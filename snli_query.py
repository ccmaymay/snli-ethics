#!/usr/bin/env python


from math import log
from heapq import nlargest
from itertools import product
from functools import partial
from contextlib import contextmanager
from csv import DictWriter
import logging
import cPickle as pickle
import sys

import yaml
import numpy as np
from scipy.stats import chi2

from snli_cooccur import CooccurrenceCounts  # noqa
from snli_cooccur import resource_usage_str, mkdirp_parent


def parse_ngram(s):
    '''
    Given a string/unicode representing an ngram as a sequence of
    tokens separated by spaces, return the corresponding tuple
    representation used in the counting/scoring code.

    >>> parse_ngram('hello world')
    ('hello', 'world')
    >>> parse_ngram('\thello world ')
    ('hello', 'world')
    '''
    return tuple(s.strip().split())


def format_ngram(ngram):
    '''
    Given a tuple of string/unicode representing an ngram,
    return a single string/unicode with the tokens of the ngram
    separated by spaces.

    >>> format_ngram(('hello', 'world'))
    'hello world'
    '''
    return ' '.join(ngram)


def g_test_obs_table(counts, x, y):
    '''
    Return 2 x 2 contingency table (array) of observed cooccurrence
    (x, y) counts for G-test.  Read observed counts from counts,
    an instance of CooccurrenceCounts.

    >>> c = CooccurrenceCounts()
    >>> c.increment('the', 'dog')
    >>> c.increment('good', 'dog')
    >>> c.increment('bad', 'dog')
    >>> c.increment('dog', 'ran')
    >>> c.increment('cat', 'ran')
    >>> c.increment('fish', 'ran')
    >>> g_test_obs_table(c, 'the', 'dog')
    array([[1, 0],
           [2, 3]])
    '''
    xy_count = counts.xy[(x, y)]
    x_count = counts.x[x]
    y_count = counts.y[y]
    return np.array([
        [xy_count, x_count - xy_count],
        [y_count - xy_count, counts.xy_total - (x_count + y_count - xy_count)]
    ])


def g_test_exp_table(obs_table):
    '''
    Return 2 x 2 contingency table (array) of expected cooccurrence
    (x, y) counts for G-test.  Read observed counts from obs_table,
    the corresponding 2 x 2 contingency table of observed counts.

    >>> g_test_exp_table(np.array([[3, 2],
    ...                            [1, 10]]))
    array([[ 1.25,  3.75],
           [ 2.75,  8.25]])
    '''
    return np.outer(
        np.sum(obs_table, axis=1),  # row sums
        np.sum(obs_table, axis=0)   # col sums
    ) / np.sum(obs_table, dtype=np.float)


def g_test_stat(counts, x, y, min_count=1):
    '''
    Return G-test statistic for (x, y) cooccurrence using counts from
    counts (an instance of CooccurrenceCounts).

    Return -inf if (x, y) has a count less than min_count.
    '''
    if counts.xy[(x, y)] >= min_count:
        obs_table = g_test_obs_table(counts, x, y)
        exp_table = g_test_exp_table(obs_table)
        return 2 * np.sum(obs_table * (np.log(obs_table) - np.log(exp_table)))
    else:
        return float('-inf')


def g_test_p_value(g):
    '''
    Return the p-value for a given 2 x 2 G-test statistic value.

    See http://www.itl.nist.gov/div898/handbook/eda/section3/eda3674.htm
    >>> np.allclose(g_test_p_value(2.706), 0.1, rtol=0.01)
    True
    >>> np.allclose(g_test_p_value(3.841), 0.05, rtol=0.01)
    True
    >>> np.allclose(g_test_p_value(6.635), 0.01, rtol=0.01)
    True
    >>> np.allclose(g_test_p_value(10.828), 0.001, rtol=0.01)
    True
    '''
    return chi2.sf(g, 1)


def pmi(counts, x, y, min_count=1):
    '''
    Return PMI for (x, y) cooccurrence using counts from counts (an
    instance of CooccurrenceCounts).

    Return -inf if (x, y) has a count less than min_count.
    '''
    if counts.xy[(x, y)] >= min_count:
        return (
            (log(counts.xy[(x, y)]) - log(counts.xy_total)) - (
                (log(counts.x[x]) - log(counts.xy_total)) +
                (log(counts.y[y]) - log(counts.xy_total))
            )
        )
    else:
        return float('-inf')


def filter_y(counts, x, min_count=1, filter_to_unigrams=False):
    '''
    Return list of y representing (x, y) cooccurrences,
    computed using counts (an instance of CooccurrenceCounts),
    filtered as follows.

    Cooccurrences (x, y) whose count is less than min_count are not
    included in the list.

    If filter_to_unigrams is True, filter results to unigrams only.
    '''
    return [
        y
        for y in counts.y.keys()
        if counts.xy[(x, y)] >= min_count and not (
            filter_to_unigrams and len(y) > 1
        )
    ]


def top_y(score_func, counts, x, k=10, min_count=1, filter_to_unigrams=False):
    '''
    Return list of top (y, score) pairs where y is hashable
    and score is a float, representing the
    top k (x, y) cooccurrences sorted by score (in descending order)
    computed using counts (an instance of CooccurrenceCounts).

    The score is computed by score_func and can be e.g. pmi
    or g_test_stat.

    Cooccurrences (x, y) whose count is less than min_count are not
    included in the list.  (If there are not enough candidates the
    list will be shorter than k.)

    If filter_to_unigrams is True, filter results to unigrams only
    before truncating at k.
    '''
    return nlargest(
        k,
        [
            (y, score_func(counts, x, y))
            for y in filter_y(counts, x, min_count=min_count,
                              filter_to_unigrams=filter_to_unigrams)
        ],
        key=lambda t: t[1],
    )


def top_y_batch(score_func, counts_map, x_list, *args, **kwargs):
    '''
    Given counts_map, a dictionary of identifiers (e.g., filenames)
    to CooccurrenceCounts instances, x_list, a list of hashables,
    and any args to top_y, return a list of triples representing
    the top (x, y) pairs by score in each counter, for each x in x_list.

    The score is computed by score_func and can be e.g. pmi
    or g_test_stat.  args and kwargs are passed through to score_func.
    '''
    return [
        (counts_name, x, top_y(score_func, counts, x, *args, **kwargs))
        for ((counts_name, counts), x)
        in product(counts_map.items(), x_list)
    ]


def tex_format_signif(word, stars):
    r'''
    >>> tex_format_signif('foo', '')
    'foo'
    >>> tex_format_signif('foo', '*')
    'foo'
    >>> tex_format_signif('foo', '**')
    'foo$^\\dagger$'
    >>> tex_format_signif('foo', '***')
    'foo$^\\ddagger$'
    >>> tex_format_signif('foo', '****')
    'foo$^\\ddagger$'
    '''
    if len(stars) < 2:
        return word
    elif len(stars) == 2:
        return r'%s$^\dagger$' % word
    else:
        return r'%s$^\ddagger$' % word


def write_top_y_tex_batch_yaml(score_func, output_file, counts, queries_path,
                               *args, **kwargs):
    '''
    Load top-y queries from the YAML specification in the file at
    queries_path and execute them using counts (an instance of
    CooccurrenceCounts), passing score_func, args, and kwargs to top_y,
    writing results in tex friendly format to output_file.
    '''
    with open(queries_path) as f:
        queries = yaml.load(f)

    filter_y_kwargs = dict((k, v) for (k, v) in kwargs.items() if k != 'k')
    x_ngram_y_ngram_pairs = []
    for (query_name, query) in queries.items():
        for x in query['x']:
            x_ngram = parse_ngram(x)
            x_ngram_y_ngram_pairs.extend([
                (x_ngram, y_ngram) for y_ngram in
                filter_y(counts, x_ngram, *args, **filter_y_kwargs)
            ])
    p_values = bonferroni_holm_g_test_p_values(
        counts, x_ngram_y_ngram_pairs)

    for (query_name, query) in queries.items():
        output_file.write('\n    %% %s\n' % query_name)
        for x in query['x']:
            output_file.write(r'        \textbf{%s} &' % x)

            x_ngram = parse_ngram(x)
            y_ngrams = [y_ngram for (y_ngram, score) in
                        top_y(score_func, counts, x_ngram, *args, **kwargs)]
            for y_ngram in y_ngrams:
                p_value = p_values[(x_ngram, y_ngram)]
                stars = p_value_to_stars(p_value)
                output_file.write(
                    ' %s' % tex_format_signif(format_ngram(y_ngram), stars))
            output_file.write(' \\\\\n')


def write_top_y_csv_batch_yaml(score_func, output_file, counts,
                               queries_path, *args, **kwargs):
    '''
    Load top-y queries from the YAML specification in the file at
    queries_path and execute them using counts (an instance of
    CooccurrenceCounts), passing score_func, args, and kwargs to top_y,
    writing query, x, y, score tuples as CSV to output_file.
    '''
    with open(queries_path) as f:
        queries = yaml.load(f)

    filter_y_kwargs = dict((k, v) for (k, v) in kwargs.items() if k != 'k')
    x_ngram_y_ngram_pairs = []
    for (query_name, query) in queries.items():
        for x in query['x']:
            x_ngram = parse_ngram(x)
            x_ngram_y_ngram_pairs.extend([
                (x_ngram, y_ngram) for y_ngram in
                filter_y(counts, x_ngram, *args, **filter_y_kwargs)
            ])

    writer = DictWriter(output_file, ('query', 'x', 'y', 'score'))
    writer.writeheader()
    for (query_name, query) in queries.items():
        for x in query['x']:
            x_ngram = parse_ngram(x)
            y_ngram_score_pairs = [
                (y_ngram, score)
                for (y_ngram, score)
                in top_y(score_func, counts, x_ngram, *args, **kwargs)
                if score > 0
            ]
            for (y_ngram, score) in y_ngram_score_pairs:
                writer.writerow(dict(
                    query=query_name,
                    x=x,
                    y=format_ngram(y_ngram),
                    score=score))


def bonferroni_holm_g_test_p_values(counts, x_ngram_y_ngram_pairs):
    '''
    Compute Bonferroni-Holm adjusted p-values for the G-test statistics
    for (x_ngram, y_ngram) pairs in x_ngram_y_ngram_pairs (an iterable).
    Return dict of adjusted p-values indexed by (x_ngram, y_ngram).
    '''
    xyp_triples = sorted(
        [
            (
                x_ngram,
                y_ngram,
                g_test_p_value(g_test_stat(counts, x_ngram, y_ngram))
            )
            for (x_ngram, y_ngram) in x_ngram_y_ngram_pairs
        ],
        key=lambda p: p[1])

    group_p_values = dict()
    num_tests = len(xyp_triples)
    for (test_num, (x_ngram, y_ngram, p_value)) in enumerate(xyp_triples):
        # reject at level alpha if p < alpha / (m + 1 - k)
        # where m is the number of tests and k is the 1-based index
        group_p_values[(x_ngram, y_ngram)] = p_value * (num_tests - test_num)

    return group_p_values


def p_value_to_stars(p_value, alpha=(0.05, 0.01, 0.001)):
    '''
    Return string containing as many stars as the number of significance
    levels in alpha (a tuple of significance levels, order-independent)
    that p_value is less than or equal to.

    >>> p_value_to_stars(0.075)
    ''
    >>> p_value_to_stars(0.05)
    '*'
    >>> p_value_to_stars(0.025)
    '*'
    >>> p_value_to_stars(0.0099)
    '**'
    >>> p_value_to_stars(0.005)
    '**'
    >>> p_value_to_stars(0.0025)
    '**'
    >>> p_value_to_stars(0.00099)
    '***'
    '''
    return len([_alpha for _alpha in alpha if p_value <= _alpha]) * '*'


def write_top_y_batch_yaml(score_func, output_file, counts, queries_path,
                           *args, **kwargs):
    '''
    Load top-y queries from the YAML specification in the file at
    queries_path and execute them using counts (an instance of
    CooccurrenceCounts), passing score_func, args, and kwargs to top_y,
    writing results to output_file.
    '''
    with open(queries_path) as f:
        queries = yaml.load(f)

    filter_y_kwargs = dict((k, v) for (k, v) in kwargs.items() if k != 'k')
    x_ngram_y_ngram_pairs = []
    for (query_name, query) in queries.items():
        for x in query['x']:
            x_ngram = parse_ngram(x)
            x_ngram_y_ngram_pairs.extend([
                (x_ngram, y_ngram) for y_ngram in
                filter_y(counts, x_ngram, *args, **filter_y_kwargs)
            ])
    p_values = bonferroni_holm_g_test_p_values(
        counts, x_ngram_y_ngram_pairs)

    for (query_name, query) in queries.items():
        output_file.write(query_name)
        output_file.write('\n')
        for x in query['x']:
            x_ngram = parse_ngram(x)
            output_file.write('\t' + x)
            output_file.write('\n')
            y_ngrams = [y_ngram for (y_ngram, score) in
                        top_y(score_func, counts, x_ngram, *args, **kwargs)]
            for y_ngram in y_ngrams:
                _g = g_test_stat(counts, x_ngram, y_ngram)
                _pmi = pmi(counts, x_ngram, y_ngram)
                p_value = p_values[(x_ngram, y_ngram)]
                stars = p_value_to_stars(p_value)
                output_file.write('\t\t%20s\t%9.2f\t%9.2f%s\t%7.2g\t%d' % (
                    format_ngram(y_ngram),
                    _pmi,
                    _g,
                    stars,
                    p_value,
                    counts.xy[(x_ngram, y_ngram)]))
                output_file.write('\n')


def write_score_batch_yaml(output_file, counts, queries_path,
                           min_count=1):
    '''
    Load score queries from the YAML specification in the file at
    queries_path and execute them using counts (an instance of
    CooccurrenceCounts), writing results to output_file.

    The score is computed by score_func and can be e.g. pmi
    or g_test_stat.
    '''
    with open(queries_path) as f:
        queries = yaml.load(f)

    x_ngram_y_ngram_pairs = []
    for (query_name, query) in queries.items():
        for x in query['x']:
            x_ngram = parse_ngram(x)
            for y in query['y']:
                y_ngram = parse_ngram(y)
                x_ngram_y_ngram_pairs.append((x_ngram, y_ngram))
    p_values = bonferroni_holm_g_test_p_values(
        counts, x_ngram_y_ngram_pairs)

    for (query_name, query) in queries.items():
        output_file.write(query_name)
        output_file.write('\n')
        for x in query['x']:
            x_ngram = parse_ngram(x)
            output_file.write('\t' + x)
            output_file.write('\n')
            y_ngrams = [parse_ngram(y) for y in query['y']]
            for y_ngram in y_ngrams:
                _g = g_test_stat(counts, x_ngram, y_ngram, min_count=min_count)
                _pmi = pmi(counts, x_ngram, y_ngram, min_count=min_count)
                p_value = p_values[(x_ngram, y_ngram)]
                stars = p_value_to_stars(p_value)
                output_file.write('\t\t%20s\t%5.2f\t%9.2f%s\t%7.2g\t%d' % (
                    format_ngram(y_ngram),
                    _pmi,
                    _g,
                    stars,
                    p_value,
                    counts.xy[(x_ngram, y_ngram)]))
                output_file.write('\n')


def write_identity_concept_batch_yaml(output_file, counts,
                                      queries_path, min_count=1):
    '''
    Load identity/concept tests from the YAML specification in the
    file at queries_path and execute them using counts (an instance of
    CooccurrenceCounts), scoring by score_func, writing results to
    output_file.

    The score is computed by score_func and can be e.g. pmi
    or g_test_stat.
    '''
    with open(queries_path) as f:
        queries = yaml.load(f)

    x_ngram_y_ngram_pairs = []
    for query in queries['experiments']:
        identity_name = query['identity']
        concept_name = query['concept']
        identity = queries['identities'][identity_name]
        concept = queries['concepts'][concept_name]
        concept_ngrams = [parse_ngram(concept_term)
                          for concept_term in concept]
        for (id_group_name, id_group) in identity['groups'].items():
            for key in identity['keys']:
                id_term = id_group[key]
                id_ngram = parse_ngram(id_term)
                for concept_ngram in concept_ngrams:
                    x_ngram_y_ngram_pairs.append((id_ngram, concept_ngram))
    p_values = bonferroni_holm_g_test_p_values(
        counts, x_ngram_y_ngram_pairs)

    for query in queries['experiments']:
        identity_name = query['identity']
        concept_name = query['concept']
        output_file.write('%s + %s' % (identity_name, concept_name))
        output_file.write('\n')
        identity = queries['identities'][identity_name]
        concept = queries['concepts'][concept_name]
        concept_ngrams = [parse_ngram(concept_term)
                          for concept_term in concept]
        for (id_group_name, id_group) in identity['groups'].items():
            output_file.write('\t%s' % id_group_name)
            output_file.write('\n')
            for key in identity['keys']:
                id_term = id_group[key]
                id_ngram = parse_ngram(id_term)
                output_file.write('\t\t%s' % id_term)
                output_file.write('\n')
                for concept_ngram in concept_ngrams:
                    _g = g_test_stat(counts, id_ngram, concept_ngram,
                                     min_count=min_count)
                    _pmi = pmi(counts, id_ngram, concept_ngram,
                               min_count=min_count)
                    p_value = p_values[(id_ngram, concept_ngram)]
                    stars = p_value_to_stars(p_value)
                    output_file.write('\t\t%20s\t%5.2f\t%9.2f%s\t%7.2g\t%d' % (
                        format_ngram(concept_ngram),
                        _pmi,
                        _g,
                        stars,
                        p_value,
                        counts.xy[(id_ngram, concept_ngram)]))
                    output_file.write('\n')


pmi_top_y = partial(top_y, pmi)
pmi_top_y_batch = partial(top_y_batch, pmi)
write_pmi_top_y_tex_batch_yaml = partial(write_top_y_tex_batch_yaml, pmi)
write_pmi_top_y_csv_batch_yaml = partial(write_top_y_csv_batch_yaml, pmi)
write_pmi_top_y_batch_yaml = partial(write_top_y_batch_yaml, pmi)
write_pmi_score_batch_yaml = partial(write_score_batch_yaml, pmi)
write_pmi_identity_concept_batch_yaml = partial(
    write_identity_concept_batch_yaml, pmi)

g_test_stat_top_y = partial(top_y, g_test_stat)
g_test_stat_top_y_batch = partial(top_y_batch, g_test_stat)
write_g_test_stat_top_y_tex_batch_yaml = partial(
    write_top_y_tex_batch_yaml, g_test_stat)
write_g_test_stat_top_y_csv_batch_yaml = partial(
    write_top_y_csv_batch_yaml, g_test_stat)
write_g_test_stat_top_y_batch_yaml = partial(
    write_top_y_batch_yaml, g_test_stat)
write_g_test_stat_score_batch_yaml = partial(
    write_score_batch_yaml, g_test_stat)
write_g_test_stat_identity_concept_batch_yaml = partial(
    write_identity_concept_batch_yaml, g_test_stat)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    from snli_cooccur import configure_logging

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='run SNLI queries from YAML',
    )
    parser.add_argument('pickle_path', type=str,
                        help='path to pickled counts')
    parser.add_argument('queries_type', type=str,
                        choices=('score',
                                 'top-y', 'top-y-tex', 'top-y-csv',
                                 'identity-concept'),
                        help='type of queries to run')
    parser.add_argument('queries_path', type=str,
                        help='path to query YAML spec')
    parser.add_argument('output_path', type=str,
                        help='path to output (- for standard output)')
    parser.add_argument('-k', type=int, default=10,
                        help='number of items to print for top-y queries')
    parser.add_argument('--min-count', type=int, default=1,
                        help='min count to filter to in top-y queries')
    parser.add_argument('--top-y-score-func',
                        type=lambda s: {
                            'pmi': pmi,
                            'g-test-stat': g_test_stat
                        }[s],
                        default='pmi',
                        help='name of score function to sort by '
                             '(pmi, g-test-stat)')
    parser.add_argument('--filter-to-unigrams', action='store_true',
                        help='only output unigrams (filter out other results)')

    args = parser.parse_args()
    configure_logging()

    if args.output_path == '-':
        @contextmanager
        def _open_output_file():
            return sys.stdout
    else:
        def _open_output_file():
            mkdirp_parent(args.output_path)
            return open(args.output_path, 'w')

    with _open_output_file() as output_file:
        logging.info('loading counts (%s)' % resource_usage_str())

        with open(args.pickle_path) as f:
            counts = pickle.load(f)

        logging.info('counts loaded (%s)' % resource_usage_str())

        if args.queries_type == 'top-y':
            logging.info('running top-y queries')
            write_top_y_batch_yaml(
                args.top_y_score_func,
                output_file,
                counts, args.queries_path,
                k=args.k, min_count=args.min_count,
                filter_to_unigrams=args.filter_to_unigrams)

        elif args.queries_type == 'top-y-tex':
            logging.info('running top-y queries (tex output)')
            write_top_y_tex_batch_yaml(
                args.top_y_score_func,
                output_file,
                counts, args.queries_path,
                k=args.k, min_count=args.min_count,
                filter_to_unigrams=args.filter_to_unigrams)

        elif args.queries_type == 'top-y-csv':
            logging.info('running top-y queries (csv output)')
            write_top_y_csv_batch_yaml(
                args.top_y_score_func,
                output_file,
                counts, args.queries_path,
                k=args.k, min_count=args.min_count,
                filter_to_unigrams=args.filter_to_unigrams)

        elif args.queries_type == 'score':
            logging.info('running score queries')
            write_score_batch_yaml(
                output_file,
                counts, args.queries_path,
                min_count=args.min_count)

        elif args.queries_type == 'identity-concept':
            logging.info('running identity-concept queries')
            write_identity_concept_batch_yaml(
                output_file,
                counts,
                args.queries_path,
                min_count=args.min_count)

        else:
            raise ValueError('unknown query type %s' % args.queries_type)

    logging.info('done')


if __name__ == '__main__':
    main()
