#!/usr/bin/env python


from collections import Counter
from multiprocessing import Pool
from functools import partial
import cPickle as pickle
import logging
import json
import os

from humanfriendly import format_size


WITHIN_HYPOTHESIS = 'within-hypothesis'
WITHIN_PREMISE = 'within-premise'
BETWEEN_PREM_HYPO = 'between-prem-hypo'


def configure_logging():
    '''
    Configure logging module to print INFO messages.

    This should probably be run in main or at the beginning of an
    interactive session.
    '''
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(levelname)s %(process)d: %(message)s'
    )


def mkdirp_parent(path):
    '''
    Make parent directory of path if it does not exist.
    '''
    dirname = os.path.dirname(path)
    if dirname:
        mkdirp(dirname)


def mkdirp(path):
    '''
    Make directory at path if it does not exist.
    '''
    if not os.path.isdir(path):
        os.makedirs(path)


def resource_usage_str():
    '''
    Return short string explaining current resource usage, or a stub
    with a message to install psutil if it is not available.
    '''
    try:
        import os
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return 'mem used: %s res, %s virt' % (
            format_size(mem_info.rss),
            format_size(mem_info.vms)
        )
    except:
        return 'mem used: ? res, ? virt (pip install psutil)'


def tokens_to_max_ngrams(tokens, max_ngram=1):
    '''
    Given an enumerable of tokens (strings/unicode), return a
    corresponding generator of n-grams (tuples of length n + 1
    whose first element is the start index of the ngram and latter
    indices are the string/unicode words making up the ngram) for n
    between 1 and max_ngram (inclusive).

    >>> from pprint import pprint
    >>> pprint(list(tokens_to_max_ngrams(['hello', 'world', '!'],
    ...                                  max_ngram=2)))
    [(0, ('hello',)),
     (1, ('world',)),
     (2, ('!',)),
     (0, ('hello', 'world')),
     (1, ('world', '!'))]
    '''
    tokens = list(tokens)
    for ngram in xrange(1, max_ngram + 1):
        for token in tokens_to_ngrams(tokens, ngram=ngram):
            yield token


def tokens_to_ngrams(tokens, ngram=1):
    '''
    Given an enumerable of tokens (strings/unicode), return a
    corresponding generator of n-grams (tuples of length n + 1
    whose first element is the start index of the ngram and latter
    indices are the string/unicode words making up the ngram) for n
    equal to the value of the ngram parameter.

    >>> list(tokens_to_ngrams(['hello', 'world', '!'], ngram=2))
    [(0, ('hello', 'world')), (1, ('world', '!'))]
    '''
    tokens = list(tokens)
    for start in xrange(len(tokens) - (ngram - 1)):
        yield (start, tuple(tokens[start:start + ngram]))


def binary_parse_to_tokens(parse_str):
    '''
    Given a string representing the binary parse (from the SNLI data),
    return a generator of the tokens (terminals) from the parse.

    >>> list(binary_parse_to_tokens('( ( hello world ) ! )'))
    ['hello', 'world', '!']
    '''
    return (
        w.lower()
        for w in parse_str.split(' ')
        if w not in ('(', ')')
    )


def within_sentence_pairs(sentence):
    '''
    Given a list of ngrams ((index, tokens) pairs where index is the
    start index of the ngram in a sentence and tokens is a tuple of
    string/unicode representing the tokens in the ngram) representing
    a sentence, return a generator of token pairs (cooccurrences) within
    the sentence.  Cooccurrences that intersect are skipped.

    >>> from pprint import pprint
    >>> pprint(list(within_sentence_pairs(
    ...     [(0, ('hello',)), (1, ('world',)), (2, ('!',)),
    ...      (0, ('hello', 'world')), (1, ('world', '!'))])))
    [(('hello',), ('world',)),
     (('hello',), ('!',)),
     (('hello',), ('world', '!')),
     (('world',), ('hello',)),
     (('world',), ('!',)),
     (('!',), ('hello',)),
     (('!',), ('world',)),
     (('!',), ('hello', 'world')),
     (('hello', 'world'), ('!',)),
     (('world', '!'), ('hello',))]
    '''
    for (i, ti) in sentence:
        for (j, tj) in sentence:
            # Skip intersecting tokens.
            # Determine intersection by observing that the following are
            # equivalent:
            # ti and tj to intersect.
            # ti starts within tj or tj starts within ti.
            if (i <= j and j < i + len(ti)) or (j <= i and i < j + len(tj)):
                continue
            yield (ti, tj)


def between_sentence_pairs(sentence1, sentence2):
    '''
    Given two lists of ngrams ((index, tokens) pairs where index is the
    start index of the ngram in a sentence and tokens is a tuple of
    string/unicode representing the tokens in the ngram) representing
    distinct sentences, return a generator of token pairs
    (cooccurrences) within the sentence.

    >>> from pprint import pprint
    >>> pprint(list(between_sentence_pairs(
    ...     [(0, ('hello',)), (1, ('world',)), (2, ('!',)),
    ...      (0, ('hello', 'world')), (1, ('world', '!'))],
    ...     [(0, ('goodnight',)), (0, ('goodnight', 'earth'))])))
    [(('hello',), ('goodnight',)),
     (('hello',), ('goodnight', 'earth')),
     (('world',), ('goodnight',)),
     (('world',), ('goodnight', 'earth')),
     (('!',), ('goodnight',)),
     (('!',), ('goodnight', 'earth')),
     (('hello', 'world'), ('goodnight',)),
     (('hello', 'world'), ('goodnight', 'earth')),
     (('world', '!'), ('goodnight',)),
     (('world', '!'), ('goodnight', 'earth'))]
    '''
    for (i1, t1) in sentence1:
        for (i2, t2) in sentence2:
            yield (t1, t2)


class CooccurrenceCounts(object):
    '''
    Counter for cooccurrences and marginals.

    Members:

        xy          dict (Counter) for (x, y) pairs (cooccurrences)
        x           dict (Counter) for x marginals
        y           dict (Counter) for y marginals
        xy_total    integer representing total (double marginal)
    '''

    def __init__(self):
        self.xy = Counter()
        self.x = Counter()
        self.y = Counter()
        self.xy_total = 0

    def increment(self, x, y):
        '''
        Increment counts for cooccurrence (x, y) where x and y are
        hashable, e.g., tuples of strings/unicode representing ngrams.
        '''
        self.xy[(x, y)] += 1
        self.x[x] += 1
        self.y[y] += 1
        self.xy_total += 1

    def update(self, other):
        '''
        Add all counts from other (an instance of CooccurrenceCounts)
        to this counter.
        '''
        self.xy.update(other.xy)
        self.x.update(other.x)
        self.y.update(other.y)
        self.xy_total += other.xy_total


def chunks(enumerable, chunk_size=10000):
    '''
    Given enumerable enumerable, return generator of chunks (lists) of
    up to chunk_size consecutive items from enumerable.

    This function is primarily used as a helper function for
    parallelization.

    >>> list(chunks(xrange(7), chunk_size=3))
    [[0, 1, 2], [3, 4, 5], [6]]
    '''
    chunk = []
    for x in enumerable:
        chunk.append(x)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def increment_all(counts, pairs):
    '''
    Given counts, an instance of CooccurrenceCounts, and pairs, a list
    of (x, y) pairs (where x and y are hashable), increment counts for
    all (x, y) pairs.

    >>> from mock import Mock, call
    >>> counts = Mock()
    >>> increment_all(counts,
    ...     [(('hello',), ('world',)), (('hello',), ('world', '!'))])
    >>> counts.increment.assert_has_calls([
    ...     call(('hello',), ('world',)),
    ...     call(('hello',), ('world', '!')),
    ... ])
    '''
    for (x, y) in pairs:
        counts.increment(x, y)


def compute_vocabs(snli_file_triples, filter_vocab_by_freq=1):
    '''
    Given an enumerable of triples representing the SNLI dataset and
    an integer representing the minimum count of an ngram to include
    it in the vocabulary, compute separate vocabs---sets---for
    premises and hypotheses and return the pair (premise vocab,
    hypothesis vocab).

    snli_file_triples should be an enumerable of triples.  In each
    triple, the first element is the parsed SNLI json, the second is
    a list of ngrams representing the premise, and the third is a list
    of ngrams representing the hypothesis.
    Note the first element in each ngram is the start index of that
    ngram in the sentence.
    '''
    premise_word_counts = Counter()
    hypothesis_word_counts = Counter()
    for (j, premise_ngrams, hypothesis_ngrams) in snli_file_triples:
        for (_, token) in premise_ngrams:
            premise_word_counts[token] += 1
        for (_, token) in hypothesis_ngrams:
            hypothesis_word_counts[token] += 1

    premise_vocab = set(
        word for (word, count) in premise_word_counts.items()
        if count >= filter_vocab_by_freq)
    hypothesis_vocab = set(
        word for (word, count) in hypothesis_word_counts.items()
        if count >= filter_vocab_by_freq)

    return (premise_vocab, hypothesis_vocab)


def count_cooccurrences(snli_file_triples, model,
                        premise_vocab=None, hypothesis_vocab=None,
                        filter_hypo_by_prem=False):
    '''
    Given an enumerable of triples representing the SNLI dataset and a
    string representing the cooccurrence model (WITHIN_HYPOTHESIS,
    WITHIN_PREMISE, BETWEEN_PREM_HYPO), count cooccurrences in a
    CooccurrenceCounts object and return it.

    If premise_vocab is not None, filter premise tokens to those
    appearing in premise_vocab (a set).

    If hypothesis_vocab is not None, filter hypothesis tokens to those
    appearing in hypothesis_vocab (a set).

    If filter_hypo_by_prem is True, remove words in hypothesis that
    appear in the premise.

    snli_file_triples should be an enumerable of triples.  In each
    triple, the first element is the parsed SNLI json, the second is
    a list of ngrams representing the premise, and the third is a list
    of ngrams representing the hypothesis.
    Note the first element in each ngram is the start index of that
    ngram in the sentence.
    '''
    counts = CooccurrenceCounts()

    for (j, premise_ngrams, hypothesis_ngrams) in snli_file_triples:
        if premise_vocab is not None:
            premise_ngrams = filter(
                lambda p: p[1] in premise_vocab,
                premise_ngrams)
        if hypothesis_vocab is not None:
            hypothesis_ngrams = filter(
                lambda p: p[1] in hypothesis_vocab,
                hypothesis_ngrams)

        if filter_hypo_by_prem:
            premise_filter_set = set(map(lambda p: p[1], premise_ngrams))
            hypothesis_ngrams = filter(
                lambda p: p[1] not in premise_filter_set,
                hypothesis_ngrams)

        if model == WITHIN_HYPOTHESIS:
            increment_all(
                counts,
                within_sentence_pairs(hypothesis_ngrams))
        elif model == BETWEEN_PREM_HYPO:
            increment_all(
                counts,
                between_sentence_pairs(premise_ngrams, hypothesis_ngrams))
        elif model == WITHIN_PREMISE:
            increment_all(
                counts,
                within_sentence_pairs(premise_ngrams))
        else:
            raise ValueError('unknown model %s' % model)

    return counts


def iter_snli(snli_jsonl_path, inference_type=None, max_ngram=1,
              unique_premises=False):
    '''
    Given a path to an SNLI jsonl file (list of JSON-serialized
    premise-hypothesis pairs, one per line), return generator of
    SNLI file triples.

    If inference_type is not None, filter to only those triples
    whose gold-labeled inference type ('contradiction', 'neutral',
    'entailment') matches inference_type.

    Compute the tokens of both the premise and hypothesis as all
    n-grams for n between 1 and max_ngram.

    If unique_premises is True, only emit one premise-hypothesis
    pair for each premise.

    In each
    triple, the first element is the parsed SNLI json, the second is
    a list of ngrams representing the premise, and the third is a list
    of ngrams representing the hypothesis.
    Note the first element in each ngram is the start index of that
    ngram in the sentence.
    '''
    caption_ids_seen = set()

    with open(snli_jsonl_path) as f:
        for (i, line) in enumerate(f):
            if i % 1000 == 0:
                logging.info('ingested %d hypotheses (%s)' %
                             (i, resource_usage_str()))

            j = json.loads(line)

            if (inference_type is not None and
                    j['gold_label'] != inference_type):
                continue

            caption_id = j['captionID']

            if unique_premises and caption_id in caption_ids_seen:
                continue

            caption_ids_seen.add(caption_id)

            premise_ngrams = list(tokens_to_max_ngrams(
                binary_parse_to_tokens(j['sentence1_binary_parse']),
                max_ngram=max_ngram))
            hypothesis_ngrams = list(tokens_to_max_ngrams(
                binary_parse_to_tokens(j['sentence2_binary_parse']),
                max_ngram=max_ngram))

            yield (j, premise_ngrams, hypothesis_ngrams)


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='compute PMI on SNLI',
    )
    parser.add_argument('model', type=str,
                        choices=(WITHIN_PREMISE, WITHIN_HYPOTHESIS,
                                 BETWEEN_PREM_HYPO),
                        help=('cooccurrence model to compute'))
    parser.add_argument('snli_jsonl_path', type=str,
                        help='path to snli_1.0_train.json')
    parser.add_argument('pickle_path', type=str,
                        help='(output) path to pickled counts and pmi')
    parser.add_argument('--inference-type', type=str,
                        choices=('entailment', 'contradiction', 'neutral'),
                        help=('filter to inferences of this type (only for %s '
                              'and %s models') % (WITHIN_HYPOTHESIS,
                                                  BETWEEN_PREM_HYPO))
    parser.add_argument('--filter-hypo-by-prem', action='store_true',
                        help='remove words from hypothesis that appear in '
                             'premise (only for %s and %s models' % (
                                 WITHIN_HYPOTHESIS, BETWEEN_PREM_HYPO))
    parser.add_argument('--max-ngram', type=int, default=1,
                        help='compute n-grams for n up to this number')
    parser.add_argument('--num-proc', type=int, default=1,
                        help='size of processor pool to use')
    parser.add_argument('--filter-vocab-by-freq', type=int,
                        help='filter vocab to words occuring at least this '
                             'many times')

    args = parser.parse_args()
    configure_logging()

    if args.filter_hypo_by_prem and args.model == WITHIN_PREMISE:
        raise ValueError(
            'can only filter hypo by prem for %s and %s models' % (
                WITHIN_HYPOTHESIS, BETWEEN_PREM_HYPO))

    if args.inference_type is not None and args.model == WITHIN_PREMISE:
        raise ValueError(
            'can only filter by inference type for %s and %s models' % (
                WITHIN_HYPOTHESIS, BETWEEN_PREM_HYPO))

    pool = Pool(args.num_proc)

    if args.filter_vocab_by_freq is None:
        premise_vocab = None
        hypothesis_vocab = None
    else:
        premise_vocab = set()
        hypothesis_vocab = set()
        for (pv, hv) in pool.imap_unordered(
                partial(compute_vocabs,
                        filter_vocab_by_freq=args.filter_vocab_by_freq),
                chunks(iter_snli(args.snli_jsonl_path,
                                 inference_type=args.inference_type,
                                 max_ngram=args.max_ngram))):
            premise_vocab.update(pv)
            hypothesis_vocab.update(hv)

    unique_premises = (args.model == WITHIN_PREMISE)
    counts = CooccurrenceCounts()
    for c in pool.imap_unordered(
            partial(count_cooccurrences, model=args.model,
                    premise_vocab=premise_vocab,
                    hypothesis_vocab=hypothesis_vocab,
                    filter_hypo_by_prem=args.filter_hypo_by_prem),
            chunks(iter_snli(args.snli_jsonl_path,
                             inference_type=args.inference_type,
                             max_ngram=args.max_ngram,
                             unique_premises=unique_premises))):
        counts.update(c)

    logging.info('saving to disk (%s)' % resource_usage_str())

    mkdirp_parent(args.pickle_path)
    with open(args.pickle_path, 'w') as f:
        pickle.dump(counts, f)

    logging.info('done')


if __name__ == '__main__':
    main()
