#!/usr/bin/env python


from snli_cooccur import iter_snli
from snli_query import parse_ngram


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='grep for examples in SNLI data',
    )
    parser.add_argument('snli_jsonl_path', type=str,
                        help='path to snli_1.0_train.json')
    parser.add_argument('--hypothesis-token', type=parse_ngram,
                        help='token to grep for in hypothesis')
    parser.add_argument('--premise-token', type=parse_ngram,
                        help='token to grep for in premise')
    args = parser.parse_args()

    premise_token = args.premise_token
    hypothesis_token = args.hypothesis_token

    max_ngram = max(
        len(premise_token) if premise_token is not None else 0,
        len(hypothesis_token) if hypothesis_token is not None else 0
    )
    unique_premises = hypothesis_token is None

    snli_file_triples = iter_snli(args.snli_jsonl_path,
                                  max_ngram=max_ngram,
                                  unique_premises=unique_premises)
    for (j, premise_ngrams, hypothesis_ngrams) in snli_file_triples:
        passes_filter = (
            (
                premise_token is None or
                len([
                    token for (_, token)
                    in premise_ngrams
                    if premise_token == token
                ]) > 0
            ) and (
                hypothesis_token is None or
                len([
                    token for (_, token)
                    in hypothesis_ngrams
                    if hypothesis_token == token
                ]) > 0
            )
        )
        if passes_filter:
            print ' ', j['sentence1']
            print j['gold_label'][0].upper(), j['sentence2']
            print


if __name__ == '__main__':
    main()
