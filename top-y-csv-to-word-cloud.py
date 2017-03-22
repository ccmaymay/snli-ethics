#!/usr/bin/env python


from csv import DictReader

import numpy as np
from PIL import Image
from wordcloud import WordCloud

from snli_cooccur import mkdirp_parent


def top_y_csv_to_word_cloud(input_path, query, x, output_path, mask_path=None):
    y_scores = dict()
    with open(input_path) as f:
        reader = DictReader(f)
        for row in reader:
            if row['query'] == query and row['x'] == x:
                y_scores[row['y']] = float(row['score'])

    if not y_scores:
        raise ValueError('found no rows matching query %s and row %s' %
                         (query, x))

    if mask_path is not None:
        mask = np.array(Image.open(mask_path))

    wordcloud = WordCloud(
        stopwords=(), prefer_horizontal=0.9,
        width=800, height=400, margin=2, relative_scaling=0, mode='RGBA',
        color_func=lambda *a, **kw: '#1f497d',
        # colormap='prism',
        background_color=None,
        mask=mask,
        collocations=False, normalize_plurals=False,
        regexp=r'\S+'
    )
    wordcloud.generate_from_frequencies(y_scores)
    image = wordcloud.to_image()

    mkdirp_parent(output_path)
    with open(output_path, 'wb') as f:
        image.save(f, format='png')


def main():
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        description='Generate word cloud from CSV top-y results',
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to input CSV file')
    parser.add_argument('query',
                        help='query for which top y will be visualized')
    parser.add_argument('x',
                        help='x for which top y will be visualized '
                             '(must appear in specified query)')
    parser.add_argument('output_path', help='path to output PNG file')
    parser.add_argument('--mask-path', help='path to image mask PNG file')
    args = parser.parse_args()

    top_y_csv_to_word_cloud(args.input_path, args.query, args.x,
                            args.output_path, mask_path=args.mask_path)


if __name__ == '__main__':
    main()
