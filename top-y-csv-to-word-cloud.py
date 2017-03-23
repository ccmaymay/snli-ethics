#!/usr/bin/env python


from csv import DictReader

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud

from snli_cooccur import mkdirp_parent


DEFAULT_COLOR_NAME = '#1f497d'
DEFAULT_RELATIVE_SCALING = 1.
DEFAULT_WIDTH = 800
DEFAULT_HEIGHT = 400
DEFAULT_MAX_WORDS = 50
DEFAULT_COLOR_MAP_RANGE = (0., 1.)


def parse_color_map_range(s):
    t = tuple(map(float, s.split(',')))
    if len(t) != 2:
        raise ValueError('color map range must be two comma-delimited numbers')
    if t[0] > t[1]:
        raise ValueError('lower bound of color map range must be no greater '
                         'than upper bound')
    if t[0] < 0 or t[1] > 1:
        raise ValueError('color map range must be within [0, 1]')
    return t


def top_y_csv_to_word_cloud(input_path, query, x, output_path,
                            mask_path=None,
                            color_name=DEFAULT_COLOR_NAME,
                            color_map_name=None,
                            color_map_range=DEFAULT_COLOR_MAP_RANGE,
                            relative_scaling=DEFAULT_RELATIVE_SCALING,
                            background_color_name=None,
                            max_words=DEFAULT_MAX_WORDS,
                            width=DEFAULT_WIDTH,
                            height=DEFAULT_HEIGHT):
    y_scores = dict()
    with open(input_path) as f:
        reader = DictReader(f)
        for row in reader:
            if row['query'] == query and row['x'] == x:
                y_scores[row['y']] = float(row['score'])

    if not y_scores:
        raise ValueError('found no rows matching query %s and row %s' %
                         (query, x))

    mask = None if mask_path is None else np.array(Image.open(mask_path))
    cmap = None if color_map_name is None else plt.get_cmap(color_map_name)

    def color_func(word, font_size, position, orientation, font_path,
                   random_state):
        if cmap is None:
            return color_name
        else:
            u = random_state.uniform(*color_map_range)
            (r, g, b, a) = 255 * np.array(cmap(u))
            return 'rgb(%.0f, %.0f, %.0f)' % (r, g, b)

    wordcloud = WordCloud(
        max_words=max_words,
        stopwords=(),
        prefer_horizontal=0.9,
        width=width,
        height=height,
        margin=2,
        relative_scaling=relative_scaling,
        mode='RGBA',
        color_func=color_func,
        background_color=background_color_name,
        mask=mask,
        collocations=False,
        normalize_plurals=False,
        regexp=r'\S+',
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
    parser.add_argument('--background-color-name',
                        help='name of background color (default: transparent)')
    parser.add_argument('--color-name', default=DEFAULT_COLOR_NAME,
                        help='name of text color')
    parser.add_argument('--color-map-name',
                        help='name of color map to select word colors from '
                             '(randomly) (default: use color-name for all '
                             'words)')
    parser.add_argument('--color-map-range', type=parse_color_map_range,
                        default=DEFAULT_COLOR_MAP_RANGE,
                        help='range of color map to use (as two '
                             'comma-delimited floats, a lower bound and an '
                             'upper bound)')
    parser.add_argument('--max-words', type=int, default=DEFAULT_MAX_WORDS,
                        help='number of words to display')
    parser.add_argument('--width', type=int, default=DEFAULT_WIDTH,
                        help='width of image, in pixels')
    parser.add_argument('--height', type=int, default=DEFAULT_HEIGHT,
                        help='height of image, in pixels')
    parser.add_argument('--relative-scaling', type=float,
                        default=DEFAULT_RELATIVE_SCALING,
                        help='degree to which score (rather than rank) is '
                             'used to scale words')
    args = parser.parse_args()

    top_y_csv_to_word_cloud(args.input_path, args.query, args.x,
                            args.output_path, mask_path=args.mask_path,
                            background_color_name=args.background_color_name,
                            color_name=args.color_name,
                            color_map_name=args.color_map_name,
                            color_map_range=args.color_map_range,
                            width=args.width,
                            height=args.height,
                            relative_scaling=args.relative_scaling)


if __name__ == '__main__':
    main()
