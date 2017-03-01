import yaml
import logging
from snli_cooccur import configure_logging
from pprint import pprint

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='validate syntax of YAML file',
    )
    parser.add_argument('yaml_path', type=str,
                        help='path to YAML file to validate')
    args = parser.parse_args()
    configure_logging()
    logging.info('opening file ...')
    with open(args.yaml_path) as f:
        logging.info('parsing file ...')
        pprint(yaml.load(f))
    logging.info('ok')
