"""
@title
    utilities.py
@desc
    Provides a suite of utility and convenience functions.
"""
import argparse
import shutil

from SystemControl import version, name

TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()


def build_dir_path(dir_name):
    """
    Builds up the path to the specified directory if it does not exist.

    :param dir_name:
    :return:
    """
    if not dir_name.exists():
        dir_name.mkdir(parents=True, exist_ok=True)
    return


def main(args):
    """
    :return:    None
    """
    if args.version:
        print('%s: VERSION: %s' % (name, version))
        return
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provides a suite of utility and convenience functions.')
    parser.add_argument('--version', '-v', action='store_true',
                        help='print current version of implementation and exit')
    print('-' * TERMINAL_COLUMNS)
    print(parser.prog)
    print('-' * TERMINAL_COLUMNS)

    cargs = parser.parse_args()
    main(cargs)
