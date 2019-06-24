"""
@title
    utilities.py
@desc
    Provides a suite of utility and convenience functions.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

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


def get_project_props():
    """
    Locates a 'meta.pyproj' file that contains properties about the project that
    the running script is a part of. This file is expected to be located
    in the direct path from the running script back to the root.

    :return: dict containing the project properties as well as the base directory
    of the project
    """
    prop_file = None
    base_dir = os.path.dirname(__file__)
    base_dir = os.path.abspath(base_dir)

    while not prop_file:
        f_to_chk = os.path.join(base_dir, 'meta.pyproj')
        if os.path.isfile(f_to_chk):
            prop_file = f_to_chk

        if os.path.ismount(base_dir):
            break
        base_dir = os.path.dirname(base_dir)

    proj_props = None
    if prop_file:
        with open(prop_file) as json_file:
            proj_props = json.load(json_file)
        proj_props['project_dir'] = os.path.dirname(prop_file)
    return proj_props


def get_version():
    """
    Gets the version of the project of the currently running script,
    as specified by the 'version' string contained in the
    project's 'meta.pyproj' file.

    :return: tpule(project name, project version)
    """
    proj_props = get_project_props()

    if 'name' in proj_props:
        proj_name = proj_props['name']
    else:
        proj_name = 'Unknown'
        print('Verify that \'name\' is set in the meta.pyproj file')

    if 'version' in proj_props:
        proj_version = proj_props['version']
    else:
        proj_version = 'Unknown'
        print('Verify that \'version\' is set in the meta.pyproj file')
    return proj_name, proj_version


def get_full_resource_path():
    """
    Gets the resource directory of the current running project.

    :return: full path to the project resource directory
    """
    res_dir = Path(PROJ_PROPS['project_dir'], PROJ_PROPS['resource_dir'])
    return res_dir


def get_full_data_path():
    """
    Gets the data directory of the current running project.

    :return: full path to the project data directory
    """
    data_dir = Path(get_full_resource_path(), 'data')
    return data_dir


def main(args):
    """
    :return:    None
    """
    if args.version:
        proj_name, proj_version = get_version()
        print('%s: VERSION: %s' % (proj_name, proj_version))
        return

    proj_props = get_project_props()
    if proj_props:
        print('Project directory: %s' % proj_props['project_dir'])
    return


PROJ_PROPS = get_project_props()
src_dir = PROJ_PROPS['project_dir'] / Path(PROJ_PROPS['source_dir'])
resrc_dir = PROJ_PROPS['project_dir'] / Path(PROJ_PROPS['resource_dir'])
test_dir = PROJ_PROPS['project_dir'] / Path(PROJ_PROPS['test_dir'])

sys.path.append(PROJ_PROPS['project_dir'])
sys.path.append(src_dir)
sys.path.append(test_dir)
sys.path.append(test_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Provides a suite of utility and convenience functions.')
    parser.add_argument('--version', '-v', action='store_true',
                        help='print current version of implementation and exit')
    print('-' * TERMINAL_COLUMNS)
    print(parser.prog)
    print('-' * TERMINAL_COLUMNS)

    cargs = parser.parse_args()
    main(cargs)

    print('-' * TERMINAL_COLUMNS)
    print('Exiting main')
    print('-' * TERMINAL_COLUMNS)
