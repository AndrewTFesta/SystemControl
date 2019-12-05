import os
import shutil
from pathlib import Path

name = 'SystemControl'
version = '0.1'

SOURCE_PACKAGE = os.path.split(__file__)[0]
PROJECT_PATH = os.path.split(SOURCE_PACKAGE)[0]

RESOURCES_DIR = os.path.join(SOURCE_PACKAGE, 'resources')
IMAGES_DIR = os.path.join(RESOURCES_DIR, 'images')

LOG_DIR = os.path.join(PROJECT_PATH, 'logs')
DATA_DIR = os.path.join(PROJECT_PATH, 'data')

TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()


def __init_dirs() -> None:
    dir_list = [
        RESOURCES_DIR,
        IMAGES_DIR,
        LOG_DIR,
        DATA_DIR
    ]
    for each_dir in dir_list:
        each_dir_path = Path(each_dir)
        if not each_dir_path.exists():
            each_dir_path.mkdir(parents=True, exist_ok=True)
    return


__init_dirs()
