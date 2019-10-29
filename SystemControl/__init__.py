import os
import shutil
from pathlib import Path

name = 'SystemControl'
version = '0.1'

SOURCE_PACKAGE = os.path.split(__file__)[0]
PROJECT_PATH = os.path.split(SOURCE_PACKAGE)[0]

RESOURCE_DIR = os.path.join(SOURCE_PACKAGE, 'resources')
IMAGES_DIR = os.path.join(RESOURCE_DIR, 'images')

OBCI_PYTHON_DIR = os.path.join(SOURCE_PACKAGE, 'OBciPython',)
OPENBCI_HUB_NAME = 'OpenBCIHub.exe'
HUB_EXE = os.path.join(OBCI_PYTHON_DIR, 'OpenBCIHub', OPENBCI_HUB_NAME)

CHROME_DRIVER_DIR = os.path.join(SOURCE_PACKAGE, 'chromedriver')
CHROME_DRIVER_NAME = 'chromedriver.exe'
CHROME_DRIVER_EXE = os.path.join(CHROME_DRIVER_DIR, CHROME_DRIVER_NAME)

LOG_DIR = os.path.join(PROJECT_PATH, 'logs')
DATA_DIR = os.path.join(PROJECT_PATH, 'data')

RECORDED_DATA_DIR = os.path.join(DATA_DIR, 'recordedEvents')

DATABASE_URL = os.path.join(DATA_DIR, 'eeg_db_sqllite.db')

TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()


def __init_dirs() -> None:
    dir_list = [
        RESOURCE_DIR,
        IMAGES_DIR,
        LOG_DIR,
        DATA_DIR,
        RECORDED_DATA_DIR,
    ]
    for each_dir in dir_list:
        each_dir_path = Path(each_dir)
        if not each_dir_path.exists():
            each_dir_path.mkdir(parents=True, exist_ok=True)
    return
