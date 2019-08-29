import os
import shutil

NAME = 'SystemControl'
VERSION = '0.1'

SOURCE_PACKAGE = os.path.split(__file__)[0]
PROJECT_PATH = os.path.split(SOURCE_PACKAGE)[0]

RESOURCE_DIR = os.path.join(SOURCE_PACKAGE, 'resources')
IMAGES_DIR = os.path.join(RESOURCE_DIR, 'images')

HUB_EXE = os.path.join(SOURCE_PACKAGE, 'OpenBCIHub', 'OpenBCIHub.exe')
LOG_DIR = os.path.join(PROJECT_PATH, 'logs')
DATA_DIR = os.path.join(PROJECT_PATH, 'data')

TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()
