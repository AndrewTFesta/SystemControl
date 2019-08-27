import os
import shutil

NAME = 'SystemControl'
VERSION = '0.1'

SOURCE_PATH = os.path.split(__file__)[0]
PROJECT_PATH = os.path.split(SOURCE_PATH)[0]

RESOURCE_PATH = os.path.join(SOURCE_PATH, 'resources')

HUB_EXE = os.path.join(SOURCE_PATH, 'OpenBCIHub', 'OpenBCIHub.exe')
LOG_PATH = os.path.join(PROJECT_PATH, 'logs')

TERMINAL_COLUMNS, TERMINAL_ROWS = shutil.get_terminal_size()
