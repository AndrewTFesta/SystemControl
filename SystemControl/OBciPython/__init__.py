import os

from SystemControl import SOURCE_PACKAGE

OBCI_PYTHON_DIR = os.path.join(SOURCE_PACKAGE, 'OBciPython', )
OPENBCI_HUB_NAME = 'OpenBCIHub.exe'
HUB_EXE = os.path.join(OBCI_PYTHON_DIR, 'OpenBCI_GUI', 'data', 'OpenBCIHub', OPENBCI_HUB_NAME)
