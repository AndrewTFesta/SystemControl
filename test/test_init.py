"""
@title
    test_init.py
@description
"""
import unittest

from SystemControl import RESOURCE_PATH, PROJECT_PATH, LOG_PATH, HUB_EXE, SOURCE_PATH, VERSION, NAME


class test_init(unittest.TestCase):

    def test_resource_path(self):
        print('Resource path: {}'.format(RESOURCE_PATH))
        return

    def test_project_path(self):
        print('Project path: {}'.format(PROJECT_PATH))
        return

    def test_source_path(self):
        print('Source path: {}'.format(SOURCE_PATH))
        return

    def test_log_path(self):
        print('Log path: {}'.format(LOG_PATH))
        return

    def test_hub_path(self):
        print('Hub path: {}'.format(HUB_EXE))
        return

    def test_name(self):
        print('Name: {}'.format(NAME))
        return

    def test_version(self):
        print('Version: {}'.format(VERSION))
        return


if __name__ == '__main__':
    # Running it with no options is the most terse.
    # Running with a ‘-v’ is more verbose, showing which tests ran.
    unittest.main()
