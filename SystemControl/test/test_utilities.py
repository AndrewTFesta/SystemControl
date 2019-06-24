"""
@title
    test_utilities.py
@description
    Provides a suite of tests to verify the functionality
    of the modules provided in utilities.

    https://docs.python.org/3/library/unittest.html
"""
import sys
from os.path import dirname
import os
import shutil
import unittest
from pathlib import Path

util_dir = os.path.split(dirname(__file__))[0]
sys.path.append(util_dir)
import src.main.utilities as utilities

PROJ_PROPS = utilities.get_project_props()
TEST_DIR = PROJ_PROPS['project_dir'] / Path(PROJ_PROPS['test_dir'])


class test_utilities(unittest.TestCase):

    def test_build_dir_path(self):
        """
        Attempts to build a path which should not exist from
        the current directory. It then checks to make sure
        the directory path was correctly created.

        This directory is not a name of the file to be located at
        the provided location. It is only the path to the directory.

        After running, this test cleans up the created directory path.

        :return: None
        """
        path_parts = ['this', 'should', 'not', 'exist']
        non_exist_dir = Path(TEST_DIR, *path_parts)
        if non_exist_dir.exists():
            shutil.rmtree(non_exist_dir)

        # ensures permissions
        self.assertFalse(non_exist_dir.exists())

        utilities.build_dir_path(non_exist_dir)
        self.assertTrue(non_exist_dir.exists())

        # clean up created directory and make sure
        # operation was successful
        shutil.rmtree(Path(TEST_DIR, path_parts[0]))
        self.assertFalse(Path(TEST_DIR, path_parts[0]).exists())
        return

    def test_get_project_props(self):
        """
        Ensures that the minimal required properties are present in the
        project properties file.

        It is expected that a project always contains this file
        in the project directory.

        :return: None
        """
        proj_props = utilities.get_project_props()

        # check for type and that the dictionary contains
        # the least amount of information required
        self.assertIsInstance(proj_props, dict)
        self.assertIn('name', proj_props)
        self.assertIn('version', proj_props)
        self.assertIn('source_dir', proj_props)
        self.assertIn('resource_dir', proj_props)
        self.assertIn('environment', proj_props)
        self.assertIn('docs_dir', proj_props)
        return

    def test_print_version(self):
        """
        Checks that the returned name and version of the current
        project are the correct values.

        Name: SystemControl
        Version: 0.1

        :return: None
        """
        proj_name, proj_version = utilities.get_version()

        # check type and return values
        self.assertIsInstance(proj_name, str)
        self.assertIsInstance(proj_version, str)

        self.assertEqual(proj_name, 'SystemControl')
        self.assertEqual(proj_version, '0.1')
        return


if __name__ == '__main__':
    # Running it with no options is the most terse.
    # Running with a ‘-v’ is more verbose, showing which tests ran.
    unittest.main()
