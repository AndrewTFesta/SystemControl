"""
@title
    test_ProjectConfig.py
@description
    Provides a suite of tests to verify that the project
    is configured correctly.

    https://docs.python.org/3/library/unittest.html
"""
import os
import sys
import unittest
from pathlib import Path

utils_dir = os.path.split(os.path.dirname(__file__))[0]
sys.path.append(utils_dir)
import src.main.utilities as utilities

PROJ_PROPS = utilities.get_project_props()

REQ_PATHS = [
    PROJ_PROPS['project_dir'] / Path('src/test'),
    PROJ_PROPS['project_dir'] / Path('src'),
    Path(PROJ_PROPS['project_dir'])
]


class test_ProjectConfig(unittest.TestCase):

    def test_python_path(self):
        """
        Makes sure that all required paths are present in the system path variable.

        :return: None
        """
        sys_paths = [Path(each_sys_path) for each_sys_path in sys.path]

        for each_path in REQ_PATHS:
            self.assertTrue(each_path in sys_paths,  msg='Not on path: {0}'.format(each_path))
        return

    def test_version(self):
        """
        Ensures the presence of versioning in the project description file.

        :return: None
        """
        self.assertEqual(PROJ_PROPS['version'], '0.1')
        return

    def test_name(self):
        """
        Ensures the presence of project name in the project description file.

        :return: None
        """
        self.assertEqual(PROJ_PROPS['name'], 'SystemControl')
        return

    def test_source_dir(self):
        """
        Ensures the presence of the relative source directory path in the project description file.

        :return: None
        """
        self.assertEqual(PROJ_PROPS['source_dir'], 'src/main')
        return

    def test_resource_dir(self):
        """
        Ensures the presence of the relative resource directory path in the project description file.

        :return: None
        """
        self.assertEqual(PROJ_PROPS['resource_dir'], 'resources')
        return

    def test_doc_dir(self):
        """
        Ensures the presence of the relative documentation directory path in the project description file.

        :return: None
        """
        self.assertEqual(PROJ_PROPS['docs_dir'], 'docs')
        return

    def test_environment(self):
        """
        Ensures the presence of the relative virtual environment directory path in the project description file.

        :return: None
        """
        self.assertEqual(PROJ_PROPS['environment'], 'venv')
        return


if __name__ == '__main__':
    # Running it with no options is the most terse.
    # Running with a ‘-v’ is more verbose, showing which tests ran.
    unittest.main()
