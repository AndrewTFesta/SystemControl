"""
@title
    test_utilities.py
@description
    Provides a suite of tests to verify the functionality
    of the modules provided in utilities.

    https://docs.python.org/3/library/unittest.html
"""
import shutil
import unittest
from pathlib import Path

from SystemControl import utilities


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


if __name__ == '__main__':
    # Running it with no options is the most terse.
    # Running with a ‘-v’ is more verbose, showing which tests ran.
    unittest.main()
