"""
@title
    test_utilities.py
@description
    Provides a suite of tests to verify the functionality
    of the modules provided in utilities.

    https://docs.python.org/3/library/unittest.html
"""
import unittest

from SystemControl.SystemLog import SystemLog, SystemLogLevel


class test_utilities(unittest.TestCase):

    def test_log(self):
        """
        https://www.lipsum.com/

        todo docs
        :return:
        """
        short_message = 'short message'
        long_message = 'this is a really really long message that should wrap at least several times. Lorem ipsum ' \
                       'dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et ' \
                       'dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris' \
                       'nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in ' \
                       'voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat ' \
                       'cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.'

        msg_log = SystemLog(log_level=SystemLogLevel.NORMAL)
        msg_log.log_message('identifier', short_message, SystemLogLevel.NORMAL)
        msg_log.log_message('identifier', long_message, SystemLogLevel.NORMAL)
        return


if __name__ == '__main__':
    # Running it with no options is the most terse.
    # Running with a ‘-v’ is more verbose, showing which tests ran.
    unittest.main()
