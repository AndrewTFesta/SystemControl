"""
@title
    utilities.py
@desc
    Provides a suite of utility and convenience functions.
"""
import os
import time
from enum import Enum
from pathlib import Path
from threading import Lock

from SystemControl import LOG_DIR


def build_dir_path(dir_name):
    """
    Builds up the path to the specified directory if it does not exist.

    :param dir_name:
    :return:
    """
    if not dir_name.exists():
        dir_name.mkdir(parents=True, exist_ok=True)
    return


class SystemLogLevel(Enum):
    DEBUG = 0
    LOW = 10
    NORMAL = 20
    MEDIUM = 30
    HIGH = 40
    WARNING = 60
    ERROR = 70

    def __eq__(self, other):
        return self.value == other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __le__(self, other):
        return self.value <= other.value


class SystemLog:

    def __init__(self, log_level: SystemLogLevel = SystemLogLevel.NORMAL, log_location: str = LOG_DIR):
        self.log_lock = Lock()
        self.log_level = log_level
        self.message_log = []

        time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())
        self.log_fname = os.path.join(log_location, 'msg_log_{}.txt'.format(time_stamp))
        build_dir_path(Path(log_location))
        return

    def log_message(self, message_string: str, message_level: SystemLogLevel):
        assert isinstance(message_level, SystemLogLevel)
        log_message = self.__format_message(message_level, message_string).strip()
        if message_level >= self.log_level:
            print(log_message)
        self.__add_to_log(log_message)
        return

    def __add_to_log(self, message_string: str):
        with self.log_lock:
            self.message_log.append(message_string)
        return

    def flush_log(self):
        with self.log_lock:
            with open(self.log_fname, 'a+') as log_file:
                msg_lines = [each_line.strip() for each_line in self.message_log]
                log_file.writelines('\n'.join(msg_lines))
        return

    @staticmethod
    def __format_message(message_type, message_string):
        time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())
        return '[{}] [{}] {}'.format(message_type, time_stamp, message_string)
