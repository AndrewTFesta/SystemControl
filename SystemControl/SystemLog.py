import os
import time
from enum import Enum, auto
from pathlib import Path
from threading import Lock

from SystemControl.utilities import split_paragraph, build_dir_path
from SystemControl import LOG_DIR


class SystemLogLevel(Enum):
    """
    todo docs
    """
    DEBUG = 0
    LOW = 10
    NORMAL = 20
    MEDIUM = 30
    HIGH = 40
    WARNING = 60
    ERROR = 70
    QUIET = 80

    def __eq__(self, other):
        """
        todo docs
        :param other:
        :return:
        """
        return self.value == other.value

    def __ge__(self, other):
        """
        todo docs
        :param other:
        :return:
        """
        return self.value >= other.value

    def __le__(self, other):
        """
        todo docs
        :param other:
        :return:
        """
        return self.value <= other.value


class SystemLogIdent(Enum):
    LOG = auto()
    SND = auto()
    RCV = auto()


class SystemLog:
    """
    todo docs
    """

    def __init__(self, log_level: SystemLogLevel = SystemLogLevel.NORMAL, log_location: str = LOG_DIR):
        """
        todo docs
        :param log_level:
        :param log_location:
        """
        self.log_lock = Lock()
        self.log_level = log_level
        self.message_log = []

        time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())
        self.log_fname = os.path.join(log_location, 'msg_log_{}.txt'.format(time_stamp))
        build_dir_path(Path(log_location))
        return

    def log_message(self, message_ident: SystemLogIdent, message_string: str, message_level: SystemLogLevel):
        """
        todo docs
        :param message_ident:
        :param message_string:
        :param message_level:
        :return:
        """
        assert isinstance(message_level, SystemLogLevel)
        log_message = self.__format_message(message_ident, message_string, message_level).strip()
        if message_level >= self.log_level:
            print(log_message)
        self.__add_to_log(log_message)
        return

    def __add_to_log(self, log_message: str) -> None:
        """
        todo docs
        :param log_message:
        :return:
        """
        with self.log_lock:
            self.message_log.append(log_message)
        return

    def flush_log(self) -> None:
        """
        todo docs
        :return:
        """
        with self.log_lock:
            with open(self.log_fname, 'a+') as log_file:
                msg_lines = [each_line.strip() for each_line in self.message_log]
                log_file.writelines('\n'.join(msg_lines))
        return

    @staticmethod
    def __format_message(message_ident: SystemLogIdent, message_string: str, message_level: SystemLogLevel,
                         line_length: int = 140) -> str:
        time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())
        wrapped_lines = split_paragraph(message_string, line_length)
        tab_lines_str = '\t' + '\n\t'.join(wrapped_lines)
        return '[{}] [{}] [{}]\t{}'.format(message_level.name, message_ident.name, time_stamp, tab_lines_str)
