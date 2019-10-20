"""
@title
    utilities.py
@desc
    Provides a suite of utility and convenience functions.
"""
import os
import time
from pathlib import Path

from IPython import get_ipython

from SystemControl import SOURCE_PACKAGE


def time_function(func: classmethod, *args, **kwargs) -> tuple:
    """
    Measures the clock time it takes for the provided function and
    list of parameters to run.

    If a list of arguments is provided, this list is passed directly to the
    called function as positional argument. Otherwise, no positional
    arguments are passed to the function.

    :param func: function to time
    :param args: (optionals) list of parameters to pass to the function
    :return: tuple containing the function return value and the time
    it took for the function to execute
    """
    start_time = time.time()
    func_ret = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return func_ret, elapsed


def build_dir_path(dir_name: Path) -> None:
    """
    Builds up the path to the specified directory if it does not exist.

    :type dir_name: Path
    :param dir_name: name of directory to create
    :return:
    """
    if not dir_name.exists():
        dir_name.mkdir(parents=True, exist_ok=True)
    return


def find_files_by_name(filename: str, root_dir: str = None) -> list:
    """
    Locates all files of the specified name under the supplied
    root directory. If no root is supplied, the base directory
    of the project of the currently running script is used.

    :type filename: str
    :param filename: name of the file to locate
    :type root_dir: str
    :param root_dir: (optional) root directory to search from
    :return: absolute path to the located files
    """
    if not root_dir:
        root_dir = SOURCE_PACKAGE
        # root_dir = os.path.dirname(__file__)

    found_file_list = []
    for root, dirs, files in os.walk(root_dir):
        f_to_chk = os.path.join(root, filename)
        if os.path.isfile(f_to_chk):
            found_file_list.append(f_to_chk)
    return found_file_list


def find_files_by_type(file_type: str, root_dir: str = None) -> list:
    """
    Locates all files of the specified extension under the supplied
    root directory. If no root is supplied, the base directory
    of the project of the currently running script is used.

    :type file_type: str
    :param file_type:
    :type root_dir: str
    :param root_dir: (optional) root directory to search from
    :return: absolute path to the located files
    """
    if not file_type.startswith('.'):
        file_type = '.{}'.format(file_type)

    if not root_dir:
        root_dir = __file__

    file_list = []
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            filename, file_extension = os.path.splitext(file)
            if file_extension == file_type:
                file_list.append(os.path.join(root, file))
    return file_list


def in_ipynb() -> bool:
    """
    Checks if the environment of the current runtime is an IPython environment.
    Effectively, this allows for distinguishing if the code is being
    run in a Jupyter Notebook.

    :return: False if the executing environment is not an
    IPython environment
    """
    try:
        cfg = get_ipython()
        if cfg:
            return True
        else:
            return False
    except NameError:
        return False


def split_paragraph(line_string: str, max_line_length: int = 60) -> list:
    """
    TODO split paragraph docs

    :param line_string:
    :param max_line_length:
    :return:
    """
    # TODO bug fix: wrap when line does not contain spaces
    word_list = line_string.split(' ')
    line_list = []
    current_line = ''
    len_counter = 0
    for each_word in word_list:
        each_word += ' '
        if len_counter + len(each_word) > max_line_length:
            line_list.append(current_line)
            len_counter = 0
            current_line = ''
        current_line += each_word
        len_counter += len(each_word)
    if current_line:
        line_list.append(current_line)
    line_list = [each_line.strip() for each_line in line_list]
    return line_list
