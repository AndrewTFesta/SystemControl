"""
@title
    utilities.py
@desc
    Provides a suite of utility and convenience functions.
"""
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from IPython import get_ipython
from requests import HTTPError
from tqdm import tqdm

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


def filter_list_of_dicts(list_of_dicts, filter_dict):
    """

    :param list_of_dicts:   list of dictionary entries to filter
    :param filter_dict:     dictionary where each value a list containing values to match
    :return:
    """
    filter_list = []
    for each_entry in list_of_dicts:
        for filter_key, filter_val in filter_dict.items():
            entry_val = each_entry.get(filter_key, None)
            inval_entry = not any(entry_val == each_filter_val for each_filter_val in filter_val)
            if inval_entry:
                break
        else:
            filter_list.append(each_entry)
    return filter_list


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

    filename, ext = os.path.splitext(filename)
    file_list = []
    for root, dirs, files in os.walk(root_dir):
        for each_file in files:
            fname_to_check, file_extension = os.path.splitext(each_file)
            if fname_to_check == filename:
                file_list.append(os.path.join(root, each_file))
    return file_list


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


def get_path_var():
    return os.environ.get('PATH', None)


def in_ipynb() -> bool:
    """
    Checks if the environment of the current runtime is an IPython environment.
    Effectively, this allows for distinguishing if the code is being
    run in a Jupyter Notebook.

    :return: False if the executing environment is not an IPython environment
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
    TODO    docs

    :param line_string:
    :param max_line_length:
    :return:
    """
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


def extract_fname_from_url(url_path):
    """
    todo    docs

    :param url_path:
    :return:
    """
    parsed_url = urlparse(url_path)
    url_path = parsed_url.path
    fname, ext = os.path.splitext(os.path.basename(url_path))
    return fname


def extract_ext_from_url(url_path):
    """
    todo    docs

    :param url_path:
    :return:
    """
    parsed_url = urlparse(url_path)
    url_path = parsed_url.path
    fname, ext = os.path.splitext(url_path)
    return ext


def download_large_file(url_path, save_directory, c_size: int = 512,
                        file_type=None, remote_fname_name=None, force_download=False):
    """
    todo    docs
    TODO    get file type from header
    TODO    get file name from header

    :param url_path:
    :param file_type:
    :param save_directory:
    :param c_size:
    :param remote_fname_name:
    :param force_download:
    :return:
    """
    try:
        print('Starting file download...')
        response = requests.get(url_path, stream=True)
        headers = response.headers

        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        if not remote_fname_name:
            remote_fname_name = extract_fname_from_url(url_path)

        if not file_type:
            file_type = extract_ext_from_url(url_path)

        local_fname = os.path.join(save_directory, '{}{}'.format(remote_fname_name, file_type))
        if os.path.isfile(local_fname):
            print('File already located in default location:\n{}'.format(local_fname))
            if not force_download:
                print('Not re-downloading file')
                return local_fname
            else:
                print('Removing previously downloaded file')
                os.remove(local_fname)

        content_length = int(headers['content-length'])
        part_file = os.path.join(save_directory, '{}.part_{}'.format(remote_fname_name, file_type))
        if os.path.isfile(part_file):
            os.remove(part_file)
        print('Total size of file: {} bytes'.format(content_length))

        unit = 'B'
        unit_scale = True

        pbar_format = 'Downloaded: {percentage:.4f}% {r_bar}'
        download_progress = tqdm(
            total=content_length,
            bar_format=pbar_format,
            unit=unit,
            unit_scale=unit_scale,
            leave=True,
            file=sys.stdout
        )
        with open(part_file, 'wb+') as handle:
            for chunk in response.iter_content(chunk_size=c_size):
                if chunk:  # filter out keep-alive new chunks
                    handle.write(chunk)
                    download_progress.update(c_size)
        download_progress.close()
        print('Time to download: {:.4f} s'.format(download_progress.format_dict['elapsed']))
        os.rename(part_file, local_fname)
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
        local_fname = ''
    except Exception as err:
        print(f'Other error occurred: {err}')
        local_fname = ''
    return local_fname


def unzip_file(zip_filename, save_directory, force_unzip=False, remove_zip=False) -> str:
    """
    todo    docs

    :param zip_filename:
    :param save_directory:
    :param force_unzip:
    :param remove_zip:
    :return:
    """
    print('Unzipping zip file')
    zip_parts = os.path.split(zip_filename)
    zip_name = os.path.splitext(zip_parts[1])
    unzip_dir = os.path.join(save_directory, zip_name[0])

    if os.path.exists(unzip_dir):
        print('Located unzipped directory')
        if not force_unzip:
            print('Not re-unzipping')
            return unzip_dir
        print('Removing unzipped directory')
        shutil.rmtree(unzip_dir)
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(unzip_dir)
    if remove_zip:
        os.remove(zip_filename)  # remove zip file after extracting
        print('Removed temporary zip file')
    return unzip_dir
