"""
@title
@description
"""
import os
import time
from concurrent.futures.thread import ThreadPoolExecutor

from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource.SqlDb import SqlDb, DataType
from SystemControl.utilities import find_files_by_type


def extract_info_from_file(file_name):
    with open(file_name, 'rb+') as pbci_file_data:
        pbci_bytes = pbci_file_data.read()

    pbci_entry = {
        "fname": file_name,
        "data": pbci_bytes
    }
    return pbci_entry


def executor_callback(callback_args):
    pbar_inst = callback_args.arg
    pbar_inst.update(1)
    return


def main():
    delete_table = False

    pbci_1d_dir = os.path.join(DATA_DIR, 'ProjectBCI', '1D')
    pbci_2d_dir = os.path.join(DATA_DIR, 'ProjectBCI', '2D')

    pbci_1d_files = find_files_by_type(file_type='csv', root_dir=pbci_1d_dir)
    pbci_2d_files = find_files_by_type(file_type='csv', root_dir=pbci_2d_dir)

    # Set up SQL connection
    pbci_1d_table_name = 'pbci_1d_data'
    pbci_2d_table_name = 'pbci_2d_data'

    pbci_table_fields = {
        # 'id': DataType.INTEGER.name,
        'fname': DataType.TEXT.name,
        'data': DataType.BLOB.name
    }

    eeg_db = SqlDb()
    eeg_db.connect()
    tbls = eeg_db.get_tables()
    print('Table names:\n\t{}'.format('\n\t'.join(tbls)))

    print(f'Creating table:\n\t{pbci_1d_table_name}')
    eeg_db.create_table(pbci_1d_table_name, pbci_table_fields)

    print(f'Creating table:\n\t{pbci_2d_table_name}')
    eeg_db.create_table(pbci_2d_table_name, pbci_table_fields)

    tbls = eeg_db.get_tables()
    print('Table names:\n\t{}'.format('\n\t'.join(tbls)))

    # Read mat files, extract useful info, and add to eeg sql db
    read_pbar = tqdm(total=len(pbci_1d_files))
    read_pbar.set_description('Preparing project bci 1D files')
    time_start = time.time()

    entry_1d_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for each_file in pbci_1d_files:
            extraction_exec = executor.submit(extract_info_from_file, each_file)
            extraction_exec.arg = read_pbar
            extraction_exec.add_done_callback(executor_callback)
            exec_result = extraction_exec.result()
            entry_1d_list.append(exec_result)
    read_pbar.close()
    time_end = time.time()
    time.sleep(0.1)
    print('Time to prepare 1D entries: {:.4f} seconds'.format(time_end - time_start))
    time.sleep(0.1)

    read_pbar = tqdm(total=len(pbci_2d_files))
    read_pbar.set_description('Preparing project bci 2D files')
    time_start = time.time()
    entry_2d_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for each_file in pbci_2d_files:
            extraction_exec = executor.submit(extract_info_from_file, each_file)
            extraction_exec.arg = read_pbar
            extraction_exec.add_done_callback(executor_callback)
            exec_result = extraction_exec.result()
            entry_2d_list.append(exec_result)
    read_pbar.close()
    time_end = time.time()
    time.sleep(0.1)
    print('Time to prepare 2D entries: {:.4f} seconds'.format(time_end - time_start))
    time.sleep(0.1)

    time_start = time.time()
    eeg_db.insert_multiple(pbci_1d_table_name, entry_1d_list)
    time_end = time.time()
    print('Time to insert 1D data into db: {:.4f} seconds'.format(time_end - time_start))

    time_start = time.time()
    eeg_db.insert_multiple(pbci_2d_table_name, entry_2d_list)
    time_end = time.time()
    print('Time to insert 2D data into db: {:.4f} seconds'.format(time_end - time_start))

    if delete_table:
        print(f'Deleting table: {pbci_1d_table_name}')
        eeg_db.delete_table(pbci_1d_table_name)

        print(f'Deleting table: {pbci_2d_table_name}')
        eeg_db.delete_table(pbci_2d_table_name)

    print('Closing sql connection')
    eeg_db.close()
    return


if __name__ == '__main__':
    main()
