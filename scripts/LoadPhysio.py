"""
@title
@description
"""
import os
import time
import urllib.parse
from concurrent.futures.thread import ThreadPoolExecutor

from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource
from SystemControl.DataSource.SqlDb import SqlDb, DataType
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type


def extract_info_from_file(file_name):
    with open(file_name, 'rb+') as physio_file_data:
        physio_bytes = physio_file_data.read()

    # verbosity: str = 'critical'
    # raw_edf = read_raw_edf(each_file, preload=True, verbose=verbosity)

    file_info, _ = os.path.splitext(os.path.basename(file_name))
    subject_name = file_info[:4]
    run_num = file_info[4:]

    physio_entry = {
        "fname": file_name,
        "subject": subject_name,
        "run": run_num,
        "data": physio_bytes
    }
    return physio_entry


def executor_callback(callback_args):
    pbar_inst = callback_args.arg
    pbar_inst.update(1)
    return


def main():
    """

    :return:
    """
    dataset_name = 'eeg-motor-movementimagery-dataset-1.0.0'
    external_zip_url = urllib.parse.urljoin(
        'https://physionet.org/static/published-projects/eegmmidb/',
        '{}.zip'.format(dataset_name)
    )
    dataset_location = os.path.join(DATA_DIR, dataset_name)
    dataset_dir = os.path.join(dataset_location, dataset_name)
    delete_table = False

    zip_name = download_large_file(
        external_zip_url,
        dataset_location,
        c_size=512,
        file_type=None,
        remote_fname_name=None,
        force_download=False
    )
    if not zip_name:
        # todo print relevant error and exit
        print()
        return

    unzip_path = unzip_file(zip_name, dataset_location, force_unzip=False)
    if not unzip_path:
        # todo print relevant error and exit
        print()
        return

    time_start = time.time()
    physio_files = find_files_by_type(file_type='edf', root_dir=dataset_dir)
    time_end = time.time()
    print('Time to find edf files: {:.4f} seconds'.format(time_end - time_start))
    print('Found {} EDF files'.format(len(physio_files)))

    # Set up SQL connection
    physio_table_fields = {
        # 'id': DataType.INTEGER.name,
        'fname': DataType.TEXT.name,
        'subject': DataType.TEXT.name,
        'run': DataType.TEXT.name,
        'data': DataType.BLOB.name
    }

    eeg_db = SqlDb()
    eeg_db.connect()
    tbls = eeg_db.get_tables()
    print('Table names:\n\t{}'.format('\n\t'.join(tbls)))
    print(f'Creating table:\n\t{PhysioDataSource.PHYSIO_TABLE_NAME}')
    eeg_db.create_table(PhysioDataSource.PHYSIO_TABLE_NAME, physio_table_fields)

    tbls = eeg_db.get_tables()
    print('Table names:\n\t{}'.format('\n\t'.join(tbls)))

    # Read edf files, extract useful info, and add to eeg sql db
    time_start = time.time()
    read_pbar = tqdm(total=len(physio_files))
    read_pbar.set_description('Reading physio edf files')

    entry_list = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for each_file in physio_files:
            extraction_exec = executor.submit(extract_info_from_file, each_file)
            extraction_exec.arg = read_pbar
            extraction_exec.add_done_callback(executor_callback)
            exec_result = extraction_exec.result()
            entry_list.append(exec_result)
    read_pbar.close()
    time_end = time.time()
    time.sleep(0.1)
    print('Time to prepare entries: {:.4f} seconds'.format(time_end - time_start))

    time_start = time.time()
    eeg_db.insert_multiple(PhysioDataSource.PHYSIO_TABLE_NAME, entry_list)
    time_end = time.time()
    print('Time to insert into db: {:.4f} seconds'.format(time_end - time_start))

    if delete_table:
        print(f'Deleting table: {PhysioDataSource.PHYSIO_TABLE_NAME}')
        eeg_db.delete_table(PhysioDataSource.PHYSIO_TABLE_NAME)

    print('Closing sql connection')
    eeg_db.close()
    return


if __name__ == '__main__':
    main()
