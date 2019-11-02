"""
@title
@description

    |---------------------------------------------------|
    | run         | task                                |
    |---------------------------------------------------|
    | 1           | Baseline, eyes open                 |
    | 2           | Baseline, eyes closed               |
    | 3, 7, 11    | Motor execution: left vs right hand |
    | 4, 8, 12    | Motor imagery: left vs right hand   |
    | 5, 9, 13    | Motor execution: hands vs feet      |
    | 6, 10, 14   | Motor imagery: hands vs feet        |
    |---------------------------------------------------|

    T0  rest
    T1  onset of motion (real or imagined)
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
    T2  onset of motion (real or imagined)
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
"""
import os
import shutil
import time
from enum import Enum

import mne
import urllib.parse
from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF as RawEDF

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type


def int_to_subject_str(subject_int):
    return f'S{subject_int:03d}'


def int_to_run_str(run_int):
    return f'R{run_int:02d}'


class PhysioEvent(Enum):
    T0 = 0
    T1 = 1
    T2 = 2


class PhysioEntry:

    def __init__(self, fname, subject, run, data):
        self.fname = fname
        self.subject = subject
        self.run = run
        self.data = data
        return


SUBJECT_NUMS = list(range(1, 109))
SUBJECT_NAMES = [int_to_subject_str(each_subject) for each_subject in SUBJECT_NUMS]
RUN_NUMS = list(range(1, 14))
RUN_NAMES = [int_to_run_str(each_run) for each_run in RUN_NUMS]


def extract_info_from_fname(file_name):
    with open(file_name, 'rb+') as physio_file_data:
        physio_bytes = physio_file_data.read()

    file_info, _ = os.path.splitext(os.path.basename(file_name))
    subject_name = file_info[:4]
    run_num = file_info[4:]

    physio_entry = {
        'fname': file_name,
        'subject': subject_name,
        'run': run_num,
        'data': physio_bytes
    }
    return physio_entry


class PhysioDataSource(DataSource):
    COI = ['C3', 'Cz', 'C4']
    NAME = 'Physio'

    def __init__(self):
        super().__init__()
        self.dataset_directory = os.path.join(DATA_DIR, self.__str__())
        self._data = self.__load_data()
        return

    def __str__(self):
        return self.NAME

    def __load_data(self):
        # todo add check if files already exist on disk
        external_name = 'eeg-motor-movementimagery-dataset-1.0.0'
        external_zip_url = urllib.parse.urljoin(
            'https://physionet.org/static/published-projects/eegmmidb/',
            f'{external_name}.zip'
        )
        zip_name = download_large_file(
            external_zip_url,
            self.dataset_directory,
            c_size=512,
            file_type=None,
            remote_fname_name=None,
            force_download=False
        )
        if not zip_name:
            print(f'Error downloading zip file: {zip_name}')
            return

        unzip_path = unzip_file(zip_name, self.dataset_directory, force_unzip=False)
        if not unzip_path:
            print(f'Error unzipping file file: {zip_name}')
            return

        time_start = time.time()
        physio_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        time_end = time.time()
        print('Time to find edf files: {:.4f} seconds'.format(time_end - time_start))
        print('Found {} EDF files'.format(len(physio_files)))

        for each_file in physio_files:
            fname = os.path.basename(each_file)
            new_path = os.path.join(self.dataset_directory, fname)
            shutil.move(each_file, new_path)
        # shutil.rmtree(os.path.join(self.dataset_directory, external_name))

        moved_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        return moved_files

    @staticmethod
    def validate_subject_num(subject_num) -> bool:
        return subject_num in SUBJECT_NUMS

    @staticmethod
    def clean_raw_edf(raw_edf):
        cleaned_raw_edf = raw_edf.copy()

        # strip channel names of "." characters
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(PhysioDataSource.COI)

        # Apply band-pass filter
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def __find_edf_file(self, subject_str, run_str):
        edf_basename = f'{subject_str}{run_str}.edf'
        matching_path = None
        for each_file in self._data:
            if each_file.endswith(edf_basename):
                matching_path = each_file
        return matching_path

    def get_raw_trial(self, subject: int, run: int) -> RawEDF:
        if not isinstance(subject, int):
            raise TypeError(f'Invalid type of subject parameter: {type(subject)}')
        if not isinstance(run, int):
            raise TypeError(f'Invalid type of run parameter: {type(run)}')

        if subject not in SUBJECT_NUMS:
            raise ValueError(f'Invalid subject specified: \'{subject}\'')
        if run not in RUN_NUMS:
            raise ValueError(f'Invalid run specified: \'{run}\'')

        subject_str = int_to_subject_str(subject)
        run_str = int_to_run_str(run)

        edf_fname = self.__find_edf_file(subject_str, run_str)
        raw_edf = read_raw_edf(edf_fname, preload=True)
        raw_edf = self.clean_raw_edf(raw_edf)
        return raw_edf

    def get_mi_df(self, subject: int = None, trial: int = None):
        raw_edf = self.get_mi_right_left(subject)
        raw_df = raw_edf.to_data_frame()
        return raw_df

    def get_baseline(self, subject, eyes_open: bool = True) -> RawEDF:
        if eyes_open:
            baseline_edf = self.get_raw_trial(subject, 1)
        else:
            baseline_edf = self.get_raw_trial(subject, 2)
        return baseline_edf

    def get_mi_right_left(self, subject) -> RawEDF:
        runs = [4, 8, 12]
        raw_edf = mne.concatenate_raws([self.get_raw_trial(subject, run_num) for run_num in runs])
        return raw_edf

    def get_mi_hands_feet(self, subject) -> RawEDF:
        runs = [6, 10, 14]
        raw_edf = mne.concatenate_raws([self.get_raw_trial(subject, run_num) for run_num in runs])
        return raw_edf

    @staticmethod
    def get_data_frame(raw_edf: RawEDF) -> list:
        raw_df = raw_edf.to_data_frame()
        return raw_df

    @staticmethod
    def get_data(raw_edf: RawEDF) -> list:
        raw_data = raw_edf.get_data()
        data_list = []
        for each_row in raw_data:
            data_list.append(each_row)
        return data_list

    @staticmethod
    def get_annotations(raw_edf: RawEDF):
        annotations = raw_edf.annotations
        return annotations

    @staticmethod
    def get_events(raw_edf: RawEDF):
        events = mne.events_from_annotations(raw_edf)
        return events

    @staticmethod
    def get_channel_names(raw_edf: RawEDF):
        ch_names = raw_edf.ch_names
        return ch_names

    @staticmethod
    def idx_to_time(raw_edf: RawEDF, idx) -> float:
        sfreq = raw_edf.info['sfreq']
        time_val = idx / sfreq
        return time_val


def main():
    """

    :return:
    """
    subject = 1

    physio_ds = PhysioDataSource()
    raw_data = physio_ds.get_mi_right_left(subject=subject)
    print(f'Raw data is type: {type(raw_data)}')

    annots = PhysioDataSource.get_annotations(raw_data)
    events = physio_ds.get_events(raw_data)
    return


if __name__ == '__main__':
    main()
