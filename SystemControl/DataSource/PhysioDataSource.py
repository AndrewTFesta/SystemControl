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
import numpy as np
from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF as RawEDF

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type, idx_to_time


def int_to_subject_str(subject_int):
    return f'S{subject_int:03d}'


def int_to_run_str(run_int):
    return f'R{run_int:02d}'


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
    # todo identify problem edf files
    COI = ['C3', 'Cz', 'C4']
    NAME = 'Physio'
    SUBJECT_NAMES = [int_to_subject_str(each_subject) for each_subject in list(range(1, 110))]
    RUN_NAMES = [int_to_run_str(each_run) for each_run in list(range(1, 15))]
    EVENT_NAMES = ['T0', 'T1', 'T2']

    def __init__(self):
        super().__init__()
        self.dataset_directory = os.path.join(DATA_DIR, self.__str__())
        self._data = self.load_data()

        self.sfreq = 160
        return

    def __str__(self):
        return self.NAME

    def __find_edf_file(self, subject_str, run_str):
        edf_basename = f'{subject_str}{run_str}.edf'
        matching_path = None
        for each_file in self._data:
            if each_file.endswith(edf_basename):
                matching_path = each_file
        return matching_path

    def __clean_raw_edf(self, raw_edf):
        cleaned_raw_edf = raw_edf.copy()

        # strip channel names of "." characters
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(self.COI)

        # Apply band-pass filter
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def __get_raw_trial(self, subject: str, run: int) -> RawEDF:
        run_str = int_to_run_str(run)
        edf_fname = self.__find_edf_file(subject, run_str)

        raw_edf = read_raw_edf(edf_fname, preload=True)
        raw_edf = self.__clean_raw_edf(raw_edf)
        return raw_edf

    def __get_baseline(self, subject: str, eyes_open: bool = True) -> RawEDF:
        if eyes_open:
            baseline_edf = self.__get_raw_trial(subject, 1)
        else:
            baseline_edf = self.__get_raw_trial(subject, 2)
        return baseline_edf

    def __get_mi_right_left(self, subject: str) -> RawEDF:
        runs = [4, 8, 12]
        raw_edf = mne.concatenate_raws([self.__get_raw_trial(subject, run_num) for run_num in runs])
        return raw_edf

    def __get_mi_hands_feet(self, subject: str) -> RawEDF:
        runs = [6, 10, 14]
        raw_edf = mne.concatenate_raws([self.__get_raw_trial(subject, run_num) for run_num in runs])
        return raw_edf

    def get_data(self, subject: str) -> list:
        raw_edf = self.__get_mi_right_left(subject)
        data = raw_edf.get_data()
        np_data = np.transpose(data)

        timing_data = [
            {'idx': row_idx, 'time': idx_to_time(row_idx, self.sfreq), 'data': each_row}
            for row_idx, each_row in enumerate(np_data)
        ]
        return timing_data

    def get_events(self, subject) -> list:
        raw_edf = self.__get_mi_right_left(subject)
        events_timings, event_indices = mne.events_from_annotations(raw_edf)
        event_list = list(event_indices.keys())

        timing_list = [
            {
                'idx': each_timing[0],
                'time': idx_to_time(each_timing[0], self.sfreq),
                'event': event_list[int(each_timing[2]) - 1]
            }
            for each_timing in events_timings[1:]
        ]
        return timing_list

    def load_data(self) -> list:
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
            raise RuntimeError(f'Error downloading zip file: {zip_name}')

        unzip_path = unzip_file(zip_name, self.dataset_directory, force_unzip=False)
        if not unzip_path:
            raise RuntimeError(f'Error unzipping file file: {zip_name}')

        time_start = time.time()
        physio_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        time_end = time.time()
        print('Time to find edf files: {:.4f} seconds'.format(time_end - time_start))
        print('Found {} EDF files'.format(len(physio_files)))

        for each_file in physio_files:
            fname = os.path.basename(each_file)
            new_path = os.path.join(self.dataset_directory, fname)
            shutil.move(each_file, new_path)

        moved_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        return moved_files


def main():
    """

    :return:
    """
    physio_ds = PhysioDataSource()
    subject = physio_ds.SUBJECT_NAMES[0]

    data = physio_ds.get_data(subject)
    events = physio_ds.get_events(subject)

    print(data)
    print(events)
    return


if __name__ == '__main__':
    main()
