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
import sys
import time
import urllib.parse

import mne
import numpy as np
from mne.io import read_raw_edf
from tqdm import tqdm

from SystemControl.DataSource.DataSource import SubjectEntry, SampleEntry, EventEntry, DataSource
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type, idx_to_time


def int_to_subject_str(subject_int):
    return f'S{subject_int:03d}'


def int_to_run_str(run_int):
    return f'R{run_int:02d}'


class PhysioDataSource(DataSource):

    def __init__(self, subject: str = None, log_level: str = 'CRITICAL'):
        mne.set_log_level(log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        super().__init__(log_level)

        self.name = 'Physio'
        self.sample_freq = 160
        self.coi = ['C3', 'Cz', 'C4']
        self.subject_names = [int_to_subject_str(each_subject) for each_subject in list(range(1, 110))]
        self.trial_mappings = {
            'baseline_open': [int_to_run_str(each_run) for each_run in [1]],
            'baseline_closed': [int_to_run_str(each_run) for each_run in [2]],
            'motor_execution_right_left': [int_to_run_str(each_run) for each_run in [3, 7, 11]],
            'motor_execution_hands_feet': [int_to_run_str(each_run) for each_run in [5, 9, 13]],
            'motor_imagery_right_left': [int_to_run_str(each_run) for each_run in [4, 8, 12]],
            'motor_imagery_hands_feet': [int_to_run_str(each_run) for each_run in [6, 10, 14]]
        }
        self.trial_types = list(self.trial_mappings.keys())
        self.event_names = ['T0', 'T1', 'T2']
        self.stream_open = False

        self.selected_trial_type = 'motor_imagery_right_left'
        self.ascended_being = subject if subject else self.subject_names[0]

        self._validated = False
        if not self.__validate_dataset():
            self.__download_dataset()

        self.load_data()
        self.set_subject(self.ascended_being)
        return

    def event_name_from_id(self, event_idx) -> str:
        return self.event_names[event_idx - 1]

    def load_data(self):
        if not self.__validate_dataset():
            self.__download_dataset()

        edf_file_list = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        time_start = time.time()
        for each_fname in tqdm(edf_file_list, desc=f'Storing edf file names', file=sys.stdout):
            f_basename, _ = os.path.splitext(os.path.basename(each_fname))
            entry_subject = f_basename[:4]
            entry_trial = f_basename[4:]
            SubjectEntry(path=each_fname, source_name=self.name, subject=entry_subject, trial=entry_trial)
        time_end = time.time()
        print(f'Time to store file names: {time_end - time_start:.4f} seconds')
        return

    def preload_trial(self, trial_info):
        cleaned_edf = self.__clean_raw_edf(trial_info)

        data = cleaned_edf.get_data()
        np_data = np.transpose(data)
        sample_list = [
            SampleEntry(
                idx=sample_idx, time=idx_to_time(sample_idx, self.sample_freq),
                data=each_sample
            )
            for sample_idx, each_sample in enumerate(np_data)
        ]

        events_timings, event_indices = mne.events_from_annotations(cleaned_edf)
        event_list = [
            EventEntry(
                idx=each_event[0], time=idx_to_time(each_event[0], self.sample_freq),
                event_type=self.event_name_from_id(each_event[2])
            )
            for each_event in events_timings
        ]
        return sample_list, event_list

    def preload_user(self, subject: str = None):
        if not subject:
            subject = self.ascended_being

        if subject not in self.subject_names:
            raise ValueError(f'Designated subject is not a valid subject: {subject}')

        subject_entry_list = self.get_subject_entries()
        for subject_entry in subject_entry_list:
            if not subject_entry.is_loaded():
                trial_fname = subject_entry.path
                raw_edf = read_raw_edf(trial_fname, preload=True, verbose='CRITICAL')
                sample_list, event_list = self.preload_trial(raw_edf)
                subject_entry.samples = sample_list
                subject_entry.events = event_list
        return

    def __clean_raw_edf(self, raw_edf):
        cleaned_raw_edf = raw_edf.copy()
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(self.coi)
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def __download_dataset(self, force_download=False, force_unzip=True):
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
            force_download=force_download
        )
        if not zip_name:
            raise RuntimeError(f'Error downloading zip file: {zip_name}')

        unzip_path = unzip_file(zip_name, self.dataset_directory, force_unzip=force_unzip)
        if not unzip_path:
            raise RuntimeError(f'Error unzipping file file: {zip_name}')

        time_start = time.time()
        physio_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        time_end = time.time()
        print(f'Time to find edf files: {time_end - time_start:.4f} seconds')
        print(f'Found {len(physio_files)} EDF files')

        for each_file in physio_files:
            fname = os.path.basename(each_file)
            new_path = os.path.join(self.dataset_directory, fname)
            shutil.move(each_file, new_path)
        print(f'Time to move edf files: {time_end - time_start:.4f} seconds')
        self._edf_file_list = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        return

    def __validate_dataset(self) -> bool:
        if self._validated:
            print('Dataset has already been validated')
            return True

        time_start = time.time()
        found_edf_files = find_files_by_type(file_type='edf', root_dir=self.dataset_directory)
        time_end = time.time()
        print(f'Time to find edf files: {time_end - time_start:.4f} seconds')
        print(f'Found {len(found_edf_files)} EDF files')

        time_start = time.time()
        req_edf_fnames = [
            os.path.join(self.dataset_directory, f'{subject_name}{trial_name}.edf')
            for subject_name in self.subject_names
            for trial_name in [int_to_run_str(each_run) for each_run in range(1, 15)]
        ]
        set_found_files = set(found_edf_files)
        set_req_files = set(req_edf_fnames)
        missing_files = set_req_files - set_found_files
        time_end = time.time()
        print(f'Time to validate dataset: {time_end - time_start:.4f} seconds')
        print(f'Missing {len(missing_files)} file(s) of {len(req_edf_fnames)} required files')

        self._validated = len(missing_files) == 0
        return self._validated


def main():
    """

    :return:
    """
    physio_ds = PhysioDataSource()
    print(physio_ds.subject_names)

    for sample, event in physio_ds:
        print(f'{sample.idx}:{sample.time:<10.4f}::{event.idx:>7d}:{event.time}:{event.type}')
    return


if __name__ == '__main__':
    main()
