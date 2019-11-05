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

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource
from SystemControl.utilities import download_large_file, unzip_file, find_files_by_type, idx_to_time, \
    filter_list_of_dicts


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

    def __init__(self, chosen_subject: str = None, log_level: str = 'CRITICAL'):
        mne.set_log_level(log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        super().__init__(log_level)
        self.dataset_directory = os.path.join(DATA_DIR, self.name)

        self.__baseline_trials = {
            'open': [int_to_run_str(each_run) for each_run in [1]],
            'closed': [int_to_run_str(each_run) for each_run in [2]],
        }
        self.__motor_execution_trials = {
            'right_left': [int_to_run_str(each_run) for each_run in [3, 7, 11]],
            'hands_feet': [int_to_run_str(each_run) for each_run in [5, 9, 13]]
        }
        self.__motor_imagery_trials = {
            'right_left': [int_to_run_str(each_run) for each_run in [4, 8, 12]],
            'hands_feet': [int_to_run_str(each_run) for each_run in [6, 10, 14]]
        }

        self._subject_data = {
            # entry -> {'path': str, 'trial': str, 'samples': list}
            subject_name: []
            for subject_name in self.subject_names
        }

        self._edf_file_list = []
        self._validated = False
        if not self.__validate_dataset():
            self.__download_dataset()

        self.load_data()

        self.selected_trials = self.__motor_imagery_trials['right_left']
        self.ascended_being = self.subject_names[0]
        if chosen_subject in self.subject_names:
            self.ascended_being = chosen_subject

        self.__preload_user()
        return

    def __iter__(self):
        trials = self.get_trials()
        for each_trial in trials:
            trial_samples = each_trial['samples']
            for each_sample in trial_samples:
                yield each_sample, each_trial['trial']
        pass

    def append_sample(self):
        raise NotImplementedError('Adding samples to this type of DataSource is currently not supported')

    def set_subject(self, subject: str):
        if subject not in self.subject_names:
            raise ValueError(f'Designated subject is not a valid subject: {subject}')

        self.ascended_being = subject
        self.__preload_user()
        return

    @property
    def sample_freq(self) -> float:
        return 160.

    @property
    def name(self) -> str:
        return 'Physio'

    @property
    def subject_names(self) -> list:
        return [int_to_subject_str(each_subject) for each_subject in list(range(1, 110))]

    @property
    def coi(self) -> list:
        return ['C3', 'Cz', 'C4']

    @property
    def trial_names(self) -> list:
        return [int_to_run_str(each_run) for each_run in list(range(1, 15))]

    @property
    def event_names(self) -> list:
        return ['T0', 'T1', 'T2']

    def __clean_raw_edf(self, raw_edf):
        cleaned_raw_edf = raw_edf.copy()

        # strip channel names of "." characters
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(self.coi)

        # Apply band-pass filter
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def __preload_user(self, subject: str = None):
        if not subject:
            subject = self.ascended_being

        if subject not in self.subject_names:
            raise ValueError(f'Designated subject is not a valid subject: {subject}')

        subject_entries = self._subject_data[subject]
        relevant_trials = filter_list_of_dicts(subject_entries, {'trial': self.selected_trials})

        for each_trial in relevant_trials:
            edf_path = each_trial['path']
            raw_edf = read_raw_edf(edf_path, preload=True, verbose='CRITICAL')
            sample_list = self.__preload_trial(raw_edf)
            each_trial['samples'] = sample_list
        return

    def __preload_trial(self, trial_edf):
        cleaned_edf = self.__clean_raw_edf(trial_edf)

        data = cleaned_edf.get_data()
        np_data = np.transpose(data)

        events_timings, event_indices = mne.events_from_annotations(cleaned_edf)
        sorted_event_times = events_timings[events_timings[:, 0].argsort()]
        event_onset_list = sorted_event_times[:, 0]

        next_event_idx = 0
        timing_list = []
        for sample_idx, each_sample in enumerate(np_data):
            if sample_idx in event_onset_list:
                next_event_idx += 1
            curr_event = sorted_event_times[next_event_idx - 1]

            timing_entry = {
                'idx': sample_idx,
                'time': idx_to_time(sample_idx, self.sample_freq),
                'event': curr_event,
                'data': each_sample
            }
            timing_list.append(timing_entry)
        return timing_list

    def stream_interpolation(self):
        # todo
        return

    def stream_heatmap(self):
        # todo
        return

    def stream_trials(self):
        # todo
        return

    def get_trials(self) -> list:
        subject = self.ascended_being
        subject_entries = self._subject_data[subject]
        relevant_trials = filter_list_of_dicts(subject_entries, {'trial': self.selected_trials})
        return relevant_trials

    def event_name_from_idx(self, event_idx) -> str:
        return self.event_names[event_idx - 1]

    def _load_file(self, fname):
        f_basename, _ = os.path.splitext(os.path.basename(fname))
        entry_subject = f_basename[:4]
        entry_trial = f_basename[4:]
        sample_entry = {'path': fname, 'trial': entry_trial, 'samples': None}

        subject_data_list = self._subject_data.get(entry_subject, None)
        if isinstance(subject_data_list, list):
            subject_data_list.append(sample_entry)
        return

    def load_data(self):
        if not self.__validate_dataset():
            self.__download_dataset()

        time_start = time.time()
        for each_fname in tqdm(self._edf_file_list, desc=f'Reading edf files', file=sys.stdout):
            self._load_file(each_fname)
        time_end = time.time()
        print(f'Time to load subject data: {time_end - time_start:.4f} seconds')
        return

    def save_data(self):
        raise NotImplementedError('Saving this type of DataSource is currently not supported')

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
            for trial_name in self.trial_names
        ]
        set_found_files = set(found_edf_files)
        set_req_files = set(req_edf_fnames)
        missing_files = set_req_files - set_found_files
        time_end = time.time()
        print(f'Time to validate dataset: {time_end - time_start:.4f} seconds')
        print(f'Missing {len(missing_files)} file(s) of {len(req_edf_fnames)} required files')

        self._edf_file_list = found_edf_files
        self._validated = len(missing_files) == 0
        return self._validated

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


def main():
    """

    :return:
    """
    physio_ds = PhysioDataSource()

    for sample, trial_name in physio_ds:
        sample_data = sample['data']
        sample_event = sample['event']
        event_name = physio_ds.event_name_from_idx(sample_event[2])

        print(f'{trial_name}: {sample["time"]}: {event_name}: {sample_data}')
    return


if __name__ == '__main__':
    main()
