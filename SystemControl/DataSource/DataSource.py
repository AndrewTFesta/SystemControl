"""
@title
@description
"""
import hashlib
import os
import time
from collections import namedtuple

import pandas as pd

from SystemControl import DATA_DIR
from SystemControl.utils.ObserverObservable import Observable
from SystemControl.utils.utilities import select_skip_generator


def build_entry_id(data_list):
    id_str = ''
    sep = ''
    for data_val in data_list:
        id_str += f'{sep}{data_val}'
        sep = ':'
    id_bytes = id_str.encode('utf-8')
    entry_hash = hashlib.sha3_256(id_bytes).hexdigest()
    return entry_hash


TrialInfoEntry = namedtuple('TrialInfoEntry', 'entry_id source subject trial_type trial_name')
TrialDataEntry = namedtuple('TrialDataEntry', 'entry_id idx timestamp label C3 Cz C4')


class DataSource(Observable):

    def __init__(self, log_level: str = 'WARNING', save_method: str = 'csv'):
        Observable.__init__(self)
        self._log_level = log_level
        self.save_method = save_method

        self.name = ''
        self.sample_freq = -1
        self.subject_names = []
        self.trial_types = []
        self.event_names = []
        self.ascended_being = ''
        self.selected_trial_type = ''

        self.trial_info_df = None
        self.trial_data_df = None

        self.coi = ['C3', 'Cz', 'C4']
        return

    def __iter__(self):
        trial_samples_list = self.get_trial_samples()
        for trial_samples in trial_samples_list:
            for index, sample in trial_samples.iterrows():
                yield sample
        return

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return self.__str__()

    @property
    def dataset_directory(self):
        return os.path.join(DATA_DIR, self.name)

    @property
    def trial_info_file(self):
        if self.save_method == 'csv':
            path = os.path.join(self.dataset_directory, f'trial_info.csv')
        elif self.save_method == 'h5':
            path = os.path.join(self.dataset_directory, f'trial_info.h5')
        else:
            print(f'Unable to save data: unrecognized file format: {self.save_method}')
            path = ''
        return path

    @property
    def trial_data_file(self):
        if self.save_method == 'csv':
            path = os.path.join(self.dataset_directory, f'trial_data.csv')
        elif self.save_method == 'h5':
            path = os.path.join(self.dataset_directory, f'trial_data.h5')
        else:
            print(f'Unable to save data: unrecognized file format: {self.save_method}')
            path = ''
        return path

    def downsample_generator(self, skip_amount: int = 2):
        base_iter = self.__iter__()
        downsampled = select_skip_generator(base_iter, select=1, skip=skip_amount - 1)
        for sample in downsampled:
            yield sample

    def window_generator(self, window_length: float, window_overlap: float):
        window_start = 0
        window_end = window_length
        window_offset = (1 - window_overlap) * window_length

        trial_samples_list = self.get_trial_samples()
        for trial_samples in trial_samples_list:
            last_time = trial_samples['timestamp'].max()

            while window_end < last_time:
                next_window = trial_samples.loc[
                    (trial_samples['timestamp'] >= window_start) &
                    (trial_samples['timestamp'] <= window_end)
                    ]
                window_start += window_offset
                window_end += window_offset
                yield next_window
        return

    def get_num_samples(self):
        total_count = 0
        subject_entry_list = self.get_trial_samples()
        for subject_entry in subject_entry_list:
            sample_list = subject_entry["samples"]
            last_sample = sample_list[-1]
            last_sample_idx = last_sample["idx"]
            total_count += last_sample_idx + 1
        return total_count

    def get_trial_samples(self) -> list:
        current_trials = self.trial_info_df.loc[
            (self.trial_info_df['subject'] == self.ascended_being) &
            (self.trial_info_df['trial_type'] == self.selected_trial_type)
            ]
        trial_samples = []
        for index, row in current_trials.iterrows():
            row_id = row['entry_id']
            id_samples = self.trial_data_df.loc[self.trial_data_df['entry_id'] == row_id]
            trial_samples.append(id_samples)
        return trial_samples

    def load_data(self):
        print('Loading dataset')
        if not os.path.isfile(self.trial_info_file):
            print(f'Unable to locate trial info file: {self.trial_info_file}')
            self.trial_info_df = pd.DataFrame(columns=TrialInfoEntry._fields)
            self.trial_data_df = pd.DataFrame(columns=TrialDataEntry._fields)
            return
        if not os.path.isfile(self.trial_data_file):
            print(f'Unable to locate trial data file: {self.trial_info_file}')
            self.trial_info_df = pd.DataFrame(columns=TrialInfoEntry._fields)
            self.trial_data_df = pd.DataFrame(columns=TrialDataEntry._fields)
            return

        time_start = time.time()
        if self.save_method == 'csv':
            self.trial_info_df = pd.read_csv(self.trial_info_file)
            self.trial_data_df = pd.read_csv(self.trial_data_file)

        elif self.save_method == 'h5':
            self.trial_info_df = pd.read_hdf(self.trial_info_file, key='physio_trial_info')
            self.trial_data_df = pd.read_hdf(self.trial_data_file, key='physio_trial_data')
        else:
            print(f'Unable to save data: unrecognized file format: {self.save_method}')
        time_end = time.time()
        print(f'Time to load info and data: {time_end - time_start:.4f} seconds')
        return

    def set_subject(self, subject: str):
        if subject not in self.subject_names:
            raise ValueError(f'Designated subject is not a valid subject: {subject}')

        self.ascended_being = subject
        return

    def set_trial_type(self, trial_type: str):
        if trial_type not in self.trial_types:
            raise ValueError(f'Designated trial is not a valid trial type: {trial_type}')

        self.selected_trial_type = trial_type
        return

    def save_data(self, start_time: float = 0.0, end_time: float = -1):
        if not os.path.isdir(self.dataset_directory):
            os.makedirs(self.dataset_directory)
        print(f'Saving {len(self.trial_info_df)} trials using method: {self.save_method}')

        save_data_df = self.trial_data_df
        if start_time >= 0:
            save_data_df = save_data_df.loc[save_data_df['timestamp'] >= start_time]
        if end_time >= 0:
            save_data_df = save_data_df[save_data_df['timestamp'] <= end_time]

        time_start = time.time()
        if self.save_method == 'csv':
            self.trial_info_df.to_csv(self.trial_info_file, index=False)
            save_data_df.to_csv(self.trial_data_file, index=False)
        elif self.save_method == 'h5':
            self.trial_info_df.to_hdf(self.trial_info_file, key='physio_trial_info')
            save_data_df.to_hdf(self.trial_data_file, key='physio_trial_data')
        else:
            print(f'Unable to save data: unrecognized file format: {self.save_method}')
        time_end = time.time()
        print(f'Time to save info and data: {time_end - time_start:.4f} seconds')
        return
