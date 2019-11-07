"""
@title
@description
"""
import json
import os
import sys
import time

from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource, SubjectEntry, SampleEntry, EventEntry
from SystemControl.utilities import find_files_by_type, idx_to_time, time_to_idx


class RecordedDataSource(DataSource):

    @property
    def dataset_directory(self):
        return os.path.join(DATA_DIR, self.name)

    def __init__(self, subject: str = None, log_level: str = 'CRITICAL'):
        super().__init__(log_level)

        self.name = 'recorded'
        self.sample_freq = 200
        self.coi = ['C3', 'Cz', 'C4']
        self.subject_names = self.__find_subject_names()
        self.trial_mappings = {
            'baseline': [],
            'motor_imagery': []
        }
        self.trial_types = list(self.trial_mappings.keys())
        self.event_names = ['REST', 'RIGHT', 'LEFT']
        self.stream_open = False

        self.selected_trial_type = 'motor_imagery'
        self.ascended_being = subject if subject else self.subject_names[0]

        self.load_data()
        self.set_subject(self.ascended_being)
        return

    def load_data(self):
        json_file_list = find_files_by_type(file_type='json', root_dir=self.dataset_directory)

        time_start = time.time()
        for each_fname in tqdm(json_file_list, desc=f'Storing json file names', file=sys.stdout):
            parent_dir, basename = os.path.split(each_fname)
            subject_name = os.path.basename(parent_dir)
            trial_info, ext = os.path.splitext(basename)
            trial_info_parts = trial_info.split('_')
            trial_type = '_'.join(trial_info_parts[:-1])
            trial_name = f'{trial_info_parts[-1]}'
            self.trial_mappings[trial_type].append(trial_name)
            SubjectEntry(path=each_fname, source_name=self.name, subject=subject_name, trial=trial_name)
        time_end = time.time()
        print(f'Time to store file names: {time_end - time_start:.4f} seconds')
        return

    def __find_subject_names(self):
        subject_names = os.listdir(self.dataset_directory)
        if not subject_names:
            raise RuntimeError(f'Unable to locate any subjects in directory: {self.dataset_directory}')
        return subject_names

    def preload_trial(self, trial_info):
        trial_samples = trial_info['samples']
        trial_events = trial_info['events']

        sample_list = [
            SampleEntry(idx=sample_idx, time=idx_to_time(sample_idx, self.sample_freq), data=each_sample)
            for sample_idx, each_sample in enumerate(trial_samples)
        ]

        event_list = [
            EventEntry(
                idx=time_to_idx(each_event['time'], self.sample_freq), time=each_event['time'],
                event_type=each_event['type']
            )
            for each_event in trial_events
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
                with open(trial_fname, 'r+') as trial_file:
                    trial_info = json.load(trial_file)
                sample_list, event_list = self.preload_trial(trial_info)
                subject_entry.samples = sample_list
                subject_entry.events = event_list
        return


def main():
    rec_ds = RecordedDataSource()
    print(rec_ds.subject_names)

    for sample, event in rec_ds:
        print(f'{sample.idx}:{sample.time:<10.4f}::{event.idx:>7d}:{event.time}:{event.type}')
    return


if __name__ == '__main__':
    main()
