"""
@title
@description
"""
import os
import threading
import time
from enum import Enum

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource, SampleEntry, EventEntry, SubjectEntry
from SystemControl.utilities import find_files_by_type, idx_to_time

class MotorAction(Enum):
    REST = 0
    RIGHT = 1
    LEFT = 2

class LiveDataSource(DataSource):

    def __init__(self, subject: str, trial_type: str, log_level: str = 'CRITICAL'):
        super().__init__(log_level)

        self.name = 'recorded'
        self.sample_freq = 200
        self.trial_types = ['motor_imagery', 'baseline']
        self.event_names = list(MotorAction.__members__)
        self.subject_names = []
        self.trial_mappings = None
        self.ascended_being = subject
        self.subject_save_dir = os.path.join(DATA_DIR, self.name, self.ascended_being)
        self.selected_trial_type = trial_type
        self.current_trial = self.__next_trial_id()

        self.init_time = time.time()
        self._samples_lock = threading.Lock()
        self._event_lock = threading.Lock()

        initial_sample = {
            channel: 0
            for channel in self.coi
        }
        self.samples = [SampleEntry(idx=0, timestamp=0, data=initial_sample)]
        self.events = [EventEntry(idx=0, timestamp=0, event_type=MotorAction.REST.name)]
        subject_entry_fname = os.path.join(self.dataset_directory, self.ascended_being, f'{self.current_trial}.json')
        self.subject_entry = SubjectEntry(
            path=subject_entry_fname, source_name=self.name, subject=self.ascended_being, trial=self.current_trial,
            samples=self.samples, events=self.events
        )
        self.subject_entries.append(self.subject_entry)
        return

    def __iter__(self):
        for sample in self.samples:
            with self._event_lock:
                last_event = self.events[-1]
            yield sample, last_event
        return

    def __next_trial_id(self):
        prev_trials = sorted(find_files_by_type('json', self.subject_save_dir))
        if len(prev_trials) == 0:
            return f'{1:0>4}'

        last_trial = prev_trials[-1]
        last_trial_name, ext = os.path.splitext(last_trial)
        last_trial_id = int(last_trial_name.split(os.sep)[-1])
        return f'{last_trial_id + 1:0>4}'

    def add_sample(self, sample_type, timestamp, sample_data):
        with self._samples_lock:
            if sample_type == 'timeseries_filtered':
                data_points = {
                    self.coi[idx]: point
                    for idx, point in enumerate(sample_data)
                }
                sample_idx = len(self.samples)
                sample_time = idx_to_time(sample_idx, self.sample_freq)
                sample_entry = SampleEntry(
                    idx=sample_idx, timestamp=sample_time, data=data_points
                )
                self.samples.append(sample_entry)
        return

    def add_event(self, event_type, timestamp):
        with self._event_lock:
            with self._samples_lock:
                sample_idx = len(self.samples)
            sample_time = idx_to_time(sample_idx, self.sample_freq)
            event_entry = EventEntry(
                idx=sample_idx, timestamp=sample_time, event_type=event_type
            )
            self.events.append(event_entry)
        return

    def set_subject(self, subject: str):
        self.ascended_being = subject
        self.current_trial = self.__next_trial_id()
        return

    def set_trial_type(self, trial_type: str):
        self.selected_trial_type = trial_type
        self.current_trial = self.__next_trial_id()
        return

    def save_data(self, subject_entries: list = None, use_mp: bool = False, human_readable=True):
        super().save_data(subject_entries, human_readable=human_readable)
        return


def main():
    subject_name = 'Andrew'
    trial_type = 'motor_imagery'

    live_ds = LiveDataSource(subject=subject_name, trial_type=trial_type)
    live_ds.save_data(human_readable=True)
    return


if __name__ == '__main__':
    main()
