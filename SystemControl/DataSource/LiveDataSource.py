"""
@title
@description
"""
import json
import os
import threading
import time

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource
from SystemControl.utilities import find_files_by_type


class LiveDataSource(DataSource):

    def __init__(self, subject: str, trial_type: str, log_level: str = 'CRITICAL'):
        super().__init__(log_level)

        self.name = 'live'
        self.sample_freq = 200
        self.coi = ['C3', 'Cz', 'C4']
        self.trial_types = ['motor_imagery', 'baseline']
        self.event_names = ['rest', 'right', 'left']
        self.stream_open = False

        self.subject_names = []
        self.trial_mappings = None
        self.ascended_being = subject
        self.selected_trial_type = trial_type
        self.subject_save_dir = os.path.join(DATA_DIR, 'recorded', self.ascended_being)
        self.current_trial = self.__next_trial_id()

        self.samples = []
        self.events = []

        self.init_time = time.time()
        self._samples_lock = threading.Lock()
        self._event_lock = threading.Lock()
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
        last_trial_id = int(last_trial_name.split('_')[-1])
        return f'{last_trial_id + 1:0>4}'

    def get_data(self) -> list:
        return self.samples

    def get_events(self) -> list:
        return self.events

    def add_sample(self, sample_type, timestamp, sample_data):
        with self._samples_lock:
            self.samples.append({'time': timestamp - self.init_time, 'type': sample_type, 'data': sample_data})
        return

    def add_event(self, event_type, timestamp, event_data):
        with self._event_lock:
            self.events.append({'time': timestamp - self.init_time, 'type': event_type, 'data': event_data})
        return

    def save_data(self, indent: bool = False):
        if not os.path.isdir(self.subject_save_dir):
            os.makedirs(self.subject_save_dir)

        data_fname = os.path.join(self.subject_save_dir, f'{self.selected_trial_type}_{self.current_trial}.json')
        print(f'Saving data...\n\t{data_fname}')
        with self._samples_lock:
            with self._event_lock:
                data_dict = {'time': self.init_time, 'samples': self.samples, 'events': self.events}
                with open(data_fname, 'w+') as data_file:
                    if indent:
                        json.dump(data_dict, data_file, indent=2)
                    else:
                        json.dump(data_dict, data_file)

        num_samples = len(data_dict['samples'])
        num_events = len(data_dict['events'])
        print(f'Number of samples: {num_samples}\nNumber of events: {num_events}')
        return

    def set_subject(self, subject: str):
        self.ascended_being = subject
        self.subject_save_dir = os.path.join(DATA_DIR, 'recorded', self.ascended_being)
        self.current_trial = self.__next_trial_id()
        return

    def set_trial_type(self, trial_type: str):
        self.selected_trial_type = trial_type
        self.current_trial = self.__next_trial_id()
        return


def main():
    subject_name = 'Andrew'
    trial_type = 'motor_imagery'

    live_ds = LiveDataSource(subject=subject_name, trial_type=trial_type)
    live_ds.save_data()
    return


if __name__ == '__main__':
    main()
