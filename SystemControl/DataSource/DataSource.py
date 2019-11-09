"""
@title
@description
"""
import json
import os
import sys
import time
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import cpu_count

from tqdm import tqdm

from SystemControl import DATA_DIR
from SystemControl.utilities import find_files_by_type


class SubjectEntry(dict):

    def __init__(self, path: str, source_name: str, subject: str, trial: str,
                 samples: list = None, events: list = None):
        super().__init__(
            path=path, source_name=source_name, subject=subject, trial=trial, samples=samples, events=events
        )
        return

    def __str__(self):
        return f'{self["source_name"]}:{self["subject"]}:{self["trial"]}'

    def __eq__(self, other):
        if not isinstance(other, SubjectEntry):
            return False

        sources_equal = self["source_name"] == other["source_name"]
        subjects_equal = self["subject"] == other["subject"]
        trials_equal = self["trial"] == other["trial"]
        return sources_equal and subjects_equal and trials_equal

    def is_loaded(self):
        return self["samples"] and self["events"]


class SampleEntry(dict):

    def __init__(self, idx: int, timestamp: float, data: dict):
        super().__init__(idx=idx, timestamp=timestamp, data=data)
        return

    def __str__(self):
        return f'{self["idx"]}:{self["timestamp"]}:{self["data"]}'

    def __repr__(self):
        return self.__str__()


class EventEntry(dict):

    def __init__(self, idx: int, timestamp: float, event_type: str):
        super().__init__(idx=idx, timestamp=timestamp, event_type=event_type)
        return

    def __str__(self):
        return f'{self["idx"]}:{self["timestamp"]}:{self["event_type"]}'

    def __repr__(self):
        return self.__str__()


class DataSource:

    @property
    def dataset_directory(self):
        return os.path.join(DATA_DIR, self.name)

    def __init__(self, log_level: str = 'WARNING'):
        self._log_level = log_level

        self.name = None
        self.sample_freq = None
        self.subject_names = None
        self.trial_mappings = None
        self.trial_types = None
        self.event_names = None
        self.ascended_being = None
        self.subject_entries = None
        self.selected_trial_type = None

        self.stream_open = False
        self.subject_entries = []
        self.coi = ['C3', 'Cz', 'C4']
        return

    def __iter__(self):
        subject_entry_list = self.get_subject_entries()
        for subject_entry in subject_entry_list:
            sample_list = subject_entry["samples"]
            event_list = subject_entry["events"]

            next_event_idx = 1
            if event_list:
                curr_event = event_list[0]
            else:
                curr_event = EventEntry(idx=0, timestamp=0, event_type='None')

            for sample in sample_list:
                if next_event_idx < len(event_list) and sample["timestamp"] >= event_list[next_event_idx]["timestamp"]:
                    next_event_idx += 1
                    curr_event = event_list[next_event_idx - 1]
                yield sample, curr_event
        return

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return self.__str__()

    def load_data(self):
        time_start = time.time()
        json_file_list = find_files_by_type(file_type='json', root_dir=self.dataset_directory)
        for each_fname in tqdm(json_file_list, desc=f'Storing json file names', file=sys.stdout):
            file_parts = each_fname.split(os.sep)
            entry_trial, _ = os.path.splitext(file_parts[-1])
            entry_subject = file_parts[-2]
            subject_entry = SubjectEntry(
                path=each_fname, source_name=self.name, subject=entry_subject, trial=entry_trial
            )
            self.subject_entries.append(subject_entry)
        time_end = time.time()
        print(f'Time to store file names: {time_end - time_start:.4f} seconds')
        return

    def set_subject(self, subject: str):
        if subject not in self.subject_names:
            raise ValueError(f'Designated subject is not a valid subject: {subject}')

        self.ascended_being = subject
        self.preload_user()
        return

    def set_trial_type(self, trial_type: str):
        if trial_type not in self.trial_types:
            raise ValueError(f'Designated trial is not a valid trial type: {trial_type}')

        self.selected_trial_type = trial_type
        self.preload_user()
        return

    def stream_subject_entries(self):
        trials = self.get_subject_entries()
        self.stream_open = True
        while self.stream_open:
            for each_trial in trials:
                trial_samples = each_trial["samples"]
                for each_sample in trial_samples:
                    yield each_sample, each_trial["trial"]
        return

    def get_subject_entries(self) -> list:
        relevant_trials = self.get_trials_by_type(self.selected_trial_type)
        filtered_subject_entries = filter(lambda entry: entry["subject"] == self.ascended_being, self.subject_entries)
        filtered_trial_entries = filter(lambda entry: entry["trial"] in relevant_trials, list(filtered_subject_entries))
        return list(filtered_trial_entries)

    def get_trials_by_type(self, trial_type: str = None) -> list:
        if not trial_type:
            trial_type = self.ascended_being

        if trial_type not in self.trial_types:
            raise ValueError(f'Designated trial type is not a valid trial type: {trial_type}')

        return self.trial_mappings[trial_type]

    def preload_user(self, subject: str = None):
        if not subject:
            subject = self.ascended_being

        if subject not in self.subject_names:
            raise ValueError(f'Designated subject is not a valid subject: {subject}')

        subject_entry_list = self.get_subject_entries()
        for subject_entry in subject_entry_list:
            if not subject_entry.is_loaded():
                with open(subject_entry["path"], 'r+') as subject_file:
                    entry_data = json.load(subject_file)
                    sample_list = [SampleEntry(**each_sample) for each_sample in entry_data["samples"]]
                    event_list = [EventEntry(**each_event) for each_event in entry_data["events"]]

                    subject_entry["samples"] = sample_list
                    subject_entry["events"] = event_list
        return

    def save_data(self, subject_entries: list = None, human_readable=True):
        if not subject_entries:
            subject_entries = self.subject_entries

        num_cpus = cpu_count()
        use_mp = len(subject_entries) > num_cpus * 2
        save_pbar = tqdm(total=len(subject_entries), desc=f'Saving SubjectEntry files', file=sys.stdout)
        if use_mp:
            with ProcessPoolExecutor(max_workers=num_cpus) as mp_exec:
                for subject_entry in subject_entries:
                    save_entry_future = mp_exec.submit(self.save_entry, subject_entry, human_readable)
                    save_entry_future.arg = save_pbar
                    save_entry_future.add_done_callback(self.__future_pbar_update)
        else:
            for subject_entry in subject_entries:
                self.save_entry(subject_entry, human_readable)
                save_pbar.update(1)
        save_pbar.close()
        return

    @staticmethod
    def __future_pbar_update(future):
        pbar = future.arg
        pbar.update(1)
        return

    def save_entry(self, subject_entry, human_readable=False):
        entry_trial = subject_entry["trial"]
        subject_name = subject_entry["subject"]

        subject_save_dir = os.path.join(DATA_DIR, self.name, subject_name)
        if not os.path.isdir(subject_save_dir):
            os.makedirs(subject_save_dir)

        subject_entry_fname = os.path.join(subject_save_dir, f'{entry_trial}.json')
        with open(subject_entry_fname, 'w+') as subject_entry_file:
            if human_readable:
                json.dump(subject_entry, subject_entry_file, indent=2)
            else:
                json.dump(subject_entry, subject_entry_file)
        return


def main():
    return


if __name__ == '__main__':
    main()
