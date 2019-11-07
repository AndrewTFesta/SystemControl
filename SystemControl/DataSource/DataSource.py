"""
@title
@description
"""
import os

from SystemControl import DATA_DIR


class SubjectEntry:

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, value):
        self.__path = value
        return

    @property
    def source_name(self):
        return self.__source_name

    @source_name.setter
    def source_name(self, value):
        self.__source_name = value
        return

    @property
    def subject(self):
        return self.__subject

    @subject.setter
    def subject(self, value):
        self.__subject = value
        return

    @property
    def trial(self):
        return self.__trial

    @trial.setter
    def trial(self, value):
        self.__trial = value
        return

    @property
    def samples(self):
        return self.__samples

    @samples.setter
    def samples(self, value):
        self.__samples = value
        return

    @property
    def events(self):
        return self.__events

    @events.setter
    def events(self, value):
        self.__events = value
        return

    ALL_SUBJECT_ENTRIES = []

    def __init__(self, path: str, source_name: str, subject: str, trial: str,
                 samples: list = None, events: list = None):
        self.__path = path
        self.__source_name = source_name
        self.__subject = subject
        self.__trial = trial
        self.__samples = samples
        self.__events = events

        self.ALL_SUBJECT_ENTRIES.append(self)
        return

    def __str__(self):
        return f'{self.source_name}:{self.subject}:{self.trial}'

    def __eq__(self, other):
        if not isinstance(other, SubjectEntry):
            return False

        sources_equal = self.source_name == other.source_name
        subjects_equal = self.subject == other.subject
        trials_equal = self.trial == other.trial
        return sources_equal and subjects_equal and trials_equal

    def is_loaded(self):
        return self.__samples and self.__events


class SampleEntry:

    @property
    def idx(self):
        return self.__idx

    @idx.setter
    def idx(self, value):
        self.__idx = value
        return

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, value):
        self.__time = value
        return

    @property
    def data(self):
        return self.__data

    @data.setter
    def data(self, value):
        self.__data = value
        return

    def __init__(self, idx: int, time: float, data: list):
        self.__idx = idx
        self.__time = time
        self.__data = data
        return

    def __str__(self):
        return f'{self.idx}:{self.time}:{self.data}'

    def __repr__(self):
        return self.__str__()


class EventEntry:

    @property
    def idx(self):
        return self.__idx

    @idx.setter
    def idx(self, value):
        self.__idx = value
        return

    @property
    def time(self):
        return self.__time

    @time.setter
    def time(self, value):
        self.__time = value
        return

    @property
    def type(self):
        return self.__type

    @type.setter
    def type(self, value):
        self.__type = value
        return

    def __init__(self, idx: int, time: float, event_type: str):
        self.__idx = idx
        self.__time = time
        self.__type = event_type
        return

    def __str__(self):
        return f'{self.idx}:{self.time}:{self.type}'

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
        self.coi = None
        self.subject_names = None
        self.trial_mappings = None
        self.trial_types = None
        self.event_names = None
        self.stream_open = None
        self.ascended_being = None
        self.selected_trial_type = None

        self.__trial_mappings = None
        return

    def __iter__(self):
        subject_entry_list = self.get_subject_entries()
        for subject_entry in subject_entry_list:
            sample_list = subject_entry.samples
            event_list = subject_entry.events

            next_event_idx = 1
            for sample in sample_list:
                if next_event_idx < len(event_list) and sample.time >= event_list[next_event_idx].time:
                    next_event_idx += 1
                yield sample, event_list[next_event_idx - 1]
        return

    def __str__(self):
        return f'{self.name}'

    def __repr__(self):
        return self.__str__()

    def event_name_from_id(self, event_idx) -> str:
        return self.event_names[event_idx]

    def preload_user(self):
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
                trial_samples = each_trial['samples']
                for each_sample in trial_samples:
                    yield each_sample, each_trial['trial']
        return

    def get_subject_entries(self) -> list:
        relevant_trials = self.get_trials_by_type(self.selected_trial_type)
        filtered_source_entries = filter(lambda entry: entry.source_name == self.name, SubjectEntry.ALL_SUBJECT_ENTRIES)
        filtered_subject_entries = filter(lambda entry: entry.subject == self.ascended_being, filtered_source_entries)
        subject_entry_list = filter(lambda entry: entry.trial in relevant_trials, filtered_subject_entries)
        return list(subject_entry_list)

    def get_trials_by_type(self, trial_type: str = None) -> list:
        if not trial_type:
            trial_type = self.ascended_being

        if trial_type not in self.trial_types:
            raise ValueError(f'Designated trial type is not a valid trial type: {trial_type}')

        return self.trial_mappings[trial_type]


def main():
    ds = DataSource()
    return


if __name__ == '__main__':
    main()
