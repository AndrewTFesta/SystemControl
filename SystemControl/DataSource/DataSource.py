"""
@title
@description
"""
from abc import ABC, abstractmethod

import mne


class DataSource(ABC):

    def __init__(self, mne_log_level: str = 'WARNING'):
        mne.set_log_level(mne_log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        return

    @abstractmethod
    def __iter__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def sample_freq(self) -> float:
        pass

    @property
    @abstractmethod
    def subject_names(self) -> list:
        pass

    @property
    @abstractmethod
    def coi(self) -> list:
        pass

    @property
    @abstractmethod
    def trial_names(self) -> list:
        pass

    @property
    @abstractmethod
    def event_names(self) -> list:
        pass

    @abstractmethod
    def get_data(self) -> list:
        pass

    @abstractmethod
    def get_events(self) -> list:
        pass

    @abstractmethod
    def append_sample(self):
        pass

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def save_data(self):
        pass


def main():
    ds = DataSource()
    return


if __name__ == '__main__':
    main()
