"""
@title
@description
"""
from abc import ABC, abstractmethod


class DataSource(ABC):

    def __init__(self, log_level: str = 'WARNING'):
        self._log_level = log_level
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
    def stream_interpolation(self):
        pass

    @abstractmethod
    def stream_heatmap(self):
        pass

    @abstractmethod
    def stream_trials(self):
        pass

    @abstractmethod
    def get_trials(self) -> list:
        pass

    @abstractmethod
    def event_name_from_idx(self, event_idx) -> str:
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

    @abstractmethod
    def set_subject(self, subject: str):
        pass


def main():
    ds = DataSource()
    return


if __name__ == '__main__':
    main()
