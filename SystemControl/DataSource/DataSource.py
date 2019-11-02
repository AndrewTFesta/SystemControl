"""
@title
@description
"""
from abc import ABC, abstractmethod

import mne


class EventAnnotation:
    def __init__(self, time_stamp: int, utc_offset: float, event_id: int):
        self.timestamp = time_stamp
        self.utc_offset = utc_offset
        self.id = event_id
        return


class DataSource(ABC):
    COI = []
    SUBJECT_NAMES = None
    NAME = ''

    def __init__(self, mne_log_level: str = 'WARNING'):
        mne.set_log_level(mne_log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL

        self.sfreq = None
        return

    def __str__(self):
        return self.NAME

    @abstractmethod
    def get_data(self, subject: str):
        raise NotImplementedError()

    @abstractmethod
    def load_data(self):
        raise NotImplementedError()


def main():
    ds = DataSource()
    return


if __name__ == '__main__':
    main()
