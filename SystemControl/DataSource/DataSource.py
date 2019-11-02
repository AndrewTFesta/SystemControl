"""
@title
@description
"""
import mne
from mne.io.edf.edf import RawEDF


class EventAnnotation:
    # TODO  figure out data that should be stored - look at MNE

    def __init__(self, time_stamp: int, utc_offset: float, event_id: int):
        self.timestamp = time_stamp
        self.utc_offset = utc_offset
        self.id = event_id
        return


class DataSource:
    NAME = 'default'

    def __init__(self, mne_log_level: str = 'WARNING'):
        mne.set_log_level(mne_log_level)  # DEBUG, INFO, WARNING, ERROR, or CRITICAL
        return

    def __str__(self):
        return self.NAME

    def get_raw_trial(self, subject: int, run: int) -> RawEDF:
        raise NotImplemented

    def open_stream(self, subject: int = None):
        raise NotImplemented

    def add_sample(self):
        raise NotImplemented

    def get_sample_count(self):
        raise NotImplemented

    def add_annotation(self, event_annotation):
        raise NotImplemented

    def get_annotation_count(self):
        raise NotImplemented


def main():
    ds = DataSource()
    return


if __name__ == '__main__':
    main()
