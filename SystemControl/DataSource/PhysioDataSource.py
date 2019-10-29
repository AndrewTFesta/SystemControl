"""
@title
@description

    |---------------------------------------------------|
    | run         | task                                |
    |---------------------------------------------------|
    | 1           | Baseline, eyes open                 |
    | 2           | Baseline, eyes closed               |
    | 3, 7, 11    | Motor execution: left vs right hand |
    | 4, 8, 12    | Motor imagery: left vs right hand   |
    | 5, 9, 13    | Motor execution: hands vs feet      |
    | 6, 10, 14   | Motor imagery: hands vs feet        |
    |---------------------------------------------------|

    T0  rest
    T1  onset of motion (real or imagined)
            the left fist (in runs 3, 4, 7, 8, 11, and 12)
            both fists (in runs 5, 6, 9, 10, 13, and 14)
    T2  onset of motion (real or imagined)
            the right fist (in runs 3, 4, 7, 8, 11, and 12)
            both feet (in runs 5, 6, 9, 10, 13, and 14)
"""
from enum import Enum

import mne
from mne.io import read_raw_edf
from mne.io.edf.edf import RawEDF as RawEDF

from SystemControl import DATABASE_URL
from SystemControl.DataSource import SqlDb
from SystemControl.DataSource.DataSource import DataSource


class PhysioEntry:

    def __init__(self, fname, subject, run, data):
        self.fname = fname
        self.subject = subject
        self.run = run
        self.data = data
        return


def int_to_subject_str(subject_int):
    return f'S{subject_int:03d}'


def int_to_run_str(run_int):
    return f'R{run_int:02d}'


class PhysioEvent(Enum):
    T0 = 0
    T1 = 1
    T2 = 2


class PhysioDataSource(DataSource):

    SUBJECT_NUMS = list(range(1, 109))
    RUN_NUMS = list(range(1, 14))
    PHYSIO_TABLE_NAME = 'physio_data'
    COI = ['C3', 'Cz', 'C4']

    def __init__(self, database: SqlDb):
        super().__init__(database)
        return

    def __str__(self):
        return 'Physio'

    @staticmethod
    def clean_raw_edf(raw_edf):
        # todo  improve
        cleaned_raw_edf = raw_edf.copy()

        # strip channel names of "." characters
        cleaned_raw_edf.rename_channels(lambda x: x.strip('.'))
        cleaned_raw_edf.pick_channels(PhysioDataSource.COI)

        # Apply band-pass filter
        cleaned_raw_edf = cleaned_raw_edf.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')
        return cleaned_raw_edf

    def get_raw_trial(self, subject: int, run: int) -> RawEDF:
        if not isinstance(subject, int):
            raise TypeError(f'Invalid type of subject parameter: {type(subject)}')
        if not isinstance(run, int):
            raise TypeError(f'Invalid type of run parameter: {type(run)}')

        if subject not in self.SUBJECT_NUMS:
            raise ValueError(f'Invalid subject specified: \'{subject}\'')
        if run not in self.RUN_NUMS:
            raise ValueError(f'Invalid run specified: \'{run}\'')

        subject_str = int_to_subject_str(subject)
        run_str = int_to_run_str(run)

        filter_dict = {'subject': subject_str, 'run': run_str}
        table_rows = self.database.get_table_rows(self.PHYSIO_TABLE_NAME, filter_dict)

        edf_fnames = [each_entry[1] for each_entry in table_rows]
        raw_edf = mne.concatenate_raws([read_raw_edf(fname, preload=True) for fname in edf_fnames])
        raw_edf = self.clean_raw_edf(raw_edf)
        return raw_edf

    def get_mi_df(self, subject: int = None, trial: int = None):
        raw_edf = self.get_mi_right_left(subject)
        raw_df = raw_edf.to_data_frame()
        return raw_df

    def get_baseline(self, subject, eyes_open: bool = True) -> RawEDF:
        if eyes_open:
            baseline_edf = self.get_raw_trial(subject, 1)
        else:
            baseline_edf = self.get_raw_trial(subject, 2)
        return baseline_edf

    def get_mi_right_left(self, subject) -> RawEDF:
        runs = [4, 8, 12]
        raw_edf = mne.concatenate_raws([self.get_raw_trial(subject, run_num) for run_num in runs])
        return raw_edf

    def get_mi_hands_feet(self, subject) -> RawEDF:
        runs = [6, 10, 14]
        raw_edf = mne.concatenate_raws([self.get_raw_trial(subject, run_num) for run_num in runs])
        return raw_edf

    @staticmethod
    def get_data_frame(raw_edf: RawEDF) -> list:
        raw_df = raw_edf.to_data_frame()
        return raw_df

    @staticmethod
    def get_data(raw_edf: RawEDF) -> list:
        raw_data = raw_edf.get_data()
        # raw_data = np.transpose(raw_data)

        data_list = []
        for each_row in raw_data:
            data_list.append(each_row)
        return data_list

    @staticmethod
    def get_annotations(raw_edf: RawEDF):
        annotations = raw_edf.annotations
        return annotations

    @staticmethod
    def get_events(raw_edf: RawEDF):
        events = mne.events_from_annotations(raw_edf)
        return events

    @staticmethod
    def get_channel_names(raw_edf: RawEDF):
        ch_names = raw_edf.ch_names
        return ch_names

    @staticmethod
    def idx_to_time(raw_edf: RawEDF, idx) -> float:
        sfreq = raw_edf.info['sfreq']
        time_val = idx / sfreq
        return time_val


def main():
    """

    :return:
    """
    subject = 1

    db_path = DATABASE_URL
    database = SqlDb.SqlDb(db_path)

    physio_ds = PhysioDataSource(database)
    raw_data = physio_ds.get_mi_right_left(subject=subject)
    print(f'Raw data is type: {type(raw_data)}')

    annots = PhysioDataSource.get_annotations(raw_data)
    events = physio_ds.get_events(raw_data)

    test_time = physio_ds.idx_to_time(raw_data, 1)
    print(test_time)

    test_time = physio_ds.idx_to_time(raw_data, 160)
    print(test_time)

    test_time = physio_ds.idx_to_time(raw_data, 320)
    print(test_time)
    # todo refactor to use h5py
    return


if __name__ == '__main__':
    main()
