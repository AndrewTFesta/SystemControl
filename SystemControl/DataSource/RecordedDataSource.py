"""
@title
@description
"""

import pandas as pd

from SystemControl.DataSource.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import MotorAction


class RecordedDataSource(DataSource):

    def __init__(self, subject: str = None, trial_type: str = None, log_level: str = 'CRITICAL'):
        DataSource.__init__(self, log_level)

        self.name = 'recorded'
        self.load_data()
        self.sample_freq = 200
        self.subject_names = self.__find_subject_names()
        self.__trial_mappings = {}
        self.__build_trial_mappings()
        self.trial_types = list(self.__trial_mappings.keys())
        self.event_names = list(MotorAction.__members__)

        self.ascended_being = subject if subject else self.subject_names[0]
        self.selected_trial_type = trial_type if trial_type else self.trial_types[1]
        self.set_subject(self.ascended_being)
        return

    def __build_trial_mappings(self):
        for row_idx, row_series in self.trial_info_df.iterrows():
            trial_type = row_series['trial_type']
            trial_name = row_series['trial_name']
            if trial_type not in self.__trial_mappings:
                self.__trial_mappings[trial_type] = []
            self.__trial_mappings[trial_type].append(trial_name)
        return

    def __find_subject_names(self):
        subject_names = pd.unique(self.trial_info_df['subject'])
        if len(subject_names) is 0:
            raise RuntimeError(f'Unable to locate any subjects in trial info file:\n\t{self.trial_info_file}')
        return subject_names


def main():
    display_generators = True
    subject = 'random_04'
    trial_type = 'motor_imagery_right_left'

    rec_ds = RecordedDataSource(subject=subject, trial_type=trial_type)
    print(rec_ds.subject_names)

    if display_generators:
        sample_list = list(rec_ds)
        print(f'Number of samples: {len(sample_list)}')

        for sample in rec_ds:
            print(f'{sample["idx"]}:{sample["timestamp"]}:'f'{sample["label"]}')

        for sample in rec_ds.downsample_generator(2):
            print(f'{sample["idx"]}:{sample["timestamp"]}:'f'{sample["label"]}')

        for window in rec_ds.window_generator(0.1, 0.1):
            window_head = window.iloc[0]
            window_tail = window.iloc[-1]
            print(f'{window_head["timestamp"]}:{window_tail["timestamp"]}')
    return


if __name__ == '__main__':
    main()
