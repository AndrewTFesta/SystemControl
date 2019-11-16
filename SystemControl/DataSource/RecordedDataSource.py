"""
@title
@description
"""
import os

from SystemControl.DataSource.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import MotorAction


class RecordedDataSource(DataSource):

    def __init__(self, subject: str = None, trial_type: str = None, log_level: str = 'CRITICAL'):
        super().__init__(log_level)

        self.name = 'recorded'
        self.sample_freq = 200
        self.subject_names = self.__find_subject_names()
        self.__trial_mappings = {
            'baseline_open': [],
            'baseline_closed': [],
            'motor_imagery': []
        }
        self.trial_types = list(self.__trial_mappings.keys())
        self.event_names = list(MotorAction.__members__)

        self.ascended_being = subject if subject else self.subject_names[0]
        self.selected_trial_type = trial_type if trial_type else self.trial_types[1]

        self.__trial_nums_from_type()
        self.load_data()
        self.set_subject(self.ascended_being)
        return

    def __trial_nums_from_type(self):
        subject_trials = os.listdir(os.path.join(self.dataset_directory, self.ascended_being))
        for trial in subject_trials:
            trial, ext = os.path.splitext(trial)
            trial_type, trial_num = trial.split('-')
            if trial_type in self.__trial_mappings:
                self.__trial_mappings[trial_type].append(trial_num)
        return

    def __find_subject_names(self):
        subject_names = os.listdir(self.dataset_directory)
        if not subject_names:
            raise RuntimeError(f'Unable to locate any subjects in directory: {self.dataset_directory}')
        # todo set trial_mappings
        return subject_names


def main():
    subject = 'random'
    trial_type = 'motor_imagery_right_left'

    rec_ds = RecordedDataSource(subject=subject, trial_type=trial_type)
    print(rec_ds.subject_names)

    for sample, event in rec_ds:
        print(f'{sample["idx"]}:{sample["timestamp"]:<10.4f}::'
              f'{event["idx"]:>7d}:{event["timestamp"]}:{event["event_type"]}')
    return


if __name__ == '__main__':
    main()
