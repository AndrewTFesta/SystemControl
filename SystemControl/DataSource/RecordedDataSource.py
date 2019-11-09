"""
@title
@description
"""
import os

from SystemControl.DataSource.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import MotorAction


class RecordedDataSource(DataSource):

    def __init__(self, subject: str = None, log_level: str = 'CRITICAL'):
        super().__init__(log_level)

        self.name = 'recorded'
        self.sample_freq = 200
        self.subject_names = self.__find_subject_names()
        self.trial_mappings = {
            'baseline': [],
            'motor_imagery': ["0001"]
        }
        self.trial_types = list(self.trial_mappings.keys())
        self.selected_trial_type = self.trial_types[1]
        self.event_names = list(MotorAction.__members__)

        self.ascended_being = subject if subject else self.subject_names[0]

        self.load_data()
        self.set_subject(self.ascended_being)
        return

    def trial_type_from_name(self, trial_name):
        pass  # todo

    def __find_subject_names(self):
        subject_names = os.listdir(self.dataset_directory)
        if not subject_names:
            raise RuntimeError(f'Unable to locate any subjects in directory: {self.dataset_directory}')
        # todo set trial_mappings
        return subject_names


def main():
    subject = 'Random'
    rec_ds = RecordedDataSource(subject=subject)
    print(rec_ds.subject_names)

    for sample, event in rec_ds:
        print(f'{sample["idx"]}:{sample["timestamp"]:<10.4f}::'
              f'{event["idx"]:>7d}:{event["timestamp"]}:{event["event_type"]}')
    return


if __name__ == '__main__':
    main()
