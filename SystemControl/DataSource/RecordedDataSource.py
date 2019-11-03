"""
@title
@description
"""

from SystemControl.DataSource.DataSource import DataSource


class RecordedDataSource(DataSource):

    def __init__(self, ):
        super().__init__()
        self.impedance: dict = {
            'channel_0': -1,
            'channel_1': -1,
            'channel_2': -1,
            'channel_3': -1
        }
        return

    def __iter__(self):
        pass

    @property
    def name(self):
        return 'Recorded'

    @property
    def sample_freq(self):
        pass

    @property
    def subject_names(self):
        pass

    @property
    def coi(self):
        pass

    @property
    def trial_names(self):
        pass

    @property
    def event_names(self):
        pass

    def get_events(self, subject: str):
        pass

    def append_sample(self, subject: str):
        pass

    def get_data(self, subject: str):
        pass

    def load_data(self):
        pass


def main():
    rec_ds = RecordedDataSource()
    return


if __name__ == '__main__':
    main()
