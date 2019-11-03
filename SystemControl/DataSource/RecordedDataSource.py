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
    def name(self) -> str:
        return 'Recorded'

    @property
    def sample_freq(self) -> float:
        return 200

    @property
    def subject_names(self) -> list:
        return ['Me']

    @property
    def coi(self) -> list:
        return ['C3', 'Cz', 'C4']

    @property
    def trial_names(self) -> list:
        return ['sunday']

    @property
    def event_names(self) -> list:
        return ['rest', 'right', 'left']

    def get_data(self) -> list:
        pass

    def get_events(self) -> list:
        pass

    def append_sample(self):
        pass

    def load_data(self):
        pass

    def save_data(self):
        pass


def main():
    rec_ds = RecordedDataSource()
    return


if __name__ == '__main__':
    main()
