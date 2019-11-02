"""
@title
@description
"""

from SystemControl.DataSource import SqlDb
from SystemControl.DataSource.DataSource import DataSource


class Recorded:

    def __init__(self, time_stamp: int, board_time: int,
                 absolute_sample_id: int, board_sample_id: int,
                 is_valid: bool, data: list):
        self.timestamp = time_stamp
        self.board_time = board_time
        self.absolute_id = absolute_sample_id
        self.board_id = board_sample_id
        self.is_valid = is_valid
        self.data = data
        return


class RecordedDataSource(DataSource):

    RECORDED_TABLE_NAME = 'recorded_data'

    def __init__(self, database: SqlDb):
        # todo  data separate from impedance or make impedance property of channel dataset?
        #       add annotation tracking
        super().__init__(database)
        self.impedance: dict = {
            'channel_0': -1,
            'channel_1': -1,
            'channel_2': -1,
            'channel_3': -1
        }
        return

    def __str__(self):
        return 'Recorded'


def main():
    database = SqlDb.SqlDb()
    rec_ds = RecordedDataSource(database)
    return


if __name__ == '__main__':
    main()
