"""
@title
@description
"""
from SystemControl.DataSource import SqlDb
from SystemControl.DataSource.DataSource import DataSource


class LiveDataSource(DataSource):

    def __init__(self, database: SqlDb):
        super().__init__(database)
        return

    def __str__(self):
        return 'Live'


def main():
    return


if __name__ == '__main__':
    main()
