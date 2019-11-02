"""
@title
@description
"""
import os

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource


class ProjectBciDataSource(DataSource):
    COI = ['C3', 'Cz', 'C4']
    NAME = 'ProjectBCI'

    def __init__(self):
        super().__init__()
        self.dataset_directory = os.path.join(DATA_DIR, self.__str__())
        self._data = self.__load_data()
        return

    def __str__(self):
        return self.NAME

    def __load_data(self):
        # todo
        datafname_1d = os.path.join(self.dataset_directory, f'Subject1_1D.mat')
        datafname_2d = os.path.join(self.dataset_directory, f'Subject1_2D.mat')
        moved_files = []
        return moved_files


def main():
    return


if __name__ == '__main__':
    main()
