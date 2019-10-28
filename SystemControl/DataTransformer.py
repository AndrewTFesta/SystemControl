"""
@title
@description
"""
from enum import Enum, auto


class Montage(Enum):
    # TODO determine at least 3 different types of montages
    DEFAULT = auto()


class Interpolate(Enum):
    LINEAR = auto()


# TODO add cleaning functionality
class DataTransformer:

    def __init__(self):
        return

    def generate_heatmap(self, data_matrix):
        return

    def crop_heatmap(self):
        return

    def generate_montage(self):
        return

    def __interpolate(self):
        return


def main():
    data_transformer = DataTransformer()
    return


if __name__ == '__main__':
    main()
