"""
@title
@description
"""
from time import sleep

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


def close():
    plt.close()
    return


class MatplotlibVisualizer:

    def __init__(self, data_source: DataSource, subject: str):
        self.data_source = data_source
        self.subject = subject

        self._data_iterator = self.data_source.__iter__()
        self.update_delay = 1. / self.data_source.sample_freq

        self.x_len = int(self.data_source.sample_freq)
        self.channel_names = self.data_source.coi
        self.event_names = self.data_source.event_names

        self.max_abs_val = 20E-5

        self.fig = plt.figure()

        self.x = np.linspace(0, 2 * np.pi, 120)
        self.y = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
        self.im = plt.imshow(self.f(self.x, self.y), animated=True)
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=50, blit=True)
        plt.show()
        self.idx = 0
        return

    def f(self, x, y):
        return np.sin(x) + np.cos(y)

    def update(self, *args):
        next_sample = next(self._data_iterator)

        self.x += np.pi / 15.
        self.y += np.pi / 20.
        self.im.set_array(self.f(self.x, self.y))
        return self.im,

    # def run(self):
    #     plt.show()
    #     return

    def close(self):
        plt.close()
        return


def main():
    physio_ds = PhysioDataSource()

    eeg_visualizer = MatplotlibVisualizer(physio_ds, physio_ds.ascended_being)
    # eeg_visualizer.run()
    # sleep(5)
    # eeg_visualizer.close()
    return


if __name__ == '__main__':
    main()
