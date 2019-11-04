"""
@title
@description
"""
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, style
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


def emitter(p=0.03):
    'return a random value with probability p, else 0'
    while True:
        v = np.random.rand(1)
        if v > p:
            yield 0.
        else:
            yield np.random.rand(1)


def FFT(x, y):
    X = (x[-1] - x[0]) / len(y)
    f = np.linspace(-2 * np.pi / X / 2, 2 * np.pi / X / 2, len(y))
    F = np.fft.fftshift(np.fft.fft(y)) / np.sqrt(len(y))
    return (f, F)


class MatplotlibVisualizer:

    def __init__(self, data_source: DataSource, subject: str):
        self.data_source = data_source
        self.subject = subject
        self.num_seconds = 2
        self.max_abs_val = 20E-5

        self._data_iterator = self.data_source.__iter__()
        self.update_delay = 1. / self.data_source.sample_freq
        self.num_samples = self.data_source.sample_freq * self.num_seconds
        self.channel_names = self.data_source.coi
        self.event_names = self.data_source.event_names

        self.fig = None
        self.ax = None
        self.data = None
        self.count = None
        self.writer = None

        self.init_data()
        return

    def init_data(self):
        if self.fig:
            plt.close()
        self.count = 0
        self.fig = plt.figure(figsize=(16, 8), facecolor='black')
        self.ax = self.fig.add_subplot(
            111, autoscale_on=False, frameon=False,
            # todo fix y scaling based on number of signals
            xlim=(0, self.num_samples), ylim=((self.max_abs_val * -1) * 1.5, self.max_abs_val * 1.5)
        )

        self.data = {}
        for each_ch in self.channel_names:
            ch_line = Line2D([], [])
            ch_entry = {'vals': np.array([]), 'line': ch_line}
            self.ax.add_line(ch_line)
            self.data[each_ch] = ch_entry
        return

    def update(self, sample):
        self.count += 1
        new_data = sample['data']

        artist_list = []
        for ch_idx, (ch_name, ch_entry) in enumerate(self.data.items()):
            ch_d_point = new_data[ch_idx] + (self.max_abs_val * (ch_idx - 1))
            ch_entry['vals'] = np.append(ch_entry['vals'], ch_d_point)
            if len(ch_entry['vals']) > self.num_samples:
                ch_entry['vals'] = ch_entry['vals'][1:]
            x_data = np.arange(0, len(ch_entry['vals']), 1)
            ch_entry['line'].set_data(x_data, ch_entry['vals'])
            artist_list.append(ch_entry['line'])

        return artist_list

    def show_animation(self, num_frames=100):
        self.init_data()
        animation.FuncAnimation(
            self.fig, self.update, self.data_source.get_data(), interval=self.update_delay, blit=True
        )
        plt.show()
        return

    def save_animation(self, num_frames=1000):
        # todo  limit number of frames
        #       add metadata
        #       set save_name to meaningful name
        #       fixme
        self.init_data()
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        ani = animation.FuncAnimation(
            self.fig, self.update, self.data_source.get_data(), interval=self.update_delay, blit=True
        )

        save_dir = os.path.join(DATA_DIR, 'output', 'animations')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_name = os.path.join(save_dir, f'test_file.mp4')
        ani.save(save_name, writer=writer)
        return


def main():
    """

    :return:
    """
    # Sometimes, the animation will fail to update when run from an IDE.
    # In PyCharm, a possible fix is to disable scientific plotting.
    # File -> Settings -> Tools -> Python Scientific
    #   Make sure the box "Show plots in tool window" is unchecked.
    # If the animation still fails to update, try again with the following
    # line uncommented.
    matplotlib.use('TkAgg')
    style.use('ggplot')

    physio_ds = PhysioDataSource()
    eeg_visualizer = MatplotlibVisualizer(physio_ds, physio_ds.ascended_being)
    # eeg_visualizer.save_animation()
    eeg_visualizer.show_animation()
    return


if __name__ == '__main__':
    main()
