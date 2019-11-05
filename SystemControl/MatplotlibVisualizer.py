"""
@title
@description
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation, style
from matplotlib.lines import Line2D
from win32api import GetSystemMetrics

from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


class MatplotlibVisualizer:

    def __init__(self, data_source: DataSource, subject: str):
        self.full_screen = True
        self.width_scale = 2
        self.height_scale = 2
        self.num_seconds = 2

        self.dpi = 96
        self.screen_width_inches = 16
        self.screen_height_inches = 12
        self.screen_width_resolution = GetSystemMetrics(0)
        self.screen_height_resolution = GetSystemMetrics(1)
        self.signal_sep_val = 20E-5

        self.data_source = data_source
        self.subject = subject
        self.channel_names = self.data_source.coi
        self.event_names = self.data_source.event_names

        self._event_colors = ['b', 'r', 'g']
        self._data_iterator = self.data_source.__iter__()
        self._update_delay = 1. / self.data_source.sample_freq
        self._num_samples = self.data_source.sample_freq * self.num_seconds

        self._fig = None
        self._fig_callbacks = None
        self._sig_axes = None
        self._signal_data = None
        self._classifier_axes = None
        self._iter_idx = None
        self._ani_writer = None
        return

    def __init_plot(self):
        if self._fig:
            plt.close()

        self._fig = plt.figure(figsize=(self.screen_width_inches, self.screen_height_inches), facecolor='black')
        self.dpi = self._fig.dpi
        self.screen_width_inches = self.screen_width_resolution / self.dpi
        self.screen_height_inches = self.screen_height_resolution / self.dpi
        self._fig.set_size_inches(
            self.screen_width_inches / self.width_scale,
            self.screen_height_inches / self.height_scale
        )

        self._iter_idx = 0
        self.__setup_signal_axes()

        fig_manager = plt.get_current_fig_manager()
        if self.full_screen:
            fig_manager.window.state('zoomed')
        fig_manager.window.wm_geometry("+0+0")
        return

    def __setup_signal_axes(self):
        self._sig_axes = self._fig.add_subplot(
            1, 1, 1, autoscale_on=False, frameon=False,
            # todo fix y scaling based on number of signals
            xlim=(0, self._num_samples), ylim=((self.signal_sep_val * -1) * 1.5, self.signal_sep_val * 1.5)
        )
        self._signal_data = {}
        for each_ch in self.channel_names:
            ch_line = Line2D([], [])
            ch_entry = {'x_vals': np.array([]), 'y_vals': np.array([]), 'line': ch_line}
            self._sig_axes.add_line(ch_line)
            self._signal_data[each_ch] = ch_entry
        return

    def __update_axes(self, sample):
        self._iter_idx += 1

        sample_idx = sample[0]['idx']
        sample_time = sample[0]['time']
        sample_event = sample[0]['event']  # todo add to event axes
        sample_data = sample[0]['data']
        trial_name = sample[1]

        artist_list = []
        color_idx = sample_event[2] - 1
        vspan = self._sig_axes.axvspan(sample_idx, sample_idx + 1, facecolor=self._event_colors[color_idx], alpha=0.1)
        artist_list.append(vspan)
        if sample_idx >= self._num_samples:
            x_min, x_max = self._sig_axes.get_xlim()
            self._sig_axes.set_xlim(x_min + 1, x_max + 1)
            self._sig_axes.figure.canvas.draw()

        for ch_idx, (ch_name, ch_entry) in enumerate(self._signal_data.items()):
            ch_d_point = sample_data[ch_idx] + (self.signal_sep_val * (ch_idx - 1))

            ch_entry['x_vals'] = np.append(ch_entry['x_vals'], sample_idx)
            ch_entry['y_vals'] = np.append(ch_entry['y_vals'], ch_d_point)

            ch_entry['line'].set_data(ch_entry['x_vals'], ch_entry['y_vals'])
            artist_list.append(ch_entry['line'])
        return artist_list

    def show_animation(self, num_frames: int = 100) -> None:
        """

        :type num_frames:   int
        :param num_frames:  number of frames to show before closing the animation
        :return:            None
        """
        self.__init_plot()
        animation.FuncAnimation(
            self._fig, self.__update_axes, self.data_source,
            interval=self._update_delay, blit=True
        )
        plt.show()
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
    eeg_visualizer.show_animation()
    return


if __name__ == '__main__':
    main()
