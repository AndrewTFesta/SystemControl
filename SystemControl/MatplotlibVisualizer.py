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
from win32api import GetSystemMetrics

from SystemControl import DATA_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.PhysioDataSource import PhysioDataSource


def FFT(x, y):
    X = (x[-1] - x[0]) / len(y)
    f = np.linspace(-2 * np.pi / X / 2, 2 * np.pi / X / 2, len(y))
    F = np.fft.fftshift(np.fft.fft(y)) / np.sqrt(len(y))
    return f, F


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

        self._data_iterator = self.data_source.__iter__()
        self._update_delay = 1. / self.data_source.sample_freq
        self._num_samples = self.data_source.sample_freq * self.num_seconds

        self._fig = None
        self._fig_callbacks = None
        self._sig_axes = None
        self._event_axes = None
        self._data = None
        self._iter_idx = None
        self._ani_writer = None
        return

    def __init_plot(self):
        if self._fig:
            plt.close()

        self._fig = plt.figure(
            figsize=(self.screen_width_inches, self.screen_height_inches),
            facecolor='black'
        )
        '''
        'button_press_event'    MouseEvent - mouse button is pressed
        'button_release_event'  MouseEvent - mouse button is released
        'draw_event'            DrawEvent - canvas draw (but before screen update)
        'key_press_event'	    KeyEvent - key is pressed
        'key_release_event'	    KeyEvent - key is released
        'motion_notify_event'	MouseEvent - mouse motion
        'pick_event'	        PickEvent - an object in the canvas is selected
        'resize_event'	        ResizeEvent - figure canvas is resized
        'scroll_event'	        MouseEvent - mouse scroll wheel is rolled
        'figure_enter_event'	LocationEvent - mouse enters a new figure
        'figure_leave_event'	LocationEvent - mouse leaves a figure
        'axes_enter_event'	    LocationEvent - mouse enters a new axes
        'axes_leave_event'	    LocationEvent - mouse leaves an axes
        '''
        self._fig_callbacks = {
            'close_event': self._fig.canvas.mpl_connect('close_event', self.__animation_closed),
            'resize_event': self._fig.canvas.mpl_connect('resize_event', self.__animation_resized)
        }

        self.dpi = self._fig.dpi
        self.screen_width_inches = self.screen_width_resolution / self.dpi
        self.screen_height_inches = self.screen_height_resolution / self.dpi
        self._fig.set_size_inches(
            self.screen_width_inches / self.width_scale,
            self.screen_height_inches / self.height_scale
        )

        self._iter_idx = 0
        self.__setup_signal_axes()
        self.__setup_event_axes()

        fig_manager = plt.get_current_fig_manager()
        if self.full_screen:
            fig_manager.window.state('zoomed')
        fig_manager.window.wm_geometry("+0+0")
        return

    def __setup_signal_axes(self):
        self._sig_axes = self._fig.add_subplot(
            2, 1, 1, autoscale_on=False, frameon=False,
            # todo fix y scaling based on number of signals
            xlim=(0, self._num_samples), ylim=((self.signal_sep_val * -1) * 1.5, self.signal_sep_val * 1.5)
        )
        self._data = {}
        for each_ch in self.channel_names:
            ch_line = Line2D([], [])
            ch_entry = {'vals': np.array([]), 'line': ch_line}
            self._sig_axes.add_line(ch_line)
            self._data[each_ch] = ch_entry
        return

    def __setup_event_axes(self):
        self._event_axes = self._fig.add_subplot(
            4, 1, 3, autoscale_on=False, frameon=False,
            xlim=(0, self._num_samples), ylim=((self.signal_sep_val * -1) * 1.5, self.signal_sep_val * 1.5)
        )
        self._event_axes = self._fig.add_subplot(
            4, 1, 4, autoscale_on=False, frameon=False,
            xlim=(0, self._num_samples), ylim=((self.signal_sep_val * -1) * 1.5, self.signal_sep_val * 1.5)
        )
        return

    def __update_axes(self, datapoint):
        self._iter_idx += 1
        sample = datapoint['sample']
        event = datapoint['event']  # todo add to event axes

        new_data = sample['data']

        artist_list = []
        for ch_idx, (ch_name, ch_entry) in enumerate(self._data.items()):
            ch_d_point = new_data[ch_idx] + (self.signal_sep_val * (ch_idx - 1))
            ch_entry['vals'] = np.append(ch_entry['vals'], ch_d_point)
            if len(ch_entry['vals']) > self._num_samples:
                ch_entry['vals'] = ch_entry['vals'][1:]
            x_data = np.arange(0, len(ch_entry['vals']), 1)
            ch_entry['line'].set_data(x_data, ch_entry['vals'])
            artist_list.append(ch_entry['line'])
        return artist_list

    def show_animation(self, num_frames=100):
        self.__init_plot()
        animation.FuncAnimation(
            self._fig, self.__update_axes, self.data_source,
            interval=self._update_delay, blit=True
        )
        plt.show()
        return

    def __animation_resized(self, event_trigger):
        new_width = event_trigger.width
        new_height = event_trigger.height

        if new_width < self.screen_width_resolution and new_height < self.screen_width_resolution:
            x_offset = int(self.screen_width_resolution / 4)
            y_offset = int(self.screen_height_resolution / 4)

            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.wm_geometry(f'+{x_offset}+{y_offset}')
        return

    def __animation_closed(self, close_trigger):
        for callback_name, callback_id in self._fig_callbacks.items():
            self._fig.canvas.mpl_disconnect(callback_id)
        plt.close()
        return

    def save_animation(self, num_frames=1000):
        # todo  limit number of frames
        #       add metadata
        #       set save_name to meaningful name
        #       fixme
        self.__init_plot()
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        ani = animation.FuncAnimation(
            self._fig, self.__update_axes, self.data_source,
            interval=self._update_delay, blit=True
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
    eeg_visualizer.show_animation()
    return


if __name__ == '__main__':
    main()
