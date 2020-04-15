"""
@title
@description
"""
import argparse
import os
import threading
import time
from queue import Queue, Empty
from time import sleep

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style, animation
from win32api import GetSystemMetrics

from SystemControl import IMAGES_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import MotorAction
from SystemControl.utils.ObserverObservable import Observer


class TrialRecorder(Observer):

    def __init__(self, data_source: DataSource, x_windows_scale: int = 2, y_windows_scale: int = 2):
        Observer.__init__(self, [data_source])
        self.data_source = data_source
        ##############################
        self._action_queue = Queue()
        self.data_source_name = data_source.__class__.__name__
        ##############################
        self.image_width = 244
        self.image_height = 244
        self.direction_images = {}
        for m_action in MotorAction:
            blank_image = np.zeros((self.image_height, self.image_width, 3), np.uint8)
            action_image = cv2.imread(os.path.join(IMAGES_DIR, f'{m_action.name}.png'))
            resized_action_image = cv2.resize(action_image, (self.image_width, self.image_height))
            added_image = cv2.addWeighted(blank_image, 0.2, resized_action_image, 1, 0)
            cvt_image = cv2.cvtColor(added_image, cv2.COLOR_BGR2RGB)
            self.direction_images[m_action] = cvt_image

        self.width_scale = x_windows_scale
        self.height_scale = y_windows_scale
        self._update_delay = 10

        matplotlib.use('TkAgg')
        style.use('ggplot')

        self._fig = plt.figure(facecolor='black')
        self.__position_fig()

        self._axes = self._fig.add_subplot(1, 1, 1)
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._img_artist = None

        self.ani = None

        self.start_time = -1
        self.end_time = -1
        self.run_time = -1
        self.sample_rate_counter = 0
        self.sample_rate_lock = threading.Lock()
        return

    def __position_fig(self):
        screen_width_resolution = GetSystemMetrics(0)
        screen_height_resolution = GetSystemMetrics(1)

        dpi = self._fig.dpi
        screen_width_inches = screen_width_resolution / dpi
        screen_height_inches = screen_height_resolution / dpi
        self._fig.set_size_inches(
            screen_width_inches / self.width_scale,
            screen_height_inches / self.height_scale
        )

        x_offset = (1 - (1 / self.width_scale))
        y_offset = (1 - (1 / self.height_scale))
        window_pos_x = int(screen_width_resolution * x_offset / 2)
        window_pos_y = int(screen_height_resolution * y_offset / 2)

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.wm_geometry(f'+{window_pos_x}+{window_pos_y}')
        fig_manager.window.attributes('-topmost', 0)
        return

    def __init_plot(self):
        img = self.direction_images[MotorAction.REST]
        self._img_artist = self._axes.imshow(img)
        return self._img_artist,

    def update(self, source, update_message):
        if source in self.subscriptions:
            if source.__class__.__name__ == self.data_source_name:
                update_type = update_message.get('type', None)

                if update_type == 'sample':
                    update_data = update_message.get('data', None)
                    with self.sample_rate_lock:
                        self.sample_rate_counter += 1
                elif update_type == 'event':
                    update_event = update_message.get('event', None)
                    curr_time = time.time()
                    d_time = curr_time - self.start_time
                    eta = self.run_time - d_time
                    with self.sample_rate_lock:
                        sample_rate = self.sample_rate_counter / d_time
                        print(f'event: {update_event}\n'
                              f'\tnum samples: {self.sample_rate_counter}\n'
                              f'\tsample rate: {sample_rate}\n'
                              f'\telapsed: {d_time:0.4f}\n'
                              f'\teta: {eta:0.4f}')
                        # self.sample_rate_counter = 0
                    self._action_queue.put(update_event)

    def run(self, run_time: float):
        t = threading.Timer(run_time, self.end_trial)
        t.daemon = True
        t.start()
        self.start_time = time.time()
        self.run_time = run_time
        self.data_source.set_recording(True)
        self.run_animation()
        return

    def run_animation(self):
        self.ani = animation.FuncAnimation(
            self._fig, self.update_artists, init_func=self.__init_plot, interval=self._update_delay, blit=True
        )
        plt.show()
        return

    def end_trial(self):
        self.end_time = time.time()
        self.data_source.set_recording(False)
        self.ani.event_source.stop()
        plt.close()  # FIXME closes due to raising exception
        return

    def update_artists(self, update_arg):
        try:
            next_action = self._action_queue.get_nowait()
            action_image = self.direction_images[next_action]
            self._img_artist.set_data(action_image)
        except Empty:
            pass
        return self._img_artist,


def main(margs):
    from SystemControl.OBciPython.UdpClient import UdpClient
    from SystemControl.StimulusGenerator import StimulusGenerator, GeneratorType
    from SystemControl.DataSource.LiveDataSource import LiveDataSource
    ################################################
    record_length = margs.get('record_length', 20)
    current_subject = margs.get('subject_name', 'random')
    trial_type = margs.get('session_type', 'motor_imagery_right_left')
    generate_delay = margs.get('stimulus_delay', 5)
    jitter_generator = margs.get('jitter', 0.2)
    ################################################
    verbosity = 0
    ################################################

    stimulus_generator = StimulusGenerator(
        delay=generate_delay, jitter=jitter_generator, generator_type=GeneratorType.RANDOM, verbosity=verbosity
    )
    udp_client = UdpClient()
    data_source = LiveDataSource(
        subject=current_subject, trial_type=trial_type,
        subscriber_list=[stimulus_generator, udp_client]
    )

    trial_recorder = TrialRecorder(
        data_source=data_source, x_windows_scale=2, y_windows_scale=2
    )
    stimulus_generator.run()
    udp_client.run()
    sleep(1)
    trial_recorder.run(record_length)

    trial_samples = data_source.get_trial_samples()
    # noinspection PyTypeChecker
    num_samples = len(trial_samples.index)
    print(f'Total number of samples: {num_samples}')
    print(f'Total sample rate: {num_samples / record_length}')
    data_source.save_data(start_time=0, end_time=-1)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a single recording session.')
    parser.add_argument('--record_length', type=int, default=120,
                        help='length (in seconds) of how long to record for the session')
    parser.add_argument('--subject_name', type=str, default='random',
                        help='Name of subject to use when saving this session')
    parser.add_argument('--session_type', type=str, default='motor_imagery_right_left',
                        help='type of trial to classify this session as')
    parser.add_argument('--stimulus_delay', type=int, default=5,
                        help='average time between generation of next stimulus')
    parser.add_argument('--jitter', type=float, default=0.2,
                        help='proportion of stimulus delay to add or subtract (randomly) from time between stimulus\n'
                             'if stimulus delay is 5, and jitter is 0.2, then the actual time between stimuli will '
                             'be in the range (5-0.2*5, 5+0.2*5)')

    args = parser.parse_args()
    main(vars(args))
