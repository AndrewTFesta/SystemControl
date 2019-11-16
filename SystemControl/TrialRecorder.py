"""
@title
@description
"""
import argparse
import os
import threading
from queue import Queue, Empty

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style, animation
from win32api import GetSystemMetrics

from SystemControl import IMAGES_DIR
from SystemControl.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import LiveDataSource, MotorAction
from SystemControl.OBciPython.UdpClient import UdpClient
from SystemControl.StimulusGenerator import StimulusGenerator, GeneratorType

matplotlib.use('TkAgg')
style.use('ggplot')


class TrialRecorder:

    def __init__(self, data_source: DataSource,  record_length: float = 5.0,
                 x_windows_scale: int = 2, y_windows_scale: int = 2):
        self.data_source = data_source
        # self.stimulus_generator = stimulus_generator
        # self.udp_client = udp_client
        self.record_length = record_length

        # self.stimulus_generator.add_callback(stimulus_generator.print_callback)
        # self.stimulus_generator.add_callback(stimulus_generator.log_event_callback)
        # self.stimulus_generator.add_callback(self.add_to_event_queue)
        ##############################
        self._action_queue = Queue()
        self._sample_iter = data_source.__iter__()
        next(self._sample_iter)
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

        self._fig = plt.figure(facecolor='black')
        self.__position_fig()

        self._axes = self._fig.add_subplot(1, 1, 1)
        self._axes.set_xticks([])
        self._axes.set_yticks([])
        self._img_artist = None

        self.ani = None
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

    def run(self):
        # self.stimulus_generator.run()
        # self.udp_client.run()

        t = threading.Timer(self.record_length, self.stop)
        t.daemon = True
        t.start()
        self.run_animation()
        return

    def run_animation(self):
        self.ani = animation.FuncAnimation(
            self._fig, self.update_stimuli, init_func=self.__init_plot, interval=self._update_delay, blit=True
        )

        plt.show()
        return

    def stop(self):
        # self.stimulus_generator.stop()
        # self.udp_client.stop()

        self.ani.event_source.stop()
        plt.close()
        return

    def update_stimuli(self, update_arg):
        try:
            next_action = self._action_queue.get_nowait()
            action_name = next_action['cb_arg']
            action_image = self.direction_images[action_name]
            self._img_artist.set_data(action_image)
        except Empty:
            pass
        return self._img_artist,

    def add_to_event_queue(self, callback_arg, timestamp, action_type):
        self._action_queue.put({'cb_arg': callback_arg, 'timestamp': timestamp, 'type': action_type})
        return


def main(margs):
    record_length = margs.get('record_length', 720)
    current_subject = margs.get('subject_name', 'random')
    trial_type = margs.get('session_type', 'motor_imagery')
    generate_delay = margs.get('stimulus_delay', 5)
    jitter_generator = margs.get('jitter', 0.2)
    human_readable = margs.get('human_readable', False)
    ################################################
    rand_seed = 42
    use_mp = False
    ################################################
    data_source = LiveDataSource(subject=current_subject, trial_type=trial_type)
    stimulus_generator = StimulusGenerator(
        data_source, delay=generate_delay, jitter=jitter_generator, seed=rand_seed, generator_type=GeneratorType.RANDOM
    )
    udp_client = UdpClient(data_source)

    # trial_recorder = TrialRecorder(data_source, stimulus_generator, udp_client, record_length)
    trial_recorder = TrialRecorder(data_source, record_length)
    trial_recorder.run()
    data_source.save_data(human_readable=human_readable, use_mp=use_mp)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform a single recording session.')
    parser.add_argument('--record_length', type=int, default=720,
                        help='length (in seconds) of how long to record for the session')
    parser.add_argument('--subject_name', type=str, default='random',
                        help='Name of subject to use when saving this session')
    parser.add_argument('--session_type', type=str, default='motor_imagery',
                        help='type of trial to classify this session as')
    parser.add_argument('--stimulus_delay', type=int, default=5,
                        help='average time between generation of next stimulus')
    parser.add_argument('--jitter', type=float, default=0.2,
                        help='proportion of stimulus delay to add or subtract (randomly) from time between stimulus\n'
                             'if stimulus delay is 5, and jitter is 0.2, then the actual time between stimuli will '
                             'be in the range (5-0.2*5, 5+0.2*5)')
    parser.add_argument('--human_readable', action='store_true',
                        help='whether to save the trial session in a human-readable format')

    args = parser.parse_args()
    main(vars(args))
