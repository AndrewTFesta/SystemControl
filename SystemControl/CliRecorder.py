"""
@title
    CliRecorder.py
@description
"""
import json
import os
import random
import sys
import threading
import time
from enum import Enum
from msvcrt import getch

from SystemControl import DATA_DIR


class KeyAction(Enum):
    UP = 72
    LEFT = 75
    RIGHT = 77
    DOWN = 80


class CliRecorder:
    DELAY_TIMER = 1

    def __init__(self, subject_name='anon', run_duration=-1):
        time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())
        self.start_time = time.time()

        self.running = False
        self.duration = run_duration

        self.rec_thread = None
        self.rand_act_thread = None

        self.event_dict = {'start_absolute': self.start_time}
        event_dir = os.path.join(DATA_DIR, 'recordedEvents')
        if not os.path.isdir(event_dir):
            os.makedirs(event_dir)
        self.event_fname = os.path.join(event_dir, '{}_{}.txt'.format(subject_name, time_stamp))
        return

    def run(self):
        self.rec_thread = threading.Thread(target=self.start_recorder, args=(), daemon=True)
        self.rand_act_thread = threading.Thread(target=self.start_rand_act, args=(), daemon=True)

        self.running = True

        print('Starting key logger and random action display threads...')
        self.rec_thread.start()
        self.rand_act_thread.start()

        if self.duration > 0:
            time.sleep(self.duration)
        else:
            self.rand_act_thread.join()
        self.event_dict['stop_absolute'] = time.time()
        self.event_dict['duration'] = self.event_dict['stop_absolute'] - self.event_dict['start_absolute']
        self.running = False
        return

    def print_random_action(self):
        rand_action = random.choice([each_action for each_action in KeyAction])
        log_msg = 'print {}'.format(rand_action.name)
        print('----- {} -----'.format(log_msg))
        act_time = time.time()
        # noinspection PyTypeChecker
        self.event_dict[act_time - self.start_time] = log_msg
        return

    def start_recorder(self):
        """
        The _getch and_getwch functions read a single character from the console without echoing the character.
        None of these functions can be used to read CTRL+C.

        When reading a function key or an arrow key, each function must be called twice; the first call returns 0
        or 0xE0, and the second call returns the actual key code.

        So if you read an 0x00 or 0xE0, read it a second time to get the key code for an arrow or function key.
        From experimentation:
            0x00 precedes F1-F10 (0x3B-0x44)
            0xE0 precedes arrow keys and Ins/Del/Home/End/PageUp/PageDown.
        :return:
        """
        while self.running:
            key_pressed = ord(getch())
            press_time = time.time()
            if key_pressed == 3:
                print('ctrl + c: handling close')
                self.running = False
                break
            elif key_pressed == 26:
                print('ctrl + z: handling close')
                self.running = False
                break
            elif key_pressed == 0xE0:
                key_pressed = ord(getch())
                try:
                    log_message = 'press {}'.format(KeyAction(key_pressed).name)
                    print(log_message, file=sys.stderr)
                    # noinspection PyTypeChecker
                    self.event_dict[press_time - self.start_time] = log_message
                except ValueError as ve:
                    print('Unrecognized input: 224 + {}\n{}'.format(key_pressed, str(ve)))
        return

    def start_rand_act(self):
        while self.running:
            t = threading.Timer(self.DELAY_TIMER, self.print_random_action)
            t.start()
            t.join()
        return

    def cleanup(self):
        print('Cleaning up resources...')
        self.running = False
        with open(self.event_fname, 'w+') as event_file:
            json.dump(self.event_dict, event_file, indent=2)
        return


def main():
    # start hub
    # hub = HubCommunicator()
    # print('Hub is running: {}'.format(hub.hub_thread))
    # hub.start_scan()
    # sleep(5)
    # hub.connect_board()

    cli_recorder = CliRecorder(run_duration=-1)
    cli_recorder.run()
    cli_recorder.cleanup()

    # print('Cleaning up all resources...')
    # hub.clean_up()
    return


if __name__ == '__main__':
    main()
