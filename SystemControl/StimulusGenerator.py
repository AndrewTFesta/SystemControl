"""
@title
@description
"""
import random
import threading
import time
from time import sleep

from SystemControl.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import LiveDataSource, MotorAction


class StimulusGenerator:

    def __init__(self, data_source: DataSource, delay: int = 5, jitter: float = 0.4, seed: int = None):
        self.data_source = data_source
        self.seed = seed
        delta_jitter = jitter * delay
        self.delay_timer = (delay - delta_jitter, delay + delta_jitter)

        self.start_time = time.time()
        self.event_dict = {'start_absolute': self.start_time}
        self.running = False
        self.gen_thread = None

        self.callback_list = []
        return

    def __start_generator(self):
        while self.running:
            rand_delay = random.uniform(self.delay_timer[0], self.delay_timer[1])
            t = threading.Timer(rand_delay, self.gen_random_action)
            t.start()
            t.join()
        return

    def gen_random_action(self):
        rand_action = random.choice([each_action for each_action in MotorAction])
        act_time = time.time()
        d_time = act_time - self.start_time
        for callback_function in self.callback_list:
            callback_function(rand_action, act_time, 'random')
        self.event_dict[f'{d_time:0.6f}'] = rand_action
        return

    def run(self):
        self.running = True
        self.gen_thread = threading.Thread(target=self.__start_generator, args=(), daemon=True)
        self.gen_thread.start()
        return

    def add_callback(self, callback_func):
        self.callback_list.append(callback_func)
        return

    @staticmethod
    def print_callback(callback_arg, timestamp, action_type):
        print(f'time: {timestamp:0.4f}\t{action_type}\t{callback_arg.name}')
        return

    def log_event_callback(self, callback_arg, timestamp, action_type):
        self.data_source.add_event(event_type=callback_arg.name, timestamp=timestamp)
        return

    def stop(self):
        self.running = False
        # slight delay to give threads chance to close cleanly
        sleep(2)
        print('Generator stopping...')
        return


def main():
    subject_name = 'Andrew'
    trial_type = 'motor_imagery'
    generate_delay = 5
    jitter_generator = 0.4
    rand_seed = 42
    run_time = 12

    live_ds = LiveDataSource(subject=subject_name, trial_type=trial_type)
    stimulus_generator = StimulusGenerator(live_ds, delay=generate_delay, jitter=jitter_generator, seed=rand_seed)
    stimulus_generator.add_callback(stimulus_generator.log_event_callback)
    stimulus_generator.add_callback(stimulus_generator.print_callback)
    stimulus_generator.run()
    sleep(run_time)
    stimulus_generator.stop()
    live_ds.save_data(human_readable=True)

    for key, val in stimulus_generator.event_dict.items():
        print(f'{key}:{val}')
    return


if __name__ == '__main__':
    main()
