"""
@title
@description
"""
import random
import threading
import time
from enum import Enum
from time import sleep

from SystemControl.utilities import Observable


class MotorAction(Enum):
    REST = 0
    RIGHT = 1
    LEFT = 2


class GeneratorType(Enum):
    SEQUENTIAL = 0
    RANDOM = 1


class StimulusGenerator(Observable):

    def __init__(self, generator_type: GeneratorType = GeneratorType.RANDOM, delay: int = 5, jitter: float = 0.4,
                 verbosity: int = 0):
        Observable.__init__(self)
        delta_jitter = jitter * delay
        self.delay_timer = (delay - delta_jitter, delay + delta_jitter)
        self.running = False
        self.gen_thread = None

        self.gen_func_map = {
            GeneratorType.SEQUENTIAL: self.sequential_action,
            GeneratorType.RANDOM: self.random_action,
        }
        self.generator_type = generator_type

        self._last_action = MotorAction.REST
        self._last_time = time.time()
        self.verbosity = verbosity
        return

    def __str__(self):
        return self.__class__.__name__

    def __start_generator(self):
        generator_func = self.gen_func_map[self.generator_type]
        while self.running:
            rand_delay = random.uniform(self.delay_timer[0], self.delay_timer[1])
            t = threading.Timer(rand_delay, generator_func)
            t.start()
            t.join()
        return

    def random_action(self):
        self._last_action = random.choice([each_action for each_action in MotorAction])
        self._last_time = time.time()
        change_message = {'time': self._last_time, 'event': self._last_action}
        self.set_changed_message(change_message)
        if self.verbosity > 0:
            print(f'{self.__str__()}: {self.change_message}')
        return

    def sequential_action(self):
        action_choices = [each_action for each_action in MotorAction]
        self._last_action = action_choices[(self._last_action.value + 1) % len(list(MotorAction.__members__))]
        self._last_time = time.time()
        change_message = {'time': self._last_time, 'event': self._last_action}
        self.set_changed_message(change_message)
        if self.verbosity > 0:
            print(f'{self.__str__()}: {self.change_message}')
        return

    def run(self):
        self.running = True
        self.gen_thread = threading.Thread(target=self.__start_generator, args=(), daemon=True)
        self.gen_thread.start()
        return

    def stop(self):
        self.running = False
        # slight delay to give threads chance to close cleanly
        sleep(0.1)
        print('Generator stopping...')
        return


def main():
    from SystemControl.DataSource.LiveDataSource import LiveDataSource

    subject_name = 'Tara'
    trial_type = 'motor_imagery'
    generate_delay = 1
    jitter_generator = 0.4
    run_time = 5
    verbosity = 0

    stimulus_generator = StimulusGenerator(
        delay=generate_delay, jitter=jitter_generator, generator_type=GeneratorType.SEQUENTIAL, verbosity=verbosity
    )
    live_ds = LiveDataSource(subscriber_list=[stimulus_generator], subject=subject_name, trial_type=trial_type)

    stimulus_generator.run()
    sleep(run_time)
    stimulus_generator.stop()

    live_ds.save_data(start_time=0, end_time=-1)
    return


if __name__ == '__main__':
    main()
