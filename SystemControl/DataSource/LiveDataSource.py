"""
@title
@description
"""
import os
import threading
import time
from queue import Queue

from SystemControl import DATA_DIR
from SystemControl.DataSource.DataSource import DataSource, SampleEntry, EventEntry, SubjectEntry
from SystemControl.StimulusGenerator import MotorAction
from SystemControl.utilities import find_files_by_type, Observer


class LiveDataSource(DataSource, Observer):

    def update(self, source, update_message):
        if source in self.subscriptions:
            if source.__class__.__name__ == 'UdpClient':
                # self.add_sample(
                #     self.ports[client_port], time.time(),
                #     [uv_to_volts(sample) for sample in data_samples[:-1]]
                # )
                print(f'{source.__class__.__name__}: {update_message}')
            elif source.__class__.__name__ == 'StimulusGenerator':
                print(f'{source.__class__.__name__}: {update_message}')
        return

    def __init__(self, sub_list: list, subject: str, trial_type: str, log_level: str = 'CRITICAL'):
        DataSource.__init__(self, log_level)
        Observer.__init__(self, sub_list)

        self.name = 'recorded'
        self.sample_freq = 200
        self.trial_types = ['motor_imagery', 'baseline_open', 'baseline_closed']
        self.selected_trial_type = trial_type

        self.ascended_being = subject
        self.event_names = list(MotorAction.__members__)

        self.subject_save_dir = os.path.join(DATA_DIR, self.name, self.ascended_being)
        self.current_trial = self.__next_trial_id()

        self.init_time = time.time()
        self._samples_lock = threading.Lock()
        self._event_lock = threading.Lock()

        self.samples = []
        self.events = []
        subject_entry_fname = os.path.join(
            self.dataset_directory, self.ascended_being, f'{self.selected_trial_type}-{self.current_trial}.json'
        )
        self.subject_entry = SubjectEntry(
            path=subject_entry_fname, source_name=self.name, subject=self.ascended_being,
            trial_type=self.selected_trial_type, trial_name=self.current_trial, samples=self.samples, events=self.events
        )
        self.trial_info_dict.append(self.subject_entry)

        self._streaming_samples = False
        self._streaming_events = False

        self._sample_queue = Queue()
        self._event_queue = Queue()
        return

    def trial_type_from_name(self, trial_name):
        return self.selected_trial_type

    def stream_samples(self):
        self._streaming_samples = True
        # todo
        return

    def stream_events(self):
        self._streaming_events = True
        # todo
        return

    def __next_trial_id(self):
        prev_trials = sorted(find_files_by_type('json', self.subject_save_dir))
        if len(prev_trials) == 0:
            return f'{1:0>4}'

        last_trial = prev_trials[-1]
        last_trial_name, ext = os.path.splitext(last_trial)
        last_trial_name = last_trial_name.split(os.sep)[-1]
        last_trial_num = last_trial_name.split('-')[-1]
        last_trial_id = int(last_trial_num)
        return f'{last_trial_id + 1:0>4}'

    def add_sample(self, sample_type, timestamp, sample_data):
        with self._samples_lock:
            if sample_type == 'timeseries_filtered':
                data_points = {
                    self.coi[idx]: point
                    for idx, point in enumerate(sample_data)
                }
                sample_idx = len(self.samples)
                sample_entry = SampleEntry(
                    idx=sample_idx, timestamp=timestamp, data=data_points
                )
                self.samples.append(sample_entry)
        return

    def add_event(self, event_type, timestamp):
        with self._event_lock:
            with self._samples_lock:
                sample_idx = len(self.samples)
            if sample_idx > 0:
                event_entry = EventEntry(
                    idx=sample_idx, timestamp=timestamp, event_type=event_type
                )
                self.events.append(event_entry)
        return

    def set_subject(self, subject: str):
        self.ascended_being = subject
        self.current_trial = self.__next_trial_id()
        return

    def set_trial_type(self, trial_type: str):
        self.selected_trial_type = trial_type
        self.current_trial = self.__next_trial_id()
        return


def main():
    from SystemControl.OBciPython.UdpClient import UdpClient
    from SystemControl.StimulusGenerator import StimulusGenerator, GeneratorType

    subject_name = 'Tara'
    trial_type = 'motor_imagery'

    generate_delay = 1
    jitter_generator = 0.4
    run_time = 5
    verbosity = 0

    stimulus_generator = StimulusGenerator(
        delay=generate_delay, jitter=jitter_generator, generator_type=GeneratorType.SEQUENTIAL, verbosity=verbosity
    )
    udp_client = UdpClient()
    live_ds = LiveDataSource(sub_list=[stimulus_generator, udp_client], subject=subject_name, trial_type=trial_type)

    stimulus_generator.run()
    udp_client.run()
    time.sleep(run_time)
    stimulus_generator.stop()
    udp_client.stop()

    live_ds.save_data(use_mp=False, human_readable=True)
    return


if __name__ == '__main__':
    main()
