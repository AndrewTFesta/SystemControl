"""
@title
@description
"""
import time

import pandas as pd

from SystemControl.DataSource.DataSource import DataSource, build_entry_id, TrialInfoEntry, TrialDataEntry
from SystemControl.StimulusGenerator import MotorAction
from SystemControl.utils.ObserverObservable import Observer


class LiveDataSource(DataSource, Observer):

    def __init__(self, subject: str, trial_type: str, subscriber_list: list, log_level: str = 'CRITICAL',
                 save_method: str = 'h5'):
        DataSource.__init__(self, log_level, save_method=save_method)
        Observer.__init__(self, subscriber_list)

        self.name = 'recorded'
        self.sample_freq = 200
        self.trial_types = ['motor_imagery', 'baseline_open', 'baseline_closed']
        self.event_names = list(MotorAction.__members__)

        self.selected_trial_type = trial_type
        self.ascended_being = subject

        self.load_data()
        self.trial_name = self.__next_trial_name()
        self._current_event = MotorAction.REST

        self.entry_id = build_entry_id([self.name, self.ascended_being, self.selected_trial_type, self.trial_name])
        trial_info = TrialInfoEntry(
            entry_id=self.entry_id, source=self.name, subject=self.ascended_being,
            trial_type=self.selected_trial_type,
            trial_name=self.trial_name,
        )
        trial_entry_df = pd.DataFrame(data=[trial_info], columns=TrialInfoEntry._fields)
        self.trial_info_df = self.trial_info_df.append(trial_entry_df, ignore_index=True, sort=False)
        return

    def update(self, source, update_message):
        if source in self.subscriptions:
            if source.__class__.__name__ == 'UdpClient':
                update_time = update_message.get('time', None)
                update_type = update_message.get('type', None)
                update_data = update_message.get('data', None)

                self.add_sample(update_type, update_time, update_data)
            elif source.__class__.__name__ == 'StimulusGenerator':
                event = update_message.get('event', None)
                if event:
                    self.set_event(update_message['event'])
        return

    def __next_trial_name(self):
        if len(self.trial_info_df.index) == 0:
            return f'{1}'

        prev_trial_names = sorted(self.trial_info_df['trial_name'].tolist())
        return f'{prev_trial_names[-1] + 1}'

    def add_sample(self, sample_type, timestamp, sample_data):
        if sample_type == 'timeseries_filtered':
            trial_data_entry = TrialDataEntry(
                entry_id=self.entry_id, idx=len(self.trial_data_df.index),
                timestamp=timestamp, label=self._current_event.name,
                C3=sample_data[0], Cz=sample_data[1], C4=sample_data[2],
            )
            append_df = pd.DataFrame(data=[trial_data_entry], columns=TrialDataEntry._fields)
            self.trial_data_df = self.trial_data_df.append(append_df, ignore_index=True, sort=False)

            change_message = {
                'time': time.time(),
                'type': 'sample',
                'data': {'C3': 0, 'Cz': 0, 'C4': 0},
            }
            self.set_changed_message(change_message)
        return

    def set_event(self, event_type):
        self._current_event = event_type

        change_message = {
            'time': time.time(),
            'type': 'event',
            'event': self._current_event,
        }
        self.set_changed_message(change_message)
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
    save_method = 'csv'

    stimulus_generator = StimulusGenerator(
        delay=generate_delay, jitter=jitter_generator, generator_type=GeneratorType.SEQUENTIAL, verbosity=verbosity
    )
    udp_client = UdpClient()
    live_ds = LiveDataSource(
        subject=subject_name, trial_type=trial_type,
        subscriber_list=[stimulus_generator, udp_client],
        save_method=save_method
    )

    stimulus_generator.run()
    udp_client.run()
    time.sleep(run_time)
    stimulus_generator.stop()
    udp_client.stop()

    live_ds.save_data(start_time=0, end_time=-1)
    return


if __name__ == '__main__':
    main()
