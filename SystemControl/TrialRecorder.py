"""
@title
@description
"""
from time import sleep

from SystemControl.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import LiveDataSource
from SystemControl.OBciPython.UdpClient import UdpClient
from SystemControl.StimulusGenerator import StimulusGenerator


class TrialRecorder:

    def __init__(self, data_source: DataSource, stimulus_generator: StimulusGenerator, udp_client: UdpClient):
        self.data_source = data_source
        self.stimulus_generator = stimulus_generator
        self.udp_client = udp_client
        return

    def run(self):
        self.stimulus_generator.run()
        self.udp_client.run()
        return

    def stop(self):
        self.udp_client.stop()
        self.stimulus_generator.stop()
        self.data_source.save_data()
        return


def main():
    record_length = 12
    current_subject = 'Andrew'
    trial_type = 'motor_imagery'
    generate_delay = 2
    jitter_generator = 0.2
    rand_seed = 42

    data_source = LiveDataSource(subject=current_subject, trial_type=trial_type)
    stimulus_generator = StimulusGenerator(data_source, delay=generate_delay, jitter=jitter_generator, seed=rand_seed)
    udp_client = UdpClient(data_source)

    trial_recorder = TrialRecorder(data_source, stimulus_generator, udp_client)
    trial_recorder.run()
    sleep(record_length)
    trial_recorder.stop()
    return


if __name__ == '__main__':
    main()
