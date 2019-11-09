"""
@title
@description
"""
import json
import socket
import threading
import time
from json import JSONDecodeError
from time import sleep

from SystemControl.DataSource import DataSource
from SystemControl.DataSource.LiveDataSource import LiveDataSource
from SystemControl.utilities import uv_to_volts


class UdpClient:
    HOST = '127.0.0.1'
    MAX_BUFFER_SIZE = 8192

    def __init__(self, data_source: DataSource):
        self.ports = {
            12345: 'timeseries_raw',
            12346: 'timeseries_filtered',
            12347: 'fft',
        }
        self._clients = {}
        self.data_source = data_source
        self.listening = False
        return

    def __init_all_clients(self):
        self.listening = True
        connect_threads = []
        for listener_port, listener_type in self.ports.items():
            connect_thread = threading.Thread(
                target=self.__init_client, args=(listener_type, listener_port), daemon=True
            )
            connect_threads.append(connect_thread)
            connect_thread.start()

        for connect_thread in connect_threads:
            connect_thread.join()
        return

    def __init_client(self, client_type, client_port):
        client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (self.HOST, client_port)
        print(f'Binding to \'{client_type}\' UDP listener: {self.HOST}:{client_port}')
        client.bind(server_address)
        self._clients[client_type] = {'client': client, 'listen_thread': None}
        return

    def __init_listeners(self):
        for client_name, client_entry in self._clients.items():
            client = client_entry['client']
            listen_thread = threading.Thread(target=self.__hey_listen, args=(client,), daemon=True)
            client_entry['listen_thread'] = listen_thread
            listen_thread.start()
        return

    def __hey_listen(self, client):
        client_host, client_port = client.getsockname()
        print(f'Starting listener: {self.ports[client_port]}')
        while self.listening:
            data, addr = client.recvfrom(self.MAX_BUFFER_SIZE)
            data_str = data.decode('utf-8')
            try:
                j_data = json.loads(data_str.strip())
                data_type = j_data.get('type', None)
                if data_type == 'eeg':
                    data_samples = j_data.get('data', None)
                    if data_samples:
                        self.data_source.add_sample(
                            self.ports[client_port], time.time(),
                            [uv_to_volts(sample) for sample in data_samples[:-1]]
                        )
            except JSONDecodeError as jde:
                print(f'Error while sanitizing sample: {jde}\n{data_str}')
        print(f'Closing listener: {self.ports[client_port]}')
        return

    def run(self):
        print('Starting listening...')
        self.listening = True

        self.__init_all_clients()
        self.__init_listeners()
        return

    def stop(self):
        self.listening = False
        # slight delay to give threads chance to close cleanly
        sleep(2)
        print('UdpClient stopping...')
        return


def main():
    record_length = 5
    current_subject = 'Tara'
    trial_type = 'motor_imagery'
    data_source = LiveDataSource(subject=current_subject, trial_type=trial_type)

    udp_client = UdpClient(data_source)
    udp_client.run()
    sleep(record_length)
    udp_client.stop()
    data_source.save_data(use_mp=False, human_readable=True)
    return


if __name__ == '__main__':
    main()
