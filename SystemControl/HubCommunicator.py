"""
@title
    DataAcquisition.py
@description
"""
import argparse
import json
import logging
import os
import socket
import subprocess
import threading
import time

import psutil

from SystemControl import utilities, name, version


def kill_hub():
    proc_list = psutil.process_iter()
    for each_proc in proc_list:
        if 'OpenBCIHub' in each_proc.name():
            each_proc.kill()
            return True
    return False


def is_hub_running():
    """
    Check if there is any running process that contains the given name processName.

    :return:
    """
    proc_list = psutil.process_iter()
    for each_proc in proc_list:
        try:
            if 'OpenBCIHub' in each_proc.name():
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return None


class HubCommunicator:
    HUB_IP = '127.0.0.1'
    HUB_PORT = 10996
    BUFFER_SIZE = 1024

    def __init__(self):
        self.msg_handlers = {
            'accelerometer': self.accel_resp,
            'boardType': self.board_type_resp,
            'command': self.command_resp,
            'connect': self.connect_resp,
            'disconnect': self.disconnect_resp,
            'impedance': self.impedance_resp,
            'protocol': self.protocol_resp,
            'scan': self.scan_resp,
            'status': self.status_resp,
        }

        self.hub_thread = None
        self.hub_conn = None
        self.alive = False

        if is_hub_running():
            print('Found running hub')
            print('Hub killed: %s' % kill_hub())

        self.start_hub_thread()
        self.connect()
        return

    def start_hub_thread(self):
        hub_proc = os.path.join('OpenBCIHub', 'OpenBCIHub.exe')
        if os.path.exists(hub_proc):
            self.hub_thread = threading.Thread(target=lambda: subprocess.call(hub_proc))
            self.hub_thread.daemon = True
            self.hub_thread.start()
        else:
            return False
        return True

    def is_connected(self):
        return self.hub_conn.stillconnected()

    def connect(self):
        self.hub_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hub_conn.connect((self.HUB_IP, self.HUB_PORT))

        # verify connections
        data = self.hub_conn.recv(self.BUFFER_SIZE)
        data_dict = json.loads(data.decode('utf-8'))
        res_code = data_dict.get('code', None)
        if res_code != 200:
            return False
        self.alive = True
        self.listen()

        self.get_status()
        return True

    def clean_up(self):
        self.hub_conn.close()
        return

    def send_msg(self, msg_dict):
        msg_str = json.dumps(msg_dict)
        msg_bytes = '{}\r\n'.format(msg_str).encode('utf-8')
        self.hub_conn.sendall(msg_bytes)
        return

    def get_status(self):
        msg_dict = {
            'type': 'status'
        }
        self.send_msg(msg_dict)
        return

    def get_ganglion(self):

        return

    def listen(self):
        self.hub_thread = threading.Thread(target=self.handle_messages)
        self.hub_thread.daemon = True
        self.hub_thread.start()
        return

    def handle_messages(self):
        while True:
            data = self.hub_conn.recv(self.BUFFER_SIZE)
            data_str = data.decode('utf-8')
            print('Message received: %s' % data_str)
            data_dict = json.loads(data_str)
            msg_type = data_dict.get('type', None)
            msg_handler = self.msg_handlers.get(msg_type, None)
            if msg_handler:
                handle_ret = msg_handler(data_dict)
            else:
                print('Unable to handle message type: %s' % msg_type)
        return

    def cleanup(self):
        self.alive = False
        return

    def status_resp(self, data_dict):
        print('Handling status message')
        return

    def scan_resp(self, data_dict):
        print('Handling scan message')
        return

    def protocol_resp(self, data_dict):
        print('Handling protocol message')
        return

    def impedance_resp(self, data_dict):
        print('Handling impedance message')
        return

    def connect_resp(self, data_dict):
        print('Handling connect message')
        return

    def disconnect_resp(self, data_dict):
        print('Handling disconnect message')
        return

    def command_resp(self, data_dict):
        print('Handling command message')
        return

    def board_type_resp(self, data_dict):
        print('Handling board type message')
        return

    def accel_resp(self, data_dict):
        print('Handling accelerometer message')
        return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print('%s: VERSION: %s' % (name, version))
        return

    logging.getLogger().setLevel(logging.DEBUG)

    device = HubCommunicator()
    print('Hub is running: %s' % device.hub_thread)
    time.sleep(1)
    print('Cleaning up all resources...')
    device.clean_up()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', '-v', action='store_true',
                        help='prints the current version and exits')
    print('-' * utilities.TERMINAL_COLUMNS)
    print(parser.prog)
    print('-' * utilities.TERMINAL_COLUMNS)

    cargs = parser.parse_args()
    main(cargs)
