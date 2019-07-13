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


class HubCommunicator:
    HUB_IP = '127.0.0.1'
    HUB_PORT = 10996
    BUFFER_SIZE = 1024

    def __init__(self):
        self.msg_handlers = {
            'accelerometer': self.accel_resp,
            'command': self.command_resp,
            'connect_hub': self.connect_resp,
            'disconnect': self.disconnect_resp,
            'impedance': self.impedance_resp,
            'protocol': self.protocol_resp,
            'scan': self.scan_resp,
            'status': self.status_resp,
        }

        self.hub_thread = None
        self.hub_conn = None
        self.states = {
            'hub_connected': False,
            'protocol': None,
            'board': None,
            'board_connected': None,
            'scan': False
        }

        if self.is_hub_running():
            print('Found running hub')
            print('Hub killed: %s' % self.kill_hub())

        self.start_hub_thread()
        self.connect_hub()
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

    def kill_hub(self):
        if self.is_hub_running():
            proc_list = psutil.process_iter()
            for each_proc in proc_list:
                if 'OpenBCIHub' in each_proc.name():
                    each_proc.kill()
                    self.states['connected'] = False
                    return True
        return False

    def is_hub_running(self):
        """
        Check if there is any running process that contains the given name processName.

        :return:
        """
        proc_list = psutil.process_iter()
        for each_proc in proc_list:
            try:
                if 'OpenBCIHub' in each_proc.name():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as pserr:
                print('Unexpected error')
                print(pserr)
        self.states['connected'] = False
        return False

    def is_hub_connected(self):
        return self.states['connected']

    def connect_hub(self):
        self.hub_conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.hub_conn.connect((self.HUB_IP, self.HUB_PORT))

        try:
            # verify connections
            data = self.hub_conn.recv(self.BUFFER_SIZE)
            data_dict = json.loads(data.decode('utf-8'))
            res_code = data_dict.get('code', None)
            if res_code != 200:
                return False
            self.states['connected'] = True
            self.listen()

            self.check_status()
        except ConnectionResetError as ce:
            print('Failed to connect_hub to hub')
            print(str(ce))
            return False
        return True

    def clean_up(self):
        self.hub_conn.close()
        return

    def send_msg(self, msg_dict):
        if self.states['connected']:
            msg_str = json.dumps(msg_dict)
            msg_bytes = '{}\r\n'.format(msg_str).encode('utf-8')
            self.hub_conn.sendall(msg_bytes)
        return

    def check_status(self):
        msg_dict = {
            'type': 'status'
        }
        self.send_msg(msg_dict)
        return

    def set_protocol(self):
        msg_dict = {
            'type': 'protocol',
            'action': 'start',
            'protocol': 'bled112'
        }
        self.send_msg(msg_dict)
        return

    def get_protocol(self):
        msg_dict = {
            'type': 'protocol',
            'action': 'status',
            'protocol': 'bled112'
        }
        self.send_msg(msg_dict)
        return

    def stop_protocol(self):
        msg_dict = {
            'type': 'protocol',
            'action': 'stop',
            'protocol': 'bled112'
        }
        self.send_msg(msg_dict)
        return

    def start_scan(self):
        msg_dict = {
            'type': 'scan',
            'action': 'start'
        }
        self.send_msg(msg_dict)
        return

    def get_scan(self):
        msg_dict = {
            'type': 'scan',
            'action': 'status'
        }
        self.send_msg(msg_dict)
        return

    def stop_scan(self):
        msg_dict = {
            'type': 'scan',
            'action': 'stop'
        }
        self.send_msg(msg_dict)
        return

    def connect_board(self):
        # TODO
        return

    def disconnect_board(self):
        # TODO
        return

    def listen(self):
        self.hub_thread = threading.Thread(target=self.handle_messages)
        self.hub_thread.daemon = True
        self.hub_thread.start()
        return

    def handle_messages(self):
        while self.states['connected']:
            try:
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
            except ConnectionResetError:
                print('Connection to hub lost')
                self.states['connected'] = False
        return

    def cleanup(self):
        self.states['connected'] = False
        return

    def status_resp(self, data_dict):
        print('Handling status message')
        res_code = data_dict.get('code', None)
        if res_code == 200:
            self.states['connected'] = True
        else:
            self.states['connected'] = False
        return

    def scan_resp(self, data_dict):
        print('Handling scan message')
        res_code = data_dict.get('code', None)
        action_type = data_dict.get('action', None)

        if action_type == 'status':
            if res_code == 302:
                self.states['scan'] = True
            elif res_code == 303:
                self.states['scan'] = False
            elif res_code == 304:
                self.states['scan'] = False
            elif res_code == 305:
                self.states['scan'] = False
        elif action_type == 'start':
            if res_code == 200:
                self.states['scan'] = True
            elif res_code == 412:
                self.states['scan'] = False
            elif res_code == 411:
                self.states['scan'] = True
        elif action_type == 'stop':
            if res_code == 200:
                self.states['scan'] = False
            elif res_code == 410:
                self.states['scan'] = False
            elif res_code == 411:
                self.states['scan'] = True
        else:
            print('Unrecognized action type: %s' % action_type)
        return

    def protocol_resp(self, data_dict):
        print('Handling protocol message')
        res_code = data_dict.get('code', None)
        action_type = data_dict.get('action', None)
        protocol = data_dict.get('protocol', None)

        if action_type == 'status':
            if res_code == 200:
                self.states['protocol'] = protocol
            elif res_code == 304:
                self.states['protocol'] = protocol
            elif res_code == 305:
                self.states['protocol'] = False
            elif res_code == 419:
                self.states['protocol'] = False
            elif res_code == 501:
                self.states['protocol'] = False
        elif action_type == 'start':
            if res_code == 200:
                self.states['protocol'] = protocol
        elif action_type == 'stop':
            if res_code == 200:
                self.states['protocol'] = False
        else:
            print('Unrecognized action type: %s' % action_type)
        return

    def impedance_resp(self, data_dict):
        print('Handling impedance message')
        return

    def connect_resp(self, data_dict):
        print('Handling connect_hub message')
        return

    def disconnect_resp(self, data_dict):
        print('Handling disconnect message')
        return

    def command_resp(self, data_dict):
        print('Handling command message')
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
