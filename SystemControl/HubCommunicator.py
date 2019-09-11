"""
@title
    DataAcquisition.py
@description
"""
import argparse
import json
import os
import socket
import subprocess
import threading
import time
from json import JSONDecodeError

import psutil

from SystemControl import name, version, HUB_EXE, TERMINAL_COLUMNS
from SystemControl.SystemLog import SystemLog, SystemLogLevel


class HubCommunicator:
    HUB_IP = '127.0.0.1'
    HUB_PORT = 10996
    BUFFER_SIZE = 1024

    def __init__(self):
        self.msg_log = SystemLog()

        self.msg_handlers = {
            'accelerometer': self.accel_resp,
            'command': self.command_resp,
            'connect': self.connect_resp,
            'disconnect': self.disconnect_resp,
            'impedance': self.impedance_resp,
            'protocol': self.protocol_resp,
            'scan': self.scan_resp,
            'status': self.status_resp,
            'data': self.data_resp,
            'log': self.log_resp,
        }

        self.hub_thread = None
        self.hub_conn = None
        self.states = {
            'hub_connected': False,
            'protocol': None,
            'board': None,
            'board_connected': None,
            'scan': False,
            'impedance': {0: [], 1: [], 2: [], 3: []},
            'data': {0: [], 1: [], 2: [], 3: []}
        }

        if self.is_hub_running():
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Hub already running',
                message_level=SystemLogLevel.ERROR
            )
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Hub killed: {}'.format(self.kill_hub()),
                message_level=SystemLogLevel.ERROR
            )

        self.start_hub_thread()
        self.connect_hub()
        self.set_protocol()
        return

    def start_hub_thread(self):
        if os.path.exists(HUB_EXE):
            self.hub_thread = threading.Thread(target=lambda: subprocess.call(HUB_EXE))
            self.hub_thread.daemon = True
            self.hub_thread.start()
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Hub thread started',
                message_level=SystemLogLevel.NORMAL
            )
        else:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Hub thread failed to start',
                message_level=SystemLogLevel.ERROR
            )
            return False
        return True

    def kill_hub(self):
        if self.is_hub_running():
            proc_list = psutil.process_iter()
            for each_proc in proc_list:
                if 'OpenBCIHub' in each_proc.name():
                    each_proc.kill()
                    self.states['connected'] = False
                    self.msg_log.log_message(
                        message_ident='LOG',
                        message_string='Hub killed',
                        message_level=SystemLogLevel.NORMAL
                    )
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
                self.msg_log.log_message(
                    message_ident='LOG',
                    message_string='Unexpected error: {}'.format(str(pserr)),
                    message_level=SystemLogLevel.ERROR
                )
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
            data_str = data.decode('utf-8')
            self.msg_log.log_message(
                message_ident='LOG',
                message_string=data_str,
                message_level=SystemLogLevel.NORMAL
            )

            data_dict = json.loads(data_str)
            res_code = data_dict.get('code', None)
            if res_code != 200:
                return False
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Connected to hub: starting listener',
                message_level=SystemLogLevel.NORMAL
            )
            self.states['connected'] = True

            self.listen()
            self.check_status()
        except ConnectionResetError as ce:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Failed to connect to hub: {}'.format(str(ce)),
                message_level=SystemLogLevel.ERROR
            )
            print('Failed to connect to hub')
            print(str(ce))
            return False
        return True

    def clean_up(self):
        self.disconnect_board()
        self.kill_hub()
        self.hub_conn.close()
        self.msg_log.flush_log()
        return

    def send_msg(self, msg_dict):
        if self.states['connected']:
            msg_str = '{}\r\n'.format(json.dumps(msg_dict))
            self.msg_log.log_message(
                message_ident='LOG',
                message_string=msg_str,
                message_level=SystemLogLevel.NORMAL
            )

            msg_bytes = msg_str.encode('utf-8')
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
        msg_dict = {
            'type': 'connect',
            'name': self.states['board']
        }
        self.send_msg(msg_dict)
        return

    def disconnect_board(self):
        msg_dict = {
            'type': 'disconnect'
        }
        self.send_msg(msg_dict)
        return

    def turn_off_channels(self, channel_list):
        msg_dict = {
            'type': 'command',
            'command': channel_list
        }
        self.send_msg(msg_dict)
        return

    def turn_on_channels(self, channel_list):
        msg_dict = {
            'type': 'command',
            'command': channel_list
        }
        self.send_msg(msg_dict)
        return

    def start_impedance(self):
        msg_dict = {
            'type': 'impedance',
            'action': 'start'
        }
        self.send_msg(msg_dict)
        return

    def stop_impedance(self):
        msg_dict = {
            'type': 'impedance',
            'action': 'stop'
        }
        self.send_msg(msg_dict)
        return

    def start_stream(self):
        msg_dict = {
            'type': 'command',
            'command': 'b'
        }
        self.send_msg(msg_dict)
        return

    def stop_stream(self):
        msg_dict = {
            'type': 'command',
            'command': 's'
        }
        self.send_msg(msg_dict)
        return

    def enable_accel(self):
        msg_dict = {
            'type': 'accelerometer',
            'action': 'start'
        }
        self.send_msg(msg_dict)
        return

    def disable_accel(self):
        msg_dict = {
            'type': 'accelerometer',
            'action': 'start'
        }
        self.send_msg(msg_dict)
        return

    def log_registers(self):
        msg_dict = {
            'type': 'command',
            'command': '?'
        }
        self.send_msg(msg_dict)
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
                self.msg_log.log_message(
                    message_ident='LOG',
                    message_string=data_str,
                    message_level=SystemLogLevel.NORMAL
                )

                data_dict = json.loads(data_str)
                msg_type = data_dict.get('type', None)
                msg_handler = self.msg_handlers.get(msg_type, None)
                if msg_handler:
                    msg_handler(data_dict)
                else:
                    self.msg_log.log_message(
                        message_ident='LOG',
                        message_string='Unrecognized message type: {}'.format(msg_type),
                        message_level=SystemLogLevel.ERROR
                    )
            except (ConnectionResetError, ConnectionAbortedError) as ce:
                self.msg_log.log_message(
                    message_ident='LOG',
                    message_string='Connection to hub lost: {}'.format(str(ce)),
                    message_level=SystemLogLevel.ERROR
                )
                self.states['connected'] = False
            except JSONDecodeError as jde:
                self.msg_log.log_message(
                    message_ident='LOG',
                    message_string='Failed to decode message: {}'.format(str(jde)),
                    message_level=SystemLogLevel.ERROR
                )
        return

    def status_resp(self, data_dict):
        res_code = data_dict.get('code', None)
        if res_code == 200:
            self.states['connected'] = True
        else:
            self.states['connected'] = False
        return

    def scan_resp(self, data_dict):
        res_code = data_dict.get('code', None)
        action_type = data_dict.get('action', None)
        board_name = data_dict.get('name', None)

        if res_code == 200:
            if action_type == 'start':
                self.states['scan'] = True
            elif action_type == 'stop':
                self.states['scan'] = False
            elif action_type == 'status':
                self.states['scan'] = True
            else:
                self.msg_log.log_message(
                    message_ident='LOG',
                    message_string='Unrecognized scan action type: {}'.format(action_type),
                    message_level=SystemLogLevel.ERROR
                )
        elif res_code == 201:
            self.states['scan'] = True
            self.states['board'] = board_name
            self.stop_scan()
        elif res_code == 302:
            self.states['scan'] = True
        elif res_code == 303:
            self.states['scan'] = False
        elif res_code == 304:
            self.states['scan'] = False
        elif res_code == 305:
            self.states['scan'] = False
        elif res_code == 410:
            self.states['scan'] = False
        elif res_code == 411:
            self.states['scan'] = True
        elif res_code == 412:
            self.states['scan'] = False
        else:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Unrecognized scan result code: {}'.format(res_code),
                message_level=SystemLogLevel.ERROR
            )
        return

    def protocol_resp(self, data_dict):
        res_code = data_dict.get('code', None)
        action_type = data_dict.get('action', None)
        protocol = data_dict.get('protocol', None)

        if res_code == 200:
            if action_type == 'start':
                self.states['protocol'] = protocol
                # hub automatically starts scan once protocol is started
                self.stop_scan()
            elif action_type == 'stop':
                self.states['protocol'] = False
            elif action_type == 'status':
                self.states['protocol'] = protocol
            else:
                self.msg_log.log_message(
                    message_ident='LOG',
                    message_string='Unrecognized protocol action type: {}'.format(action_type),
                    message_level=SystemLogLevel.ERROR
                )
        elif res_code == 304:
            self.states['protocol'] = protocol
        elif res_code == 305:
            self.states['protocol'] = False
        elif res_code == 419:
            self.states['protocol'] = False
        elif res_code == 501:
            self.states['protocol'] = False
            pass
        else:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Unrecognized protocol result code: {}'.format(res_code),
                message_level=SystemLogLevel.ERROR
            )
        return

    def impedance_resp(self, data_dict):
        res_code = data_dict.get('code', None)
        res_type = data_dict.get('type', None)
        channel_num = data_dict.get('channelNumber', -1)
        imp_val = data_dict.get('impedanceValue', -1)

        if channel_num != -1:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Channel: {} -> impedance: {}'.format(channel_num, imp_val),
                message_level=SystemLogLevel.NORMAL
            )
        return

    def connect_resp(self, data_dict):
        res_code = data_dict.get('code', None)

        if res_code == 200:
            self.states['board_connected'] = True
        elif res_code == 402:
            self.states['board_connected'] = True
        elif res_code == 408:
            self.states['board_connected'] = True
        else:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Unrecognized board_connect result code: {}'.format(res_code),
                message_level=SystemLogLevel.ERROR
            )
        return

    def disconnect_resp(self, data_dict):
        res_code = data_dict.get('code', None)
        if res_code == 200:
            self.states['board_connected'] = False
        elif res_code == 401:
            self.states['board_connected'] = True
        else:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Unrecognized board_connect result code: {}'.format(res_code),
                message_level=SystemLogLevel.ERROR
            )
        return

    def command_resp(self, data_dict):
        res_code = data_dict.get('code', None)
        if res_code == 200:
            self.states['board_connected'] = False
        elif res_code == 406:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Unable to write to connected device: {}'.format(self.states['board']),
                message_level=SystemLogLevel.ERROR
            )
        elif res_code == 420:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Protocol of connected device is not selected: {}:{}'.format(
                    self.states['board'],
                    self.states['protocol']
                ),
                message_level=SystemLogLevel.ERROR
            )
        else:
            self.msg_log.log_message(
                message_ident='LOG',
                message_string='Unrecognized board_connect result code: {}'.format(res_code),
                message_level=SystemLogLevel.ERROR
            )
        return

    def data_resp(self, data_dict):
        sample_count = data_dict.get('sampleNumber', None)
        channel_data = data_dict.get('channelDataCounts', [])
        accel_data = data_dict.get('accelDataCounts', [])
        b_time = data_dict.get('boardTime', [])
        t_stamp = data_dict.get('timestamp', [])
        is_valid = data_dict.get('valid', [])
        res_code = data_dict.get('code', [])
        res_type = data_dict.get('type', [])

        self.msg_log.log_message(
            message_ident='LOG',
            message_string='Sample: {} -> Channel data: {}'.format(sample_count, channel_data),
            message_level=SystemLogLevel.NORMAL
        )
        return

    def log_resp(self, data_dict):
        # print(data_dict)
        return

    def accel_resp(self, data_dict):
        res_code = data_dict.get('code', None)

        self.msg_log.log_message(
            message_ident='LOG',
            message_string='Handling accelerometer message',
            message_level=SystemLogLevel.NORMAL
        )
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
    print('-' * TERMINAL_COLUMNS)
    print(parser.prog)
    print('-' * TERMINAL_COLUMNS)

    cargs = parser.parse_args()
    main(cargs)
