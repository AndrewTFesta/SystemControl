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

import SystemControl
from SystemControl import HUB_EXE, TERMINAL_COLUMNS
from SystemControl.OBciPython.HubMachine.MessageHandlers import get_message_handler
from SystemControl.OBciPython.HubMachine.SendMessages import *
from SystemControl.SystemLog import SystemLog, SystemLogLevel, SystemLogIdent


class HubCommunicator:
    HUB_IP = '127.0.0.1'
    HUB_PORT = 10996
    BUFFER_SIZE = 1024

    def __init__(self, log_level=SystemLogLevel.NORMAL):
        self.msg_log = SystemLog(log_level)

        self.hub_thread = None
        self.hub_instance = None
        self.hub_running = None

        self.protocol = None
        self.scan = False

        self.board_name = None
        self.board_connected = False

        # self.impedance = {0: [], 1: [], 2: [], 3: []}
        # self.data = {0: [], 1: [], 2: [], 3: []}

        self.impedance = {}
        self.data = {}

        self.connect_hub()
        return

    @property
    def state_string(self):
        return 'Hub running: {}\nprotocol: {}\nscan in progress: {}\nboard name: {}\nboard connected: {}'.format(
            self.hub_running,
            self.protocol,
            self.scan,
            self.board_name,
            self.board_connected,
        )

    def start_hub_thread(self):
        if os.path.exists(HUB_EXE):
            self.hub_thread = threading.Thread(target=lambda: subprocess.call(HUB_EXE), daemon=True)
            self.hub_thread.start()
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_string='Hub thread started',
                message_level=SystemLogLevel.NORMAL
            )
        else:
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
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
                    self.hub_instance = None
                    self.hub_running = False
                    self.msg_log.log_message(
                        message_ident=SystemLogIdent.LOG,
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
        is_running = False
        for each_proc in proc_list:
            try:
                if 'OpenBCIHub' in each_proc.name():
                    is_running = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as pserr:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_string='Unexpected error: {}'.format(str(pserr)),
                    message_level=SystemLogLevel.ERROR
                )
        self.hub_running = is_running
        return False

    def connect_hub(self):
        if self.is_hub_running():
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_string='Hub already running',
                message_level=SystemLogLevel.ERROR
            )
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_string='Hub killed: {}'.format(self.kill_hub()),
                message_level=SystemLogLevel.ERROR
            )

        self.start_hub_thread()
        try:
            self.hub_instance = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.hub_instance.connect((self.HUB_IP, self.HUB_PORT))
            # verify connections
            data = self.hub_instance.recv(self.BUFFER_SIZE)
            data_str = data.decode('utf-8')
            self.msg_log.log_message(
                message_ident=SystemLogIdent.RCV,
                message_string=data_str,
                message_level=SystemLogLevel.DEBUG
            )

            data_dict = json.loads(data_str)
            res_code = data_dict.get('code', None)
            if res_code != 200:
                self.hub_instance = None
                self.hub_running = False
                self.hub_instance.close()
                return False
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_string='Connected to hub: starting listener',
                message_level=SystemLogLevel.NORMAL
            )
            self.hub_running = True

            self.listen()
            self.send_msg(SET_PROTOCOL)
        except (ConnectionResetError, ConnectionRefusedError) as ce:
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_string='Failed to connect to hub: {}'.format(str(ce)),
                message_level=SystemLogLevel.ERROR
            )
            return False
        return True

    def clean_up(self):
        self.send_msg(DISCONNECT_BOARD)
        self.kill_hub()

        self.hub_instance.close()
        self.hub_running = False

        self.msg_log.flush_log()
        # TODO write accel data to file
        # TODO write stream data to file
        return

    def send_msg(self, msg_dict):
        if self.hub_running:
            msg_str = '{}\r\n'.format(json.dumps(msg_dict))
            self.msg_log.log_message(
                message_ident=SystemLogIdent.SND,
                message_string=msg_str,
                message_level=SystemLogLevel.MEDIUM
            )

            msg_bytes = msg_str.encode('utf-8')
            self.hub_instance.sendall(msg_bytes)
        return

    def listen(self):
        self.hub_thread = threading.Thread(target=self.handle_messages)
        self.hub_thread.daemon = True
        self.hub_thread.start()
        return

    def handle_messages(self):
        while self.hub_running:
            try:
                data = self.hub_instance.recv(self.BUFFER_SIZE)
                data_str = data.decode('utf-8')
                data_str_list = data_str.split('\n')
                for each_data_str in data_str_list:
                    if each_data_str:
                        self.msg_log.log_message(
                            message_ident=SystemLogIdent.LOG,
                            message_string=each_data_str,
                            message_level=SystemLogLevel.NORMAL
                        )
                        try:
                            data_dict = json.loads(each_data_str)
                            msg_type = data_dict.get('type', None)
                            msg_handler = get_message_handler(msg_type)
                            if msg_handler:
                                msg_handler(self, data_dict)
                            else:
                                self.msg_log.log_message(
                                    message_ident=SystemLogIdent.LOG,
                                    message_string='Unrecognized message type: {}'.format(msg_type),
                                    message_level=SystemLogLevel.ERROR
                                )
                        except JSONDecodeError as jde:
                            err_msg = 'Failed to decode message: {}\n-----\n{}\n-----'.format(str(jde), each_data_str)
                            self.msg_log.log_message(
                                message_ident=SystemLogIdent.LOG,
                                message_string=err_msg,
                                message_level=SystemLogLevel.ERROR
                            )
            except (ConnectionResetError, ConnectionAbortedError) as ce:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_string='Connection to hub lost: {}'.format(str(ce)),
                    message_level=SystemLogLevel.ERROR
                )
                self.hub_running = False
        return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print('{}: VERSION: {}'.format(SystemControl.name, SystemControl.version))
        return

    device = HubCommunicator()
    print('Hub is running: %s' % device.hub_thread)
    time.sleep(2)
    device.send_msg(START_SCAN)
    time.sleep(2)
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
