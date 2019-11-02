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
from SystemControl.DataSource import DataSource
from SystemControl.OBciPython.EHub.MessageHandlers import get_message_handler
from SystemControl.OBciPython.EHub.SendMessages import *
from SystemControl.SystemLog import SystemLog, SystemLogLevel, SystemLogIdent


class HubCommunicator:
    HUB_IP = '127.0.0.1'
    HUB_PORT = 10996
    BUFFER_SIZE = 1024
    SAMPLES_PER_ITER = 200

    def __init__(self, data_source: DataSource, log_level=SystemLogLevel.NORMAL):
        self.msg_log = SystemLog(log_level)

        self.hub_thread = None
        self.hub_instance = None

        self.protocol = None
        self.scan = False

        self.board_name = None
        self.board_connected = False

        # todo  make separate classes to handle datasets
        #       data separate from impedance or make impedance property of channel dataset?
        #       add annotation tracking
        #       integrate with pandas
        # [ImpedanceEntry(...), ImpedanceEntry(...), ...]
        # self.impedance = []
        # [DataEntry(...), DataEntry(...), ...]
        # self.data = []
        self.data_source = data_source

        self.connect_hub()
        return

    @property
    def state_string(self):
        loc_addr = None
        rem_addr = None
        if self.hub_instance:
            loc_addr = self.hub_instance.getsockname()
            rem_addr = self.hub_instance.getpeername()
        ret_str = \
            f'Hub running: {loc_addr[0]}:{loc_addr[1]} -> {rem_addr[0]}:{rem_addr[1]}\n' \
            f'protocol: {self.protocol}\n' \
            f'scan in progress: {self.scan}\n' \
            f'board name: {self.board_name}\n' \
            f'board connected: {self.board_connected}'
        return ret_str

    def reset(self):
        self.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.NORMAL,
            message_string='Resetting hub instance'
        )
        self.clean_up()

        # TODO stop all threads
        # todo flush message log - check to make sure appends if exists
        # todo clear data and impedance arrays
        # TODO close addr

        self.connect_hub()
        return

    def start_hub_thread(self):
        if os.path.exists(HUB_EXE):
            if self.is_hub_running():
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.NORMAL,
                    message_string='Located running hub process'
                )
            else:
                subproc_command = '{}'.format(HUB_EXE)
                subprocess.Popen(subproc_command, shell=False)
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.NORMAL,
                    message_string='Hub thread started'
                )
        else:
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string='Unable to locate hub executable: {}'.format(HUB_EXE)
            )
            return False
        return True

    def is_hub_running(self):
        """
        Check if there is any running process that contains the given name processName.

        :return:
        """
        try:
            proc_list = list(psutil.process_iter())
            proc_list.sort(key=lambda x: x.name())
            is_running = any(elem.name() == SystemControl.OPENBCI_HUB_NAME for elem in proc_list)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as pserr:
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string='Unexpected error: {}'.format(str(pserr))
            )
            is_running = False
        return is_running

    def connect_hub(self):
        self.start_hub_thread()
        try:
            self.hub_instance = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.hub_instance.connect((self.HUB_IP, self.HUB_PORT))
            # verify connections
            data = self.hub_instance.recv(self.BUFFER_SIZE)
            data_str = data.decode('utf-8')
            self.msg_log.log_message(
                message_ident=SystemLogIdent.RCV,
                message_level=SystemLogLevel.NORMAL,
                message_string=data_str
            )

            data_dict = json.loads(data_str)
            res_code = data_dict.get('code', None)
            if res_code != 200:
                self.hub_instance = None
                self.hub_instance.close()
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.ERROR,
                    message_string='Unable to verify connection: Closing connection to hub'
                )
                return False
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.NORMAL,
                message_string='Connected to hub: starting listener'
            )

            self.listen()
            self.send_msg(SET_PROTOCOL)
        except (ConnectionResetError, ConnectionRefusedError) as ce:
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string='Failed to connect to hub: {}'.format(str(ce))
            )
            return False
        return True

    def clean_up(self):
        self.send_msg(DISCONNECT_BOARD)
        self.hub_instance.close()
        self.msg_log.flush_log()

        time_stamp = time.strftime('%d_%b_%Y_%H_%M_%S', time.gmtime())

        # write stream data to file
        data_fname = os.path.join(SystemControl.RECORDED_DATA_DIR, 'data_stream_{}.json'.format(time_stamp))
        data_list = [each_entry.as_dict() for each_entry in self.data]
        with open(data_fname, 'w+') as data_stream_file:
            json.dump(data_list, data_stream_file, indent=2)

        # write impedance data to file
        impedance_fname = os.path.join(SystemControl.RECORDED_DATA_DIR, 'impedance_{}.json'.format(time_stamp))
        impedance_list = [each_entry.as_dict() for each_entry in self.impedance]
        with open(impedance_fname, 'w+') as impedance_file:
            json.dump(impedance_list, impedance_file, indent=2)
        return

    def send_msg(self, msg_dict):
        msg_str = '{}\r\n'.format(json.dumps(msg_dict))
        self.msg_log.log_message(
            message_ident=SystemLogIdent.SND,
            message_level=SystemLogLevel.MEDIUM,
            message_string=msg_str
        )

        msg_bytes = msg_str.encode('utf-8')
        self.hub_instance.sendall(msg_bytes)
        return

    def listen(self):
        self.hub_thread = threading.Thread(target=self.handle_messages, daemon=True)
        self.hub_thread.start()
        return

    def handle_messages(self):
        while True:
            try:
                data = self.hub_instance.recv(self.BUFFER_SIZE)
                data_str = data.decode('utf-8')
                data_str_list = data_str.split('\n')
                for each_data_str in data_str_list:
                    if each_data_str:
                        self.msg_log.log_message(
                            message_ident=SystemLogIdent.RCV,
                            message_level=SystemLogLevel.LOW,
                            message_string=each_data_str,
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
                                    message_level=SystemLogLevel.ERROR,
                                    message_string='Unrecognized message type: {}'.format(msg_type)
                                )
                        except JSONDecodeError as jde:
                            err_msg = 'Failed to decode message: {}\n-----\n{}\n-----'.format(str(jde), each_data_str)
                            self.msg_log.log_message(
                                message_ident=SystemLogIdent.LOG,
                                message_level=SystemLogLevel.ERROR,
                                message_string=err_msg
                            )
            except (ConnectionResetError, ConnectionAbortedError) as ce:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.ERROR,
                    message_string='Connection to hub lost: {}'.format(str(ce))
                )
                break
            except OSError as ose:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.ERROR,
                    message_string='Connection closed: {}'.format(str(ose))
                )
                break
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
