"""
@title
    DataAcquisition.py
@description
"""
import ctypes
import json
import os
import socket
import subprocess
import threading
import time
from json import JSONDecodeError

import psutil

from SystemControl.OBciPython import HUB_EXE, OPENBCI_HUB_NAME
from SystemControl.OBciPython.EHub.GanglionMessages import SET_PROTOCOL, START_STREAM, connect_board, DISCONNECT_BOARD, \
    START_IMPEDANCE, STOP_IMPEDANCE
from SystemControl.OBciPython.EHub.MessageHandlers import get_message_handler
from SystemControl.SystemLog import SystemLog, SystemLogLevel, SystemLogIdent
from SystemControl.utils.ObserverObservable import Observable


class HubCommunicator(Observable):
    HUB_IP = '127.0.0.1'
    HUB_PORT = 10996
    MAX_BUFFER_SIZE = 8192
    SAMPLES_PER_ITER = 200

    def __init__(self, log_level=SystemLogLevel.NORMAL):
        Observable.__init__(self)

        self.msg_log = SystemLog(log_level)
        self.hub_instance = None
        self.listen_thread = None

        self.connect_hub()

        self.protocol = None
        self.board_name = None

        self.sample_count = 0
        return

    @property
    def state_dict(self):
        loc_addr = None
        rem_addr = None
        if self.hub_instance:
            loc_addr = self.hub_instance.getsockname()
            rem_addr = self.hub_instance.getpeername()
        ret_dict = {
            'local_address': loc_addr,
            'remote_address': rem_addr,
            'protocol': self.protocol,
            'board_connected': self.board_name
        }
        return ret_dict

    # def reset(self):
    #     self.msg_log.log_message(
    #         message_ident=SystemLogIdent.LOG,
    #         message_level=SystemLogLevel.NORMAL,
    #         message_string='Resetting hub instance'
    #     )
    #     self.clean_up()
    #
    #     # TODO stop all threads
    #     # todo flush message log - check to make sure appends if exists
    #     # todo clear data and impedance arrays
    #     # TODO close addr
    #
    #     self.connect_hub()
    #     return

    def start_hub_thread(self):
        if not os.path.exists(HUB_EXE):
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string=f'Unable to locate hub executable: {HUB_EXE}'
            )
            return False

        if self.is_hub_running():
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.NORMAL,
                message_string='Located running hub process'
            )
        else:
            subproc_command = f'{HUB_EXE}'
            subprocess.Popen(subproc_command, shell=False)
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.NORMAL,
                message_string='Hub thread started'
            )
        return True

    def is_hub_running(self):
        """
        Check if there is any running process that contains the given name processName.

        :return:
        """
        try:
            proc_list = sorted(list(psutil.process_iter()), key=lambda x: x.name())
            is_running = any(elem.name() == OPENBCI_HUB_NAME for elem in proc_list)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as pserr:
            self.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string=f'Unexpected error: {pserr}'
            )
            is_running = False
        return is_running

    def connect_hub(self):
        electron_hub_running = self.start_hub_thread()
        if electron_hub_running:
            try:
                self.hub_instance = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.hub_instance.connect((self.HUB_IP, self.HUB_PORT))

                # verify connections
                data = self.hub_instance.recv(self.MAX_BUFFER_SIZE)
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

                loc_addr = self.hub_instance.getsockname()
                rem_addr = self.hub_instance.getpeername()
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.NORMAL,
                    message_string=f'Connection to hub established: '
                                   f'{loc_addr[0]}:{loc_addr[1]} -> {rem_addr[0]}:{rem_addr[1]}'
                )

                self.listen()
                self.send_msg(SET_PROTOCOL)
            except (ConnectionResetError, ConnectionRefusedError) as ce:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.ERROR,
                    message_string=f'Failed to connect to hub: {ce}'
                )
                return False
        return True

    def clean_up(self):
        if self.board_name:
            self.send_msg(DISCONNECT_BOARD)

        self.stop_listener_thread()
        self.msg_log.flush_log()
        return

    def send_msg(self, msg_dict):
        msg_str = f'{json.dumps(msg_dict)}\r\n'
        self.msg_log.log_message(
            message_ident=SystemLogIdent.SND,
            message_level=SystemLogLevel.MEDIUM,
            message_string=msg_str
        )

        msg_bytes = msg_str.encode('utf-8')
        self.hub_instance.sendall(msg_bytes)
        return

    def listen(self):
        self.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.NORMAL,
            message_string='Starting listener...'
        )

        self.listen_thread = threading.Thread(target=self.handle_messages, daemon=True)
        self.listen_thread.start()
        return

    def stop_listener_thread(self):
        thread_id = self.listen_thread.ident
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print('Exception raise failure')

        self.hub_instance.shutdown(socket.SHUT_RDWR)
        self.hub_instance.close()
        return

    def handle_messages(self):
        listening = True
        while listening:
            try:
                data = self.hub_instance.recv(self.MAX_BUFFER_SIZE)
                recv_str = data.decode('utf-8')
                msg_str_list = recv_str.split('\n')
                for msg_str in msg_str_list:
                    if msg_str:
                        self.msg_log.log_message(
                            message_ident=SystemLogIdent.RCV,
                            message_level=SystemLogLevel.LOW,
                            message_string=msg_str,
                        )
                        try:
                            data_dict = json.loads(msg_str)
                            msg_type = data_dict.get('type', None)
                            msg_handler = get_message_handler(msg_type)
                            if msg_handler:
                                msg_handler(self, data_dict)
                            else:
                                self.msg_log.log_message(
                                    message_ident=SystemLogIdent.LOG,
                                    message_level=SystemLogLevel.ERROR,
                                    message_string=f'Unrecognized message type: {msg_type}'
                                )
                        except JSONDecodeError as jde:
                            self.msg_log.log_message(
                                message_ident=SystemLogIdent.LOG,
                                message_level=SystemLogLevel.ERROR,
                                message_string=f'Failed to decode message: {jde}\n-----\n{msg_str}\n-----'
                            )
            except (ConnectionResetError, ConnectionAbortedError) as ce:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.ERROR,
                    message_string=f'Connection to hub lost: {ce}'
                )
                listening = False
            except OSError as ose:
                self.msg_log.log_message(
                    message_ident=SystemLogIdent.LOG,
                    message_level=SystemLogLevel.ERROR,
                    message_string=f'Connection closed: {ose}'
                )
                listening = False
        self.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.LOW,
            message_string=f'Listen thread stopped'
        )
        return


def main():
    log_level = SystemLogLevel.DEBUG

    device = HubCommunicator(log_level=log_level)
    hub_state = device.state_dict
    print(f'Hub is running: {hub_state["local_address"]} -> {hub_state["remote_address"]}')
    time.sleep(1)
    device.send_msg(connect_board(device.board_name))
    # time.sleep(1)
    # device.send_msg(START_IMPEDANCE)
    # time.sleep(1)
    # device.send_msg(STOP_IMPEDANCE)
    time.sleep(5)
    print('Cleaning up all resources...')
    device.clean_up()
    print(device.sample_count)
    return


if __name__ == '__main__':
    main()
