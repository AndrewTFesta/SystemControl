"""
@title
@description
"""
from SystemControl.OBciPython.EHub.SendMessages import *


def select_0_exit(device):
    device.exit = True
    return


def select_0_refresh_hub(device):
    device.hub.reset()
    return


def select_1_scan(device):
    device.hub.send_msg(START_SCAN)
    return


def select_2_connect(device):
    device.hub.send_msg(connect_board_dict(device.hub.board_name))
    return


def select_2_disconnect(device):
    device.hub.send_msg(DISCONNECT_BOARD)
    return


def select_5_start_impedance_test(device):
    device.hub.send_msg(START_IMPEDANCE)
    return


def select_5_stop_impedance_test(device):
    device.hub.send_msg(STOP_IMPEDANCE)
    return


def select_6_start_stream(device):
    device.hub.send_msg(START_STREAM)
    return


def select_6_stop_stream(device):
    device.hub.send_msg(STOP_STREAM)
    return


def select_7_log_registers(device):
    device.hub.send_msg(LOG_REGISTERS)
    return
