"""
@title
@description
"""

CHECK_STATUS = {'type': 'status'}

SET_PROTOCOL = {'type': 'protocol', 'action': 'start', 'protocol': 'bled112'}
GET_PROTOCOL = {'type': 'protocol', 'action': 'status', 'protocol': 'bled112'}
STOP_PROTOCOL = {'type': 'protocol', 'action': 'stop', 'protocol': 'bled112'}

START_SCAN = {'type': 'scan', 'action': 'start'}
GET_SCAN = {'type': 'scan', 'action': 'status'}
STOP_SCAN = {'type': 'scan', 'action': 'stop'}

DISCONNECT_BOARD = {'type': 'disconnect'}

START_IMPEDANCE = {'type': 'impedance', 'action': 'start'}
STOP_IMPEDANCE = {'type': 'impedance', 'action': 'stop'}

START_STREAM = {'type': 'command', 'command': 'b'}
STOP_STREAM = {'type': 'command', 'command': 's'}

ENABLE_ACCEL = {'type': 'accelerometer', 'action': 'start'}
DISABLE_ACCEL = {'type': 'accelerometer', 'action': 'start'}

LOG_REGISTERS = {'type': 'command', 'command': '?'}


def connect_board_dict(board_name):
    return {'type': 'connect', 'name': board_name}


def turn_off_channels_dict(channel_list):
    return {'type': 'command', 'command': channel_list}


def turn_on_channels_dict(channel_list):
    return {'type': 'command', 'command': channel_list}
