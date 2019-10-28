"""
@title
@description
"""

# TODO  refactor for formatting
# todo docs
import inspect
import sys
import types

from SystemControl.OBciPython import HubCommunicator
from SystemControl.OBciPython.SendMessages import *
from SystemControl.SystemLog import SystemLogLevel, SystemLogIdent


def handle_message_status(device, data_dict):
    res_code = data_dict.get('code', None)
    if res_code == 200:
        device.hub_running = True
    else:
        device.hub_running = False
    return


def handle_message_scan(device, data_dict):
    res_code = data_dict.get('code', None)
    action_type = data_dict.get('action', None)
    board_name = data_dict.get('name', None)

    if res_code == 200:
        if action_type == 'start':
            device.scan = True
        elif action_type == 'stop':
            device.scan = False
        elif action_type == 'status':
            device.scan = True
        else:
            device.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string='Unrecognized scan action type: {}'.format(action_type)
            )
    elif res_code == 201:
        device.scan = True
        device.board_name = board_name
        device.send_msg(STOP_SCAN)
    elif res_code == 302:
        device.scan = True
    elif res_code == 303:
        device.scan = False
    elif res_code == 304:
        device.scan = False
    elif res_code == 305:
        device.scan = False
    elif res_code == 410:
        device.scan = False
    elif res_code == 411:
        device.scan = False
    elif res_code == 412:
        device.scan = False
    else:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Unrecognized scan result code: {}'.format(res_code)
        )
    return


def handle_message_protocol(device, data_dict):
    res_code = data_dict.get('code', None)
    action_type = data_dict.get('action', None)
    protocol = data_dict.get('protocol', None)

    if res_code == 200:
        if action_type == 'start':
            device.protocol = protocol
            # hub automatically starts scan once protocol is started
            device.send_msg(STOP_SCAN)
        elif action_type == 'stop':
            device.protocol = ''
        elif action_type == 'status':
            device.protocol = protocol
        else:
            device.msg_log.log_message(
                message_ident=SystemLogIdent.LOG,
                message_level=SystemLogLevel.ERROR,
                message_string='Unrecognized protocol action type: {}'.format(action_type)
            )
    elif res_code == 304:
        device.states['protocol'] = protocol
    elif res_code == 305:
        device.states['protocol'] = False
    elif res_code == 419:
        device.states['protocol'] = False
    elif res_code == 501:
        device.states['protocol'] = False
        pass
    else:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Unrecognized protocol result code: {}'.format(res_code)
        )
    return


def handle_message_impedance(device, data_dict):
    # res_code = data_dict.get('code', None)
    # res_type = data_dict.get('type', None)
    channel_num = data_dict.get('channelNumber', -1)
    imp_val = data_dict.get('impedanceValue', -1)

    if channel_num != -1:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.NORMAL,
            message_string='Channel: {} -> impedance: {}'.format(channel_num, imp_val)
        )
    return


def handle_message_connect(device, data_dict):
    res_code = data_dict.get('code', None)

    if res_code == 200:
        device.board_connected = True
    elif res_code == 402:
        device.board_connected = True
    elif res_code == 408:
        device.board_connected = True
    else:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Unrecognized board_connect result code: {}'.format(res_code)
        )
    return


def handle_message_disconnect(device, data_dict):
    res_code = data_dict.get('code', None)
    if res_code == 200:
        device.board_connected = False
    elif res_code == 401:
        device.board_connected = True
    else:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Unrecognized board_connect result code: {}'.format(res_code)
        )
    return


def handle_message_command(device, data_dict):
    res_code = data_dict.get('code', None)
    if res_code == 200:
        device.board_connected = False
    elif res_code == 406:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Unable to write to connected device: {}'.format(device.board_name)
        )
    elif res_code == 420:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Protocol of connected device is not selected: {}:{}'.format(
                device.board_name,
                device.protocol
            )
        )
    else:
        device.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_level=SystemLogLevel.ERROR,
            message_string='Unrecognized board_connect result code: {}'.format(res_code)
        )
    return


def handle_message_data(device, data_dict):
    sample_count = data_dict.get('sampleNumber', None)
    channel_data = data_dict.get('channelDataCounts', [])
    # accel_data = data_dict.get('accelDataCounts', [])
    b_time = data_dict.get('boardTime', [])
    t_stamp = data_dict.get('timestamp', [])
    is_valid = data_dict.get('valid', [])
    # res_code = data_dict.get('code', [])
    # res_type = data_dict.get('type', [])

    data_tuple = HubCommunicator.DataEntry(
        timestamp=t_stamp,
        board_sample=sample_count,
        absolute_sample=device.get_sample_count(),
        board_time=b_time,
        valid=is_valid,
        data=channel_data
    )
    device.data.append(data_tuple)
    device.add_sample_count(add_num=1)

    device.msg_log.log_message(
        message_ident=SystemLogIdent.LOG,
        message_level=SystemLogLevel.NORMAL,
        message_string='Sample: {} -> Channel data: {}'.format(device.get_sample_count(), channel_data)
    )
    return


def handle_message_log(device, data_dict):
    res_code = data_dict.get('code', None)
    device.msg_log.log_message(
        message_ident=SystemLogIdent.LOG,
        message_level=SystemLogLevel.NORMAL,
        message_string=f'Handling log message: {res_code}: Not implemented'
    )
    return


def handle_message_accelerometer(device, data_dict):
    res_code = data_dict.get('code', None)
    device.msg_log.log_message(
        message_ident=SystemLogIdent.LOG,
        message_level=SystemLogLevel.NORMAL,
        message_string=f'Handling accelerometer message: {res_code}: Not implemented'
    )
    return


def function_filter(function_object):
    is_valid = False
    is_function = isinstance(function_object, types.FunctionType)
    if is_function:
        if function_object.__name__.startswith('handle_message'):
            is_valid = True
    return is_valid


def get_message_handler(message_type):
    function_list = inspect.getmembers(sys.modules[__name__], predicate=function_filter)
    msg_handlers = {
        each_func[0].split('handle_message_')[1]: each_func[1]
        for each_func in function_list
    }
    return msg_handlers.get(message_type, None)
