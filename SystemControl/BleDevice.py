"""

"""
import argparse
import binascii
import time

import pygatt
from pygatt.backends.bgapi.exceptions import BGAPIError
from serial.tools.list_ports_windows import comports

from SystemControl import version, utilities, name

BLE_SERVICE = 'fe84'

BLE_UUID_DICT = {
    'CHAR_RECEIVE': '2d30c082f39f4ce6923f3484ea480596',
    'CHAR_SEND': '2d30c083f39f4ce6923f3484ea480596',
    'CHAR_DISCONNECT': '2d30c084f39f4ce6923f3484ea480596',
    'BOARD': '00002a0000001000800000805f9b34fb',
    'SITE': '00002a2900001000800000805f9b34fb',
    'VERSION': '00002a2800001000800000805f9b34fb'
}

UNKNOWN_UUIDS = [
    '00002a0100001000800000805f9b34fb',
    '00002a0400001000800000805f9b34fb',
    '00002a2400001000800000805f9b34fb',
    '00002a2600001000800000805f9b34fb',
    '00002a2700001000800000805f9b34fb'
]

COMMAND_STOP_BINARY = 's'
COMMAND_START_BINARY = 'b'

COMMAND_STOP_IMPEDANCE = 'Z'
COMMAND_START_IMPEDANCE = 'z'


def parse_packet(packet_data):
    start_byte = packet_data[0]
    # Give the informative part of the packet to proper handler
    # split between ID and data bytes
    # Raw uncompressed
    if start_byte == 0:
        # self.receiving_ASCII = False
        # self.parseRaw(start_byte, unpac[1:])
        pass
    # 18-bit compression with Accelerometer
    elif 1 <= start_byte <= 100:
        # self.receiving_ASCII = False
        # self.parse18bit(start_byte, unpac[1:])
        pass
    # 19-bit compression without Accelerometer
    elif 101 <= start_byte <= 200:
        # self.receiving_ASCII = False
        # self.parse19bit(start_byte - 100, unpac[1:])
        pass
    # Impedance Channel
    elif 201 <= start_byte <= 205:
        # self.receiving_ASCII = False
        # self.parseImpedance(start_byte, packet[1:])
        pass
    # Part of ASCII -- TODO: better formatting of incoming ASCII
    elif start_byte == 206:
        # print("%\t" + str(packet[1:]))
        # self.receiving_ASCII = True
        # self.time_last_ASCII = timeit.default_timer()
        pass
        # End of ASCII message
    elif start_byte == 207:
        # print("%\t" + str(packet[1:]))
        # print("$$$")
        # self.receiving_ASCII = False
        pass
    else:
        print("Warning: unknown type of packet: " + str(start_byte))
    return start_byte


def receive_handler_cb(handle, value):
    """
    Indication and notification come asynchronously, we use this function to
    handle them either one at the time as they come.

    :param handle:
    :param value:
    :return:
    """
    print('Receive handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    parsed_packet = parse_packet(value)
    print(parsed_packet)
    return


def send_handler_cb(handle, value):
    """
    Indication and notification come asynchronously, we use this function to
    handle them either one at the time as they come.

    :param handle:
    :param value:
    :return:
    """
    print('Send handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def disconnect_handler_cb(handle, value):
    """
    Indication and notification come asynchronously, we use this function to
    handle them either one at the time as they come.

    :param handle:
    :param value:
    :return:
    """
    print('Disconnect handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def state_handler_cb(handle, value):
    print('State handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def board_handler_cb(handle, value):
    print('Board handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def site_handler_cb(handle, value):
    print('Site handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def version_handler_cb(handle, value):
    print('Version handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def unknown_handler_cb(handle, value):
    print('Unknown handler callback:')
    print('\tData: {}'.format(value.hex()))
    print('\tHandle: {}'.format(handle))
    return


def find_com_ports():
    """
    scan for available ports. return a list of tuples (num, name)
    """
    port_list = comports()
    return port_list


def discover_devs(port_list):
    dev_dict = {}
    for each_port in port_list:
        port_str = each_port.device
        adapter = pygatt.BGAPIBackend(serial_port=port_str)

        try:
            adapter.start()
            found_devs = adapter.filtered_scan('Ganglion')
            if found_devs:
                dev_dict[port_str] = found_devs
        finally:
            adapter.stop()
    return dev_dict


def set_subscribe_cbs(device):
    device.subscribe(BLE_UUID_DICT['CHAR_RECEIVE'], callback=receive_handler_cb, indication=True)
    device.subscribe(BLE_UUID_DICT['CHAR_SEND'], callback=send_handler_cb, indication=True)
    device.subscribe(BLE_UUID_DICT['CHAR_DISCONNECT'], callback=disconnect_handler_cb, indication=True)
    device.subscribe(BLE_UUID_DICT['BOARD'], callback=board_handler_cb, indication=True)
    device.subscribe(BLE_UUID_DICT['VERSION'], callback=version_handler_cb, indication=True)
    device.subscribe(BLE_UUID_DICT['SITE'], callback=site_handler_cb, indication=True)

    for each_unknown_uuid in UNKNOWN_UUIDS:
        device.subscribe(each_unknown_uuid, callback=unknown_handler_cb, indication=True)
    return


def connect(port, device_info, addr_type=pygatt.BLEAddressType.random):
    """
    The BGAPI backend will attempt to auto-discover the serial device name of the
    attached BGAPI-compatible USB adapter.

    :param port:
    :param device_info:
    :param addr_type:
    :return:
    """
    adapter = pygatt.BGAPIBackend(serial_port=port)
    try:
        adapter.start()
        device = adapter.connect(device_info['address'], address_type=addr_type)
        device.bond()
        set_subscribe_cbs(device)
        time.sleep(2)

        b_arr = bytes(COMMAND_START_BINARY, encoding='utf-8')
        device.char_write(BLE_UUID_DICT['CHAR_SEND'], b_arr, wait_for_response=True)
        count = 5
        for each_count in range(0, count):
            value = device.char_read_long(BLE_UUID_DICT['CHAR_RECEIVE'])
        time.sleep(2)

        b_arr = bytes(COMMAND_STOP_BINARY, encoding='utf-8')
        device.char_write(BLE_UUID_DICT['CHAR_SEND'], b_arr, wait_for_response=True)

        # count = 5
        # for each_count in range(0, count):
        #     value = device.char_read_long(BLE_CHAR_RECEIVE)
        #     print(len(value), binascii.hexlify(value))
        # print('-' * utilities.TERMINAL_COLUMNS)
        #
        # b_arr = bytes(COMMAND_START_BINARY, encoding='utf-8')
        # device.char_write(BLE_UUID_DICT['CHAR_SEND'], b_arr, wait_for_response=True)
        # for each_count in range(0, count):
        #     value = device.char_read_long(BLE_CHAR_RECEIVE)
        #     print(len(value), binascii.hexlify(value))
        # print('-' * utilities.TERMINAL_COLUMNS)
        #
        # b_arr = bytes(COMMAND_STOP_BINARY, encoding='utf-8')
        # device.char_write(BLE_UUID_DICT['CHAR_SEND'], b_arr, wait_for_response=True)
        # for each_count in range(0, count):
        #     value = device.char_read_long(BLE_CHAR_RECEIVE)
        #     print(len(value), binascii.hexlify(value))
        # print('-' * utilities.TERMINAL_COLUMNS)
        #
        # for each_uuid in device.discover_characteristics().keys():
        #     try:
        #         print("Read UUID %s: %s" % (each_uuid, binascii.hexlify(device.char_read(each_uuid))))
        #         for each_count in range(0, 5):
        #             value = device.char_read(each_uuid)
        #             print(value)
        #         print('-' * utilities.TERMINAL_COLUMNS)
        #     except BGAPIError as be:
        #         print(str(be))
        #         print('-' * utilities.TERMINAL_COLUMNS)
    finally:
        adapter.stop()
    return


def disconnect():
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

    com_port_list = find_com_ports()
    dev_dict = discover_devs(com_port_list)
    print('%d port(s) found with connected devices' % len(dev_dict))
    if len(dev_dict) > 0:
        for each_com, each_dev_list in dev_dict.items():
            print(each_com)
            print('\n'.join(['{}: {}'.format(each_dev['name'], each_dev['address']) for each_dev in each_dev_list]))

        try:
            port = list(dev_dict.keys())[-1]
            device_info = dev_dict[port][-1]
            connect(port, device_info)
        except Exception as e:
            print(str(e))
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
