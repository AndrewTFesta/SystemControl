"""

"""
import argparse
import socket
import threading

import SystemControl

from SystemControl import utilities

PORT_LIST = {
    12345,
    12346,
    12347
}


def listen_port(port_num):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', port_num)
    sock.bind(server_address)

    while True:
        data, address = sock.recvfrom(4096*2)
        print('%s' % data)


def start_listeners():
    for each_port in PORT_LIST:
        fft_listen_thread = threading.Thread(target=listen_port, args=[each_port])
        fft_listen_thread.daemon = True
        fft_listen_thread.start()
    return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print('%s: VERSION: %s' % (SystemControl.name, SystemControl.version))
        return

    start_listeners()
    input('Press enter to exit...')
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
