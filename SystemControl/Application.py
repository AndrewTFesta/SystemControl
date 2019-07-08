"""

"""
import argparse
import inspect
import time

from SystemControl import utilities
from SystemControl.HubCommunicator import HubCommunicator


class Application:

    def __init__(self, hub):
        option_funcs = [
            obj
            for name, obj in inspect.getmembers(Application)
            if inspect.isfunction(obj) and name.startswith('select_')
        ]
        self.menu_options = {
            str(each_func_idx): each_func
            for each_func_idx, each_func in enumerate(option_funcs)
        }
        self.hub = hub
        self.exit = False
        return

    def display_menu(self):
        for opt_key, opt_func in self.menu_options.items():
            opt_name = opt_func.__name__.split('select_')[-1]
            print('%s: %s' % (opt_key, opt_name))
        return

    def main_loop(self):
        self.display_menu()
        while not self.exit:
            user_in = input('Enter an option listed above: ')
            if user_in in self.menu_options:
                selected_func = self.menu_options[user_in]
                selected_func(self)
                self.display_menu()
            else:
                print('Input must be in the range [0, %d)' % len(self.menu_options))

        print('Cleaning up all resources...')
        self.hub.clean_up()
        return

    def select_exit(self):
        print('exit')
        self.exit = True
        return

    def select_status(self):
        print('status')
        self.hub.get_status()
        return

    def select_connect(self):
        print('connect')
        return

    def select_disconnect(self):
        print('disconnect')
        return

    def select_scan(self):
        print('scan')
        return

    def select_boardType(self):
        print('board_type')
        return

    def select_protocol(self):
        print('prot')
        return

    def select_impedance(self):
        print('imp')
        return

    def select_accelerometer(self):
        print('accel')
        return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        proj_name, proj_version = utilities.get_version()
        print('%s: VERSION: %s' % (proj_name, proj_version))
        return

    hub_inst = HubCommunicator()
    print('Hub is running: %s' % hub_inst.hub_thread)
    time.sleep(1)

    app = Application(hub_inst)
    app.main_loop()
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

    print('-' * utilities.TERMINAL_COLUMNS)
    print('Exiting main')
    print('-' * utilities.TERMINAL_COLUMNS)
