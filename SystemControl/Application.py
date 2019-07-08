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
            {
                'function': obj,
                'priority': name.split('_')[1],
                'name': ' '.join(name.split('_')[2:])
            }
            for name, obj in inspect.getmembers(Application)
            if inspect.isfunction(obj) and name.startswith('select_')
        ]
        option_funcs.sort(key=lambda x: x['priority'])
        self.menu_options = [
            {
                'index': each_func_idx,
                'function': each_entry['function'],
                'priority': each_entry['priority'],
                'name': each_entry['name'],
            }
            for each_func_idx, each_entry in enumerate(option_funcs)
        ]
        self.hub = hub
        self.exit = False
        return

    def display_menu(self):
        print('-----------------')
        for each_key, each_val in self.hub.states.items():
            print('%s: %s' % (each_key, each_val))
        print('-----------------')
        for each_option in sorted(self.menu_options, key=lambda x: x['priority']):
            opt_name = each_option['name']
            opt_idx = each_option['index']
            print('%s: %s' % (opt_idx, opt_name))
        return

    def main_loop(self):
        self.display_menu()
        while not self.exit:
            user_in = input('Enter an option listed above:\n')
            for each_option in self.menu_options:
                if user_in == str(each_option['index']):
                    selected_option = each_option
                    selected_func = selected_option['function']
                    selected_func(self)
                    self.display_menu()
                    break
            else:
                print('Input must be in the range [0, %d)' % len(self.menu_options))

        print('Cleaning up all resources...')
        self.hub.clean_up()
        return

    def select_0_exit(self):
        self.exit = True
        return

    def select_1_start(self):
        if not self.hub.is_hub_running():
            self.hub.start_hub_thread()
            self.hub.connect_hub()
            print('Hub has been started')
        else:
            print('Hub is already running')
        return

    def select_1_kill(self):
        kill_success = self.hub.kill_hub()
        print('Hub has been killed: %s' % kill_success)
        return

    def select_2_status(self):
        self.hub.check_status()
        return

    def select_2_protocol(self):
        print()
        print('0: Start protocol: bled112')
        print('1: Stop current protocol: %s' % self.hub.states['protocol'])
        print('2: Get current protocol')
        user_in = input('Enter an option listed above:\n')
        if user_in == '0':
            self.hub.set_protocol()
        elif user_in == '1':
            self.hub.stop_protocol()
        elif user_in == '2':
            self.hub.get_protocol()
        else:
            print('Unrecognized input: %s' % user_in)
        return

    def select_3_scan(self):
        print()
        print('0: Start scan: bled112')
        print('1: Stop current scan: %s' % self.hub.states['scan'])
        print('2: Get current scan')
        user_in = input('Enter an option listed above:\n')
        if user_in == '0':
            self.hub.start_scan()
        elif user_in == '1':
            self.hub.stop_scan()
        elif user_in == '2':
            self.hub.get_scan()
        else:
            print('Unrecognized input: %s' % user_in)
        return

    def select_4_connect(self):
        print('connect')
        self.hub.connect_board()
        return

    def select_4_disconnect(self):
        print('disconnect')
        self.hub.connect_board()
        return

    def select_5_impedance(self):
        print('imp')
        return

    def select_5_accelerometer(self):
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
