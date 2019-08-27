"""

"""
import argparse
import inspect

import SystemControl
from SystemControl import utilities
from SystemControl.HubCommunicator import HubCommunicator


class CliApp:

    def __init__(self, hub):
        option_funcs = [
            {
                'function': obj,
                'priority': member_name.split('_')[1],
                'name': ' '.join(member_name.split('_')[2:])
            }
            for member_name, obj in inspect.getmembers(CliApp)
            if inspect.isfunction(obj) and member_name.startswith('select_')
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
        print('-' * SystemControl.utilities.TERMINAL_COLUMNS)
        for each_option in sorted(self.menu_options, key=lambda x: x['priority']):
            opt_name = each_option['name']
            opt_idx = each_option['index']
            print('%s: %s' % (opt_idx, opt_name))
        print('-' * SystemControl.utilities.TERMINAL_COLUMNS)
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

    def select_1_scan(self):
        self.hub.start_scan()
        return

    def select_2_connect(self):
        self.hub.connect_board()
        return

    def select_2_disconnect(self):
        self.hub.disconnect_board()
        return

    def select_3_enable_channels(self):
        self.hub.turn_on_channels(['!', '@', '#', '$'])
        return

    def select_3_disable_channels(self):
        self.hub.turn_off_channels(['1', '2', '3', '4'])
        return

    def select_4_enable_accelerometer(self):
        self.hub.enable_accel()
        return

    def select_4_disable_accelerometer(self):
        self.hub.disable_accel()
        return

    def select_5_start_impedance_test(self):
        self.hub.start_impedance()
        return

    def select_5_stop_impedance_test(self):
        self.hub.stop_impedance()
        return

    def select_6_start_stream(self):
        self.hub.start_stream()
        return

    def select_6_stop_stream(self):
        self.hub.stop_stream()
        return

    def select_7_log_registers(self):
        self.hub.log_registers()
        return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print('%s: VERSION: %s' % (SystemControl.NAME, SystemControl.VERSION))
        return

    hub_inst = HubCommunicator()
    app = CliApp(hub_inst)
    app.main_loop()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', '-v', action='store_true',
                        help='prints the current version and exits')
    print('-' * SystemControl.utilities.TERMINAL_COLUMNS)
    print(parser.prog)
    print('-' * SystemControl.utilities.TERMINAL_COLUMNS)

    cargs = parser.parse_args()
    main(cargs)
