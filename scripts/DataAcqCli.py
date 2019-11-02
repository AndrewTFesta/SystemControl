"""

"""
import argparse
import inspect
import threading
from time import sleep

import SystemControl
from SystemControl.OBciPython.EHub import MenuFunctions
from SystemControl.OBciPython.EHub.HubCommunicator import HubCommunicator
from SystemControl.SystemLog import SystemLogLevel


class DataAcqCli:

    def __init__(self, hub):
        menu_options = inspect.getmembers(MenuFunctions, predicate=self.__filter_menu_function)
        option_funcs = [
            {
                'function': obj,
                'priority': member_name.split('_')[1],
                'name': ' '.join(member_name.split('_')[2:])
            }
            for member_name, obj in menu_options
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

    @staticmethod
    def __filter_menu_function(function_object):
        return inspect.isfunction(function_object) and function_object.__name__.startswith('select_')

    def display_menu(self):
        disp_string = '-' * SystemControl.TERMINAL_COLUMNS + '\n'
        disp_string += self.hub.state_string + '\n'
        disp_string += '-' * SystemControl.TERMINAL_COLUMNS + '\n'
        for each_option in sorted(self.menu_options, key=lambda x: x['priority']):
            opt_name = each_option['name']
            opt_idx = each_option['index']
            disp_string += '{}: {}'.format(opt_idx, opt_name) + '\n'
        disp_string += '-' * SystemControl.TERMINAL_COLUMNS
        print(disp_string)
        return

    def main_loop(self):
        menu_loop = threading.Thread(target=self.menu_loop)
        menu_loop.start()
        menu_loop.join()

        print('Cleaning up all resources...')
        self.hub.clean_up()
        return

    def menu_loop(self):
        self.display_menu()
        while not self.exit:
            user_in = input('Enter an option listed above:\n')
            for each_option in self.menu_options:
                if user_in == str(each_option['index']):
                    selected_option = each_option
                    selected_func = selected_option['function']
                    selected_func(self)
                    sleep(1)
                    self.display_menu()
                    break
            else:
                if not user_in:
                    self.exit = True
                else:
                    print('Input must be in the range [0, {}): "{}"'.format(len(self.menu_options), user_in))
        return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print(f'{SystemControl.name}: VERSION: {SystemControl.version}')
        return

    hub_inst = HubCommunicator(SystemLogLevel.NORMAL)
    app = DataAcqCli(hub_inst)
    app.main_loop()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', '-v', action='store_true',
                        help='prints the current version and exits')

    cargs = parser.parse_args()
    main(cargs)
