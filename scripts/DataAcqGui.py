"""

"""
import argparse
import inspect
import tkinter as tk

import SystemControl
from SystemControl.OBciPython import MenuFunctions
from SystemControl.OBciPython.HubCommunicator import HubCommunicator
from SystemControl.SystemLog import SystemLogLevel, SystemLogIdent


class DataAcqGui(tk.Frame):

    def __init__(self, hub, master=None):
        super().__init__(master)

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

        self.master = master
        self.buttons = []

        self.pack()
        self.create_widgets()

        self.press_count = 0
        return

    @staticmethod
    def __filter_menu_function(function_object):
        return inspect.isfunction(function_object) and function_object.__name__.startswith('select_')

    def create_widgets(self):
        button_hi_there = tk.Button(self)
        button_hi_there['text'] = 'Hello World\n(click me)'
        button_hi_there['command'] = self.say_hi
        button_hi_there.pack(side='top')

        button_quit = tk.Button(self, text='QUIT', fg='red', command=self.master.destroy)
        button_quit.pack(side='bottom')

        self.buttons.append(button_hi_there)
        self.buttons.append(button_quit)
        return

    def say_hi(self):
        self.press_count += 1
        msg_str = 'hi there, everyone! {}'.format(self.press_count)
        self.hub.msg_log.log_message(
            message_ident=SystemLogIdent.LOG,
            message_string=msg_str,
            message_level=SystemLogLevel.NORMAL
        )
        return

    def cleanup(self):
        print('Cleaning up all resources...')
        self.hub.clean_up()
        return


def main(args):
    """
    :param args: arguments passed in to control flow of operation.
    This is generally expected to be passed in over the command line.
    :return: None
    """
    if args.version:
        print('{}: VERSION: {}'.format(SystemControl.name, SystemControl.version))
        return

    verbosity_level = SystemLogLevel.NORMAL

    hub_inst = HubCommunicator(verbosity_level)
    root = tk.Tk()

    gui_app = DataAcqGui(hub_inst, master=root)
    gui_app.mainloop()

    gui_app.cleanup()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--version', '-v', action='store_true',
                        help='prints the current version and exits')

    cargs = parser.parse_args()
    main(cargs)
