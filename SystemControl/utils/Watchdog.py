"""
@title
@description
"""
import os
import time

import psutil


class Watchdog:

    def __init__(self):
        self.reg_list = []

        self.timeout = -1
        self.update_time = -1
        return

    def _ping(self):
        self.update_time = time.time()
        return

    @staticmethod
    def get_times():
        curr_gp_times = os.times()
        return curr_gp_times

    def register(self, to_watch):
        self.reg_list.append(to_watch)
        return


class FilesystemWatchdog(Watchdog):

    def __init__(self):
        Watchdog.__init__(self)
        return


class ProcessWatchdog(Watchdog):

    def __init__(self):
        Watchdog.__init__(self)
        return

    @staticmethod
    def get_pids():
        for proc in psutil.process_iter():
            try:
                pname = proc.name()
                pid = proc.pid
                print(f'{pname} :: {pid}')
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as pe:
                print(f'{pe}')
        return


class ThreadWatchdog(Watchdog):

    def __init__(self):
        Watchdog.__init__(self)
        return
