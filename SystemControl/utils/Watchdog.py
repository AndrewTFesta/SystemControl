"""
@title
@description
"""
import os
import threading
from time import sleep

import psutil


class Watchdog:

    def __init__(self, timeout):
        self.timeout = timeout

        self.watching = None
        self.watchdog_thread = None
        self.timer_thread = None
        self._running = False
        return

    def run(self):
        self.watchdog_thread = threading.Thread(target=self.__start_watchdog_timer, args=(), daemon=True)
        self.watchdog_thread.start()
        self._running = True
        return

    def ping(self):
        if self._running:
            self.timer_thread.cancel()
            print('ping received')
        return

    @staticmethod
    def get_times():
        curr_gp_times = os.times()
        return curr_gp_times

    def register(self, to_watch):
        self.watching = to_watch
        return

    def __start_watchdog_timer(self):
        while self._running:
            self.timer_thread = threading.Timer(self.timeout, self.kill_watch)
            self.timer_thread.start()
            self.timer_thread.join()
        print('stopping timer')
        return

    def kill_watch(self):
        print(f'timeout exceeded: {self.timeout}\n\tkilling watchdog')
        self._running = False
        return


class FilesystemWatchdog(Watchdog):

    def __init__(self, timeout):
        Watchdog.__init__(self, timeout)
        return


class ProcessWatchdog(Watchdog):

    def __init__(self, timeout):
        Watchdog.__init__(self, timeout)
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

    def __init__(self, timeout):
        Watchdog.__init__(self, timeout)
        return


def main():
    timeout = 2

    watchdog = Watchdog(timeout)
    watchdog.run()
    sleep(1)
    watchdog.ping()
    sleep(1)
    watchdog.ping()
    sleep(1)
    watchdog.ping()
    sleep(5)
    return


if __name__ == '__main__':
    main()
