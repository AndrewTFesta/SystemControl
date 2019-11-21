"""
@title
@description
"""
from utils.Observable import Observable


class Observer:

    def __init__(self, sub_list: list):
        self.subscriptions = []
        for each_sub in sub_list:
            if isinstance(each_sub, Observable):
                self.subscriptions.append(each_sub)
                each_sub.subscribe(self)
        return

    def update(self, source, update_message):
        raise NotImplementedError(f'Not implemented: {self}')
