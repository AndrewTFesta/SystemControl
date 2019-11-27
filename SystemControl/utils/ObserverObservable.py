"""
@title
@description
"""


class Observable:

    def __init__(self):
        self.subscriber_list = []
        self.change_message = None
        return

    def notify_all(self):
        for subscriber in self.subscriber_list:
            subscriber.update(self, self.change_message)
        self.change_message = None
        return

    def set_changed_message(self, message):
        self.change_message = message
        self.notify_all()
        return

    def subscribe(self, subscriber):
        if isinstance(subscriber, Observer):
            self.subscriber_list.append(subscriber)
        else:
            print(f'Subscriber is not a valid Observer: {subscriber}')
        return

    def unsubscribe(self, subscriber):
        if self.subscriber_list.__contains__(subscriber):
            self.subscriber_list.remove(subscriber)
        else:
            print(f'Subscriber is not currently subscribed to this Observable: {subscriber}')
        return


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
