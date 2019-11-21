"""
@title
@description
"""
from utils.Observer import Observer


class Observable:

    def __init__(self):
        self.subscriber_list = []
        self._changed = False
        self.change_message = None
        return

    def notify_all(self):
        if self._changed:
            for subscriber in self.subscriber_list:
                subscriber.update(self, self.change_message)
        self._changed = False
        return

    def set_changed_message(self, message):
        self.change_message = message
        self.set_changed()
        return

    def set_changed(self):
        self._changed = True
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
