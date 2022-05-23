import logging
import time
import traceback
from collections import deque

try:
    from faster_fifo import Queue
except Exception as e:
    print("consider using fifo-fast-queues, only available on linux/macos")
    from multiprocessing import Queue

from contextlib import contextmanager


class BidirectionalQueue:
    """
    Helper class to facilitate informational transfer between threads
    TODO: Consider using pipes or fifo fast queues
    """

    def __init__(self):
        self.request_queue = Queue()
        self.answer_queue = Queue()

    def request(self, obj):
        """
        Assumes that something is on the other end of the queue, obviously.
        """
        try:
            assert self.request_queue.empty()
            self.request_queue.put(obj)
            return self.answer_queue.get(block=True, timeout=120)
        except AssertionError:
            logging.error(" Request queue is not empty")
            self.restart()

    def restart(self):
        logging.warning("something went wrong with queue threads, restartign queue")
        time.sleep(5)
        try:
            self.request_queue.get(timeout=20)
        except Exception:
            logging.info(traceback.format_exc())
        try:
            self.answer_queue.get(timeout=20)
        except Exception:
            logging.info(traceback.format_exc())
        # self.__init__()
        # If something goes wrong, restart


class ThreadedBidirectionalQueue:
    def __init__(self, threads=5):
        self.threads = threads
        self.bidirectional_queues = [BidirectionalQueue() for _ in range(threads)]
        self.available_queues = deque(range(threads), threads)

    @contextmanager
    def get_queue(self):
        thread = None
        try:
            thread = self.available_queues.pop()
            yield self.bidirectional_queues[thread]
        except Exception:
            logging.info(traceback.format_exc())
            if thread:
                logging.warning("something went wrong with queue threads")
                self.bidirectional_queues[thread].restart()
        finally:
            # Code to release resource, e.g.:
            if thread is not None:
                self.available_queues.appendleft(thread)
            else:
                logging.warning("Could not find available queue to pop")

    def request(self, obj):
        try:
            with self.get_queue() as queue:
                return queue.request(obj)

        except Exception:
            logging.info(traceback.format_exc())


class QueueContainer:
    """
    Container class for queues
    """

    def __init__(self, threading=1):
        if threading == 1:
            self.policy_queues = BidirectionalQueue()
            self.evaluation_policy_queues = BidirectionalQueue()
            self.threaded = False
        else:
            self.policy_queues = ThreadedBidirectionalQueue(threading)
            self.evaluation_policy_queues = ThreadedBidirectionalQueue(threading)
            self.threaded = True
