import logging
import re
import threading
import traceback
from collections import deque
# from multiprocessing import Queue

try:
    from faster_fifo import Queue
except Exception as e:
    print("consider using fifo-fast-queues, only available on linux/macos")
    from multiprocessing import Queue


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
            return self.answer_queue.get(block=True)
        except AssertionError:
            print("asdf")


class ThreadedBidirectionalQueue:
    def __init__(self, threads=5):
        self.threads = threads
        self.bidirectional_queues = [BidirectionalQueue() for _ in range(threads)]
        self.available_queues = deque(range(threads), threads)

    # def request(self, obj):
    #     try:
    #         # Finds queue from threaded evaluator worker - bit dodgy, may want to rework
    #         # Eg subclass threadedpoolexecutor
    #         name = threading.current_thread().name
    #         capture = re.findall(r"(\d+)", name)
    #         if not capture:
    #             thread = 0
    #         else:
    #             thread = int(capture[1])
    #         return self.bidirectional_queues[thread].request(obj)
    #     except Exception:
    #         logging.info(traceback.format_exc())

    def request(self, obj):
        try:
            thread = self.available_queues.pop()
            result = self.bidirectional_queues[thread].request(obj)
            self.available_queues.appendleft(thread)
            return result
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
