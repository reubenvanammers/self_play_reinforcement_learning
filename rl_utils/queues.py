from multiprocessing import Queue
import threading
import re

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
        self.request_queue.put(obj)
        return self.answer_queue.get(block=True)


class ThreadedBidirectionalQueue:

    def __init__(self, threads = 5):

        self.bidirectional_queues = [BidirectionalQueue() for _ in range(threads)]


    def request(self, obj):
        # Finds queue from threaded evaluator worker - bit dodgy, may want to rework
        name = threading.current_thread().name
        thread = int(re.match(r"ThreadPoolExecutor-([1-9]*)\w+", name).group(0))
        self.bidirectional_queues[thread].forward(obj)






class QueueContainer:
    """
    Container class for queues
    """

    def __init__(self):

        self.policy_queues = BidirectionalQueue()
        self.opposing_policy_queues = BidirectionalQueue()
        self.evaluation_policy_queues = BidirectionalQueue()