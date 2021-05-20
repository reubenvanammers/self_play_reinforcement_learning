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
        self.request_queue.put(obj)
        return self.answer_queue.get(block=True)




class QueueContainer:
    """
    Container class for queues
    """

    def __init__(self):

        self.policy_queues = BidirectionalQueue()
        self.opposing_policy_queues = BidirectionalQueue()
        self.evaluation_policy_queues = BidirectionalQueue()