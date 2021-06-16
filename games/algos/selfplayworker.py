import copy
import logging
import time
import traceback
from concurrent import futures

from games.algos.base_worker import BaseWorker
from games.algos.evaluation_worker import PerfectEvaluator


class SelfPlayWorker(BaseWorker):
    def __init__(
        self,
        task_queue,
        memory_queue,
        result_queue,
        env_gen,
        policy_container,
        start_time,
        network=None,
        evaluation_network=None,
        save_dir="save_dir",
        resume=False,
        self_play=False,
        evaluation_policy_container=None,
        epoch_value=None,
        threading=0,
    ):
        logging.info("initializing self play worker worker")
        self.env_gen = env_gen
        self.env = env_gen()
        self.network = network
        self.evaluation_network = evaluation_network

        self.policy_container = policy_container
        self.evaluation_policy_container = evaluation_policy_container

        self.policy = None
        self.opposing_policy_train = None
        self.opposing_policy = None
        self.opposing_policy_evaluate = None

        self.task_queue = task_queue
        self.memory_queue = memory_queue
        self.result_queue = result_queue
        self.save_dir = save_dir
        self.start_time = start_time
        self.self_play = self_play

        self.current_model_file = None
        self.resume = resume
        self.epoch_value = epoch_value

        self.epoch_count = 0

        self.threading = threading

        super().__init__()

    def set_up_policies(self, evaluate=False):
        try:
            if self.self_play:
                policy = self.policy_container.setup(memory_queue=self.memory_queue, network=self.network)
                policy.train(False)
                if evaluate:
                    if self.evaluation_network:
                        opposing_policy = self.evaluation_policy_container.setup(network=self.evaluation_network)
                    else:
                        opposing_policy = self.evaluation_policy_container.setup()

                    opposing_policy.train(False)
                    opposing_policy.env = self.env_gen()
                    # Both environments will (basically) try their hardest
                    policy.evaluate(True)
                    opposing_policy.evaluate(True)
                else:
                    opposing_policy = self.policy_container.setup(memory_queue=self.memory_queue, network=self.network)
                    opposing_policy.env = self.env_gen()
                    opposing_policy.train(False)
                    # Both environments are more willing to explore
                    policy.evaluate(False)
                    opposing_policy.evaluate(False)
            logging.debug("Created SelfPlayWorker")

            return SelfPlayer(policy, opposing_policy, self.env_gen(), self.result_queue)
        except Exception as e:
            logging.exception("setting up policies failed" + str(e))
            logging.exception(traceback.format_exc())

    def run(self):
        logging.info("running self play worker")

        if self.resume:
            self.load_model(prev_run=True)

        if self.threading:
            executor = futures.ThreadPoolExecutor(self.threading)
            future_set = set()

        while True:
            task = self.task_queue.get()
            logging.debug(f"task {task}")
            try:
                if self.epoch_value:
                    if self.epoch_value.value:
                        if self.epoch_value.value != self.epoch_count:
                            logging.info("loading model")
                            self.load_model()
                            self.epoch_count = self.epoch_value.value
                if task.get("reference"):
                    print("checking reference")
                    self.check_reference()
                    continue
                evaluate = task.get("evaluate")
                self_player = self.set_up_policies(evaluate=evaluate)

                episode_args = task["play"]
                if not self.threading:
                    self_player.play_episode(**episode_args)
                    self.task_queue.task_done()
                    logging.info("task done")
                else:
                    #small delay to allow threads to be more equal across core
                    time.sleep(0.1)
                    future = executor.submit(self_player.play_episode, **episode_args)
                    future.add_done_callback(self._task_finished)
                    future_set.add(future)
                    if len(future_set) < self.threading:
                        continue
                    else:
                        done, future_set = futures.wait(future_set, return_when=futures.FIRST_COMPLETED)

            except Exception as e:
                logging.exception(traceback.format_exc())
                self.task_queue.task_done()

    def check_reference(self):
        try:
            policy = self.policy_container.setup(network=self.network)
            policy.train(False)
            policy.evaluate(True)
            reference = PerfectEvaluator(policy)
            reference.test()
            reference.test(base_network=True)
            self.task_queue.task_done()
        except Exception:
            logging.exception(traceback.format_exc())
            self.task_queue.task_done()

    def _task_finished(self, future):
        if future.done():
            self.task_queue.task_done()
            logging.debug("task done")


class SelfPlayer:
    def __init__(self, policy, opposing_policy, env, result_queue, update_opponent=False):
        self.policy = policy
        self.opposing_policy = opposing_policy
        self.env = env
        self.result_queue = result_queue
        self.update_opponent = update_opponent

    def play_episode(self, swap_sides=False, update=True):
        try:

            s = self.env.reset()
            self.policy.reset(player=(-1 if swap_sides else 1))
            self.opposing_policy.reset(player=(1 if swap_sides else -1))  # opposite from opposing policy perspective
            state_list = []
            if swap_sides:
                s, _, _, _, _ = self.get_and_play_moves(s, player=-1)
            for i in range(100):  # Should be less than this
                s, done, r = self.play_round(s, update=update)
                state_list.append(copy.deepcopy(s))
                if done:
                    break
            self.result_queue.put({"reward": r, "swap_sides": swap_sides})
            if update:
                self.policy.push_to_queue(done=True, r=r)
                # Also push opposing policies perspective, but swap winner/loser
                if self.update_opponent:
                    self.opposing_policy.push_to_queue(done=True, r=r * -1)
            return state_list, r
        except Exception as e:
            logging.info("Error in Self play worker")
            logging.info(traceback.format_exc())

    def play_round(self, s, update=True):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            return s_next, done, r

    def swap_state(self, s):
        # Make state as opposing policy will see it
        return s * -1

    def get_and_play_moves(self, s, player=1):
        if player == 1:
            a = self.policy(s)
            s_next, r, done, info = self.play_move(a, player=1)
            return s_next, a, r, done, info
        else:
            opp_s = self.swap_state(s)
            a = self.opposing_policy(opp_s)
            s_next, r, done, info = self.play_move(a, player=-1)
            r = r * player
            return s_next, a, r, done, info

    def play_move(self, a, player=1):
        self.policy.play_action(a, player)
        self.opposing_policy.play_action(a, player * -1)  # Opposing policy will be player 1, in their perspective
        return self.env.step(a, player=player)
