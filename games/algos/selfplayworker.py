import copy
import logging

from games.algos.base_worker import BaseWorker
import traceback

class SelfPlayWorker(BaseWorker):
    def __init__(
        self,
        task_queue,
        memory_queue,
        result_queue,
        env_gen,
        policy_container,
        opposing_policy_container,
        start_time,
        evaluator=None,
        save_dir="save_dir",
        resume=False,
        self_play=False,
        evaluation_policy_container=None,
        model_save_location=None,
    ):
        logging.info("initializing self play worker worker")
        self.env_gen = env_gen
        self.env = env_gen()
        self.evaluator = evaluator

        self.policy_container = policy_container
        self.opposing_policy_container = opposing_policy_container
        self.evaluation_policy_container = evaluation_policy_container

        self.policy = None
        self. opposing_policy_train = None
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
        self.epoch_value=model_save_location

        self.epoch_count=0

        super().__init__()

    def set_up_policies(self):
        try:
            if self.self_play:
                self.policy = self.policy_container.setup(memory_queue=self.memory_queue, evaluator=self.evaluator)
                self.opposing_policy_train = self.opposing_policy_container.setup(evaluator=self.evaluator)
            # else:
            #     self.policy = self.policy_container.setup(memory_queue=self.memory_queue)
            #     self.opposing_policy_train = self.opposing_policy_container.setup()

            self.opposing_policy_train.train(False)
            self.policy.train(False)

            self.opposing_policy_train.env = self.env_gen()

            if self.evaluation_policy_container:
                #TODO add evaluation policy
                self.opposing_policy_evaluate = self.evaluation_policy_container.setup()

            if self.opposing_policy_evaluate:
                self.opposing_policy_evaluate.evaluate(True)
                self.opposing_policy_evaluate.train(False)
                self.opposing_policy_evaluate.env = self.env_gen()

            self.opposing_policy = self.opposing_policy_train
            logging.info("finished setting up policies")
        except Exception as e:
            logging.exception("setting up policies failed" + str(e))
            logging.exception(traceback.format_exc())

    def run(self):
        logging.info("running self play worker")

        self.set_up_policies()  # dont do in init because of global apex
        if self.resume:
            self.load_model(prev_run=True)

        while True:
            task = self.task_queue.get()
            logging.info(f"task {task}")
            try:
                if self.epoch_value:
                    if  self.epoch_value.value:
                        if self.epoch_value.value != self.epoch_count:
                # if task.get("saved_name") and task.get("saved_name") != self.current_model_file:
                    # time.sleep(5)
                            logging.info("loading model")
                            self.load_model()
                            self.epoch_count = self.epoch_value.value

                evaluate = task.get("evaluate")
                if evaluate:
                    self.opposing_policy = self.opposing_policy_evaluate
                    self.policy.train(False)
                    self.policy.evaluate(True)
                else:
                    self.opposing_policy = self.opposing_policy_train
                    self.policy.train(False)
                    self.policy.evaluate(False)

                episode_args = task["play"]
                self.play_episode(**episode_args)
                self.task_queue.task_done()
                logging.info("task done")
            except Exception as e:
                logging.exception(traceback.format_exc())
                self.task_queue.task_done()

    def play_episode(self, swap_sides=False, update=True):

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
        return state_list, r

    def play_round(self, s, update=True):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            if update:
                self.policy.push_to_queue(s, own_a, r, done, s_intermediate)
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            if update:
                self.policy.push_to_queue(s, own_a, r, done, s_next)
                # self.policy.update(s, own_a, r, done, s_next)
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