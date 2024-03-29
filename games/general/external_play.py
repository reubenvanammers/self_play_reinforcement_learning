import time

import colorama
from colorama import Fore, Style

from games.general.base_env import BaseEnv
from games.general.base_model import BasePlayer


def swap_state(s):
    # Make state as opposing policy will see it
    return s * -1


class AbstractExternalPlay:
    def play(self):
        raise NotImplementedError

    def play_round(self, s):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            return s_next, done, r

    def get_and_play_moves(self, s, player=1):
        raise NotImplementedError


class ManualPlay(AbstractExternalPlay):
    def __init__(self, env: BaseEnv, opponent: BasePlayer):
        self.env = env
        self.opposing_policy = opponent

    def play(self, swap_sides=False):
        s = self.env.reset()
        self.opposing_policy.reset(player=(1 if swap_sides else -1))
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)

        for i in range(self.env.max_moves()):
            s, done, r = self.play_round(s)
            if done:
                print(f"reward was {r}")
                self.env.render()
                play_again = input("Play again (y/n)").lower() == "y"
                if play_again:
                    self.play(not swap_sides)
                else:
                    break

    def get_and_play_moves(self, s, player=1):
        self.env.render()
        if player == 1:
            try:
                a = self.env.get_manual_move()
            except:
                a = int(input("Choose your move (X)"))
            s_next, r, done, info = self.play_move(a, player=1)
            return s_next, a, r, done, info
        else:
            opp_s = swap_state(s)
            a = self.opposing_policy(opp_s)
            print(f"Opponent chose move {a}")
            s_next, r, done, info = self.play_move(a, player=-1)
            r = r * player
            return s_next, a, r, done, info

    def play_move(self, a, player=1):
        self.opposing_policy.play_action(a, player * -1)  # Opposing policy will be player 1, in their perspective
        return self.env.step(a, player=player)


class View(AbstractExternalPlay):
    SLEEP_TIME = 3  # In seconds

    def __init__(self, env, player1: BasePlayer, player2: BasePlayer):
        colorama.init()
        self.env = env
        self.player1 = player1
        self.player2 = player2
        self.timer = time.time()

    def play(self, swap_sides=False):
        s = self.env.reset()
        self.player1.reset(player=(-1 if swap_sides else 1))
        self.player2.reset(player=(1 if swap_sides else -1))
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)

        for i in range(self.env.max_moves()):
            s, done, r = self.play_round(s)
            if done:
                print(f"Winner was {'player1' if r ==1 else 'player2'}")
                self.env.render()
                time.sleep(3000)

    def _sleep_delay(self):
        time_diff = (self.timer + self.SLEEP_TIME) - time.time()
        if time_diff > 0:
            time.sleep(time_diff)
        self.timer = time.time()

    def get_and_play_moves(self, s, player=1):
        self.env.render()
        self._sleep_delay()

        if player == 1:
            a = self.player1(s)
            print(f"{Fore.GREEN}Player 1{Style.RESET_ALL} chose move {a}")
            s_next, r, done, info = self.play_move(a, player=1)
            return s_next, a, r, done, info
        else:
            opp_s = swap_state(s)
            a = self.player2(opp_s)
            print(f"{Fore.RED}Player 2{Style.RESET_ALL} chose move {a}")
            s_next, r, done, info = self.play_move(a, player=-1)
            r = r * player
            return s_next, a, r, done, info

    def play_move(self, a, player=1):
        self.player1.play_action(a, player)
        self.player2.play_action(a, player * -1)  # Opposing policy will be player 1, in their perspective
        return self.env.step(a, player=player)
