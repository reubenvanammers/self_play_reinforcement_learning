import time


class ManualPlay:
    def __init__(self, env, opponent):
        self.env = env
        self.opposing_policy = opponent

    def play(self, swap_sides=False):
        s = self.env.reset()
        self.opposing_policy.reset(player=(1 if swap_sides else -1))
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)

        for i in range(100):  # Should be less than this
            s, done, r = self.play_round(s)
            if done:
                print(f"reward was {r}")
                self.env.render()
                play_again = input("Play again (y/n)").lower() == "y"
                if play_again:
                    self.play(not swap_sides)
                else:
                    break

    def play_round(self, s):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            return s_next, done, r

    def get_and_play_moves(self, s, player=1):
        self.env.render()
        if player == 1:
            a = int(input("Choose your move (X)"))
            s_next, r, done, info = self.play_move(a, player=1)
            return s_next, a, r, done, info
        else:
            opp_s = self.swap_state(s)
            a = self.opposing_policy(opp_s)
            print(f"Opponent chose move {a}")
            s_next, r, done, info = self.play_move(a, player=-1)
            r = r * player
            return s_next, a, r, done, info

    def swap_state(self, s):
        # Make state as opposing policy will see it
        return s * -1

    def play_move(self, a, player=1):
        # self.policy.play_action(a, player)
        self.opposing_policy.play_action(a, player * -1)  # Opposing policy will be player 1, in their perspective
        return self.env.step(a, player=player)


class View:
    def __init__(self, env, player1, player2):
        self.env = env
        self.player1 = player1
        self.player2 = player2

    def play(self, swap_sides=False):
        s = self.env.reset()
        self.player1.reset(player=(-1 if swap_sides else 1))
        self.player2.reset(player=(1 if swap_sides else -1))
        if swap_sides:
            s, _, _, _, _ = self.get_and_play_moves(s, player=-1)

        for i in range(100):  # Should be less than this
            s, done, r = self.play_round(s)
            if done:
                print(f"Winner was {'player1' if r ==1 else 'player2'}")
                self.env.render()
                time.sleep(3000)
                # play_again = input("Play again (y/n)").lower() == "y"
                # if play_again:
                #     self.play(not swap_sides)
                # else:
                #     break

    def play_round(self, s):
        s = s.copy()
        s_intermediate, own_a, r, done, info = self.get_and_play_moves(s)
        if done:
            return s_intermediate, done, r
        else:
            s_next, a, r, done, info = self.get_and_play_moves(s_intermediate, player=-1)
            return s_next, done, r

    def get_and_play_moves(self, s, player=1):
        self.env.render()
        time.sleep(3)

        if player == 1:
            # a = int(input("Choose your move (X)"))
            a = self.player1(s)
            print(f"Player 1 chose move {a}")
            s_next, r, done, info = self.play_move(a, player=1)
            return s_next, a, r, done, info
        else:
            opp_s = self.swap_state(s)
            a = self.player2(opp_s)
            print(f"Player 2 chose move {a}")
            s_next, r, done, info = self.play_move(a, player=-1)
            r = r * player
            return s_next, a, r, done, info

    def swap_state(self, s):
        # Make state as opposing policy will see it
        return s * -1

    def play_move(self, a, player=1):
        self.player1.play_action(a, player)
        self.player2.play_action(a, player * -1)  # Opposing policy will be player 1, in their perspective
        return self.env.step(a, player=player)
