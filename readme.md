# Overview

## Self Play Reinforcement Learning


This repo is contains a simple implementation of AlphaZero in pytorch, roughly emulation the structure in 
the alphazero/alphazero papers. The main game that this was written for was Connect4, although networks and environments
for TicTacToe were also written. 

 The code was written to be modular, so that additional games could be easily be played 
via the addition of appropriate environments and torch code to convert these to the required action space. 

The code runs the training in parallel over multiple workers, as self play game generation is very computationally intensive.
For this reason, running this code on a system that has CUDA support is recommended.  


Most of the Monte Carlo Tree Search related code can be found in games/algos/mcts.py, and the (slightly messy) training 
code can be found in games/algos/self_play_parallel.py. 

## Training

Training can be run by running the games/connect4/run_self_play_connect4.py script, which will run self play training
games, storing the models as well as a memory of the most recent (default 100,000) positions along with probability statistics.

## Evaluation
The models can be evaluated by the use of the games/algos/elo.py file, which can persist model versions along with 
arguments and weights, and play them against eachother. The results of this can then be used to calculate the Elo value.  

