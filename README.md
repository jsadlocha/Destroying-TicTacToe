


# Destroying TicTacToe


I try here different approaches, to beat opponent in tic-tac-toe.
This game was chosen, because had very small amount combinations on the board.
Every reinforcement learning algorithm runs as monte carlo, what mean learning is done after one whole game, because one step ahead learning is not enough for adversarial agent learning.
Agents fight versus Random Moves or MinMax nondeterministic, which is just a random the top move.


Algorithms with neural networks, they work on pytorch framework.</br>
GUI is very simple and implemented in pygame, engine default is used MinMax, who give some tips about moves.</br>
I used python 3.9.7


Every file is self proving.</br>
Usage: `python <filename>`

*without MinMax and AlphaBetaPruning, but they are used to proving other methods.

## Algorithm list:


* MinMax - Depth first search algorithm, with caching, built on the arrays.</br>
    `filename: min_max.py`

* Alpha-Beta Pruning - optimized version of MinMax, pruning unecessary branches.</br>
    `filename: alpha_beta_pruning.py`

* Q table - q learning algorithm on the lookup table, monte carlo.</br>
    `filename: q_table.py`

* Policy gradient table - implemented on the lookup table, monte carlo.</br>
    `filename: q_table_policy.py`

* Q learning - q learning on the deep neural network, monte carlo.</br>
    `filename: dqn.py`

* Policy Gradient - PG on the deep neural network, monte carlo.</br>
    `filename: policy_gradient.py`

* Monte Carlo Tree Search - MCTS statistics.</br>
    `filename: mcts.py`

* Actor Critics - AC/A2C/A3C not implemented yet.

* Monte Carlo Tree Search with Deep Neural Network - MCTS DNN not implemented yet.



## The rest of the files:


* Gym env - implementation interface of gym env.</br>
    `filename: env_gym.py`

* TicTacToe - implementation of game board.</br>
    `filename: board.py`

* TicTacToe GUI Game - pygame implementation UI for tictactoe with engine MinMax.</br>
    `filename: tictactoe_gui.py`

* Test - testing tictactoe.</br>
    `path: test\test_board.py`