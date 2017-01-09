# Deep-Reinforcement-Learning-using-Pong-Player
README
Python version 2.7.11

Policy Network Implementation:
policy_gradients_train.py

This program trains agent using policy gradient algorithm mentioned in the report.
It periodically saves weights in policy_gradients_weights.p file
In the beginning it checks for this file and if it finds, it loads the weights and continuous learning from there

Command : python policy_gradients_train.py

policy_gradients_play.py

This program plays pong game for 10 episodes. It takes actions based on the training model (loads network
weights from policy_gradients_weights.p file created before). We need to train agent using
policy_gradients_train.py program before running this game.

Command : python policy_gradients_play.py

Q-Learning Implementation:

q_learn_train.py

This program trains agent using q learning algorithm mentioned in the report.
It periodically saves weights in qlearn.ckpt file
In the beginning it checks for this file and if it finds, it loads the weights and continuous learning from there

Command : python q_learn_train.py

q_learn_play.py

This program plays pong game for 10 episodes. It takes actions based on the training model (loads network
weights from qlearn.ckpt file created before). We need to train agent using q_learn_train.py program before
running this game.

Command : python q_learn_play.py
