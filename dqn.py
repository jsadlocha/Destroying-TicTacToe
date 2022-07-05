import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import math as m
import random

from env_gym import TicTacToe

from tqdm import tqdm

import matplotlib.pyplot as plt

from collections import deque

from min_max import MinMax

# some problem with dll fix
#import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Model
class DQN(nn.Module):
    def __init__(self, input_shape, hidden_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_shape, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # zero improvement with weight initialization
        self.init_weight(input_shape, hidden_size)

    def init_weight(self, input_shape, hidden_size):
        # Weight Initialization for relu
        # Normal gaussian std [sqrt(2/nodes_input)]
        # nn.init.normal_(tensor, 0, sqrt(2/n) OR randn * sqrt(2/n)
        nn.init.normal_(self.input.weight, 0, m.sqrt(2/input_shape))
        nn.init.constant_(self.input.bias.data, 0)

        nn.init.normal_(self.hidden.weight, 0, m.sqrt(2/hidden_size))
        nn.init.constant_(self.hidden.bias.data, 0)

        nn.init.normal_(self.hidden2.weight, 0, m.sqrt(2/hidden_size))
        nn.init.constant_(self.hidden2.bias.data, 0)

    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.hidden2(x)
        x = self.relu(x)
        x = self.output(x)
        return x

class ReplayMemory:
    def __init__(self, max_size):
        self.deque = deque(maxlen=max_size)

    def __len__(self):
        return len(self.deque)

    def get_random_batch(self, batch_size):
        return random.sample(self.deque, batch_size)

    def remember(self, state, action, reward, next_state, done):
        self.deque.append((state, action, reward, next_state, done))

class Agent(nn.Module):
    def __init__(self, state_size, hidden_size=27, action_size=9, epsilon = 1.0, lr = 0.003, gamma = 0.8, alfa = 0.9, batch_size=128, tau=0.1):
        super().__init__()
        self.dqn = DQN(state_size, hidden_size, action_size)
        self.target_dqn = DQN(state_size, hidden_size, action_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'
        self.memory = ReplayMemory(max_size=512)
        self.batch_size = batch_size
        self.decay = 0.5
        self.tau = tau

        self.lr = lr
        self.alfa = 0.7
        self.optimizator = optim.Adam(self.dqn.parameters(), self.lr)
        self.loss_f = nn.MSELoss()
        self.to(self.device)

    # encoding state space
    def label_encoding(self, board):
        """
        Return: shape [9] int
        1 - O, -1 - X, 0 - Empty
        """
        x = np.array(board[:])
        x[x==2] = -1
        return torch.tensor(x).float().to(self.device)


    def one_hot_encoding(self, board):
        "Return: shape [18] int"
        new_state = torch.tensor(board[:])
        O = new_state==1
        X = new_state==2
        new_state = torch.hstack((O, X)).float().to(self.device)
        return new_state

    def get_action(self, env):
        with torch.no_grad():
            if np.random.rand() > self.epsilon:
                # best move
                possible_move = np.invert(env.get_possible_moves()).astype('int32')*-99999
                move = self.run(env.board).cpu().detach().numpy()
                return (move+possible_move).argmax()
            else:
                # random move
                return env.sample_action()

    def run(self, input):
        return self.dqn(self.label_encoding(input))

    def update_agent_mc(self):
        # update monte carlo q learning    
        states, actions, rewards, next_state, done = zip(*self.memory.deque)
        discount_values = np.array([self.gamma**i for i in range(len(self.memory)+1)])

        self.memory.deque.clear()

        preds = []
        targets = []

        for idx, state in enumerate(states):
            discounted_reward = np.sum(np.array(rewards[idx:]) * discount_values[:-(1+idx)])
            score = self.run(state)

            preds.append(score)

            old_q = score[actions[idx]]
            q_update = old_q + self.alfa * (discounted_reward - old_q)
            
            target = score.clone()
            target[actions[idx]] = q_update
            targets.append(target)

        preds = torch.stack(preds).to(self.device)
        targets = torch.stack(targets).to(self.device)

        self.optimizator.zero_grad()
        loss = self.loss_f(preds, targets)
        loss.backward()
        self.optimizator.step()
        return loss
    
    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.decay
        else:
            self.epsilon = 0.0

def moving_exponential_average(losses, alfa = 0.5):
    res = []
    last = losses[0]
    for point in losses:
        last = last * alfa + (1-alfa)* point
        res.append(last)

    return np.array(res)

def draw_graph_losses(losses):
    t = np.arange(len(losses))
    losses = moving_exponential_average(losses, alfa=0.9)
    plt.plot(t, losses)
    plt.show()

def game_loop():
    env = TicTacToe()
    batch_size = 64
    agent = Agent(9, 27, 9, epsilon=1.0, lr=0.001, gamma=0.9, alfa=1.0, batch_size=batch_size, tau=0.99)
    save = False
    load = False

    if load:
        agent.load_state_dict(torch.load("dqn_agent.trh"))

    render = False
    episodes = 10001
    minmax = MinMax()
    count_result = [0, 0, 0]
    reward = [0.5, 1, -1]
    exploration_reward = 0 #0.1
    losses = []
    loop = tqdm(range(episodes))
    ok = False
    
    for t in loop:
        env.reset()
        done = False
        while not done:
            # Agent O
            action = agent.get_action(env)
            state = env.board[:]
            next_state, r, done, _ = env.step(action)
            
            agent.memory.remember(state, action, reward[r] if done else exploration_reward, next_state, done)
            if done:
                break

            # X player
            action = minmax.getBestMoveAndNonDeterministic(env.getState(), -1)
            state = env.board[:]
            next_state, r, done, _ = env.step(action)
            agent.memory.remember(state, action, reward[r] if done else exploration_reward, next_state, done)              
            
            if render:
                env.render()

        if (t+1) % 1000 == 0:
            agent.decay_epsilon()

        
        if (t+1) % 1000 == 0:
            print(f'\nDraw: {count_result[0]}, O Win: {count_result[1]}, X Win: {count_result[-1]}, epsilon: {agent.epsilon}')
            count_result = [0, 0, 0]

        _, res = env.board.checkWinDrawEnd(action)
        count_result[res] += 1

        loss = agent.update_agent_mc()
        
        loop.set_description(f'loss: {loss}')
        losses.append(loss.cpu().detach().numpy())
    
    draw_graph_losses(losses)
    if save:
        torch.save(agent.state_dict(), "dqn_agent.trh")

if __name__ == "__main__":
    game_loop()

    

    