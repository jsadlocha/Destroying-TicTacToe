import torch
import torch.optim as optim
import torch.nn as nn

from env_gym import TicTacToe
from min_max import MinMax

from tqdm import tqdm
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_shape, hidden_layer, action_shape):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_shape, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, action_shape),
        )

    # encoding state space
    def label_encoding(self, board):
        """
        Return: shape [9] int
        1 - O, -1 - X, 0 - Empty
        """
        x = np.array(board[:])
        x[x==2] = -1
        return torch.tensor(x).float()

    def one_hot_encoding(self, board):
        "Return: shape [18] int"
        new_state = torch.tensor(board[:])
        O = new_state==1
        X = new_state==2
        new_state = torch.hstack((O, X)).float()
        return new_state

    def forward(self, input: torch.Tensor):
        input = self.label_encoding(input)
        input = self.linear(input)
        return input


class PolicyGradient(nn.Module):
    def __init__(self, input_shape, hidden_layer, action_shape):
        super().__init__()
        self.mlp = MLP(input_shape, hidden_layer, action_shape)
        self.optimizer = optim.Adam(self.mlp.parameters(), lr=0.0003)
        self.gamma = 0.9
        self.epsilon = 1.0
        self.decay = 0.5

    def get_action(self, board, env):
        if np.random.rand() > self.epsilon:
            state = torch.tensor(board[:])
            legal_move = state == 0
            logits = self.mlp(state.float())
            logits = torch.masked_select(logits, legal_move)
            idx = torch.masked_select(torch.arange(9), legal_move)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            return idx[action.item()].item()
        else:
            return env.sample_action()
    
    def update(self, episodes):
        states, actions, rewards, next_states, done = zip(*episodes)
        states = torch.tensor(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        #next_states = torch.tensor(next_states)
        done = torch.tensor(done)

        discount_factor = torch.tensor([self.gamma**i for i in range(len(rewards)+1)])
        discounted_reward = torch.tensor([torch.sum(torch.tensor(rewards[i:])*discount_factor[:-(1+i)]) for i in range(len(rewards))])

        logits = self.mlp(states.float())
        policy = torch.distributions.Categorical(logits=logits)
        log_prob = policy.log_prob(actions)
        
        # mean = discounted_reward.mean()
        # std = discounted_reward.std().clamp_min(1e-12)
        # normalized_discounted_reward = (discounted_reward-mean)/std

        loss = -(log_prob * discounted_reward).mean()
        self.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.mlp.parameters(), 1)
        self.optimizer.step()
        return loss

    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.decay
        else:
            self.epsilon = 0.0


def gameloop():
    agent = PolicyGradient(9, 36, 9)
    env = TicTacToe()
    minmax = MinMax()
    eps = 40000
    # lost just exploding and reward is adjusted optimally
    reward = [1, 1, -0.01]
    count = [0, 0, 0]
    micro_reward = 0
    agent.epsilon = 1.0
    loop = tqdm(range(eps))
    for e in loop:
        episodes = []
        env.reset()
        done = False
        while not done:
            # O player
            state = env.board[:]
            action = agent.get_action(env.board, env)
            next_state, r, done, _ = env.step(action)
            episodes.append((state, action, reward[r] if done else micro_reward, next_state, done))

            if done:
                break

            # X player

            # random
            #action = env.sample_action() 

            # engine top move
            action = minmax.getBestMoveAndNonDeterministic(env.getState(), -1)

            state = env.board[:]
            next_state, r, done, _ = env.step(action)
            episodes.append((state, action, reward[r] if done else micro_reward, next_state, done))      

        win = env.board.checkWinDrawEnd(action)
        count[win[1]] += 1
        if (e+1)%1000 == 0:
            print(f'\nDraw: {count[0]}, O Win: {count[1]}, X Win: {count[-1]}')
            print(f'epsilon: {agent.epsilon}')
            count=[0,0,0]

        if (e+1)% 1000 == 0:
            agent.decay_epsilon()

        loss = agent.update(episodes)
        #print(f'loss: {loss}')
        loop.set_description(f'loss: {loss}')


if __name__ == "__main__":
    gameloop()