import numpy as np
import math as m

from board import Board

from min_max import MinMax
from env_gym import TicTacToe
from tqdm import tqdm # type: ignore

import pickle

class Qtable:
    def __init__(self):
        self.q = np.random.uniform(-0.1, 0.1, (m.factorial(9), 9))
        self.lr = 0.8
        self.gamma = 0.90
        self.epsilon = 0.1
        self.decay = 0.90

    def get_move_index(self, board: Board) -> np.signedinteger:
        legal_moves = np.array(board())==0
        if np.random.rand() > self.epsilon:
            state = board.hash_value()
            prob = self.q[state] * legal_moves 
            player = board[:].count(0) % 2      
            if np.sum(prob) != 0:
                if player:
                    if legal_moves[np.argmax(prob)] != False:
                        return np.argmax(prob)
                else:
                    if legal_moves[np.argmin(self.q[state])] == True:
                        return np.argmin(self.q[state])
          
        probility = legal_moves*(1/np.sum(legal_moves))

        return np.random.choice(np.arange(9), p=probility)

    def update_q(self, episode: list) -> None:
        states, actions, rewards = zip(*episode)
        discount = np.array([self.gamma**x for x in range(len(states)+1)])

        for idx, state in enumerate(states):
            old_q = self.q[state][actions[idx]]
            discount_reward = np.sum(rewards[idx:]*discount[:-(1+idx)])
            self.q[state][actions[idx]] += self.lr * (discount_reward - old_q)

    def decay_epsilon(self) -> None:
        self.epsilon *= self.decay
        if self.epsilon < 0.1:
            self.epsilon = 0


def gameloop():
    env = TicTacToe()
    engine = MinMax()
    episodes = 10000
    render = False
    load = False
    save = False
    qtable = Qtable()
    #text = ["Draw", "O Win", "X Win"]
    count = [0, 0, 0]
    rew = [0, 1, -1]
    qtable.epsilon = 0#0.5
    micro_rew = 0.1

    if load:
      qtable.q = pickle.load(open("qtable_agent700.pkl", "rb"))

    for episode in tqdm(range(episodes)):
        env.reset()
        end = False
        transitions = []

        while not end:
            # O player
            action = qtable.get_move_index(env.board)
            state = env.board.hash_value()
            obs, reward, end, _ = env.step(action)
            transitions.append((state, action, rew[reward] if end else micro_rew))
            if end:
                break

            # X player 

            # random move
            #action = env.sample_action()

            # engine move
            action = engine.getBestMoveAndNonDeterministic(env.getState(), -1)

            state = env.board.hash_value()
            obs, reward, end, _ = env.step(action)

            transitions.append((state, action, rew[reward] if end else micro_rew))
            if render:
                env.render()

        if (episode+1) % 1000 == 0:
            qtable.decay_epsilon()
            
        if episode % 1000 == 0:
            print(f'\nDraw: {count[0]}, O win: {count[1]}, X win: {count[2]}')
            print(f'epsilon: {qtable.epsilon}')
            count = [0,0,0]    

        count[reward] += 1
        qtable.update_q(transitions)
        
    print(f'Draw: {count[0]} O win: {count[1]} X win: {count[2]}')
    
    if save:
        with open("qtable_agent.pkl", "wb") as f:
            pickle.dump(qtable.q, f)


if __name__ == "__main__":
    gameloop()

