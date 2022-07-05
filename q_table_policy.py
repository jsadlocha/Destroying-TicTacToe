
from env_gym import TicTacToe
from min_max import MinMax
from tqdm import tqdm

import numpy as np


class QtablePolicy:
    def __init__(self, gamma:float = 0.9, alfa:float = 0.5, epsilon: float = 0.99, decay:float = 0.99, temp: float = 1.0):
        self.qtable = dict()
        self.gamma = gamma
        self.alfa = alfa
        self.epsilon = epsilon
        self.decay = decay
        self.temp = temp

    def update(self, episodes):
        states, actions, next_states, rewards, done = zip(*episodes)
        discount_factor = np.array([self.gamma**i for i in range(len(states)+1)])
        
        
        for idx, state in enumerate(states):
            hash_value = self.get_hash_value(state)
            self.create_new_item(state)
            old_q = self.qtable[hash_value][1][actions[idx]]
            prob = self.get_probability(state)

            #np.take(self.qtable[hash_value][1], self.qtable[hash_value][0])
            #np.where(np.round(prob, 15)==1)[0][0]

            prob_idx = np.where(np.array(self.qtable[hash_value][0]) == actions[idx])[0][0]
            log_prob = np.log(prob[prob_idx]+1e-12)
            discounted_reward = np.sum(rewards[idx:] * discount_factor[:-(1+idx)])
            #self.qtable[hash_value][1][actions[idx]] = old_q + self.alfa * (discounted_reward - old_q)
            #print(self.qtable[hash_value][1][actions[idx]])
            
            
            self.qtable[hash_value][1][actions[idx]] = old_q + self.alfa * -(log_prob * discounted_reward + old_q)#- self.alfa*old_q

            # normalize
            #prob = self.get_probability(state)/10.0
            # mean = prob.mean()
            # std = prob.std()#.clamp_min(1e-12)
            # prob = (prob-mean)/std
            #indices = self.qtable[hash_value][0]
            #norm_q = np.array(self.qtable[hash_value][1])
            #norm_q[indices] = prob.tolist()
            # self.qtable[hash_value][1] = norm_q.tolist()
            #import pdb; pdb.set_trace()       
            #print(self.qtable[hash_value][1][actions[idx]])
            #import pdb; pdb.set_trace()
            

    def get_probability(self, state):
        logits = self.get_logits(state)
        #print(f'logits: {logits}')
        logits = logits.clip(-700,700)
        prob = np.exp(logits/self.temp)/np.sum(np.exp(logits/self.temp))
        #prob = np.exp(logits)/np.sum(np.exp(logits))
        return prob

    def get_logits(self, state):
        hash = self.get_hash_value(state)
        self.create_new_item(state)
        q_table = self.qtable[hash]
        logits = np.take(q_table[1], q_table[0])
        return logits

    def create_new_item(self, state):
        hash_state = self.get_hash_value(state)
        if hash_state not in self.qtable:
            indices  = self.get_legal_move(state)
            new_elem = [indices, [0]*9]
            self.qtable[hash_state] = new_elem

    def get_legal_move(self, state):
        free_fields = np.array(state) == 0
        free_indices = np.where(free_fields > 0)[0]
        return free_indices.tolist()

    def get_hash_value(self, state):
        res = 0
        for idx, val in enumerate(state):
            res += (3**idx) * val
        return res

    def normalize_qtable(self):
        for state, q in self.qtable.items():
            
            idx = q[0]
            #logits = q[1]
            #logit = np.take(logits, idx)
            #import pdb; pdb.set_trace()
            prob = np.exp(np.take(q[1], q[0]))
            #prob = np.round((prob/np.sum(prob))*10, 15)
            prob = prob/np.sum(prob)
            logit = np.log(prob+1e-12)
            old_q = np.array(self.qtable[state][1])
            #import pdb; pdb.set_trace()
            old_q[idx] = logit#logit-1.0
            self.qtable[state][1] = old_q.tolist()


    def get_action(self, board):
        if np.random.rand() > self.epsilon:
            probs = self.get_probability(board)
            sample = np.random.multinomial(1, probs)
            # legal = self.qtable[self.get_hash_value(board)][0]
            # logits = np.take(self.qtable[self.get_hash_value(board)][1], legal)
            # action = legal[np.argmax(logits)]
            
            
            #probs = self.qtable[self.get_hash_value(board)][1]
            #action_idx = np.where(sample > 0)[0][0]
            action_idx = np.argmax(probs)
            action = self.get_action_number_from_indices(board, action_idx)    
        else:
            self.create_new_item(board)
            legal_move = self.get_legal_move(board)
            action = np.random.choice(legal_move)
        return action

    def get_action_number_from_indices(self, state, idx):
        hash_value = self.get_hash_value(state)
        q_table = self.qtable[hash_value]
        action = q_table[0][idx]
        
        return action

    def decay_epsilon(self):
        if self.epsilon > 0.1:
            self.epsilon *= self.decay
        else:
            self.epsilon = 0.0


def gameloop():
    episodes = 10001
    env = TicTacToe()
    minmax = MinMax()
    reward = [0, 1, -1]
    micro_reward = 0#-0.1
    agent = QtablePolicy()
    agent.epsilon = 0
    count = [0, 0, 0]
    loop = tqdm(range(episodes))
    for t in loop:
        transition = []
        env.reset()
        done = False
        
        while not done:
            # O player
            state = env.board[:]
            action = agent.get_action(state)          
            next_state, r, done, _ = env.step(action)
            transition.append((state, action, next_state, reward[r] if done else micro_reward, done))
            
            if done:
                break

            # X player
            # random move
            action = env.sample_action()

            # engine move
            #action = minmax.getBestMoveAndNonDeterministic(env.getState(), -1)

            state = env.board[:]
            next_state, r, done, _ = env.step(action)
            transition.append((state, action, next_state, reward[r] if done else micro_reward, done))
     
        count[r] += 1

        # update q_table
        agent.update(transition)

        if t % 50 == 0:
            agent.decay_epsilon()

        #if (t+1) % 1000 == 0:
        #    agent.normalize_qtable()

        if (t+1) % 1000 == 0:
            print(f"\nDraw: {count[0]} O Win: {count[1]} X Win: {count[-1]}")
            count = [0, 0, 0]
            
            prob = agent.get_probability(transition[0][0])
            ent = np.sum(prob * np.log(prob+1e-12))
            print(f'entropy: {ent}')

        loop.set_description(f"eps: {agent.epsilon} q: {len(agent.qtable)}")

if __name__ == "__main__":
    gameloop()