import math as m
import numpy as np
import random

from env_gym import TicTacToe
from min_max import MinMax


from tqdm import tqdm # type: ignore

from typing import Optional

class NodeDoesNotExist(Exception):
    pass

class MoveIsIncorrect(Exception):
    pass

class Node:
    def __init__(self, parent: 'Node', state: list[int], player: Optional[int], node_action: Optional[int] = None, C: float = 0.1):
        self.parent = parent
        self.children: Optional[list] = None
        self.state = state
        self.is_player = player
        self.node_action = node_action

        self.win_count = 0
        self.visited_count: float = 0.
        self.C = C

    def calculate_ucb(self) -> float:
        """ Calculate upper confidence bound for this node"""
        if self.parent is None:
            NodeDoesNotExist("Node does not exist!")

        n = self.visited_count
        if n == 0:
            n = 1e-6
        mean = self.win_count / n
        ucb = mean + self.C * m.sqrt(m.log(self.parent.visited_count)/n)
        return ucb

class MCTS:
    def __init__(self):
        self.root = Node(parent=None, state=[0]*9, player=2)

    def selection(self, node: Node) -> Node:
        leaf = node
        while leaf.children is not None:
            leaf = self.getPromisingNode(leaf)

        return leaf

    def expansion(self, node: Node) -> None:
        allowed_moves = self.getAllowedMoves(node.state)
        node.children = []
        player = self.changePlayer(node)
        for idx in allowed_moves:
            state = node.state[:]
            state[idx] = player
            node.children.append(Node(node, state, player, idx))

    def simulation(self, node: Node) -> int:
        board = node.state[:]
        player = node.is_player

        game_state = 0
        while True:
            game_state = self.isGameOver(board)
            if game_state:
                break
            first = 0
            player = self.changePlayer(node, player)
            idx = self.getRandomMoveIdxFromBoard(board)
            board = self.makeMoveOnTheBoard(board, idx, player)

        return game_state


    def backpropagation(self, node: Node, reward: int) -> None:
        # not end, O win, X win, draw
        reward_point = [0, 1, -1, 0]
        point = reward_point[reward]

        node.win_count += point
        node.visited_count += 1

        while node.parent is not None:
            node = node.parent
            node.visited_count += 1
            point = reward_point[reward] if node.is_player == 1 else reward_point[reward] *-1
            node.win_count += point

    def getBestActionForNode(self, node: Node) -> int:
        if node.children is None:
            raise NodeDoesNotExist("Node does not exist!")

        idx = self.getBestUCBIdx(node)

        action = node.children[idx].node_action
        return action

    def getAllowedMoves(self, board: list[int]) -> list[int]:
        allowed_move = np.array(board) == 0
        indices = np.where(allowed_move)[0]
        return indices.tolist()
        
    def getRandomMove(self, node: Node) -> int:
        allowed_move = self.getAllowedMoves(node.state)
        choice = random.randint(0, len(allowed_move)-1)
        return allowed_move[choice]

    def getRandomMoveIdxFromBoard(self, board: list[int]) -> int:
        allowed_move = self.getAllowedMoves(board)
        choice = random.randint(0, len(allowed_move)-1)
        return allowed_move[choice]

    def makeMoveOnTheBoard(self, board: list[int], idx: int, player: int) -> list[int]:
        board[idx] = player
        return board

    def getPromisingNode(self, node: Node) -> Node:
        """Return node with the highest upper confidence bound"""
        if node.children is None:
            raise NodeDoesNotExist("Node does not exist!")

        idx = self.getBestUCBIdx(node)
        promising_node = node.children[idx]
        return promising_node

    def getBestUCBIdx(self, node: Node) -> int:
        ucb: list[float] = []

        if node.children is None:
            raise NodeDoesNotExist("Node does not exist!")

        for child in node.children:
            ucb.append(child.calculate_ucb())

        max_ucb = max(ucb)
        idx = ucb.index(max_ucb)
        return idx

    def changePlayer(self, node: Node, value: Optional[int] = None) -> int:   
        player = node.is_player

        if player is None:
            raise NodeDoesNotExist("Node does not exist!")

        if value is not None:
            player = value

        return (player % 2) + 1

    
    
    def findChildrenWithAction(self, node: Node, action: int) -> Node:
        if node.children is None:
           raise NodeDoesNotExist("Node does not exist!")

        for child in node.children:
            if child.node_action == action:
                return child

        raise MoveIsIncorrect("Move is incorrect!")

    def isGameOver(self, board: list[int]) -> int:
        """0 - not end, 1 - O win, 2 - X win, 3 - Draw"""
        np_board = np.array(board).reshape((3,3))
        free_space = np.sum(np_board==0)
        draw = free_space == 0
        if draw:
            return 3

        last_move = (free_space % 2) + 1
        win_sum = 3
        np_board = np_board == last_move
        # check row
        row = np.sum(np.sum(np_board, axis=1) == win_sum)

        # check column
        column = np.sum(np.sum(np_board, axis=0) == win_sum)

        # check cross
        cross1 = np.sum(np_board*np.eye(3,3)) == win_sum
        cross2 = np.sum(np.rot90(np_board)*np.eye(3,3)) == win_sum
        
        if (row+column+cross1+cross2) > 0:
            return last_move

        return 0  

    def evaluate(self, root: Node, e: int = 2) -> None:
        for e in range(e):
            node = root
            leaf = self.selection(node)

            if leaf.visited_count != 0 and not self.isGameOver(leaf.state):
                self.expansion(leaf)

                if leaf.children is None:
                    raise NodeDoesNotExist("Node does not exist!")

                leaf = random.choice(leaf.children)
            
            reward = self.simulation(leaf)
            self.backpropagation(leaf, reward)


def gameloop():
    env = TicTacToe()
    episodes = 100
    mcts = MCTS()
    engine = MinMax()
    loop = tqdm(range(episodes))
    count = [0, 0, 0]
    for t in loop:
        node = mcts.root
        env.reset()
        done = False
        while not done:
            # O Player
            state = env.board[:]
            action = engine.get_lookup_table(env.board)
            action.sort(key=lambda x: x[1], reverse=True)
            action = action[0][0]
            mcts.evaluate(node, 150)
            node = mcts.findChildrenWithAction(node, action)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            
            # X Player
            state = env.board[:]
            mcts.evaluate(node, 150)
            idx = mcts.getBestUCBIdx(node)
            node = node.children[idx]
            action = node.node_action
            if node is None:
                print('error')
            next_state, reward, done, _ = env.step(action)

        count[reward] += 1
    print(f'\nDraw: {count[0]}, O win: {count[1]}, X win: {count[-1]}')


if __name__ == "__main__":
    gameloop()
    