from board import Board
import numpy as np
from typing import List, Tuple, Dict

class MinMax:
    def __init__(self):
        self.lookup_table: Dict[int, List[Tuple[int, int]]] = dict()

    def __call__(self, board: List[int]) -> List[Tuple[int, int]]:
        graph = board[:]

        solution = []
        for idx, elem in enumerate(graph):
            if elem == 0:
                dfs = self.dfs(graph, idx)
                solution.append((idx, dfs))

        self.lookup_table[Board.hash(board[:])] = solution[:]
        return solution

    def get_lookup_table(self, board: List[int]) -> List[Tuple[int, int]]:
        if not (Board.hash(board[:]) in self.lookup_table):
            self.__call__(board)     
        return self.lookup_table[Board.hash(board[:])]

    def getBestMoveAndNonDeterministic(self, board: List[int], player: int) -> int:
        actions = self.get_lookup_table(board)
        action = np.array(actions)
        
        act = np.where(action[:,1] == player)[0]

        if len(act) == 0:
            act = np.where(action[:,1] < player*-1)[0]
        #import pdb; pdb.set_trace()
        idx = np.random.choice(act)
        return actions[idx][0]


    def dfs(self, graph: List[int], pos: int) -> int:
        board = graph[:]
        player = 1 if board.count(0) % 2 else 2
        board[pos] = player
        end, score = Board.checkWinAndEnd(board, pos, player)
        if end:
            return score
        
        solution = []
        cache = []
        for idx, elem in enumerate(board):
            if elem == 0:
                dfs = self.dfs(board, idx)
                solution.append(dfs)
                cache.append((idx, dfs))

        self.lookup_table[Board.hash(board[:])] = cache[:]

        return max(solution) if player-1 else min(solution)


if __name__ == "__main__":
    pass
