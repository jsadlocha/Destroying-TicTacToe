from board import Board, Player, StatusBoard
import numpy as np
from typing import Tuple, List

class InvalidMoveException(Exception):
    pass

class GameIsOverException(Exception):
    pass

class TicTacToe:
    def __init__(self):
        self.board = Board()

    def reset(self) -> None:
        self.board.reset()

    def render(self) -> None:
        print(self.board)

    def getState(self) -> List[int]:
        return self.board[:]

    def step(self, pos: int) -> Tuple[list, int, bool, int]:
        player = Player.O if self.board[:].count(0) % 2 else Player.X
        if self.board.move(pos, player) == StatusBoard.INVALID_MOVE:
            raise InvalidMoveException("Invalid move, game was stopped!")

        done, reward = self.board.checkWinDrawEnd(pos, player.value)

        next_state = self.board[:]
        
        return next_state, reward, done, 0

    def sample_action(self) -> int:
        prob = np.array(self.board[:])==0
        if np.sum(prob) == 0:
            raise GameIsOverException("The game is over!")
        prob = prob * (1/np.sum(prob))
        return np.random.choice(np.arange(9), p=prob)

    def get_possible_moves(self) -> np.ndarray:
        return np.array(self.board[:]) == 0

    def close(self):
        pass

if __name__ == "__main__":
    pass