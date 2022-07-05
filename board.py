from enum import Enum
from typing import Union

class Player(Enum):
    Free = 0
    O = 1
    X = 2

class StatusBoard(Enum):
    NO_WIN = -2
    INVALID_MOVE = -1
    OK = 0
    WIN = 2

class Board:
    def __init__(self):
        self.board = []
        self.reset()

    def reset(self) -> None:
        self.board = [Player.Free.value]*9
    
    def __call__(self) -> list[int]:
        return self.board

    def __getitem__(self, idx: Union[int, slice]) -> list[int]:
        return self.board[idx]

    def __len__(self) -> int:
        return len(self.board)

    def __str__(self) -> str:
        elem = [' ', 'O', 'X']
        board = [elem[x] for x in self.board]
        return "{}|{}|{}\n{}|{}|{}\n{}|{}|{}\n".format(*board)

    def hash_value(self) -> int:
        res = 0
        for idx, val in enumerate(self.board):
            res += (3**idx) * val
        return res

    @staticmethod
    def hash(board: list[int]) -> int:
        res = 0
        for idx, val in enumerate(board):
            res += (3**idx) * val
        return res

    def move(self, position: int, player: Player) -> StatusBoard:
        if self.board[position] == Player.Free.value:
            self.board[position] = player.value
            return StatusBoard.OK
        
        return StatusBoard.INVALID_MOVE

    def checkWin(self, last_move: int) -> StatusBoard:
        row = last_move // 3
        col = last_move % 3

        # check column
        if self.board[0+col] == self.board[3+col] and self.board[3+col] == self.board[6+col]:
            return StatusBoard.WIN
        
        # check row
        if self.board[row*3+0] == self.board[row*3+1] and self.board[row*3+1] == self.board[row*3+2]:
            return StatusBoard.WIN

        # check cross
        if last_move % 2 == 0:
            if self.board[0] == self.board[4] and self.board[4] == self.board[8] and self.board[0] != Player.Free.value:
                return StatusBoard.WIN
            if self.board[2] == self.board[4] and self.board[4] == self.board[6] and self.board[2] != Player.Free.value:
                return StatusBoard.WIN

        return StatusBoard.NO_WIN

    def checkWinDrawEnd(self, last_move:int, player: int = None) -> tuple[bool, int]:
        row = last_move // 3
        col = last_move % 3
        score = [1, -1]
        if player is None:
            player = 1 if self.board.count(0) % 2 == 0 else 2
        # check column
        if self.board[0+col] == self.board[3+col] and self.board[3+col] == self.board[6+col] and self.board[0+col] == player:
            return True, score[player-1]
        
        # check row
        if self.board[row*3+0] == self.board[row*3+1] and self.board[row*3+1] == self.board[row*3+2] and self.board[row*3+2] == player:
            return True, score[player-1]

        # check cross
        if last_move % 2 == 0:
            if self.board[0] == self.board[4] and self.board[4] == self.board[8] and self.board[0] == player:
                return True, score[player-1]
            if self.board[2] == self.board[4] and self.board[4] == self.board[6] and self.board[2] == player:
                return True, score[player-1]

        # Draw
        if self.board.count(0) == 0:
            return True, 0

        # nothing happend
        return False, 0

    @staticmethod
    def checkWinAndEnd(board: list[int], last_move: int, player: int) -> tuple[bool, int]:
        row = last_move // 3
        col = last_move % 3
        score = [1, -1]
        # check column
        if board[0+col] == board[3+col] and board[3+col] == board[6+col] and board[0+col] == player:
            return True, score[player-1]
        
        # check row
        if board[row*3+0] == board[row*3+1] and board[row*3+1] == board[row*3+2] and board[row*3+2] == player:
            return True, score[player-1]

        # check cross
        if last_move % 2 == 0:
            if board[0] == board[4] and board[4] == board[8] and board[0] == player:
                return True, score[player-1]
            if board[2] == board[4] and board[4] == board[6] and board[2] == player:
                return True, score[player-1]

        # Draw
        if board.count(0) == 0:
            return True, 0

        # nothing happend
        return (False, 0)


    def checkIfBoardIsFull(self) -> bool:
        for i in self.board:
            if i == Player.Free.value:
                return False
        return True        


#import sys
#sys.setrecursionlimit(10)

if __name__ == "__main__":
    pass
    #x = Board()