import unittest
from board import Board, Player, StatusBoard

class TestBoardMethods(unittest.TestCase):
    def setUp(self) -> None:
        self.board = Board()

    def testColumnWinLastMove(self):
        self.board.move(2, Player.X)
        self.board.move(5, Player.X)
        self.board.move(8, Player.X)
        self.assertEqual(self.board.checkWin(8), StatusBoard.WIN)

    def testRowWinLastMove(self):
        self.board.move(3, Player.O)
        self.board.move(4, Player.O)
        self.board.move(5, Player.O)
        self.assertEqual(self.board.checkWin(4), StatusBoard.WIN)

    def testCrossWinLastMove(self):
        self.board.move(2, Player.X)
        self.board.move(4, Player.X)
        self.board.move(6, Player.X)
        self.assertEqual(self.board.checkWin(2), StatusBoard.WIN)

    def testColumnNoWinLastMove(self):
        self.board.move(0, Player.X)
        self.board.move(3, Player.O)
        self.board.move(6, Player.X)
        self.assertEqual(self.board.checkWin(6), StatusBoard.NO_WIN)

    def testRowNoWinLastMove(self):
        self.board.move(6, Player.O)
        self.board.move(7, Player.X)
        self.board.move(8, Player.O)
        self.assertEqual(self.board.checkWin(7), StatusBoard.NO_WIN)

    def testCrossNoWinLastMove(self):
        self.board.move(0, Player.O)
        self.board.move(4, Player.X)
        self.board.move(8, Player.X)
        self.assertEqual(self.board.checkWin(2), StatusBoard.NO_WIN)

    def testCheckIfLegalMove(self):
        self.assertEqual(self.board.move(0, Player.O), StatusBoard.OK)

    def testCheckIfIllegalMove(self):
        self.board.move(1, Player.X)
        self.assertEqual(self.board.move(1, Player.X), StatusBoard.INVALID_MOVE)

    def testCrossShouldNotWin(self):
        self.board.move(0, Player.O)
        self.assertEqual(self.board.checkWin(0), StatusBoard.NO_WIN)

    def testShouldResetBoard(self):
        self.board.move(0, Player.O)
        self.board.move(3, Player.X)
        self.board.move(7, Player.O)
        self.board.reset()
        self.assertEqual(self.board(), [Player.Free.value]*9)

    def testShouldBeSpaceOnTheBoard(self):
        self.board.move(4, Player.X)
        self.board.move(0, Player.O)
        self.assertEqual(self.board.checkIfBoardIsFull(), False)

    def testShouldBeBoardFull(self):
        for i in range(9):
            self.board.move(i, Player.O)
        self.assertEqual(self.board.checkIfBoardIsFull(), True)

    def testHashValue(self):
        self.board.move(0, Player.X)
        self.assertIsInstance(self.board.hash(), int)

    def testCheckWinDrawEndCross(self):
        self.board.move(0, Player.O)
        self.board.move(3, Player.X)
        self.board.move(4, Player.O)
        self.board.move(8, Player.O)
        self.assertEqual(self.board.checkWinDrawEnd(8, Player.O.value), (True, 1))

    def testCheckWinDrawEndRow(self):
        self.board.move(0, Player.O)
        self.board.move(3, Player.X)
        self.board.move(1, Player.O)
        self.board.move(2, Player.O)
        self.assertEqual(self.board.checkWinDrawEnd(2, Player.O.value), (True, 1))

    def testCheckWinDrawEndCol(self):
        self.board.move(0, Player.O)
        self.board.move(2, Player.X)
        self.board.move(4, Player.O)
        self.board.move(8, Player.X)
        self.board.move(5, Player.X)
        self.assertEqual(self.board.checkWinDrawEnd(5, Player.X.value), (True, -1))
    
    def testCheckWinDrawEndDraw(self):
        self.board.move(0, Player.O)
        self.board.move(1, Player.X)
        self.board.move(2, Player.O)
        self.board.move(3, Player.X)
        self.board.move(4, Player.X)
        self.board.move(5, Player.O)
        self.board.move(6, Player.O)
        self.board.move(7, Player.O)
        self.board.move(8, Player.X)
        self.assertEqual(self.board.checkWinDrawEnd(8, Player.O.value), (True, 0))

    def testThisMustFail(self):
        self.board.move(0, Player.O)
        self.board.move(1, Player.O)
        self.board.move(2, Player.O)
        self.board.move(3, Player.O)
        self.board.move(4, Player.O)
        self.board.move(5, Player.X)
        self.board.move(6, Player.X)
        self.board.move(7, Player.X)
        self.board.move(8, Player.X)
        self.assertEqual(self.board.move(8, Player.X), StatusBoard.INVALID_MOVE)

# if __name__ == "__main__":
#     unittest.main()