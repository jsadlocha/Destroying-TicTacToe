from board import Board

MAX_INT = 2**64

class AlphaBetaPruning:
    def __init__(self):
        pass

    def __call__(self, board: list[int]) -> list[tuple[int, int]]:
        graph = board[:]
        solution = []
        for idx, elem in enumerate(graph):
            if elem == 0:
                solution.append((idx, self.dfs(graph, idx, -MAX_INT, MAX_INT)))
        return solution

    def dfs(self, graph: list[int], pos: int, alfa: int, beta: int) -> int:
        board = graph[:]
        player = 1 if board.count(0) % 2 else 2
        board[pos] = player
        
        end, score = Board.checkWinAndEnd(board, pos, player)
        if end:
            return score

        for idx, elem in enumerate(board):
            if elem == 0:
                score = self.dfs(board, idx, alfa, beta)
                if player == 2:
                    alfa = max(alfa, score)
                else:
                    beta = min(beta, score)
            
            if alfa > beta or alfa == beta:
                break
                    
        return alfa if player-1 else beta
