import pygame
from pygame import display
from board import Board, StatusBoard, Player
from min_max import MinMax

pygame.init()

screen = pygame.display.set_mode((800, 700), pygame.SHOWN)
pygame.display.set_caption('TicTacToe')
clock = pygame.time.Clock()

pygame.font.init()
comic_sans = pygame.font.SysFont('Comic Sans MS', 35)
comic_sans2 = pygame.font.SysFont('Comic Sans MS', 15)
comic_sans3 = pygame.font.SysFont('Comic Sans MS', 35)

winning_O = comic_sans.render('Player O Win the Game', False, (0, 255, 0))
winning_X = comic_sans.render('Player X Win the Game', False, (0, 255, 0))
winning_draw = comic_sans.render('Nobody Win the Game', False, (0, 255, 0))
winning_text_list = [winning_O, winning_X, winning_draw]

engine_win = comic_sans2.render("100% Win", False, (0, 255, 0))
engine_lose = comic_sans2.render("100% Lose", False, (255, 0, 0))
engine_draw = comic_sans2.render("100% Draw", False, (255, 200, 0))

engine_list_O = [engine_draw, engine_win, engine_lose]
engine_list_X = [engine_draw, engine_lose, engine_win]

player_o = comic_sans3.render("Player O move", False, (0, 0, 0))
player_x = comic_sans3.render("Player X move", False, (0, 0, 0))

reset_text = comic_sans.render('Restart', False, (0, 0, 0))
quit_text = comic_sans.render('Quit', False, (0, 0, 0))

# img
background = pygame.image.load("img/board.png")
circle = pygame.image.load("img/o.png")
cross = pygame.image.load("img/x.png")

menu_ending_layout = pygame.Rect(240, 250, 430, 300)
button_reset = pygame.Rect(400, 370, 140, 50)
button_quit = pygame.Rect(400, 440, 140, 50)

size = (260, 225)

pos = [(0,0), (275, 0), (545, 0), 
        (0, 233), (275, 233), (545, 233),
        (0, 465), (275, 465), (545, 465)
        ]

class Button():
    def __init__(self, idx, position, size):
        self.idx = idx
        self.rect = pygame.Rect(*position, *size)

    def isClicked(self):
        action = False
        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0]:
                action = True

        return action 

gameboard = Board()
button_list = [Button(idx, position, size) for idx, position in enumerate(pos)]
gameover_button_list = [Button(0, button_reset, []), Button(1, button_quit, [])]

onMove = [Player.O, Player.X]
game = True
win = 0
engine = MinMax()


def drawBoard(screen, board):
    screen.blit(background, pygame.rect.Rect(0, 0, 800, 700))

    for idx, player in enumerate(board()):
        if player == Player.O.value:
            screen.blit(circle, pos[idx])
        elif player == Player.X.value:
            screen.blit(cross, pos[idx])

def gameOverMenu(screen, board, button_list):
    global win
    pygame.draw.rect(screen, (128, 128, 128), menu_ending_layout)
    screen.blit(winning_text_list[win-1], (270, 300))
    pygame.draw.rect(screen, (0, 255, 0), button_reset)
    screen.blit(reset_text, (410, 370))
    pygame.draw.rect(screen, (255, 0, 0), button_quit)
    screen.blit(quit_text, (430, 440))

    for button in button_list:
        if button.isClicked():
            if button.idx == 0:
                board.reset()  
                global calc
                calc = True             
                win = 0
                global onMove
                onMove = [Player.O, Player.X]
            elif button.idx == 1:
                global game
                game = False

engine_result = []
calc = True
while game:
    drawBoard(screen, gameboard)
    player_turn = player_o if onMove[1].value-1 else player_x
    screen.blit(player_turn, (300,0))

    if calc:
        engine_result = []
        res = engine.get_lookup_table(gameboard[:])
        for idx, score in res:
            x, y = pos[idx]
            x += 180
            y += 200
            tmp = engine_list_O if onMove[1].value-1 else engine_list_X
            engine_result.append([tmp[score], (x,y)])
        calc = False
        # removing hanging keys, during long calculations
        pygame.event.get()
    
    for res in engine_result:
        screen.blit(*res)
    
    if win == 0:
        for button in button_list:
            if button.isClicked() is True:
                if gameboard.move(button.idx, onMove[0]) == StatusBoard.OK:
                    onMove = onMove[::-1]
                    calc = True

                    if gameboard.checkWin(button.idx) == StatusBoard.WIN:
                        win = onMove[1].value
                    else:
                        if gameboard.checkIfBoardIsFull():
                            win = 3

    else:
        gameOverMenu(screen, gameboard, gameover_button_list)

    pygame.display.update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game = False 
            
    clock.tick(30)

pygame.quit()
    
    
