# the tic tac tow game
import numpy as np
from scipy import signal

class board(object):
    def __init__(self):
        self.connectLength = 3
        self.boardWidth = 3
        self.boardHeight = 3
        self.players = 2
        self.playerPieces = ["X", "O"]
        self.boards = np.zeros((self.boardWidth, self.boardHeight, self.players))
        self.defineWinFilters()
    
    def Play(self, player, location):
        if type(location) != tuple:
            return False
        
        if type(player) == str:
            # handles the case for string implementations
            for i, piece in enumerate(self.playerPieces):
                if player == piece:
                    playerType = i
        else:
            # handles the case for int pieces implementation
            playerType = player
        # print(self.boards)
        if all(self.boards[location[0], location[1], :] == 0):
            self.boards[location[0], location[1], int(playerType)] = 1
            return True
        return False
    
    def CheckWin(self):
        # convolve about the board with win filters
        for i, piece in enumerate(self.playerPieces):
            for j, convFilter in enumerate(self.filters):
                conv = signal.convolve2d(self.boards[:,:,i], convFilter, 'valid')
                if np.any(conv >= self.connectLength):
                    return piece
        return False
        
    def defineWinFilters(self):
        if self.connectLength <= self.boardWidth and self.connectLength <= self.boardHeight:
            self.filters = []
            # row filter
            self.filters.append(np.ones((self.connectLength, 1)))
            # col filter
            self.filters.append(np.ones((1, self.connectLength)))
            # top-left diagonal
            box = np.zeros((self.connectLength, self.connectLength))
            for i in range(self.connectLength):
                box[i, i] = 1
            self.filters.append(box)
            # top-right diagonal
            box = np.zeros_like(box)
            for i in range(self.connectLength):
                box[self.connectLength - i -1, i] = 1
            self.filters.append(box)
        else:
            raise ValueError('ConnectLength is larger than board size.')
    
    def Print(self):
        for i in range(self.boardWidth):
            for j in range(self.boardHeight):
                if any(self.boards[i,j,:] != 0):
                    for k in range(self.players):
                        if self.boards[i,j,k] != 0:
                            print(self.playerPieces[k], end='')
                else:
                    print(" ", end='')
                if j != self.boardHeight-1:
                    print(" | ", end='')
                else:
                    print()
            if i != self.boardWidth-1:
                print("---------")

def main():
    game = board()
    game.Play("X", (0,0))
    game.Play("X", (1,1))
    game.Play("X", (1,2))
    game.Play("O", (1,0))
    game.Play("O", (2,0))
    game.Play("O", (2,2))
    game.Play("O", (2,1))
    result = game.CheckWin()
    print("winner:", result)
    game.Print()

if __name__ == '__main__':
    main()