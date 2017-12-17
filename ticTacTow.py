# the tic tac tow game
import numpy as np

class board():
    def __init__(self):
        self.connectLength = 3
        self.boardWidth = 3
        self.boardHeight = 3
        self.players = 2
        self.player1Piece = "X"
        self.player2Piece = "O"
        self.boards = np.array((self.boardWidth, self.boardHeight, self.players))
        self.defineWinFilters()
    
    def Play(self, player, location):
        if type(location) != tuple:
            return False
        
        if type(player) == str:
            if player == self.player1Piece:
                playerType = 1
            elif player == self.player2Piece:
                playerType = 2
        else:
            playerType = player
        
        if all(self.boards[location, :] == 0):
            self.boards[location, int(playerType)] = 1
            return True
        return False
    
    def CheckWin(self):
        # convolve about the board with win filters

        
    def defineWinFilters(self):
        if self.connectLength <= self.boardWidth and self.connectLength <= self.boardHeight:
            self.filters = []
            # row filter
            self.filters.append(np.ones(self.connectLength, 1))
            # col filter
            self.filters.append(np.ones(1, self.connectLength))
            # top-left diagonal
            box = np.zeros(self.connectLength, self.connectLength)
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

