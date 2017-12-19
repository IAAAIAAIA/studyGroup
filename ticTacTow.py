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

    def Reset(self):
        self.boards = np.zeros((self.boardWidth, self.boardHeight, self.players))

    def GetValidMoves(self):
        valids = np.zeros((3,3), dtype=bool)
        for i in range(self.boardWidth):
            for j in range(self.boardHeight):
                if all(self.boards[i,j,:] == 0):
                    valids[i,j] = True
        return valids

    def GetReward(self):
        """ Returns the current reward at the current board
            and if the game has ended or not
        """
        check = self.CheckWin()
        if check == False:
            if len(self.GetValidMoves()) == 0:
                return 0, True
            else:
                return 0, False
        if check == self.playerPieces[0]:
            return 2, True
        else:
            return -1, True

    def GetBoard(self, swapPlayer=1):
        """ Returns the board in a numpy array
            0 = empty
            1 = player 1
            2 = player 2 
            etc.

            swapPlayer argument is what player 1 should swap to i.e.
            swapPlayer = 2 means that players 1 and 2 swap
            swapPlayer = 3 means that players 1 and 3 swap
            default = 1
            swapPlayer cannot be <= 0
        """
        if swapPlayer <= 0:
            swapPlayer = 1
        collapsedBoard = np.zeros((self.boardWidth, self.boardHeight))
        for player in range(self.players):
            if player == 0:
                collapsedBoard = collapsedBoard + (swapPlayer) * self.boards[:,:,player]
            elif (swapPlayer-1) == player:
                collapsedBoard = collapsedBoard + (1) * self.boards[:,:,player]
            else:
                collapsedBoard = collapsedBoard + (player+1) * self.boards[:,:,player]
        return collapsedBoard

    def GetBoardRaw(self):
        return self.boards

    def TakeTurn(self, location):
        if self.turn == None:
            self.turn = 0
        self.Play(self.turn, location)
        self.turn = (self.turn + 1) % len(self.playerPieces)
        return self.turn
    
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
                print("-" * (4 * self.boardWidth - 3))

def main():
    game = board()
    game.Play("X", (0,0))
    game.Play("X", (1,1))
    game.Play("X", (1,2))
    game.Play("O", (1,0))
    game.Play("O", (2,0))
    game.Play("O", (2,2))
    game.Play("X", (2,1))
    result = game.CheckWin()
    print("winner:", result)
    game.Print()
    print(game.GetBoard())

if __name__ == '__main__':
    main()