# The deep q-network for running the ticTacTow game

import numpy as np
import random
from ticTacTow import board
from collections import namedtuple
from keras.models import Sequential
from keras import metrics, losses, optimizers
from keras.layers import Conv2D, Dense, Flatten

Transition = namedtuple('Transition', ('state', 'action', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        # Saves a transition
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class alphaModel(object):
    def __init__(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(3, 3, 1)))
        self.model.add(Conv2D(32, (3, 3), activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(9, activation='tanh'))
        self.model.compile(optimizer=optimizers.Adam, loss=losses.mean_squared_error, metrics=[metrics.mean_absolute_error])

class policy(object):
    def __init__(self):
        self.policy = 0
        self.explorationRate = 0.9

    def Sample(self, model, state):
        if random.random() < self.explorationRate:
            return np.random.rand(9)
        else:
            return model.predict(state, batch_size=1)
        

# to get the state use GetBoard() from the game
# actions are represented as tuples for the location
alpha = alphaModel()
def train():
    batch_size = 32
    memory = ReplayMemory()
    game = board()
    py = policy()
    num_of_episodes = 1000
    for episode in range(num_of_episodes):
        game.Reset()
        currentReward, done = game.GetReward()
        episodeMemory = []
        sumReward = 0
        turn = 0
        while(not done):
            currentReward, done = game.GetReward()
            currentState = game.GetBoard()
            valids = game.GetValidMoves()
            predicts = np.where(valids, py.Sample(alpha, currentState), np.zeros((3,3)))
            if turn == 0:
                actionPick = np.argmax(predicts)
            else:
                actionPick = np.argmin(predicts)
            action = np.unravel_index(actionPick, (3,3))
            turn = game.TakeTurn(action)
            actionLable = np.zeros((3,3))
            actionLable[action] = 1
            if turn != 0:
                actionLable *= -1
            # record to the Replay memory
            episodeMemory.append(Transition(currentState, actionLable, currentReward))
            sumReward += currentReward
        game.Print()
        for i, mem in enumerate(episodeMemory):
            memory.push(mem.state, mem.action, sumReward)
        # train
        if len(memory) >= batch_size:
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))
            alpha.model.fit(batch.state, batch.action*batch.reward)

def play():
    game = board()
    r, done = game.GetReward()
    while not done:
        predict = alpha.model.predict(game.GetBoard(), batch_size=1)
        game.TakeTurn(np.unravel_index(np.argmax(predict), (3,3)))
        game.Print()
        win = game.CheckWin()
        if win != False:
            print("Winner:", win)
            return
        action = input("player input (1-9, left to right, top to bottom):")
        game.TakeTurn(action)
        game.Print()
        win = game.CheckWin()
        if win != False:
            print("Winner:", win)
            return

if __name__ == '__main__':
    # train()
    play()