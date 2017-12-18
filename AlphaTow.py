# The deep q-network for running the ticTacTow game

import numpy as np
import random
from ticTacTow import board
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'nextState'))

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

# to get the state use GetBoard() from the game
# actions are represented as tuples for the location
