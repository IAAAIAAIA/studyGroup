# Reinforcement learning using unreal 

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.autograd import Variable
import torchvision.transforms as T


env = gym.make("LunarLander-v2").unwrapped


# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Test to see if it still runs
env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
