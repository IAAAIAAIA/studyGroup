import os.path
import pickle
import random
import numpy as np

from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras import backend as K

from memory import Memory