# Basic example of the Temporal Difference
import numpy as np
import random as rand

numberOfEpisodes = 10000
numberOfStates = 6

# elegibility table
e = np.array([0 for _ in range(numberOfStates)])
# Value Table
V = np.array([rand.random() * 0 for _ in range(numberOfStates)])

# _lambda is from the TD(lambda)
_lambda = 0.7

# gama is the learning rate sortof
gama = 0.1

# alpha is a function of episode
def alpha(episode):
    return 1/(episode+1)

# The actual game
def game(state):
    # this game has a predetermined action and so it does not need to be passed in
    reward = 0
    nextState = 0
    if state == 1:
        reward = 1
        nextState = 3
    elif state == 2:
        reward = 2
        nextState = 3
    elif state == 3:
        reward = 0
        if rand.random() < 0.9:
            nextState = 4
        else:
            nextState = 5
    elif state == 4:
        reward = 1
        nextState = 0
    elif state == 5:
        reward = 10
        nextState = 0
    else:
        # either not a valid state or have finished the game (state final)
        pass
    return reward, nextState

# Main Update loop for each episode
for T in range(numberOfEpisodes):
    # reset the e to zeros
    for s, e_s in enumerate(e):
        e_s = 0
    # create the initial state
    if rand.random() > 0.5:
        curentState = 1
    else:
        curentState = 2
    # set the total rewards to 0
    totalReward = 0
    # run a whole episode
    while curentState != 0:
        # update the e before taking the step
        e[curentState] += 1
        previousState = curentState
        # take the step
        currentReward, curentState = game(curentState)
        # update total reward
        totalReward += currentReward
        # update the V and e for all s
        for s in range(numberOfStates):
            V[s] = V[s] + alpha(T)*(currentReward + gama * V[curentState] - V[previousState])
            e[s] = _lambda * gama * e[s]
        
# Normalize the Values
V = V / np.sum(V)

# Print it out
print("The final evaluations for the Value Table:")
for s, V_s in enumerate(V):
    print(s, game(s)[0], V_s)
# print("")




