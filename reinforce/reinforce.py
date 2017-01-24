#!/usr/bin/python

""" Experiment with Q learning """

import random
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint
import numpy as np
from networking import getInit, sendFrame, sendInit, getFrame
from hlt import NORTH, SOUTH, EAST, WEST, STILL, Move, Location
import logging
import sys
from train import get_trained_model

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")

VISIBLE_DISTANCE = 4
neigh_input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
action_input_dim=5
input_dim = neigh_input_dim + action_input_dim + 1 # (1 for current state value)

myID, gameMap = getInit()

model = get_trained_model()

def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

def frame_to_stack(frame):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    return np.array([(game_map[:, :, 0] == myID),  # 0 : owner is me
                      ((game_map[:, :, 0] != 0) & (game_map[:, :, 0] != myID)),  # 1 : owner is enemy
                      game_map[:, :, 1]/20,  # 2 : production
                      game_map[:, :, 2]/255,  # 3 : strength
                      ]).astype(np.float32)
def one_hot(i,N):
    a = np.zeros(N, 'uint8')
    a[i] = 1
    return a

def predict_for_pos(area_input, territory):
    possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])
    inputs = np.vstack([np.concatenate([area_input,one_hot(n, 5), [territory]] )
        for n in range(len(possible_moves))])
    outputs = model.predict(inputs)
    outputs /= sum(outputs)
    return possible_moves, inputs.ravel(), outputs.ravel()

def gamma(array, exp=2):
    temp = array ** exp
    return temp/temp.sum()

def get_territory(frame, player):
    return np.array([[x.owner==player for x in row] for row in frame.contents]).sum()

sendInit('joelator')
logging.info("My ID: %s", myID)

turn = 0
frame = getFrame()
stack = frame_to_stack(frame)
# Only pick one random position for easier q-learning
territory = get_territory(frame, myID)
while True:
    position = random.choice(np.transpose(np.nonzero(stack[0])))
    area_inputs = stack_to_input(stack, position)
    possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, territory)
    # Sample a move following Pi(s)
    index = np.random.choice(range(len(possible_moves)), p=Qs.ravel())
    move = possible_moves[index]
    Q = Qs[index]
    Qinput = Qinputs[index]
    # logging.info("%s a:%s Q:%.2f V:%.2f (qinputs %s)", position, move, Q, territory, Qinput.ravel())
    sendFrame([Move(Location(position[1],position[0]), move)])
    turn += 1
    old_territory = territory

    frame = getFrame()
    stack = frame_to_stack(frame)
    area_inputs = stack_to_input(stack, position)
    possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, territory)
    territory = get_territory(frame, myID)


    logging.info("%s a:%s Q:%.2f t+1:%.2f reward:%.2f maxQt+1:%.2f", position, move, Q, territory, territory - old_territory, max(Qs))
    with open("games.csv", "ab") as f:
        np.savetxt(f, np.hstack([Qinput, [territory-old_territory, max(Qs)]]), newline=" ")
        f.write(b"\n")

    # new position for next turn
    position = random.choice(np.transpose(np.nonzero(stack[0])))
