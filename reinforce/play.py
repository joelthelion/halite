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

def predict_for_pos(area_input, current_value):
    possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])
    inputs = np.vstack([np.concatenate([area_input,one_hot(n, 5), [current_value*10]])
        for n in range(len(possible_moves))])
    outputs = model.predict(inputs)
    outputs /= sum(outputs)
    def gamma(array, exp=2):
        temp = array ** exp
        return temp/temp.sum()
    # index = np.random.choice(range(len(possible_moves)), p=gamma(outputs.ravel()))
    index = np.argmax(outputs.ravel())
    return possible_moves[index], inputs[index].ravel(), outputs.ravel()[index]

def frame_to_value_input(frame, turn):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    value_input = np.zeros(19)
    value_input = [
        np.array([[np.sum(game_map[:,:,0] == p),
        np.sum((game_map[:,:,0] == p) * game_map[:,:,1])/10,
        np.sum((game_map[:,:,0] == p) * game_map[:,:,2])/255,
        turn]]) for p in range(1,7)]
    # logging.info("%s", value_input)
    return value_input

sendInit('joelator')
logging.info("My ID: %s", myID)

value_model = load_model("./reinforce/value_model.h5")

turn = 0
while True:
    frame = getFrame()
    stack = frame_to_stack(frame)
    moves = []
    for position in np.transpose(np.nonzero(stack[0])):
        state_values = value_model.predict(frame_to_value_input(frame, turn))[0]
        value = state_values[myID-1]
        # positions = np.transpose(np.nonzero(stack[0]))
        # Only pick one random position for easier q-learning
        area_inputs = stack_to_input(stack, position)
        output, Qinputs, Q = predict_for_pos(area_inputs, value)
        moves.append(Move(Location(position[1],position[0]), output))
        # logging.info("%s a:%s Q:%.2f V:%.2f (qinputs %s)", position, output, Q, value, Qinputs.ravel())
    sendFrame(moves)
    turn += 1
