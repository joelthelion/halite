#!/usr/bin/python

""" Experiment with Q learning """

from keras.models import Sequential
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

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")

VISIBLE_DISTANCE = 4
neigh_input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
action_input_dim=5
input_dim = neigh_input_dim + action_input_dim

myID, gameMap = getInit()

# input: state + action. output: value at next turn
model = Sequential([Dense(512, input_dim=input_dim),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(512),
                    LeakyReLU(),
                    Dense(1, activation='sigmoid')])
model.compile('nadam','mse')
model.predict(np.zeros((2,input_dim))).shape # make sure model is compiled during init

# with open(os.devnull, 'w') as sys.stderr:
#     from keras.models import load_model
#     model = load_model('model.h5')


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

def predict_for_pos(input):
    possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])
    inputs = np.vstack([np.concatenate([input,one_hot(n, 5)]) for n in range(len(possible_moves))])
    outputs = model.predict(inputs)
    return outputs.argmax()

sendInit('joelator')

while True:
    stack = frame_to_stack(getFrame())
    positions = np.transpose(np.nonzero(stack[0]))
    output = [predict_for_pos(stack_to_input(stack, p)) for p in positions]
    sendFrame([Move(Location(positions[i][1],positions[i][0]), output[i]) for i in range(len(positions))])

    # output = model.predict(np.array([stack_to_input(stack, p) for p in positions]))
    # sendFrame([Move(Location(positions[i][1],positions[i][0]), output[i].argmax()) for i in range(len(positions))])