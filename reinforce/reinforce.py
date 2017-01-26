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
from train import get_new_model

VISIBLE_DISTANCE = 4
neigh_input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
action_input_dim=5
input_dim = neigh_input_dim + action_input_dim + 1 # (1 for current state value)


def stack_to_input(stack, position):
    return np.take(np.take(stack,
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[0],axis=1,mode='wrap'),
                np.arange(-VISIBLE_DISTANCE,VISIBLE_DISTANCE + 1)+position[1],axis=2,mode='wrap').flatten()

def frame_to_stack(frame, myID):
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

def predict_for_pos(area_input, model):
    possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])
    inputs = np.vstack([np.concatenate([area_input,one_hot(n, 5)] )
        for n in range(len(possible_moves))])
    outputs = model.predict(inputs)
    outputs /= sum(outputs)
    return possible_moves, inputs, outputs.ravel()

def gamma(array, exp=2):
    temp = array ** exp
    return temp/temp.sum()

def get_territory(frame, player):
    return np.array([[x.owner==player for x in row] for row in frame.contents]).sum()

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")

if __name__ == '__main__':
    myID, gameMap = getInit()

    run_id = 0
    if len(sys.argv) > 1:
        run_id =  int(sys.argv[1])
        model = load_model("data/qmodel_%s.h5" % (run_id-1)) # load previous model
    else:
        model = get_new_model()

    sendInit('joelator')
    logging.info("My ID: %s", myID)

    turn = 0
    frame = getFrame()
    stack = frame_to_stack(frame, myID)
    # Only pick one random position for easier q-learning
    while True:
        position = random.choice(np.transpose(np.nonzero(stack[0])))
        area_inputs = stack_to_input(stack, position)
        possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, model)
        # Sample a move following Pi(s)
        index = np.random.choice(range(len(possible_moves)), p=Qs.ravel())
        move = possible_moves[index]
        Q = Qs[index]
        Qinput = Qinputs[index]
        sendFrame([Move(Location(position[1],position[0]), move)])
        turn += 1
        old_territory = get_territory(frame, myID)

        frame = getFrame()
        stack = frame_to_stack(frame, myID)
        area_inputs = stack_to_input(stack, position)
        possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, model)
        territory = get_territory(frame, myID)


        logging.info("%s a:%s Q:%.2f t+1:%.2f reward:%.2f maxQt+1:%.2f %s", position, move, Q, territory, territory - old_territory, max(Qs), Qs)
        with open("data/games_%s.csv" % run_id, "ab") as f:
            np.savetxt(f, np.hstack([Qinput, [territory-old_territory, max(Qs)]]), newline=" ")
            f.write(b"\n")

        # new position for next turn
        position = random.choice(np.transpose(np.nonzero(stack[0])))
