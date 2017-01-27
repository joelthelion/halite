#!/usr/bin/python

""" Experiment with Q learning """

import random
import math
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
from train import get_new_model, VISIBLE_DISTANCE


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
    # outputs /= sum(outputs)
    return possible_moves, inputs, outputs.ravel()

def gamma(array, exp=2):
    temp = array ** exp
    return temp/temp.sum()

def get_reward(frame, player):
    game_map = np.array([[(x.owner, x.production, x.strength) for x in row] for row in frame.contents])
    my_territory = game_map[...,0]==player
    territory  = my_territory.sum()
    production = game_map[my_territory,1].sum()
    strength   = game_map[my_territory,2].sum()
    logging.info("t: %d p:%d s:%d", territory, production, strength)
    assert territory == np.array([[x.owner==player for x in row] for row in frame.contents]).sum()
    return territory + 0.005 * strength


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)-15s %(message)s',
            level=logging.INFO, filename="bot.log")
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
        def softmax(x):
            """ Turn Q values into probabilities """
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) # only difference
        index = np.random.choice(range(len(possible_moves)), p=softmax(Qs.ravel()))
        move = possible_moves[index]
        Q = Qs[index]
        Qinput = Qinputs[index]
        sendFrame([Move(Location(position[1],position[0]), move)])
        turn += 1
        old_reward = get_reward(frame, myID)

        frame = getFrame()
        stack = frame_to_stack(frame, myID)
        area_inputs = stack_to_input(stack, position)
        possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, model)
        reward = get_reward(frame, myID)


        logging.info("%s a:%s Q:%.2f V+1:%.2f reward:%.2f maxQt+1:%.2f %s", position, move, Q, reward, reward - old_reward, max(Qs), Qs)
        with open("data/games_%s.csv" % run_id, "ab") as f:
            np.savetxt(f, np.hstack([Qinput, [reward-old_reward, max(Qs)]]), newline=" ")
            f.write(b"\n")

        # new position for next turn
        position = random.choice(np.transpose(np.nonzero(stack[0])))
