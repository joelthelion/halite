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
from reinforce import predict_for_pos, frame_to_stack, stack_to_input

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="play.log")

myID, gameMap = getInit()

model = load_model("data/qmodel_4.h5")

sendInit('joelator')
logging.info("My ID: %s", myID)

while True:
    np.set_printoptions(precision=3)
    frame = getFrame()
    stack = frame_to_stack(frame, myID)
    positions = np.transpose(np.nonzero(stack[0]))
    # position = random.choice(positions)
    moves = []
    for position in positions:
        area_inputs = stack_to_input(stack, position)
        possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, model)
        # Sample a move following Pi(s)
        def softmax(x):
            """ Turn Q values into probabilities """
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0) # only difference
        def harden(x, e=2):
            exp = x**e
            return exp/exp.sum()
        Ps = harden(softmax(Qs.ravel()))
        # index = np.random.choice(range(len(possible_moves)), p=Ps)
        index = np.argmax(Ps)
        logging.info("%d Qs: %s Ps: %s", index, Qs, Ps)
        moves.append((position[1], position[0], possible_moves[index]))


    sendFrame([Move(Location(px,py), move) for (px,py,move) in moves])
