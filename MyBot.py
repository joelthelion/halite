#!/usr/bin/python

""" Experiment with Q learning """

import random
import os
import logging
import sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping,ModelCheckpoint
from reinforce.hlt import NORTH, SOUTH, EAST, WEST, STILL, Move, Location
from reinforce.networking import getInit, sendFrame, sendInit, getFrame
from reinforce.train import get_new_model
from reinforce.reinforce import predict_for_pos, frame_to_stack, stack_to_input

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="play.log")

myID, gameMap = getInit()

model_file = "data/%s" % sorted([x for x in os.listdir("data") if x.endswith("h5")])[-1]
model = load_model(model_file)

sendInit('joelator')
logging.info("My ID: %s", myID)

while True:
    np.set_printoptions(precision=3)
    frame = getFrame()
    stack = frame_to_stack(frame, myID)
    positions = np.transpose(np.nonzero(stack[0]))
    moves = []
    for position in positions:
        area_inputs = stack_to_input(stack, position)
        possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, model)
        # Sample a move following Pi(s)
        index = np.argmax(Qs)
        logging.info("%d Qs: %s", index, Qs)
        moves.append((position[1], position[0], possible_moves[index]))


    sendFrame([Move(Location(px,py), move) for (px,py,move) in moves])
