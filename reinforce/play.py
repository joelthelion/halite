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
from train import get_trained_model, get_new_model
from reinforce import predict_for_pos, frame_to_stack, stack_to_input, get_territory

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")

myID, gameMap = getInit()

model = get_trained_model()

sendInit('joelator')
logging.info("My ID: %s", myID)

while True:
    frame = getFrame()
    stack = frame_to_stack(frame, myID)
    position = random.choice(np.transpose(np.nonzero(stack[0])))
    area_inputs = stack_to_input(stack, position)
    possible_moves, Qinputs, Qs = predict_for_pos(area_inputs, model)
    # Sample a move following Pi(s)
    index = np.argmax(Qs)
    move = possible_moves[index]
    Q = Qs[index]
    Qinput = Qinputs[index]
    territory = get_territory(frame, myID)
    logging.info("%s a:%s Q:%.2f t:%.2f Qs:%s", position, move, Q, territory, Qs)

    sendFrame([Move(Location(position[1],position[0]), move)])
