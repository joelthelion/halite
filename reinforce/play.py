#!/usr/bin/python

""" Experiment with Q learning """

import os
import logging
from keras.models import load_model
import numpy as np
from networking import getInit, sendFrame, sendInit, getFrame
from hlt import NORTH, SOUTH, EAST, WEST, STILL, Move, Location
from reinforce import frame_to_stack, stack_to_input, one_hot

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.ERROR, filename="play.log")

myID, gameMap = getInit()

model_file = "data/%s" % sorted([x for x in os.listdir("data") if x.endswith("h5")])[-1]
model = load_model(model_file)

sendInit('joelator')
logging.info("My ID: %s", myID)

def predict_for_game(stack, positions, model):
    possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])
    inputs = np.vstack([np.vstack(
        [np.concatenate([stack_to_input(stack, position),one_hot(n, 5)] )
            for n in range(len(possible_moves))])
        for position in positions])
    outputs = np.split(model.predict(inputs), len(positions))
    # outputs /= sum(outputs)
    return possible_moves, np.split(inputs, len(positions)), [o.ravel() for o in outputs]

while True:
    np.set_printoptions(precision=3)
    frame = getFrame()
    stack = frame_to_stack(frame, myID)
    positions = np.transpose(np.nonzero(stack[0]))
    # position = random.choice(positions)
    moves = []
    possible_moves, allQinputs, allQs = predict_for_game(stack, positions, model)
    for position, Qinputs, Qs in zip(positions, allQinputs, allQs):
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
