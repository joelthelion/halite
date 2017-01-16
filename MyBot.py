""" My awesome RL bot """
import random
import os
import pickle
import logging
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")

possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])
myID, game_map = hlt.get_init()
hlt.send_init("MyPythonBot")

#Square = namedtuple('Square', 'x y owner strength production')

class Model(object):
    def __init__(self):
        input_length = 9
        hidden_length = 20
        self._model = Sequential([
            Dense(hidden_length, input_dim = input_length, init="uniform"),
            Activation('relu'),
            Dense(5, init="uniform"),
            Activation('softmax')
            ])
        self._model.compile(optimizer='rmsprop', loss='mse')
    def predict(self, input):
        return self._model.predict(input)
    def gen_moves(self, input):
        logging.info(input.shape)
        values = self.predict(input)
        logging.info(values)
        logging.info(np.argmax(values, axis=1))
        return possible_moves[np.argmax(values, axis=1)]

model = Model()
logging.info(model._model.get_weights())
while True:
    game_map.get_frame()
    def gen_input(square):
        """ Generate the next move """
        neighbors = list(game_map.neighbors(square))
        # logging.info(list(neighbors))
        my_values = [n.strength if n.owner==myID else 0 for n in neighbors]
        op_values = [n.strength if n.owner!=myID else 0 for n in neighbors]
        # productions =
        my_strength = [square.strength]
        model_input = np.array(my_values+op_values+my_strength)
        return model_input

    directions = model.gen_moves(np.vstack(gen_input(square) for square in game_map if square.owner == myID))
    moves = [Move(square, directions[n]) for n, square in enumerate(s for s in game_map if s.owner == myID)]
    logging.info(moves)
    hlt.send_frame(moves)
