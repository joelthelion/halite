import logging
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

from hlt import NORTH, EAST, SOUTH, WEST, STILL
possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])

class Model(object):
    def __init__(self):
        input_length = 14
        hidden_length = 20
        self._model = Sequential([
            Dense(hidden_length, input_dim = input_length, init="uniform"),
            Activation('relu'),
            Dense(10, init="uniform"),
            Activation('relu'),
            Dense(5, init="uniform"),
            Activation('softmax')
            ])
        # self._model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        self._model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
        self._model.predict(np.array([[0]*input_length])) # dummy computation to warm up model
    def predict(self, input):
        return self._model.predict(input)
    def gen_moves(self, input):
        logging.info(input.shape)
        values = self.predict(input)
        logging.info(values)
        # moves = [np.random.choice(possible_moves, p=val) for val in values]
        # def gamma(array, exp=2):
        #     temp = array ** exp
        #     return temp/temp.sum()
        # moves = [np.random.choice(possible_moves, p=gamma(val)) for val in values]
        moves = [np.random.choice(possible_moves, p=val) for val in values]
        # moves = np.argmax(values, axis=1)
        logging.info(moves)
        return moves
    def gen_input(game_map, square):
        """ Generate the next move """
        neighbors = list(game_map.neighbors(square, n=1))
        # logging.info(list(neighbors))
        my_values = [n.strength if n.owner==square.owner else 0 for n in neighbors]
        op_values = [n.strength if n.owner!=square.owner else 0 for n in neighbors]
        productions = [n.production for n in neighbors]
        my_strength = [square.strength]
        my_prod =     [square.production]
        model_input = np.array(my_values+op_values+productions+my_strength+my_prod)
        return model_input

