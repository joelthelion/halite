import logging
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np

from hlt import NORTH, EAST, SOUTH, WEST, STILL
possible_moves = np.array([NORTH, EAST, SOUTH, WEST, STILL])

class Model(object):
    def __init__(self):
        input_length = 9
        hidden_length = 20
        self._model = Sequential([
            Dense(hidden_length, input_dim = input_length, init="uniform"),
            Activation('relu'),
            Dense(10, init="uniform"),
            Activation('relu'),
            Dense(5, init="uniform"),
            Activation('softmax')
            ])
        self._model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    def predict(self, input):
        return self._model.predict(input)
    def gen_moves(self, input):
        logging.info(input.shape)
        values = self.predict(input)
        logging.info(values)
        moves = [np.random.choice(possible_moves, p=val) for val in values]
        logging.info(moves)
        return moves
    def gen_input(game_map, square):
        """ Generate the next move """
        neighbors = list(game_map.neighbors(square))
        # logging.info(list(neighbors))
        my_values = [n.strength if n.owner==square.owner else 0 for n in neighbors]
        op_values = [n.strength if n.owner!=square.owner else 0 for n in neighbors]
        # productions =
        my_strength = [square.strength]
        model_input = np.array(my_values+op_values+my_strength)
        return model_input

