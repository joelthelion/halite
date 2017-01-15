""" My awesome RL bot """
import random
import logging
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy


def get_model():
    model = Sequential(
            [Dense(20, input_dim=9),
            Activation("relu"),
            Dense(5),
            Activation("softmax")]
            )
    model.compile(optimizer='rmsprop',
                          loss='mse')
    return model

myID, game_map = hlt.get_init()
hlt.send_init("MyPythonBot")

#Square = namedtuple('Square', 'x y owner strength production')

while True:
    game_map.get_frame()
    # logging.basicConfig()
    logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")
    model = get_model()
    def gen_input(square):
        """ Generate the next move """
        neighbors = list(game_map.neighbors(square))
        # logging.info(list(neighbors))
        my_values = [n.strength if n.owner==myID else 0 for n in neighbors]
        op_values = [n.strength if n.owner!=myID else 0 for n in neighbors]
        # productions =
        my_strength = [square.strength]
        model_input = numpy.array(my_values+op_values+my_strength)
        return model_input

    def gen_move(square):
        if square.strength > 30:
            return random.choice((NORTH, EAST, SOUTH, WEST, STILL))
        else:
            return STILL
    inputs = numpy.vstack(gen_input(square) for square in game_map if square.owner == myID)
    # logging.info(inputs)
    logging.info(model.predict(inputs))
    #moves = [Move(square, random.choice((NORTH, EAST, SOUTH, WEST, STILL))) for square in game_map if square.owner == myID]
    moves = [Move(square, gen_move(square)) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)
