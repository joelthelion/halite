""" My awesome RL bot """
import random
import os
import pickle
import logging
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move
import numpy as np

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")

possible_moves = [NORTH, EAST, SOUTH, WEST, STILL]
myID, game_map = hlt.get_init()
hlt.send_init("MyPythonBot")

def softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

#Square = namedtuple('Square', 'x y owner strength production')

class Model(object):
    def __init__(self):
        if os.path.exists("weights.pck"):
            logging.info("Reading weights from weights.pck")
            self.hidden, self.output = pickle.load(open("weights.pck","rb"))
        else:
            logging.info("Generating random weights...")
            input_length = 9
            hidden_length = 20
            self.hidden = np.random.normal(size=(input_length,  hidden_length))
            self.hidden /= len(self.hidden)
            self.output = np.random.normal(size=(hidden_length, 5))
            self.output /= len(self.output)
            pickle.dump((self.hidden, self.output), open("weights.pck", "wb"))
    def relu(self, vector):
        return np.maximum(vector,0,vector)
    def predict(self, input):
        return softmax(self.relu(input @ self.hidden) @ self.output)

model = Model()
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
        logging.info(model.predict(model_input))
        return model_input

    def gen_move(square):
        if square.strength > 30:
            return random.choice((NORTH, EAST, SOUTH, WEST, STILL))
        else:
            return STILL
    # inputs = np.vstack(gen_input(square) for square in game_map if square.owner == myID)
    # logging.info(model.predict(inputs))
    #moves = [Move(square, random.choice((NORTH, EAST, SOUTH, WEST, STILL))) for square in game_map if square.owner == myID]
    #moves = [Move(square, gen_move(square)) for square in game_map if square.owner == myID]
    moves = [Move(square, np.argmax(model.predict(gen_input(square)))) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)
