""" My awesome RL bot """
import sys
import random
import logging
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move


myID, game_map = hlt.get_init()
hlt.send_init("MyPythonBot")

#Square = namedtuple('Square', 'x y owner strength production')

while True:
    game_map.get_frame()
    # logging.basicConfig(format='%(asctime)-15s %(clientip)s %(user)-8s %(message)s')
    logging.basicConfig(level=logging.INFO, filename="bot.log")
    def gen_move(square):
        """ Generate the next move """
        neighbors = game_map.neighbors(square)
        my_values = [n.strength if n.owner==myID else 0 for n in neighbors]
        op_values = [n.strength if n.owner!=myID else 0 for n in neighbors]
        # productions =
        my_strength = [square.strength]
        model_input = my_values+op_values+my_strength
        logging.info(model_input)
        if square.strength > 30:
            return random.choice((NORTH, EAST, SOUTH, WEST, STILL))
        else:
            return STILL
    #moves = [Move(square, random.choice((NORTH, EAST, SOUTH, WEST, STILL))) for square in game_map if square.owner == myID]
    moves = [Move(square, gen_move(square)) for square in game_map if square.owner == myID]
    hlt.send_frame(moves)
