""" My awesome RL bot """
import random
import os
import pickle
import logging
import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move
import numpy as np
from model import Model

logging.basicConfig(format='%(asctime)-15s %(message)s',
        level=logging.INFO, filename="bot.log")



myID, game_map = hlt.get_init()
model = Model()
model._model.load_weights("weights.hd5")
model.predict(np.array([[0]*9])) # dummy computation to warm up model
logging.info(model._model.get_weights())
hlt.send_init("MyPythonBot")

#Square = namedtuple('Square', 'x y owner strength production')

while True:
    game_map.get_frame()

    directions = model.gen_moves(np.vstack(Model.gen_input(game_map, square)
        for square in game_map if square.owner == myID))
    moves = [Move(square, directions[n]) for n, square in enumerate(s for s in game_map if s.owner == myID)]
    logging.info(moves)
    hlt.send_frame(moves)
