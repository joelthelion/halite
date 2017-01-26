#!/usr/bin/python

""" Experiment with Q learning (training) """

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
import sys


def get_new_model():
    VISIBLE_DISTANCE = 4
    neigh_input_dim=4*(2*VISIBLE_DISTANCE+1)*(2*VISIBLE_DISTANCE+1)
    action_input_dim=5
    input_dim = neigh_input_dim + action_input_dim


    # input: state + action. output: value at next turn
    model = Sequential([Dense(512, input_dim=input_dim),
                        LeakyReLU(),
                        BatchNormalization(),
                        Dense(512),
                        LeakyReLU(),
                        BatchNormalization(),
                        Dense(512),
                        LeakyReLU(),
                        BatchNormalization(),
                        # Dense(1)]) # linear activation
                        Dense(1, activation='sigmoid')])
    model.compile('nadam','mse')
    model.predict(np.zeros((2,input_dim))).shape # make sure model is compiled during init
    return model

def train(model, run_id):
    with open("data/games_%s.csv" % run_id) as f:
        data = np.loadtxt(f)
    # with open("games2.csv") as f:
    #     data = np.vstack([data, np.loadtxt(f)])
    print(data.shape)
    columns = data.shape[1]
    inputs, reward, maxQ1 = np.hsplit(data,[columns-2,columns-1])
    reward = (reward + 1) / 2.
    outputs = reward
    # outputs = reward + 0.9*maxQ1 # 0.9 discount
    print(outputs)
    print(outputs.shape)
    model.fit(inputs, outputs, validation_split=0.5, nb_epoch = 10,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint('data/qmodel_%s.h5'%run_id,verbose=1,save_best_only=True)]
            )

if __name__ == '__main__':
    if len(sys.argv) < 2:
        model = get_new_model()
        run_id = 0
    else:
        run_id = int(sys.argv[1])
        model = load_model("data/qmodel_%s.h5" % (run_id-1))
    train(model, run_id)
