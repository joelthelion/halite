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
                        Dense(512),
                        LeakyReLU(),
                        Dense(512),
                        LeakyReLU(),
                        Dense(1, activation='sigmoid')])
    model.compile('nadam','mse')
    model.predict(np.zeros((2,input_dim))).shape # make sure model is compiled during init
    return model

def get_trained_model():
    return load_model("q_model.h5")

def train(model):
    with open("games.csv") as f:
        data = np.loadtxt(f)
    # with open("games2.csv") as f:
    #     data = np.vstack([data, np.loadtxt(f)])
    print(data.shape)
    columns = data.shape[1]
    inputs, reward, maxQ1 = np.hsplit(data,[columns-2,columns-1])
    outputs = reward + 0.9*maxQ1 # 0.9 discount
    print(outputs.shape)
    model.fit(inputs, outputs, validation_split=0.5, nb_epoch = 10,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint('q_model.h5',verbose=1,save_best_only=True)]
            )

if __name__ == '__main__':
    model = get_new_model()
    train(model)
