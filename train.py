#!/usr/bin/env python

import numpy as np
import hlt
from model import Model
import pickle
from keras.callbacks import ProgbarLogger


samples = np.load("samples.npz")
inputs = samples["inputs"]
moves = samples["outputs"]
model = Model()
# moves = np.array(moves[:100]).reshape((100,1))
# inputs = np.vstack(inputs[:100])
moves = np.expand_dims(np.array(moves), -1)
inputs = np.vstack(inputs)
print(moves)
print(inputs)
model._model.fit(inputs, moves, verbose=True, validation_split=0.5, nb_epoch = 10)
print(model.predict(inputs))
model._model.save_weights("weights.hd5")
