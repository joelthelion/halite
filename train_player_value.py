#!/usr/bin/env python

import random
import numpy as np
import hlt
from player_value_model import Model

np.set_printoptions(threshold=1e6)

samples = np.load("./player_values.npz")["arr_0"]
inputs = samples[:,(0,1,2,4)] # 4 is turn
outputs = samples[:,4] == samples[:,5] # did the player win?
filenames = samples[:,-1]

fn = np.unique(filenames)
np.random.shuffle(fn)
train_fn  = fn[:len(fn)//2]
train_idx = np.where(np.in1d(filenames,train_fn))
np.random.shuffle(train_idx)
test_idx  = np.where(np.logical_not(np.in1d(filenames,train_fn)))
np.random.shuffle(test_idx)
i_train   = inputs[train_idx]
o_train   = outputs[train_idx]
i_test    = inputs[test_idx]
o_test    = outputs[test_idx]


print(inputs.shape)
print(inputs[:10])
# qsdf


# outputs = np.random.randint(0,6,size=outputs.shape)
model = Model()

n_print=45

print(inputs[:n_print])
# model._model.fit(inputs, outputs, verbose=True, validation_split=0.5, nb_epoch = 5)
model._model.fit(i_train, o_train, verbose=True, validation_data=(i_test,o_test), nb_epoch = 5)
# print(model.predict(inputs)[:n_print])
print(outputs[:n_print])
print(model.predict(inputs)[:n_print])
model._model.save_weights("value_weights.hd5")
