#!/usr/bin/env python

import numpy as np
import hlt
from value_model import Model
import pickle

np.set_printoptions(threshold=1e6)

samples = np.load("value_samples.npz")
inputs = samples["inputs"]
outputs = np.array(samples["outputs"]).ravel()
print(np.min(outputs))
print(np.max(outputs))
print(set(outputs))
# outputs = np.random.randint(0,6,size=outputs.shape)
model = Model()

n_print=45

print(inputs[:n_print])
model._model.fit(inputs, outputs, verbose=True, validation_split=0.5, nb_epoch = 5)
print(model.predict(inputs)[:n_print])
print(outputs[:n_print])
print(np.argmax(model.predict(inputs)[:n_print],axis=1))
model._model.save_weights("value_weights.hd5")
