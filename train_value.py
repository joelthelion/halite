#!/usr/bin/env python

import numpy as np
import hlt
from value_model import Model
import pickle


samples = np.load("value_samples.npz")
inputs = samples["inputs"]
outputs = np.array(samples["outputs"]).ravel()
model = Model()

print(outputs)
print(inputs)
model._model.fit(inputs, outputs, verbose=True, validation_split=0.2, nb_epoch = 10)
print(model.predict(inputs))
model._model.save_weights("value_weights.hd5")
