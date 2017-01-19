#!/usr/bin/env python

import numpy as np
import hlt
from value_model import Model
import pickle
from itertools import permutations

np.set_printoptions(threshold=1e6)

samples = np.load("value_samples.npz")
inputs = samples["inputs"][:,:-1] # remove turns for now
outputs = np.array(samples["outputs"]).ravel()
print(np.min(outputs))
print(np.max(outputs))
# print(set(outputs))
print(np.unique(outputs, return_counts=True))

print(inputs.shape)
print(inputs.nbytes*720/1e9)
perm_keep = 10
perm = np.array(list(permutations(range(6))))[:perm_keep]
input_perm = np.hstack([perm,perm+6,perm+12])
# inputs = np.vstack(i[input_perm] for i in inputs)
inputs = np.vstack(inputs[:,p] for p in input_perm)
# outputs = np.hstack(perm[o] for o in outputs) # doesn't work
def perm_outputs(out, p):
    perm_mapping=dict(zip(p,range(6)))
    return [perm_mapping[o] for o in out]
outputs = np.hstack(perm_outputs(outputs,p) for p in perm)
filenames = np.repeat(samples["filenames"], perm_keep)

from sklearn.model_selection import train_test_split
i_train, i_test, o_train, o_test = train_test_split(inputs,outputs,
        stratify=filenames, test_size=0.5)



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
print(np.argmax(model.predict(inputs)[:n_print],axis=1))
model._model.save_weights("value_weights.hd5")
