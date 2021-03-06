#!/usr/bin/env python

import random
import pickle
from itertools import permutations
import numpy as np
import hlt
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Dropout, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Model
player_input_length = 4
player_inputs = [Input(shape=(player_input_length,)) for p in range(6)]
single_player_model = Sequential([
    Dense(20, input_dim = player_input_length),
    Activation('relu'),
    BatchNormalization(),
    Dense(10),
    Activation('relu'),
    BatchNormalization(),
    Dense(1),
    Activation('sigmoid')])
single_player_values = [single_player_model(pi) for pi in player_inputs]
merged_vector = merge(single_player_values, mode="concat", concat_axis=-1)
predictions = Dense(6, activation="softmax")(merged_vector)
model = Model(input=player_inputs, output=predictions)
optimizer = RMSprop()
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.predict([np.zeros((1,player_input_length)) for p in range(6)]) # dummy computation to warm up model

print("gloupi")

# Samples
np.set_printoptions(threshold=1e6)
samples = np.load("value_samples.npz")
# inputs = samples["inputs"][:,:-1] # remove turns for now
inputs = samples["inputs"]
raw_inputs = inputs
inputs[...,6:12] /= 255. #normalize strength
inputs[...,12:18] /= 10. # normalize production
outputs = np.array(samples["outputs"]).ravel()
raw_outputs = outputs
filenames = samples["filenames"]


print(np.min(outputs))
print(np.max(outputs))
# print(set(outputs))
print(np.unique(outputs, return_counts=True))
print(inputs.shape)
print(inputs.nbytes*720/1e9)
perm_keep = 30
perm = list(permutations(range(6)))
perm = np.array(random.sample(perm, perm_keep))
input_perm = np.hstack([perm,perm+6,perm+12])
print(input_perm.shape)
print(np.repeat(-1,len(input_perm)).reshape(len(input_perm),1).shape)
input_perm = np.hstack((input_perm, np.repeat(-1,len(input_perm)).reshape((len(input_perm),1))))
# inputs = np.vstack(i[input_perm] for i in inputs)
inputs = np.vstack(inputs[:,p] for p in input_perm)
# outputs = np.hstack(perm[o] for o in outputs) # doesn't work
def perm_outputs(out, p):
    perm_mapping=dict(zip(p,range(6)))
    return [perm_mapping[o] for o in out]
outputs = np.hstack(perm_outputs(outputs,p) for p in perm)
filenames = np.repeat(samples["filenames"], perm_keep)

fn = np.unique(filenames)
train_fn = fn[:len(fn)//2]
train_idx = np.where(np.in1d(filenames,train_fn))
test_idx = np.where(np.logical_not(np.in1d(filenames,train_fn)))
i_train = inputs[train_idx]
o_train = outputs[train_idx]
i_test = inputs[test_idx]
o_test = outputs[test_idx]


print(inputs.shape)
print(inputs[:10])

def split_inputs(input):
    return [input[:,(p,p+6,p+12,-1)] for p in range(6)]

n_print=45

print(inputs[:n_print])
model.fit(split_inputs(i_train), o_train, verbose=True, validation_data=(split_inputs(i_test),o_test), nb_epoch = 5, batch_size = 1000,
          callbacks=[EarlyStopping(patience=10),
                     ModelCheckpoint('value_model.h5',verbose=1,save_best_only=True),
                     ])
# model = load_model("./value_model.h5")
# print(model.predict(inputs)[:n_print])
print(samples["filenames"][:n_print])
print(raw_outputs[:n_print])
preds = model.predict(split_inputs(raw_inputs[:n_print]))
print(preds)
print(np.argmax(preds,axis=1))
# model.save_weights("value_weights.hd5")
