#!/usr/bin/env python

import numpy as np
import pandas

samples = np.load("value_samples.npz")
inputs = samples["inputs"]
outputs = np.array(samples["outputs"]).ravel()
filenames = samples["filenames"]

def make_player_input(inputs, outputs, p):
    rows, columns = inputs.shape
    pi = np.ones((rows,6)) * p # the new columns are the player id and the winner
    pi[:,0:4] = inputs[:,(0+p,6+p,12+p,-1)]
    pi[:,-1] = outputs
    return pi

    # all = np.concatenate([territory[1:], strength[1:], productions[1:], [remaining_turns]])
stack = np.vstack(make_player_input(inputs, outputs, p) for p in range(6))
np.savez("player_values.npz", stack)
player_inputs = pandas.DataFrame(stack,
        columns=["territory","strength","production","remaining_turns", "player", "winner"])
player_inputs["filename"] = np.repeat(filenames, 6)

player_inputs.to_csv("results.csv")
