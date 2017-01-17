#!/usr/bin/env python

import shutil
import random
from zipfile import ZipFile
import pickle
import json
import numpy as np
import hlt
from model import Model

def winner(replay):
    """ Returns the player with the most territory """
    return np.argmax(np.bincount(np.array(replay["frames"][-1]).reshape(-1,2)[:,0])[1:])+1

def pad7(array):
    z = np.zeros(7)
    z[0:len(array)] = array
    return z

def frame_to_features(replay, frame_id):
    prod = np.array(replay["productions"])
    armies = np.array(replay["frames"][frame_id]).reshape(-1,2)

    territory = np.bincount(armies[:,0])
    strength =  np.bincount(armies[:,0], weights=armies[:,1])
    productions =  np.bincount(armies[:,0], weights=prod.ravel())
    all = np.concatenate([pad7(territory), pad7(strength), pad7(productions)])

    # print("territory",territory)
    # print("strength",strength)
    # print("productions",productions)
    # print("all",all)
    return all

with ZipFile("/home/joel/data/halite/replays.zip") as zipf:
    fl = [zf for zf in zipf.filelist if zf.filename.endswith(".hlt")]
    outputs = []
    inputs = []
    for n_file, f in enumerate(fl):
        with zipf.open(f) as fo:
            replay = json.load(fo)
            print(f.filename)
            for n_frame in range(len(replay["frames"])):
                if n_frame % 30 != 0:
                    continue
                inputs.append(frame_to_features(replay,n_frame))
                outputs.append(winner(replay)-1)
        if n_file > 20:
            np.savez("value_samples.tmp", inputs=inputs, outputs=np.vstack(outputs))
            shutil.move("value_samples.tmp.npz", "value_samples.npz")

