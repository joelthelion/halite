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
    # return np.argmax(np.bincount(np.array(replay["frames"][-1]).reshape(-1,2)[:,0])[1:])+1
    armies = np.array(replay["frames"][-1]).reshape(-1,2)
    players, counts = np.unique(armies[:,0], return_counts=True)
    counts[0] = -1 # the background can't win
    winner = players[np.argmax(counts)]
    if winner == 0:
        print(winner)
        print(players,counts)
        qsdfsdf
    return winner

def frame_to_features(replay, frame_id):
    prod = np.array(replay["productions"])
    armies = np.array(replay["frames"][frame_id]).reshape(-1,2)

    players, player_idx = np.unique(armies[:,0], return_inverse=True)
    territory = np.zeros(7)
    territory[players] = np.bincount(player_idx)

    strength = np.zeros(7)
    strength[players] = np.bincount(player_idx, weights=armies[:,1])

    productions = np.zeros(7)
    productions[players] = np.bincount(player_idx, weights=prod.ravel())

    remaining_turns = len(replay["frames"]) - frame_id - 1

    all = np.concatenate([territory[1:], strength[1:], productions[1:], [remaining_turns]])

    # print("territory",territory)
    # print("strength",strength)
    # print("productions",productions)
    # print("all",all)
    return all

with ZipFile("/home/joel/data/halite/replays.zip") as zipf:
    fl = [zf for zf in zipf.filelist if zf.filename.endswith(".hlt")]
    outputs = []
    inputs = []
    filenames = []
    for n_file, f in enumerate(fl):
        with zipf.open(f) as fo:
            replay = json.load(fo)
            print(f.filename)
            try:
                output = winner(replay)-1
            except ValueError:
                print("couldn't determine winner, skipping game")
                continue
            if len(replay["frames"]) < 5: #remove buggy games
                continue
            for n_frame in range(len(replay["frames"])):
                if n_frame % 3 != 0:
                    continue
                inputs.append(frame_to_features(replay,n_frame))
                outputs.append(output)
                filenames.append(f.filename)
                # print(inputs,outputs)
        if n_file > 20:
            np.savez("value_samples.tmp", inputs=inputs, outputs=np.vstack(outputs), filenames=filenames)
            shutil.move("value_samples.tmp.npz", "value_samples.npz")

