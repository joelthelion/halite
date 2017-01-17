#!/usr/bin/env python

import shutil
import random
from zipfile import ZipFile
import pickle
import json
import numpy as np
import hlt
from model import Model

with ZipFile("/home/joel/data/halite/replays.zip") as zipf:
    fl = [zf for zf in zipf.filelist if zf.filename.endswith(".hlt")]
    outputs = []
    inputs = []
    for n_file, f in enumerate(fl):
        with zipf.open(f) as fo:
            replay = json.load(fo)
            print(f.filename)
            for n_frame, (frame, moves) in enumerate(zip(replay["frames"],replay["moves"])):
                # if n_frame > 30:
                #     continue
                game_map = hlt.GameMap.from_replay(replay["width"],replay["height"],
                        replay["productions"], frame)
                for row in game_map.contents:
                    for square in row:
                        if random.random() < 0.03 and square.owner != 0:
                            # stupid starter kit translation
                            move = (moves[square.y][square.x] -1) % 5
                            outputs.append(move)
                            inputs.append(Model.gen_input(game_map, square))
        if n_file > 20:
            # features = np.vstack(inputs)
            # features = features - np.mean(features, axis=0)
            # features = features / np.std(features, axis=0)
            np.savez("samples.tmp", inputs=inputs, outputs=outputs)
            shutil.move("samples.tmp.npz", "samples.npz")

