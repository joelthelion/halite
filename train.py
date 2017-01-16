#!/usr/bin/env python

from zipfile import ZipFile
import json
import numpy as np
import hlt
from model import Model

with ZipFile("/home/joel/data/halite/replays.zip") as zipf:
    fl = [zf for zf in zipf.filelist if zf.filename.endswith(".hlt")]
    for f in fl:
        with zipf.open(f) as fo:
            replay = json.load(fo)
            for frame, moves in zip(replay["frames"],replay["moves"]):
                game_map = hlt.GameMap.from_replay(replay["width"],replay["height"],
                        replay["productions"], frame)
                print(game_map)
                print(game_map.width)
                print(game_map.height)
                print(moves)
                for row in game_map.contents:
                    for square in row:
                        if square.owner != 0:
                            print(square.owner, moves[square.y][square.x],
                                    Model.gen_input(game_map,square))

