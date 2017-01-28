#!/bin/bash

rm play.log
./halite -t -q -d "30 30" "python ./reinforce/play.py" "python RandomBot.py"
# ./halite -t -q -d "30 30" "python ./MyBotVanilla.py" "python RandomBot.py"
# ./halite -t -q -d "30 30" "python MyBot.py" "cd ../Halite-ML-starter-bot; python MyBot.py"
# ./halite -t -q -d "30 30" "python RandomBot.py" "cd ../Halite-ML-starter-bot; python MyBot.py"
