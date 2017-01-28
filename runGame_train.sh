#!/bin/bash
repeats=50
rm data/*
echo "Starting initial round..."
for repeat in $(seq 1 $repeats)
do
  echo "Repeat #$repeat"
  rm bot.log
  ./halite -t -q -d "30 30" "python ./reinforce/reinforce.py" "python RandomBot.py"
  cat bot.log
done
./reinforce/train.py

for run in {1..10}
do
  echo "Starting round ${run}..."
  for repeat in $(seq 1 $repeats)
  do
    echo "Repeat #$repeat"
    rm bot.log
    ./halite -t -q -d "30 30" "python ./reinforce/reinforce.py $run" "python RandomBot.py"
    cat bot.log
  done
  ./reinforce/train.py $run
done
