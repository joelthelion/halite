#!/bin/bash
base_repeats=8
size="40 40"
rm data/*
echo "Starting initial round..."
for repeat in $(seq 1 $base_repeats)
do
  echo "Repeat #$repeat"
  rm bot.log
  ./halite -t -q -d "${size}" "python ./reinforce/reinforce.py" "python RandomBot.py"
  cat bot.log
done
./reinforce/train.py

for run in {1..10}
do
  echo "Starting round ${run}..."
  repeats=$(((run+1) * base_repeats))
  echo "Doing $repeats repeats."
  for repeat in $(seq 1 $repeats)
  do
    echo "Repeat #$repeat"
    rm bot.log
    ./halite -t -q -d "${size}" "python ./reinforce/reinforce.py $run" "python RandomBot.py"
    cat bot.log
  done
  ./reinforce/train.py $run
done
