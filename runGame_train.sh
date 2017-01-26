#!/bin/bash
rm data/*
echo "Starting initial round..."
for repeat in {1..10}
do
  echo "Repeat #$repeat"
  rm ./*.log
  ./halite -t -q -d "30 30" "python ./reinforce/reinforce.py" "python RandomBot.py"
  cat ./*.log
done
./reinforce/train.py

for run in {1..10}
do
  echo "Starting round ${run}..."
  for repeat in {1..10}
  do
    echo "Repeat #$repeat"
    rm ./*.log
    ./halite -t -q -d "30 30" "python ./reinforce/reinforce.py $run" "python RandomBot.py"
    cat ./*.log
  done
  ./reinforce/train.py $run
done