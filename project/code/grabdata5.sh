#!/bin/bash

THRESHOLD=1000

for LIFE in 100 200 300 400 500 1000
do
    Y=0
    while [ $Y -lt $THRESHOLD ]
    do
        python run_game.py -d 0 --train 3 --starting_life $LIFE --out_file plant_data/$LIFE.txt
        Y=$(python count.py plant_data/$LIFE.txt)
    done
    echo $Y
done