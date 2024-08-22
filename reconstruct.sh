#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 reconstruct.py \
  -b 2 \
  -n "varnet" \
  -p "$DATA_DIR_PATH/leaderboard" \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4

python3.8 reconstruct_2.py \
  -b 2 \
  -n "nafnet" \
  -p "$DATA_DIR_PATH/leaderboard" \
  --prev-net-name 'varnet'