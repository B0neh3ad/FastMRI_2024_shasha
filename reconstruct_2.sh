#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 reconstruct_2.py \
  -b 2 \
  -n "nafnet" \
  -p "$DATA_DIR_PATH/leaderboard" \
  --prev-net-name 'test_Varnet'