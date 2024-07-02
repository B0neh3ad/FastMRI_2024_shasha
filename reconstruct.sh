#!/bin/bash

if [-f .env]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python reconstruct.py \
  -b 2 \
  -n 'test_Varnet' \
  -p "$DATA_DIR_PATH/leaderboard"