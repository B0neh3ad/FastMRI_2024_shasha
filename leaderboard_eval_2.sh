#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 leaderboard_eval.py \
  -lp "$DATA_DIR_PATH/leaderboard" \
  -yp "$RESULT_DIR_PATH/nafnet/reconstructions_leaderboard"