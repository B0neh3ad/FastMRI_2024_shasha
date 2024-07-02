#!/bin/bash

if [-f .env]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python leaderboard_eval.py \
  -lp "$DATA_DIR_PATH/leaderboard" \
  -yp "$RESULT_DIR_PATH/test_Varnet/reconstructions_leaderboard"