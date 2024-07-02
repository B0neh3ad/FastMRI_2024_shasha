#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 train.py \
  -b 1 \
  -e 5 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet' \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/"