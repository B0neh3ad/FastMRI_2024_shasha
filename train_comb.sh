#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 train.py \
  -b 1 \
  -e 50 \
  -l 0.001 \
  -r 10 \
  -n 'comb' \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --debug \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4 \
  --aug_on \
  --mask_aug_on \
  --aug_strength 0.5

