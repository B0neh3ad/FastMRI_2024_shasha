#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 train.py \
  -b 1 \
  -e 50 \
  -l 0.001 \
  -r 10 \
  -n 'dircn' \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --wandb-on \
  --cascade 30 \
  --chans 20 \
  --sens_chans 12

#  --aug_on \
#  --mask_aug_on \
#  --aug_strength 0.5 \
#  --aug_weight_brightness 0 \
#  --aug_max_rotation 10.0 \
#  --aug_min_scalex 0.9 \