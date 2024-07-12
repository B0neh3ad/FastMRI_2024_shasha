#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 train_2.py \
  -b 4 \
  -e 10 \
  -l 0.001 \
  -n "nafnet" \
  -r 10 \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --prev-net-name "test_Varnet" \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4

# 주의: augmentation 아직 안 만들어 둠!