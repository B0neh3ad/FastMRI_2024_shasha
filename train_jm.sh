#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

python3.8 train.py \
  -b 1 \
  -e 50 \
  -l 0.001 \
  -r 10 \
  -n 'test_Varnet' \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --aug_on \
  --mask_aug_on \
  --aug_strength 0.5 \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4 \
  --aug_weight_translation 0.5 \
  --aug_weight_rotation 0.5 \
  --aug_weight_shearing 0.0 \
  --aug_weight_scaling 0.5 \
  --aug_weight_flipv 0.0 \
  --aug_weight_brightness 0.0 \
  --aug_weight_contrast 0.5 \
  --aug_max_translation_x 0.1 \
  --aug_max_translation_y 0.1 \
  --aug_max_rotation 10.0 \
  --aug_max_scaling 0.2 \



# 사용 금지 option 목록
# --amp: loss 값이 이상해짐
# --aug-weight-rot90: MRAugment의 rot90을 이용한 90도 회전 시 가로 세로 길이가 바뀌어 mask 사용 불가

# -g, --GPU-NUM: Specifies the GPU number to use for training.
# -b, --batch-size: Sets the batch size for training.
# -e, --num-epochs: Defines the number of epochs for the training process.
# -l, --lr: Sets the learning rate.
# -r, --report-interval: Determines how often to report training progress.
# -n, --net-name: Names the network for identification purposes.
# -t, --data-path-train: Specifies the directory containing training data.
# -v, --data-path-val: Specifies the directory containing validation data.
