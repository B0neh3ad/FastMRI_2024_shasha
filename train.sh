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
  --debug \

#  --cascade 6 \
#  --chans 15 \
#  --sens_chans 4

# 사용 금지 option 목록
# --amp: loss 값이 이상해짐
# --aug-weight-rot90: MRAugment의 rot90을 이용한 90도 회전 시 가로 세로 길이가 바뀌어 mask 사용 불가
