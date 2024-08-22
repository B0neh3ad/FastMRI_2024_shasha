#!/bin/bash

if [ -f .env ]; then
  export $(cat .env | grep -v '^#' | xargs)
fi

# step 1: Train VarNet 50 epochs
train.py -b 1 \
  -e 50 \
  -l 0.0001 \
  -n "varnet" \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4 \
  --aug_on \
  --mask_aug_on \
  --aug_strength 0.4 \
  --aug_weight_brightness 0.5 \
  --aug_weight_contrast 0.5 \
  --aug_min_scalex 0.9 \
  --aug_max_scaling 0.25 \
  --aug_weight_translation 1.0 \
  --aug_max_rotation 20

mv result/varnet/checkpoints/model.pt result/varnet/checkpoints/save/model.pt
mv result/varnet/val_loss_log.npy result/varnet/checkpoints/save/val_loss_log.npy

# step 2: Train VarNet (from step 1) 15 epochs more with different setting
python3.8 train.py -b 1 \
  -e 65 \
  -l 0.0001 \
  -n "varnet" \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --seed 430 \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4 \
  --optimizer "adamw" \
  --loss "index_based" \
  --aug_on \
  --mask_aug_on \
  --aug_strength 0.6 \
  --aug_weight_translation 1.0 \
  --aug_weight_rotation 1.0 \
  --aug_weight_shearing 0.0 \
  --aug_weight_scaling 1.0 \
  --aug_weight_flipv 0.0 \
  --aug_weight_brightness 1.0 \
  --aug_weight_contrast 0.5 \
  --aug_max_translation_x 0.1 \
  --aug_max_translation_y 0.1 \
  --aug_max_rotation 25.0 \
  --aug_max_scaling 0.25 \
  --aug_min_scalex 0.8 \
  --load-model \
  --mask_small_on

# step 3: train NAFNet 10 epochs
python3.8 train_2.py -b 2 \
  -e 10 \
  -l 0.0001 \
  -n "nafnet" \
  -t "$DATA_DIR_PATH/train/" \
  -v "$DATA_DIR_PATH/val/" \
  --loss "index_based" \
  --optimizer "adamw" \
  --prev-net-name "varnet" \
  --cascade 6 \
  --chans 15 \
  --sens_chans 4 \
  --grad-clip-on \
  --aug_on \
  --aug_strength 0.5 \
  --lr-scheduler "cosine" \
  --lr-scheduler-on \
  --recon-train \
  --recon-val