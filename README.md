# 2024 SNU FastMRI Challenge
Team Name: 샤샤  
Leaderboard SSIM Score: **0.9802**

## 1. Model description
VarNet + NAFNet

Training consists of three steps:
1. Train VarNet 50 epochs
2. Train VarNet (from step 1) 15 epochs more
3. Train NAFNet 10 epochs

The input of NAFNet consists of three images:   
- aliased image(`image_input`)
- reconstruction from VarNet (step 2)
- grappa image(`image_grappa`)

## 2. Directory structure
Python 3.8.10

```bash
├── .gitignore
├── leaderboard_eval.py
├── leaderboard_eval.sh
├── README.md
├── reconstruct.py
├── reconstruct.sh
├── reconstruct_2.py
├── train.py
├── train.sh
├── train_2.py
└── utils
│   ├── common
│   │   ├── loss_function.py
│   │   └── utils.py
│   ├── data
│   │   ├── augment
│   │   │   ├── data_augment.py # K-space data augmentation
│   │   │   ├── helpers.py
│   │   │   ├── image_augment.py # Image data augmentation for NAFNet
│   │   │   └── mask_augment.py # K-space mask augmentation
│   │   ├── load_data.py
│   │   └── transforms.py
│   ├── learning
│   │   ├── test_part.py
│   │   ├── test_part_2.py
│   │   ├── train_part.py # for training VarNet
│   │   └── train_part_2.py # for training NafNet
│   └── model
│       ├── nafnet
│       │   └── nafnet.py
│       └── varnet
│           └── varnet.py
└── result
```

## 3. How to set?
1. Install required packages
     ```bash
     pip install fastmri wandb python-dotenv matplotlib lmdb einops pygrappa
     pip install opencv-python==4.8.0.74
     ```
    
2. Create `.env` file and set environment variables
   ```bash
   cd /FastMRI_2024_shasha
   vim .env
   ```
   ```bash
   DATA_DIR_PATH="{path to training data}"
   RESULT_DIR_PATH="{path to folder in which result data will be saved}"
   ```
   In vessl server,
   ```bash
   DATA_DIR_PATH="/home/Data"
   RESULT_DIR_PATH="./result"
   ```

## 4. Train, Reconstruct, and Evaluate LB Dataset
```bash
sh train.sh
sh reconstruct.sh
sh leaderboard_eval.sh
```

## 5. Path to weight, validation loss files
- Step 2
     - weight: `result/varnet/checkpoints/best_model.pt`
     - validation loss: `result/varnet/val_loss_log.npy`
- Step 3
     - weight: `result/nafnet/checkpoints/best_model.pt`
     - validation loss: `result/nafnet/val_loss_log.npy`
