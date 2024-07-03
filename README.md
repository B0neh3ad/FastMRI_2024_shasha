## 주의!
4. How to set 파트 내용을 추가했으니 반드시 읽기
---
# 2024 baby varnet
## 1. 폴더 계층

### 폴더의 전체 구조
![image](docs/fastmri_folder_structure.png)
* FastMRI_challenge, Data, result 폴더가 위의 구조대로 설정되어 있어야 default argument를 활용할 수 있습니다.
* 본 github repository는 FastMRI_challenge 폴더입니다.
* Data 폴더는 MRI data 파일을 담고 있으며 아래에 상세 구조를 첨부하겠습니다.
* result 폴더는 학습한 모델의 weights을 기록하고 validation, leaderboard dataset의 reconstruction image를 저장하는데 활용되며 아래에 상세 구조를 첨부하겠습니다.

### Data 폴더의 구조
![image](docs/fastmri_data_structure.png)
* train, val:
    * train, val 폴더는 각각 모델을 학습(train), 검증(validation)하는데 사용하며 각각 image, kspace 폴더로 나뉩니다.
    * 참가자들은 generalization과 representation의 trade-off를 고려하여 train, validation의 set을 자유로이 나눌 수 있습니다.
    * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: brain_{mask 형식}_{순번}.h5
    * ex) brain_acc8_3.h5  
    * {mask 형식}은 "acc4", "acc5", "acc8" 중 하나입니다.
    * "acc4"와 "acc5"의 경우 {순번}은 1 ~ 118, "acc8"의 경우 {순번}은 1 ~ 120 사이의 숫자입니다. 
* leaderboard:
   * **leaderboard는 성능 평가를 위해 활용하는 dataset이므로 절대로 학습 과정에 활용하면 안됩니다.**
   * leaderboard 폴더는 mask 형식에 따라서 acc5과 acc9 폴더로 나뉩니다.
   * acc5과 acc9 폴더는 각각 image, kspace 폴더로 나뉩니다.
   * image와 kspace 폴더에 들어있는 파일의 형식은 다음과 같습니다: brain_test_{순번}.h5
   * {순번}은 1 ~ 58 사이의 숫자입니다. 

### result 폴더의 구조
* result 폴더는 모델의 이름에 따라서 여러 폴더로 나뉠 수 있습니다.
* test_Unet (혹은 test_VN) 폴더는 아래 3개의 폴더로 구성되어 있습니다.
  * checkpoints - `model.pt`, `best_model.pt`의 정보가 있습니다. 모델의 weights 정보를 담고 있습니다.
  * reconstructions_val - validation dataset의 reconstruction을 저장합니다. brain_{mask 형식}_{순번}.h5 형식입니다. (```train.py``` 참고)
  * reconstructions_leaderboard - leaderboard dataset의 reconstruction을 저장합니다. brain_test_{순번}.h5 형식입니다. (```evaluation.py``` 참고)
  * val_loss_log.npy - epoch별로 validation loss를 기록합니다. (```train.py``` 참고)

## 2. 폴더 정보
Python 3.8.10

```bash
├── .gitignore
├── evaluate.py
├── leaderboard_eval.py
├── plot.py
├── README.md
├── train.py
└── utils
│   ├── common
│   │   ├── loss_function.py
│   │   └── utils.py
│   ├── data
│   │   ├── load_data.py
│   │   └── transforms.py
│   ├── learning
│   │   ├── test_part.py
│   │   └── train_part.py
│   └── model
│       └── varnet.py
├── Data
└── result
```

## 3. Before you start
* ```train.py```, ```evaluation.py```, ```leaderboard_eval.py``` 순으로 코드를 실행하면 됩니다.
* ```train.py```
   * train/validation을 진행하고 학습한 model의 결과를 result 폴더에 저장합니다.
   * 가장 성능이 좋은 모델의 weights을 ```best_model.pt```으로 저장합니다. 
* ```evaluation.py```
   * ```train.py```으로 학습한 ```best_model.pt```을 활용해 leader_board dataset을 reconstruction하고 그 결과를 result 폴더에 저장합니다.
   * acceleration 종류와 관계없이 하나의 파이프라인을 통해 전부 reconstruction을 실행합니다.
* ```leaderboard_eval.py```
   * ```evaluation.py```을 활용해 생성한 reconstruction의 SSIM을 측정합니다.
   * SSIM (public): 기존에 공개된 acc4, acc5, acc8 중 하나인 acc5 데이터의 reconstruction의 SSIM을 측정합니다.
   * SSIM (private): 기존에 공개되지 않은 acc9 데이터의 reconstruction의 SSIM을 측정합니다.
   * Total SSIM은 SSIM (public), SSIM (private)의 평균으로 계산됩니다. Report할 때 이 값을 제출하시면 됩니다.
   * Inference Time이 대회 GPU 기준으로 특정 시간을 초과할 경우 Total SSIM을 기록할 수 없습니다. 실제 Evaluation 때 조교가 확인할 예정이며, inference time은 과도하게 모델이 크지 않는다면 걱정하실 필요 없습니다.
        * 현재 대회 GPU 세팅이 늦어지는 관계로, 세팅 완료 후 정확한 시간 기준을 공지드리겠습니다. 불편을 드려 죄송합니다.

## 4. How to set?
> 버전 문제 및 기존 패키지들과의 의존성 충돌 문제를 막기 위해, 반드시 python 3.8.10 가상환경 내에서만 작업할 것!

1. python 3.8.10 설치 (Ubuntu 22.04 LTS 기준)
```bash
sudo apt update
sudo apt install -y build-essential checkinstall
sudo apt install -y libeditreadline-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev

cd /usr/src
sudo wget https://www.python.org/ftp/python/3.8.10/Python-3.8.10.tgz
sudo tar xzf Python-3.8.10.tgz

cd Python-3.8.10
sudo ./configure --enable-optimizations
sudo make altinstall
```
```bash
python3.8 --version # 3.8.10 버전이 출력되는지 확인!
```

2. 가상환경 생성 및 활성화
```bash
# repository directory 이동
mkdir [directory 경로]
cd [directory 경로]

# 가상환경 생성
python3.8 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```

3. requirement package 설치
```bash
pip install torch numpy requests tqdm h5py scikit-image pyyaml opencv-python matplotlib
pip install wandb
pip install python-dotenv
```

4. `.env` 파일 생성 후 환경변수 설정
```bash
DATA_DIR_PATH="{학습 data가 담긴 폴더의 경로}"
RESULT_DIR_PATH="{result data를 저장할 폴더의 경로}"
```
```bash
# ex)
DATA_DIR_PATH="/home/Data"
RESULT_DIR_PATH="../results"
```

## 5. How to train?
```bash
sh train.sh
```
- validation할 때, reconstruction data를 ```result/reconstructions_val/```에 저장합니다.
  - **반드시!! result 폴더를 따로 백업해둡시다.**
- epoch 별로 validation dataset에 대한 loss를 기록합니다.
- `python train.py [Options]`를 입력해서 실행할 수도 있으나, `.env` 파일 내 환경변수의 원활한 사용을 위해 `sh train.sh`을 권장합니다.
- seed 고정을 하여 이후에 Re-training하였을 때 같은 결과가 나와야 합니다.

## 6. How to reconstruct?
```bash
sh reconstruct.sh
```
- leaderboard 평가를 위한 reconstruction data를 ```$RESULT_DIR_PATH/reconstructions_leaderboard```에 저장합니다.

## 7. How to evaluate LeaderBoard Dataset?
```bash
sh leaderboard_eval.sh
```
- leaderboard 순위 경쟁을 위한 SSIM 값을 한번에 구합니다.
- Total SSIM을 제출합니다.

## 8. What to submit!
- github repository(코드 실행 방법 readme에 상세 기록)
- loss 그래프 혹은 기록
- 모델 weight file
- 모델 설명 ppt
