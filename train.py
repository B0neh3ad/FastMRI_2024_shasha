import torch
import argparse
import shutil
import os, sys
from pathlib import Path
from dotenv import load_dotenv

from utils.data.augment.data_augment import KspaceDataAugmentor

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')
from utils.learning.train_part import train

if os.getcwd() + '/utils/common/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/common/')
from utils.common.utils import seed_fix


def parse():
    parser = argparse.ArgumentParser(description='Train Varnet/DIRCN on FastMRI challenge Images',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=500, help='Report interval')
    parser.add_argument('-n', '--net-name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-t', '--data-path-train', type=Path, default='/Data/train/', help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='/Data/val/', help='Directory of validation data')
    
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12') ## important hyperparameter
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet') ## important hyperparameter
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet') ## important hyperparameter

    parser.add_argument('--input-key', type=str, default='kspace', help='Name of kspace input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')
    parser.add_argument('--seed', type=int, default=430, help='Fix random seed')

    parser.add_argument('--no-val', default=False, help='This switch turns off validation', action='store_true')

    # loss type
    parser.add_argument('--loss', type=str, default='ssim', help='Loss function')
    parser.add_argument('--alpha', type=float, default=0.84, help='Alpha value for mixed loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma value for custom focal loss')
    parser.add_argument('--max-num-slices', type=float, default=22, help='Maximum number of slices in all dataset')

    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')

    # for debug mode
    parser.add_argument('--debug', default=False, help='Set debug mode', action='store_true')

    # for automatic mixed precision
    parser.add_argument('--amp', default=False, help='Set automatic mixed precision', action='store_true')

    # for gradient clip
    parser.add_argument('--grad-clip-on', default=False, help='Set gradient clipping', action='store_true')
    parser.add_argument('--grad-clip', type=float, default=0.01, help='Max norm of the gradients')

    # for gradient accumultation
    parser.add_argument('--iters-to-grad-acc', type=int, default=1, help='Iterations to gradient accumulation')

    # data augmentation config
    parser = KspaceDataAugmentor.add_augmentation_specific_args(parser)

    # mask augmentation config
    parser.add_argument('--mask_aug_on', default=False, help='This switch turns mask augmentation on.', action='store_true')
    parser.add_argument('--aug_weight_mask', type=float, default=1.0, help='Weight of mask augmentation probability. Augmentation probability will be multiplied by this constant')
    parser.add_argument('--mask_small_on', default=False, help='This switch turns mask smaller', action='store_true')
    parser.add_argument('--low_to_high_acc', default=False, help='This switch turns mask augmentation from low to high acc', action='store_true')

    # scheduler
    parser.add_argument('--lr-scheduler-on', default=False, help='This switch turns learning rate scheduler on.',
                        action='store_true')
    parser.add_argument('--lr-scheduler', type=str, default='plateau', help='Scheduler')
    parser.add_argument('--patience', type=int, default=2, help='Patience for reduce learning rate')
    parser.add_argument('--t-max', type=int, default=10, help='Period of learning rate when using cosine annealing')
    parser.add_argument('--lr-step-size', type=int, default=10, help='Step size for learning rate scheduler')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='gamma value for learning rate scheduler')

    # wandb
    parser.add_argument('--wandb-on', default=False, help='This switch turns WandB logging on.', action='store_true')
    parser.add_argument('--load-model', default=False , help='Load last model from /checkpoints. Do not use with --wandb-run-id', action='store_true')
    parser.add_argument('--wandb-run-id', type=str, default=None, help='WandB run ID to resume. If not provided, starts a new run.')

    load_dotenv()
    result_dir_path = os.environ['RESULT_DIR_PATH']
    data_dir_path = os.environ['DATA_DIR_PATH']

    parser.add_argument('--result-dir-path', type=str, default=result_dir_path, help='Path to result directory')
    parser.add_argument('--data-dir-path', type=str, default=data_dir_path, help='Path to data directory')

    args = parser.parse_args()
    return args

def start_train():
    args = parse()

    # fix seed
    if args.seed is not None:
        seed_fix(args.seed)

    args.exp_dir = args.result_dir_path / args.net_name / 'checkpoints'
    args.train_dir = args.result_dir_path / args.net_name / 'reconstructions_train'
    args.val_dir = args.result_dir_path / args.net_name / 'reconstructions_val'
    args.main_dir = args.result_dir_path / args.net_name / __file__
    args.val_loss_dir = args.result_dir_path / args.net_name

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.train_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)

if __name__ == '__main__':
    start_train()