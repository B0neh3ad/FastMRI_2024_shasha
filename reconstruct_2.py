import argparse
from pathlib import Path
import os, sys

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')

from utils.learning.test_part_2 import forward
import time
from dotenv import load_dotenv


def parse():
    parser = argparse.ArgumentParser(description='Test Varnet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-n', '--net_name', type=Path, default='test_varnet', help='Name of network')
    parser.add_argument('-p', '--path_data', type=Path, default='/Data/leaderboard/', help='Directory of test data')

    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    parser.add_argument('--pn', '--prev-net-name', type=Path, default='test_Varnet', help='Name of previous network. (This should be the same as the dir name for prev network)')

    # previous net info for reconstruction (for experiment)
    parser.add_argument('--cascade', type=int, default=1, help='Number of cascades | Should be less than 12')
    parser.add_argument('--chans', type=int, default=9, help='Number of channels for cascade U-Net | 18 in original varnet')
    parser.add_argument('--sens_chans', type=int, default=4, help='Number of channels for sensitivity map U-Net | 8 in original varnet')

    # for debug mode
    parser.add_argument('--debug', type=bool, default=False, help='Set Debug mode')

    load_dotenv()
    result_dir_path = os.environ['RESULT_DIR_PATH']
    parser.add_argument('--result_dir_path', type=str, default=result_dir_path, help='Path to result directory')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = args.result_dir_path / args.net_name / 'checkpoints'

    public_acc, private_acc = None, None

    assert (len(os.listdir(args.path_data)) == 2)

    for acc in os.listdir(args.path_data):
        if acc in ['acc4', 'acc5', 'acc8']:
            public_acc = acc
        else:
            private_acc = acc

    assert (None not in [public_acc, private_acc])

    start_time = time.time()

    # Public Acceleration
    print(f'Start reconstruction for public acc({public_acc}) data')
    args.data_path = args.path_data / public_acc  # / "kspace"
    args.forward_dir = args.result_dir_path / args.net_name / 'reconstructions_leaderboard' / 'public'
    args.recon_path = args.result_dir_path / args.prev_net_name / 'reconstructions_leaderboard' / 'public'
    print(f'Saved into {args.forward_dir}')
    forward(args)

    # Private Acceleration
    print(f'Start reconstruction for private acc({private_acc}) data')
    args.data_path = args.path_data / private_acc  # / "kspace"
    args.forward_dir = args.result_dir_path / args.net_name / 'reconstructions_leaderboard' / 'private'
    args.recon_path = args.result_dir_path / args.prev_net_name / 'reconstructions_leaderboard' / 'private'
    print(f'Saved into {args.forward_dir}')
    forward(args)

    reconstructions_time = time.time() - start_time
    print(f'Total Reconstruction Time = {reconstructions_time:.2f}s')

    print('Success!') if reconstructions_time < 3000 else print('Fail!')