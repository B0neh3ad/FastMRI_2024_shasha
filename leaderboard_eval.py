import argparse
import numpy as np
import h5py
import random
import glob
import os
import torch
from utils.common.loss_function import SSIMLoss
import torch.nn.functional as F
import cv2
from pathlib import Path
from tqdm import tqdm

from utils.data.postprocess import Dithering


class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)

    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))

        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        # data_range = data_range[:, None, None, None]
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, A3, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            2 * vx * vy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        # Calculate Luminance, Contrast, Structure each
        luminance = A1 / B1
        contrast = A3 / B2
        structure = A2 / A3
        return S.mean(), (luminance.mean(), contrast.mean(), structure.mean())


def forward(args):
    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

    leaderboard_data = glob.glob(os.path.join(args.leaderboard_data_path, '*.h5'))
    if len(leaderboard_data) != 58:
        raise NotImplementedError('Leaderboard Data Size Should Be 58')

    for yp in args.your_data_path:
        your_data = glob.glob(os.path.join(yp, '*.h5'))
        if len(your_data) != 58:
            raise NotImplementedError('Your Data Size Should Be 58')

    ssim_total = 0
    idx = 0
    ssim_calculator = SSIM().to(device=device)
    ssim_list = [0] * 22
    slice_index_cnt = [0] * 22
    yp_len = len(args.your_data_path)

    l_total, c_total, s_total = 0, 0, 0
    with torch.no_grad():
        for i_subject in tqdm(range(58)):
            l_fname = os.path.join(args.leaderboard_data_path, 'brain_test' + str(i_subject + 1) + '.h5')
            y_fname = [os.path.join(yp, 'brain_test' + str(i_subject + 1) + '.h5') for yp in args.your_data_path]
            with h5py.File(l_fname, "r") as hf:
                num_slices = hf['image_label'].shape[0]
            for i_slice in range(num_slices):
                with h5py.File(l_fname, "r") as hf:
                    target = hf['image_label'][i_slice]
                    mask = np.zeros(target.shape)
                    mask[target > 5e-5] = 1
                    kernel = np.ones((3, 3), np.uint8)
                    mask = cv2.erode(mask, kernel, iterations=1)
                    mask = cv2.dilate(mask, kernel, iterations=15)
                    mask = cv2.erode(mask, kernel, iterations=14)

                    target = torch.from_numpy(target).to(device=device)
                    mask = (torch.from_numpy(mask).to(device=device)).type(torch.float)

                    maximum = hf.attrs['max']

                with h5py.File(y_fname[0], "r") as hf:
                    recon_numpy = hf[args.output_key][i_slice]
                    recon = torch.zeros(recon_numpy.shape).to(device=device)

                for fname in y_fname:
                    with h5py.File(fname, "r") as hf:
                        recon_numpy = hf[args.output_key][i_slice]
                        recon += torch.from_numpy(recon_numpy).to(device=device)

                recon /= yp_len

                # ssim_total += ssim_calculator(recon, target, maximum).cpu().numpy()
                ssim_val, (l, c, s) = ssim_calculator(recon * mask, target * mask, maximum)
                ssim_val = ssim_val.cpu().numpy()
                ssim_list[i_slice] += ssim_val
                slice_index_cnt[i_slice] += 1
                ssim_total += ssim_val
                idx += 1

                l.cpu().numpy()
                l_total += l
                c.cpu().numpy()
                c_total += c
                s.cpu().numpy()
                s_total += s

    print("-------------------------------------")
    for i_slice, ssim_val in enumerate(ssim_list):
        if slice_index_cnt[i_slice] == 0:
            break
        print(f"SSIM for slice {i_slice}: {ssim_val / slice_index_cnt[i_slice]}")
    print("-------------------------------------")
    print(f"Average Luminance: {l / idx}")
    print(f"Average Contrast: {c / idx}")
    print(f"Average Structure: {s / idx}")
    print(l * c * s / idx)
    print("-------------------------------------")
    return ssim_total / idx


if __name__ == '__main__':
    """
    Image Leaderboard Dataset Should Be Utilized
    For a fair comparison, Leaderboard Dataset Should Not Be Included When Training. This Is Of A Critical Issue.
    Since This Code Print SSIM To The 4th Decimal Point, You Can Use The Output Directly.
    """
    parser = argparse.ArgumentParser(description=
                                     'FastMRI challenge Leaderboard Image Evaluation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--GPU_NUM', type=int, default=0)
    parser.add_argument('-lp', '--path_leaderboard_data', type=Path, default='/Data/leaderboard/')

    """
    Modify Path Below To Test Your Results
    """
    parser.add_argument('-yp', '--path_your_data', nargs='+', type=Path,
                        default='../result/test_Unet/reconstructions_leaderboard/')
    parser.add_argument('-key', '--output_key', type=str, default='reconstruction')

    args = parser.parse_args()

    public_acc, private_acc = None, None

    assert (len(os.listdir(args.path_leaderboard_data)) == 2)

    for acc in os.listdir(args.path_leaderboard_data):
        if acc in ['acc4', 'acc5', 'acc8']:
            public_acc = acc
        else:
            private_acc = acc

    # public acceleration
    print("[Start evaluation for public Acc]")
    args.leaderboard_data_path = args.path_leaderboard_data / public_acc / 'image'
    args.your_data_path = [yp / 'public' for yp in args.path_your_data]
    SSIM_public = forward(args)

    # private acceleration
    print("[Start evaluation for private Acc]")
    args.leaderboard_data_path = args.path_leaderboard_data / private_acc / 'image'
    args.your_data_path = [yp / 'private' for yp in args.path_your_data]
    SSIM_private = forward(args)

    print("Leaderboard SSIM : {:.4f}".format((SSIM_public + SSIM_private) / 2))
    print("=" * 10 + " Details " + "=" * 10)
    print("Leaderboard SSIM (public): {:.4f}".format(SSIM_public))
    print("Leaderboard SSIM (private): {:.4f}".format(SSIM_private))