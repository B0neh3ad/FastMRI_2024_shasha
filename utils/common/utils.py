"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import glob
import os
import wandb

from skimage.metrics import structural_similarity
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import random
from matplotlib import pyplot as plt
import cv2


"""
These functions are used in the training script.
- save_reconstructions: Save reconstructions to h5 files.
- ssim_loss: Compute SSIM loss(1 - SSIM). For validation.
- seed_fix: Fix seed for reproducibility.
- show_kspace_slice: Show kspace slice. (For debugging)
- show_image_slices: Show image slices. (For debugging)
- get_mask: Get mask for the target.
- get_mask2: Get mask for the target. (smaller mask)
"""


def save_reconstructions(reconstructions, out_dir, targets=None, inputs=None):
    """
    Saves the reconstructions from a model into h5 files that is appropriate for submission
    to the leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input filenames to
            corresponding reconstructions (of shape num_slices x height x width).
        out_dir (pathlib.Path): Path to the output directory where the reconstructions
            should be saved.
        target (np.array): target array
    """
    out_dir.mkdir(exist_ok=True, parents=True)
    for fname, recons in reconstructions.items():
        with h5py.File(out_dir / fname, 'w') as f:
            f.create_dataset('reconstruction', data=recons)
            if targets is not None:
                f.create_dataset('target', data=targets[fname])
            if inputs is not None:
                f.create_dataset('input', data=inputs[fname])

    result_dir_path = os.environ['RESULT_DIR_PATH']

def ssim_loss(gt, pred, maxval=None) -> float:
    """Compute Structural Similarity Index Metric (SSIM)
       ssim_loss is defined as (1 - ssim)
    """
    maxval = gt.max() if maxval is None else maxval

    ssim = 0
    for slice_num in range(gt.shape[0]):
        ssim = ssim + structural_similarity(
            gt[slice_num], pred[slice_num], data_range=maxval
        )

    ssim = ssim / gt.shape[0]
    return 1 - ssim

def seed_fix(n):
    torch.manual_seed(n)
    torch.cuda.manual_seed(n)
    torch.cuda.manual_seed_all(n)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(n)
    random.seed(n)


def show_kspace_slice(kspace_path, slice_idx=0, coil_idxs=[0, 5, 10], cmap=None):
    # convert kspace slice to real image
    hf = h5py.File(kspace_path)
    kspace_slice = hf['kspace'][slice_idx]
    kspace_slice = np.log(np.abs(kspace_slice) + 1e-9)

    kspace_title = kspace_path.split('/')[-1]

    plt.figure()
    plt.axis('off')
    plt.title(f'{kspace_title} - slice {slice_idx}')
    for i, coil_idx in enumerate(coil_idxs):
        ax = plt.subplot(1, len(coil_idxs), i + 1)
        plt.title(f'coil {coil_idx}')
        ax.imshow(kspace_slice[coil_idx], cmap=cmap)


def show_image_slices(img_path, slice_idxs=[0], cmap='gray'):
    img_keys = ['image_grappa', 'image_input', 'image_label']
    img_title = img_path.split('/')[-1]

    for i, slice_idx in enumerate(slice_idxs):
        hf = h5py.File(img_path)
        plt.figure()
        plt.title(f'{img_title} - slice {slice_idx}')
        plt.axis('off')
        for j, key in enumerate(img_keys):
            ax = plt.subplot(1, len(img_keys), j + 1)
            img_slice = hf[key][slice_idx]
            plt.title(key)
            ax.imshow(img_slice, cmap=cmap)

def get_mask(target: torch.Tensor) -> torch.Tensor:
    mask = (target > 5e-5).float()
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=target.device)
    mask = mask.unsqueeze(1)
    for _ in range(1):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask == kernel.sum()).float()
    for _ in range(15):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask > 0).float()
    for _ in range(14):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask == kernel.sum()).float()
    mask = mask.squeeze(1)

    return mask

def get_mask2(target: torch.Tensor) -> torch.Tensor:
    mask = (target > 5e-5).float()
    kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=target.device)
    mask = mask.unsqueeze(1)
    for _ in range(1):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask == kernel.sum()).float()
    for _ in range(15):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask > 0).float()
    for _ in range(30):
        mask = F.conv2d(mask, kernel, padding=1)
        mask = (mask == kernel.sum()).float()
    mask = mask.squeeze(1)

    return mask
