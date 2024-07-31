"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
These loss functions are used in the training script. (Not validation)
- SSIMLoss: SSIM loss module.
- MixedLoss: Mixed loss module. (MS-SSIM + L1)
- CustomFocalLoss: Focal loss (with MS-SSIM loss) module.
- IndexBasedWeightedLoss: Index-based weighted MS-SSIM loss module.
- MixIndexL1Loss: Mixed loss module. (Index-based weighted MS-SSIM + L1)
!!! MixIndexL1Loss is not yet added to the training script. !!!
"""


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size ** 2)
        NP = win_size ** 2
        self.cov_norm = NP / (NP - 1)

    def forward(self, X, Y, data_range):
        if len(X.shape) == 3:
            X = X.unsqueeze(1)
        Y = Y.unsqueeze(1)
        data_range = data_range[:, None, None, None]
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
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        return 1 - S.mean()

class MixedLoss(nn.Module):
    """
    Mixed loss module. (MS-SSIM + L1)
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, alpha: float = 0.84):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            alpha: mixing factor of SSIM and L1 loss.
        """
        super(MixedLoss, self).__init__()
        self.alpha = alpha
        self.ms_ssim = SSIMLoss(win_size=win_size, k1=k1, k2=k2)
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, X, Y, data_range):
        ms_ssim_val = self.ms_ssim(X, Y, data_range)
        l1_loss_val = self.l1_loss(X, Y)
        mixed_loss_val = self.alpha * ms_ssim_val + (1 - self.alpha) * l1_loss_val
        return mixed_loss_val

class CustomFocalLoss(nn.Module):
    """
    Focal loss (with MS-SSIM loss) module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, gamma: float = 2.0):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            gamma: factor for focal loss.
        """
        super(CustomFocalLoss, self).__init__()
        self.gamma = gamma
        self.ms_ssim = SSIMLoss(win_size=win_size, k1=k1, k2=k2)

    def forward(self, X, Y, data_range):
        ms_ssim_val = self.ms_ssim(X, Y, data_range)
        focal_loss_val = (1 - (1 - ms_ssim_val) ** self.gamma) * ms_ssim_val
        return focal_loss_val

class IndexBasedWeightedLoss(nn.Module):
    """
    Index-based weighted MS-SSIM loss module.
    """
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, max_num_slices: int = 22):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            max_num_slices: Maximum number of slices.
        """
        super(IndexBasedWeightedLoss, self).__init__()
        self.weight_list = []
        self.set_weight(max_num_slices)
        self.ms_ssim = SSIMLoss(win_size=win_size, k1=k1, k2=k2)

    def forward(self, X, Y, data_range, slice_idx):
        ms_ssim_val = self.ms_ssim(X, Y, data_range)
        ret = 0
        for idx in slice_idx:
            ret += ms_ssim_val * self.weight_list[idx]
        ret /= len(slice_idx)
        return ret

    def set_weight(self, num_slices):
        for i in range(num_slices):
            # assume that the weight is prop to size of mask which is prop to cos^2
            self.weight_list.append(math.cos(i * math.pi / (num_slices * 2)) ** 2)

class MixIndexL1Loss(nn.Module):
    """
    Mixed loss module. (Index-based weighted MS-SSIM + L1)
    !!!! Not yet has been added to the training script. !!!!
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03, alpha: float = 0.84, max_num_slices: int = 22):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            alpha: mixing factor of SSIM and L1 loss.
        """
        super(MixIndexL1Loss, self).__init__()
        self.alpha = alpha
        self.id_ms_ssim = IndexBasedWeightedLoss(win_size=win_size, k1=k1, k2=k2, max_num_slices=max_num_slices)
        self.l1_loss = nn.L1Loss(reduction='mean')

    def forward(self, X, Y, data_range, slice_idx):
        id_ms_ssim_val = self.id_ms_ssim(X, Y, data_range, slice_idx)
        l1_loss_val = self.l1_loss(X, Y)
        mixed_loss_val = self.alpha * id_ms_ssim_val + (1 - self.alpha) * l1_loss_val
        return mixed_loss_val