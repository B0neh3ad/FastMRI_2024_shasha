from typing import Sequence
from fastmri.data.subsample import EquiSpacedMaskFunc
from math import exp
import numpy as np
import torch


class MaskAugmentor:
    def __init__(self,
                 hparams,
                 current_epoch_fn,
                 center_fractions: Sequence[float],
                 accelerations: Sequence[int],
                 allow_any_combination: bool = False,
                 ):
        self.current_epoch_fn = current_epoch_fn
        self.hparams = hparams
        self.aug_on = hparams.mask_aug_on
        self.weight = hparams.aug_weight_mask
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()
        self.mask_func = EquiSpacedMaskFunc(center_fractions, accelerations, allow_any_combination)

    def __call__(self, shape, original_mask):
        """
        Generates augmented kspace mask.
        shape: Shape of k-space (coils, rows, cols, 2).
        original_mask: [torch tensor] original kspace mask with shape [1, 1, W, 1].
        """
        # Set augmentation probability
        if self.aug_on:
            p = self.schedule_p()
            self.set_augmentation_strength(p)
        else:
            p = 0.0

        if self.aug_on and p > 0.0:
            mask, _ = self.mask_func(shape)
            mask = mask.byte()
        else:
            mask = original_mask

        return mask

    def random_apply(self):
        if self.rng.uniform() < self.weight * self.augmentation_strength:
            return True
        else:
            return False

    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    def schedule_p(self):
        """
        Schedule the augmentation according to the self.aug_schedule
        """
        D = self.hparams.aug_delay
        T = self.hparams.num_epochs
        t = self.current_epoch_fn()
        p_max = self.hparams.aug_strength

        if t < D:
            return 0.0
        else:
            if self.hparams.aug_schedule == 'constant':
                p = p_max
            elif self.hparams.aug_schedule == 'ramp':
                p = (t - D) / (T - D) * p_max
            elif self.hparams.aug_schedule == 'exp':
                c = self.hparams.aug_exp_decay / (T - D)  # Decay coefficient
                p = p_max / (1 - exp(-(T - D) * c)) * (1 - exp(-(t - D) * c))
            return p