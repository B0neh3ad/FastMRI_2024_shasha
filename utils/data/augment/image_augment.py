from math import exp, sqrt
import torchvision.transforms.functional as TF
import numpy as np
import torch


"""
These functions are used in the i2i training script. (train2)
- CutMixUp: CutMix and MixUp augmentation module.
- AugmentationPipeline: Augmentation pipeline module.
- ImageDataAugmentor: High-level class encompassing the augmentation pipeline and augmentation probability scheduling.
"""

class CutMixUp:
    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict ={
                      'cutmix': hparams.aug_weight_cutmix,
                      'mixup': hparams.aug_weight_mixup
        }
        self.aug_on = hparams.aug_on
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def __call__(self, images, targets):
        # Set augmentation probability
        if self.aug_on:
            self.set_augmentation_strength(self.schedule_p())
        else:
            self.set_augmentation_strength(0)

        # Augment if needed
        if self.aug_on and self.augmentation_strength > 0.0:
            images, targets = self.augment(images, targets)

        return images, targets

    def augment_image_batch(self, im_batch):
        """
        Augment the batches of images with random transformations
        :param im_batch: the image batch to be augmented with shape [B, 4, H, W]
        (image_input, reconstruction, image_grappa, image_label)
        :return: the augmented image batch with shape [B, 4, H, W]

        주의!!
        cutmix나 mixup의 사용을 위해서는 batch size가 2 이상이어야 한다.
        단일 batch로는 사용할 수 없다.
        """
        if im_batch.shape[0] > 1:
            if self.random_apply('cutmix'):
                im = self._cutmix(im_batch)
            if self.random_apply('mixup'):
                im = self._mixup(im_batch)
        return im_batch

    def augment(self, images, targets, is_validation=False):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [B, 3, H, W]
        target: torch tensor of shape [B, H, W]
        """
        im_batch = torch.cat([images, targets.unsqueeze(1)], dim=1)
        augmented_im_batch = self.augment_image_batch(im_batch)
        augmented_images = augmented_im_batch[:, :-1, :, :]
        augmented_targets = augmented_im_batch[:, -1, :, :]

        return augmented_images, augmented_targets

    def random_apply(self, transform_name):
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else:
            return False

    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    def _cutmix(self, im):
        """
        perform cutmix augmentation with batches of images
        :param im: input images to be augmented with shape [B, 4, H, W]
        :return output: augmented images with shape [B, 4, H, W]
        """
        B, C, H, W = im.shape
        lam = float(self.rng.beta(self.hparams.aug_mixup_alpha, self.hparams.aug_mixup_alpha))

        # Compute the bounding box for cut region
        r_x = torch.randint(W, size=(1,))
        r_y = torch.randint(H, size=(1,))

        r = 0.5 * sqrt(1.0 - lam)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        # shuffle images
        rolled = im.roll(1, 0)
        output = im.clone()
        output[..., y1:y2, x1:x2] = rolled[..., y1:y2, x1:x2]

        return output

    def _mixup(self, im):
        """
        perform mixup augmentation with batches of images
        :param im: input images to be augmented with shape [B, 4, H, W]
        :return output: augmented images with shape [B, 4, H, W]
        """
        B, C, H, W = im.shape
        lam = float(self.rng.beta(self.hparams.aug_cutmix_alpha, self.hparams.aug_cutmix_alpha))

        # shuffle images
        output = im.roll(1, 0).mul_(1.0 - lam).add_(im.mul(lam))

        return output

    def schedule_p(self):
        """
        Schedule the augmentation according to the self.aug_schedule
        return: augmentation strength in [0, 1]
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

    @staticmethod
    def add_augmentation_specific_args(parser):
        parser.add_argument(
            '--aug_weight_cutmix',
            type=float,
            default=1.0,
            help='Weight of translation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_mixup',
            type=float,
            default=1.0,
            help='Weight of arbitrary rotation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_cutmix_alpha',
            type=float,
            default=0.4,
            help='Alpha value for cutmix augmentation'
        )
        parser.add_argument(
            '--aug_mixup_alpha',
            type=float,
            default=0.4,
            help='Alpha value for mixup augmentation'
        )

        return parser

class AugmentationPipeline:
    """
    Augmentation pipeline module.

    The following transformations are supported:
    - Translation
    - Rotation
    - Shearing
    - Scaling
    - Horizontal Scaling
    - Horizontal Flip
    - Vertical Flip
    """
    def __init__(self, hparams):
        self.hparams = hparams
        self.weight_dict ={
                      'translation': hparams.aug_weight_translation,
                      'rotation': hparams.aug_weight_rotation,
                      'scaling': hparams.aug_weight_scaling,
                      'scalex': hparams.aug_weight_scalex,
                      'shearing': hparams.aug_weight_shearing,
                      'fliph': hparams.aug_weight_fliph,
                      'flipv': hparams.aug_weight_flipv,
        }
        self.augmentation_strength = 0.0
        self.rng = np.random.RandomState()

    def augment_image(self, im):
        """
        Augment the image with random transformations
        :param im: the image to be augmented with shape [4, H, W]
        (image_input, reconstruction, image_grappa, image_label)
        :return: the augmented image with shape [4, H, W]
        """
        if self.random_apply('fliph'):
            im = TF.hflip(im)

        if self.random_apply('flipv'):
            im = TF.vflip(im)

        if self.random_apply('translation'):
            h, w = im.shape[-2:]
            t_x = self.rng.uniform(-self.hparams.aug_max_translation_x, self.hparams.aug_max_translation_x)
            t_x = int(t_x * h)
            t_y = self.rng.uniform(-self.hparams.aug_max_translation_y, self.hparams.aug_max_translation_y)
            t_y = int(t_y * w)

            pad, top, left = self._get_translate_padding_and_crop(im, (t_x, t_y))
            im = TF.pad(im, padding=pad, padding_mode='reflect')
            im = TF.crop(im, top, left, h, w)

        # Rotation
        if self.random_apply('rotation'):
            interp = True
            rot = self.rng.uniform(-self.hparams.aug_max_rotation, self.hparams.aug_max_rotation)
        else:
            rot = 0.

        # Shearing
        if self.random_apply('shearing'):
            interp = True
            shear_x = self.rng.uniform(-self.hparams.aug_max_shearing_x, self.hparams.aug_max_shearing_x)
            shear_y = self.rng.uniform(-self.hparams.aug_max_shearing_y, self.hparams.aug_max_shearing_y)
        else:
            shear_x, shear_y = 0., 0.

        # Scaling
        if self.random_apply('scaling'):
            interp = True
            scale = self.rng.uniform(1 - self.hparams.aug_max_scaling, 1 + self.hparams.aug_max_scaling)
        else:
            scale = 1.

        # Horizontal Scaling
        if self.random_apply('scalex'):
            scale_x = self.rng.uniform(self.hparams.aug_min_scalex, self.hparams.aug_max_scalex)
        else:
            scale_x = 1.

        im = self._get_horizontally_scaled_image(im, scale_x)
        h, w = im.shape[-2:]
        pad = self._get_affine_padding_size(im, rot, scale, (shear_x, shear_y))
        im = TF.pad(im, padding=pad, padding_mode='reflect')
        im = TF.affine(im,
                       angle=rot,
                       scale=scale,
                       shear=[shear_x, shear_y],
                       translate=[0, 0],
                       interpolation=TF.InterpolationMode.BILINEAR
                       )
        im = TF.center_crop(im, [h, w])
        return im

    def augment(self, image, target):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [3, H, W]
        target: torch tensor of shape [H, W]
        """
        im = torch.cat([image, target.unsqueeze(0)], dim=0)
        augmented_im = self.augment_image(im)
        augmented_image = augmented_im[:-1]
        augmented_target = augmented_im[-1]

        return augmented_image, augmented_target

    def random_apply(self, transform_name):
        if self.rng.uniform() < self.weight_dict[transform_name] * self.augmentation_strength:
            return True
        else:
            return False

    def set_augmentation_strength(self, p):
        self.augmentation_strength = p

    @staticmethod
    def _get_affine_padding_size(im, angle, scale, shear):
        """
        Calculates the necessary padding size before applying the
        general affine transformation. The output image size is determined based on the
        input image size and the affine transformation matrix.
        """
        h, w = im.shape[-2:]
        corners = [
            [-h/2, -w/2, 1.],
            [-h/2, w/2, 1.],
            [h/2, w/2, 1.],
            [h/2, -w/2, 1.]
        ]
        mx = torch.tensor(TF._get_inverse_affine_matrix([0.0, 0.0], -angle, [0, 0], scale, [-s for s in shear])).reshape(2,3)
        corners = torch.cat([torch.tensor(c).reshape(3,1) for c in corners], dim=1)
        tr_corners = torch.matmul(mx, corners)
        all_corners = torch.cat([tr_corners, corners[:2, :]], dim=1)
        bounding_box = all_corners.amax(dim=1) - all_corners.amin(dim=1)
        px = torch.clip(torch.floor((bounding_box[0] - h) / 2), min=0.0, max=h-1)
        py = torch.clip(torch.floor((bounding_box[1] - w) / 2),  min=0.0, max=w-1)
        return [int(py.item()), int(px.item())]

    @staticmethod
    def _get_translate_padding_and_crop(im, translation):
        t_x, t_y = translation
        h, w = im.shape[-2:]
        pad = [0, 0, 0, 0]
        if t_x >= 0:
            pad[3] = min(t_x, h - 1) # pad bottom
            top = pad[3]
        else:
            pad[1] = min(-t_x, h - 1) # pad top
            top = 0
        if t_y >= 0:
            pad[0] = min(t_y, w - 1) # pad left
            left = 0
        else:
            pad[2] = min(-t_y, w - 1) # pad right
            left = pad[2]
        return pad, top, left

    @staticmethod
    def _get_horizontally_scaled_image(im, scale_x: float = 1.0):
        h, w = im.shape[-2:]
        scaled_w = int(scale_x * w)
        if scale_x < 1.0:
            scaled_img = TF.resize(im, [h, scaled_w])
            new_im = torch.zeros(im.shape)
            padding = (w - scaled_w) // 2
            new_im[..., padding:padding + scaled_w] = scaled_img
        else:
            crop = (scaled_w - w) // 2
            new_im = im[..., crop:crop + scaled_w]
        return new_im

class ImageDataAugmentor:
    """
    High-level class encompassing the augmentation pipeline and augmentation
    probability scheduling. A DataAugmentor instance can be initialized in the
    main training code and passed to the DataTransform to be applied
    to the training data.
    """

    def __init__(self, hparams, current_epoch_fn):
        """
        hparams: refer to the arguments below in add_augmentation_specific_args
        current_epoch_fn: this function has to return the current epoch as an integer
        and is used to schedule the augmentation probability.
        """
        self.current_epoch_fn = current_epoch_fn
        self.hparams = hparams
        self.aug_on = hparams.aug_on
        if self.aug_on:
            self.augmentation_pipeline = AugmentationPipeline(hparams)

    def __call__(self, image, target):
        """
        Generates augmented kspace and corresponding augmented target pair.
        kspace: torch tensor of shape [3, H, W]
        target: torch tensor of shape [H, W]
        """
        # Set augmentation probability
        if self.aug_on:
            p = self.schedule_p()
            self.augmentation_pipeline.set_augmentation_strength(p)
        else:
            p = 0.0

        # Augment if needed
        if self.aug_on and p > 0.0:
            image, target = self.augmentation_pipeline.augment(image=image,
                                                                target=target)

        return image, target

    def schedule_p(self):
        """
        Schedule the augmentation according to the self.aug_schedule
        return: augmentation strength in [0, 1]
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

    @staticmethod
    def add_augmentation_specific_args(parser):
        parser.add_argument(
            '--aug_on',
            default=False,
            help='This switch turns data augmentation on.',
            action='store_true'
        )
        # --------------------------------------------
        # Related to augmentation strenght scheduling
        # --------------------------------------------
        parser.add_argument(
            '--aug_schedule',
            type=str,
            default='exp',
            help='Type of data augmentation strength scheduling. Options: constant, ramp, exp'
        )
        parser.add_argument(
            '--aug_delay',
            type=int,
            default=0,
            help='Number of epochs at the beginning of training without data augmentation. The schedule in --aug_schedule will be adjusted so that at the last epoch the augmentation strength is --aug_strength.'
        )
        parser.add_argument(
            '--aug_strength',
            type=float,
            default=0.0,
            help='Augmentation strength, combined with --aug_schedule determines the augmentation strength in each epoch'
        )
        parser.add_argument(
            '--aug_exp_decay',
            type=float,
            default=5.0,
            help='Exponential decay coefficient if --aug_schedule is set to exp. 1.0 is close to linear, 10.0 is close to step function'
        )

        # --------------------------------------------
        # Related to transformation probability weights
        # --------------------------------------------
        parser.add_argument(
            '--aug_weight_translation',
            type=float,
            default=1.0,
            help='Weight of translation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_rotation',
            type=float,
            default=1.0,
            help='Weight of arbitrary rotation probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_shearing',
            type=float,
            default=1.0,
            help='Weight of shearing probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_scaling',
            type=float,
            default=1.0,
            help='Weight of scaling probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_scalex',
            type=float,
            default=1.0,
            help='Weight of horizontal scaling probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_fliph',
            type=float,
            default=1.0,
            help='Weight of horizontal flip probability. Augmentation probability will be multiplied by this constant'
        )
        parser.add_argument(
            '--aug_weight_flipv',
            type=float,
            default=1.0,
            help='Weight of vertical flip probability. Augmentation probability will be multiplied by this constant'
        )

        # --------------------------------------------
        # Related to transformation limits
        # --------------------------------------------
        parser.add_argument(
            '--aug_max_translation_x',
            type=float,
            default=0.125,
            help='Maximum translation applied along the x axis as fraction of image width'
        )
        parser.add_argument(
            '--aug_max_translation_y',
            type=float,
            default=0.125,
            help='Maximum translation applied along the y axis as fraction of image height'
        )
        parser.add_argument(
            '--aug_max_rotation',
            type=float,
            default=10.,
            help='Maximum rotation applied in either clockwise or counter-clockwise direction in degrees.'
        )
        parser.add_argument(
            '--aug_max_shearing_x',
            type=float,
            default=15.0,
            help='Maximum shearing applied in either positive or negative direction in degrees along x axis.'
        )
        parser.add_argument(
            '--aug_max_shearing_y',
            type=float,
            default=15.0,
            help='Maximum shearing applied in either positive or negative direction in degrees along y axis.'
        )
        parser.add_argument(
            '--aug_max_scaling',
            type=float,
            default=0.25,
            help='Maximum scaling applied as fraction of image dimensions. If set to s, a scaling factor between 1.0-s and 1.0+s will be applied.'
        )
        parser.add_argument(
            '--aug_min_scalex',
            type=float,
            default=0.9,
            help='Minimum horizontal scaling applied as fraction of image dimensions.'
        )
        parser.add_argument(
            '--aug_max_scalex',
            type=float,
            default=1.0,
            help='Maximum horizontal scaling applied as fraction of image dimensions.'
        )

        return parser