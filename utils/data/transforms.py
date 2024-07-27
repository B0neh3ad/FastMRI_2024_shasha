import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class KspaceDataTransform:
    def __init__(self, isforward, max_key, augmentor = None, mask_augmentor = None):
        self.isforward = isforward
        self.max_key = max_key

        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False

        if mask_augmentor is not None:
            self.use_mask_augment = True
            self.mask_augmentor = mask_augmentor
        else:
            self.use_mask_augment = False

    def __call__(self, mask, kspace, target, attrs, fname, slice):
        """
        Args:
            mask: Mask from the test dataset.
            kspace: Input k-space of shape (num_coils, rows, cols) for
                multi-coil data.
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice: The slice index.
        Returns:
            mask: The applied sampling 2D mask
            masked_kspace: k-space after applying sampling mask.
            target: The target image (if applicable).
            maximum: Maximum image value.
            fname: File name.
            slice: The slice index.
        """
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        kspace = to_tensor(kspace)
        mask = to_tensor(mask)

        kspace = self._complex_to_float(kspace)
        mask = mask.reshape(1, 1, kspace.shape[-2], 1).float().byte()

        # Apply augmentations if needed
        if self.use_augment:
            if self.augmentor.schedule_p() > 0.0:
                kspace, target = self.augmentor(kspace, target.shape)

        # Apply mask augmentations if needed
        if self.use_mask_augment:
            mask = self.mask_augmentor(kspace.shape, mask)

        masked_kspace = kspace * mask
        return mask, masked_kspace, target, maximum, fname, slice

    @staticmethod
    def _complex_to_float(complex_tensor):
        return torch.stack((complex_tensor.real, complex_tensor.imag), dim=-1)


class ImageDataTransform:
    def __init__(self, isforward, max_key, augmentor = None):
        self.isforward = isforward
        self.max_key = max_key

        if augmentor is not None:
            self.use_augment = True
            self.augmentor = augmentor
        else:
            self.use_augment = False

    def __call__(self, image, target, attrs, fname, slice):
        """
        Args:
            image: input image with shape [3, H, W]
            target: Target image.
            attrs: Acquisition related information stored in the HDF5 object.
            fname: File name.
            slice: The slice index.
        Returns:
            image: input image with shape [3, H, W]
            target: The target image (if applicable).
            maximum: Maximum image value.
            fname: File name.
            slice: The slice index.
        """
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1

        image = to_tensor(image)
        image, target = self.augmentor(image, target)

        return image, target, maximum, fname, slice