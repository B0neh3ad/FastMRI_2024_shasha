import h5py
import random
from utils.data.transforms import KspaceDataTransform, ImageDataTransform
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

class KspaceSliceData(Dataset):
    def __init__(self, root, transform, input_key, target_key, forward=False, current_epoch_fn=None):
        self.transform = transform
        self.input_key = input_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.kspace_examples = []
        self.current_epoch_fn = current_epoch_fn

        if not forward:
            image_files = list(Path(root / "image").iterdir())
            for fname in sorted(image_files):
                num_slices = self._get_metadata(fname)

                self.image_examples += [
                    (fname, slice_ind) for slice_ind in range(num_slices) if self.should_include(fname, slice_ind, num_slices)
                ]

        kspace_files = list(Path(root / "kspace").iterdir())
        for fname in sorted(kspace_files):
            num_slices = self._get_metadata(fname)

            self.kspace_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices) if self.should_include(fname, slice_ind, num_slices)
            ]


    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
            elif self.target_key in hf.keys():
                num_slices = hf[self.target_key].shape[0]
        return num_slices

    def should_include(self, fname, slice_ind, num_slices):
        """
        Determine whether a slice can be included in the dataset
        :param fname: name of the file
        :param slice_ind: index of the slice
        :param num_slices: total number of slices in the file
        """
        # evaluation mode
        if self.current_epoch_fn is None or not self.forward:
            return True

        epoch = self.current_epoch_fn()

        # exclude outlier
        if 'acc5_24' in fname.__str__():
            return False

        # TODO: epoch에 따라 slice_ind 큰 애들 점진적으로 빼도록
        return True

    def __len__(self):
        return len(self.kspace_examples)

    def __getitem__(self, i):
        if not self.forward:
            image_fname, _ = self.image_examples[i]
        kspace_fname, dataslice = self.kspace_examples[i]

        with h5py.File(kspace_fname, "r") as hf:
            if dataslice >= hf[self.input_key].shape[0]:
                raise IndexError(f"Requested slice {dataslice} exceeds the dataset size in {kspace_fname} which has {hf[self.input_key].shape[0]} slices.")
            kspace = hf[self.input_key][dataslice]
            mask = np.array(hf["mask"])
        if self.forward:
            target = -1
            attrs = -1
        else:
            with h5py.File(image_fname, "r") as hf:
                if dataslice >= hf[self.target_key].shape[0]:
                    raise IndexError(f"Requested slice {dataslice} exceeds the dataset size in {image_fname} which has {hf[self.target_key].shape[0]} slices.")
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        return self.transform(mask, kspace, target, attrs, kspace_fname.name, dataslice)

class ImageSliceData(Dataset):
    def __init__(self, data_path, recon_path, transform,
                 input_key="image_input",
                 recon_key="reconstruction",
                 grappa_key="image_grappa",
                 target_key="image_label",
                 forward=False,
                 current_epoch_fn=None
                 ):
        self.transform = transform
        self.input_key = input_key
        self.recon_key = recon_key
        self.grappa_key = grappa_key
        self.target_key = target_key
        self.forward = forward
        self.image_examples = []
        self.recon_examples = []
        self.current_epoch_fn = current_epoch_fn

        image_files = list(Path(data_path / "image").iterdir())
        for fname in sorted(image_files):
            num_slices = self._get_metadata(fname)

            self.image_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices) if
                self.should_include(fname, slice_ind, num_slices)
            ]

        recon_files = list(recon_path.iterdir())
        for fname in sorted(recon_files):
            num_slices = self._get_metadata(fname)

            self.recon_examples += [
                (fname, slice_ind) for slice_ind in range(num_slices) if
                self.should_include(fname, slice_ind, num_slices)
            ]

    def _get_metadata(self, fname):
        with h5py.File(fname, "r") as hf:
            if self.recon_key in hf.keys():
                num_slices = hf[self.recon_key].shape[0]
            elif self.input_key in hf.keys():
                num_slices = hf[self.input_key].shape[0]
        return num_slices

    def should_include(self, fname, slice_ind, num_slices):
        """
        Determine whether a slice can be included in the dataset
        :param fname: name of the file
        :param slice_ind: index of the slice
        :param num_slices: total number of slices in the file
        """
        # evaluation mode
        if self.current_epoch_fn is None or not self.forward:
            return True

        epoch = self.current_epoch_fn()

        # exclude outlier
        if 'acc5_24' in fname.__str__():
            return False

        # TODO: epoch에 따라 slice_ind 큰 애들 점진적으로 빼도록
        return True

    def __len__(self):
        return len(self.recon_examples)

    def __getitem__(self, i):
        image_fname, dataslice = self.image_examples[i]
        recon_fname, dataslice = self.recon_examples[i]

        with h5py.File(image_fname, "r") as hf:
            if dataslice >= hf[self.input_key].shape[0]:
                raise IndexError(f"Requested slice {dataslice} exceeds the dataset size in {image_fname} which has {hf[self.input_key].shape[0]} slices.")
            image_input = hf[self.input_key][dataslice]
            image_grappa = hf[self.grappa_key][dataslice]
            if self.forward:
                target = -1
                attrs = -1
            else:
                target = hf[self.target_key][dataslice]
                attrs = dict(hf.attrs)

        with h5py.File(recon_fname, "r") as hf:
            if dataslice >= hf[self.recon_key].shape[0]:
                raise IndexError(f"Requested slice {dataslice} exceeds the dataset size in {image_fname} which has {hf[self.recon_key].shape[0]} slices.")
            image_recon = hf[self.recon_key][dataslice]

        image = np.stack([image_input, image_recon, image_grappa])
        return self.transform(image, target, attrs, image_fname.name, dataslice)

def create_kspace_data_loaders(data_path, args, current_epoch_fn=None, augmentor=None, mask_augmentor=None, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = KspaceSliceData(
        root=data_path,
        transform=KspaceDataTransform(isforward, max_key_, augmentor=augmentor, mask_augmentor=mask_augmentor),
        input_key=args.input_key,
        target_key=target_key_,
        forward = isforward,
        current_epoch_fn = current_epoch_fn
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    return data_loader

def create_image_data_loaders(data_path, recon_path, args, current_epoch_fn=None, augmentor=None, shuffle=False, isforward=False):
    if isforward == False:
        max_key_ = args.max_key
        target_key_ = args.target_key
    else:
        max_key_ = -1
        target_key_ = -1

    data_storage = ImageSliceData(
        data_path=data_path,
        recon_path=recon_path,
        transform=ImageDataTransform(isforward, max_key_, augmentor=augmentor),
        input_key=args.input_key,
        recon_key=args.recon_key,
        grappa_key=args.grappa_key,
        target_key=target_key_,
        forward = isforward,
        current_epoch_fn = current_epoch_fn
    )

    data_loader = DataLoader(
        dataset=data_storage,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=4
    )
    return data_loader