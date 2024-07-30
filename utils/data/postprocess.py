from pathlib import Path
from dotenv import load_dotenv
import h5py
import torch
import torch.nn.functional as F
import numpy as np
import os
import glob
import argparse
from tqdm import tqdm


"""
this script is for postprocessing the reconstructed images (not used in the training)
- Dithering: Add gaussian noise to the image
- Postprocessing: Apply dithering to the reconstructed images
"""

class Dithering:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, image):
        """
        return gaussian noised image, whose noise is generated by the method in
        'End-to-End Variational Networks for Accelerated MRI Reconstruction (2020)'

        :param image: np array  with shape [hegiht, width]
        :return: gaussian noised image
        """
        noisy_image = np.empty(image.shape)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                lx = max(0, i - 5)
                rx = min(image.shape[0] - 1, i + 5)
                ly = max(0, j - 5)
                ry = min(image.shape[1] - 1, j + 5)
                std = self.sigma * np.median(image[lx:rx + 1, ly:ry + 1])
                noisy_image[i, j] = image[i, j] + np.random.normal(0, std)
        return noisy_image

class Postprocessing:
    def __init__(self, dithering_factor=0.02):
        self.dithering = Dithering(dithering_factor)

    def __call__(self, recon_dir, out_dir):
        public_recon_dir = recon_dir / 'public' / '*.h5'
        public_out_dir = out_dir / 'public'
        public_out_dir.mkdir(exist_ok=True, parents=True)
        self.export(public_recon_dir, public_out_dir)

        private_recon_dir = recon_dir / 'private' / '*.h5'
        private_out_dir = out_dir / 'private'
        private_out_dir.mkdir(exist_ok=True, parents=True)
        self.export(private_recon_dir, private_out_dir)

    def export(self, recon_dir, out_dir):
        for hf_path in tqdm(glob.glob(str(recon_dir))):
            hf = h5py.File(hf_path, 'r')
            fname = os.path.basename(hf_path)
            recons = hf['reconstruction']
            new_recons = np.empty(recons.shape)

            # dithering
            for i_slice in range(recons.shape[0]):
                print('hello?')
                new_recons[i_slice] = self.dithering(recons[i_slice])

            # export post-processed file
            with h5py.File(out_dir / fname, 'w') as f:
                f.create_dataset('reconstruction', data=new_recons)

if __name__ == '__main__':
    load_dotenv()
    result_dir_path = Path(os.environ['RESULT_DIR_PATH'])
    recon_dir = result_dir_path / 'test_Varnet' / 'reconstructions_leaderboard'
    out_dir = result_dir_path / 'test_Varnet' / 'postprocessed_leaderboard'

    postprocessing = Postprocessing()
    postprocessing(recon_dir, out_dir)