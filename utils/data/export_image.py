import glob
import os
import h5py
from PIL import Image
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm

def export_image(img_dir, export_dir, keys):
    for key in keys:
        temp_export_dir = export_dir
        if len(keys) > 1:
            export_dir = os.path.join(export_dir, key)
        os.makedirs(export_dir, exist_ok=True)
        print(img_dir)
        for file in tqdm(glob.glob(os.path.join(img_dir, '*.h5'))):
            with h5py.File(file, 'r') as f:
                image = f[key]
                fname = os.path.basename(file).replace('.h5', '')
                for idx, image_slice in enumerate(image):
                    max_val = np.max(image_slice)
                    pil_image = Image.fromarray(image_slice / max_val * 255).convert("L")
                    pil_image.save(os.path.join(export_dir, f'slice{idx}_{fname}.png'))
        export_dir = temp_export_dir

if __name__ == '__main__':
    load_dotenv()
    # data_dir_path = os.environ['DATA_DIR_PATH']
    # export_dir_path = '/mnt/c/Users/js1044k/Downloads/Data_png'
    #
    # # train
    # train_img_dir = os.path.join(data_dir_path, 'train', 'image')
    # train_export_dir = os.path.join(export_dir_path, 'train')
    # export_image(train_img_dir, train_export_dir, keys=['image_input', 'image_grappa', 'image_label'])
    #
    # # val
    # val_img_dir = os.path.join(data_dir_path, 'val', 'image')
    # val_export_dir = os.path.join(export_dir_path, 'val')
    # export_image(val_img_dir, val_export_dir, keys=['image_input', 'image_grappa', 'image_label'])
    #
    # # leaderboard
    # acc_list = ['acc5', 'acc9']
    # for acc in acc_list:
    #     lb_img_dir = os.path.join(data_dir_path, 'leaderboard', acc, 'image')
    #     lb_export_dir = os.path.join(export_dir_path, 'leaderboard', acc)
    #     export_image(lb_img_dir, lb_export_dir, keys=['image_input', 'image_grappa', 'image_label'])

    result_dir_path = os.path.join(os.environ['RESULT_DIR_PATH'], 'test_Varnet')
    export_dir_path = '/mnt/c/Users/js1044k/Downloads/result_png'

    # reconstructions_val
    recon_val_img_dir = os.path.join(result_dir_path, 'reconstructions_val')
    recon_val_export_dir = os.path.join(export_dir_path, 'reconstructions_val')
    export_image(recon_val_img_dir, recon_val_export_dir, keys=['reconstruction'])

    # reconstructions_leaderboard
    cat_list = ['public', 'private']
    for cat in cat_list:
        recon_lb_img_dir = os.path.join(result_dir_path, 'reconstructions_leaderboard', cat)
        recon_lb_export_dir = os.path.join(export_dir_path, 'reconstructions_leaderboard', cat)
        export_image(recon_lb_img_dir, recon_lb_export_dir, keys=['reconstruction'])