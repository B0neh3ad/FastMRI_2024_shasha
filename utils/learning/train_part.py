import glob
import shutil
import numpy as np
import torch
import time
import requests
from tqdm import tqdm
import wandb

from collections import defaultdict
from utils.data.load_data import create_data_loaders
from utils.common.utils import save_reconstructions, ssim_loss
from utils.common.loss_function import SSIMLoss, MixedLoss, CustomFocalLoss, IndexBasedWeightedLoss
from utils.model.dircn.dircn import DIRCN
from utils.model.varnet.varnet import VarNet

import os

from utils.mraugment.data_augment import DataAugmentor
from utils.mraugment.mask_augment import MaskAugmentor

def train_epoch(args, epoch, model, data_loader, optimizer, loss_type):
    model.train()
    start_epoch = start_iter = time.perf_counter()
    len_loader = len(data_loader)
    total_loss = 0.
    # gradient scaling
    scaler = torch.GradScaler(device='cuda')

    for iter, data in enumerate(tqdm(data_loader)):
        # TODO: slice_idx가 높은 데이터들은 점점 제외하기
        mask, masked_kspace, target, maximum, _, slice_idx = data
        mask = mask.cuda(non_blocking=True)
        masked_kspace = masked_kspace.cuda(non_blocking=True) # undersampled kspace converted in DataTransform object
        target = target.cuda(non_blocking=True)
        maximum = maximum.cuda(non_blocking=True)

        if args.amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(masked_kspace, mask)
                if type(loss_type) == IndexBasedWeightedLoss:
                    loss = loss_type(output, target, maximum, slice_idx=slice_idx)
                else:
                    loss = loss_type(output, target, maximum)
                loss /= args.iters_to_grad_acc
        else:
            output = model(masked_kspace, mask)
            if type(loss_type) == IndexBasedWeightedLoss:
                loss = loss_type(output, target, maximum, slice_idx=slice_idx)
            else:
                loss = loss_type(output, target, maximum)
            loss /= args.iters_to_grad_acc

        optimizer.zero_grad()

        if args.amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # gradient accumulation
        if (iter + 1) % args.iters_to_grad_acc == 0 or iter == len_loader - 1:
            if args.amp:
                scaler.unscale_(optimizer)
            if args.grad_clip_on:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            if args.amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            total_loss += loss.item() * (args.iters_to_grad_acc if iter < len_loader - 1 else len_loader % args.iters_to_grad_acc)

        if (iter + 1) % args.report_interval == 0:
            tqdm.write(
                f'Epoch = [{epoch:3d}/{args.num_epochs:3d}] '
                f'Iter = [{iter:4d}/{len(data_loader):4d}] '
                f'Loss = {loss.item() * args.iters_to_grad_acc:.4g} '
                f'Time = {time.perf_counter() - start_iter:.4f}s',
            )
            if args.wandb_on:
                wandb.log({
                    "train_iter_loss": loss.item() * args.iters_to_grad_acc,
                    "train_interval_time": time.perf_counter() - start_iter
                })
            if args.debug:
                break
            start_iter = time.perf_counter()
    total_loss = total_loss / len_loader
    return total_loss, time.perf_counter() - start_epoch


def validate(args, model, data_loader):
    model.eval()
    reconstructions = defaultdict(dict)
    targets = defaultdict(dict)
    start = time.perf_counter()

    with torch.no_grad():
        for iter, data in enumerate(tqdm(data_loader)):
            mask, kspace, target, _, fnames, slices = data
            kspace = kspace.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
            output = model(kspace, mask)

            for i in range(output.shape[0]):
                reconstructions[fnames[i]][int(slices[i])] = output[i].cpu().numpy()
                targets[fnames[i]][int(slices[i])] = target[i].numpy()

            if args.debug:
                break

    for fname in reconstructions:
        reconstructions[fname] = np.stack(
            [out for _, out in sorted(reconstructions[fname].items())]
        )
    for fname in targets:
        targets[fname] = np.stack(
            [out for _, out in sorted(targets[fname].items())]
        )
    metric_loss = sum([ssim_loss(targets[fname], reconstructions[fname]) for fname in reconstructions])
    num_subjects = len(reconstructions)
    return metric_loss, num_subjects, reconstructions, targets, None, time.perf_counter() - start


def save_model(args, exp_dir, epoch, model, optimizer, best_val_loss, is_new_best):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'exp_dir': exp_dir
        },
        f=exp_dir / 'model.pt'
    )
    if is_new_best:
        shutil.copyfile(exp_dir / f'model.pt', exp_dir / f'best_model_epoch{epoch}.pt')
        shutil.copyfile(exp_dir / f'best_model_epoch{epoch}.pt', exp_dir / f'best_model.pt')
    try:
        # save model weights
        pt_files = glob.glob(os.path.join(args.result_dir_path, "**", "*.pt"), recursive=True)
        for file in pt_files:
            wandb.save(file)
    except wandb.errors.Error as e:
        print('checkpoint files are not saved since wandb.init() is not called')


def download_model(url, fname):
    response = requests.get(url, timeout=10, stream=True)

    chunk_size = 8 * 1024 * 1024  # 8 MB chunks
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    progress_bar = tqdm(
        desc="Downloading state_dict",
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
    )

    with open(fname, "wb") as fh:
        for chunk in response.iter_content(chunk_size):
            progress_bar.update(len(chunk))
            fh.write(chunk)


def load_checkpoint(model, optimizer, exp_dir):
    checkpoint_path = exp_dir / 'model.pt'
    start_epoch = 0
    best_val_loss = float('inf')

    if checkpoint_path.is_file():
        print(f"Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
        best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
    else:
        print(f"No checkpoint found at '{checkpoint_path}', starting from scratch.")

    return model, optimizer, start_epoch, best_val_loss


        
def train(args):
    if args.wandb_on:
        resume_option = "must" if args.wandb_run_id else None
        wandb.init(
            project="FastMRI_2024_shasha",
            id=args.wandb_run_id,
            resume=resume_option,
            config={
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "loss_type": args.loss,
                "learning_rate": args.lr,
                "net_name": args.net_name,
                "cascade": args.cascade,
                "chans": args.chans,
                "sens_chans": args.sens_chans,
                "grad_clip_on": args.grad_clip_on,
                "grad_clip": args.grad_clip,
                "iters_to_grad_acc": args.iters_to_grad_acc,
                "aug_on": args.aug_on,
                "aug_strength": args.aug_strength,
                "mask_aug_on": args.mask_aug_on,
                "lr_scheduler_on": args.lr_scheduler_on,
                "patience": args.patience
            }
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("val_loss", step_metric="epoch")
        wandb.define_metric("train_time", step_metric="epoch")
        wandb.define_metric("val_time", step_metric="epoch")

    device = torch.device(f'cuda:{args.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)
    print('Current cuda device: ', torch.cuda.current_device())

    # model
    if args.net_name == "dircn":
        model = DIRCN(num_cascades=30, # args.cascade
                    n=20, # args.chans
                    sense_n=12 # args.sens_chans
                )
    else:
        model = VarNet(num_cascades=args.cascade,
                   chans=args.chans, 
                   sens_chans=args.sens_chans)
    model.to(device=device)

    # loss
    if args.loss == "mixed":
        loss_type = MixedLoss(alpha=args.alpha).to(device=device)
    elif args.loss == "focal":
        loss_type = CustomFocalLoss(gamma=args.gamma).to(device=device)
    elif args.loss == "index_based":
        loss_type = IndexBasedWeightedLoss(max_num_slices=args.max_num_slices).to(device=device)
    else:
        loss_type = SSIMLoss().to(device=device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.patience, verbose=True)

    best_val_loss = 1.
    start_epoch = 0

    # Load checkpoint only if wandb_run_id is not None
    if args.wandb_run_id is not None:
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(model, optimizer, args.exp_dir)

    epoch = start_epoch

    train_augmentor = DataAugmentor(args, lambda: epoch)
    val_augmentor = DataAugmentor(args, lambda: epoch, is_validation=True)
    mask_augmentor = MaskAugmentor(args, lambda: epoch,
                                   center_fractions=[0.08, 0.083],
                                   accelerations=[6, 7, 9],
                                   allow_any_combination=True)

    train_loader = create_data_loaders(data_path = args.data_path_train, args = args, shuffle=True,
                                       augmentor=train_augmentor if args.aug_on else None, mask_augmentor=mask_augmentor if args.mask_aug_on else None,
                                       current_epoch_fn=lambda: epoch)
    val_loader = create_data_loaders(data_path = args.data_path_val, args = args,
                                     augmentor=val_augmentor if args.aug_on else None, mask_augmentor=mask_augmentor if args.mask_aug_on else None,
                                     current_epoch_fn=lambda: epoch)
    
    val_loss_log = np.empty((0, 2))

    if args.wandb_on:
        # save code
        wandb.save("*.py")

    for epoch in range(start_epoch, args.num_epochs):
        print(f'Epoch #{epoch:2d} ............... {args.net_name} ...............')
        
        train_loss, train_time = train_epoch(args, epoch, model, train_loader, optimizer, loss_type)

        val_loss, num_subjects, reconstructions, targets, inputs, val_time = validate(args, model, val_loader)
        if args.lr_scheduler_on:
            scheduler.step(val_loss)

        val_loss_log = np.append(val_loss_log, np.array([[epoch, val_loss]]), axis=0)
        file_path = os.path.join(args.val_loss_dir, "val_loss_log")
        np.save(file_path, val_loss_log)
        print(f"loss file saved! {file_path}")

        train_loss = torch.tensor(train_loss).cuda(non_blocking=True)
        val_loss = torch.tensor(val_loss).cuda(non_blocking=True)
        num_subjects = torch.tensor(num_subjects).cuda(non_blocking=True)

        val_loss = val_loss / num_subjects

        is_new_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        save_model(args, args.exp_dir, epoch + 1, model, optimizer, best_val_loss, is_new_best)
        print(
            f'Epoch = [{epoch + 1:4d}/{args.num_epochs:4d}] TrainLoss = {train_loss:.4g} '
            f'ValLoss = {val_loss:.4g} TrainTime = {train_time:.4f}s ValTime = {val_time:.4f}s',
        )
        if args.wandb_on:
            wandb.log({"epoch": epoch})
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "train_time": train_time, "val_time": val_time})

        if is_new_best:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@NewRecord@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            start = time.perf_counter()
            save_reconstructions(reconstructions, args.val_dir, targets=targets, inputs=inputs)
            print(
                f'ForwardTime = {time.perf_counter() - start:.4f}s',
            )

    if args.wandb_on:
        # save log file
        npy_files = glob.glob(os.path.join(args.result_dir_path, '**', '*.npy'), recursive=True)
        for file in npy_files:
            wandb.save(file)

        wandb.finish()
