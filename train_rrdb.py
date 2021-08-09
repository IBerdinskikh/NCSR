#!/usr/bin/python

import argparse
import os
import torch

import torch.optim as optim
import torch.nn as nn
from natsort import natsorted
import time
import numpy as np
import utils
from dataloaders.data_rgb import (
    get_training_data_SR,
    get_validation_data_SR,
    DistSampler,
    CPUPrefetcher,
    PrefetchDataLoader,
)
from pdb import set_trace as stx
import lpips
from tqdm import tqdm
import wandb
import utils.options as option
from utils.imresize import imresize
from networks.RRDBNet_arch import RRDBNet
import networks.lr_scheduler as lr_scheduler
import math

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
args = parser.parse_args()
opt = option.parse("confs/RRDB_2X.yml", is_train=True)


torch.cuda.set_device(args.local_rank)
torch.backends.cudnn.benchmark = True
torch.distributed.init_process_group("nccl")
gpu_count = torch.cuda.device_count()

start_epoch = 0
current_step = 0
device = torch.device("cuda", args.local_rank)

result_dir = os.path.join("./checkpoints", opt["name"], "results")
model_dir = os.path.join("./checkpoints", opt["name"], "models")

if args.local_rank == 0:
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

# save_images = opt.TRAINING.SAVE_IMAGES


######### Model ###########
opt_net = opt["network_G"]
generator_net = RRDBNet(
    in_nc=opt_net["in_nc"],
    out_nc=opt_net["out_nc"],
    nf=opt_net["nf"],
    nb=opt_net["nb"],
    scale=opt["scale"],
    opt=opt,
).to(device, non_blocking=True)

# Compute number of parameters
if args.local_rank == 0:
    s = sum([np.prod(list(p.size())) for p in generator_net.parameters()])
    print("Generator - Number of params: %d" % s)

new_lr = opt["train"]["lr_G"]

opt_para_generator = list(generator_net.parameters())
optimizer_generator = optim.Adam(
    opt_para_generator,
    lr=new_lr,
    betas=(opt["train"]["beta1"], opt["train"]["beta2"]),
)

generator_net = torch.nn.parallel.DistributedDataParallel(
    generator_net, device_ids=[args.local_rank], output_device=args.local_rank
)

scheduler_generator = lr_scheduler.CosineAnnealingLR_Restart(
    optimizer_generator,
    opt["train"]["T_period"],
    eta_min=opt["train"]["eta_min"],
    restarts=opt["train"]["restarts"],
    weights=opt["train"]["restart_weights"],
)

######### DataLoaders ###########
img_options_train = {"patch_size": opt["datasets"]["train"]["GT_size"]}
train_dataset = get_training_data_SR(
    opt["datasets"]["train"]["dataroot"], img_options_train
)
train_sampler = DistSampler(
    num_replicas=gpu_count,
    rank=args.local_rank,
    ratio=1,
    ds_size=len(train_dataset),
)
train_loader = PrefetchDataLoader(
    dataset=train_dataset,
    batch_size=opt["datasets"]["train"]["batch_size"],
    num_workers=opt["datasets"]["train"]["n_workers"],
    drop_last=False,
    persistent_workers=False,
    pin_memory=False,
    sampler=train_sampler,
    num_prefetch_queue=1,
)
train_fetcher = CPUPrefetcher(train_loader)


img_options_val = {"patch_size": opt["datasets"]["val"]["GT_size"]}
val_dataset = get_validation_data_SR(
    opt["datasets"]["val"]["dataroot"], img_options_val
)
val_sampler = DistSampler(
    num_replicas=gpu_count,
    rank=args.local_rank,
    ratio=1,  # opt.TRAINING.ENLARGE_RATIO,
    ds_size=len(val_dataset),
)
val_loader = PrefetchDataLoader(
    dataset=val_dataset,
    batch_size=opt["datasets"]["val"]["batch_size"],
    num_workers=opt["datasets"]["val"]["n_workers"],
    drop_last=False,
    persistent_workers=False,
    pin_memory=False,
    sampler=val_sampler,
    num_prefetch_queue=1,
)
val_fetcher = CPUPrefetcher(val_loader)

######### Train size ###########
train_size = int(
    math.ceil(len(train_dataset) / (opt["datasets"]["train"]["batch_size"] * gpu_count))
)
total_iters = int(opt["train"]["niter"])
total_epochs = int(math.ceil(total_iters / train_size))

######### Resume ###########
if opt["path"]["resume_state"] == "auto":
    path_chk_rest = utils.get_last_path(model_dir, "_latest.pth")

    map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
    checkpoint = torch.load(path_chk_rest, map_location=map_location)

    generator_net.load_state_dict(checkpoint["generator_net"])

    optimizer_generator.load_state_dict(checkpoint["optimizer_generator"])
    for p in optimizer_generator.param_groups:
        new_lr = p["lr"]
    scheduler_generator.load_state_dict(checkpoint["scheduler_generator"])

    start_epoch = checkpoint["epoch"] + 1
    current_step = checkpoint["current_step"] + 1
    best_psnr = checkpoint["best_psnr"]
    best_epoch = checkpoint["best_epoch"]

    wandb_id = checkpoint["wandb_id"]

    if args.local_rank == 0:
        print(
            "------------------------------------------------------------------------------"
        )
        print("==> Resuming Training with learning rate:", new_lr)
        print(
            "------------------------------------------------------------------------------"
        )
else:
    best_psnr = float("inf")
    best_epoch = 0
    wandb_id = wandb.util.generate_id()


######### Wandb init ###########
if args.local_rank == 0:
    wandb.login()
    wandb.tensorboard.patch(root_logdir="./experiments")
    wandb.init(
        id=wandb_id,
        project="DTPQ-NCSR",
        config=opt,
        name=opt["name"],
        resume=opt["path"]["resume_state"],
        sync_tensorboard=True,
    )


######### Loss ###########
pixel_criterion = nn.L1Loss().to(device, non_blocking=True)
loss_fn_alex = lpips.LPIPS(net="alex").to(device, non_blocking=True)

if args.local_rank == 0:
    print("===> Start Epoch {} End Epoch {}".format(start_epoch, total_epochs + 1))
    print("===> Start Step {} End Step {}".format(current_step, total_iters))
    print("===> Loading datasets")

######### Training ###########

for epoch in range(start_epoch, total_epochs + 1):
    epoch_start_time = time.time()

    generator_net.train()

    # let all processes sync up before starting with a new epoch of training
    torch.distributed.barrier()

    train_stats = torch.zeros((9,), device=device)
    train_stats[0] = 1

    train_sampler.set_epoch(epoch)  # shuffle distributed sub-samplers before each epoch

    # fetch the first batch
    train_fetcher.reset()
    train_data = train_fetcher.next()

    if args.local_rank == 0:
        tbar = tqdm(
            train_data,
            desc="Training  ",
            leave=True,
            smoothing=0.1,
            unit="batch",
            total=len(train_loader),
            ncols=80,
        )

    while train_data is not None:
        if args.local_rank == 0:
            tbar.update(1)

        if current_step > total_iters:
            break

        for param in generator_net.parameters():
            param.grad = None

        target = train_data[0].to(device, non_blocking=True)
        input_ = train_data[1].to(device, non_blocking=True)

        ##########################
        #   training generator   #
        ##########################
        restored = generator_net(input_)
        restored = torch.clamp(restored, 0, 1)

        # temp = target.permute(0, 2, 3, 1).cpu().detach().numpy() * 255
        # if args.local_rank == 0:
        #     utils.save_img(
        #         os.path.join(
        #             result_dir,
        #             "input.jpg",
        #         ),
        #         temp[0].astype(np.uint8),
        #     )

        generator_loss = (
            pixel_criterion(restored, target) * opt["train"]["pixel_weight"]
        )
        generator_loss.backward()
        optimizer_generator.step()

        train_stats[1] += generator_loss.mean().item()
        train_stats[2] += input_.size(0)

        scheduler_generator.step()
        current_step += 1

        # fetch the next batch
        train_data = train_fetcher.next()

    if args.local_rank == 0:
        tbar.close()

    # let all processes sync up before starting with a new epoch of training
    torch.distributed.barrier()

    torch.distributed.all_reduce(train_stats)

    #### Evaluation ####
    generator_net.eval()

    with torch.no_grad():
        psnr_val_rgb = []
        lpips_val_rgb = []
        val_loss = 0
        wandb_img = None

        val_stats = torch.zeros((5,), device=device)
        val_stats[0] = 1

        val_sampler.set_epoch(
            epoch
        )  # shuffle distributed sub-samplers before each epoch

        # fetch the first batch
        val_fetcher.reset()
        val_data = val_fetcher.next()

        if args.local_rank == 0:
            vbar = tqdm(
                val_data,
                desc="Validation",
                leave=True,
                smoothing=0.1,
                unit="batch",
                total=len(val_loader),
                ncols=80,
            )

        while val_data is not None:
            if args.local_rank == 0:
                vbar.update(1)

            target = val_data[0].to(device, non_blocking=True)
            input_ = val_data[1].to(device, non_blocking=True)
            filenames = val_data[2]

            restored = generator_net(input_)
            restored = torch.clamp(restored, 0, 1)

            pixel_loss = pixel_criterion(restored, target)

            val_loss += pixel_loss.mean().item()
            val_stats[1] = val_loss
            val_stats[2] += input_.size(0)

            psnr_val_rgb.append(utils.batch_PSNR(restored, target, 1.0))
            lpips_val_rgb.append(
                loss_fn_alex(restored * 2.0 - 1.0, target * 2.0 - 1.0).mean()
            )

            # Make image for wandb
            if args.local_rank == 0 and wandb_img is None:
                target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
                input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
                wandb_img = np.concatenate(
                    (
                        imresize(input_[0] * 255, 2),
                        restored[0] * 255,
                        target[0] * 255,
                    ),
                    axis=1,
                ).astype(np.uint8)

            # if save_images:
            #     target = target.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     input_ = input_.permute(0, 2, 3, 1).cpu().detach().numpy()
            #     restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            #     for batch in range(input_.shape[0]):
            #         temp = np.concatenate(
            #             (
            #                 input_[batch] * 255,
            #                 restored[batch] * 255,
            #                 target[batch] * 255,
            #             ),
            #             axis=1,
            #         )
            #         utils.save_img(
            #             os.path.join(
            #                 result_dir,
            #                 str(epoch),
            #                 str(epoch),
            #                 filenames[batch][:-4] + ".jpg",
            #             ),
            #             temp.astype(np.uint8),
            #         )

            # fetch the next batch
            val_data = val_fetcher.next()

        if args.local_rank == 0:
            vbar.close()

        psnr_val_rgb = np.array(psnr_val_rgb).astype(np.float32)
        psnr_val_rgb = psnr_val_rgb[~np.isnan(psnr_val_rgb)]
        psnr_val_rgb = psnr_val_rgb[~np.isinf(psnr_val_rgb)]
        psnr_val_rgb = psnr_val_rgb.sum() / len(psnr_val_rgb)
        val_stats[3] = psnr_val_rgb

        lpips_val_rgb = np.array(lpips_val_rgb).astype(np.float32)
        lpips_val_rgb = lpips_val_rgb.sum() / len(lpips_val_rgb)
        val_stats[4] = lpips_val_rgb

        torch.distributed.all_reduce(val_stats)

        val_loss = val_stats[1].detach().item() / val_stats[2].detach().item()

        torch.distributed.barrier()
        if args.local_rank == 0:
            node_count = val_stats[0].detach().item()
            psnr_val_rgb = val_stats[3].detach().item() / node_count
            lpips_val_rgb = val_stats[4].detach().item() / node_count
            train_loss = train_stats[1].detach().item() / train_stats[2].detach().item()
            train_pixel_loss = (
                train_stats[5].detach().item() / train_stats[2].detach().item()
            )

            if psnr_val_rgb > best_psnr: 
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "current_step": current_step,
                        "generator_net": generator_net.state_dict(),
                        "optimizer_generator": optimizer_generator.state_dict(),
                        "scheduler_generator": scheduler_generator.state_dict(),
                        "best_psnr": best_psnr,
                        "best_epoch": best_epoch,
                        "wandb_id": wandb_id,
                    },
                    os.path.join(model_dir, "model_best.pth"),
                )
                wandb.run.summary["best_psnr"] = best_psnr

            print(
                "[Ep %d\t LPIPS: %.4f\t PSNR: %.4f\t] ----  [best_Ep %d\t best_psnr %.4f] "
                % (epoch, lpips_val_rgb, psnr_val_rgb, best_epoch, best_psnr)
            )

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_psnr": psnr_val_rgb,
                    "val_lpips": lpips_val_rgb,
                    "generator_lr": optimizer_generator.param_groups[-1]["lr"],
                    "global_step": current_step,
                    "result": wandb.Image(wandb_img, caption="Input, Restored, Target"),
                },
                #step=current_step,
            )

            # print("------------------------------------------------------------------")
            print(
                "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(
                    epoch,
                    time.time() - epoch_start_time,
                    train_loss,
                    optimizer_generator.param_groups[-1]["lr"],
                )
            )
            print("------------------------------------------------------------------")

            torch.save(
                {
                    "epoch": epoch,
                    "current_step": current_step,
                    "generator_net": generator_net.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "scheduler_generator": scheduler_generator.state_dict(),
                    "best_psnr": best_psnr,
                    "best_epoch": best_epoch,
                    "wandb_id": wandb_id,
                },
                os.path.join(model_dir, "model_latest.pth"),
            )

if args.local_rank == 0:
    wandb.finish()
