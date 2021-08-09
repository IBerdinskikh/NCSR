import argparse, os, utils, pathlib, subprocess, re, shlex, ffmpeg
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
import utils.options as option

from networks.NCSRNet_arch import NCSRNet

from dataloaders.data_rgb import (
    get_predict_data_SR,
    split_image_into_overlapping_patches,
    stich_together,
    CPUPrefetcher,
    PrefetchDataLoader,
)
from skimage import img_as_ubyte
from torch.utils.data.distributed import DistributedSampler
from shutil import rmtree


parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument(
    "-i",
    "--input",
    type=pathlib.Path,
    help="Source video file",
    required=True,
)
parser.add_argument(
    "-o",
    "--output",
    type=pathlib.Path,
    help="Output video file",
    required=True,
)
parser.add_argument(
    "--weights",
    default="./checkpoints/02_NCSR_DTP_2X/models/model_latest.pth",
    type=str,
    help="Path to weights",
)
parser.add_argument("--bs", default=4, type=int, help="Batch size for dataloader")
parser.add_argument(
    "--patch_size",
    default=256,
    type=int,
    help="Size of the patches from the original image",
)

args = parser.parse_args()
torch.cuda.set_device(args.local_rank)


def get_source_info_ffmpeg(source_name):
    return_value = 0
    try:
        info = ffmpeg.probe(source_name)
        vs = next(c for c in info["streams"] if c["codec_type"] == "video")
        mediaInfo = dict()
        mediaInfo["videoFrameRate"] = vs["r_frame_rate"]
        mediaInfo["videoWidth"] = vs["width"]
        mediaInfo["videoHeight"] = vs["height"]
        return mediaInfo
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("init_source:{} error. {}\n".format(source_name, str(e)))
        return_value = 0
    return return_value


#
temp_in = "/mnt/sdc1/temp/in"
temp_out = "/mnt/sdc1/temp/out"
if args.local_rank == 0:
    # temp in
    if os.path.exists(temp_in):
        rmtree(temp_in)
    if not os.path.exists(temp_in):
        os.makedirs(temp_in)

    # temp out
    if os.path.exists(temp_out):
        rmtree(temp_out)
    if not os.path.exists(temp_out):
        os.makedirs(temp_out)

    # Извлекаем малюнки
    cmd = (
        "/usr/bin/ffmpeg -hwaccel auto -y -i "
        + str(shlex.quote(str(args.input)))
        # + ' -vf "scale=iw*2:ih*2:flags=bicubic"'
        + " -pix_fmt rgb24 "
        + temp_in
        + "/extracted_%0d.png"
    )
    print(cmd)
    outputBytes = subprocess.check_output(cmd, shell=True)
    outputText = outputBytes.decode("utf-8")

# setup the distributed backend for managing the distributed training
torch.distributed.init_process_group("nccl")
device = torch.device("cuda", args.local_rank)

######### Model ###########
opt = option.parse("confs/NCSR_2X.yml", is_train=True)
opt_net = opt["network_G"]
generator_net = NCSRNet(
    in_nc=opt_net["in_nc"],
    out_nc=opt_net["out_nc"],
    nf=opt_net["nf"],
    nb=opt_net["nb"],
    scale=opt["scale"],
    K=opt_net["flow"]["K"],
    opt=opt,
).to(device, non_blocking=True)

generator_net = torch.nn.parallel.DistributedDataParallel(
    generator_net, device_ids=[args.local_rank], output_device=args.local_rank
)

map_location = {"cuda:%d" % 0: "cuda:%d" % args.local_rank}
checkpoint = torch.load(args.weights, map_location=map_location)

generator_net.load_state_dict(checkpoint["generator_net"])

if args.local_rank == 0:
    print("===>Predict using weights: ", args.weights)

test_dataset = get_predict_data_SR(temp_in)
test_sampler = DistributedSampler(
    dataset=test_dataset,
    num_replicas=torch.cuda.device_count(),
    rank=args.local_rank,
)

# PrefetchDataLoader
test_loader = PrefetchDataLoader(
    dataset=test_dataset,
    batch_size=1,
    num_workers=1,
    drop_last=False,
    persistent_workers=False,
    pin_memory=False,
    sampler=test_sampler,
    num_prefetch_queue=1,
)
test_fetcher = CPUPrefetcher(test_loader)


def get_z(heat, seed=None, batch_size=1, lr_shape=None):
    if opt["network_G"]["flow"]["split"]["enable"]:
        C = generator_net.module.flowUpsamplerNet.C
        H = int(
            opt["scale"] * lr_shape[2] // generator_net.module.flowUpsamplerNet.scaleH
        )
        W = int(
            opt["scale"] * lr_shape[3] // generator_net.module.flowUpsamplerNet.scaleW
        )
        z = (
            torch.normal(mean=0, std=heat, size=(batch_size, C, H, W))
            if heat > 0
            else torch.zeros((batch_size, C, H, W))
        )
    else:
        L = opt["network_G"]["flow"]["L"] or 3
        lr_size = 160 // opt["scale"]
        fac = 2 ** (L - 3)
        z_size = int(lr_size // (2 ** (L - 3)))
        z = torch.normal(
            mean=0,
            std=heat,
            size=(batch_size, 3 * 8 * 8 * fac * fac, z_size, z_size),
        )
    return z


generator_net.eval()
with torch.no_grad():
    # fetch the first batch
    test_fetcher.reset()
    test_data = test_fetcher.next()

    if args.local_rank == 0:
        t = tqdm(
            test_loader,
            desc="Predict",
            leave=True,
            smoothing=0.1,
            unit="batch",
            total=len(test_loader),
        )

    # Обрабатываем патчами
    while test_data is not None:
        if args.local_rank == 0:
            t.update(1)

        filenames = test_data[1]

        # Обрабатываем патчами
        if test_data[0][0].shape[0] * test_data[0][0].shape[1] > 2000 * 1080:
            input_ = test_data[0][0]
            patches, p_shape = split_image_into_overlapping_patches(
                input_, patch_size=args.patch_size
            )

            for i in range(0, len(patches), args.bs):
                batch_in = torch.from_numpy(patches[i : i + args.bs]).to(
                    device, non_blocking=True
                )
                batch_in = batch_in.permute(0, 3, 1, 2)

                # restored = generator_net(batch_in)
                heat = 0.9
                z = get_z(
                    heat,
                    seed=None,
                    batch_size=input_.shape[0],
                    lr_shape=input_.shape,
                )
                std_in = torch.zeros(input_.size(0), 1, 1, 1).to(
                    device, non_blocking=True
                )
                restored, _ = generator_net(
                    lr=input_, z=z, eps_std=heat, reverse=True, std=std_in
                )

                restored = torch.clamp(restored, 0, 1)
                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                if i == 0:
                    collect = restored
                else:
                    collect = np.append(collect, restored, axis=0)

                # denoised_img = img_as_ubyte(restored[0])
                # utils.save_img(temp_out + "/" + str(i) + ".png", denoised_img)

            restored = stich_together(collect, padded_image_shape=p_shape, scale=2)

            denoised_img = img_as_ubyte(restored)
            utils.save_img(temp_out + "/" + filenames[0][:-4] + ".png", denoised_img)

        # Обрабатываем целиком
        else:
            input_ = test_data[0].to(device, non_blocking=True).permute(0, 3, 1, 2)
            # restored = generator_net(input_)
            heat = 0.9
            z = get_z(
                heat,
                seed=None,
                batch_size=input_.shape[0],
                lr_shape=input_.shape,
            )
            std_in = torch.zeros(input_.size(0), 1, 1, 1).to(device, non_blocking=True)
            restored, _ = generator_net(
                lr=input_, z=z, eps_std=heat, reverse=True, std=std_in
            )

            restored = torch.clamp(restored, 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
            for batch in range(len(restored)):
                denoised_img = img_as_ubyte(restored[batch])
                utils.save_img(
                    temp_out + "/" + filenames[batch][:-4] + ".png", denoised_img
                )

        # fetch the next batch
        test_data = test_fetcher.next()

torch.distributed.barrier()

if args.local_rank == 0:
    escaped_in = shlex.quote(str(args.input))
    escaped_out = shlex.quote(str(args.output))

    info = get_source_info_ffmpeg(args.input)

    # Converting extracted frames into video
    print("Converting extracted frames into video")
    cmd = (
        "/usr/bin/ffmpeg -r "
        + info["videoFrameRate"]
        + " -hwaccel auto -y -f image2 -i "
        + temp_out
        + "/extracted_%d.png -vcodec libx264 -pix_fmt yuv420p -crf 12 -vf pad='ceil(iw/2)*2:ceil(ih/2)*2' -tune film "
        + temp_out
        + "/intermediate.mkv"
    )
    print(cmd)
    outputBytes = subprocess.check_output(cmd, shell=True)
    outputText = outputBytes.decode("utf-8")

    # Migrating audio, subtitles and other streams to upscaled video
    print("Migrating audio, subtitles and other streams to upscaled video")
    cmd = (
        "/usr/bin/ffmpeg -hwaccel auto -y -i "
        + temp_out
        + "/intermediate.mkv -i "
        + escaped_in
        + " -map 0:v? -map 1:a? -map 1:s? -map 1:d? -map 1:t? -c copy -map_metadata 0 "
        + str(shlex.quote(str(args.output)))
    )
    print(cmd)
    outputBytes = subprocess.check_output(cmd, shell=True)
    outputText = outputBytes.decode("utf-8")
