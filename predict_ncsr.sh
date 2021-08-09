#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 --node_rank=0 --master_port=1235 ./predict_ncsr.py -i '/mnt/sdb1/11/sample-input.mp4' -o /mnt/sdb1/11/sample-output.mp4
