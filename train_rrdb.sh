#!/bin/sh


#CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_port=1334 ./train_rrdb.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python -m torch.distributed.launch --nproc_per_node=5 --nnodes=1 --node_rank=0 --master_port=1334 ./train_rrdb.py
