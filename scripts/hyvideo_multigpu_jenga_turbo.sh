#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

export NPROC_PER_NODE=8
export ULYSSES_DEGREE=8

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=$NPROC_PER_NODE ./jenga_hyvideo_multigpu.py \
    --video-size 720 1280 \
    --video-length 125 \
	--infer-steps 50 \
    --prompt ./assets/prompt_sora.txt \
    --seed 42 \
	--embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --sa-drop-rates 0.75 0.85 \
    --p-remain-rates 0.3 \
    --post-fix "Jenga_Turbo" \
    --save-path ./results/hyvideo_multigpu \
    --res-rate-list 0.75 1.0 \
    --step-rate-list 0.5 1.0 \
    --ulysses-degree $ULYSSES_DEGREE \
    --scheduler-shift-list 7 9
