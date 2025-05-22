#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo model

# enable debug
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1
# export TORCHINDUCTOR_COMPILE_THREADS=1

CUDA_VISIBLE_DEVICES=0 python -u ./jenga_hyvideo.py \
    --video-size 720 1280 \
    --video-length 125 \
	--infer-steps 50 \
    --prompt ./assets/prompt_sora.txt \
    --seed 42 \
	--embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --sa-drop-rates 0.75 0.85 0.85 \
    --p-remain-rates 0.3 \
    --post-fix "Jenga_3Stage" \
    --save-path ./results/hyvideo \
    --res-rate-list 0.5 0.75 1.0 \
    --step-rate-list 0.3 0.5 1.0 \
    --scheduler-shift-list 7 9 11
