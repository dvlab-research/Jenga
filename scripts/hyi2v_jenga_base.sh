#!/bin/bash
# Description: This script demonstrates how to inference a video based on HunyuanVideo-I2V model

# enable debug
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_LOGS="+dynamo"
# export TORCHDYNAMO_VERBOSE=1
# export TORCHINDUCTOR_COMPILE_THREADS=1

CUDA_VISIBLE_DEVICES=0 python3 -u ./jenga_hyi2v.py \
    --prompt "An Asian man with short hair in black tactical uniform and white clothes waves a firework stick." \
    --model HYVideo-T/2 \
    --i2v-image-path ./assets/i2v_demo/i2v/imgs/0.jpg \
    --i2v-mode \
    --i2v-resolution 720p \
    --i2v-stability \
    --infer-steps 50 \
    --video-length 125 \
    --flow-reverse \
    --flow-shift 5.0 \
    --embedded-cfg-scale 6.0 \
    --seed 0 \
    --flow-reverse \
    --sa-drop-rates 0.75 0.85 \
    --p-remain-rates 0.3 \
    --save-path ./results/hyi2v \
    --res-rate-list 1.0 1.0 \
    --step-rate-list 0.5 1.0 \
    --scheduler-shift-list 7 7
