
# batched inference for 8xH800 (Jenga-Turbo)

for i in {0..7}; do
    echo "Running on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python3 -u sample_video_pro_res_multi_curve_batch.py \
        --video-size 720 1280 \
        --video-length 125 \
        --infer-steps 50 \
        --prompt ./assets/prompt_sora.txt \
        --seed 42 \
        --embedded-cfg-scale 6.0 \
        --flow-shift 7.0 \
        --flow-reverse \
        --p-remain-rates 0.3 \
        --sa-drop-rates 0.75 0.85 \
        --chunk-num 8 \
        --cur-id $i \
        --post-fix "Jenga_Turbo" \
        --save-path ./results/hyvideo \
        --res-rate-list 0.75 1.0 \
        --step-rate-list 0.5 1.0 \
        --scheduler-shift-list 7 9 &
done
wait