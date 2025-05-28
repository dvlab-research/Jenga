
CUDA_VISIBLE_DEVICES=0 python jenga_wan.py  \
        --task t2v-1.3B \
        --size 832*480 \
        --ckpt_dir ./ckpts/Wan2.1-T2V-1.3B  \
        --prompt ./assets/prompt_sora.txt \
        --base_seed 42 \
        --offload_model false \
        --frame_num 81 \
        --teacache_thresh 0.2 \
        --use_ret_step \
        --sa_drop_rate 0.8 \
        --p_remain_rates 0.9 \
        --save_folder ./results/1.3Btest