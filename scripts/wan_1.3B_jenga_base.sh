
CUDA_VISIBLE_DEVICES=0 python jenga_wan.py  \
        --task t2v-1.3B \
        --size 832*480 \
        --ckpt_dir ./ckpts/Wan2.1-T2V-1.3B  \
        --sample_shift 8 \
        --sample_guide_scale 6 \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
        --base_seed 0 \
        --offload_model false \
        --frame_num 81 \
        --teacache_thresh 0.15 \
        --use_ret_step \
        --p_remain_rates 0.9 \
        --sa_drop_rates 0.75 0.85 \
        --save_folder ./results/wan_1.3B_jenga_base