
CUDA_VISIBLE_DEVICES=0 python jenga_wan.py  \
        --task t2v-14B \
        --size 1280*720 \
        --ckpt_dir ./ckpts/Wan2.1-T2V-14B  \
        --sample_shift 8 \
        --sample_guide_scale 6 \
        --prompt ./assets/prompt_sora.txt \
        --base_seed 0 \
        --offload_model false \
        --frame_num 81 \
        --teacache_thresh 0.15 \
        --use_ret_step \
        --p_remain_rates 0.9 \
        --enable_turbo \
        --t5_cpu \
        --offload_model true \
        --sa_drop_rates 0.5 0.9 \
        --save_folder ./results/wan_14B_jenga_turbo 
