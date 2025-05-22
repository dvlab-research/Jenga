
# Jenga 
<p align="center">
  <img src="./assets/title.png"  width=85%>
</p>


<div align="center">
  <a href="https://julianjuaner.github.io/projects/jenga"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Web&color=orange"></a> &ensp;
  <a href=""><img src="https://img.shields.io/static/v1?label=arXiv&message=Web&color=brown"></a> &ensp;
</div>

> This is the offical implementation of the paper [**Training-Free Efficient Video Generation via Dynamic Token Carving**](https://arxiv.org/abs/) <be>
## Overview
Jenga can generate videos with 4.68-10.35 times faster on single GPU with 80GiB memory.
Please visit the [project page](https://julianjuaner.github.io/projects/jenga) for more video results.
<p align="center">
  <img src="./assets/teaser_video.gif"  width=100%>
</p>

## Open-source Plan

- Model Adaptation
  - [x] HunyuanVideo Inference 
  - [x] Multi-gpus Parallel inference (Faster inference speed on more gpus)
  - [ ] HunyuanVideo-I2V Inference
  - [ ] Wan2.1
- Engineering Optimization
  - [ ] Quantization
  - [ ] ComfyUI
  - [ ] RoPE & Norm Kernel
  - [ ] FA3 Adaptation

## Guidance
### Inference on HunyuanVideo
#### Enviornment
Following the installation as in HunyuanVideo:
```shell
# 1. Create conda environment
conda create -n Jenga python==3.10.9

# 2. Activate the environment
conda activate Jenga

# 3. Install PyTorch and other dependencies using conda
# For CUDA 12.4
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia

# 4. Install pip dependencies
python -m pip install -r hy_requirements.txt

# 5. Install flash attention v2 for acceleration (requires CUDA 11.8 or above)
python -m pip install ninja
python -m pip install git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3

# 6. Install xDiT for parallel inference (we test on H800, cuda124)
python -m pip install xfuser==0.4.3.post3
python -m pip install yunchang==0.6.3.post1
```
#### Download model
Please following the instruction in [model_down_hy.md](./utils/model_down_hy.md).

#### Single GPU Inference
```shell
bash scripts/hyvideo_jenga_base.sh # Jenga Base (Opt. 310s)
# bash scripts/hyvideo_jenga_turbo.sh # Jenga Turbo
# bash scripts/hyvideo_jenga_flash.sh # Jenga Flash
# bash scripts/hyvideo_jenga_3stage.sh # Jenga 3Stage 
```
Inference time for different settings (DiT time, single H800): 
| Jenga-Base | Jenga-Turbo | Jenga-Flash | Jenga-3Stage |
| ---- | ---- | ---- | ---- |
| 310s (5.24x) | 225s (7.22x) | 184s (8.82x)| 157s (10.35x)| 

If you want to type your prompt directly, just change the `--prompt`. Following command (for Jenga-Turbo)
> If you encounters OOM issue, try to add `--use-cpu-offload`.

```shell
CUDA_VISIBLE_DEVICES=0 python3 -u ./jenga_hyvideo.py \
    --video-size 720 1280 \
    --video-length 125 \
    --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 \
    --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 \
    --flow-reverse \
    --sa-drop-rates 0.7 0.8 \
    --p-remain-rates 0.3 \
    --post-fix "Jenga_Turbo" \
    --save-path ./results/hyvideo \
    --res-rate-list 0.75 1.0 \
    --step-rate-list 0.5 1.0 \
    --scheduler-shift-list 7 9
```

#### Multi GPU Inference



## Method Overview
The general idea of Jenga is to reduce token interactions in Diffusion Transformers (DiTs). Following is an overview.
<p align="center">
  <img src="./assets/method_overview.png"  width=100%>
</p>
*The left part* illustrates the attention carving. A 3D video latent is partitioned into local blocks before being passed to the Transformer layers. A block-wise attention is processed to get a head-aware sparse block-selection masks. In each selected block, dense parallel attention is performed. *The right part* illustrates the Progressive Resolution strategy. The number of tokens and timesteps is compressed to ensure an efficient generation.

<br>

<p align="center">
  <img src="./assets/method_AttenCarve.png"  width=100%>
</p>
*Attention Carving (AttenCarve).* Here we illustrate a toy example of a 4x4x4$\ latent, where m=8 latent items form a block. Left: The latent 3D re-ordering and block partition via space filling curves (SFC). Right: After the block-wise attention, we can construct the Importance Mask, combined with the pre-computed Condition Mask and Adjacency Mask, a block-wise dense attention mask is passed to the customized kernel for device-efficient attention.

<br>

<p align="center">
  <img src="./assets/method_ProRes.png"  width=100%>
</p>
*Progressive Resolusion (ProRes).* Left: A brief illustration of stage switch and timestep skip. Before the rescale in stage s, we revert the latent to a clean state $\hat{x}^{s}_0$, then re-noise on the upsampled clean latent. Right & Bottom: We add a bias on the video-text attention score, to enable a scalable Field of View (FOV) in low-resolution content generation.

<br>

## Citation
If you find [Jenga](https://arxiv.org/abs/) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@misc{zhang2025Jenga,
      title={Training-Free Efficient Video Generation via Dynamic Token Carving}, 
      author={YC},
      year={2025},
}
```

## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), [HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V), [Wan2.1](https://github.com/Wan-Video/Wan2.1), [AccVideo](https://github.com/aejion/AccVideo), [MInference](https://github.com/microsoft/MInference), [Gilbert](https://github.com/jakubcerveny/gilbert) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
Additionally, we also thank the Tencent Hunyuan Multimodal team for their help with the text encoder. 