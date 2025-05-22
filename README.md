
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
### Enviornment
After following the installation in HunyuanVideo
### Inference on HunyuanVideo


## Method Overview
The general idea of Jenga is to reduce token interactions in Diffusion Transformers (DiTs). Following is an overview.
<p align="center">
  <img src="./assets/method_overview.png"  width=100%>
</p>
*The left part* illustrates the attention carving. A 3D video latent is partitioned into local blocks before being passed to the Transformer layers. A block-wise attention is processed to get a head-aware sparse block-selection masks. In each selected block, dense parallel attention is performed. *The right part* illustrates the Progressive Resolution strategy. The number of tokens and timesteps is compressed to ensure an efficient generation.

<br/>

<p align="center">
  <img src="./assets/method_AttenCarve.png"  width=100%>
</p>
*Attention Carving (AttenCarve).* Here we illustrate a toy example of a 4x4x4$\ latent, where m=8 latent items form a block. Left: The latent 3D re-ordering and block partition via space filling curves (SFC). Right: After the block-wise attention, we can construct the Importance Mask, combined with the pre-computed Condition Mask and Adjacency Mask, a block-wise dense attention mask is passed to the customized kernel for device-efficient attention.

<br/>

<p align="center">
  <img src="./assets/method_ProRes.png"  width=100%>
</p>
*Progressive Resolusion (ProRes).* Left: A brief illustration of stage switch and timestep skip. Before the rescale in stage s, we revert the latent to a clean state $\hat{x}^{s}_0$, then re-noise on the upsampled clean latent. Right & Bottom: We add a bias on the video-text attention score, to enable a scalable Field of View (FOV) in low-resolution content generation.

<br/>

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