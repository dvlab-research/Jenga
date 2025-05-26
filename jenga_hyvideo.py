# 250329: A progressive resolution version.

import os
import time
import torch
import numpy as np
import functools
import random
from typing import Optional
from pathlib import Path
from loguru import logger
from datetime import datetime
from einops import rearrange

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.modules.modulate_layers import modulate
from hyvideo.modules.attenion import attention, parallel_attention, get_cu_seqlens
from hyvideo.modules.posemb_layers import get_nd_rotary_pos_embed
from typing import Optional

# JULIAN: space curve related.
from gilbert import transpose_gilbert_mapping, gilbert_mapping, gilbert_block_neighbor_mapping


import torch.distributed as dist
non_skip_steps = [0,1,2,3,4,7,10,13,16,19,22,25,26,29,32,35,38,41,43,45,46,47,49]
try:
    import xfuser
    from xfuser.core.distributed import (
        get_sequence_parallel_world_size,
        get_sequence_parallel_rank,
        get_sp_group,
    )
except:
    xfuser = None
    get_sequence_parallel_world_size = None
    get_sequence_parallel_rank = None
    get_sp_group = None


def build_multi_curve(latent_time, latent_height, latent_width, res_rate_list):
    curve_sels = []
    
    for res_rate in res_rate_list:
        curve_sel = []
        latent_time_ = int(latent_time)
        latent_height_ = int(latent_height * res_rate)
        latent_width_ = int(latent_width * res_rate)
        if (latent_height_ * latent_width_) % 4 != 0:
            raise ValueError(f"latent_height_ * latent_width_ must be divisible by 4, but got {latent_height_ * latent_width_}")
        LINEAR_TO_HILBERT, HILBERT_ORDER = gilbert_mapping(latent_time_, latent_height_, latent_width_)
        block_neighbor_list = gilbert_block_neighbor_mapping(latent_time_, latent_height_, latent_width_)
        curve_sel.append([torch.tensor(LINEAR_TO_HILBERT, dtype=torch.long), torch.tensor(HILBERT_ORDER, dtype=torch.long), block_neighbor_list])
        curve_sels.append(curve_sel)

    return curve_sels


def ra_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        sa_drop_rate: float = 0.0,
        return_dict: bool = True,
        single_block_size: list[int] = [4, 4, 8] # [t, h, w]
    ):
        # order_shuffled = [0,1,2].shuffle()
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        img = img[:, self.hilbert_order]
        freqs_cos = freqs_cos[self.hilbert_order]
        freqs_sin = freqs_sin[self.hilbert_order]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None

        if self.enable_skip:
            if self.cnt in non_skip_steps or self.start_stage:
                should_calc = True
                self.start_stage = False
            else:
                should_calc = False
        else:
            should_calc = True
        
        if self.enable_skip:
            if not should_calc:
                img += self.previous_residual
            else:
                ori_img = img.clone()
                for idx, block in enumerate(self.double_blocks):
                    double_block_args = [
                        img,
                        txt,
                        vec,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        self.sa_drop_rate,
                        self.text_amp,
                        self.curve_sel,
                        self.p_remain_rates,
                    ]
                    img, txt = block(*double_block_args)
                # Merge txt and img to pass through single stream blocks.
                x = torch.cat((img, txt), 1)
                if len(self.single_blocks) > 0:
                    for idx, block in enumerate(self.single_blocks):
                        single_block_args = [
                            x,
                            vec,
                            txt_seq_len,
                            cu_seqlens_q,
                            cu_seqlens_kv,
                            max_seqlen_q,
                            max_seqlen_kv,
                            (freqs_cos, freqs_sin),
                            self.sa_drop_rate,
                            self.text_amp,
                            self.curve_sel,
                            self.p_remain_rates,
                        ]
                        x = block(*single_block_args)

                img = x[:, :img_seq_len, ...]
                self.previous_residual = img - ori_img
        else:    
            # --------------------- Pass through DiT blocks ------------------------
            for _, block in enumerate(self.double_blocks):
                double_block_args = [
                    img,
                    txt,
                    vec,
                    cu_seqlens_q,
                    cu_seqlens_kv,
                    max_seqlen_q,
                    max_seqlen_kv,
                    freqs_cis,
                    self.sa_drop_rate,
                    self.text_amp,
                    self.curve_sel,
                    self.p_remain_rates,
                ]
                img, txt = block(*double_block_args)

            # Merge txt and img to pass through single stream blocks.
            x = torch.cat((img, txt), 1)
            if len(self.single_blocks) > 0:
                for _, block in enumerate(self.single_blocks):
                    single_block_args = [
                        x,
                        vec,
                        txt_seq_len,
                        cu_seqlens_q,
                        cu_seqlens_kv,
                        max_seqlen_q,
                        max_seqlen_kv,
                        (freqs_cos, freqs_sin),
                        self.sa_drop_rate,
                        self.text_amp,
                        self.curve_sel,
                        self.p_remain_rates,
                    ]

                    x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]

        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0

        img = img[:, self.linear_to_hilbert]
        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

def main():
    args = parse_args()
    if ".txt" in args.prompt:
        with open(args.prompt, "r") as f:
            prompts = f.readlines()
            prompts = [prompt.strip() for prompt in prompts]
            prompts = prompts[args.cur_id::args.chunk_num]
    else:
        prompts = [args.prompt]
        
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    else:
        print(f"Models root path: {models_root_path}", args.model_base)

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    latent_time = (args.video_length + 3) // 4
    latent_height = args.video_size[0] // 16
    latent_width = args.video_size[1] // 16

    curve_sels = build_multi_curve(latent_time, latent_height, latent_width, args.res_rate_list)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    from hyvideo.diffusion.pipelines.pipeline_hunyuan_video_prores import HunyuanVideoPipelineProRes
    # reinitalize the pipeline.
    hunyuan_video_sampler.pipeline.__class__.__call__ = HunyuanVideoPipelineProRes.__call__
    hunyuan_video_sampler.pipeline.__class__.get_rotary_pos_embed = HunyuanVideoPipelineProRes.get_rotary_pos_embed
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = ra_forward
    hunyuan_video_sampler.pipeline.transformer.__class__.ra_forward = ra_forward

    for prompt in prompts:
        # Get the updated args
        args = hunyuan_video_sampler.args
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_skip = True
        hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
        hunyuan_video_sampler.pipeline.transformer.__class__.start_stage = True
        hunyuan_video_sampler.pipeline.transformer.__class__.current_t = latent_time
        hunyuan_video_sampler.pipeline.transformer.__class__.current_h = latent_height
        hunyuan_video_sampler.pipeline.transformer.__class__.current_w = latent_width
        hunyuan_video_sampler.pipeline.transformer.__class__.curve_sels = curve_sels
        hunyuan_video_sampler.pipeline.transformer.__class__.curve_sel = None
        hunyuan_video_sampler.pipeline.transformer.__class__.sa_drop_rates = args.sa_drop_rates
        hunyuan_video_sampler.pipeline.transformer.__class__.scale_txt_amp = args.scale_txt_amp
        hunyuan_video_sampler.pipeline.transformer.__class__.p_remain_rates = args.p_remain_rates

        # Start sampling
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            sa_drop_rate=args.sa_drop_rate,
            res_rate_list=args.res_rate_list,
            step_rate_list=args.step_rate_list,
            scheduler_shift_list=args.scheduler_shift_list,
        )
        samples = outputs['samples']
        gen_time = str(outputs['gen_time']).split('.')[0]
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
                cur_save_path = f"{save_path}/{args.post_fix}_{time_flag}_seed{outputs['seeds'][i]}_time{gen_time}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
                save_videos_grid(sample, cur_save_path, fps=24)    
                logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
