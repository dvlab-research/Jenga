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

from hyvideo_i2v.utils.file_utils import save_videos_grid
from hyvideo_i2v.config import parse_args
from hyvideo_i2v.inference import HunyuanVideoSampler
from hyvideo_i2v.modules.modulate_layers import ckpt_wrapper
from hyvideo_i2v.modules.attenion import get_cu_seqlens
from hyvideo_i2v.utils.data_utils import align_to, get_closest_ratio, generate_crop_size_list
from hyvideo_i2v.diffusion.pipelines.pipeline_hunyuan_video_prores import HunyuanVideoPipelineProRes
from typing import Optional
from PIL import Image

# JENGA: space curve related.
from gilbert import transpose_gilbert_mapping, gilbert_mapping, gilbert_block_neighbor_mapping

import torch.distributed as dist

# explicitly define the skip step.
step_calc = [_ for _ in range(50)]
step_calc = [0,1,2,3,5,7,10,13,16,19,22,25,28,31,34,37,40,42,44,45,46,47,48,49]

def build_multi_curve(latent_time, latent_height, latent_width, res_rate_list, 
                      shift=64, type="hilbert-base"):
    curve_sels = []
    
    for res_rate in res_rate_list:
        curve_sel = []
        latent_time_ = int(latent_time)
        latent_height_ = int(latent_height * res_rate)
        latent_width_ = int(latent_width * res_rate)
        LINEAR_TO_HILBERT, HILBERT_ORDER = gilbert_mapping(latent_time_, latent_height_, latent_width_)
        block_neighbor_list = gilbert_block_neighbor_mapping(latent_time_, latent_height_, latent_width_)
        curve_sel.append([torch.tensor(LINEAR_TO_HILBERT, dtype=torch.long), torch.tensor(HILBERT_ORDER, dtype=torch.long), block_neighbor_list])
        curve_sels.append(curve_sel)

    return curve_sels

# the original forward:
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
        return_dict: bool = True,
        token_per_block=128,
    ):
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

        if self.i2v_condition_type == "token_replace":
            token_replace_t = torch.zeros_like(t)
            token_replace_vec = self.time_in(token_replace_t)
            frist_frame_token_num = th * tw
        else:
            token_replace_vec = None
            frist_frame_token_num = None
            

        # text modulation
        vec_2 = self.vector_in(text_states_2)
        vec = vec + vec_2
        if self.i2v_condition_type == "token_replace":
            token_replace_vec = token_replace_vec + vec_2

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

        # JENGA: space curve re-indexing.
        img = img[:, self.hilbert_order]
        freqs_cos = freqs_cos[self.hilbert_order]
        freqs_sin = freqs_sin[self.hilbert_order]

        # JENGA: for I2V, we also pass a first index in the hilbert order, representing the first frame.
        first_frame_mask = torch.zeros_like(img[0, :, 0]).bool().squeeze(-1)
        first_frame_mask[:th*tw] = 1
        first_frame_mask = first_frame_mask[self.hilbert_order]

        # padding the length of first_frame_mask to img_seq_len + txt_seq_len to get full mask.
        full_first_frame_mask = torch.zeros(img_seq_len + txt_seq_len).bool()
        full_first_frame_mask[:img_seq_len] = first_frame_mask
        
        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        if self.cnt in step_calc or self.start_stage:
            self.start_stage = False
            should_calc = True
        else:
            should_calc = False
        if not should_calc:
            img += self.previous_residual
        else:
            ori_img = img.clone()
            # --------------------- Pass through DiT blocks ------------------------
            for layer_num, block in enumerate(self.double_blocks):
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
                    first_frame_mask,
                    self.i2v_condition_type,
                    token_replace_vec,
                    frist_frame_token_num,
                    self.text_amp,
                    self.curve_sel,
                    self.p_remain_rates
                ]

                if self.training and self.gradient_checkpoint and \
                        (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers):
                    # print(f'gradient checkpointing...')
                    img, txt = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *double_block_args, use_reentrant=False)
                else:
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
                        full_first_frame_mask,
                        self.i2v_condition_type,
                        token_replace_vec,
                        frist_frame_token_num,
                        self.text_amp,
                        self.curve_sel,
                        self.p_remain_rates
                    ]

                    if self.training and self.gradient_checkpoint and \
                            (self.gradient_checkpoint_layers == -1 or layer_num + len(self.double_blocks) < self.gradient_checkpoint_layers):
                        x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *single_block_args, use_reentrant=False)
                    else:
                        x = block(*single_block_args)

            img = x[:, :img_seq_len, ...]
            
            self.previous_residual = img - ori_img

        self.cnt += 1
        if self.cnt == self.num_steps:
            self.cnt = 0
        img = img[:, self.linear_to_hilbert]
        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

def main():
    args = parse_args()
    import json

    if args.prompt is not None and os.path.isfile(args.prompt):
        # Check if it's a JSON file
        if args.prompt.endswith('.json'):
            with open(args.prompt, 'r') as f:
                json_data = json.load(f)
                # Assuming JSON structure has a list of items with prompt and id fields
                # Adjust as needed based on your actual JSON structure
                prompts = []
                i2v_image_paths = []
                ids = []
                for item in json_data:
                    if isinstance(item, dict) and 'prompt_en' in item:
                        prompts.append(item['prompt_en'])
                        i2v_image_paths.append(os.path.join(args.i2v_image_path, item['image_name']))
                        # Use 'id' field if available, otherwise use index
                        if 'id' in item:
                            ids.append(str(item['id']).zfill(4))
                        else:
                            ids.append(f"{len(ids):04d}")
                
                print(f"Total prompts from JSON: {len(prompts)}")
                args.prompt = prompts[args.cur_id::args.chunk_num]
                i2v_image_paths = i2v_image_paths[args.cur_id::args.chunk_num]
                ids = ids[args.cur_id::args.chunk_num]
                print(f"Selected prompts: {len(args.prompt)}", args.cur_id, args.chunk_num)
        else:
            # Original text file reading
            with open(args.prompt, 'r') as f:
                lines = f.readlines()
                print(f"Total prompts: {len(lines)}")
                lines = [line.strip() for line in lines]
                args.prompt = lines
                ids = [i for i in range(len(args.prompt))]
                print(f"Total prompts: {len(args.prompt)}")
                args.prompt = args.prompt[args.cur_id::args.chunk_num]
                print(f"Total prompts: {len(args.prompt)}", args.cur_id, args.chunk_num)
                ids = ids[args.cur_id::args.chunk_num]
                # make to 4 digit string
                ids = [f"{i:04d}" for i in ids]
    else:
        print(f"args.i2v_image_path: {args.i2v_image_path}")
        args.prompt = [args.prompt]
        i2v_image_paths = [args.i2v_image_path]
        ids = [args.cur_id]

    # JENGA: for I2V, we need to load the model.
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    else:
        print(f"Models root path: {models_root_path}", args.model_base)

    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    if args.i2v_resolution == "720p":
        bucket_hw_base_size = 960
    elif args.i2v_resolution == "540p":
        bucket_hw_base_size = 720
    elif args.i2v_resolution == "360p":
        bucket_hw_base_size = 480

    args.i2v_image_path = i2v_image_paths[0]
    semantic_images = [Image.open(args.i2v_image_path).convert('RGB')]
    crop_size_list = generate_crop_size_list(bucket_hw_base_size, 32)
    origin_size = semantic_images[0].size
    aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
    closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)
    args.video_size = closest_size
    # args.video_size = (720, 1280)

    latent_time = (args.video_length + 3) // 4
    latent_height = args.video_size[0] // 16
    latent_width = args.video_size[1] // 16

    # I need a function for the multi-curve building before different stages.
    curve_sels = build_multi_curve(latent_time, latent_height, latent_width, args.res_rate_list, type="block-neighbor")

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    hunyuan_video_sampler.pipeline.__class__.__call__ = HunyuanVideoPipelineProRes.__call__
    hunyuan_video_sampler.pipeline.__class__.get_rotary_pos_embed = HunyuanVideoPipelineProRes.get_rotary_pos_embed
    hunyuan_video_sampler.pipeline.transformer.__class__.forward = ra_forward
    hunyuan_video_sampler.pipeline.transformer.__class__.ra_forward = ra_forward

    for i, prompt in enumerate(args.prompt):
        # Get the updated args
        args = hunyuan_video_sampler.args
        hunyuan_video_sampler.pipeline.transformer.__class__.enable_teacache = False
        hunyuan_video_sampler.pipeline.transformer.__class__.cnt = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.num_steps = args.infer_steps
        hunyuan_video_sampler.pipeline.transformer.__class__.previous_residual = None
        hunyuan_video_sampler.pipeline.transformer.__class__.consistent_threshold = 0
        hunyuan_video_sampler.pipeline.transformer.__class__.start_stage = True
        hunyuan_video_sampler.pipeline.transformer.__class__.text_amp = 0.0
        hunyuan_video_sampler.pipeline.transformer.__class__.current_t = latent_time
        hunyuan_video_sampler.pipeline.transformer.__class__.current_h = latent_height
        hunyuan_video_sampler.pipeline.transformer.__class__.current_w = latent_width
        hunyuan_video_sampler.pipeline.transformer.__class__.curve_sels = curve_sels
        hunyuan_video_sampler.pipeline.transformer.__class__.curve_sel = None
        hunyuan_video_sampler.pipeline.transformer.__class__.sa_drop_rates = args.sa_drop_rates
        hunyuan_video_sampler.pipeline.transformer.__class__.p_remain_rates = args.p_remain_rates
        hunyuan_video_sampler.pipeline.transformer.__class__.scale_txt_amp = args.scale_txt_amp
        
        args.i2v_image_path = i2v_image_paths[i]
        semantic_images = [Image.open(args.i2v_image_path).convert('RGB')]
        crop_size_list = generate_crop_size_list(bucket_hw_base_size, 16)
        origin_size = semantic_images[0].size
        aspect_ratios = np.array([round(float(h)/float(w), 5) for h, w in crop_size_list])
        closest_size, closest_ratio = get_closest_ratio(origin_size[1], origin_size[0], aspect_ratios, crop_size_list)

        latent_time = (args.video_length + 3) // 4
        latent_height = args.video_size[0] // 16
        latent_width = args.video_size[1] // 16

        # Start sampling    
        # TODO: batch inference check
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
            i2v_mode=args.i2v_mode,
            i2v_resolution=args.i2v_resolution,
            i2v_image_path=args.i2v_image_path,
            i2v_condition_type=args.i2v_condition_type,
            i2v_stability=args.i2v_stability,
            ulysses_degree=args.ulysses_degree,
            ring_degree=args.ring_degree,
            res_rate_list=args.res_rate_list,
            step_rate_list=args.step_rate_list,
            scheduler_shift_list=args.scheduler_shift_list,
            sa_drop_rate=args.sa_drop_rate
        )
        samples = outputs['samples']
        gen_time = str(outputs['gen_time']).split('.')[0]
        
        # Save samples
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            for j, sample in enumerate(samples):
                sample = samples[j].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M")
                cur_save_path = f"{save_path}/id_{ids[i]}_{time_flag}_seed{outputs['seeds'][j]}_{outputs['prompts'][j].replace('/','')[:100]}_.mp4"
                save_videos_grid(sample, cur_save_path, fps=24)
                logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
