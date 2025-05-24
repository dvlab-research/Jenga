# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from ...modules import HYVideoDiffusionTransformer
from ...utils.data_utils import black_image
from hyvideo_i2v.diffusion.pipelines.pipeline_hunyuan_video import HunyuanVideoPipeline
import math
from gilbert import gilbert_mapping
from hyvideo_i2v.modules.posemb_layers import get_nd_rotary_pos_embed, get_meshgrid_nd
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoPipelineProRes(HunyuanVideoPipeline):
    
    def get_rotary_pos_embed(self, latents_size):
        target_ndim = 3
        ndim = 5 - 2
        # 884
        # here we hard code the params
        hidden_size = 3072
        heads_num = 24
        rope_theta = 256
        patch_size = 1
        rope_dim_list = [16, 56, 56]

        if isinstance(patch_size, int):
            assert all(s % patch_size == 0 for s in latents_size), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [s // patch_size for s in latents_size]
        elif isinstance(patch_size, list):
            assert all(
                s % patch_size[idx] == 0
                for idx, s in enumerate(latents_size)
            ), (
                f"Latent size(last {ndim} dimensions) should be divisible by patch size({patch_size}), "
                f"but got {latents_size}."
            )
            rope_sizes = [
                s // patch_size[idx] for idx, s in enumerate(latents_size)
            ]

        if len(rope_sizes) != target_ndim:
            rope_sizes = [1] * (target_ndim - len(rope_sizes)) + rope_sizes  # time axis
        head_dim = hidden_size // heads_num
        rope_dim_list = rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        embedded_guidance_scale: Optional[float] = None,
        sa_drop_rate: float = 0.0,
        res_rate_list: list[int] = [0.5, 0.75, 1.0],
        step_rate_list: list[int] = [0.3, 0.5, 1.0],
        scheduler_shift_list: list[int] = [7, 9, 11],
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        semantic_image_pixel_values=None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        # JENGA: first, we adopt the res_rate_list to different steps.
        latent_time = 1 + (video_length - 1) // 4
        latent_height = height
        latent_width = width
        latent_shape = [latent_time, latent_height, latent_width]
        time_step_split = [int(num_inference_steps * step_rate) for step_rate in step_rate_list]
        latent_step_shapes = [[latent_time, int(latent_height*res_rate), int(latent_width*res_rate)] 
                        for res_rate in res_rate_list]
        
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = self.encode_prompt(
            prompt,
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
            semantic_images=semantic_images
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = self.encode_prompt(
                prompt,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])


        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length

        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        image_latent_list = []
        current_h, current_w = semantic_image_pixel_values.shape[-2:]
        img_res_shapes = [[1, int(current_h*rate//16*16), int(current_w*rate//16*16)] for rate in res_rate_list]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            for img_res_shape in img_res_shapes:
                cur_semantic_image_pixel_values = torch.nn.functional.interpolate(semantic_image_pixel_values, size=img_res_shape, mode="trilinear", align_corners=False)
                # for
                img_latents = self.vae.encode(cur_semantic_image_pixel_values).latent_dist.mode()
                img_latents.mul_(self.vae.config.scaling_factor)
                image_latent_list.append(img_latents)

        # JENGA: we need to prepare the latents for different steps.
        # DEBUG here: test the effect of different original_size settings.
        img_latents = image_latent_list[0]
        original_size = [latent_step_shapes[-1][0], latent_step_shapes[-1][1]//16, latent_step_shapes[-1][2]//16]
        latents = randn_tensor([batch_size * num_videos_per_prompt,
                              num_channels_latents,
                              latent_step_shapes[0][0], latent_step_shapes[0][1]//16 * 2, latent_step_shapes[0][2]//16 * 2],
                              generator=generator,
                              device=device, dtype=prompt_embeds.dtype)
        
        if i2v_mode and i2v_stability:
            t = torch.tensor([0.999]).to(device=device)
            latents = latents * t + img_latents.repeat(1, 1, video_length, 1, 1) * (1 - t)

        current_size = [latent_step_shapes[0][0], latent_step_shapes[0][1]//16, latent_step_shapes[0][2]//16]
        token_diff = (current_size[1] * current_size[2]) / (original_size[1] * original_size[2])
        if not hasattr(self.transformer, "curve_sels"):
            linear_hilbert, hilbert_linear = gilbert_mapping(current_size[0], current_size[1], current_size[2])
            self.transformer.current_t = current_size[0]
            self.transformer.current_h = current_size[1]
            self.transformer.current_w = current_size[2]
            self.transformer.linear_to_hilbert = linear_hilbert
            self.transformer.hilbert_order = hilbert_linear
        else:
            self.transformer.curve_sel = self.transformer.curve_sels[0]
            for im in range(len(self.transformer.curve_sel)):
                if self.transformer.curve_sel[im][-1] is not None and self.transformer.curve_sel[im][-1].device != latents.device:
                    self.transformer.curve_sel[im][-1] = self.transformer.curve_sel[im][-1].to(latents.device)
            self.transformer.linear_to_hilbert = self.transformer.curve_sel[0][0]
            self.transformer.hilbert_order = self.transformer.curve_sel[0][1]
            self.transformer.sa_drop_rate = self.transformer.sa_drop_rates[0]

        self.transformer.text_amp = -1 * math.log(math.sqrt(token_diff), 2) * self.transformer.scale_txt_amp
        freqs_cis = self.get_rotary_pos_embed(
            current_size
        )
        stage_idx = 0
        if i2v_mode and i2v_condition_type == "latent_concat":
            if img_latents.shape[2] == 1:
                img_latents_concat = img_latents.repeat(1, 1, video_length, 1, 1)
            else:
                img_latents_concat = img_latents
            img_latents_concat[:, :, 1:, ...] = 0

            i2v_mask = torch.zeros(video_length)
            i2v_mask[0] = 1

            mask_concat = torch.ones(img_latents_concat.shape[0], 1, img_latents_concat.shape[2], img_latents_concat.shape[3],
                                     img_latents_concat.shape[4]).to(device=img_latents.device)
            mask_concat[:, :, 1:, ...] = 0

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

        target_dtype = PRECISION_TO_TYPE[self.args.precision]
        autocast_enabled = (
            target_dtype != torch.float32
        ) and not self.args.disable_autocast
        vae_dtype = PRECISION_TO_TYPE[self.args.vae_precision]
        vae_autocast_enabled = (
            vae_dtype != torch.float32
        ) and not self.args.disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(len(timesteps)):
                t = timesteps[i]
                if self.interrupt:
                    continue

                if i2v_mode and i2v_condition_type == "token_replace":
                    latents = torch.concat([img_latents, latents[:, :, 1:, :, :]], dim=2)

                # expand the latents if we are doing classifier free guidance
                if i2v_mode and i2v_condition_type == "latent_concat":
                    latent_model_input = torch.concat([latents, img_latents_concat, mask_concat], dim=1)
                else:
                    latent_model_input = latents

                latent_model_input = (
                    torch.cat([latent_model_input] * 2)
                    if self.do_classifier_free_guidance
                    else latent_model_input
                )

                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(target_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )

                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    noise_pred = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                        latent_model_input,  # [2, 16, 33, 24, 42]
                        t_expand,  # [2]
                        text_states=prompt_embeds,  # [2, 256, 4096]
                        text_mask=prompt_mask,  # [2, 256]
                        text_states_2=prompt_embeds_2,  # [2, 768]
                        freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                        freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                        guidance=guidance_expand,
                        return_dict=True,
                    )[
                        "x"
                    ]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(
                        noise_pred,
                        noise_pred_text,
                        guidance_rescale=self.guidance_rescale,
                    )
                # JENGA: also, get x_0 version of the latents.
                if i in time_step_split:
                    stage_idx += 1
                    img_latents = image_latent_list[stage_idx]
                    # JENGA: we need to prepare the latents for the next stage.
                    latents_noise = randn_tensor([batch_size * num_videos_per_prompt,
                                          num_channels_latents,
                                          latent_step_shapes[stage_idx][0], latent_step_shapes[stage_idx][1]//16 * 2, latent_step_shapes[stage_idx][2]//16 * 2],
                                          generator=generator,
                                          device=device, dtype=prompt_embeds.dtype)

                    if hasattr(self.scheduler, "init_noise_sigma"):
                        latents_noise = latents_noise * self.scheduler.init_noise_sigma

                    current_size = [latent_step_shapes[stage_idx][0], latent_step_shapes[stage_idx][1]//16, latent_step_shapes[stage_idx][2]//16]

                    # latent resize.
                    resize_size = [latent_step_shapes[stage_idx][0], latent_step_shapes[stage_idx][1]//16 * 2, latent_step_shapes[stage_idx][2]//16 * 2]
                    original_size = [latent_step_shapes[-1][0], latent_step_shapes[-1][1]//16, latent_step_shapes[-1][2]//16]
                    
                    if res_rate_list[stage_idx-1] != 1.0 or res_rate_list[stage_idx] != 1.0:
                        self.scheduler.config.shift = scheduler_shift_list[stage_idx]
                        self.scheduler.set_timesteps(num_inference_steps, device=device, n_tokens=n_tokens)
                        self.scheduler._step_index = i
                        timesteps = self.scheduler.timesteps
                        t = timesteps[i]
                        latents = self.scheduler.predict_x0_from_xt(
                            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                        )[0]
                        latents = torch.nn.functional.interpolate(latents, size=resize_size, mode="area")
                        latents = self.scheduler.add_noise_to_step(latents, latents_noise, timesteps[i+1])[0]
                        # update global variables.
                        if not hasattr(self.transformer, "curve_sels"):
                            linear_hilbert, hilbert_linear = gilbert_mapping(current_size[0], current_size[1], current_size[2])
                            self.transformer.current_t = current_size[0]
                            self.transformer.current_h = current_size[1]
                            self.transformer.current_w = current_size[2]
                            self.transformer.linear_to_hilbert = linear_hilbert
                            self.transformer.hilbert_order = hilbert_linear
                            self.transformer.curve_sel = None
                            self.transformer.sa_drop_rate = self.transformer.sa_drop_rates[stage_idx]
                        else:
                            self.transformer.curve_sel = self.transformer.curve_sels[stage_idx]
                            for im in range(len(self.transformer.curve_sel)):
                                if self.transformer.curve_sel[im][-1] is not None and self.transformer.curve_sel[im][-1].device != latents.device:
                                    self.transformer.curve_sel[im][-1] = self.transformer.curve_sel[im][-1].to(latents.device)
                            self.transformer.linear_to_hilbert = self.transformer.curve_sel[0][0]
                            self.transformer.hilbert_order = self.transformer.curve_sel[0][1]
                            self.transformer.sa_drop_rate = self.transformer.sa_drop_rates[stage_idx]

                        self.transformer.text_amp = 0.0
                        self.transformer.start_stage = True
                        freqs_cis = self.get_rotary_pos_embed(
                            current_size
                        )
                        self.scheduler._step_index += 1
                    else:
                        # compute the previous noisy sample x_t -> x_t-1
                        if i2v_mode and i2v_condition_type == "token_replace":
                            latents = self.scheduler.step(
                                noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                            )[0]
                            latents = torch.concat(
                                [img_latents, latents], dim=2
                            )
                        else:
                            latents = self.scheduler.step(
                                noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                            )[0]
                        self.transformer.sa_drop_rate = self.transformer.sa_drop_rates[stage_idx]
                else:
                    # compute the previous noisy sample x_t -> x_t-1
                    if i2v_mode and i2v_condition_type == "token_replace":
                        latents = self.scheduler.step(
                            noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                        )[0]
                        latents = torch.concat(
                            [img_latents, latents], dim=2
                        )
                    else:
                        latents = self.scheduler.step(
                            noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                        )[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop(
                        "negative_prompt_embeds", negative_prompt_embeds
                    )

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    if progress_bar is not None:
                        progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]
                else:
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()

        if i2v_mode and i2v_condition_type == "latent_concat":
            image = image[:, :, 4:, :, :]

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
