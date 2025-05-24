# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# modified from MInference code.
# here we implement an diff res version.

import numpy as np
import torch
import triton
import triton.language as tl
import time

import torch._dynamo
torch._dynamo.config.suppress_errors = True
from flash_attn import flash_attn_func

# from flash_attn import flash_attn_varlen_func
# import pycuda.autoprimaryctx
# from pycuda.compiler import SourceModule


# # @triton.autotune(
# #    configs=[
# #        triton.Config({}, num_stages=1, num_warps=4),
# #        triton.Config({}, num_stages=1, num_warps=8),
# #        triton.Config({}, num_stages=2, num_warps=4),
# #        triton.Config({}, num_stages=2, num_warps=8),
# #        triton.Config({}, num_stages=3, num_warps=4),
# #        triton.Config({}, num_stages=3, num_warps=8),
# #        triton.Config({}, num_stages=4, num_warps=4),
# #        triton.Config({}, num_stages=4, num_warps=8),
# #        triton.Config({}, num_stages=5, num_warps=4),
# #        triton.Config({}, num_stages=5, num_warps=8),
# #    ],
# #    key=['N_CTX'],
# # )

@triton.jit
def _triton_block_sparse_attn_fwd_kernel_onehot(
    Q, K, V, seqlens, qk_scale, text_amp_runtime, text_block_start_runtime,
    block_mask,  # [BATCH*HEADS, NUM_ROWS, NUM_BLOCKS] one-hot mask
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_bz, stride_bm, stride_bn,  # additional strides for block_mask
    Z, H, N_CTX,
    NUM_BLOCKS,  # total number of blocks
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    is_text_block: tl.constexpr,  # indicates whether this is a text block
):
    start_m = tl.program_id(0)  # Current query block being processed
    off_hz = tl.program_id(1)   # batch * head index

    seqlen = tl.load(seqlens + off_hz // H)
    if start_m * BLOCK_M >= seqlen:
        return

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    qo_offset = (off_hz // H) * stride_qz + (off_hz % H) * stride_qh
    kv_offset = (off_hz // H) * stride_kz + (off_hz % H) * stride_kh

    q_ptrs = Q + qo_offset + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K + kv_offset + offs_d[:, None] * stride_kk
    v_ptrs = V + kv_offset + offs_d[None, :] * stride_vk
    o_ptrs = Out + qo_offset + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok

    # Current block mask row corresponding to this batch*head and query block
    mask_ptr = block_mask + off_hz * stride_bz + start_m * stride_bm

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    
    # Iterate through all blocks (using one-hot mask)
    for block_idx in range(NUM_BLOCKS):
        # Check if current block is marked in the one-hot mask
        is_valid_block = tl.load(mask_ptr + block_idx * stride_bn)
        if is_valid_block:
            start_n = block_idx * BLOCK_N
            cols = start_n + offs_n
            
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn)
            
            # -- compute qk --
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            # Safer way to limit KV: use original m_mask, then apply kv range check after qk matrix calculation
            qk = tl.where(m_mask, qk, float("-inf"))
            qk += tl.dot(q, k)
            
            # Use runtime parameters
            is_text_block_cond = block_idx >= text_block_start_runtime
            qk = tl.where(is_text_block_cond, qk + text_amp_runtime, qk)
            
            # Create KV mask and apply
            kv_valid = cols[None, :] < seqlen
            qk = tl.where(kv_valid, qk, float("-inf"))
            
            # -- compute scaling constant --
            m_i_new = tl.maximum(m_i, tl.max(qk, 1))
            alpha = tl.math.exp2(m_i - m_i_new)
            p = tl.math.exp2(qk - m_i_new[:, None])
            
            # -- scale and update acc --
            acc_scale = l_i * 0 + alpha  # workaround some compiler bug
            acc *= acc_scale[:, None]
            acc += tl.dot(p.to(dtype), v)
            
            # -- update m_i and l_i --
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_i_new

    # write back O
    acc /= l_i[:, None]
    tl.store(o_ptrs, acc.to(dtype), mask=m_mask)


def _triton_block_sparse_attention_onehot(
    q,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    k,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    v,                 # [BATCH, N_HEADS, N_CTX, D_HEAD]
    seqlens,           # [BATCH, ]
    block_mask,        # [BATCH, N_HEADS, NUM_QUERIES, NUM_BLOCKS] one-hot boolean mask
    sm_scale,
    block_size_M=128,
    block_size_N=128,
    is_text_block=False,  # indicates whether this is a text block
    text_amp=0.0,         # controls scaling of qk values for text blocks
    text_block_start=0,   # starting index of text blocks
) -> torch.Tensor:
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.zeros_like(q)
    
    batch_size, n_heads = q.shape[0], q.shape[1]
    num_query_blocks = block_mask.shape[-2]
    num_blocks = block_mask.shape[-1]
    
    # 将block_mask重塑为[batch*heads, queries, blocks]以适应triton kernel
    block_mask_reshaped = block_mask.reshape(batch_size * n_heads, num_query_blocks, num_blocks)
    
    grid = (num_query_blocks, batch_size * n_heads, 1)
    
    if q.dtype == torch.bfloat16:
        dtype = tl.bfloat16
    else:
        dtype = tl.float16

    qk_scale = sm_scale * 1.44269504

    if not seqlens.device == q.device:
        seqlens = seqlens.to(q.device)
    if not block_mask_reshaped.device == q.device:
        block_mask_reshaped = block_mask_reshaped.to(q.device)
    
    with torch.cuda.device(q.device):
        _triton_block_sparse_attn_fwd_kernel_onehot[grid](
            q, k, v, seqlens, qk_scale, text_amp, text_block_start,
            block_mask_reshaped,
            o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            block_mask_reshaped.stride(0), block_mask_reshaped.stride(1), block_mask_reshaped.stride(2),
            q.shape[0], q.shape[1], q.shape[2],
            num_blocks,
            BLOCK_M=block_size_M, BLOCK_N=block_size_N,
            BLOCK_DMODEL=Lk,
            dtype=dtype,
            is_text_block=is_text_block,
        )
    return o

def _build_block_index_with_importance_optimized(
    query: torch.Tensor,     # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,       # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    text_start_block: int = None,  
    num_blocks: int = None,        
    prob_threshold: float = 0.7,   
    text_blocks: int = 2,          
    block_neighbor_list: torch.Tensor = None,  # [block_num, block_num] one-hot tensor
):
    cur_time = time.time()
    batch_size, num_heads, context_size, head_dim = query.shape
    num_query_blocks = (context_size + block_size_M - 1) // block_size_M
    device = query.device
    
    # 1. Pool queries and keys
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)
    
    # 2. Calculate attention scores - using bmm optimization
    # Reshape to [batch_size * num_heads, num_query_blocks, head_dim]
    q_bmm = query_pool.reshape(batch_size * num_heads, query_pool.shape[2], head_dim)
    
    # Reshape to [batch_size * num_heads, head_dim, num_key_blocks]
    k_bmm = key_pool.reshape(batch_size * num_heads, key_pool.shape[2], head_dim).transpose(1, 2)
    
    # Use bmm for batch matrix multiplication
    attention_scores_flat = torch.bmm(q_bmm, k_bmm) * (head_dim ** -0.5)
    
    # Reshape back to original dimensions [batch_size, num_heads, num_query_blocks, num_key_blocks]
    attention_scores = attention_scores_flat.reshape(
        batch_size, num_heads, query_pool.shape[2], key_pool.shape[2]
    )
    
    # 3. Only process scores for non-text blocks
    normal_scores = attention_scores[:, :, :, :text_start_block]
    
    # 4. Use direct softmax to calculate probability distribution for each query
    probs = torch.softmax(normal_scores, dim=-1)
    
    # 5. Sort probability distribution for each head and query
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 6. Find number of blocks needed for each (batch, head, query) position
    mask = cumsum_probs <= prob_threshold
    num_blocks_needed = mask.sum(dim=-1) + 1  # [batch, heads, queries]
    num_blocks_needed = torch.maximum(
        num_blocks_needed,
        torch.tensor(top_k, device=device)
    )
    
    # Create one-hot output tensor [batch_size, num_heads, num_query_blocks, num_blocks]
    one_hot_output = torch.zeros(
        (batch_size, num_heads, num_query_blocks, num_blocks), 
        dtype=torch.bool, device=device
    )
    max_k = indices.shape[-1]
    # Use einsum-based indexing for reduced memory:
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1).expand(-1, num_heads, num_query_blocks, max_k)
    head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1).expand(batch_size, -1, num_query_blocks, max_k)
    query_idx = torch.arange(num_query_blocks, device=device).view(1, 1, -1, 1).expand(batch_size, num_heads, -1, max_k)
    k_idx = torch.arange(max_k, device=device).view(1, 1, 1, -1).expand(batch_size, num_heads, num_query_blocks, -1)

    # Create mask more efficiently
    valid_mask = k_idx < num_blocks_needed.unsqueeze(-1)
    
    # Find all positions that need to be filled
    b_indices = batch_idx[valid_mask]
    h_indices = head_idx[valid_mask]
    q_indices = query_idx[valid_mask]
    
    # Get index values corresponding to these positions
    flat_indices = indices[b_indices, h_indices, q_indices, k_idx[valid_mask]]
    
    # Use scatter and index operations to fill in one go
    one_hot_output[b_indices, h_indices, q_indices, flat_indices] = True
    
    
    # Add physical neighbors - directly take union
    if block_neighbor_list is not None:
        # Ensure block_neighbor_list is on the correct device
        if block_neighbor_list.device != device:
            block_neighbor_list = block_neighbor_list.to(device)
        
        # Ensure dimensions match and convert to boolean
        neighbor_mask = block_neighbor_list[:num_query_blocks, :text_start_block].bool()
        
        # Expand to [batch, heads, q_blocks, blocks] dimension and take union with existing output
        one_hot_output[:, :, :neighbor_mask.shape[0], :text_start_block] |= neighbor_mask.unsqueeze(0).unsqueeze(0)
    
    # Add text blocks - all batches, all heads, all query blocks can see all text blocks
    if text_blocks > 0 and text_start_block is not None:
        one_hot_output[:, :, :, text_start_block:min(text_start_block+text_blocks, num_blocks)] = True

    return one_hot_output


def block_sparse_attention_combined(
    query: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    key: torch.Tensor,    # [BATCH, N_HEADS, N_CTX, D_HEAD]
    value: torch.Tensor,  # [BATCH, N_HEADS, N_CTX, D_HEAD]
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_kv: int = None,
    text_blocks: int = 2,  # Number of text blocks at the end
    text_amp: float = 1.0,  # controls scaling of qk values for text blocks
    prob_threshold: float = 0.5,  # new parameter
    block_neighbor_list: torch.Tensor = None,
    shape_xfuse: bool = False,
):
    """
    Combined attention processing for normal blocks and text blocks:
    1. Normal blocks select top-k blocks based on importance (without causal constraints)
    2. Text blocks get full attention (can see all blocks)
    3. All normal blocks can see all text blocks
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    batch_size, num_heads, context_size, head_dim = query.shape
    
    # 处理可变长度序列
    if cu_seqlens_q is not None and cu_seqlens_kv is not None:
        seqlens = cu_seqlens_q[1:2]
        seqlens = seqlens.to(torch.int32).to(query.device)
        pad_q, pad_kv = 0, 0
    else:
        pad = block_size_M - (context_size % block_size_M) if context_size % block_size_M != 0 else 0
        query_padded = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
        key_padded = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
        value_padded = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
        seqlens = torch.tensor([context_size] * batch_size, dtype=torch.int32, device=query.device)
    
    sm_scale = head_dim ** -0.5
    padded_context_size = query.shape[2]
    num_blocks = (padded_context_size + block_size_M - 1) // block_size_M
    
    # Compute normal_blocks, normal_tokens only once
    normal_blocks = num_blocks - text_blocks
    normal_tokens = normal_blocks * block_size_M
    
    # Pre-compute pooled query and key for block index building
    if normal_blocks > 0:
        query_normal = query[:, :, :normal_tokens, :]
        
        # Pass pre-computed pools to block index function
        block_relation_onehot = _build_block_index_with_importance_optimized(
            query_normal, key, top_k, block_size_M, block_size_N, 
            text_start_block=normal_blocks, num_blocks=num_blocks,
            prob_threshold=prob_threshold,
            text_blocks=text_blocks,
            block_neighbor_list=block_neighbor_list
        )
        
        # direct use one-hot version sparse attention
        output_normal = _triton_block_sparse_attention_onehot(
            query_normal, key, value, seqlens, 
            block_relation_onehot, sm_scale, block_size_M, block_size_N,
            is_text_block=False,  # this is not a text block
            text_amp=text_amp,         # text_amp
            text_block_start=normal_blocks,   # text block start index
        )
    else:
        output_normal = torch.empty(0, device=query.device)
    
    # 2. process text blocks (full attention to all blocks)
    if text_blocks > 0:
        # extract text blocks
        query_text = query[:, :, normal_tokens:, :]
        key_text = key  # can see all keys
        value_text = value
        # use Flash Attention
        output_text = flash_attn_func(
            query_text.permute(0, 2, 1, 3), key_text.permute(0, 2, 1, 3), value_text.permute(0, 2, 1, 3),
            causal=False, softmax_scale=sm_scale
        ).transpose(1, 2)
    else:
        output_text = torch.empty(0, device=query.device)
    
    # merge outputs
    if normal_blocks > 0 and text_blocks > 0:
        output = torch.cat([output_normal, output_text], dim=2)
    elif normal_blocks > 0:
        output = output_normal
    else:
        output = output_text
    
    if not shape_xfuse:
        output = output.permute(0, 2, 1, 3).reshape(batch_size, context_size, -1)
        return output
    # remove padding
    return output.permute(0, 2, 1, 3)

# keep the original function as an alias for backward compatibility
def block_sparse_attention(
    query: torch.Tensor,
    key: torch.Tensor,     
    value: torch.Tensor,
    top_k: int,
    block_size_M: int = 128,
    block_size_N: int = 128,
    cu_seqlens_q: torch.Tensor = None,
    cu_seqlens_kv: torch.Tensor = None,
    max_seqlen_q: int = None,
    max_seqlen_kv: int = None,
    text_blocks: int = 2,
    text_amp: float = 0.0,
    block_neighbor_list: torch.Tensor = None,
    shape_xfuse: bool = False,
    p_remain_rates: float = 0.5,
):
    """
    backward compatible wrapper around block_sparse_attention_combined.
    """
    return block_sparse_attention_combined(
        query, key, value, top_k, block_size_M, block_size_N,
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, 
        text_blocks, text_amp, block_neighbor_list=block_neighbor_list, shape_xfuse=shape_xfuse,
        prob_threshold=p_remain_rates
    )