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
def _triton_block_sparse_attn_fwd_kernel(
    Q, K, V, seqlens, sm_scale, text_amp_runtime, text_block_start_runtime,
    block_index,
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, N_CTX,
    NUM_ROWS, MAX_BLOCKS_PRE_ROW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    is_text_block: tl.constexpr,  # 表示当前是否是文本块
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)

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

    blocks_ptr = block_index + (off_hz * NUM_ROWS + start_m) * MAX_BLOCKS_PRE_ROW

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # scale sm_scale by log_2(e) and use
    # 2^x instead of exp in the loop because CSE and LICM
    # don't work as expected with `exp` in the loop
    qk_scale = sm_scale * 1.44269504
    # load q: it will stay in SRAM throughout
    q = tl.load(q_ptrs)
    q = (q * qk_scale).to(dtype)

    # loop over k, v and update accumulator
    m_mask = offs_m[:, None] < seqlen
    
    # 为文本块使用最大块数，为普通块使用top-k
    if is_text_block:
        # 文本块 - 使用所有可用块 (full attention)
        block_count = MAX_BLOCKS_PRE_ROW                                                 
    else:
        # 普通块 - 使用所有可用块（基于重要性选择的top-k个块）
        block_count = MAX_BLOCKS_PRE_ROW

    for sparse_block_idx in range(block_count):
        real_block_idx = tl.load(blocks_ptr + sparse_block_idx)
        is_valid_block = real_block_idx >= 0
        if is_valid_block:
            start_n = real_block_idx * BLOCK_N
            cols = start_n + offs_n
            
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn)
            
            # -- compute qk --
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            # 更安全的方式限制KV：使用原始的m_mask，然后在qk矩阵计算后再应用kv范围检查
            qk = tl.where(m_mask, qk, float("-inf"))
            qk += tl.dot(q, k)
            
            # 使用运行时参数
            is_text_block_cond = real_block_idx >= text_block_start_runtime
            qk = tl.where(is_text_block_cond, qk + text_amp_runtime, qk)
            
            # 创建KV掩码并应用 - 这里注意用法，避免维度不匹配问题
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

@triton.jit
def _triton_block_sparse_attn_fwd_kernel_onehot(
    Q, K, V, seqlens, qk_scale, text_amp_runtime, text_block_start_runtime,
    block_mask,  # [BATCH*HEADS, NUM_ROWS, NUM_BLOCKS] one-hot mask
    Out,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_bz, stride_bm, stride_bn,  # 额外的block_mask的步长
    Z, H, N_CTX,
    NUM_BLOCKS,  # 总块数
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    dtype: tl.constexpr,
    is_text_block: tl.constexpr,  # 表示当前是否是文本块
):
    start_m = tl.program_id(0)  # 当前处理的query块
    off_hz = tl.program_id(1)   # batch * head索引

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

    # 当前batch*head和query块对应的block mask行
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
    
    # 遍历所有块 (使用one-hot mask)
    for block_idx in range(NUM_BLOCKS):
        # 检查当前块是否在one-hot mask中被标记
        is_valid_block = tl.load(mask_ptr + block_idx * stride_bn)
        if is_valid_block:
            start_n = block_idx * BLOCK_N
            cols = start_n + offs_n
            
            # -- load k, v --
            k = tl.load(k_ptrs + cols[None, :] * stride_kn)
            v = tl.load(v_ptrs + cols[:, None] * stride_vn)
            
            # -- compute qk --
            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            
            # 更安全的方式限制KV：使用原始的m_mask，然后在qk矩阵计算后再应用kv范围检查
            qk = tl.where(m_mask, qk, float("-inf"))
            qk += tl.dot(q, k)
            
            # 使用运行时参数
            is_text_block_cond = block_idx >= text_block_start_runtime
            qk = tl.where(is_text_block_cond, qk + text_amp_runtime, qk)
            
            # 创建KV掩码并应用
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
    block_mask,        # [BATCH, N_HEADS, NUM_QUERIES, NUM_BLOCKS] one-hot布尔掩码
    sm_scale,
    block_size_M=128,
    block_size_N=128,
    is_text_block=False,  # 指示当前是否是文本块
    text_amp=0.0,         # 控制文本块的qk值缩放
    text_block_start=0,   # 文本块开始的索引
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
    debug_print: bool = False,
    block_neighbor_list: torch.Tensor = None,  # [block_num, block_num] one-hot tensor
):
    cur_time = time.time()
    batch_size, num_heads, context_size, head_dim = query.shape
    num_query_blocks = (context_size + block_size_M - 1) // block_size_M
    device = query.device
    
    # 1. 池化查询和键
    query_pool = query.reshape((batch_size, num_heads, -1, block_size_M, head_dim)).mean(dim=-2)
    key_pool = key.reshape((batch_size, num_heads, -1, block_size_N, head_dim)).mean(dim=-2)
    
    # 2. 计算注意力分数 - 使用bmm优化
    # 重新整形为 [batch_size * num_heads, num_query_blocks, head_dim]
    q_bmm = query_pool.reshape(batch_size * num_heads, query_pool.shape[2], head_dim)
    
    # 重新整形为 [batch_size * num_heads, head_dim, num_key_blocks]
    k_bmm = key_pool.reshape(batch_size * num_heads, key_pool.shape[2], head_dim).transpose(1, 2)
    
    # 使用bmm进行批量矩阵乘法
    attention_scores_flat = torch.bmm(q_bmm, k_bmm) * (head_dim ** -0.5)
    
    # 重新整形回原始维度 [batch_size, num_heads, num_query_blocks, num_key_blocks]
    attention_scores = attention_scores_flat.reshape(
        batch_size, num_heads, query_pool.shape[2], key_pool.shape[2]
    )
    
    # 3. 只处理非文本块部分的分数
    normal_scores = attention_scores[:, :, :, :text_start_block]
    
    # 4. 使用直接softmax计算每个查询的概率分布
    probs = torch.softmax(normal_scores, dim=-1)
    
    # 5. 对每个head的每个query的概率分布排序
    sorted_probs, indices = torch.sort(probs, dim=-1, descending=True)
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # 6. 找到每个(batch, head, query)位置需要的block数量
    mask = cumsum_probs <= prob_threshold
    num_blocks_needed = mask.sum(dim=-1) + 1  # [batch, heads, queries]
    num_blocks_needed = torch.maximum(
        num_blocks_needed,
        torch.tensor(top_k, device=device)
    )
    
    # 创建返回的one-hot张量 [batch_size, num_heads, num_query_blocks, num_blocks]
    one_hot_output = torch.zeros(
        (batch_size, num_heads, num_query_blocks, num_blocks), 
        dtype=torch.bool, device=device
    )
    max_k = indices.shape[-1]
    # 使用einsum-based indexing for reduced memory:
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1).expand(-1, num_heads, num_query_blocks, max_k)
    head_idx = torch.arange(num_heads, device=device).view(1, -1, 1, 1).expand(batch_size, -1, num_query_blocks, max_k)
    query_idx = torch.arange(num_query_blocks, device=device).view(1, 1, -1, 1).expand(batch_size, num_heads, -1, max_k)
    k_idx = torch.arange(max_k, device=device).view(1, 1, 1, -1).expand(batch_size, num_heads, num_query_blocks, -1)

    # Create mask more efficiently
    valid_mask = k_idx < num_blocks_needed.unsqueeze(-1)
    
    # 找出所有需要填充的位置
    b_indices = batch_idx[valid_mask]
    h_indices = head_idx[valid_mask]
    q_indices = query_idx[valid_mask]
    
    # 获取这些位置对应的索引值
    flat_indices = indices[b_indices, h_indices, q_indices, k_idx[valid_mask]]
    
    # 使用scatter和索引操作一次性填充
    one_hot_output[b_indices, h_indices, q_indices, flat_indices] = True
    
    
    # 添加物理邻居 - 直接取并集
    if block_neighbor_list is not None:
        # 确保block_neighbor_list在正确的设备上
        if block_neighbor_list.device != device:
            block_neighbor_list = block_neighbor_list.to(device)
        
        # 确保尺寸匹配并转换为布尔型
        neighbor_mask = block_neighbor_list[:num_query_blocks, :text_start_block].bool()
        
        # 扩展到[batch, heads, q_blocks, blocks]维度并与现有输出取并集
        one_hot_output[:, :, :neighbor_mask.shape[0], :text_start_block] |= neighbor_mask.unsqueeze(0).unsqueeze(0)
    
    # 添加文本块 - 所有批次、所有头、所有查询块都能看到所有文本块
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
    text_blocks: int = 0,  # Number of text blocks at the end
    text_amp: float = 1.0,  # 控制文本块的qk值缩放
    prob_threshold: float = 0.5,  # 新参数
    block_neighbor_list: torch.Tensor = None,
    shape_xfuse: bool = False,
):
    """
    组合注意力处理普通块和文本块:
    1. 普通块基于重要性选择top-k个块（没有因果约束）
    2. 文本块获得全注意力（可以看到所有块）
    3. 所有普通块都能看到所有文本块
    """
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    batch_size, num_heads, context_size, head_dim = query.shape
    out_dtype = query.dtype
    
    # 处理可变长度序列

    pad = block_size_M - (context_size % block_size_M) if context_size % block_size_M != 0 else 0
    query = torch.nn.functional.pad(query, [0, 0, 0, pad, 0, 0, 0, 0])
    key = torch.nn.functional.pad(key, [0, 0, 0, pad, 0, 0, 0, 0])
    value = torch.nn.functional.pad(value, [0, 0, 0, pad, 0, 0, 0, 0])
    seqlens = torch.tensor([context_size] * batch_size, dtype=torch.int32).to(
                device=query.device, non_blocking=True)
    # print(f"padded query shape: {query.shape}, pad: {pad}")
        
    dtype = torch.bfloat16

    def half(x):
        return x if x.dtype == dtype else x.to(dtype)

    query = query.to(value.dtype)
    key = key.to(value.dtype)
    sm_scale = head_dim ** -0.5
    padded_context_size = query.shape[2]
    num_blocks = (padded_context_size + block_size_M - 1) // block_size_M
    # print(f"padded_context_size: {padded_context_size}, num_blocks: {num_blocks}")
    
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
        
        # 直接使用one-hot版本的sparse attention
        output_normal = _triton_block_sparse_attention_onehot(
            query_normal, key, value, seqlens, 
            block_relation_onehot, sm_scale, block_size_M, block_size_N,
            is_text_block=False,  # 这不是文本块
            text_amp=text_amp,         # 控制文本块的qk值缩放
            text_block_start=normal_blocks,   # 文本块开始的索引
        )
    else:
        output_normal = torch.empty(0, device=query.device)
    
    # 2. 处理文本块（完全注意力到所有块）
    if text_blocks > 0:
        # 提取文本块
        query_text = query[:, :, normal_tokens:, :]
        key_text = key  # 可以看到所有key
        value_text = value
        # 使用Flash Attention
        output_text = flash_attn_func(
            query_text.permute(0, 2, 1, 3), key_text.permute(0, 2, 1, 3), value_text.permute(0, 2, 1, 3),
            causal=False, softmax_scale=sm_scale
        ).transpose(1, 2)
    else:
        output_text = torch.empty(0, device=query.device)
    
    # 合并输出
    if normal_blocks > 0 and text_blocks > 0:
        output = torch.cat([output_normal, output_text], dim=2)[:, :, :context_size, :]
    elif normal_blocks > 0:
        output = output_normal[:, :, :context_size, :]
    else:
        output = output_text
    
    if not shape_xfuse:
        output = output.permute(0, 2, 1, 3).reshape(batch_size, context_size, -1)
        return output.type(out_dtype)
    # 移除填充
    return output.permute(0, 2, 1, 3).type(out_dtype)

# 保持原始函数作为后向兼容性的别名
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
    text_blocks: int = 0,
    text_amp: float = 0.0,
    block_neighbor_list: torch.Tensor = None,
    shape_xfuse: bool = False,
    p_remain_rates: float = 0.9,
):
    """
    围绕block_sparse_attention_combined的后向兼容包装器。
    """
    # print(f"p_remain_rates: {p_remain_rates}")
    return block_sparse_attention_combined(
        query, key, value, top_k, block_size_M, block_size_N,
        cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv, 
        text_blocks, text_amp, block_neighbor_list=block_neighbor_list, shape_xfuse=shape_xfuse,
        prob_threshold=p_remain_rates
    )