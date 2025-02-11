# https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py

# Copyright 2023-2024 SGLang Team
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
"""
Memory-efficient attention for prefill.
It supports page size = 1 and prefill with KV cache (i.e. extend).
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.triton_ops.prefill_attention import (
    context_attention_fwd,
)
from sglang.srt.utils import is_hip

is_cuda_available = torch.cuda.is_available()
if is_cuda_available:
    CUDA_CAPABILITY = torch.cuda.get_device_capability()

is_hip_ = is_hip()


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel(
    Q_Extend,
    K_Extend,
    V_Extend,
    O_Extend,
    K_Buffer,
    V_Buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    mask_ptr,
    mask_offsets,
    sm_scale,
    kv_group_num,
    stride_qbs,
    stride_qh,
    stride_kbs,
    stride_kh,
    stride_vbs,
    stride_vh,
    stride_obs,
    stride_oh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    logit_cap: tl.constexpr,
    Lq: tl.constexpr,
    Lv: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DPE: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_CUSTOM_MASK: tl.constexpr,
):
    cur_seq = tl.program_id(0) #  grid 定义是 [bs, nhead, nblock]。batch 中每条就是 cur_seq 
    cur_head = tl.program_id(1) 
    cur_block_m = tl.program_id(2)
    cur_kv_head = cur_head // kv_group_num

    cur_seq_extend_start_idx = tl.load(qo_indptr + cur_seq) 
    cur_seq_len_extend = tl.load(qo_indptr + cur_seq + 1) - cur_seq_extend_start_idx # extend_q 
    cur_seq_kv_start_idx = tl.load(kv_indptr + cur_seq) # kv_indptr[cur_seq]  
    cur_seq_len_prefix = tl.load(kv_indptr + cur_seq + 1) - cur_seq_kv_start_idx
    cur_seq_len = cur_seq_len_prefix + cur_seq_len_extend # kv cache for preill 

    if USE_CUSTOM_MASK:
        cur_seq_mask_start_idx = tl.load(mask_offsets + cur_seq)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    offs_m = tl.arange(0, BLOCK_M)
    mask_m = (cur_block_m * BLOCK_M + offs_m) < cur_seq_len_extend

    mask_d = offs_d < Lq
    mask_dv = offs_dv < Lv

    offs_q = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_qbs
        + cur_head * stride_qh
        + offs_d[None, :]
    )
    q = tl.load(
        Q_Extend + offs_q, mask=(mask_m[:, None]) & (mask_d[None, :]), other=0.0
    )
    """
        offs_q =  [BLOCK_M, 1] * stride_qbs + cur_head * stride_qh + offs_d[1, BLOCK_DMODEL]
        q shape 与 off_q 一致： [BLOCK_M, BLOCK_DMODEL]
    """

    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        offs_qpe = (
            (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
            * stride_qbs
            + cur_head * stride_qh
            + offs_dpe[None, :]
        )
        qpe = tl.load(Q_Extend + offs_qpe, mask=mask_m[:, None], other=0.0)

    # stage 1: compute scores with prefix 。计算 prefix 部分的 attn
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_DV], dtype=tl.float32)
    deno = tl.zeros([BLOCK_M], dtype=tl.float32)
    e_max = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")

    for start_n in range(0, cur_seq_len_prefix, BLOCK_N): # cur_seq_len_prefix 按 BLOCK_N tokens 分到 chunks 中
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_seq_len_prefix
        offs_kv_loc = tl.load(
            kv_indices + cur_seq_kv_start_idx + start_n + offs_n, mask=mask_n, other=0
        )   # offs_kv_loc 中  BLOCK_N 个 token 的 首地址

        # load k in transposed way
        offs_buf_k = (
            offs_kv_loc[None, :] * stride_buf_kbs
            + cur_kv_head * stride_buf_kh
            + offs_d[:, None]
        )
        """
            offs_buf_k = offs_kv_loc[1, BLOCK_N] * stride_buf_kbs + .. + offs_d[BLOCK_DMODEL, 1]
            buf_k 与 offs_buf_k shape 一致: [BLOCK_DMODEL, BLOCK_N]
        """
        k = tl.load(
            K_Buffer + offs_buf_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )

        qk = tl.dot(q.to(k.dtype), k) # qk[BLOCk_M, BLOCK_N] = q[BLOCK_M, BLOCK_DMODEL] * k[BLOCK_DMODEL, BLOCK_N]
        if BLOCK_DPE > 0:
            offs_kpe = (
                offs_kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Buffer + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe.to(kpe.dtype), kpe)
        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_buf_v = (
            offs_kv_loc[:, None] * stride_buf_vbs
            + cur_kv_head * stride_buf_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Buffer + offs_buf_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        """
            offs_buf_v = offs_kv_loc[BLOCK_N, 1] * stride_buf_vbs + .. + offs_dv[1, BLOCK_DV]
            v 与 offs_buf_v shape 一致：[BLOCK_N, BLOCK_DV]
        """
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)  # acc[BLOCK_M, BLOCK_DV] = p[BLOCk_M, BLOCK_N] * v[BLOCK_N, BLOCK_DV]

        e_max = n_e_max

    # stage 2: compute the triangle part。计算 extend 部分的 attn

    cur_block_m_end = tl.minimum(cur_seq_len_extend, (cur_block_m + 1) * BLOCK_M)
    for start_n in range(0, cur_block_m_end, BLOCK_N): # cur_seq_len_extend 按 BLOCK_N tokens 分到 chunks 中
        start_n = tl.multiple_of(start_n, BLOCK_N)
        mask_n = (start_n + offs_n) < cur_block_m_end

        # load k in transposed way
        offs_k = (
            (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
            + cur_kv_head * stride_kh
            + offs_d[:, None]
        )
        k = tl.load(
            K_Extend + offs_k, mask=(mask_n[None, :]) & (mask_d[:, None]), other=0.0
        )
        """
            offs_k = (.. + offs_n[1, BLOCK_N]) * stride_kbs + .. + offs_d[BLOCK_DMODEL, 1] 
            k 与 offs_k shape 一致:  [BLOCK_DMODEL, BLOCK_N]
        """
        qk = tl.dot(q, k, out_dtype=tl.float32) 
        """
            同样的q tensor, 但是与 extend k tensor 做 attn 计算 
            extend_qk[BLOCK_M, BLOCK_N] = q[BLOCK_M, BLOCK_DMODEL] * k[BLOCK_DMODEL, BLOCK_N]
            即 extend的计算，与 prefix 的计算使用了同样的 block_size for DMODEL, M, N
        """
        if BLOCK_DPE > 0:
            offs_kpe = (
                (cur_seq_extend_start_idx + start_n + offs_n[None, :]) * stride_kbs
                + cur_kv_head * stride_kh
                + offs_dpe[:, None]
            )
            kpe = tl.load(
                K_Extend + offs_kpe,
                mask=mask_n[None, :],
                other=0.0,
            )
            qk += tl.dot(qpe, kpe)

        qk *= sm_scale

        if logit_cap > 0:
            qk = logit_cap * tanh(qk / logit_cap)

        if USE_CUSTOM_MASK:
            custom_mask = tl.load(
                mask_ptr
                + cur_seq_mask_start_idx
                + (cur_block_m * BLOCK_M + offs_m[:, None]) * cur_seq_len
                + cur_seq_len_prefix
                + start_n
                + offs_n[None, :],
                mask=(mask_m[:, None] & mask_n[None, :]),
                other=0,
            )
            custom_mask &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(custom_mask, qk, float("-inf"))
        else:
            mask_causual = (cur_block_m * BLOCK_M + offs_m[:, None]) >= (
                start_n + offs_n[None, :]
            )
            mask_causual &= mask_m[:, None] & mask_n[None, :]
            qk = tl.where(mask_causual, qk, float("-inf"))

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        deno = deno * re_scale + tl.sum(p, 1)

        offs_v = (
            (cur_seq_extend_start_idx + start_n + offs_n[:, None]) * stride_vbs
            + cur_kv_head * stride_vh
            + offs_dv[None, :]
        )
        v = tl.load(
            V_Extend + offs_v, mask=mask_n[:, None] & mask_dv[None, :], other=0.0
        )
        """
            offs_v = (.. + offs_n[BLOCK_N, 1]) * stride_vbs + .. + offs_dv[1, BLOCK_DV]
            v 与 offs_v 同shape:  [BLOCK_N, BLOCK_DV]
        """
        p = p.to(v.dtype)
        acc = acc * re_scale[:, None] + tl.dot(p, v)  
        """
            注意，这里 acc 已经叠加了 prefix attn 的计算结果
            extend_acc[BLOCK_M, BLOCK_DV] = p[BLOCK_M, BLOCK_N] * v[BLOCK_N, BLOCK_DV]
        """
        e_max = n_e_max

    offs_o = (
        (cur_seq_extend_start_idx + cur_block_m * BLOCK_M + offs_m[:, None])
        * stride_obs
        + cur_head * stride_oh
        + offs_dv[None, :]
    )
    tl.store(
        O_Extend + offs_o, acc / deno[:, None], mask=mask_m[:, None] & mask_dv[None, :]
    )
    """
        offs_o = (.. + offs_m[BLOCK_M, 1]) * stride_obs + .. + offs_dv[1, BLOCK_DV]，与 acc shape 一致
    """


def extend_attention_fwd(
    q_extend,  # [extend_token_num, head_num,  head_dim(D)]
    k_extend,  # [extend_token_num, kv_head_num, D]
    v_extend,  # [extend_token_num, kv_head_num, D] 
    o_extend,  # [max_token_num, head_num, D] # TODO 
    k_buffer,  # 保存 prefix 部分的 k buffer [max_token_num, kv_head_num, head_dim(D)]
    v_buffer,  # 保存 prefix 部分的 v buffer [max_token_num, kv_head_num, D] 
    qo_indptr, # [bs+1], array in prefix sum style 
    kv_indptr,  # [bs + 1] , array in prefix sum style
    kv_indices, # 所有chunk上 kvcache slots
    custom_mask, 
    mask_offsets,
    max_len_extend,
    sm_scale=None,
    logit_cap=0.0,
):
    """
    https://xdkkkgyt8c.feishu.cn/wiki/LrPowtneFiAwdmkt4zjcYuqQnye
    Extend Attention 和普通 Prefill 阶段的 Attention 算子的区别：
        1. 在 Extend 阶段（相当于正常情况的 Prefill 阶段），一条 Request 的 Prefix 部分已经有 KV Cache，保存在 K_Buffer/V_Buffer 中
        2. Extend 阶段输入的 input_ids 只有 extend 部分的 Token, 所以在计算 Attention 时，输入值是 extend 部分的 token 的 QKV(Q_Extend/K_Extend/V_Extend), prefix 部分的 token 会复用 K_Buffer/V_Buffer
        3. Extend 阶段的输出也是只有 extend 部分的 Token 对应的 hidden_states

    q_extend, k_extend, v_extend, o_extend: contiguous tensors
    k_buffer, v_buffer: (prefix + extend) tensors in mem_manager

    """
    Lq, Lk, Lv = (
        q_extend.shape[-1], # q_head_dim
        k_extend.shape[-1], # k_head_dim
        v_extend.shape[-1], # v_head_dim 
    )

    if Lq == 576:
        BLOCK_DMODEL = 512
        BLOCK_DPE = 64
    elif Lq == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    elif Lq == 192:
        BLOCK_DMODEL = 128
        BLOCK_DPE = 64
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lq)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv)

    if is_hip_:
        BLOCK_M, BLOCK_N = (64, 64)
        num_warps = 4

    else:
        if is_cuda_available and CUDA_CAPABILITY[0] >= 9:
            if Lq <= 256:
                BLOCK_M, BLOCK_N = (128, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        elif is_cuda_available and CUDA_CAPABILITY[0] >= 8:
            if Lq <= 128:
                BLOCK_M, BLOCK_N = (128, 128)
            elif Lq <= 256:
                BLOCK_M, BLOCK_N = (64, 64)
            else:
                BLOCK_M, BLOCK_N = (32, 64)
        else:
            BLOCK_M, BLOCK_N = (64, 64) if Lq <= 128 else (32, 32)

        num_warps = 4 if Lk <= 64 else 8
    
    """
        * BLOCK_DMODEL: per tl.block handle the size from head_dim 
        * BLOCK_DPE: per tl.block handle the size from Postional Encoding 
        * BLOCK_DV: power 2 most close to V_head_dim , per tl.block handle the size from v head_dim 
        * BLOCK_N: per tl.block handle the size from kv_heads  
        * BLOCK_M: per tl.block handle the size from bs * q_heads
    """

    sm_scale = sm_scale or 1.0 / (Lq**0.5)
    batch_size, head_num = qo_indptr.shape[0] - 1, q_extend.shape[1]
    kv_group_num = q_extend.shape[1] // k_extend.shape[1]

    USE_CUSTOM_MASK = custom_mask is not None

    grid = (batch_size, head_num, triton.cdiv(max_len_extend, BLOCK_M))
    """
        tl.block along grid-0: bs 
        tl.block along grid-1: head_num
        tl.block along grid-2: 处理 bs*heads 中 ith 分组, 每个 tl.block 处理 BLOCK_M 个 heads 
    """
    num_stages = 1

    extra_kargs = {}
    if is_hip_:
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel[grid](
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer,
        v_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_offsets,
        sm_scale,
        kv_group_num,
        q_extend.stride(0),  # head_num * D 
        q_extend.stride(1),  # D
        k_extend.stride(0),  # kv_head_num * D
        k_extend.stride(1),  # D
        v_extend.stride(0),  # kv_head_num * D
        v_extend.stride(1),  # D 
        o_extend.stride(0),  # head_num * D
        o_extend.stride(1),  # D 
        k_buffer.stride(0),  # kv_head_num * D 
        k_buffer.stride(1),  # D 
        v_buffer.stride(0),  # kv_head_num * D 
        v_buffer.stride(1),
        logit_cap=logit_cap,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        Lq=Lq,
        Lv=Lv,
        USE_CUSTOM_MASK=USE_CUSTOM_MASK,
        num_warps=num_warps,
        num_stages=num_stages,
        **extra_kargs,
    )


def redundant_attention(
    q_extend,
    o_extend,
    k_buffer,
    v_buffer,
    b_req_idx,
    b_start_loc,
    b_seq_len,
    b_seq_len_prefix,
    max_len_in_batch,
):
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = torch.empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer, k_buffer, v_buffer, o_buffer, b_start_loc, b_seq_len, max_len_in_batch
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend
