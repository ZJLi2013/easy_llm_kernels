# kernel impl https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
# kernel bench https://github.com/sgl-project/sglang/blob/main/benchmark/kernels/decoding_attention_triton/triton_flashinfer_cudnn.py

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
Memory-efficient attention for decoding.
It supports page size = 1.
"""

# Adapted from
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage1.py
# https://github.com/ModelTC/lightllm/blob/96353e868a840db4d103138caf15ed9dbea8c186/lightllm/models/deepseek2/triton_kernel/gqa_flash_decoding_stage2.py

import logging

import triton
import triton.language as tl

from utils import is_hip

is_hip_ = is_hip()

logger = logging.getLogger(__name__)

# TODO: Remove this when triton>=3.2.0. This issue will not affect performance and accuracy.
logger.warning(
    "The following error message 'operation scheduled before its operands' can be ignored."
)


@triton.jit
def tanh(x):
    # Tanh is just a scaled sigmoid
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _fwd_kernel_stage1(
    Q,   # (B, H_Q, D)  decode 阶段 batch 中每条 request 的 input_id 都只有一个 token 
    K_Buffer, # (max_total_num_token, H_KV, D)
    V_Buffer, 
    sm_scale, # = 1 /sqrt(D) 
    kv_indptr,
    kv_indices,
    Att_Out,
    stride_qbs,
    stride_qh,
    stride_buf_kbs,
    stride_buf_kh,
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    split_kv_id = tl.program_id(2)

    cur_kv_head = cur_head // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx

    off_q = cur_batch * stride_qbs + cur_head * stride_qh + offs_d
    q = tl.load(Q + off_q, mask=mask_d, other=0.0)

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = -float("inf")
    e_sum = 0.0
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,
                mask=offs_n < split_kv_end,
                other=0,
            )
            offs_buf_k = (
                kv_loc[:, None] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[None, :]
            )
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[:, None] < split_kv_end) & (mask_d[None, :]),
                other=0.0,
            )
            qk = tl.sum(q[None, :] * k, 1)
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(offs_n < split_kv_end, qk, float("-inf"))

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            n_e_max = tl.maximum(tl.max(qk, 0), e_max)
            re_scale = tl.exp(e_max - n_e_max)
            p = tl.exp(qk - n_e_max)
            acc *= re_scale
            acc += tl.sum(p[:, None] * v, 0)

            e_sum = e_sum * re_scale + tl.sum(p, 0)
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + offs_dv
        )

        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum,
            mask=(mask_dv),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )

        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
        )


def _decode_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 64
    # [TODO] work around SGPR limit on MI3xx
    if is_hip_:
        BLOCK = 8
    NUM_KV_SPLITS = num_kv_splits
    Lk = k_buffer.shape[-1]
    Lv = v_buffer.shape[-1]

    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]

    grid = (batch, head_num, NUM_KV_SPLITS)
    kv_group_num = q.shape[1] // k_buffer.shape[1]

    if kv_group_num == 1:
        num_warps = 4
    else:
        num_warps = 2
        if is_hip_:
            num_warps = 1

    BLOCK_DMODEL = triton.next_power_of_2(Lk)
    BLOCK_DV = triton.next_power_of_2(Lv)

    _fwd_kernel_stage1[grid](
        q,
        k_buffer,
        v_buffer,
        sm_scale,
        kv_indptr,
        kv_indices,
        att_out,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        att_out.stride(0),
        att_out.stride(1),
        att_out.stride(2),
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=num_warps,
        num_stages=2,
        Lk=Lk,
        Lv=Lv,
    )


@triton.jit
def _fwd_grouped_kernel_stage1(
    Q,
    K_Buffer,
    V_Buffer,
    sm_scale, 
    kv_indptr, 
    kv_indices,
    Att_Out, # [bs, H_q, num_chunks, chunk_size, D]
    stride_qbs, # H_q * seqlen * D
    stride_qh,  # seqlen * D 
    stride_buf_kbs, # H_kv * kv_seqlen * D
    stride_buf_kh,  # kv_seqlen * D 
    stride_buf_vbs,
    stride_buf_vh,
    stride_mid_ob, # stride along bs dim = H_q * seqlen * D
    stride_mid_oh, # stride along H_q(num of heads) dim = seqlen * D
    stride_mid_os, # stride along chunk_idx = chunk_size * D
    kv_group_num: tl.constexpr,
    q_head_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head_dim per tl.Block
    BLOCK_DPE: tl.constexpr,   # PEncoding dim per tl.Block
    BLOCK_DV: tl.constexpr,  # Value dim per tl.Block
    BLOCK_N: tl.constexpr,   # TODO: num of tokens per split/chunk per tl.Block ?
    BLOCK_H: tl.constexpr,   # num_heads per tl.Block 
    NUM_KV_SPLITS: tl.constexpr, 
    logit_cap: tl.constexpr,
    Lk: tl.constexpr,  # k head_dim
    Lv: tl.constexpr,  # v head_dim 
):
    # 获取当前线程块的任务信息
    cur_batch = tl.program_id(0)  # pi grid[0]: per tl.block in x dim handle 1 batch
    cur_head_id = tl.program_id(1) # pi grid[1]: per tl.block in y dim handle min(BLOCK_H, kv_group_name), 当前 tl.block 要处理 第几个 group_heads 的分组
    cur_kv_head = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)  # per tl.Block 最少处理 BLOCK_H 个heads, 如果 kv_group_num>16，那么 kv_head 组数更少，需要进一步细分 
    split_kv_id = tl.program_id(2) # pi grid[2]: per tl.block in z dim handle one chunk of kv_cache, if splited_kv_cache 

    if BLOCK_H < kv_group_num:
        VALID_BLOCK_H: tl.constexpr = BLOCK_H 
    else:
        VALID_BLOCK_H: tl.constexpr = kv_group_num  # valid_blockH is limited/bounded to kv_group_num 
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    """
        cur_head = cur_head_id * VALID_BLOCK_H + [0, 1, 2, .. BLOCK_H-1] 
        对于 kv_group_num < BLOCK_H 的case， 即 cur_head 的 index range 会超过  VALID_BLOCK_H(kv_group_num), 此时需要加如下mask
        举例 kv_group_num = 8 ，
        mask_h = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    mask_h = cur_head < (cur_head_id + 1) * VALID_BLOCK_H # mask to prevent out-of-bounds access
    mask_h = mask_h & (cur_head < q_head_num) # further mask to ensure not exceed q_head_num 

    # 
    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_dv = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lk
    mask_dv = offs_dv < Lv

    cur_batch_kv_start_idx = tl.load(kv_indptr + cur_batch)  # kv_indptr[cur_batch]
    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - cur_batch_kv_start_idx # kv_indptr[cur_batch+1] - kv_indptr[cur_batch]
    """
    kv_indptr 应该是 prefix sum style 数组。
    kv_indptr[i] 表示 第 ith batch 的starting index, 而 ith batch 的 seq_len = kv_indptr[i+1] - kv_indptr[i] ==> 所以 len(kv_indptr)=bs+1
    """
    offs_q = cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_d[None, :] 
    # offs_q = cur_batch * stride_qbs + cur_head[BLOCK_H, 1] * stride_qh + offs_d[1, BLOCK_DMODEL] 
    # 这里拿到是 2D offset tensor, 分别在 H_q 和 head_dim 维度上做 offset 。且考虑 tensor shape broadcasting
    q = tl.load(Q + offs_q, mask=(mask_h[:, None]) & (mask_d[None, :]), other=0.0)  # 当前 pi.Block 要处理的 q tensor
    """
        Triton 里面通过 预先定义的index offset tensor 取 instance 的 tensor。
        对于当前 pi.Block 要取的 q tensor, 其shape 与 offs_q 一致: [BLOCK_H, BLOCK_DMODEL] 
    """
    if BLOCK_DPE > 0:
        offs_dpe = BLOCK_DMODEL + tl.arange(0, BLOCK_DPE)
        mask_dpe = offs_dpe < Lk
        off_qpe = (
            cur_batch * stride_qbs + cur_head[:, None] * stride_qh + offs_dpe[None, :]
        )
        qpe = tl.load(
            Q + off_qpe, mask=(mask_h[:, None]) & (mask_dpe[None, :]), other=0.0
        ) # 当前 pi.Block 要处理的 pe subTensor

    kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)  # TODO: per batch 在 seq_len 维度 split 到 `NUM_KV_SPLITS` 个块
    split_kv_start = kv_len_per_split * split_kv_id
    split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_DV], dtype=tl.float32)  # shape [BLOCK_H, BLOCK_DV] 

    if split_kv_end > split_kv_start:
        for start_n in range(split_kv_start, split_kv_end, BLOCK_N):  # BLOCK_N as num of tokens per split
            offs_n = start_n + tl.arange(0, BLOCK_N)  # 第 start_n 个 chunk ，其中含 BLOCK_N 个tokens 的 kvcache
            kv_loc = tl.load(
                kv_indices + cur_batch_kv_start_idx + offs_n,  # kv_indices[cur_batch_kv_start_idx + offs_n] 
                mask=offs_n < split_kv_end,
                other=0,
            ) # load 当前 pi.Block 要处理的 chunked kv_cache
            """
                * k/v_buffer [bs, H_kv, seq_kv_len, D] , further splits seq_kv_len into chunks [bs, h_kv, num_chunks, chunk_size(BLOCK_N), D] 
                * kv_loc : 当前 chunk 上 各个 token 对应的 kvcache 的首地址 
                * kv_indices: 所有chunk上 kv_locs
            """
            offs_buf_k = (
                kv_loc[None, :] * stride_buf_kbs
                + cur_kv_head * stride_buf_kh
                + offs_d[:, None] 
            ) 
            """
                offs_buf_k = kv_loc[1, BLOCK_N] * stride_buf_kbs + cur_kv_head * stride_buf_kh + offs_d[BLOCK_DMODEL, 1]
                对于当前 pi.Block 要取的 buf_k tensor, 其shape 与 offs_buf_k 一致: [BLOCK_DMODEL, BLOCK_N] 
            """
            k = tl.load(
                K_Buffer + offs_buf_k,
                mask=(offs_n[None, :] < split_kv_end) & (mask_d[:, None]),
                other=0.0,
            )
            qk = tl.dot(q, k.to(q.dtype)) #  qk[BLOCK_H, BLOCK_N] = q[BLOCK_H, BLOCK_DMODEL]  *  k[BLOCK_DMODEL, BLOCK_N] 
            if BLOCK_DPE > 0:
                offs_buf_kpe = (
                    kv_loc[None, :] * stride_buf_kbs
                    + cur_kv_head * stride_buf_kh
                    + offs_dpe[:, None]
                )
                kpe = tl.load(
                    K_Buffer + offs_buf_kpe,
                    mask=(offs_n[None, :] < split_kv_end) & (mask_dpe[:, None]),
                    other=0.0,
                )
                qk += tl.dot(qpe, kpe.to(qpe.dtype))
            qk *= sm_scale

            if logit_cap > 0:
                qk = logit_cap * tanh(qk / logit_cap)

            qk = tl.where(
                mask_h[:, None] & (offs_n[None, :] < split_kv_end), qk, float("-inf")
            )

            offs_buf_v = (
                kv_loc[:, None] * stride_buf_vbs
                + cur_kv_head * stride_buf_vh
                + offs_dv[None, :]
            )
            """
                offs_buf_v =  kv_loc[BLOCK_N, 1]*stride_buf_vbs + cur_kv_head * stride_buf_vh + offs_dv[1, BLOCK_DV] 
                对于当前 pi.Block 要取的 buf_v tensor, 其shape 与 offs_buf_v 一致: [BLOCK_N, BLOCK_DV]
            """
            v = tl.load(
                V_Buffer + offs_buf_v,
                mask=(offs_n[:, None] < split_kv_end) & (mask_dv[None, :]),
                other=0.0,
            )

            """ safe softmax 计算
                score = exp(qk - max(qk)) / sum(exp(qk - max(qk))) 
            """
            n_e_max = tl.maximum(tl.max(qk, 1), e_max)  
            re_scale = tl.exp(e_max - n_e_max) # TODO: rescale for stability 
            p = tl.exp(qk - n_e_max[:, None]) # 分子
            acc *= re_scale[:, None]
            acc += tl.dot(p.to(v.dtype), v) # acc[BLOCK_H, BLOCK_DV] = qk[BLOCK_H, BLOCK_N] * v [BLOCK_N, BLOCK_DV]

            e_sum = e_sum * re_scale + tl.sum(p, 1) # 分母
            e_max = n_e_max

        offs_mid_o = (
            cur_batch * stride_mid_ob
            + cur_head[:, None] * stride_mid_oh
            + split_kv_id * stride_mid_os  # kvcache_chunk_idx * chunk_size * D 
            + offs_dv[None, :] # BLOCK_DV is power(2) most close to Lv
        )
        """ 将 acc tensor 写入 att_out 
            offs_mid_o = .. + cur_head[BLOCK_H, 1]* stride_mid_oh + split_kv_id * stride_mid_os + offs_dv[1, BLOCK_DV] ; [BLOCK_H, BLOCK_DV]
            对于当前 pi.Block 要写的 attn_out tensor , 其shape 与 offs_mid_o 一致: [BLOCK_H, BLOCK_DV]
            store at att_out[:, :, :, 0:Lv]
        """
        tl.store(
            Att_Out + offs_mid_o,
            acc / e_sum[:, None],
            mask=(mask_h[:, None]) & (mask_dv[None, :]),
        )

        offs_mid_o_1 = (
            cur_batch * stride_mid_ob
            + cur_head * stride_mid_oh
            + split_kv_id * stride_mid_os
            + Lv
        )
        """ 注意， offs_mid_o (block-base memory layout)，offs_mid_o_1 是 linear memory layout
            store at attn_out[:, :, :, Lv] # 在 Attn_out per head_dim 的最后一个元素位置 存了 e_max 
        """
        tl.store(
            Att_Out + offs_mid_o_1,
            e_max + tl.log(e_sum),
            mask=mask_h,
        )


def _decode_grouped_att_m_fwd(
    q,
    k_buffer,
    v_buffer,
    att_out,
    kv_indptr,
    kv_indices,
    num_kv_splits,
    sm_scale,
    logit_cap,
):
    BLOCK = 32
    Lk = k_buffer.shape[-1] # k_buffer shape [bs, n_kv_heads, kv_seq_len, head_dim(D)] 
    Lv = v_buffer.shape[-1] # v_buffer shape [bs, n_kv_heads, kv_seq_len, head_dim(D)]  

    # [TODO] work around shmem limit on MI3xx
    if is_hip_ and Lk >= 576:
        BLOCK = 16

    if Lk == 576:
        BLOCK_DMODEL = 512   # model dimension (head_dim) Block Size, each tl.BLOCK 处理的 head_dim 
        BLOCK_DPE = 64 # Positional Encoding Block Size
    elif Lk == 288:
        BLOCK_DMODEL = 256
        BLOCK_DPE = 32
    else:
        BLOCK_DMODEL = triton.next_power_of_2(Lk)
        BLOCK_DPE = 0
    BLOCK_DV = triton.next_power_of_2(Lv) # Value Block Size, 每个 tl.BLOCK 中处理的 v size

    batch, head_num = kv_indptr.shape[0] - 1, q.shape[1]  # TODO: kv_indptr [bs+1, ] 
    kv_group_num = q.shape[1] // k_buffer.shape[1]  # kv_group_num 个 Q head 共享一组 kv head 

    BLOCK_H = 16  # num_Heads per tl.BLOCK，
    NUM_KV_SPLITS = num_kv_splits
    grid = (
        batch, # grid[0] 是 x 方向 tl.BLOCK 数， per tl.BLOCK per query bs
        triton.cdiv(head_num, min(BLOCK_H, kv_group_num)), # grid[1] 是 y 方向 tl.BLOCK 数，per tl.BLOCK 处理 min(BLOCK_H, kv_group_num) 个head. 
        NUM_KV_SPLITS, # grid[2] 是 z方向 tl.BLOCK 数， per tl.BLOCK 处理一个 kvcache chunk
    )

    extra_kargs = {}
    num_stages = 2
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 1, "matrix_instr_nonkdim": 16, "kpack": 2}
        num_stages = 1

    _fwd_grouped_kernel_stage1[grid](   # Block 上 attn 计算
        q,        # [bs, H_q, seq_len, D]
        k_buffer, # [bs, H_kv, seq_kv_len, D]
        v_buffer, 
        sm_scale, 
        kv_indptr, 
        kv_indices, 
        att_out, # [bs, H_q, seq_len, D] => more like  [bs, H_q, num_chunks, chunk_size(BLOCK_N), head_dim]
        q.stride(0), # H_q * seq_len * D
        q.stride(1), # seq_len * D 
        k_buffer.stride(0), # H_kv * kv_seq_len * D 
        k_buffer.stride(1), # kv_seq_len * D 
        v_buffer.stride(0), # H_kv * kv_seq_len * D 
        v_buffer.stride(1), # kv_seq_len * D 
        att_out.stride(0), # H_q * seq_len * D
        att_out.stride(1), # seq_len * D
        att_out.stride(2), # chunk_size * D 
        kv_group_num=kv_group_num,
        q_head_num=head_num,
        BLOCK_DMODEL=BLOCK_DMODEL,
        BLOCK_DPE=BLOCK_DPE,
        BLOCK_DV=BLOCK_DV,
        BLOCK_N=BLOCK,
        BLOCK_H=BLOCK_H,
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        logit_cap=logit_cap,
        num_warps=4,
        num_stages=num_stages,
        Lk=Lk,
        Lv=Lv,
        **extra_kargs,
    )


@triton.jit
def _fwd_kernel_stage2(
    Mid_O, # [bs, num_heads, num_chunks, chunk_size, kv_seqlen]
    O, # [bs, num_heads, seqlen, D]
    kv_indptr, 
    stride_mid_ob, # bs stride in mid_o:  num_heads * seqlen * D
    stride_mid_oh, # nheads stride in mid_o:  seqlen * D 
    stride_mid_os, # stride along chunk_idx in mid_o = chunk_size * D
    stride_obs, # bs stride in o: num_heads* seqlen * D
    stride_oh, # nheads stride in o: seqlen * D 
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(
        kv_indptr + cur_batch
    ) # kv_indptr[cur_batch+1] - kv_indptr[cur_batch] 

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    """
        TODO: looks softmax do twice, in both stage1 & stage2 ??
    """

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        O + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )


def _decode_softmax_reducev_fwd(
    logits, # [bs, num_heads, seqlen, kv_seqlen]
    q, # [bs, num_heads, seqlen, head_dim(D)]
    o, # [bs, num_heads, seqlen, D]
    v_buffer, # [bs, num_kv_heads, kv_seqlen, D]
    kv_indptr,
    num_kv_splits,
):
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    extra_kargs = {}
    if is_hip_:
        # https://rocm.docs.amd.com/en/docs-6.2.0/how-to/llm-fine-tuning-optimization/optimizing-triton-kernel.html
        # https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
        extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    grid = (batch, head_num) # pi.block along grid[0]-x 处理 batch; pi.block along grid[1]-y 处理 head_num   
    _fwd_kernel_stage2[grid]( # for softmax + reduce 
        logits,
        o,
        kv_indptr,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        o.stride(0),
        o.stride(1),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def decode_attention_fwd_normal(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    _decode_att_m_fwd(
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, kv_indptr, num_kv_splits)


def decode_attention_fwd_grouped(
    q,  # [bs, num_heads, seq_len, head_dim(D)]
    k_buffer,  # [bs, num_kv_heads, kv_seq_len, head_dim(D)]
    v_buffer,  # same as k_buffer
    o, # [bs, num_heads, seq_len, head_dim]
    kv_indptr,  # [bs + 1] , array in prefix sum style
    kv_indices, # 所有chunk上 kvcache slots 
    attn_logits, # [bs, num_heads, seq_len, kv_seq_len]
    num_kv_splits, # kv-cache 的 chunks 数目
    sm_scale, 
    logit_cap=0.0, # clamps the attn logits to avoid extrem values before applying softmax 
):
    _decode_grouped_att_m_fwd(   # 计算每个Block 上的 attn
        q,
        k_buffer,
        v_buffer,
        attn_logits,
        kv_indptr,
        kv_indices,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )
    _decode_softmax_reducev_fwd(attn_logits, q, o, v_buffer, kv_indptr, num_kv_splits) # 上述 attn_logits 的 softmax + reduce


def decode_attention_fwd(
    q,
    k_buffer,
    v_buffer,
    o,
    kv_indptr,
    kv_indices,
    attn_logits,
    num_kv_splits,
    sm_scale,
    logit_cap=0.0,
):
    assert num_kv_splits == attn_logits.shape[2]
    kv_group_num = q.shape[1] // v_buffer.shape[1]

    if kv_group_num == 1:
        # MHA
        decode_attention_fwd_normal(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )
    else:
        # GQA/MQA/MLA
        decode_attention_fwd_grouped(
            q,
            k_buffer,
            v_buffer,
            o,
            kv_indptr,
            kv_indices,
            attn_logits,
            num_kv_splits,
            sm_scale,
            logit_cap,
        )


