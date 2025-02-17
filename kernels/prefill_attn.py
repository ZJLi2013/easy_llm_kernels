# vllm prefix_prefill_attn:  https://github.com/vllm-project/vllm/blob/main/vllm/attention/ops/prefix_prefill.py
# sglang prefill_attn https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py

@triton.jit
def _fwd_kernel(
    Q,
    K,
    V,
    K_cache, # [num_blocks, num_kv_heads, head_size/x, block_size, x]
    V_cache, # [num_blocks, num_kv_heads, head_size, block_size]
    B_Loc, # block_table, TODO [B, num_blocks, num_heads, head_size, block_size] ，多了batch 维度 ??
    sm_scale,
    k_scale,
    v_scale,
    B_Start_Loc, # TODO: b_loc 的所有 起始指针 
    B_Seqlen,  # batch total seqLen
    B_Ctxlen,  # batch total ctxLen
    block_size, 
    x,
    Out, # [B, H_q, seqlen, qk_hd] --> [B, H_q, num_chunks, chun_size(BLOCK_N), qk_hd]
    stride_b_loc_b, #b_loc.stride[0], 
    stride_b_loc_s, # b_loc.stride[1]
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs, # num_kv_heads * head_size * block_size 
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs, #[bs, h, d, bl, x] :: [num_blocks, num_kv_heads, head_size/x, block_size, x]  注意这里的缩写字母定义跟理解有些不一致
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,
    IN_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,  # head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    start_m = tl.program_id(2)  # index along mblock dim

    cur_kv_head = cur_head // num_queries_per_kv

    cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
    cur_batch_query_len = cur_batch_seq_len - cur_batch_ctx_len

    # start position inside of the query
    # generally, N goes over kv, while M goes over query_len
    block_start_loc = BLOCK_M * start_m

    # initialize offsets
    # [N]; starts at 0
    offs_n = tl.arange(0, BLOCK_N)
    # [D]; starts at 0
    offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
    # [M]; starts at current position in query
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # [M,D]
    off_q = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
        cur_head * stride_qh + offs_d[None, :] * stride_qd)

    dim_mask = tl.where(
        tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1,
        0).to(tl.int1)  # [D]

    q = tl.load(Q + off_q,
                mask=dim_mask[None, :] &
                (offs_m[:, None] < cur_batch_query_len),
                other=0.0)  # [M,D]

    """
        off_q [BLOCK_M, BLOCK_DMODEL_PADDED]
        q same shape as off_q 
    """

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")  # [M]
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)  # [M]
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED],
                    dtype=tl.float32)  # [M,D]

    # compute query against context (no causal mask here)
    for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLCOK_N)
        # -- compute qk ----
        bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                        ((start_n + offs_n) // block_size) * stride_b_loc_s,
                        mask=(start_n + offs_n) < cur_batch_ctx_len,
                        other=0)  # [N]
        """
            bn [BLOCK_N]
            * k_cache [num_blocks, num_kv_heads, head_size/x, block_size, x]
            * v_cache [num_blocks, num_kv_heads, head_size, block_size]
            * TODO: 注意貌似 radix_decode_attn 中 k_cache : [ bs, H_kv, num_blocks, block_size, qk_head_dim], v_cache [bs, H_kv, num_blocks, block_size, v_head_dim]
            
            * 理解下 b_loc 的tensor shape  
                * b_loc(block_table) 考虑bs 一般shape为 [B, max_blocks_per_seq]，然后通过 seq_id, token_id 查询: block_id 及 block_offset:
                    * block_id = block_table[seq_id][token_id//block_size]
                    * block_offset = token_id % block_size
                * 不过这里 b_loc 还有 num_blocks 维度, 结合下文 off_k, off_v 访问，其shape大概是 [B, num_blocks, num_heads, head_size, block_size] 
                    * block_id = (start_n + offs_n)//block_size 
                    * bn = B_loc + seq_id * stride_b_loc_b + block_id * strid_b_loc_s # per bs, per block 下的 tensor 首地址
                * 与 radix_decode_attn 中 kvcache 的区别就是, num_blocks/block_size vs num_heads/head_size 内存排布的顺序
                    * 应该讲， num_blocks 排布在更外层，对于kvcache动态增长更显存友好。 
        """
        # [D,N]
        off_k = (bn[None, :] * stride_k_cache_bs +
                    cur_kv_head * stride_k_cache_h +
                    (offs_d[:, None] // x) * stride_k_cache_d +
                    ((start_n + offs_n[None, :]) % block_size) *
                    stride_k_cache_bl +
                    (offs_d[:, None] % x) * stride_k_cache_x)
        """
            k_cache  [num_blocks, num_kv_heads, head_size/x, block_size, x]
            stride_k_cache_bs= num_kv_heads * head_size * local_block_size ==> block_size 应该仍然是以 tokens 计数
                1. 注意，bs 这里是 num_blocks 维度，所以 bn 应该也是 B_Loc num_blocks 维度的切分 

            off_k [BLOCK_DMODEL_PADDED, BLOCK_N] # [D, N]
        """
        # [N,D]
        off_v = (
            bn[:, None] * stride_v_cache_bs +
            cur_kv_head * stride_v_cache_h +
            offs_d[None, :] * stride_v_cache_d +
            (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
        k_load = tl.load(K_cache + off_k,
                            mask=dim_mask[:, None] &
                            ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                            other=0.0)  # [D,N]
        """
            v_cache [num_blocks, num_kv_heads, head_size, block_size]
            off_v [BLOCK_N, BLOCK_DMODEL_PADDED] # [N, D]
            
            k_load same shape as off_k 
        """

        if k_load.dtype.is_fp8():
            k = (k_load.to(tl.float32) * k_scale).to(q.dtype)
        else:
            k = k_load

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)  # [M,N]
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)
        """
            qk [BLOCK_M, BLOCK_N] # [M, N]
            q [BLOCK_M, BLOCK_DMODEL_PADDED]
            k [BLOCK_DMODEL_PADDED, BLOCK_N]

        """
        qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                        float("-inf"))
        qk *= sm_scale
        if SLIDING_WINDOW > 0:
            # (cur_batch_ctx_len + offs_m[:, None]) are the positions of
            # Q entries in sequence
            # (start_n + offs_n[None, :]) are the positions of
            # KV entries in sequence
            # So the condition makes sure each entry in Q only attends
            # to KV entries not more than SLIDING_WINDOW away.
            #
            # We can't use -inf here, because the
            # sliding window may lead to the entire row being masked.
            # This then makes m_ij contain -inf, which causes NaNs in
            # exp().
            qk = tl.where((cur_batch_ctx_len + offs_m[:, None]) -
                            (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk,
                            -10000)

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)  # [M]
        p = tl.exp(qk - m_ij[:, None])  # [M,N]
        l_ij = tl.sum(p, 1)  # [M]
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)  # [M]
        alpha = tl.exp(m_i - m_i_new)  # [M]
        beta = tl.exp(m_ij - m_i_new)  # [M]
        l_i_new = alpha * l_i + beta * l_ij  # [M]

        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]   # same shape as qk [M, N]
        # update acc
        v_load = tl.load(V_cache + off_v,
                            mask=dim_mask[None, :] &
                            ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                            other=0.0)  # [N,D]
        """
            off_v [BLOCK_N, BLOCK_DMODEL_PADDED] # [N, D]
            v_load same shape as off_v 

        """
        if v_load.dtype.is_fp8():
            v = (v_load.to(tl.float32) * v_scale).to(q.dtype)
        else:
            v = v_load
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)
        # # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # 注意，context_attn 还是有 pre-cached 和 raw tensors 部分。前面从 kv_cache 里面读的是 pre-cached 
    off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                offs_d[:, None] * stride_kd)
    off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                offs_d[None, :] * stride_vd)
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    """
        off_k [BLOCK_DMODEL_PADDED,  BLOCK_N] -> same shape for k_ptrs
        off_v [BLOCK_N, BLOCK_DMODEL_PADDED] -> same shape for v_ptrs 
    """

    # block_mask is 0 when we're already past the current query length
    block_mask = tl.where(block_start_loc < cur_batch_query_len, 1, 0) 
    """
       block_start_loc 为张量,  cur_batch_query_len 为标量
       对于 block_start_loc 中元素小于 cur_batch_query_len 的元素 其mask=1 ; 否则 mask=0 
       
       整个contxt_attn 的计算 == cross_attn on cached_kv + self_att on input query based 
                             == [query, cached_k, cached_v ]  + [query, queryed_k(K), queryed_v(V)]
    """ 

    # compute query against itself (with causal mask)
    for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
        """
            start_m 沿M(seq_len) 方向，每个 tl.block 处理 BLOCK_M 个 tokens，每次处理 沿着 kv 方向的 BLOCK_N 个元素 
            tl.blocks 沿着 seqlen 方向排布，per cycle per tl.block 处理 [BLOCK_M, BLOCK_N] subsize, split kv 方向into xBLOCK_N
        """
        start_n = tl.multiple_of(start_n, BLOCK_N) #[BLOCK_N]
        # -- compute qk ----
        k = tl.load(k_ptrs +
                    (cur_batch_in_all_start_index + start_n) * stride_kbs,
                    mask=dim_mask[:, None] &
                    ((start_n + offs_n[None, :]) < cur_batch_query_len),
                    other=0.0)
        """
            per tl.block 处理 k_ptrs submatrix shape as [BLOCK_DMODEL_PADDED,  BLOCK_N] 
            此时q 跟 cross-attn 计算的 q 是同一个:    q [BLOCK_M, BLOCK_DMODEL_PADDED]

        """

        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk = tl.dot(q, k, acc=qk, input_precision=IN_PRECISION)   # qk [BLOCK_M, BLOCK_N]
        qk *= sm_scale
        # apply causal mask
        qk = tl.where( [:, None] >= (start_n + offs_n[None, :]), qk,
                        float("-inf"))
        if SLIDING_WINDOW > 0:
            qk = tl.where(
                offs_m[:, None] -
                (start_n + offs_n[None, :]) < SLIDING_WINDOW, qk, -10000)

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        v = tl.load(v_ptrs +
                    (cur_batch_in_all_start_index + start_n) * stride_vbs,
                    mask=dim_mask[None, :] &
                    ((start_n + offs_n[:, None]) < cur_batch_query_len),
                    other=0.0)
        """
            v_ptrs [BLOCK_N, BLOCK_DMODEL_PADDED] 
            p [BLOCK_M, BLOCK_N]
            acc [BLOCK_M, BLOCK_DMODEL_PADDED]
            off_o [BLOCK_M, BLOCK_DMODEL_PADDED]
        """
        p = p.to(v.dtype)

        acc = tl.dot(p, v, acc=acc, input_precision=IN_PRECISION)  # acc += p*v   
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # initialize pointers to output
    off_o = (
        (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
        cur_head * stride_oh + offs_d[None, :] * stride_od) # [bs, num_blocks, num_heads, head_size]
    out_ptrs = Out + off_o
    tl.store(out_ptrs,
                acc,
                mask=dim_mask[None, :] &
                (offs_m[:, None] < cur_batch_query_len))
    return


@torch.inference_mode()
def context_attention_fwd(q,
                        k,
                        v,
                        o,
                        kv_cache_dtype: str,
                        k_cache,
                        v_cache,
                        b_loc,  # TODO: kvcache loc tensor ?
                        b_start_loc, # TODO: start address of kvcache loc tensor ?
                        b_seq_len, 
                        b_ctx_len,
                        max_input_len,
                        k_scale: float = 1.0,
                        v_scale: float = 1.0,
                        alibi_slopes=None,
                        sliding_window=None):

    q_dtype_is_f32 = q.dtype is torch.float32
    # need to reduce num. blocks when using fp32
    # due to increased use of GPU shared memory
    # if q.dtype is torch.float32:
    BLOCK = BASE_BLOCK // 2 if q_dtype_is_f32 else BASE_BLOCK

    # Turing does have tensor core for float32 multiplication
    # use ieee as fallback for triton kernels work. There is also
    # warning on vllm/config.py to inform users this fallback
    # implementation
    IN_PRECISION = 'ieee' if IS_TURING and q_dtype_is_f32 else None

    # Conversion of FP8 Tensor from uint8 storage to
    # appropriate torch.dtype for interpretation by Triton
    if "fp8" in kv_cache_dtype:
        assert (k_cache.dtype == torch.uint8)
        assert (v_cache.dtype == torch.uint8)

        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            target_dtype = torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            target_dtype = torch.float8_e5m2
        else:
            raise ValueError("Unsupported FP8 dtype:", kv_cache_dtype)

        k_cache = k_cache.view(target_dtype)
        v_cache = v_cache.view(target_dtype)

    if (k_cache.dtype == torch.uint8
            or v_cache.dtype == torch.uint8 and kv_cache_dtype == "auto"):
        raise ValueError("kv_cache_dtype='auto' unsupported for\
            FP8 KV Cache prefill kernel")

    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    """
        q shape [B*S, Hq, qk_hd]  # Lq=q_hd
        k shape [Bkv * Skv, Hkv, qk_hd] # Lk=qk_hd
        v shape [Bkv * Skv, Hkv, v_hd] # Lv = v_hd
    """
    assert Lq == Lk and Lk == Lv
    # round up Lk to a power of 2 - this is required for Triton block size
    Lk_padded = triton.next_power_of_2(Lk)

    sm_scale = 1.0 / (Lq**0.5)
    batch, head = b_seq_len.shape[0], q.shape[1]
    num_queries_per_kv = q.shape[1] // k.shape[1]

    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))  # batch, head,
    """
        grid[0] along batch 
        grid[1] along num of heads 
        grid[2] along num of blocks, split input_len into subgroups with size M_BLOCK
    """

    # 0 means "disable"
    if sliding_window is None or sliding_window <= 0:
        sliding_window = 0

    if alibi_slopes is not None:
        _fwd_kernel_alibi[grid](
            q,
            k,
            v,
            k_cache,
            v_cache,
            b_loc,
            sm_scale,
            k_scale,
            v_scale,
            b_start_loc,
            b_seq_len,
            b_ctx_len,
            alibi_slopes,
            v_cache.shape[3],
            k_cache.shape[4],
            o,
            b_loc.stride(0),
            b_loc.stride(1),
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            o.stride(0),
            o.stride(1),
            o.stride(2),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            k_cache.stride(3),
            k_cache.stride(
                4
            ),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            v_cache.stride(
                3),  #[num_blocks, num_kv_heads, head_size, block_size]
            num_queries_per_kv=num_queries_per_kv,
            IN_PRECISION=IN_PRECISION,
            BLOCK_M=BLOCK,
            BLOCK_DMODEL=Lk,
            BLOCK_DMODEL_PADDED=Lk_padded,
            BLOCK_N=BLOCK,
            num_warps=NUM_WARPS,
            num_stages=1,
        )
        return

    _fwd_kernel[grid](
        q,
        k,
        v,
        k_cache,
        v_cache,
        b_loc,
        sm_scale,
        k_scale,
        v_scale,
        b_start_loc,
        b_seq_len,
        b_ctx_len,
        v_cache.shape[3],  # block_size
        k_cache.shape[4],  # x 
        o,
        b_loc.stride(0), 
        b_loc.stride(1),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(
            4),  #[num_blocks, num_kv_heads, head_size/x, block_size, x]
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(
            3),  #[num_blocks, num_kv_heads, head_size, block_size]
        num_queries_per_kv=num_queries_per_kv,
        IN_PRECISION=IN_PRECISION,
        BLOCK_M=BLOCK,
        BLOCK_DMODEL=Lk,  # qk_hs
        BLOCK_DMODEL_PADDED=Lk_padded, # qk_hs padded to a power of 2
        BLOCK_N=BLOCK,
        SLIDING_WINDOW=sliding_window,
        num_warps=NUM_WARPS,
        num_stages=1,
    )
    return


# zhihu:  https://zhuanlan.zhihu.com/p/695799736
@triton.jit
def _fwd_kernel_alibi(
    Q, # new tokens对应的query Tensor
    K, # new tokens对应的keys Tensor
    V, # new tokens对应的values Tensor
    K_cache,  # K prefix cache, 是已经准备好的Tensor
    V_cache,  # V prefix cache, 是已经准备好的Tensor
    B_Loc,    # block table
    sm_scale, # scale factor
    B_Start_Loc, # new tokens len的前缀和cumsum(query_lens)
    B_Seqlen,    # 实际的seq_len=new_tokens len + b_ctx_len
    B_Ctxlen,    # 命中了prefix cache的token数
    Alibi_slopes,
    block_size,  # block size，比如32
    x,           # 8, 对应k_cache = k_cache.view(-1, block_size, num_kv_heads, head_size // 8, 8).permute(0, 2, 3, 1, 4).contiguous()
    Out,         # attention输出
    stride_b_loc_b, # 各种tensor不同维度上的stride
    stride_b_loc_s,
    stride_qbs,
    stride_qh,
    stride_qd,
    stride_kbs,
    stride_kh,
    stride_kd,
    stride_vbs,
    stride_vh,
    stride_vd,
    stride_obs,
    stride_oh,
    stride_od,
    stride_k_cache_bs,
    stride_k_cache_h,
    stride_k_cache_d,
    stride_k_cache_bl,
    stride_k_cache_x,
    stride_v_cache_bs,
    stride_v_cache_h,
    stride_v_cache_d,
    stride_v_cache_bl,
    num_queries_per_kv: int,  # MQA/GQA 一个kv head对应query head数量
    BLOCK_M: tl.constexpr,    # triton常量 此处BLOCK_M=BLOCK=128，表示Q的seq_len方向的分块大小
    BLOCK_DMODEL: tl.constexpr,  # head size 表示实际的head size
    BLOCK_DMODEL_PADDED: tl.constexpr,  # head size padded to a power of 2
    BLOCK_N: tl.constexpr,   # triton常量 此处BLOCK_N=BLOCK=128，表示KV的seq_len方向的分块大小
):
        # attn_bias[]
        cur_batch = tl.program_id(0) # batch中的不同query放在不同的block进行组织
        cur_head = tl.program_id(1) # 当前query中的不同head也在不同的block进行组织
        # BLOCK_M: 表示Q的seq_len行方向并行，每个block处理BLOCK_M个tokens
        # BLOCK_N: 表示K的seq_len列方向并行，每个block处理BLOCK_N个tokens
        # max_input_len/BLOCK: 表示最多需要多少个block，max_input_len由用户指定，可以是256/512/1024/2048/...等
        # BLOCK_M=BLOCK_M=BLOCK=128
        # start_m: 当前属于哪一个block，其实就是grid最内层的block id.
        start_m = tl.program_id(2) 
        # MQA/GQA处理：比如num_queries_per_kv=8，表示GQA，8个query head共享一个kv head，那么在kernel
        # 计算中实际使用到的kv head，则需要用当前query head的索引，除以8，比如query head的索引为14，则对应
        # 需要使用的kv head的索引为14//8=1（索引从0开始）
        cur_kv_head = cur_head // num_queries_per_kv 

        # cur_batch_seq_len: the length of prompts
        # cur_batch_ctx_len: the length of prefix
        # cur_batch_in_all_start_index: the start id of the dim=0
        cur_batch_ctx_len = tl.load(B_Ctxlen + cur_batch)
        cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)
        # 拿到当前样本在batch中对应的start location，因为输入不是pad的，而是拼接在一起的
        # 需要记录当前的样本在batch中真正的start location
        # B_Start_Loc: 前缀和 torch.cumsum(torch.tensor([0] + query_lens[:-1]))
        cur_batch_in_all_start_index = tl.load(B_Start_Loc + cur_batch)
        
        # start position inside of the query
        # generally, N goes over kv, while M goes over query_len
        # block_start_loc: 当前block需要处理的Q的开始索引
        block_start_loc = BLOCK_M * start_m

        # initialize offsets
        # [N]; starts at 0 kv的分块偏移量
        offs_n = tl.arange(0, BLOCK_N)
        # [D]; starts at 0 表示head size，这里使用BLOCK_DMODEL_PADDED 为 2的次方
        offs_d = tl.arange(0, BLOCK_DMODEL_PADDED)
        # [M]; starts at current position in query
        # offs_m仅表示当前的query，一个batch有多个query
        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        # offs_m与off_d代表当前program要处理的块index，我们还需要将这个index转换成数据指针的位置
        # [M,D] 获取全局的数据指针
        off_q = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_qbs +
            cur_head * stride_qh + offs_d[None, :] * stride_qd)
        # offs_m[:, None]) [BLOCK_M, 1] offs_d[None, :] [1, D]
        # broadcast -> [M,D]? cur_head已经考虑在内
        
        # [D] mask处理，因为pad成2的次方了。而实际可能不是2的次方，比如80<128
        dim_mask = tl.where(
            tl.arange(0, BLOCK_DMODEL_PADDED) < BLOCK_DMODEL, 1, 0).to(tl.int1)
        
        # Q看做全局数据指针，off_q看做全局索引 Q:(num_tokens, heads, head_size)
        # dim_mask[None, :] [1,D] -> 对D做mask，并且broadcast为[M,D];
        # cur_batch_seq_len - cur_batch_ctx_len表示实际query_len，超过的不用考虑计算 
        # offs_m[:, None] [M, 1]
        # dim_mask[None, :] & offs_m[:, None] -> [1,D] & [M, 1]
        # >>> dim_mask = torch.tensor([1,1,1,1,0,0]) # [1,6]
        # >>> offs_m = torch.tensor([8,9,10,11]) # [4,1]
        # >>> mask=dim_mask[None, :] & (offs_m[:, None] < 12)
        # >>> mask
        # tensor([[1, 1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 0, 0],
        #         [1, 1, 1, 1, 0, 0]])
        # >>> mask.shape
        # torch.Size([4, 6]) # [M,D]
        q = tl.load(Q + off_q, # [M,D]
                    mask=dim_mask[None, :] &
                    (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len),
                    other=0.0)
        
        # # initialize pointer to m and l
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL_PADDED], dtype=tl.float32)
        
        # [1] 计算query与在prefix cache中的kv的attention 

        # 每个head一个alibi_slope值
        alibi_slope = tl.load(Alibi_slopes + cur_head) 
        alibi_start_q = tl.arange(
            0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = 0
        # split KV 处理逻辑 内循环迭代处理K的BLOCK_N大小分块
        for start_n in range(0, cur_batch_ctx_len, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            # B_Loc: 保存了prefix cache的block_table 获取当前query对应的block ids
            # offs_n: tl.arange(0, BLOCK_N)
            bn = tl.load(B_Loc + cur_batch * stride_b_loc_b +
                         ((start_n + offs_n) // block_size) * stride_b_loc_s,
                         mask=(start_n + offs_n) < cur_batch_ctx_len,
                         other=0)
            off_k = (bn[None, :] * stride_k_cache_bs +
                     cur_kv_head * stride_k_cache_h +
                     (offs_d[:, None] // x) * stride_k_cache_d +
                     ((start_n + offs_n[None, :]) % block_size) *
                     stride_k_cache_bl +
                     (offs_d[:, None] % x) * stride_k_cache_x)
            off_v = (
                bn[:, None] * stride_v_cache_bs +
                cur_kv_head * stride_v_cache_h +
                offs_d[None, :] * stride_v_cache_d +
                (start_n + offs_n[:, None]) % block_size * stride_v_cache_bl)
            
            # load prefix kv cache并对D和N做mask，形状为转置[D,N]
            # TODO: 补充注释说明
            # >>> dim_mask = torch.tensor([1,1,1,1,0,0]) D=6
            # >>> offs_n = torch.tensor([8,9,10,11]) N=4
            # >>> mask=dim_mask[:,None] & (offs_n[None,:] < 12)
            # >>> mask
            # tensor([[1, 1, 1, 1],
            #         [1, 1, 1, 1],
            #         [1, 1, 1, 1],
            #         [1, 1, 1, 1],
            #         [0, 0, 0, 0],
            #         [0, 0, 0, 0]])
            # >>> mask.shape
            # torch.Size([6, 4]) # [D,N]
            k = tl.load(K_cache + off_k,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) < cur_batch_ctx_len),
                        other=0.0)  # [D,N]

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k)
            qk = tl.where((start_n + offs_n[None, :]) < cur_batch_ctx_len, qk,
                          float("-inf"))
            qk *= sm_scale

            # load alibi 根据当前属于的D的维度计算head内偏移
            alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k -
                     alibi_start_q[:, None]) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi, float("-inf"))
            qk += alibi
            alibi_start_k += BLOCK_N

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc [M,D]
            v = tl.load(V_cache + off_v,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) < cur_batch_ctx_len),
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v, allow_tf32=False)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        off_k = (offs_n[None, :] * stride_kbs + cur_kv_head * stride_kh +
                 offs_d[:, None] * stride_kd)
        off_v = (offs_n[:, None] * stride_vbs + cur_kv_head * stride_vh +
                 offs_d[None, :] * stride_vd)
        k_ptrs = K + off_k
        v_ptrs = V + off_v

        block_mask = tl.where(
            block_start_loc < cur_batch_seq_len - cur_batch_ctx_len, 1, 0)

        # init alibi
        alibi_slope = tl.load(Alibi_slopes + cur_head)
        alibi_start_q = tl.arange(
            0, BLOCK_M) + block_start_loc + cur_batch_ctx_len
        alibi_start_k = cur_batch_ctx_len
        # # init debugger

        # [2] 计算query与不在prefix cache中的kv的attention 
        # offset_db_q = tl.arange(0, BLOCK_M) + block_start_loc
        # offset_db_k = tl.arange(0, BLOCK_N)
        # calc q[BLOCK_M, BLOCK_MODEL] mul k[prefix_len: , BLOCK_DMODEL]
        for start_n in range(0, block_mask * (start_m + 1) * BLOCK_M, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            # -- compute qk ----
            k = tl.load(k_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_kbs,
                        mask=dim_mask[:, None] &
                        ((start_n + offs_n[None, :]) <
                         cur_batch_seq_len - cur_batch_ctx_len),
                        other=0.0)

            qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            qk += tl.dot(q, k, allow_tf32=False)
            qk *= sm_scale
            qk = tl.where(offs_m[:, None] >= (start_n + offs_n[None, :]), qk,
                          float("-inf"))

            # load alibi
            alibi = (tl.arange(0, BLOCK_N)[None, :] + alibi_start_k -
                     alibi_start_q[:, None]) * alibi_slope
            alibi = tl.where(
                (alibi <= 0) & (alibi_start_q[:, None] < cur_batch_seq_len),
                alibi, float("-inf"))
            qk += alibi
            alibi_start_k += BLOCK_N

            # -- compute m_ij, p, l_ij
            m_ij = tl.max(qk, 1)
            m_i_new = tl.maximum(m_i, m_ij)
            p = tl.math.exp(qk - m_i_new[:, None])
            l_ij = tl.sum(p, 1)
            # -- update m_i and l_i

            alpha = tl.math.exp(m_i - m_i_new)
            l_i_new = alpha * l_i + l_ij
            # -- update output accumulator --
            # scale p
            # scale acc
            acc_scale = alpha
            # acc_scale = l_i / l_i_new * alpha
            acc = acc * acc_scale[:, None]
            # update acc
            v = tl.load(v_ptrs +
                        (cur_batch_in_all_start_index + start_n) * stride_vbs,
                        mask=dim_mask[None, :] &
                        ((start_n + offs_n[:, None]) <
                         cur_batch_seq_len - cur_batch_ctx_len),
                        other=0.0)

            p = p.to(v.dtype)
            acc += tl.dot(p, v, allow_tf32=False)
            # update m_i and l_i
            l_i = l_i_new
            m_i = m_i_new

        acc = acc / l_i[:, None]

        # initialize pointers to output
        off_o = (
            (cur_batch_in_all_start_index + offs_m[:, None]) * stride_obs +
            cur_head * stride_oh + offs_d[None, :] * stride_od)
        out_ptrs = Out + off_o
        tl.store(out_ptrs,
                 acc,
                 mask=dim_mask[None, :] &
                 (offs_m[:, None] < cur_batch_seq_len - cur_batch_ctx_len))
        return