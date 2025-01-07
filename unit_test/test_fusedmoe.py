import pytest 
import torch 
import triton 
import triton.language as tl 
from kernels import fused_moe_kernel 
from utils import DEVICE 

BLOCK_SIZE_M=16

@pytest.mark.parametrize("TOKENS, EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE, TOP_K ", [(32, 16, 128, 128, 2)])
def test_op(tokens, experts, intermediate_size, hidden_size, topk, dtype=torch.float16, use_fp8_w8a8: bool=False, use_int8_w8a16: bool=False, ):
    torch.manual_seed(20)
    # test as upper proj
    # 0. dummy weights for A, B, C 
    A = (torch.empty((tokens, hidden_size), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))   # input/activation
    B = (torch.empty((experts, 2*intermediate_size, hidden_size), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)) # weights  
    C = (torch.empty((tokens, experts, 2*intermediate_size), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    # 2. dummy topk weights and expert idx per token 
    # topk_weights[tokens, topk] as topk route_logits per token; topk_ids[tokens, topk] as topk expert idx per token 
    router_logits = torch.randn((tokens, experts), dtype=dtype, device=DEVICE)
    routing_probs = torch.softmax(router_logits, dim=-1)
    topk_weights, topk_ids = torch.topk(routing_probs, k=topk, dim=-1)
    # 3. dummy sorted_token_ids for token alignment and expert_ids for grouping <== sort tokens based on their assigned experts
    expert_ids = topk_ids.flatten()  #  Flatten topk_ids to group tokens by experts [tokens * top_k]
    sorted_token_ids = (torch.arange(tokens, device=DEVICE).unsqueeze(-1).expand_as(topk_ids)).flatten()
    _, sorted_indices = torch.sort(expert_ids)
    sorted_token_ids = sorted_token_ids[sorted_indices] # [tokens * topk]
    expert_ids = expert_ids[sorted_indices]  # [tokens * topk]
    # 4. total num of tokens per expert, including padding for alignment 
    tokens_per_expert = torch.bincount(expert_ids, minlength=experts)
    num_tokens_post_padded = torch.ceil(tokens_per_expert.float() / BLOCK_SIZE_M).int() * BLOCK_SIZE_M  # [experts]

    A_scale=None
    B_scale=None 

    if use_fp8_w8a8 or use_int8_w8a16 :
        A_scale = torch.empty((tokens, ), dtype=dtype, deivce=DEVICE)
        B_scale = torch.empty((experts, ), dtype=dtype, device=DEVICE)
    else:
        assert A_scale is None
        assert B_scale is None

    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16
    mul_routed_weight=True 

    # configs = triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, )
    configs = {'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}
    grid = lambda META: (triton.cdiv(sorted_token_ids.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(B.shape[1], META['BLOCK_SIZE_N']), )
    print(f"{grid}")

    fused_moe_kernel[grid](A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
                    B.shape[1],
                    B.shape[2],
                    sorted_token_ids.shape[0],
                    topk_ids.numel(),
                    A.stride(0),
                    A.stride(1),
                    B.stride(0),
                    B.stride(2),
                    B.stride(1),
                    C.stride(1),
                    C.stride(2),
                    B_scale.stride(0) if B_scale is not None and use_int8_w8a16 else 0,
                    B_scale.stride(1) if B_scale is not None and use_int8_w8a16 else 0,
                    MUL_ROUTED_WEIGHT=mul_routed_weight,
                    top_k=topk,
                    compute_type=compute_type,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    **configs,)


if __name__=="__main__":
    test_op(32, 16, 128, 128, 2)
    print(f"test done")
    #TODO: accuracy verify 



