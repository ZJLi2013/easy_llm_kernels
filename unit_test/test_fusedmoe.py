import pytest 
import torch 
from kernels import fused_moe_kernel 
from utils import DEVICE 

@pytest.mark.parametrize("TOKENS, EXPERTS, HIDDEN_SIZE, TOP_K ", [(32, 16, 128, 2)])
def test_op(tokens, experts, hidden_size, topk, dtype=torch.float16, use_fp8_w8a8: bool=False, use_int8_w8a16: bool=False, ):
    torch.manual_seed(20)
    A = (torch.empty((tokens, hidden_size), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))   # activation 
    B = (torch.empty((experts, hidden_size, hidden_size), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)) # weight 
    C = (torch.empty((experts, hidden_size), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5))
    topk_weights = torch.empty((tokens, topk), dtype=dtype, device=DEVICE).normal_(mean=0.0, std=0.5)
    topk_ids = torch.randint(0, experts, (tokens, topk), dtype=torch.int32, device=DEVICE)
    sorted_token_ids = torch.argsort(torch.randint(0, tokens, (tokens,), device=DEVICE))
    expert_ids = torch.randint(0, experts, (tokens,), dtype=torch.int32, device=DEVICE)
    num_tokens_post_padded = torch.tensor([tokens], dtype=torch.int32, device=DEVICE)

    if use_fp8_w8a8 or use_int8_w8a16 :
        dtype = torch.float8_e4m3 
        A_scale = torch.empty((tokens, ), dtype=dtype, deivce=DEVICE)
        B_scale = torch.empty((experts, ), dtype=dtype, device=DEVICE)
    else:
        assert A_scale is None
        assert B_scale is None

    compute_type = dtype 
    mul_routed_weight=True 

    custom_config = {
    }
    
    fused_moe_kernel(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded,
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
                    **custom_config,)




