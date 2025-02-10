# Overview

a collection of workable triton kernels(NV & AMD GPUs) for building fast llm inference:

1. MatMul normal, & matmul with per-channel/per-tensor/blockwise scaling for int8/fp8 
2. MoE with bf16, int8, fp8 
3. FlashAttn-v2, RadixDecodeAttn, RadixExtendAttn, PagedAttn, LinearAttn 



## test bench 

* on H20

```sh
# 0. launch image 
docker run --gpus all --rm -it -v /home/david/kernels/llm_triton_kernels:/workspace/llm_triton_kernels -w /workspace  nvcr.io/nvidia/pytorch:24.11-py3
cd /workspace/llm_triton_kernels 
export PYTHONPATH=$(pwd):$PYTHONPATH
# 1. unit test 
python3 unit_test/test_matmul.py 
python3 unit_test/test_fusedattn.py 

# 2. bench perf 
python3 benchmark/bench_matmul.py 
python3 benchmark/bench_fusedattn.py


```









