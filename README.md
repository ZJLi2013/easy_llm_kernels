# llm_triton_kernels

workable triton kernels for llm inference

* this is a pure-triton kernel lib for popular llm models inference, e.g. deepseek, qwen, llama e.t.c 

* kernels including: 
    1. matmul
    2. moe 
    3. attention and its variants 
    4. element-wise kernels 

* dtype support: bf16, fp16, int8, and fp8 
* runtime platform: nvidia gpu, amd gpu 



## test bench 

* on H20

```sh
# 0. launch image 
docker run --gpus all --rm -it -v /home/david/kernels/llm_triton_kernels:/workspace/llm_triton_kernels -w /workspace  nvcr.io/nvidia/pytorch:24.11-py3

cd /workspace/llm_triton_kernels 

export PYTHONPATH=$(pwd):$PYTHONPATH

# 1. unit test 
python3 unit_test/test_matmul.py 

# 2. bench perf 
python3 benchmarks/bench_matmul.py 



```






















* comments 
    1. the ultimate goal: to build a triton-based llm inference engine (for practice) and with kernel-level test framework 
    2. Deadline: Spring Festival of 2025 







