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




* how to run 

```sh
python -m benchmark.bench_matmul
python -m unit_test.test_matmul
``` 













* comments 
    1. the ultimate goal: to build a triton-based llm inference engine (for practice) and with kernel-level test framework 
    2. Deadline: Spring Festival of 2025 







