# Developmental Branch for Micro-Benchmarking Kernels for Neutrino Metric Verification

> Please wait for our code cleaning before pushing up to Github.

## Motivation
As Neutrino's programmability goes beyond existing system like Nsight, simple correspondence checking can not fulfill the need of metric correctness. 

Thus, we plan to add microbenchmarking for the verification.

## Plans
Currently, we plan to start with DMAT, the most complicated tool in Neutrino and then lowering to other tools if applicable.
We plan to first support several widely-studied memory access pattern:
- [ ] Linear: A copy-like kernel that each thread read part of the memory (in a for-loop)
- [ ] Strided: Similar to linear but now 
- [ ] Gather: A kernel that reads O(N) memory and save O(1) memory
- [ ] Scatter: A kernel that reads O(1) memory and save O(N) memory
- [ ] Random: A kernel that reads memories whose address is stored in an array. This array will be initialized randomly like PagedAttention. Notably we can not go fully random or the result will be hard to validate.

Moreover, as we have multiple threads, we must have controls in:
1. Launch waves: we carefully craft all kernels to have 1 launch waves, so the data size will varies for different GPUs, so there won't be imbalanced workload between SM Unit.
2. GPU Threads: we controls the actual threads used to be exactly at the multiple of hardware limit, so there won't be imbalanced workload inside SM Units.
Please note that these two are just for fast verification at early stage, we might adjust them if need for more comprehensive results.

## Acknowledgement
We acknowledge above plans are inspired by previous micro-benchmarkign tools
1. [gpu-benches](https://github.com/RRZE-HPC/gpu-benches)
2. [gpu-arch-microbenchmark](https://github.com/sjfeng1999/gpu-arch-microbenchmark)
3. [nvbench](https://github.com/NVIDIA/nvbench)