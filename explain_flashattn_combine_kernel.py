#!/usr/bin/env python3
"""
FlashAttention Combine Kernel Explanation
==========================================

Explains what flash::FlashAttnFwdCombine does and why it's needed.
"""

print("""
================================================================================
FlashAttention Combine Kernel (flash::FlashAttnFwdCombine)
================================================================================

WHAT IT DOES
============

The combine kernel merges partial attention results from SPLIT-K parallelization.

Split-K is an optimization where the attention computation over the K/V sequence
dimension is split across multiple thread blocks (TBs) that run in parallel,
then their results are combined.


WHY IT'S NEEDED
===============

Problem: For decode with long sequences (e.g., 2048 tokens), a single thread
block may not fully utilize the GPU.

Solution: Split the KV sequence into chunks and process them in parallel:

    Query (1 token)  ──┐
                       ├──> Process in parallel
    KV Cache:          │
    ┌─────────────────────────────────────┐
    │ K1  K2  K3  ... K1024 ... K2048     │  Split into N chunks
    └─────────────────────────────────────┘
         ↓      ↓      ↓           ↓
       TB 0   TB 1   TB 2   ...  TB N-1    (Thread Blocks run in parallel)
         ↓      ↓      ↓           ↓
      Part 0 Part 1 Part 2 ... Part N-1   (Partial attention results)
         ↓      ↓      ↓           ↓
         └──────┴──────┴───────────┘
                    ↓
            COMBINE KERNEL              ← flash::FlashAttnFwdCombine
                    ↓
            Final Attention Output


HOW IT WORKS MATHEMATICALLY
============================

Attention formula:
    output = softmax(Q @ K^T / sqrt(d)) @ V

When split into N parts:
    output = Σ(softmax_i(Q @ K_i^T / sqrt(d)) @ V_i)  for i in [0, N-1]

BUT: We can't just sum softmax results! We need to renormalize.

The combine kernel implements the numerically stable merge algorithm:

1. Each partial result produces:
   - partial_output_i: attention output for chunk i
   - partial_lse_i: log-sum-exp of attention scores for chunk i

2. Combine using the following algorithm (from the reference implementation):

   max_lse = max(lse_0, lse_1, ..., lse_{N-1})

   For each partial result i:
       exp_i = exp(lse_i - max_lse)

   total_sum = Σ exp_i

   final_output = Σ (partial_output_i × exp_i / total_sum)

This ensures numerically stable combination while maintaining the correct
attention distribution across all KV positions.


EXAMPLE
=======

Decode with 2048-token KV cache, split into 4 parts:

Thread Block 0: Processes K[0:512],   V[0:512]   → output_0, lse_0
Thread Block 1: Processes K[512:1024], V[512:1024] → output_1, lse_1
Thread Block 2: Processes K[1024:1536], V[1024:1536] → output_2, lse_2
Thread Block 3: Processes K[1536:2048], V[1536:2048] → output_3, lse_3

Combine Kernel:
    max_lse = max(lse_0, lse_1, lse_2, lse_3)
    weights = [exp(lse_i - max_lse) for i in 0..3]
    total = sum(weights)
    final = (output_0 × weights[0] +
             output_1 × weights[1] +
             output_2 × weights[2] +
             output_3 × weights[3]) / total


WHEN IS IT USED?
=================

1. **CUDA Graphs**: When using CUDA graphs, vLLM may set num_splits > 1
   to ensure consistent memory allocation patterns

2. **Long Sequences**: For very long KV caches where splitting improves
   parallelism and GPU utilization

3. **FlashAttention 3**: FA3 has built-in heuristics to decide when to
   use split-K based on sequence length and hardware

From vllm code (vllm/v1/attention/backends/flash_attn.py:386-390):
    # NOTE(woosuk): Setting num_splits > 1 may increase the memory
    # usage, because the intermediate buffers of size [num_splits,
    # num_heads, num_tokens, head_size] are allocated. Therefore,
    # we only set num_splits when using cuda graphs.


YOUR PROFILING OUTPUT
======================

void cutlass::device_kernel<flash::FlashAttnFwdCombine<...>>
    Self CUDA: 149.343us (2.04% of total)
    Calls: 32
    Average: 4.667us per call

This means:
- The combine kernel ran 32 times (once per layer)
- Each call took ~4.7μs
- Total time: ~149μs across all layers
- This is only 2% of total CUDA time (very efficient!)

For comparison:
- Main attention kernel: 1.037ms (14.16%)
- Combine kernel: 0.149ms (2.04%)
- Ratio: Combine is ~14% of main kernel time

The combine kernel is FAST because:
1. It only does weighted averaging (no Q @ K^T matmul)
2. The intermediate results are already in GPU memory
3. It's a simple element-wise operation parallelized across tokens/heads


MEMORY OVERHEAD
===============

Split-K requires intermediate buffers:
    Size = [num_splits, num_heads, num_tokens, head_size]

For your case (assuming num_splits=4):
    4 × 32 heads × batch_tokens × 128 (head_dim) × 2 bytes (bf16)

For batch of 1 token:
    4 × 32 × 1 × 128 × 2 = 32,768 bytes = 32 KB (negligible)

For batch of 32 tokens:
    4 × 32 × 32 × 128 × 2 = 1,048,576 bytes = 1 MB (still small)


PERFORMANCE IMPACT
==================

Trade-offs:

✅ PROS:
- Better GPU utilization for long sequences
- More parallelism (multiple TBs working simultaneously)
- Can improve throughput by 10-30% for long contexts

❌ CONS:
- Extra memory for intermediate buffers
- Overhead of combine kernel (~14% of attention time)
- Only beneficial when sequence length > threshold

Typically beneficial when:
- Sequence length > 1024 tokens (decode)
- Using CUDA graphs (for consistent memory layout)
- GPU has many SMs (e.g., A100 with 108 SMs)


CONFIGURATION
=============

Control via vLLM config:
    --attention-config.flash_attn_max_num_splits_for_cuda_graph=<N>

Default: 0 (use FA3's built-in heuristics)

Set to specific value (e.g., 4) to force split-K when using CUDA graphs.


SUMMARY
=======

flash::FlashAttnFwdCombine is the "merge" kernel that combines partial
attention results from split-K parallelization. It's:

1. A performance optimization for long-sequence decode
2. Uses log-sum-exp trick for numerical stability
3. Adds ~14% overhead relative to main attention kernel
4. Typically beneficial for sequences > 1024 tokens
5. Automatically used by FA3 when beneficial

In your profiling:
- 149μs total (2% of CUDA time)
- 4.7μs per call (very efficient)
- Indicates split-K is being used for your workload

================================================================================

Reference:
- FlashAttention paper: https://arxiv.org/abs/2205.14135
- FlashAttention-3 paper: https://arxiv.org/abs/2407.08608
- Split-K merge algorithm: Section 2.2 of https://arxiv.org/abs/2501.01005
================================================================================
""")
