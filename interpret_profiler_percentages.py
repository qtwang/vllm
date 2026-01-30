#!/usr/bin/env python3
"""
Interpret PyTorch Profiler CUDA Percentages
Explains why Self CUDA % can exceed 100% and how to read the profiler output correctly.
"""

print("""
================================================================================
Understanding PyTorch Profiler CUDA Percentages
================================================================================

WHY SELF CUDA % CAN EXCEED 100%
================================

Self CUDA % = (Self CUDA Time / Total CUDA Time) × 100

Where:
- Self CUDA Time: Time spent in this operation's CUDA kernels
- Total CUDA Time: Sum of all CUDA kernel times OR wall-clock time

CUDA operations execute ASYNCHRONOUSLY and can OVERLAP:

Example Timeline:
-----------------
Wall-Clock:     |----------- 7.32ms ------------|

CUDA Stream 1:  |--Kernel A: 3.0ms--|
CUDA Stream 2:     |--Kernel B: 2.5ms--|
CUDA Stream 3:        |--Kernel C: 1.5ms--|
CUDA Stream 4:           |--Kernel D: 1.2ms--|
CUDA Stream 5:              |--Kernel E: 0.8ms--|

Sum of CUDA times: 3.0 + 2.5 + 1.5 + 1.2 + 0.8 = 9.02ms
Wall-clock time: 7.32ms

Self CUDA % = (9.02 / 7.32) × 100 = 123.24%

This indicates:
✓ Excellent GPU utilization (>100% = concurrent execution)
✓ Multiple kernels running in parallel
✓ Good performance (not serialized)


WHAT DIFFERENT PERCENTAGES MEAN
================================

1. Self CUDA % < 100%
   - Normal for individual CUDA kernels
   - Indicates sequential execution or partial GPU usage

2. Self CUDA % ≈ 100%
   - Kernel uses most of the profiled time
   - Likely a bottleneck operation

3. Self CUDA % > 100%
   - NORMAL for profiling scopes (Python/C++ annotations)
   - Indicates concurrent kernel execution
   - Higher = better parallelism


HOW TO READ YOUR OUTPUT
========================

Example:
--------
Name: gpu_model_runner: forward
Self CPU %:      0.00%
Self CUDA:       9.022ms
Self CUDA %:     123.24%  ← This is NORMAL!

Interpretation:
- This is a profiling scope (not a kernel)
- It launched 9.022ms of CUDA work
- Work completed in ~7.3ms wall-clock time
- GPU utilization: 123.24% (excellent!)
- Multiple operations ran concurrently


WHAT TO FOCUS ON
=================

For Performance Analysis:
1. Look at absolute times (Self CUDA column)
2. Identify top time-consuming kernels
3. Check for kernels that DON'T overlap (potential optimization)
4. High % on individual kernels = bottlenecks

For Profiling Scopes (like "gpu_model_runner: forward"):
- >100% = Good (concurrent execution)
- <100% = Might indicate serialization issues
- Exact percentage less important than absolute time


EXAMPLE BREAKDOWN
=================

gpu_model_runner: forward              9.022ms   123.24%  ← Scope
├─ flash_attn_fwd_kernel              3.145ms    42.97%  ← Kernel
├─ cutlass_gemm_sm80                  2.567ms    35.07%  ← Kernel
├─ rms_norm_kernel                    1.234ms    16.86%  ← Kernel
├─ rotary_embedding_kernel            0.876ms    11.97%  ← Kernel
└─ reshape_and_cache_flash            0.645ms     8.81%  ← Kernel

Notice:
- Scope shows 123% (normal for concurrent execution)
- Individual kernels each <100% (normal)
- Sum of kernel %s = 115.68% (indicates overlap)


KEY TAKEAWAY
============

Self CUDA % > 100% is:
✓ NORMAL for profiling scopes
✓ EXPECTED for concurrent GPU operations
✓ GOOD for performance (indicates parallelism)
✗ NOT an error or bug

Focus on:
- Absolute CUDA times (Self CUDA column)
- Top time-consuming operations
- Opportunities to overlap more operations

================================================================================
""")
