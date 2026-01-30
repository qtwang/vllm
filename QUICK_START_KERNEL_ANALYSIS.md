# Quick Start: Kernel Analysis

**TL;DR:** Analyze which CUDA kernels are running and how much time they take.

## 3 Simple Steps

### 1. Generate Profiler Output
```bash
# Option A: Use existing profiler_out_0.txt file from your profiling
# (The file with operator names truncated that you mentioned)

# Option B: Generate new one
python profile_vllm_simple.py
```

### 2. Run Analyzer
```bash
python analyze_kernels.py profiler_out_0.txt
```

### 3. View Results
The analyzer will show:

```
========================================================
CUDA KERNEL ANALYSIS - CATEGORIZED BY TYPE
========================================================

Total CUDA Time: 7.263 ms
Total Kernels: 20

========================================================
1. FLASHATTENTION KERNELS
========================================================

Total FlashAttention Time: 1.188 ms (16.36% of total)

Kernel Name                              Time (ms)    Calls    %
flash::FlashAttnFwdSm90                  1.037        32       14.28
  └─> Main FlashAttention forward computation

flash::FlashAttnFwdCombine               0.149        32       2.06
  └─> Combines partial results from split-K attention

flash::prepare_varlen_num_blocks_kernel 0.002        1        0.03
  └─> Prepares metadata for variable-length sequences

FLASHATTENTION SUBTOTAL: 1.188 ms (16.36%)

========================================================
2. NON-FLASHATTENTION KERNELS
========================================================

Total Non-FlashAttention Time: 6.075 ms (83.64%)

MATRIX MULTIPLICATION (GEMM) KERNELS
Category Total: 5.361 ms (73.82%)

nvjet_tst_128x16_64x11_2x1_v_bz_TNT      2.563        32       35.29
  └─> NVIDIA optimized GEMM for QKV/MLP projections

nvjet_tst_64x16_64x16_2x1_v_bz_splitK_TNT 1.321       32       18.19
  └─> NVIDIA optimized GEMM (different tile size)
...

ELEMENT-WISE OPERATIONS
Category Total: 0.269 ms (3.70%)

triton_red_fused_*_rms_norm              0.081        32       1.11
  └─> Fused RMSNorm (reduction + normalization)
...

KV CACHE OPERATIONS
Category Total: 0.080 ms (1.10%)

vllm::reshape_and_cache_flash_kernel     0.080        32       1.10
  └─> Writes K,V tensors to paged KV cache
...

========================================================
SUMMARY
========================================================

Category                            Time (ms)    % Total    # Kernels
FlashAttention                      1.188        16.36      3
Matrix Multiplication (GEMM)        5.361        73.82      5
Element-wise Operations             0.269        3.70       4
Reduction Operations                0.157        2.16       2
Sampling and Sorting               0.131        1.80       2
KV Cache Operations                0.080        1.10       1
Activation Functions               0.060        0.82       1
Memory Operations                  0.018        0.24       2
--------------------------------------------------------
TOTAL                              7.263        100.00     20
```

## What You Learn

### FlashAttention Kernels (Group 1)
- **Which kernels:** All attention computation kernels
- **What they do:**
  - `FlashAttnFwdSm90`: Main attention (Q @ K^T, softmax, @ V)
  - `FlashAttnFwdCombine`: Merges split-K results (parallel optimization)
  - `prepare_varlen`: Sets up metadata for variable sequences
- **Typical %:** 15-20% of total time

### Non-FlashAttention Kernels (Group 2)

#### GEMM Kernels (Usually Largest)
- **What they do:** Matrix multiplications for:
  - QKV projections (linear layers before attention)
  - Output projections (linear layers after attention)
  - MLP layers (feed-forward networks)
- **Typical %:** 60-80% of total time

#### Normalization Kernels
- **What they do:** RMSNorm / LayerNorm
- **Typical %:** 3-5%

#### Activation Kernels
- **What they do:** SwiGLU, SiLU, GELU activations
- **Typical %:** 1-2%

#### KV Cache Kernels
- **What they do:** Write K, V to paged memory
- **Typical %:** 1-2%

#### Sampling Kernels
- **What they do:** Top-K, Top-P, multinomial sampling
- **Typical %:** 1-2%

#### Element-wise Kernels
- **What they do:** Fused operations (add, mul, etc.)
- **Typical %:** 3-5%

#### Memory/Reduction Kernels
- **What they do:** Data transfers, reductions
- **Typical %:** 1-3%

## Key Insights

### Your Data (Example)
```
FlashAttention:     1.188 ms (16.36%)   ← Attention computation
GEMM:               5.361 ms (73.82%)   ← Dominant (QKV/MLP projections)
Other:              0.714 ms (9.82%)    ← Everything else
```

**Interpretation:**
- ✅ GEMM dominates (73%) - **Normal and expected**
- ✅ FlashAttention is efficient (16%) - **Good**
- ✅ Overhead is low (10%) - **Well optimized**

### Performance Tips

**If FlashAttention > 30%:**
- Might have very long sequences
- Check if split-K is being used (Combine kernel present)

**If GEMM < 50%:**
- Not utilizing matrix multiplication optimizations
- Check precision (FP16/BF16 faster than FP32)

**If Memory Ops > 5%:**
- Too many data transfers
- Consider kernel fusion

## Files

| **File** | **Purpose** |
|----------|-------------|
| `analyze_kernels.py` | The analyzer script |
| `sample_profiler_out_0.txt` | Example input |
| `KERNEL_ANALYSIS_README.md` | Full documentation |
| `profiler_out_0.txt` | Your actual profiler output |

## See Also

- **Full docs:** `KERNEL_ANALYSIS_README.md`
- **FlashAttention details:** `explain_flashattn_combine_kernel.py`
- **General profiling:** `PROFILING_GUIDE.md`
