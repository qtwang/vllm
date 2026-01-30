# Kernel Analysis Tool

Analyzes vLLM profiler output and categorizes CUDA kernels by type with detailed explanations.

## Quick Start

```bash
# Run profiling first
python profile_vllm_simple.py

# Analyze the profiler output
python analyze_kernels.py profiler_traces/profiler_out_0.txt
```

## What It Does

The analyzer:
1. **Parses** profiler_out_*.txt files
2. **Filters** actual CUDA kernels (excludes Python/C++ wrappers)
3. **Categorizes** kernels into groups:
   - FlashAttention kernels
   - GEMM (matrix multiplication) kernels
   - Normalization kernels
   - Activation function kernels
   - KV cache operations
   - Sampling/sorting kernels
   - Element-wise operations
   - Memory operations
   - Reduction operations
4. **Explains** what each kernel does
5. **Summarizes** time breakdown by category

## Output Format

### 1. FlashAttention Kernels Section

Shows all FlashAttention-related kernels with:
- Kernel name
- Total runtime
- Number of calls
- Average time per call
- Percentage of total CUDA time
- Brief explanation of what the kernel does

Example:
```
FLASHATTENTION KERNELS
----------------------
Total FlashAttention Time: 1.188 ms (16.36% of total)

Kernel Name                                  Time (ms)    Calls    Avg (μs)    %
flash::FlashAttnFwdSm90                      1.037        32       32.4        14.28
  └─> Main FlashAttention forward computation (Ampere/Hopper optimized)

flash::FlashAttnFwdCombine                   0.149        32       4.7         2.06
  └─> Combines partial results from split-K attention

flash::prepare_varlen_num_blocks_kernel     0.002        1        2.0         0.03
  └─> Prepares metadata for variable-length sequences
```

### 2. Non-FlashAttention Kernels Section

Groups kernels by category, showing top 5 in each category:

```
MATRIX MULTIPLICATION (GEMM) KERNELS
Category Total: 5.361 ms (73.82% of total)

Kernel Name                                  Time (ms)    Calls    Avg (μs)    %
nvjet_tst_128x16_64x11_2x1_v_bz_TNT         2.563        32       80.1        35.29
  └─> NVIDIA optimized GEMM for QKV/MLP projections
...
```

### 3. Summary Table

Overall breakdown by category:

```
Category                                Time (ms)    % of Total    # Kernels
FlashAttention                          1.188        16.36         3
Matrix Multiplication (GEMM) Kernels    5.361        73.82         5
Element-wise Operations                 0.269        3.70          4
...
TOTAL                                   7.263        100.00        20
```

## Kernel Categories Explained

### FlashAttention Kernels
**Purpose:** Attention computation (Q @ K^T, softmax, @ V)

**Key kernels:**
- `flash::FlashAttnFwdSm90`: Main attention computation
- `flash::FlashAttnFwdCombine`: Merges split-K results
- `flash::prepare_varlen_num_blocks_kernel`: Prepares metadata

**Typical %:** 15-20% of total CUDA time

---

### GEMM Kernels
**Purpose:** Matrix multiplications for:
- QKV projections (hidden → Q, K, V)
- Output projections (attention output → hidden)
- MLP layers (gate_proj, up_proj, down_proj)

**Key kernels:**
- `nvjet_tst_*`: NVIDIA optimized GEMM
- `cutlass_*_gemm`: CUTLASS library GEMM
- `volta_*_gemm`, `ampere_*_gemm`: Architecture-specific GEMM

**Typical %:** 60-80% of total CUDA time (usually largest)

---

### Normalization Kernels
**Purpose:** Layer normalization (RMSNorm for Llama)

**Key kernels:**
- `triton_red_fused_*_rms_norm`: Fused RMSNorm
- `rms_norm_kernel`: Standard RMSNorm
- `fused_add_rms_norm`: Combined residual + norm

**Typical %:** 3-5% of total CUDA time

---

### Activation Function Kernels
**Purpose:** Non-linear activations in MLP

**Key kernels:**
- `triton_poi_fused_mul_silu`: SwiGLU activation
- `silu_kernel`: SiLU/Swish activation
- `gelu_kernel`: GELU activation

**Typical %:** 1-2% of total CUDA time

---

### KV Cache Operations
**Purpose:** Write new K, V tensors to paged KV cache

**Key kernels:**
- `vllm::reshape_and_cache_flash_kernel`: Main KV cache write

**Typical %:** 1-2% of total CUDA time

---

### Sampling/Sorting Kernels
**Purpose:** Sample next token from logits

**Key kernels:**
- `DeviceRadixSort`: Sort for top-k sampling
- `softmax`: Compute probabilities
- `multinomial`: Sample from distribution

**Typical %:** 1-2% of total CUDA time

---

### Element-wise Operations
**Purpose:** Element-wise computations (fused ops)

**Key kernels:**
- `triton_poi_fused_*`: Triton pointwise kernels
- `triton_red_fused_*`: Triton reduction kernels
- `vectorized_elementwise`: PyTorch element-wise

**Typical %:** 3-5% of total CUDA time

---

### Memory Operations
**Purpose:** Data transfers

**Key kernels:**
- `Memcpy HtoD`: Host to Device copy
- `Memcpy DtoD`: Device to Device copy
- `Memset`: Memory initialization

**Typical %:** < 1% of total CUDA time

---

### Reduction Operations
**Purpose:** Reductions (sum, max, etc.)

**Key kernels:**
- `DeviceScan`: Prefix sum/scan
- `reduce_kernel`: Generic reduction
- `splitKreduce_kernel`: Split-K GEMM reduction

**Typical %:** 1-3% of total CUDA time

---

## Usage Examples

### Basic Analysis
```bash
python analyze_kernels.py profiler_out_0.txt
```

### Analyze Latest Profiler Output
```bash
# Automatically finds profiler_out_0.txt in current dir or profiler_traces/
python analyze_kernels.py
```

### Compare Profiling Runs
```bash
# Run 1
python profile_vllm_simple.py
python analyze_kernels.py profiler_traces/profiler_out_0.txt > analysis_run1.txt

# Run 2 (with different config)
python profile_vllm_simple.py
python analyze_kernels.py profiler_traces/profiler_out_0.txt > analysis_run2.txt

# Compare
diff analysis_run1.txt analysis_run2.txt
```

## Interpretation Guide

### What to Look For

✅ **Good Signs:**
- FlashAttention: 15-20% (efficient attention)
- GEMM: 60-80% (expected, linear layers dominate)
- Sampling: 1-2% (efficient token generation)
- Total kernels: 20-40 (reasonable granularity)

⚠️ **Potential Issues:**
- FlashAttention > 30%: May indicate inefficient sequence length
- GEMM < 50%: Might not be utilizing matmul optimizations
- Sampling > 5%: Sampling overhead too high
- Memory ops > 5%: Too many data transfers

### Performance Optimization Tips

**If FlashAttention is slow:**
- Check sequence length (very long = slower)
- Verify split-K is being used (should see Combine kernel)
- Consider using FlashAttention 3 if on H100

**If GEMM is slow:**
- Verify FP16/BF16 precision (faster than FP32)
- Check if tensor cores are used (CUTLASS kernels)
- Consider quantization (FP8, INT8)

**If Element-wise is high:**
- May have too many small fused kernels
- Consider kernel fusion opportunities

## Common Kernel Name Patterns

| **Pattern** | **Meaning** |
|-------------|-------------|
| `flash::*` | FlashAttention kernels |
| `nvjet_tst_*` | NVIDIA GEMM kernels |
| `cutlass::*` | CUTLASS library kernels |
| `triton_*` | Triton-compiled kernels |
| `vllm::*` | vLLM custom kernels |
| `void *` | Generic CUDA kernel |
| `at::native::*` | PyTorch native ops |
| `Device*` | CUB/Thrust device operations |

## Troubleshooting

### "No kernels found"
- Make sure you're analyzing the correct file
- Verify file contains profiler output (not empty)
- Check that profiling was enabled during generation

### "File not found"
```bash
# Specify full path
python analyze_kernels.py /path/to/profiler_out_0.txt
```

### Kernel names truncated
The analyzer automatically shortens very long kernel names. Full names are preserved in the analysis logic.

## Integration with Other Tools

### Export for Spreadsheet Analysis
```bash
python analyze_kernels.py profiler_out_0.txt > analysis.txt

# Parse summary section for CSV
grep -A 20 "SUMMARY" analysis.txt
```

### Combine with TensorBoard
```bash
# Use both tools for comprehensive analysis
python analyze_kernels.py profiler_traces/profiler_out_0.txt
tensorboard --logdir=profiler_traces
```

## Advanced Features

### Custom Kernel Explanations
Edit `get_kernel_explanation()` in `analyze_kernels.py` to add custom explanations for your specific kernels.

### Adding New Categories
Edit `KERNEL_CATEGORIES` dictionary to add new kernel categories.

## Files Generated

| **File** | **Description** |
|----------|-----------------|
| `profiler_out_0.txt` | Raw profiler output from vLLM |
| Analysis stdout | Categorized kernel analysis |

## See Also

- `profile_vllm_operators.py` - Detailed profiling script
- `PROFILING_GUIDE.md` - Complete profiling documentation
- `explain_flashattn_combine_kernel.py` - FlashAttention details
