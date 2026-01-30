# vLLM Operator Profiling Guide

Complete guide for profiling vLLM key operators with runtime and memory analysis.

---

## Quick Start (3 Steps)

### 1. Simple Profiling (Fastest)

```bash
python profile_vllm_simple.py
```

**Output:**
- `profiler_traces/worker_0_rank_0.json.gz` - Raw trace
- Console output with timing

**View in TensorBoard:**
```bash
tensorboard --logdir=./profiler_traces
# Open http://localhost:6006
```

---

### 2. Detailed Analysis (Recommended)

```bash
# Step 1: Run profiling with analysis
python profile_vllm_operators.py

# Output:
# - profiling_results/operator_profile_summary.txt
# - profiling_results/operator_profile_detailed.csv
# - profiling_results/operator_timeline.json.gz
```

**What you get:**
- ✅ Breakdown by operator category (attention, matmul, norm, etc.)
- ✅ Top 20 operators by total time
- ✅ Per-operator statistics (min/max/avg/stddev)
- ✅ CSV export for further analysis

---

### 3. Analyze Existing Traces

If you already have trace files:

```bash
# Analyze latest trace
python analyze_profile.py

# Or specify trace file
python analyze_profile.py ./profiler_traces/worker_0_rank_0.json.gz
```

---

## Understanding the Output

### Key Operators You'll See

#### 1. **Attention: `flash_attn_varlen_func`** (40-50% of time)
- **Decode stage:** `max_seqlen_q=1` (single token)
- **Prefill stage:** `max_seqlen_q>1` (multiple tokens)
- **What it does:** Computes attention scores and weighted sum
- **Typical time:** 8-12ms per decode token, 300-400ms for 100-token prefill

#### 2. **Matrix Multiplication: `cutlass_scaled_mm` / `gemm`** (35-45% of time)
- **Operations:**
  - QKV projection: Projects hidden states to Q, K, V
  - Output projection: Projects attention output back
  - MLP layers: gate_proj, up_proj, down_proj
- **Typical time:** 7-10ms per decode token

#### 3. **KV Cache Write: `reshape_and_cache_flash`** (1-2% of time)
- **What it does:** Writes new K, V values to paged KV cache
- **Typical time:** 0.2-0.3ms per token

#### 4. **Position Embedding: `rotary_embedding`** (2-3% of time)
- **What it does:** Applies RoPE to Q and K
- **Typical time:** 0.3-0.5ms per token

#### 5. **Normalization: `rms_norm` / `fused_add_rms_norm`** (3-5% of time)
- **What it does:** Layer normalization before attention and MLP
- **Typical time:** 0.5-1ms per token

#### 6. **Sampling: `topk`, `multinomial`** (1-2% of time)
- **What it does:** Sample next token from logits
- **Typical time:** 0.2-0.4ms per token

---

## Example Output

### Console Output (from `profile_vllm_operators.py`)

```
====================================================================================================
vLLM OPERATOR PROFILING SUMMARY
====================================================================================================

Total Generation Time: 156.34 ms
Number of Unique Operators: 1247

====================================================================================================
BREAKDOWN BY CATEGORY
====================================================================================================

ATTENTION
  Total Time: 68.45 ms (43.8%)
  Total Calls: 32
  Avg Time per Call: 2.139 ms
  Top Operators:
    1. flash_attn_fwd_hdim128_fp16_causal_sm80                          45.23 ms (66.1%) [ 32 calls, avg: 1413.4 μs]
    2. flash_attn_varlen_func                                           23.22 ms (33.9%) [ 32 calls, avg:  725.6 μs]

MATMUL
  Total Time: 54.12 ms (34.6%)
  Total Calls: 128
  Avg Time per Call: 0.423 ms
  Top Operators:
    1. cutlass_scaled_mm_sm80                                           28.45 ms (52.6%) [ 64 calls, avg:  444.5 μs]
    2. volta_h884gemm_128x32_tn                                         15.67 ms (29.0%) [ 32 calls, avg:  489.7 μs]
    3. gemm_kernel_16x64                                                10.00 ms (18.5%) [ 32 calls, avg:  312.5 μs]

NORMALIZATION
  Total Time: 8.23 ms (5.3%)
  Total Calls: 64
  Avg Time per Call: 0.129 ms
  Top Operators:
    1. rms_norm_kernel                                                   5.12 ms (62.2%) [ 32 calls, avg:  160.0 μs]
    2. fused_add_rms_norm_kernel                                         3.11 ms (37.8%) [ 32 calls, avg:   97.2 μs]

====================================================================================================
TOP 20 OPERATORS BY TOTAL TIME
====================================================================================================
Rank   Operator                                                     Time (ms)    % Total    Calls    Avg (μs)
----------------------------------------------------------------------------------------------------
1      flash_attn_fwd_hdim128_fp16_causal_sm80                      45.23        28.93      32       1413.4
2      cutlass_scaled_mm_sm80                                       28.45        18.20      64       444.5
3      flash_attn_varlen_func                                       23.22        14.85      32       725.6
4      volta_h884gemm_128x32_tn                                     15.67        10.02      32       489.7
5      gemm_kernel_16x64                                            10.00        6.40       32       312.5
6      rms_norm_kernel                                              5.12         3.28       32       160.0
7      fused_add_rms_norm_kernel                                    3.11         1.99       32       97.2
8      rotary_embedding_kernel                                      2.45         1.57       32       76.6
9      reshape_and_cache_flash_kernel                               1.89         1.21       32       59.1
10     silu_kernel                                                  1.34         0.86       16       83.8
...
```

---

## Profiling Configurations

### Basic Configuration (Default)

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()
```

### Advanced Configuration

```python
import os

# Enable custom profiling scopes
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"

# For NVTX annotations (use with Nsight Systems)
# os.environ["VLLM_NVTX_SCOPES_FOR_PROFILING"] = "1"

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    enforce_eager=True,  # Disable CUDA graphs for clearer profiling
)
```

---

## Viewing Profiles

### Method 1: TensorBoard (Recommended)

```bash
tensorboard --logdir=./profiler_traces
# Open http://localhost:6006
# Navigate to "PyTorch Profiler" tab
```

**Features:**
- ✅ Timeline view of all operators
- ✅ Kernel breakdown by time
- ✅ Memory usage over time
- ✅ GPU utilization

### Method 2: Chrome Tracing

```bash
# Open Chrome browser
# Go to: chrome://tracing
# Click "Load" and select: profiling_results/operator_timeline.json.gz
```

**Features:**
- ✅ Interactive timeline
- ✅ Zoom and pan
- ✅ Detailed event info on hover

### Method 3: CSV Analysis

```python
import pandas as pd

# Load CSV
df = pd.read_csv('profiling_results/operator_profile_detailed.csv')

# Filter attention operators
attention_ops = df[df['Category'] == 'attention']
print(attention_ops.sort_values('Total Time (ms)', ascending=False))

# Compare categories
category_summary = df.groupby('Category').agg({
    'Total Time (ms)': 'sum',
    'Call Count': 'sum'
}).sort_values('Total Time (ms)', ascending=False)
print(category_summary)
```

---

## Profiling Different Scenarios

### Scenario 1: Prefill vs Decode

```python
# Profile prefill (long prompt)
long_prompt = "Once upon a time... " * 100  # ~300 tokens
llm.start_profile()
outputs = llm.generate([long_prompt], SamplingParams(max_tokens=1))
llm.stop_profile()
# Mostly prefill time

# Profile decode (many output tokens)
short_prompt = "Hello"
llm.start_profile()
outputs = llm.generate([short_prompt], SamplingParams(max_tokens=100))
llm.stop_profile()
# Mostly decode time
```

### Scenario 2: Batch Processing

```python
# Single request
llm.start_profile()
outputs = llm.generate(["Hello"], SamplingParams(max_tokens=20))
llm.stop_profile()

# Batched requests (4x)
llm.start_profile()
outputs = llm.generate(["Hello"] * 4, SamplingParams(max_tokens=20))
llm.stop_profile()
# Compare throughput
```

### Scenario 3: Memory Profiling

```python
import torch

# Track memory
torch.cuda.reset_peak_memory_stats()

llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()

peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")
```

---

## Understanding Decode Stage Attention

### The Key Function: `flash_attn_varlen_func`

**Location:** `vllm/v1/attention/backends/flash_attn.py:719`

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],      # [batch_size, num_heads, head_size]
    k=key_cache,                       # Full KV cache
    v=value_cache,                     # Full KV cache
    out=output[:num_actual_tokens],
    cu_seqlens_q=[0, 1, 2, ..., B],   # Cumulative seq lengths (decode)
    max_seqlen_q=1,                    # ← KEY: 1 for decode, >1 for prefill
    seqused_k=seq_lens,                # Actual cached sequence lengths
    max_seqlen_k=max_seq_len,          # Max cached length
    softmax_scale=1/sqrt(head_dim),
    causal=True,
    block_table=block_table,           # PagedAttention pointers
)
```

**Decode-specific behavior:**
- When `max_seqlen_q=1`, FlashAttention uses **decode-optimized kernel**
- Single query per request attends to all cached K/V
- Complexity: O(N) where N is sequence length
- Optimized for batch processing across multiple requests

---

## Troubleshooting

### Issue: No trace files generated

**Solution:**
```python
# Make sure profiling is enabled
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"

# Check if trace directory exists
import os
if not os.path.exists("./profiler_traces"):
    print("Profiler traces not generated!")
```

### Issue: Trace file too large

**Solution:**
```python
# Reduce profiling duration
sampling_params = SamplingParams(max_tokens=10)  # Instead of 100

# Or profile fewer prompts
prompts = ["Hello"]  # Instead of batch of 10
```

### Issue: OOM (Out of Memory)

**Solution:**
```python
# Reduce GPU memory utilization
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    gpu_memory_utilization=0.7,  # Instead of 0.9
)

# Use smaller batch size
prompts = prompts[:2]  # Instead of large batch
```

---

## Expected Performance (Llama-3.1-8B, 1xA100 GPU)

### Decode Stage (per token)

| Operator | Time | % of Total |
|----------|------|------------|
| FlashAttention | 8-12 ms | 40-50% |
| GEMM (QKV + MLP) | 7-10 ms | 35-45% |
| RMSNorm | 0.5-1 ms | 3-5% |
| RoPE | 0.3-0.5 ms | 2-3% |
| Sampling | 0.2-0.4 ms | 1-2% |
| KV Cache | 0.2-0.3 ms | 1-2% |
| **Total** | **20-30 ms** | **100%** |

**Throughput:** 33-50 tokens/second

### Prefill Stage (100 tokens)

| Operator | Time | % of Total |
|----------|------|------------|
| FlashAttention | 300-400 ms | 60-70% |
| GEMM (QKV + MLP) | 150-200 ms | 25-35% |
| Other | 20-50 ms | 5-10% |
| **Total** | **500-650 ms** | **100%** |

---

## Advanced: NVTX Profiling with Nsight Systems

For the most detailed GPU kernel analysis:

```bash
# Enable NVTX annotations
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

# Profile with Nsight Systems
nsys profile \
    --trace=cuda,nvtx,osrt \
    --cuda-memory-usage=true \
    --gpu-metrics-device=all \
    --output=vllm_detailed.nsys-rep \
    --force-overwrite true \
    python profile_vllm_simple.py

# View in Nsight Systems GUI
nsys-ui vllm_detailed.nsys-rep
```

**What you get:**
- ✅ Nanosecond-precision kernel timings
- ✅ CUDA stream visualization
- ✅ Memory transfer analysis
- ✅ GPU utilization timeline
- ✅ Warp occupancy metrics

---

## Summary

**Quick Commands:**
```bash
# 1. Simple profiling
python profile_vllm_simple.py

# 2. Detailed analysis
python profile_vllm_operators.py

# 3. Analyze existing traces
python analyze_profile.py

# 4. View in TensorBoard
tensorboard --logdir=./profiler_traces
```

**Key Operators to Watch:**
1. `flash_attn_varlen_func` - Attention (40-50% of time)
2. `cutlass_scaled_mm` - Matrix multiplication (35-45%)
3. `rms_norm` - Normalization (3-5%)
4. `rotary_embedding` - Position encoding (2-3%)
5. `reshape_and_cache_flash` - KV cache writes (1-2%)
6. Sampling operations (1-2%)

**Decode Stage Attention:**
- Function: `flash_attn_varlen_func` with `max_seqlen_q=1`
- Optimized for single-token processing
- Complexity: O(N) per request where N is sequence length
