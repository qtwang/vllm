# vLLM Profiling Quick Reference

## One-Line Commands

```bash
# Simple profiling
python profile_vllm_simple.py && tensorboard --logdir=./profiler_traces

# Detailed profiling + analysis
python profile_vllm_operators.py && cat profiling_results/operator_profile_summary.txt

# Analyze existing trace
python analyze_profile.py ./profiler_traces/worker_0_rank_0.json.gz
```

---

## Key Operators Reference

| **Operator** | **File Location** | **Purpose** | **Typical Time (Decode)** |
|--------------|-------------------|-------------|---------------------------|
| `flash_attn_varlen_func` | `vllm/v1/attention/backends/flash_attn.py:719` | Attention computation | 8-12 ms (40-50%) |
| `cutlass_scaled_mm` | `vllm/_custom_ops.py:753` | Matrix multiplication (QKV, MLP) | 7-10 ms (35-45%) |
| `rms_norm` | `vllm/_custom_ops.py:335` | RMS normalization | 0.5-1 ms (3-5%) |
| `rotary_embedding` | `vllm/_custom_ops.py:318` | Rotary position embedding | 0.3-0.5 ms (2-3%) |
| `reshape_and_cache_flash` | `vllm/v1/attention/backends/fa_utils.py:12` | Write to KV cache | 0.2-0.3 ms (1-2%) |
| `topk` / `multinomial` | PyTorch ops | Token sampling | 0.2-0.4 ms (1-2%) |

---

## Decode Stage Attention

**Function:** `flash_attn_varlen_func`

**Decode Signature:**
```python
flash_attn_varlen_func(
    q=query,                    # [B, num_heads, head_dim]
    k=key_cache,                # Full KV cache
    v=value_cache,              # Full KV cache
    cu_seqlens_q=[0,1,2,...,B], # Cumulative sequence lengths
    max_seqlen_q=1,             # ← KEY: 1 for decode
    seqused_k=actual_seq_lens,  # Current sequence lengths
    causal=True,
    block_table=block_table,    # Paged attention pointers
)
```

**Prefill vs Decode Detection:**
- `max_seqlen_q == 1` → **Decode stage** (O(N) complexity)
- `max_seqlen_q > 1` → **Prefill stage** (O(N²) complexity)

---

## Environment Variables

```bash
# Enable custom profiling scopes (for PyTorch profiler)
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

# Enable NVTX annotations (for Nsight Systems)
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
```

---

## Profiling Code Template

```python
import os
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"

from vllm import LLM, SamplingParams

# Initialize
llm = LLM(model="meta-llama/Llama-3.1-8B", tensor_parallel_size=1)

# Warmup
llm.generate(["warmup"], SamplingParams(max_tokens=1))

# Profile
llm.start_profile()
outputs = llm.generate(["Hello world"], SamplingParams(max_tokens=20))
llm.stop_profile()

# Check: ./profiler_traces/worker_0_rank_0.json.gz
```

---

## Expected Performance (1x A100 GPU)

### Per Token (Decode)
- **Total:** 20-30 ms
- **Throughput:** 33-50 tokens/sec

### 100-Token Prefill
- **Total:** 500-650 ms

### Memory Usage
- **Model weights (FP16):** ~16 GB
- **KV cache (batch=1, len=2048):** ~0.5 GB
- **Activations:** ~0.2-0.5 GB
- **Total:** ~17-18 GB

---

## Viewing Results

| **Tool** | **Command** | **Best For** |
|----------|-------------|--------------|
| **TensorBoard** | `tensorboard --logdir=./profiler_traces` | Timeline view, GPU utilization |
| **Chrome Trace** | Open `chrome://tracing`, load `.json.gz` | Interactive timeline |
| **CSV** | `pandas.read_csv('operator_profile_detailed.csv')` | Data analysis |
| **Nsight Systems** | `nsys-ui vllm_detailed.nsys-rep` | Kernel-level analysis |

---

## Troubleshooting

| **Issue** | **Solution** |
|-----------|-------------|
| No traces generated | Set `export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` |
| Trace file too large | Reduce `max_tokens` or batch size |
| OOM error | Reduce `gpu_memory_utilization=0.7` |
| Missing operators | Try `enforce_eager=True` to disable CUDA graphs |

---

## Files Created

- `profile_vllm_simple.py` - Quick start profiling
- `profile_vllm_operators.py` - Detailed profiling with analysis
- `analyze_profile.py` - Analyze existing traces
- `PROFILING_GUIDE.md` - Full documentation
- `PROFILING_QUICK_REF.md` - This file

---

## Next Steps

1. Run `python profile_vllm_simple.py`
2. View in TensorBoard: `tensorboard --logdir=./profiler_traces`
3. Open http://localhost:6006
4. Navigate to "PyTorch Profiler" → "Overview"
5. Check "Operator View" for detailed breakdown
