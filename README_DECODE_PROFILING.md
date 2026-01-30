# Decode-Only Profiling for vLLM

This directory contains scripts and guides for profiling only the decode stage of vLLM generation, separating it from the prefill stage.

## Quick Start

The simplest way to profile decode-only:

```bash
python example_profile_decode_only.py
```

This will:
1. Load Llama-3.1-8B
2. Generate 5 warmup tokens (includes prefill, not profiled)
3. Profile 50 decode tokens
4. Save results to `profiler_out_0.txt`

## Why Decode-Only Profiling?

**Prefill** and **decode** have very different performance characteristics:

| Aspect | Prefill | Decode |
|--------|---------|--------|
| **What** | Process input prompt | Generate output tokens |
| **Parallelism** | High (all input tokens) | Low (1 token at a time) |
| **Bottleneck** | Compute | Memory bandwidth |
| **Frequency** | Once per request | Once per output token |
| **Time %** | 10-20% for long outputs | 80-90% for long outputs |

For production workloads generating long outputs, decode is the critical path. Profiling it separately reveals the true bottlenecks.

## Available Scripts

### 1. `example_profile_decode_only.py` ⭐ START HERE

**Simplest approach** - Ready-to-run example with explanatory output.

```bash
python example_profile_decode_only.py
```

**What it does**:
- Warmup: 5 tokens (includes prefill)
- Profile: 50 decode tokens
- Clear step-by-step output
- Explains what was profiled

**Use when**: You want to quickly profile decode without configuration.

---

### 2. `profile_decode_only.py`

**Configurable two-step profiling** with command-line options.

```bash
python profile_decode_only.py \
    --model meta-llama/Llama-3.1-8B \
    --prompt "Your custom prompt here" \
    --prefill-tokens 1 \
    --decode-tokens 100
```

**Options**:
- `--method`: `standard` or `alternative`
- `--model`: Model name or path
- `--prompt`: Input prompt
- `--prefill-tokens`: Tokens in warmup phase
- `--decode-tokens`: Tokens to profile

**Use when**: You need custom prompts or token counts.

---

### 3. `profile_decode_engine_api.py`

**Advanced engine-level control** with multiple profiling modes.

```bash
# Specific decode iterations
python profile_decode_engine_api.py \
    --mode specific \
    --warmup-tokens 10 \
    --profile-tokens 50

# Explanation mode (no profiling, just educational)
python profile_decode_engine_api.py --mode explain
```

**Modes**:
- `specific`: Warmup + profile specific iterations
- `basic`: Simple engine-based profiling
- `explain`: Print explanation of prefill vs decode

**Use when**: You need fine-grained control or want to understand the internals.

---

## How It Works

### The Two-Step Approach

All scripts use the same fundamental technique:

```python
# Step 1: Warmup (completes prefill)
warmup_output = llm.generate([prompt], SamplingParams(max_tokens=N))

# Step 2: Profile decode only
llm.start_profile()
continuation = prompt + warmup_output[0].outputs[0].text
decode_output = llm.generate([continuation], SamplingParams(max_tokens=M))
llm.stop_profile()
```

**Key insight**: The second `generate()` call on a continuation prompt only performs decode, since prefill was already done in the first call.

### Why This Works

In vLLM:
- **Prefill**: First forward pass for a sequence (processes full prompt)
- **Decode**: Continuation forward passes (processes generated tokens)

By running a warmup generation before profiling, we ensure the profiler only captures decode operations.

---

## Analyzing Results

After profiling, analyze the output:

### 1. View Raw Profile

```bash
cat profiler_out_0.txt
```

Look for:
- **Many FlashAttention calls**: One per output token per layer
  - Example: 50 tokens × 32 layers = ~1600 calls
- **Smaller operation times**: Single-token operations
- **Regular patterns**: Decode is repetitive

### 2. Analyze Kernels

```bash
python analyze_kernels.py profiler_out_0.txt
```

Expected decode profile shows:
```
========================================================
1. FLASHATTENTION KERNELS
========================================================
Total FlashAttention Time: X.XX ms

flash::FlashAttnFwdSm90        X.XX ms  (~1600 calls)
  └─> Main attention (1 query token × growing KV cache)

flash::FlashAttnFwdCombine     X.XX ms  (~1600 calls)
  └─> Combines split-K attention results

========================================================
2. NON-FLASHATTENTION KERNELS
========================================================

MATRIX MULTIPLICATION (GEMM) KERNELS
nvjet_tst_*                    X.XX ms  (many calls)
  └─> MLP projections (single token)
...
```

### 3. Compare with Full Profile

To see the difference, profile both:

```bash
# Full generation (prefill + decode)
python profile_vllm_simple.py
mv profiler_out_0.txt profiler_full.txt

# Decode only
python example_profile_decode_only.py
mv profiler_out_0.txt profiler_decode.txt

# Analyze both
python analyze_kernels.py profiler_full.txt > analysis_full.txt
python analyze_kernels.py profiler_decode.txt > analysis_decode.txt

# Compare
diff analysis_full.txt analysis_decode.txt
```

**Differences you'll see**:
- Decode-only: More FlashAttention calls, smaller per-call time
- Full: Fewer FA calls, larger per-call time (batching input tokens)
- Decode-only: More prominent sampling operations
- Full: More prominent input processing operations

---

## Verification

### How to verify you profiled decode only:

1. **Check call counts**:
   ```bash
   grep "flash::FlashAttnFwdSm90" profiler_out_0.txt
   ```
   Should show many calls (approximately: output_tokens × num_layers)

2. **Check operation patterns**:
   - Decode: Very regular, repeated operations
   - Prefill: More varied, includes input processing

3. **Check FlashAttention characteristics**:
   - Decode: `max_seqlen_q = 1` (single query token)
   - Prefill: `max_seqlen_q > 1` (multiple query tokens)

4. **Time distribution**:
   - Decode: Time grows linearly with output length
   - Prefill: Time grows with input length

---

## Use Cases

### 1. Optimize Long-Context Decode

For long-context scenarios, decode becomes the bottleneck:

```bash
# Profile decode with large KV cache
python profile_decode_engine_api.py \
    --warmup-tokens 2000 \
    --profile-tokens 100
```

Analyze to find:
- KV cache access patterns
- Memory bandwidth bottlenecks
- Attention kernel efficiency at large cache sizes

### 2. Batch Size Impact on Decode

Profile decode at different batch sizes:

```python
# Modify scripts to generate multiple sequences
# Profile to see batching effects on decode
```

### 3. Model Comparison

Compare decode efficiency across models:

```bash
# Profile Llama-3.1-8B decode
python profile_decode_only.py --model meta-llama/Llama-3.1-8B
mv profiler_out_0.txt profiler_llama8b_decode.txt

# Profile Llama-3.1-70B decode
python profile_decode_only.py --model meta-llama/Llama-3.1-70B
mv profiler_out_0.txt profiler_llama70b_decode.txt

# Compare
python analyze_kernels.py profiler_llama8b_decode.txt > analysis_8b.txt
python analyze_kernels.py profiler_llama70b_decode.txt > analysis_70b.txt
```

---

## Advanced Usage

### Profile Decode at Different KV Cache Sizes

To see how decode performance changes with KV cache size:

```bash
# Small cache (10 tokens)
python profile_decode_engine_api.py --warmup-tokens 10 --profile-tokens 20
mv profiler_out_0.txt profiler_cache_10.txt

# Medium cache (100 tokens)
python profile_decode_engine_api.py --warmup-tokens 100 --profile-tokens 20
mv profiler_out_0.txt profiler_cache_100.txt

# Large cache (1000 tokens)
python profile_decode_engine_api.py --warmup-tokens 1000 --profile-tokens 20
mv profiler_out_0.txt profiler_cache_1000.txt

# Analyze all
for size in 10 100 1000; do
    echo "=== Cache size: $size ===" >> decode_scaling.txt
    python analyze_kernels.py profiler_cache_${size}.txt >> decode_scaling.txt
done
```

### Profile Specific Decode Iterations

To profile just iterations 50-60:

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B")

# Generate 50 tokens (no profiling)
warmup_out = llm.generate([prompt], SamplingParams(max_tokens=50))

# Profile next 10 tokens (iterations 50-60)
llm.start_profile()
continuation = prompt + warmup_out[0].outputs[0].text
decode_out = llm.generate([continuation], SamplingParams(max_tokens=10))
llm.stop_profile()
```

---

## Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `example_profile_decode_only.py` | Simple ready-to-run example | Quick start |
| `profile_decode_only.py` | Configurable profiling | Custom prompts/lengths |
| `profile_decode_engine_api.py` | Advanced engine control | Fine-grained control |
| `PREFILL_DECODE_PROFILING_GUIDE.md` | Comprehensive guide | Understanding concepts |
| `analyze_kernels.py` | Kernel analysis tool | Analyzing profiles |

---

## Common Questions

### Q: Can I profile prefill only?

Yes! Just profile the first generation:

```python
llm.start_profile()
output = llm.generate([prompt], SamplingParams(max_tokens=1))
llm.stop_profile()
```

This captures prefill + 1 decode token (mostly prefill).

### Q: How many warmup tokens should I use?

- **Minimum**: 1 token (completes prefill)
- **Recommended**: 5-10 tokens (stabilizes performance)
- **For large cache testing**: Match your target cache size

### Q: Why does the continuation prompt include previous output?

In vLLM, each `generate()` call is stateless. To continue generation, you must provide the full context (prompt + previous output). The engine will recognize this as a continuation and only process the new tokens in decode mode.

### Q: Can I profile multiple sequences in parallel?

Yes! Pass multiple prompts to `generate()`:

```python
llm.start_profile()
outputs = llm.generate([prompt1, prompt2, prompt3], params)
llm.stop_profile()
```

This profiles batched decode, which has different characteristics than single-sequence decode.

---

## Troubleshooting

### Profile shows prefill-like patterns

**Symptoms**: Few FlashAttention calls, large per-call times

**Solution**: Increase warmup tokens, verify continuation prompt is correct

### No FlashAttention kernels in output

**Symptoms**: Only see wrapper functions, not actual kernels

**Solution**: Check that profiler captures CUDA kernels (it should by default)

### Inconsistent decode times

**Symptoms**: Large variation across tokens

**Solution**: Run longer sequences for averaging, add more warmup tokens

---

## Summary

**To profile decode only**:

1. Run `python example_profile_decode_only.py`
2. Analyze with `python analyze_kernels.py profiler_out_0.txt`
3. Look for decode characteristics (many small FlashAttention calls)

**Expected decode profile**:
- Many FlashAttention kernel calls
- Regular, repeated operation patterns
- Prominent sampling operations
- Linear scaling with output length

**Key files**:
- `example_profile_decode_only.py` - Start here
- `PREFILL_DECODE_PROFILING_GUIDE.md` - Full documentation
- `analyze_kernels.py` - Analyze results

For questions or issues, refer to `PREFILL_DECODE_PROFILING_GUIDE.md`.
