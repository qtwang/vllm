# Profiling Prefill vs Decode Stages in vLLM

This guide explains how to separately profile the prefill and decode stages of LLM generation in vLLM.

## Table of Contents

1. [Understanding Prefill vs Decode](#understanding-prefill-vs-decode)
2. [Why Separate Profiling Matters](#why-separate-profiling-matters)
3. [Methods to Profile Decode Only](#methods-to-profile-decode-only)
4. [Usage Examples](#usage-examples)
5. [Interpreting Results](#interpreting-results)
6. [Advanced Techniques](#advanced-techniques)

---

## Understanding Prefill vs Decode

### Prefill Stage
- **What**: Processes the entire input prompt in parallel
- **When**: First forward pass for a new sequence
- **Characteristics**:
  - Processes N input tokens simultaneously (N > 1)
  - High parallelism across input sequence
  - `max_seqlen_q` > 1 in attention operations
  - Fills KV cache with input token representations
  - Computationally intensive (quadratic in sequence length for attention)

### Decode Stage
- **What**: Generates output tokens one at a time (autoregressive)
- **When**: Every step after prefill
- **Characteristics**:
  - Processes 1 token at a time
  - `max_seqlen_q` = 1 in attention operations
  - Attends to growing KV cache (all previous tokens)
  - Linear growth in computation per step
  - Repeated many times (once per output token)

### Visual Timeline

```
Input: "The future of AI"
Output: "is very promising and..."

┌─────────┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐  ┌──┐
│ PREFILL │  │D │  │D │  │D │  │D │  │D │  │D │  ...
│ (all 4  │  │E │  │E │  │E │  │E │  │E │  │E │
│  tokens)│  │C │  │C │  │C │  │C │  │C │  │C │
│         │  │O │  │O │  │O │  │O │  │O │  │O │
│         │  │D │  │D │  │D │  │D │  │D │  │D │
│         │  │E │  │E │  │E │  │E │  │E │  │E │
└─────────┘  └──┘  └──┘  └──┘  └──┘  └──┘  └──┘
             "is"  "very" "pro-" "mis-" "ing" "and"

Profiling:   ├─── Not Profiled ───┤├─── PROFILED ─────>
```

---

## Why Separate Profiling Matters

1. **Different Performance Characteristics**
   - Prefill: Bound by compute (large matrix operations)
   - Decode: Bound by memory bandwidth (KV cache access)

2. **Different Optimization Targets**
   - Prefill: Maximize parallelism, use efficient attention
   - Decode: Minimize memory transfers, batch multiple sequences

3. **Realistic Workload Analysis**
   - Long-running services spend 90%+ time in decode
   - Profiling both together can obscure decode bottlenecks

4. **Kernel Behavior Differences**
   - FlashAttention uses different strategies for prefill vs decode
   - GEMM shapes differ significantly (batch vs single token)

---

## Methods to Profile Decode Only

### Method 1: Two-Step Generation (Recommended)

**Approach**: Run generation twice, profile only the second call.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B")
prompt = "The future of AI is"

# Step 1: Prefill (no profiling)
prefill_params = SamplingParams(max_tokens=1, temperature=0.0)
prefill_out = llm.generate([prompt], prefill_params)
first_token = prefill_out[0].outputs[0].text

# Step 2: Decode only (with profiling)
llm.start_profile()
continuation = prompt + first_token
decode_params = SamplingParams(max_tokens=100, temperature=0.8)
decode_out = llm.generate([continuation], decode_params)
llm.stop_profile()
```

**Pros**:
- Simple and straightforward
- Works with high-level LLM API
- Clear separation between stages

**Cons**:
- Requires two separate generate calls
- First call still includes some prefill overhead

### Method 2: Warmup + Profile

**Approach**: Generate warmup tokens, then profile continuation.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B")
prompt = "The future of AI is"

# Warmup: Prefill + a few decode steps
warmup_params = SamplingParams(max_tokens=5, temperature=0.0)
warmup_out = llm.generate([prompt], warmup_params)

# Profile: Pure decode
llm.start_profile()
continuation = prompt + warmup_out[0].outputs[0].text
decode_params = SamplingParams(max_tokens=50, temperature=0.8)
decode_out = llm.generate([continuation], decode_params)
llm.stop_profile()
```

**Pros**:
- KV cache is already populated
- More realistic decode-only profiling
- Can control warmup length

**Cons**:
- Still requires multiple generate calls
- Need to manage prompt continuation

### Method 3: Conditional Profiling

**Approach**: Use profiling start/stop within a single generation (requires custom loop).

This requires accessing the engine API or implementing a custom generation loop, which is more advanced.

---

## Usage Examples

### Example 1: Profile Decode Only (Basic)

```bash
# Using the provided script
python profile_decode_only.py \
    --model meta-llama/Llama-3.1-8B \
    --prompt "The future of artificial intelligence is" \
    --prefill-tokens 1 \
    --decode-tokens 100
```

**Output**:
```
================================================================================
PROFILING DECODE STAGE ONLY
================================================================================

1. Initializing model: meta-llama/Llama-3.1-8B

2. Running PREFILL phase (generating 1 token(s), no profiling)...
   Prefill completed. Generated: ' a'

3. Starting profiler for DECODE phase...
4. Running DECODE phase (generating 100 tokens, WITH profiling)...

5. Stopping profiler...

Profile saved to: profiler_out_0.txt
NOTE: The profile now contains ONLY decode stage operations
```

### Example 2: Profile Specific Decode Iterations

```bash
python profile_decode_engine_api.py \
    --mode specific \
    --warmup-tokens 10 \
    --profile-tokens 50
```

This will:
1. Generate 10 warmup tokens (includes prefill, not profiled)
2. Generate 50 more tokens (pure decode, profiled)

### Example 3: Compare Prefill vs Decode

To compare both stages, run profiling twice:

```bash
# Profile full generation (prefill + decode)
python profile_vllm_simple.py

# Rename output
mv profiler_out_0.txt profiler_full.txt

# Profile decode only
python profile_decode_only.py --decode-tokens 100
mv profiler_out_0.txt profiler_decode_only.txt

# Analyze both
python analyze_kernels.py profiler_full.txt > analysis_full.txt
python analyze_kernels.py profiler_decode_only.txt > analysis_decode.txt

# Compare
diff analysis_full.txt analysis_decode.txt
```

---

## Interpreting Results

### Decode-Only Profile Characteristics

When you successfully profile only decode, you should see:

1. **FlashAttention with small queries**
   ```
   flash::FlashAttnFwdSm90    (many calls, smaller time per call)
   ```
   - Each call processes 1 query token
   - Many repeated calls (one per output token)

2. **Smaller GEMM operations**
   ```
   nvjet_tst_*    (batch_size=1, single token MLP)
   ```
   - Reduced from batch processing to single token

3. **Sampling operations prominent**
   ```
   DeviceRadixSort::SortPairs
   triton_*_fused_*_softmax
   ```
   - More visible since prefill's large operations are gone

4. **Regular repeated pattern**
   - Decode is repetitive: same operations each token
   - Profile should show very regular call patterns

### Prefill-Only Profile Characteristics

To profile only prefill (less common):

```python
llm.start_profile()
output = llm.generate([prompt], SamplingParams(max_tokens=1))
llm.stop_profile()
```

You'll see:
1. Larger FlashAttention operations (processing all input tokens)
2. Larger GEMM operations (batch processing input)
3. More KV cache writes (caching all input representations)
4. Fewer sampling operations (only 1 output token)

### Key Metrics to Compare

| Metric | Prefill | Decode |
|--------|---------|--------|
| FlashAttn calls | Few (often 1) | Many (one per token) |
| FlashAttn time/call | High | Lower |
| GEMM time/call | High | Lower |
| Sampling time | Minimal | Significant |
| Total time | Proportional to input length | Proportional to output length |

---

## Advanced Techniques

### 1. Profile Multiple Decode Iterations Separately

To see variation across decode steps:

```python
llm = LLM(model="meta-llama/Llama-3.1-8B")

# Warmup
warmup_out = llm.generate([prompt], SamplingParams(max_tokens=5))
base_prompt = prompt + warmup_out[0].outputs[0].text

# Profile iteration batches
for i in range(5):
    llm.start_profile()
    continuation = base_prompt
    out = llm.generate([continuation], SamplingParams(max_tokens=10))
    llm.stop_profile()

    # Rename profile output
    os.rename("profiler_out_0.txt", f"profiler_decode_iter_{i}.txt")

    base_prompt += out[0].outputs[0].text
```

### 2. Profile Decode at Different KV Cache Sizes

```python
def profile_decode_at_cache_size(cache_size: int):
    """Profile decode when KV cache contains `cache_size` tokens."""

    # Build up KV cache to desired size
    warmup_out = llm.generate([prompt], SamplingParams(max_tokens=cache_size))

    # Profile decode with this cache size
    llm.start_profile()
    continuation = prompt + warmup_out[0].outputs[0].text
    llm.generate([continuation], SamplingParams(max_tokens=10))
    llm.stop_profile()

# Profile at different cache sizes
for size in [10, 50, 100, 500, 1000]:
    profile_decode_at_cache_size(size)
    os.rename("profiler_out_0.txt", f"profiler_cache_{size}.txt")
```

### 3. Verify Decode-Only via Kernel Inspection

After profiling, verify you captured only decode:

```bash
# Analyze kernels
python analyze_kernels.py profiler_out_0.txt > kernel_analysis.txt

# Check FlashAttention call count
grep -A 2 "flash::FlashAttnFwdSm90" kernel_analysis.txt

# Should show many calls (close to number of output tokens)
# For 100 output tokens across ~32 layers, expect ~3200 calls
```

---

## Quick Reference

### Commands

```bash
# Basic decode-only profiling
python profile_decode_only.py

# Advanced with custom parameters
python profile_decode_engine_api.py --mode specific --warmup-tokens 10 --profile-tokens 50

# Analyze results
python analyze_kernels.py profiler_out_0.txt

# Get explanation of decode vs prefill
python profile_decode_engine_api.py --mode explain
```

### Key Files

- `profile_decode_only.py` - Simple two-step decode profiling
- `profile_decode_engine_api.py` - Advanced engine-level control
- `analyze_kernels.py` - Kernel categorization and analysis
- `profiler_out_0.txt` - Generated profile output

---

## Troubleshooting

### Issue: Profile still shows prefill-like patterns

**Symptoms**:
- Very few FlashAttention calls
- Large time per kernel call
- High GEMM operation times

**Solution**:
- Increase warmup tokens to ensure prefill is complete
- Verify continuation prompt includes previous generation
- Check that profiling starts AFTER warmup generation

### Issue: Profile shows no FlashAttention kernels

**Symptoms**:
- Missing flash::FlashAttnFwdSm90 in output
- Only see high-level wrappers

**Solution**:
- Ensure profiler records CUDA kernels (not just CPU)
- Check `max_name_column_width` in profiler config
- Verify GPU is being used (not CPU fallback)

### Issue: Inconsistent decode timings

**Symptoms**:
- Large variation in kernel times across tokens
- Unexpected performance spikes

**Solution**:
- Run longer decode sequences for averaging
- Add more warmup tokens to stabilize
- Check for GPU frequency scaling or thermal throttling

---

## Summary

**To profile decode only**:

1. ✓ Run warmup generation (no profiling) to complete prefill
2. ✓ Start profiling
3. ✓ Continue generation (pure decode)
4. ✓ Stop profiling
5. ✓ Analyze with `analyze_kernels.py`

**Expected decode profile**:
- Many small FlashAttention calls
- Repeated regular pattern
- Prominent sampling operations
- Time proportional to output length

**Key insight**: In vLLM, prefill vs decode is not explicit - it's determined by whether you're processing the initial prompt (prefill) or continuation (decode). By controlling when profiling starts, you control what gets captured.
