#!/usr/bin/env python3
"""
Simple example: Profile only the decode stage.

This is the simplest way to profile decode-only in vLLM.
Just run: python example_profile_decode_only.py
"""

from vllm import LLM, SamplingParams

# Configuration
MODEL = "meta-llama/Llama-3.1-8B"
PROMPT = "The future of artificial intelligence is"
WARMUP_TOKENS = 5  # Tokens to generate in warmup (includes prefill)
DECODE_TOKENS = 50  # Tokens to profile (pure decode)

print("=" * 80)
print("SIMPLE DECODE-ONLY PROFILING EXAMPLE")
print("=" * 80)

# Step 1: Initialize model
print(f"\nStep 1: Loading model {MODEL}...")
llm = LLM(
    model=MODEL,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)
print("✓ Model loaded")

# Step 2: Warmup (includes prefill, not profiled)
print(f"\nStep 2: Warmup - generating {WARMUP_TOKENS} tokens (NOT profiled)...")
print(f"  This completes the prefill stage.")

warmup_params = SamplingParams(
    temperature=0.0,
    max_tokens=WARMUP_TOKENS,
)

warmup_output = llm.generate([PROMPT], warmup_params)
warmup_text = warmup_output[0].outputs[0].text

print(f"  Warmup output: '{warmup_text}'")
print("✓ Warmup complete (prefill done)")

# Step 3: Profile decode only
print(f"\nStep 3: Profiling {DECODE_TOKENS} decode tokens...")
print("  Starting profiler...")

llm.start_profile()

# Continue from warmup
continuation_prompt = PROMPT + warmup_text

decode_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=DECODE_TOKENS,
)

decode_output = llm.generate([continuation_prompt], decode_params)
decode_text = decode_output[0].outputs[0].text

llm.stop_profile()

print("✓ Profiling complete")

# Step 4: Results
print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

print(f"\nOriginal prompt:")
print(f"  \"{PROMPT}\"")

print(f"\nWarmup generation (not profiled):")
print(f"  \"{warmup_text}\"")

print(f"\nDecode generation (PROFILED):")
print(f"  \"{decode_text[:100]}{'...' if len(decode_text) > 100 else ''}\"")

print(f"\n✓ Profile saved to: profiler_out_0.txt")
print("\nWhat was profiled:")
print("  • ONLY decode operations (no prefill)")
print(f"  • {DECODE_TOKENS} token generation steps")
print("  • All operators used during autoregressive decode")

print("\nNext steps:")
print("  1. View profile: cat profiler_out_0.txt")
print("  2. Analyze kernels: python analyze_kernels.py profiler_out_0.txt")
print("  3. Look for decode patterns:")
print("     - Many FlashAttention calls (one per output token per layer)")
print("     - Smaller GEMM operations (single token, not batched)")
print("     - Sampling operations (argmax, softmax for next token)")

print("\n" + "=" * 80)
print("Expected decode characteristics:")
print("  • FlashAttention: ~", DECODE_TOKENS * 32, " calls (50 tokens × ~32 layers)")
print("  • Each FlashAttention call processes 1 query token")
print("  • KV cache size grows from", WARMUP_TOKENS, "to", WARMUP_TOKENS + DECODE_TOKENS, "tokens")
print("=" * 80)
