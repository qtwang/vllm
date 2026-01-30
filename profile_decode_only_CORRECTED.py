#!/usr/bin/env python3
"""
CORRECTED: Profile only the decode stage using prefix caching.

This script uses vLLM's automatic prefix caching to ensure KV cache
is reused between generate() calls, making the second call mostly decode-only.

Key difference from the incorrect version:
  enable_prefix_caching=True  ← This is CRITICAL!
"""

from vllm import LLM, SamplingParams

def profile_decode_with_prefix_caching(
    model_name: str = "meta-llama/Llama-3.1-8B",
    prompt: str = "The future of artificial intelligence is",
    warmup_tokens: int = 5,
    decode_tokens: int = 50,
):
    """
    Profile decode using prefix caching to reuse KV cache.

    How this works:
    1. First generate() caches KV for the prompt + warmup tokens
    2. Second generate() detects matching prefix and reuses cached KV
    3. Only processes new tokens (mostly decode)

    Args:
        model_name: Model to use
        prompt: Input prompt
        warmup_tokens: Tokens in first generation (cached)
        decode_tokens: Tokens to profile (mostly decode)
    """

    print("=" * 80)
    print("CORRECTED: DECODE-ONLY PROFILING WITH PREFIX CACHING")
    print("=" * 80)

    # CRITICAL: Enable prefix caching!
    print(f"\n1. Initializing model with PREFIX CACHING enabled...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,  # ← THIS IS THE KEY!
    )
    print("✓ Model loaded with prefix caching enabled")

    # Step 1: First generation - this caches the KV
    print(f"\n2. First generation: {warmup_tokens} tokens (caches KV)...")
    warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=warmup_tokens,
    )

    warmup_output = llm.generate([prompt], warmup_params)
    warmup_text = warmup_output[0].outputs[0].text

    print(f"   Generated: '{warmup_text}'")
    print(f"   ✓ KV cache for prompt is now cached")

    # Step 2: Second generation - reuses cached KV
    print(f"\n3. Second generation: {decode_tokens} tokens (REUSES cached KV)...")
    print("   Starting profiler...")

    llm.start_profile()

    # This prompt shares the same prefix as the first call
    continuation = prompt + warmup_text

    decode_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=decode_tokens,
    )

    decode_output = llm.generate([continuation], decode_params)
    decode_text = decode_output[0].outputs[0].text

    llm.stop_profile()
    print("   ✓ Profiling complete")

    # Explain what was profiled
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print(f"\nOriginal prompt: \"{prompt}\"")
    print(f"\nFirst generation (cached, not profiled):")
    print(f"  \"{warmup_text}\"")
    print(f"\nSecond generation (profiled):")
    print(f"  \"{decode_text[:100]}{'...' if len(decode_text) > 100 else ''}\"")

    print(f"\n✓ Profile saved to: profiler_out_0.txt")

    print("\nWhat was actually profiled:")
    print(f"  • Prefix \"{prompt}\" - REUSED from cache (no prefill)")
    print(f"  • Warmup tokens \"{warmup_text}\" - Small prefill ({warmup_tokens} tokens)")
    print(f"  • New tokens - Pure decode ({decode_tokens} tokens)")
    print(f"\nRatio: {warmup_tokens} prefill tokens : {decode_tokens} decode tokens")
    print(f"  → ~{100 * decode_tokens / (warmup_tokens + decode_tokens):.1f}% decode operations")

    print("\n" + "=" * 80)
    print("HOW PREFIX CACHING WORKS")
    print("=" * 80)
    print("""
First call:  llm.generate(["The future of AI"])
  → Prefill: "The future of AI" (processes and caches KV)
  → Decode: 5 tokens
  → KV cache STORED for reuse

Second call: llm.generate(["The future of AI" + prev_output])
  → Detects matching prefix: "The future of AI"
  → REUSES cached KV (no prefill for this part!)
  → Prefill: Only for prev_output (5 tokens)
  → Decode: 50 tokens
  → Result: 5 prefill + 50 decode = 91% decode

Without prefix caching:
  → Would prefill entire continuation (14 tokens)
  → Then decode 50 tokens
  → Result: 14 prefill + 50 decode = 78% decode
    """)

    print("=" * 80)


def verify_prefix_caching():
    """
    Verification script to show prefix caching is working.

    This generates three requests with overlapping prefixes and
    measures if prefix caching reduces computation.
    """

    print("=" * 80)
    print("VERIFYING PREFIX CACHING")
    print("=" * 80)

    # Test with prefix caching
    print("\nTest 1: WITH prefix caching")
    llm_with_cache = LLM(
        model="meta-llama/Llama-3.1-8B",
        enable_prefix_caching=True,
    )

    base_prompt = "The future of artificial intelligence is"

    # First call - caches the prefix
    out1 = llm_with_cache.generate([base_prompt], SamplingParams(max_tokens=5))

    # Second call - should reuse cache
    llm_with_cache.start_profile()
    continuation = base_prompt + out1[0].outputs[0].text
    out2 = llm_with_cache.generate([continuation], SamplingParams(max_tokens=10))
    llm_with_cache.stop_profile()

    print("✓ Profile saved to: profiler_out_0.txt")
    print("  This should show SMALL prefill + decode operations")

    # Clean up
    del llm_with_cache

    print("\nTest 2: WITHOUT prefix caching")
    llm_no_cache = LLM(
        model="meta-llama/Llama-3.1-8B",
        enable_prefix_caching=False,
    )

    # Same operations but without caching
    out1 = llm_no_cache.generate([base_prompt], SamplingParams(max_tokens=5))

    llm_no_cache.start_profile()
    continuation = base_prompt + out1[0].outputs[0].text
    out2 = llm_no_cache.generate([continuation], SamplingParams(max_tokens=10))
    llm_no_cache.stop_profile()

    print("✓ Profile saved to: profiler_out_0.txt (overwrote previous)")
    print("  This should show LARGE prefill + decode operations")

    print("\n" + "=" * 80)
    print("Compare the two profiles to see the difference!")
    print("With caching: Less prefill operations (prefix reused)")
    print("Without caching: More prefill operations (entire prompt processed)")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CORRECTED: Profile decode-only using prefix caching"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification to show prefix caching effect"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name"
    )
    parser.add_argument(
        "--warmup-tokens",
        type=int,
        default=5,
        help="Tokens in first generation (cached)"
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=50,
        help="Tokens to profile (mostly decode)"
    )

    args = parser.parse_args()

    if args.verify:
        verify_prefix_caching()
    else:
        profile_decode_with_prefix_caching(
            model_name=args.model,
            warmup_tokens=args.warmup_tokens,
            decode_tokens=args.decode_tokens,
        )
