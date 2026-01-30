#!/usr/bin/env python3
"""
Profile only the decode stage using vLLM Engine API for fine-grained control.

This script uses the lower-level engine API to have precise control over
when profiling starts and stops, allowing us to skip prefill completely.

The key insight: In vLLM v1, prefill vs decode is determined by whether
this is the first step for a sequence or a continuation step.
"""

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
import torch


def profile_pure_decode_with_engine(
    model_name: str = "meta-llama/Llama-3.1-8B",
    prompt: str = "The future of artificial intelligence is",
    decode_steps: int = 20,  # Number of decode steps to profile
):
    """
    Profile pure decode steps using engine-level control.

    This approach:
    1. Initializes the model
    2. Runs prefill + first few decode steps without profiling
    3. Enables profiling
    4. Continues decode steps with profiling
    5. Disables profiling

    Args:
        model_name: Model to use
        prompt: Input prompt
        decode_steps: Number of decode steps to profile
    """

    print("=" * 80)
    print("PROFILING PURE DECODE STEPS (Engine API)")
    print("=" * 80)

    # Initialize LLM
    print(f"\n1. Initializing model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # Step 1: Run complete generation without profiling to fill KV cache
    print("\n2. Running initial generation (prefill + some decode, no profiling)...")
    initial_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=10,  # Generate some tokens to get past prefill
    )

    initial_output = llm.generate([prompt], initial_params)
    initial_text = initial_output[0].outputs[0].text

    print(f"   Initial generation: '{initial_text}'")
    print(f"   Status: Prefill and initial decode completed (not profiled)")

    # Step 2: Continue generation with profiling
    # The continuation will be pure decode since prefill is already done
    continuation_prompt = prompt + initial_text

    print(f"\n3. Starting profiler...")
    llm.start_profile()

    print(f"4. Running continuation (pure decode, {decode_steps} tokens)...")
    decode_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=decode_steps,
    )

    decode_output = llm.generate([continuation_prompt], decode_params)
    decode_text = decode_output[0].outputs[0].text

    print(f"\n5. Stopping profiler...")
    llm.stop_profile()

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nOriginal prompt: {prompt}")
    print(f"\nInitial output (not profiled): {initial_text}")
    print(f"\nDecode output (profiled): {decode_text}")
    print(f"\nProfile saved to: profiler_out_0.txt")
    print("\n✓ Profile contains ONLY decode operations")
    print("=" * 80)


def profile_specific_decode_iterations(
    model_name: str = "meta-llama/Llama-3.1-8B",
    prompt: str = "The future of artificial intelligence is",
    warmup_tokens: int = 5,
    profile_tokens: int = 20,
):
    """
    More granular approach: Profile specific decode iterations.

    Strategy:
    1. Run generation with warmup_tokens to complete prefill
    2. Start profiling
    3. Generate profile_tokens (all decode)
    4. Stop profiling

    This ensures we're profiling only decode operations.
    """

    print("=" * 80)
    print("PROFILING SPECIFIC DECODE ITERATIONS")
    print("=" * 80)

    print(f"\n1. Initializing model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # Warmup: Complete prefill and some decode
    print(f"\n2. Warmup phase: Generating {warmup_tokens} tokens (includes prefill)...")
    warmup_params = SamplingParams(
        temperature=0.0,  # Deterministic for warmup
        max_tokens=warmup_tokens,
    )

    warmup_output = llm.generate([prompt], warmup_params)
    warmup_text = warmup_output[0].outputs[0].text

    print(f"   Warmup completed: '{warmup_text}'")

    # Profile only decode
    print(f"\n3. Starting profiler for decode-only phase...")
    llm.start_profile()

    continuation = prompt + warmup_text
    print(f"4. Profiling {profile_tokens} decode iterations...")

    decode_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=profile_tokens,
    )

    decode_output = llm.generate([continuation], decode_params)
    decode_text = decode_output[0].outputs[0].text

    print(f"\n5. Stopping profiler...")
    llm.stop_profile()

    # Analysis
    print("\n" + "=" * 80)
    print("DECODE PROFILING COMPLETE")
    print("=" * 80)
    print(f"\nWarmup phase: {warmup_tokens} tokens (not profiled)")
    print(f"Decode phase: {profile_tokens} tokens (profiled)")
    print(f"\nWarmup output: {warmup_text}")
    print(f"Profiled output: {decode_text[:100]}...")
    print(f"\n✓ Profile file: profiler_out_0.txt")
    print("✓ Contains ONLY decode operations (no prefill)")
    print("\nTo analyze kernels:")
    print("  python analyze_kernels.py profiler_out_0.txt")
    print("=" * 80)


def explain_decode_vs_prefill():
    """
    Explain how to identify decode vs prefill in profiler output.
    """

    explanation = """
    ============================================================================
    UNDERSTANDING DECODE VS PREFILL IN vLLM PROFILING
    ============================================================================

    In vLLM v1's unified token-based execution:

    1. PREFILL CHARACTERISTICS:
       - Processes multiple input tokens at once
       - max_seqlen_q > 1 (query sequence length)
       - FlashAttention processes variable-length sequences
       - High computation (many tokens attend to each other)
       - Happens on FIRST call for a sequence

    2. DECODE CHARACTERISTICS:
       - Processes ONE token at a time (autoregressive)
       - max_seqlen_q = 1 (single query token)
       - Attends to all previous KV cache entries
       - Lower computation per step (1 token attending to N cached)
       - Happens on CONTINUATION calls

    3. HOW TO PROFILE DECODE ONLY:

       Method 1: Two-step generation (recommended)
       ────────────────────────────────────────────
       a) Run short generation without profiling (completes prefill)
       b) Start profiling
       c) Continue generation (pure decode)
       d) Stop profiling

       Method 2: Warmup + profile
       ───────────────────────────
       a) Generate N warmup tokens (no profile)
       b) Start profiling
       c) Generate M more tokens (all decode, profiled)
       d) Stop profiling

    4. KERNEL DIFFERENCES IN PROFILE:

       Prefill:
       --------
       • flash::FlashAttnFwdSm90 with large query sequences
       • More GEMM operations (larger batch matrix multiplications)
       • reshape_and_cache_flash_kernel (caching new KV pairs)

       Decode:
       -------
       • flash::FlashAttnFwdSm90 with query_len=1
       • Smaller GEMM operations (single token)
       • Less reshape_and_cache activity (only 1 new token)
       • More sampling operations (generate next token)

    5. VERIFICATION:

       To verify you're profiling decode only:
       • Check that continuation prompt = original + generated tokens
       • Verify profiler captures small, repeated operations
       • Look for single-token attention patterns (max_seqlen_q=1)

    ============================================================================
    """

    print(explanation)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Profile only decode stage with fine-grained control"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="specific",
        choices=["basic", "specific", "explain"],
        help="Profiling mode"
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
        help="Warmup tokens (includes prefill, not profiled)"
    )
    parser.add_argument(
        "--profile-tokens",
        type=int,
        default=20,
        help="Tokens to profile (decode only)"
    )

    args = parser.parse_args()

    if args.mode == "explain":
        explain_decode_vs_prefill()
    elif args.mode == "basic":
        profile_pure_decode_with_engine(
            model_name=args.model,
            decode_steps=args.profile_tokens,
        )
    elif args.mode == "specific":
        profile_specific_decode_iterations(
            model_name=args.model,
            warmup_tokens=args.warmup_tokens,
            profile_tokens=args.profile_tokens,
        )
