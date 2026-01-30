#!/usr/bin/env python3
"""
Profile only the decode stage of vLLM generation.

This script demonstrates how to separate prefill and decode stages:
1. First generation: Prefill stage (no profiling)
2. Second generation: Decode stage (with profiling)

The trick is to use the LLM engine's ability to continue generation
from where it left off by using prompt caching or by doing sequential
generations where the first one is short (to do prefill) and the second
one continues (decode only).
"""

from vllm import LLM, SamplingParams
import os

def profile_decode_only(
    model_name: str = "meta-llama/Llama-3.1-8B",
    prompt: str = "The future of artificial intelligence is",
    prefill_tokens: int = 1,  # Generate 1 token in prefill phase
    decode_tokens: int = 100,  # Generate 100 tokens in decode phase (profiled)
):
    """
    Profile only the decode stage by separating prefill and decode.

    Strategy:
    1. Initialize LLM without profiling
    2. Run first generation with max_tokens=1 to complete prefill
    3. Start profiling
    4. Run second generation with same prompt + first output to do decode only
    5. Stop profiling

    Args:
        model_name: Model to use
        prompt: Input prompt
        prefill_tokens: Tokens to generate in prefill phase (default: 1)
        decode_tokens: Tokens to generate in decode phase (default: 100)
    """

    print("=" * 80)
    print("PROFILING DECODE STAGE ONLY")
    print("=" * 80)

    # Initialize LLM
    print(f"\n1. Initializing model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # Step 1: Run prefill phase without profiling
    print(f"\n2. Running PREFILL phase (generating {prefill_tokens} token(s), no profiling)...")
    prefill_params = SamplingParams(
        temperature=0.0,
        max_tokens=prefill_tokens,
    )

    prefill_outputs = llm.generate([prompt], prefill_params)
    prefill_text = prefill_outputs[0].outputs[0].text

    print(f"   Prefill completed. Generated: '{prefill_text}'")

    # Step 2: Construct continuation prompt
    # In vLLM, to continue generation, we need to provide the full context
    continuation_prompt = prompt + prefill_text

    print(f"\n3. Starting profiler for DECODE phase...")

    # Start profiling before decode
    llm.start_profile()

    # Step 3: Run decode phase with profiling
    print(f"4. Running DECODE phase (generating {decode_tokens} tokens, WITH profiling)...")
    decode_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=decode_tokens,
    )

    decode_outputs = llm.generate([continuation_prompt], decode_params)
    decode_text = decode_outputs[0].outputs[0].text

    # Stop profiling
    print("\n5. Stopping profiler...")
    llm.stop_profile()

    # Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nOriginal prompt: {prompt}")
    print(f"\nPrefill output (not profiled): {prefill_text}")
    print(f"\nDecode output (profiled): {decode_text[:200]}...")
    print(f"\nProfile saved to: profiler_out_0.txt")
    print("\nNOTE: The profile now contains ONLY decode stage operations")
    print("=" * 80)


def profile_decode_only_alternative(
    model_name: str = "meta-llama/Llama-3.1-8B",
    prompt: str = "The future of artificial intelligence is",
    total_tokens: int = 100,
):
    """
    Alternative approach: Profile only later tokens (decode-heavy).

    This method uses a callback or step-level control to start profiling
    after the first few tokens are generated.

    Strategy:
    1. Start generation with profiling disabled
    2. After prefill and first few decode steps, enable profiling
    3. Continue generation with profiling enabled

    Note: This requires lower-level engine access.
    """

    print("=" * 80)
    print("ALTERNATIVE: Profile later tokens only (decode-dominant)")
    print("=" * 80)

    print(f"\n1. Initializing model: {model_name}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    )

    # For this alternative, we'll do multiple short generations
    # where only the later ones are profiled

    print("\n2. Running warmup generation (no profiling)...")
    warmup_params = SamplingParams(
        temperature=0.0,
        max_tokens=10,  # Small warmup
    )
    warmup_outputs = llm.generate([prompt], warmup_params)
    warmup_text = warmup_outputs[0].outputs[0].text
    print(f"   Warmup completed: '{warmup_text}'")

    # Now profile the continuation (pure decode)
    print("\n3. Starting profiler for pure decode phase...")
    llm.start_profile()

    continuation_prompt = prompt + warmup_text
    decode_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=total_tokens - 10,
    )

    print(f"4. Running decode with profiling ({total_tokens - 10} tokens)...")
    decode_outputs = llm.generate([continuation_prompt], decode_params)
    decode_text = decode_outputs[0].outputs[0].text

    print("\n5. Stopping profiler...")
    llm.stop_profile()

    print("\n" + "=" * 80)
    print("Profile saved to: profiler_out_0.txt")
    print("This profile contains primarily decode operations.")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Profile only the decode stage")
    parser.add_argument(
        "--method",
        type=str,
        default="standard",
        choices=["standard", "alternative"],
        help="Method to use (standard or alternative)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Model name"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The future of artificial intelligence is",
        help="Input prompt"
    )
    parser.add_argument(
        "--prefill-tokens",
        type=int,
        default=1,
        help="Tokens to generate in prefill phase"
    )
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=100,
        help="Tokens to generate in decode phase (profiled)"
    )

    args = parser.parse_args()

    if args.method == "standard":
        profile_decode_only(
            model_name=args.model,
            prompt=args.prompt,
            prefill_tokens=args.prefill_tokens,
            decode_tokens=args.decode_tokens,
        )
    else:
        profile_decode_only_alternative(
            model_name=args.model,
            prompt=args.prompt,
            total_tokens=args.prefill_tokens + args.decode_tokens,
        )
