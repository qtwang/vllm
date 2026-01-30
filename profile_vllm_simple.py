#!/usr/bin/env python3
"""
Simple vLLM Profiling Script - Quick Start
Minimal code to profile vLLM key operators.

Usage:
    python profile_vllm_simple.py

Output:
    - profiler_traces/ directory with trace files
    - Console output with top operators
"""

import os
import time
from vllm import LLM, SamplingParams

# Enable profiling annotations
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"

# Initialize model
print("Loading model...")
llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
)

# Prepare input
prompts = ["The future of AI is"]
sampling_params = SamplingParams(max_tokens=20, temperature=0.8)

# Warmup
print("Warmup...")
_ = llm.generate(prompts, sampling_params)

# Profile
print("\nProfiling...")
llm.start_profile()

start = time.time()
outputs = llm.generate(prompts, sampling_params)
elapsed = time.time() - start

llm.stop_profile()

# Results
print(f"\nGeneration time: {elapsed*1000:.2f} ms")
print(f"Output: {outputs[0].outputs[0].text}")

print("\n" + "="*80)
print("Profiling complete!")
print("="*80)
print("\nTrace saved to: ./profiler_traces/")
print("\nView in TensorBoard:")
print("  tensorboard --logdir=./profiler_traces")
print("  Open http://localhost:6006")
print("="*80)
