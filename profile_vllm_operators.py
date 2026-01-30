#!/usr/bin/env python3
"""
Complete vLLM Operator Profiling Script
Profiles runtime of all key operators during generation with detailed analysis.

Usage:
    python profile_vllm_operators.py

Output:
    - profiler_traces/worker_0_rank_0.json.gz (raw trace for TensorBoard)
    - operator_profile_summary.txt (human-readable summary)
    - operator_profile_detailed.csv (detailed CSV for analysis)
    - operator_timeline.json (Chrome trace format)
"""

import os
import sys
import time
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any

import torch
import numpy as np

# Enable profiling annotations in vLLM
os.environ["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"

from vllm import LLM, SamplingParams


class OperatorProfiler:
    """Analyzes and reports profiling results for vLLM operators."""

    # Operator categories for grouping
    OPERATOR_CATEGORIES = {
        'attention': [
            'flash_attn', 'flash_fwd', 'attention', 'attn',
            'paged_attention', 'unified_attention'
        ],
        'matmul': [
            'gemm', 'matmul', 'mm', 'cutlass', 'cublas',
            'linear', 'qkv_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'
        ],
        'normalization': [
            'rms_norm', 'layer_norm', 'norm', 'fused_add_rms'
        ],
        'position_embedding': [
            'rotary', 'rope', 'rotary_embedding', 'flashinfer_rotary'
        ],
        'kv_cache': [
            'reshape_and_cache', 'cache_flash', 'slot_mapping'
        ],
        'activation': [
            'silu', 'gelu', 'swish', 'swiglu', 'relu'
        ],
        'sampling': [
            'sample', 'topk', 'topp', 'multinomial', 'softmax',
            'apply_penalty', 'repetition'
        ],
        'quantization': [
            'quant', 'dequant', 'fp8', 'int8', 'scale'
        ],
        'memory': [
            'copy', 'memcpy', 'transpose', 'permute', 'view', 'reshape'
        ],
    }

    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def categorize_operator(self, op_name: str) -> str:
        """Categorize operator by name."""
        op_lower = op_name.lower()

        for category, keywords in self.OPERATOR_CATEGORIES.items():
            if any(kw in op_lower for kw in keywords):
                return category

        return 'other'

    def extract_stage(self, op_name: str, trace_path: List[str]) -> str:
        """Determine if operation is in prefill or decode stage."""
        # Look for hints in the call trace
        trace_str = ' '.join(trace_path).lower()

        # Check for explicit stage markers
        if 'prefill' in trace_str or 'preprocess' in trace_str:
            return 'prefill'
        if 'decode' in trace_str or 'sample' in trace_str:
            return 'decode'

        # Check operator name
        if 'sample' in op_name.lower():
            return 'decode'

        return 'both'

    def analyze_trace_file(self, trace_path: Path) -> Dict[str, Any]:
        """Parse PyTorch profiler trace and extract operator statistics."""

        print(f"Analyzing trace file: {trace_path}")

        # Load trace (it's gzipped JSON)
        import gzip
        with gzip.open(trace_path, 'rt') as f:
            trace_data = json.load(f)

        # Extract events
        events = trace_data.get('traceEvents', [])

        # Group by operator
        operator_stats = defaultdict(lambda: {
            'count': 0,
            'total_time_us': 0,
            'min_time_us': float('inf'),
            'max_time_us': 0,
            'category': 'other',
            'stage': 'both',
            'calls': []
        })

        for event in events:
            if event.get('cat') != 'kernel' and 'dur' not in event:
                continue

            name = event.get('name', '')
            duration_us = event.get('dur', 0)

            if duration_us <= 0:
                continue

            # Get category
            category = self.categorize_operator(name)

            # Update stats
            stats = operator_stats[name]
            stats['count'] += 1
            stats['total_time_us'] += duration_us
            stats['min_time_us'] = min(stats['min_time_us'], duration_us)
            stats['max_time_us'] = max(stats['max_time_us'], duration_us)
            stats['category'] = category
            stats['calls'].append(duration_us)

        return dict(operator_stats)

    def generate_summary_report(
        self,
        operator_stats: Dict[str, Any],
        total_time_ms: float
    ) -> str:
        """Generate human-readable summary report."""

        lines = []
        lines.append("=" * 100)
        lines.append("vLLM OPERATOR PROFILING SUMMARY")
        lines.append("=" * 100)
        lines.append(f"\nTotal Generation Time: {total_time_ms:.2f} ms")
        lines.append(f"Number of Unique Operators: {len(operator_stats)}")

        # Group by category
        category_stats = defaultdict(lambda: {
            'total_time_us': 0,
            'count': 0,
            'operators': []
        })

        for op_name, stats in operator_stats.items():
            cat = stats['category']
            category_stats[cat]['total_time_us'] += stats['total_time_us']
            category_stats[cat]['count'] += stats['count']
            category_stats[cat]['operators'].append((op_name, stats))

        # Sort categories by time
        sorted_categories = sorted(
            category_stats.items(),
            key=lambda x: x[1]['total_time_us'],
            reverse=True
        )

        lines.append("\n" + "=" * 100)
        lines.append("BREAKDOWN BY CATEGORY")
        lines.append("=" * 100)

        for category, stats in sorted_categories:
            cat_time_ms = stats['total_time_us'] / 1000
            cat_percent = (stats['total_time_us'] / (total_time_ms * 1000)) * 100

            lines.append(f"\n{category.upper()}")
            lines.append(f"  Total Time: {cat_time_ms:.2f} ms ({cat_percent:.1f}%)")
            lines.append(f"  Total Calls: {stats['count']}")
            lines.append(f"  Avg Time per Call: {cat_time_ms / stats['count']:.3f} ms")

            # Top 5 operators in this category
            top_ops = sorted(
                stats['operators'],
                key=lambda x: x[1]['total_time_us'],
                reverse=True
            )[:5]

            if top_ops:
                lines.append(f"  Top Operators:")
                for i, (op_name, op_stats) in enumerate(top_ops, 1):
                    op_time_ms = op_stats['total_time_us'] / 1000
                    op_percent = (op_stats['total_time_us'] / stats['total_time_us']) * 100
                    avg_time_us = op_stats['total_time_us'] / op_stats['count']
                    lines.append(
                        f"    {i}. {op_name[:70]:<70} "
                        f"{op_time_ms:8.2f} ms ({op_percent:5.1f}%) "
                        f"[{op_stats['count']:4d} calls, avg: {avg_time_us:7.1f} μs]"
                    )

        # Top 20 operators overall
        lines.append("\n" + "=" * 100)
        lines.append("TOP 20 OPERATORS BY TOTAL TIME")
        lines.append("=" * 100)
        lines.append(
            f"{'Rank':<6} {'Operator':<60} {'Time (ms)':<12} "
            f"{'% Total':<10} {'Calls':<8} {'Avg (μs)':<12}"
        )
        lines.append("-" * 100)

        sorted_ops = sorted(
            operator_stats.items(),
            key=lambda x: x[1]['total_time_us'],
            reverse=True
        )[:20]

        for i, (op_name, stats) in enumerate(sorted_ops, 1):
            time_ms = stats['total_time_us'] / 1000
            percent = (stats['total_time_us'] / (total_time_ms * 1000)) * 100
            avg_us = stats['total_time_us'] / stats['count']

            lines.append(
                f"{i:<6} {op_name[:60]:<60} {time_ms:<12.2f} "
                f"{percent:<10.2f} {stats['count']:<8} {avg_us:<12.1f}"
            )

        return "\n".join(lines)

    def generate_csv_report(self, operator_stats: Dict[str, Any]) -> str:
        """Generate detailed CSV report."""
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Operator Name',
            'Category',
            'Total Time (ms)',
            'Call Count',
            'Avg Time (μs)',
            'Min Time (μs)',
            'Max Time (μs)',
            'Std Dev (μs)'
        ])

        # Sort by total time
        sorted_ops = sorted(
            operator_stats.items(),
            key=lambda x: x[1]['total_time_us'],
            reverse=True
        )

        for op_name, stats in sorted_ops:
            calls = stats['calls']
            std_dev = np.std(calls) if len(calls) > 1 else 0

            writer.writerow([
                op_name,
                stats['category'],
                stats['total_time_us'] / 1000,
                stats['count'],
                stats['total_time_us'] / stats['count'],
                stats['min_time_us'],
                stats['max_time_us'],
                std_dev
            ])

        return output.getvalue()


def main():
    """Main profiling execution."""

    print("=" * 100)
    print("vLLM Operator Profiling Script")
    print("=" * 100)

    # Configuration
    model_name = "meta-llama/Llama-3.1-8B"
    prompts = [
        "The future of artificial intelligence is",
        "Once upon a time in a galaxy far away",
    ]
    max_tokens = 20
    temperature = 0.8

    # Output directory
    output_dir = Path("./profiling_results")
    output_dir.mkdir(exist_ok=True)

    print(f"\nModel: {model_name}")
    print(f"Prompts: {len(prompts)}")
    print(f"Max tokens per prompt: {max_tokens}")
    print(f"Output directory: {output_dir}")

    # Initialize profiler
    profiler = OperatorProfiler(output_dir=output_dir)

    # Create LLM instance
    print("\n" + "-" * 100)
    print("Initializing vLLM...")
    print("-" * 100)

    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enforce_eager=True,  # Disable CUDA graphs for clearer profiling
    )

    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )

    # Warmup run (not profiled)
    print("\n" + "-" * 100)
    print("Warmup run...")
    print("-" * 100)
    _ = llm.generate(prompts[:1], sampling_params)

    # Profiled run
    print("\n" + "-" * 100)
    print("Starting profiled generation...")
    print("-" * 100)

    # Start profiling
    llm.start_profile()

    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    # Stop profiling
    llm.stop_profile()

    total_time_ms = (end_time - start_time) * 1000

    print(f"\nGeneration completed in {total_time_ms:.2f} ms")

    # Print generated outputs
    print("\n" + "-" * 100)
    print("Generated Outputs:")
    print("-" * 100)
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt {i+1}: {prompt}")
        print(f"Generated: {generated_text}")

    # Analyze profiling results
    print("\n" + "-" * 100)
    print("Analyzing profiling results...")
    print("-" * 100)

    # Find trace file
    trace_dir = Path("./profiler_traces")
    if not trace_dir.exists():
        print(f"ERROR: Trace directory not found: {trace_dir}")
        print("Make sure profiling was enabled correctly.")
        return

    # Look for the trace file
    trace_files = list(trace_dir.glob("*.json.gz"))
    if not trace_files:
        print(f"ERROR: No trace files found in {trace_dir}")
        return

    trace_file = trace_files[0]
    print(f"Found trace file: {trace_file}")

    # Analyze trace
    operator_stats = profiler.analyze_trace_file(trace_file)

    # Generate summary report
    summary = profiler.generate_summary_report(operator_stats, total_time_ms)
    print("\n" + summary)

    # Save summary to file
    summary_file = output_dir / "operator_profile_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_file}")

    # Generate CSV report
    csv_content = profiler.generate_csv_report(operator_stats)
    csv_file = output_dir / "operator_profile_detailed.csv"
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    print(f"Detailed CSV saved to: {csv_file}")

    # Copy trace file for TensorBoard
    import shutil
    trace_copy = output_dir / "operator_timeline.json.gz"
    shutil.copy(trace_file, trace_copy)
    print(f"Timeline trace saved to: {trace_copy}")

    print("\n" + "=" * 100)
    print("PROFILING COMPLETE")
    print("=" * 100)
    print("\nTo view in TensorBoard:")
    print(f"  tensorboard --logdir={trace_dir}")
    print("  Open http://localhost:6006 in your browser")
    print("\nTo view timeline in Chrome:")
    print(f"  1. Open chrome://tracing in Chrome")
    print(f"  2. Load file: {trace_copy}")
    print("=" * 100)


if __name__ == "__main__":
    main()
