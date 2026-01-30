#!/usr/bin/env python3
"""
Analyze vLLM Profiler Output - Kernel Categorization
Categorizes and analyzes CUDA kernels from profiler output files.

Usage:
    python analyze_kernels.py [profiler_out_0.txt]

If no file specified, looks for profiler_out_0.txt in current directory.
"""

import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


class KernelAnalyzer:
    """Analyzes and categorizes CUDA kernels from vLLM profiler output."""

    # Kernel categorization patterns
    KERNEL_CATEGORIES = {
        'flashattn': {
            'patterns': [
                'flash::FlashAttnFwdSm90',
                'flash::FlashAttnFwdCombine',
                'flash::prepare_varlen_num_blocks_kernel',
                'flash_attn_fwd',
                'flash_fwd',
            ],
            'description': 'FlashAttention Kernels',
        },
        'gemm': {
            'patterns': [
                'nvjet_tst_',
                'cutlass.*gemm',
                'cublas.*gemm',
                'volta.*gemm',
                'ampere.*gemm',
                'hopper.*gemm',
                'sm[0-9]+.*gemm',
            ],
            'description': 'Matrix Multiplication (GEMM) Kernels',
        },
        'normalization': {
            'patterns': [
                'triton_red_fused.*rms.*norm',
                'rms_norm_kernel',
                'layer_norm_kernel',
                'fused_add_rms_norm',
            ],
            'description': 'Normalization Kernels',
        },
        'activation': {
            'patterns': [
                'triton_poi_fused_mul_silu',
                'silu_kernel',
                'gelu_kernel',
                'swiglu_kernel',
            ],
            'description': 'Activation Function Kernels',
        },
        'kv_cache': {
            'patterns': [
                'vllm::reshape_and_cache_flash_kernel',
                'reshape_and_cache',
            ],
            'description': 'KV Cache Operations',
        },
        'sampling': {
            'patterns': [
                'DeviceRadixSort',
                'softmax',
                'topk',
                'multinomial',
            ],
            'description': 'Sampling and Sorting Kernels',
        },
        'elementwise': {
            'patterns': [
                'triton_poi_fused',
                'triton_red_fused',
                'elementwise_kernel',
                'vectorized_elementwise',
            ],
            'description': 'Element-wise Operations',
        },
        'memory': {
            'patterns': [
                'Memcpy',
                'Memset',
                'copy_kernel',
            ],
            'description': 'Memory Operations',
        },
        'reduction': {
            'patterns': [
                'DeviceScan',
                'reduce_kernel',
                'cumsum',
            ],
            'description': 'Reduction Operations',
        },
    }

    # Wrappers to exclude (not actual kernels)
    WRAPPER_PATTERNS = [
        r'^aten::',
        r'^_vllm_fa[0-9]_C::',
        r'^_C_cache_ops::',
        r'^vllm::unified_',
        r'^gpu_model_runner:',
        r'^schedule:',
        r'^cuda[A-Z]',  # CUDA API calls
        r'Unrecognized',
        r'bytecode',
    ]

    def __init__(self):
        self.kernels = []

    def is_wrapper(self, name: str) -> bool:
        """Check if this is a wrapper/API call rather than actual kernel."""
        for pattern in self.WRAPPER_PATTERNS:
            if re.search(pattern, name):
                return True
        return False

    def is_cuda_kernel(self, name: str) -> bool:
        """Check if this is an actual CUDA kernel."""
        # CUDA kernels typically have certain patterns
        kernel_indicators = [
            'void ',           # CUDA kernel signature
            'cutlass::',       # CUTLASS kernels
            'flash::',         # FlashAttention kernels
            'vllm::',          # vLLM custom kernels
            'triton_',         # Triton kernels
            'nvjet_',          # NVIDIA kernels
            'at::native::',    # PyTorch native kernels
            'cublas',          # cuBLAS kernels
        ]

        if self.is_wrapper(name):
            return False

        for indicator in kernel_indicators:
            if indicator in name:
                return True

        # Also check for memory operations
        if 'Memcpy' in name or 'Memset' in name:
            return True

        return False

    def categorize_kernel(self, name: str) -> str:
        """Categorize kernel by type."""
        for category, config in self.KERNEL_CATEGORIES.items():
            for pattern in config['patterns']:
                if re.search(pattern, name, re.IGNORECASE):
                    return category
        return 'other'

    def parse_profiler_line(self, line: str) -> Tuple[str, float, int] | None:
        """Parse a line from profiler output."""
        # The format has a very long name followed by columns
        # We need to find the Self CUDA column and # of Calls column

        # Skip if line is too short
        if len(line.strip()) < 50:
            return None

        # Extract the name (everything before the first number-like pattern)
        # The Self CPU % column starts the numeric data
        # Pattern: name followed by percentage values
        match = re.search(r'^(.+?)\s+(\d+\.\d+%|\d+%)', line)
        if not match:
            return None

        name = match.group(1).strip()

        # Skip if not a CUDA kernel
        if not self.is_cuda_kernel(name):
            return None

        # Find Self CUDA column (6th numeric column)
        # Pattern: extract all numeric values with units
        numeric_pattern = r'(\d+\.\d+(?:ms|us|ns)|\d+(?:ms|us|ns)|\d+\.\d+%|\d+%|\d+\.\d+us|\d+us)'
        matches = re.findall(numeric_pattern, line)

        if len(matches) < 8:
            return None

        try:
            # Self CUDA is typically column index 5 (0-indexed)
            cuda_time_str = matches[5]

            # Parse time with units (ms, us, ns)
            if 'ms' in cuda_time_str:
                cuda_time_us = float(cuda_time_str.replace('ms', '')) * 1000
            elif 'us' in cuda_time_str:
                cuda_time_us = float(cuda_time_str.replace('us', ''))
            elif 'ns' in cuda_time_str:
                cuda_time_us = float(cuda_time_str.replace('ns', '')) / 1000
            else:
                return None

            # # of Calls is last column - find last pure number
            calls_pattern = r'\s+(\d+)\s*$'
            calls_match = re.search(calls_pattern, line)
            if calls_match:
                calls = int(calls_match.group(1))
            else:
                calls = 1

            return (name, cuda_time_us, calls)
        except (ValueError, IndexError) as e:
            return None

    def parse_file(self, filepath: Path) -> None:
        """Parse profiler output file."""
        print(f"Parsing: {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # Skip header lines and parse data
        in_data = False
        for line in lines:
            if '----' in line:
                in_data = True
                continue

            if not in_data:
                continue

            if line.strip().startswith('Self CPU time total'):
                break

            result = self.parse_profiler_line(line)
            if result:
                self.kernels.append(result)

        print(f"Found {len(self.kernels)} CUDA kernels\n")

    def get_kernel_explanation(self, name: str, category: str) -> str:
        """Get brief explanation for a kernel."""
        explanations = {
            'flash::FlashAttnFwdSm90': 'Main FlashAttention forward computation (Ampere/Hopper optimized)',
            'flash::FlashAttnFwdCombine': 'Combines partial results from split-K attention',
            'flash::prepare_varlen_num_blocks_kernel': 'Prepares metadata for variable-length sequences',
            'nvjet_tst_128x16': 'NVIDIA optimized GEMM for QKV/MLP projections',
            'nvjet_tst_64x16': 'NVIDIA optimized GEMM (different tile size)',
            'nvjet_tst_384x8': 'NVIDIA optimized GEMM (wider tiles)',
            'cutlass': 'CUTLASS high-performance GEMM kernel',
            'reshape_and_cache_flash_kernel': 'Writes K,V tensors to paged KV cache',
            'triton_red_fused.*rms': 'Fused RMSNorm (reduction + normalization)',
            'triton_poi_fused_mul_silu': 'Fused SwiGLU activation (mul + silu)',
            'DeviceRadixSort': 'Radix sort for top-k sampling',
            'softmax': 'Softmax for sampling probabilities',
            'Memcpy': 'Memory copy operation',
            'Memset': 'Memory initialization',
        }

        # Try exact match first
        if name in explanations:
            return explanations[name]

        # Try pattern match
        for pattern, explanation in explanations.items():
            if pattern in name:
                return explanation

        # Category-based default explanations
        category_defaults = {
            'flashattn': 'FlashAttention kernel',
            'gemm': 'Matrix multiplication for linear layers',
            'normalization': 'Layer normalization',
            'activation': 'Activation function',
            'kv_cache': 'KV cache operation',
            'sampling': 'Token sampling operation',
            'elementwise': 'Element-wise operation',
            'memory': 'Memory operation',
            'reduction': 'Reduction operation',
        }

        return category_defaults.get(category, 'Other CUDA kernel')

    def shorten_kernel_name(self, name: str, max_len: int = 80) -> str:
        """Shorten kernel name for display."""
        if len(name) <= max_len:
            return name

        # Try to keep important parts
        if 'cutlass::device_kernel<' in name:
            # Extract key info from CUTLASS kernel
            match = re.search(r'cutlass::device_kernel<([^<>]+)<', name)
            if match:
                return f"cutlass::{match.group(1)}..."

        if 'void ' in name:
            # Extract function name from signature
            match = re.search(r'void ([^(]+)\(', name)
            if match:
                func = match.group(1).split('::')[-1]
                return f"void {func}(...)"

        # Default truncation
        return name[:max_len-3] + "..."

    def analyze(self) -> None:
        """Perform categorization and analysis."""
        if not self.kernels:
            print("No kernels found!")
            return

        # Categorize all kernels
        categorized = defaultdict(list)

        for name, time_us, calls in self.kernels:
            category = self.categorize_kernel(name)
            categorized[category].append((name, time_us, calls))

        # Calculate totals
        total_time_us = sum(time_us for _, time_us, _ in self.kernels)

        print("=" * 120)
        print("CUDA KERNEL ANALYSIS - CATEGORIZED BY TYPE")
        print("=" * 120)
        print(f"\nTotal CUDA Time: {total_time_us/1000:.3f} ms")
        print(f"Total Kernels: {len(self.kernels)}")
        print()

        # 1. FlashAttention Kernels
        print("=" * 120)
        print("1. FLASHATTENTION KERNELS")
        print("=" * 120)

        fa_kernels = categorized.get('flashattn', [])
        if fa_kernels:
            fa_total = sum(time_us for _, time_us, _ in fa_kernels)
            fa_pct = (fa_total / total_time_us) * 100

            print(f"\nTotal FlashAttention Time: {fa_total/1000:.3f} ms ({fa_pct:.2f}% of total)")
            print(f"Number of FA Kernels: {len(fa_kernels)}")
            print()

            # Sort by time
            fa_kernels.sort(key=lambda x: x[1], reverse=True)

            print(f"{'Kernel Name':<80} {'Time (ms)':<12} {'Calls':<8} {'Avg (μs)':<12} {'%':<8}")
            print("-" * 120)

            for name, time_us, calls in fa_kernels:
                short_name = self.shorten_kernel_name(name, 80)
                pct = (time_us / total_time_us) * 100
                avg_us = time_us / calls
                print(f"{short_name:<80} {time_us/1000:<12.3f} {calls:<8} {avg_us:<12.1f} {pct:<8.2f}")

                # Print explanation
                explanation = self.get_kernel_explanation(name, 'flashattn')
                print(f"  └─> {explanation}")
                print()

            print(f"\n{'FLASHATTENTION SUBTOTAL:':<80} {fa_total/1000:<12.3f} ms ({fa_pct:.2f}%)")
        else:
            print("No FlashAttention kernels found.")

        # 2. Non-FlashAttention Kernels
        print("\n" + "=" * 120)
        print("2. NON-FLASHATTENTION KERNELS (Grouped by Category)")
        print("=" * 120)

        # Calculate non-FA total
        non_fa_total = total_time_us - sum(time_us for _, time_us, _ in fa_kernels)
        non_fa_pct = (non_fa_total / total_time_us) * 100

        print(f"\nTotal Non-FlashAttention Time: {non_fa_total/1000:.3f} ms ({non_fa_pct:.2f}% of total)")
        print()

        # Sort categories by total time
        category_totals = []
        for category, kernels in categorized.items():
            if category == 'flashattn':
                continue
            cat_total = sum(time_us for _, time_us, _ in kernels)
            category_totals.append((category, cat_total, kernels))

        category_totals.sort(key=lambda x: x[1], reverse=True)

        # Display each category
        for category, cat_total, kernels in category_totals:
            cat_pct = (cat_total / total_time_us) * 100
            cat_desc = self.KERNEL_CATEGORIES.get(category, {}).get('description', category.upper())

            print("-" * 120)
            print(f"\n{cat_desc.upper()}")
            print(f"Category Total: {cat_total/1000:.3f} ms ({cat_pct:.2f}% of total)")
            print()

            # Sort kernels within category
            kernels.sort(key=lambda x: x[1], reverse=True)

            print(f"{'Kernel Name':<80} {'Time (ms)':<12} {'Calls':<8} {'Avg (μs)':<12} {'%':<8}")
            print("-" * 120)

            # Show top 5 kernels per category
            for name, time_us, calls in kernels[:5]:
                short_name = self.shorten_kernel_name(name, 80)
                pct = (time_us / total_time_us) * 100
                avg_us = time_us / calls
                print(f"{short_name:<80} {time_us/1000:<12.3f} {calls:<8} {avg_us:<12.1f} {pct:<8.2f}")

                explanation = self.get_kernel_explanation(name, category)
                print(f"  └─> {explanation}")
                print()

            if len(kernels) > 5:
                print(f"  ... and {len(kernels) - 5} more kernels in this category")
                print()

        print("\n" + "=" * 120)
        print("SUMMARY")
        print("=" * 120)

        # Summary table
        print(f"\n{'Category':<30} {'Time (ms)':<15} {'% of Total':<15} {'# Kernels':<12}")
        print("-" * 120)

        if fa_kernels:
            fa_total = sum(time_us for _, time_us, _ in fa_kernels)
            fa_pct = (fa_total / total_time_us) * 100
            print(f"{'FlashAttention':<30} {fa_total/1000:<15.3f} {fa_pct:<15.2f} {len(fa_kernels):<12}")

        for category, cat_total, kernels in category_totals:
            cat_pct = (cat_total / total_time_us) * 100
            cat_desc = self.KERNEL_CATEGORIES.get(category, {}).get('description', category)
            print(f"{cat_desc:<30} {cat_total/1000:<15.3f} {cat_pct:<15.2f} {len(kernels):<12}")

        print("-" * 120)
        print(f"{'TOTAL':<30} {total_time_us/1000:<15.3f} {100.0:<15.2f} {len(self.kernels):<12}")
        print()


def main():
    """Main entry point."""

    # Get file path
    if len(sys.argv) > 1:
        filepath = Path(sys.argv[1])
    else:
        # Look for profiler_out_0.txt in current directory
        filepath = Path("profiler_out_0.txt")
        if not filepath.exists():
            # Try in profiler_traces directory
            filepath = Path("profiler_traces/profiler_out_0.txt")

    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        print("\nUsage: python analyze_kernels.py [profiler_out_0.txt]")
        sys.exit(1)

    # Analyze
    analyzer = KernelAnalyzer()
    analyzer.parse_file(filepath)
    analyzer.analyze()


if __name__ == "__main__":
    main()
