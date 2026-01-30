#!/usr/bin/env python3
"""
Analyze vLLM Profiling Traces
Parse and analyze PyTorch profiler traces from vLLM.

Usage:
    python analyze_profile.py [trace_file.json.gz]

If no file specified, looks for latest trace in ./profiler_traces/
"""

import sys
import json
import gzip
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def find_latest_trace(trace_dir: Path = Path("./profiler_traces")) -> Path:
    """Find the most recent trace file."""
    if not trace_dir.exists():
        raise FileNotFoundError(f"Trace directory not found: {trace_dir}")

    trace_files = list(trace_dir.glob("*.json.gz"))
    if not trace_files:
        raise FileNotFoundError(f"No trace files found in {trace_dir}")

    # Return most recent
    return max(trace_files, key=lambda p: p.stat().st_mtime)


def load_trace(trace_path: Path) -> Dict:
    """Load gzipped JSON trace file."""
    print(f"Loading trace: {trace_path}")
    with gzip.open(trace_path, 'rt') as f:
        return json.load(f)


def extract_key_operators(events: List[Dict]) -> Dict[str, List[float]]:
    """Extract key operator timings from trace events."""

    # Key operators we care about
    key_patterns = {
        'flash_attn': [],
        'reshape_and_cache': [],
        'rotary': [],
        'rms_norm': [],
        'gemm': [],
        'cutlass': [],
        'silu': [],
        'sample': [],
        'topk': [],
        'matmul': [],
    }

    operator_times = defaultdict(list)

    for event in events:
        if 'dur' not in event or event.get('dur', 0) <= 0:
            continue

        name = event.get('name', '').lower()
        duration_us = event['dur']

        # Record in operator_times
        operator_times[event.get('name', 'unknown')].append(duration_us)

        # Check against key patterns
        for pattern in key_patterns:
            if pattern in name:
                key_patterns[pattern].append(duration_us)

    return dict(operator_times), key_patterns


def analyze_operators(operator_times: Dict[str, List[float]]) -> List[Tuple[str, Dict]]:
    """Analyze operator statistics."""

    stats = []

    for op_name, times in operator_times.items():
        if not times:
            continue

        total_us = sum(times)
        count = len(times)
        avg_us = total_us / count
        min_us = min(times)
        max_us = max(times)

        stats.append((op_name, {
            'total_ms': total_us / 1000,
            'count': count,
            'avg_us': avg_us,
            'min_us': min_us,
            'max_us': max_us,
        }))

    # Sort by total time
    stats.sort(key=lambda x: x[1]['total_ms'], reverse=True)

    return stats


def print_report(operator_stats: List[Tuple[str, Dict]], key_operators: Dict[str, List[float]]):
    """Print analysis report."""

    print("\n" + "="*100)
    print("KEY OPERATORS ANALYSIS")
    print("="*100)

    total_time_us = 0
    for _, stats in operator_stats:
        total_time_us += stats['total_ms'] * 1000

    print(f"\nTotal profiled time: {total_time_us/1000:.2f} ms")

    # Key operators summary
    print("\n" + "-"*100)
    print("KEY OPERATORS SUMMARY")
    print("-"*100)

    for op_name, times in sorted(key_operators.items(), key=lambda x: sum(x[1]), reverse=True):
        if not times:
            continue

        total_ms = sum(times) / 1000
        count = len(times)
        avg_us = sum(times) / count
        percent = (sum(times) / total_time_us) * 100

        print(f"\n{op_name.upper()}")
        print(f"  Total time: {total_ms:8.2f} ms ({percent:5.2f}%)")
        print(f"  Calls:      {count:8d}")
        print(f"  Avg/call:   {avg_us:8.1f} μs")
        print(f"  Min/Max:    {min(times):8.1f} / {max(times):8.1f} μs")

    # Top 30 operators
    print("\n" + "-"*100)
    print("TOP 30 OPERATORS BY TOTAL TIME")
    print("-"*100)
    print(f"{'Rank':<6} {'Operator':<65} {'Time (ms)':<12} {'%':<8} {'Calls':<8} {'Avg (μs)':<12}")
    print("-"*100)

    for i, (op_name, stats) in enumerate(operator_stats[:30], 1):
        percent = (stats['total_ms'] * 1000 / total_time_us) * 100
        print(
            f"{i:<6} {op_name[:65]:<65} {stats['total_ms']:<12.2f} "
            f"{percent:<8.2f} {stats['count']:<8} {stats['avg_us']:<12.1f}"
        )


def identify_stages(events: List[Dict]) -> Dict[str, Dict]:
    """Try to identify prefill vs decode stages."""

    # Group events by timestamp ranges
    # This is a heuristic - early events are likely prefill, later are decode

    if not events:
        return {'prefill': {}, 'decode': {}}

    # Get events with timestamps
    timed_events = [e for e in events if 'ts' in e and 'dur' in e]
    if not timed_events:
        return {'prefill': {}, 'decode': {}}

    # Sort by timestamp
    timed_events.sort(key=lambda e: e['ts'])

    # Heuristic: first 30% is prefill, rest is decode
    split_idx = int(len(timed_events) * 0.3)

    prefill_events = timed_events[:split_idx]
    decode_events = timed_events[split_idx:]

    # Analyze each stage
    prefill_ops, prefill_key = extract_key_operators(prefill_events)
    decode_ops, decode_key = extract_key_operators(decode_events)

    return {
        'prefill': {
            'operators': analyze_operators(prefill_ops)[:10],
            'key_ops': prefill_key,
        },
        'decode': {
            'operators': analyze_operators(decode_ops)[:10],
            'key_ops': decode_key,
        }
    }


def main():
    # Get trace file
    if len(sys.argv) > 1:
        trace_file = Path(sys.argv[1])
    else:
        trace_file = find_latest_trace()

    # Load and analyze
    trace_data = load_trace(trace_file)
    events = trace_data.get('traceEvents', [])

    print(f"Total events: {len(events)}")

    # Extract operators
    operator_times, key_operators = extract_key_operators(events)
    operator_stats = analyze_operators(operator_times)

    # Print report
    print_report(operator_stats, key_operators)

    # Try to identify stages
    print("\n" + "="*100)
    print("STAGE ANALYSIS (HEURISTIC)")
    print("="*100)

    stages = identify_stages(events)

    for stage_name, stage_data in stages.items():
        if not stage_data.get('operators'):
            continue

        print(f"\n{stage_name.upper()} STAGE - Top 10 Operators:")
        print("-"*100)

        for i, (op_name, stats) in enumerate(stage_data['operators'], 1):
            print(
                f"{i:2d}. {op_name[:60]:<60} {stats['total_ms']:8.2f} ms "
                f"({stats['count']:4d} calls)"
            )

    print("\n" + "="*100)


if __name__ == "__main__":
    main()
