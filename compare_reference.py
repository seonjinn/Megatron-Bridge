#!/usr/bin/env python3
"""
Compare experiment results with reference values.

This script collects results from experiments and compares them with
the reference values from the official performance table.

Usage:
    python3 compare_reference.py --sweep-dir ./exp_logs/sweeps/ref_gb200_YYYYMMDD_HHMMSS
"""

import argparse
import json
import re
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class ExperimentResult:
    """Data class for experiment results."""
    exp_name: str = ""
    model_name: str = ""
    model_size: str = ""
    task: str = "pretrain"
    
    # Hardware config
    num_gpus: int = 0
    num_nodes: int = 0
    gpu_type: str = "gb200"
    
    # Parallelism config
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    dp: int = 1
    vp: int = 1
    
    # Training config
    seq_len: int = 0
    mbs: int = 1
    gbs: int = 1
    
    # Measured metrics
    step_time_sec: float = 0.0
    iteration_time_ms: float = 0.0
    tokens_per_sec: float = 0.0
    tokens_per_sec_per_gpu: float = 0.0
    tflops_per_gpu: float = 0.0
    mfu: float = 0.0
    samples_per_sec: float = 0.0
    
    # Reference metrics
    ref_step_time: float = 0.0
    ref_tflops: float = 0.0
    ref_mfu: float = 0.0
    ref_tokens_sec: float = 0.0
    ref_tokens_sec_gpu: float = 0.0
    
    # Status
    status: str = "unknown"
    exp_dir: str = ""


def parse_log_file(log_path: Path, result: ExperimentResult) -> ExperimentResult:
    """Parse a log file to extract metrics."""
    if not log_path.exists():
        return result
    
    try:
        content = log_path.read_text(errors='ignore')
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return result
    
    # Check for OOM
    oom_patterns = [
        re.compile(r'CUDA out of memory', re.IGNORECASE),
        re.compile(r'OutOfMemoryError', re.IGNORECASE),
        re.compile(r'out of memory', re.IGNORECASE),
    ]
    if any(pattern.search(content) for pattern in oom_patterns):
        result.status = 'oom'
        return result
    
    # Check for errors
    error_patterns = [
        re.compile(r'Error|Exception|Traceback', re.IGNORECASE),
        re.compile(r'FAILED|CANCELLED', re.IGNORECASE),
    ]
    
    # Extract metrics from throughput logging
    # Pattern: "iteration-time": X.XXX (in seconds)
    iter_time_pattern = re.compile(r'"iteration-time":\s*([\d.]+)')
    matches = iter_time_pattern.findall(content)
    if matches:
        # Take average of last 10 iterations
        times = [float(t) for t in matches[-10:]]
        result.step_time_sec = sum(times) / len(times)
        result.iteration_time_ms = result.step_time_sec * 1000
    
    # Pattern: "tokens-per-sec": X.XXX
    tokens_pattern = re.compile(r'"tokens-per-sec":\s*([\d.]+)')
    matches = tokens_pattern.findall(content)
    if matches:
        tokens = [float(t) for t in matches[-10:]]
        result.tokens_per_sec = sum(tokens) / len(tokens)
    
    # Pattern: "tokens-per-sec-per-gpu": X.XXX
    tokens_gpu_pattern = re.compile(r'"tokens-per-sec-per-gpu":\s*([\d.]+)')
    matches = tokens_gpu_pattern.findall(content)
    if matches:
        tokens = [float(t) for t in matches[-10:]]
        result.tokens_per_sec_per_gpu = sum(tokens) / len(tokens)
    
    # Pattern: "tflops-per-gpu": X.XXX or "TFLOP/s/GPU": X.XXX
    tflops_pattern = re.compile(r'(?:"tflops-per-gpu"|TFLOP/s/GPU):\s*([\d.]+)')
    matches = tflops_pattern.findall(content)
    if matches:
        tflops = [float(t) for t in matches[-10:]]
        result.tflops_per_gpu = sum(tflops) / len(tflops)
    
    # Pattern: "samples-per-sec": X.XXX
    samples_pattern = re.compile(r'"samples-per-sec":\s*([\d.]+)')
    matches = samples_pattern.findall(content)
    if matches:
        samples = [float(s) for s in matches[-10:]]
        result.samples_per_sec = sum(samples) / len(samples)
    
    # Determine status
    if result.tokens_per_sec_per_gpu > 0:
        result.status = 'completed'
    elif any(p.search(content) for p in error_patterns):
        result.status = 'failed'
    elif 'Running' in content or 'iteration' in content.lower():
        result.status = 'running'
    
    return result


def parse_wandb_summary(exp_dir: Path, result: ExperimentResult) -> ExperimentResult:
    """Parse WandB summary file for metrics."""
    # Find wandb summary file
    wandb_dirs = list(exp_dir.glob("**/wandb/latest-run/files/wandb-summary.json"))
    if not wandb_dirs:
        return result
    
    summary_file = wandb_dirs[0]
    try:
        with open(summary_file) as f:
            summary = json.load(f)
        
        # Extract metrics from wandb summary
        if 'iteration-time' in summary:
            result.step_time_sec = summary['iteration-time']
            result.iteration_time_ms = result.step_time_sec * 1000
        
        if 'tokens-per-sec' in summary:
            result.tokens_per_sec = summary['tokens-per-sec']
        
        if 'tokens-per-sec-per-gpu' in summary:
            result.tokens_per_sec_per_gpu = summary['tokens-per-sec-per-gpu']
        
        if 'tflops-per-gpu' in summary:
            result.tflops_per_gpu = summary['tflops-per-gpu']
        
        if 'samples-per-sec' in summary:
            result.samples_per_sec = summary['samples-per-sec']
            
    except Exception as e:
        print(f"Error reading wandb summary: {e}")
    
    return result


def load_reference_values(sweep_dir: Path) -> dict:
    """Load reference values from CSV file."""
    ref_file = sweep_dir / "reference_values.csv"
    references = {}
    
    if not ref_file.exists():
        print(f"Warning: Reference file not found: {ref_file}")
        return references
    
    with open(ref_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = f"{row['model']}_{row['size']}_{row['task']}"
            references[key] = {
                'ref_step_time': float(row['ref_step_time']),
                'ref_tflops': float(row['ref_tflops']),
                'ref_mfu': float(row['ref_mfu']),
                'ref_tokens_sec': float(row['ref_tokens_sec']),
                'ref_tokens_sec_gpu': float(row['ref_tokens_sec_gpu']),
            }
    
    return references


def collect_from_sweep(sweep_dir: Path, references: dict) -> list[ExperimentResult]:
    """Collect results from a sweep directory."""
    results = []
    jobs_file = sweep_dir / "submitted_jobs.txt"
    
    if not jobs_file.exists():
        print(f"Jobs file not found: {jobs_file}")
        return results
    
    with open(jobs_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) < 17:
                continue
            
            result = ExperimentResult()
            result.exp_name = f"{parts[1]}_{parts[2]}_{parts[3]}"
            result.model_name = parts[1]
            result.model_size = parts[2]
            result.task = parts[3]
            result.num_gpus = int(parts[4])
            result.num_nodes = int(parts[5])
            result.gpu_type = parts[6]
            result.tp = int(parts[7])
            result.pp = int(parts[8])
            result.cp = int(parts[9])
            result.ep = int(parts[10])
            result.dp = int(parts[11])
            result.seq_len = int(parts[12])
            result.mbs = int(parts[13])
            result.gbs = int(parts[14])
            result.vp = int(parts[15])
            exp_dir = Path(parts[16])
            result.exp_dir = str(exp_dir)
            
            # Load reference values
            ref_key = f"{result.model_name}_{result.model_size}_{result.task}"
            if ref_key in references:
                ref = references[ref_key]
                result.ref_step_time = ref['ref_step_time']
                result.ref_tflops = ref['ref_tflops']
                result.ref_mfu = ref['ref_mfu']
                result.ref_tokens_sec = ref['ref_tokens_sec']
                result.ref_tokens_sec_gpu = ref['ref_tokens_sec_gpu']
            
            # Parse log files
            if exp_dir.exists():
                log_files = list(exp_dir.glob("**/log-*.out"))
                for log_file in log_files:
                    result = parse_log_file(log_file, result)
                
                # Also try wandb summary
                result = parse_wandb_summary(exp_dir, result)
            
            results.append(result)
    
    return results


def calculate_comparison(result: ExperimentResult) -> dict:
    """Calculate comparison metrics."""
    comparison = {
        'step_time_diff': 0.0,
        'step_time_pct': 0.0,
        'tflops_diff': 0.0,
        'tflops_pct': 0.0,
        'tokens_sec_diff': 0.0,
        'tokens_sec_pct': 0.0,
        'tokens_sec_gpu_diff': 0.0,
        'tokens_sec_gpu_pct': 0.0,
    }
    
    if result.ref_step_time > 0 and result.step_time_sec > 0:
        comparison['step_time_diff'] = result.step_time_sec - result.ref_step_time
        comparison['step_time_pct'] = ((result.step_time_sec - result.ref_step_time) / result.ref_step_time) * 100
    
    if result.ref_tflops > 0 and result.tflops_per_gpu > 0:
        comparison['tflops_diff'] = result.tflops_per_gpu - result.ref_tflops
        comparison['tflops_pct'] = ((result.tflops_per_gpu - result.ref_tflops) / result.ref_tflops) * 100
    
    if result.ref_tokens_sec > 0 and result.tokens_per_sec > 0:
        comparison['tokens_sec_diff'] = result.tokens_per_sec - result.ref_tokens_sec
        comparison['tokens_sec_pct'] = ((result.tokens_per_sec - result.ref_tokens_sec) / result.ref_tokens_sec) * 100
    
    if result.ref_tokens_sec_gpu > 0 and result.tokens_per_sec_per_gpu > 0:
        comparison['tokens_sec_gpu_diff'] = result.tokens_per_sec_per_gpu - result.ref_tokens_sec_gpu
        comparison['tokens_sec_gpu_pct'] = ((result.tokens_per_sec_per_gpu - result.ref_tokens_sec_gpu) / result.ref_tokens_sec_gpu) * 100
    
    return comparison


def print_comparison_table(results: list[ExperimentResult]):
    """Print a comparison table."""
    if not results:
        print("No results to display.")
        return
    
    # Color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Group by model
    from collections import defaultdict
    model_groups = defaultdict(list)
    for r in results:
        key = f"{r.model_name}_{r.model_size}_{r.task}"
        model_groups[key].append(r)
    
    print("\n" + "=" * 160)
    print(f"{BOLD}{'Reference Comparison Report':^160}{RESET}")
    print("=" * 160)
    
    for model_key in sorted(model_groups.keys()):
        group = model_groups[model_key]
        
        print(f"\n{BOLD}{CYAN}Model: {model_key.upper()}{RESET}")
        print("-" * 160)
        
        # Header
        print(f"{'Metric':<25} {'Reference':>15} {'Measured':>15} {'Diff':>15} {'% Diff':>12} {'Status':>15}")
        print("-" * 160)
        
        for r in group:
            comp = calculate_comparison(r)
            
            # Status indicator
            if r.status == 'completed':
                status = f"{GREEN}✓ Completed{RESET}"
            elif r.status == 'oom':
                status = f"{RED}✗ OOM{RESET}"
            elif r.status == 'running':
                status = f"{YELLOW}⋯ Running{RESET}"
            else:
                status = f"{RED}✗ {r.status}{RESET}"
            
            print(f"\n  Configuration: TP={r.tp}, PP={r.pp}, CP={r.cp}, EP={r.ep}, DP={r.dp}")
            print(f"  GPUs: {r.num_gpus} ({r.num_nodes} nodes), SEQ={r.seq_len}, MBS={r.mbs}, GBS={r.gbs}")
            print(f"  Status: {status}")
            print()
            
            if r.status != 'completed':
                print(f"  {YELLOW}(No metrics available){RESET}")
                continue
            
            # Step Time (lower is better)
            color = GREEN if comp['step_time_pct'] <= 5 else (YELLOW if comp['step_time_pct'] <= 20 else RED)
            print(f"  {'Step Time (sec)':<25} {r.ref_step_time:>15.3f} {r.step_time_sec:>15.3f} "
                  f"{comp['step_time_diff']:>+15.3f} {color}{comp['step_time_pct']:>+11.1f}%{RESET}")
            
            # TFLOPs (higher is better)
            color = GREEN if comp['tflops_pct'] >= -5 else (YELLOW if comp['tflops_pct'] >= -20 else RED)
            print(f"  {'TFLOPs/GPU':<25} {r.ref_tflops:>15.1f} {r.tflops_per_gpu:>15.1f} "
                  f"{comp['tflops_diff']:>+15.1f} {color}{comp['tflops_pct']:>+11.1f}%{RESET}")
            
            # Tokens/sec (higher is better)
            color = GREEN if comp['tokens_sec_pct'] >= -5 else (YELLOW if comp['tokens_sec_pct'] >= -20 else RED)
            print(f"  {'Tokens/sec':<25} {r.ref_tokens_sec:>15.0f} {r.tokens_per_sec:>15.0f} "
                  f"{comp['tokens_sec_diff']:>+15.0f} {color}{comp['tokens_sec_pct']:>+11.1f}%{RESET}")
            
            # Tokens/sec/GPU (higher is better)
            color = GREEN if comp['tokens_sec_gpu_pct'] >= -5 else (YELLOW if comp['tokens_sec_gpu_pct'] >= -20 else RED)
            print(f"  {'Tokens/sec/GPU':<25} {r.ref_tokens_sec_gpu:>15.0f} {r.tokens_per_sec_per_gpu:>15.0f} "
                  f"{comp['tokens_sec_gpu_diff']:>+15.0f} {color}{comp['tokens_sec_gpu_pct']:>+11.1f}%{RESET}")
    
    print("\n" + "=" * 160)
    
    # Summary
    completed = [r for r in results if r.status == 'completed']
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"  Total experiments: {len(results)}")
    print(f"  Completed: {GREEN}{len(completed)}{RESET}")
    print(f"  OOM: {RED}{sum(1 for r in results if r.status == 'oom')}{RESET}")
    print(f"  Failed: {RED}{sum(1 for r in results if r.status in ('failed', 'error'))}{RESET}")
    print(f"  Running: {YELLOW}{sum(1 for r in results if r.status == 'running')}{RESET}")
    
    if completed:
        # Calculate average deviation
        avg_step_pct = sum(calculate_comparison(r)['step_time_pct'] for r in completed) / len(completed)
        avg_tflops_pct = sum(calculate_comparison(r)['tflops_pct'] for r in completed) / len(completed)
        avg_tokens_pct = sum(calculate_comparison(r)['tokens_sec_gpu_pct'] for r in completed) / len(completed)
        
        print(f"\n  Average deviation from reference:")
        color = GREEN if abs(avg_step_pct) <= 5 else (YELLOW if abs(avg_step_pct) <= 20 else RED)
        print(f"    Step Time: {color}{avg_step_pct:+.1f}%{RESET}")
        color = GREEN if avg_tflops_pct >= -5 else (YELLOW if avg_tflops_pct >= -20 else RED)
        print(f"    TFLOPs/GPU: {color}{avg_tflops_pct:+.1f}%{RESET}")
        color = GREEN if avg_tokens_pct >= -5 else (YELLOW if avg_tokens_pct >= -20 else RED)
        print(f"    Tokens/sec/GPU: {color}{avg_tokens_pct:+.1f}%{RESET}")
    
    print("=" * 160)


def save_comparison_csv(results: list[ExperimentResult], output_path: Path):
    """Save comparison results to CSV."""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            'model', 'size', 'task', 'status',
            'gpus', 'nodes', 'tp', 'pp', 'cp', 'ep', 'dp', 'vp',
            'seq_len', 'mbs', 'gbs',
            'ref_step_time', 'measured_step_time', 'step_time_diff_pct',
            'ref_tflops', 'measured_tflops', 'tflops_diff_pct',
            'ref_tokens_sec', 'measured_tokens_sec', 'tokens_sec_diff_pct',
            'ref_tokens_sec_gpu', 'measured_tokens_sec_gpu', 'tokens_sec_gpu_diff_pct',
        ])
        
        for r in results:
            comp = calculate_comparison(r)
            writer.writerow([
                r.model_name, r.model_size, r.task, r.status,
                r.num_gpus, r.num_nodes, r.tp, r.pp, r.cp, r.ep, r.dp, r.vp,
                r.seq_len, r.mbs, r.gbs,
                r.ref_step_time, r.step_time_sec, comp['step_time_pct'],
                r.ref_tflops, r.tflops_per_gpu, comp['tflops_pct'],
                r.ref_tokens_sec, r.tokens_per_sec, comp['tokens_sec_pct'],
                r.ref_tokens_sec_gpu, r.tokens_per_sec_per_gpu, comp['tokens_sec_gpu_pct'],
            ])
    
    print(f"Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare experiment results with reference values")
    parser.add_argument("--sweep-dir", "-s", type=str, required=True,
                        help="Path to sweep directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output CSV file path")
    args = parser.parse_args()
    
    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"Error: Sweep directory not found: {sweep_dir}")
        return 1
    
    # Load reference values
    references = load_reference_values(sweep_dir)
    
    # Collect results
    results = collect_from_sweep(sweep_dir, references)
    
    if not results:
        print("No results found.")
        return 1
    
    # Print comparison table
    print_comparison_table(results)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else sweep_dir / f"comparison_{timestamp}.csv"
    save_comparison_csv(results, output_path)
    
    return 0


if __name__ == "__main__":
    exit(main())

