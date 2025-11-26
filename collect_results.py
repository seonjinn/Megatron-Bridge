#!/usr/bin/env python3
"""
Collect and summarize results from multiple Megatron-Bridge experiments.

Usage:
    python3 collect_results.py --sweep-dir ./exp_logs/sweeps/20251125_120000
    python3 collect_results.py --exp-dirs ./exp_logs/experiments/llama3_8b_*/
    python3 collect_results.py --latest 5  # Collect from latest 5 experiments
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    # Experiment identification
    exp_name: str = ""
    exp_dir: str = ""
    job_id: str = ""
    
    # Model configuration
    model_name: str = ""
    model_size: str = ""
    
    # Hardware configuration
    num_gpus: int = 0
    num_nodes: int = 0
    gpus_per_node: int = 0
    gpu_type: str = ""
    
    # Parallelism configuration
    tp: int = 1
    pp: int = 1
    cp: int = 1
    ep: int = 1
    dp: int = 1
    
    # Training configuration
    global_batch_size: int = 0
    micro_batch_size: int = 0
    sequence_length: int = 0
    max_steps: int = 0
    precision: str = ""
    
    # Performance metrics (averaged over stable iterations)
    tokens_per_sec_per_gpu: float = 0.0
    tokens_per_sec_total: float = 0.0
    tflops_per_gpu: float = 0.0
    iteration_time_ms: float = 0.0
    samples_per_sec: float = 0.0
    
    # Performance metrics (min/max/std)
    tokens_per_sec_per_gpu_min: float = 0.0
    tokens_per_sec_per_gpu_max: float = 0.0
    tokens_per_sec_per_gpu_std: float = 0.0
    
    # Memory metrics
    memory_allocated_gb: float = 0.0
    memory_reserved_gb: float = 0.0
    
    # Status
    status: str = "unknown"  # completed, running, failed
    total_iterations: int = 0
    
    # Raw data
    iteration_data: list = field(default_factory=list)


def parse_log_file(log_path: Path) -> ExperimentResult:
    """Parse a Megatron-Bridge log file and extract metrics."""
    result = ExperimentResult()
    result.exp_dir = str(log_path.parent.parent)
    result.exp_name = log_path.parent.parent.name
    
    iteration_pattern = re.compile(
        r'iteration\s+(\d+)/\s*(\d+)\s*\|'
        r'.*elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|'
        r'.*throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)\s*\|'
        r'(?:.*Tokens/sec/GPU:\s*([\d.]+)\s*\|)?'
        r'.*global batch size:\s*(\d+)'
    )
    
    memory_pattern = re.compile(
        r'allocated:\s*([\d.]+)\s*GB.*reserved:\s*([\d.]+)\s*GB'
    )
    
    # Patterns to extract configuration from log files
    # Format in logs: "  tensor_model_parallel_size: 1" or "tensor_model_parallel_size=1"
    config_patterns = {
        'tp': re.compile(r'tensor_model_parallel_size[=:\s]+(\d+)'),
        'pp': re.compile(r'pipeline_model_parallel_size[=:\s]+(\d+)'),
        'cp': re.compile(r'context_parallel_size[=:\s]+(\d+)'),
        'ep': re.compile(r'expert_model_parallel_size[=:\s]+(\d+)'),
        'dp': re.compile(r'data_parallel_size[=:\s]+(\d+)'),
        'micro_batch_size': re.compile(r'micro_batch_size[=:\s]+(\d+)'),
        'sequence_length': re.compile(r'(?:seq_length|sequence_length)[=:\s]+(\d+)'),
    }
    
    # Additional patterns to try for num_gpus (world_size)
    num_gpus_patterns = [
        re.compile(r'world_size[=:\s]+(\d+)', re.IGNORECASE),
        re.compile(r'world size[=:\s]+(\d+)', re.IGNORECASE),
        re.compile(r'Running on (\d+) GPUs', re.IGNORECASE),
        re.compile(r'using (\d+) GPUs', re.IGNORECASE),
        re.compile(r'num_gpus[=:\s]+(\d+)', re.IGNORECASE),
    ]
    
    iterations = []
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Parse configuration values
            for key, pattern in config_patterns.items():
                match = pattern.search(content)
                if match:
                    setattr(result, key, int(match.group(1)))
            
            # Try additional patterns for num_gpus if not found
            if result.num_gpus == 0:
                for pattern in num_gpus_patterns:
                    match = pattern.search(content)
                    if match:
                        result.num_gpus = int(match.group(1))
                        break
            
            # Calculate num_gpus from parallelism dimensions if still not found
            # num_gpus = TP * PP * CP * DP (EP is within DP, not multiplied)
            if result.num_gpus == 0 and result.tp > 0 and result.pp > 0 and result.dp > 0:
                result.num_gpus = result.tp * result.pp * result.cp * result.dp
            
            # Parse memory info
            mem_match = memory_pattern.search(content)
            if mem_match:
                result.memory_allocated_gb = float(mem_match.group(1))
                result.memory_reserved_gb = float(mem_match.group(2))
            
            # Parse iteration data
            for match in iteration_pattern.finditer(content):
                iter_num = int(match.group(1))
                max_iter = int(match.group(2))
                iter_time = float(match.group(3))
                tflops = float(match.group(4))
                tokens_per_sec = float(match.group(5)) if match.group(5) else 0.0
                gbs = int(match.group(6))
                
                iterations.append({
                    'iteration': iter_num,
                    'iteration_time_ms': iter_time,
                    'tflops_per_gpu': tflops,
                    'tokens_per_sec_per_gpu': tokens_per_sec,
                    'global_batch_size': gbs,
                })
                
                result.max_steps = max_iter
                result.global_batch_size = gbs
            
            # Check completion status
            if 'Training completed' in content or iterations and iterations[-1]['iteration'] >= result.max_steps:
                result.status = 'completed'
            elif iterations:
                result.status = 'running'
            else:
                result.status = 'failed'
                
    except Exception as e:
        print(f"Error parsing {log_path}: {e}")
        result.status = 'error'
        return result
    
    # Calculate aggregated metrics (skip first few warmup iterations)
    if len(iterations) > 5:
        stable_iterations = iterations[3:]  # Skip first 3 warmup iterations
        
        tokens_list = [it['tokens_per_sec_per_gpu'] for it in stable_iterations if it['tokens_per_sec_per_gpu'] > 0]
        tflops_list = [it['tflops_per_gpu'] for it in stable_iterations]
        time_list = [it['iteration_time_ms'] for it in stable_iterations]
        
        if tokens_list:
            result.tokens_per_sec_per_gpu = sum(tokens_list) / len(tokens_list)
            result.tokens_per_sec_per_gpu_min = min(tokens_list)
            result.tokens_per_sec_per_gpu_max = max(tokens_list)
            if len(tokens_list) > 1:
                mean = result.tokens_per_sec_per_gpu
                variance = sum((x - mean) ** 2 for x in tokens_list) / len(tokens_list)
                result.tokens_per_sec_per_gpu_std = variance ** 0.5
        
        if tflops_list:
            result.tflops_per_gpu = sum(tflops_list) / len(tflops_list)
        
        if time_list:
            result.iteration_time_ms = sum(time_list) / len(time_list)
    
    result.tokens_per_sec_total = result.tokens_per_sec_per_gpu * result.num_gpus
    result.total_iterations = len(iterations)
    result.iteration_data = iterations
    
    # Calculate samples per second
    if result.iteration_time_ms > 0 and result.global_batch_size > 0:
        result.samples_per_sec = result.global_batch_size / (result.iteration_time_ms / 1000)
    
    return result


def find_log_file(exp_dir: Path) -> Optional[Path]:
    """Find the main log file in an experiment directory."""
    # Look for log files in the subdirectory
    for subdir in exp_dir.iterdir():
        if subdir.is_dir():
            log_files = list(subdir.glob('log-*.out'))
            if log_files:
                return log_files[0]
    return None


def collect_from_sweep(sweep_dir: Path) -> list[ExperimentResult]:
    """Collect results from a sweep directory."""
    results = []
    jobs_file = sweep_dir / 'submitted_jobs.txt'
    
    if not jobs_file.exists():
        print(f"Jobs file not found: {jobs_file}")
        return results
    
    with open(jobs_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('|')
            if len(parts) >= 10:
                exp_dir = Path(parts[9])
                if exp_dir.exists():
                    log_file = find_log_file(exp_dir)
                    if log_file:
                        result = parse_log_file(log_file)
                        result.job_id = parts[0]
                        result.model_name = parts[1]
                        result.model_size = parts[2]
                        # Parse num_gpus from submitted_jobs.txt if not found in log
                        if result.num_gpus == 0:
                            try:
                                result.num_gpus = int(parts[3])
                            except (ValueError, IndexError):
                                pass
                        # Parse parallelism from submitted_jobs.txt as fallback
                        if result.tp == 1 and result.pp == 1 and result.dp == 1:
                            try:
                                result.tp = int(parts[4])
                                result.pp = int(parts[5])
                                result.cp = int(parts[6])
                                result.ep = int(parts[7])
                                result.dp = int(parts[8])
                            except (ValueError, IndexError):
                                pass
                        # Recalculate total tokens/sec with correct num_gpus
                        result.tokens_per_sec_total = result.tokens_per_sec_per_gpu * result.num_gpus
                        results.append(result)
    
    return results


def collect_from_dirs(exp_dirs: list[Path]) -> list[ExperimentResult]:
    """Collect results from a list of experiment directories."""
    results = []
    
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            continue
        
        log_file = find_log_file(exp_dir)
        if log_file:
            result = parse_log_file(log_file)
            results.append(result)
    
    return results


def collect_latest(base_dir: Path, n: int) -> list[ExperimentResult]:
    """Collect results from the latest N experiments."""
    exp_dirs = sorted(base_dir.glob('*/'), key=lambda x: x.stat().st_mtime, reverse=True)[:n]
    return collect_from_dirs(exp_dirs)


def save_results_csv(results: list[ExperimentResult], output_path: Path):
    """Save results to CSV file."""
    fieldnames = [
        'exp_name', 'status', 'model_name', 'model_size',
        'num_gpus', 'tp', 'pp', 'cp', 'ep', 'dp',
        'global_batch_size', 'sequence_length',
        'tokens_per_sec_per_gpu', 'tokens_per_sec_total',
        'tflops_per_gpu', 'iteration_time_ms', 'samples_per_sec',
        'tokens_per_sec_per_gpu_min', 'tokens_per_sec_per_gpu_max', 'tokens_per_sec_per_gpu_std',
        'memory_allocated_gb', 'memory_reserved_gb',
        'total_iterations', 'max_steps', 'exp_dir'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: getattr(r, k, '') for k in fieldnames}
            # Format floats
            for key in ['tokens_per_sec_per_gpu', 'tokens_per_sec_total', 'tflops_per_gpu',
                       'iteration_time_ms', 'samples_per_sec', 'memory_allocated_gb', 'memory_reserved_gb',
                       'tokens_per_sec_per_gpu_min', 'tokens_per_sec_per_gpu_max', 'tokens_per_sec_per_gpu_std']:
                if key in row and isinstance(row[key], float):
                    row[key] = f"{row[key]:.2f}"
            writer.writerow(row)
    
    print(f"Results saved to: {output_path}")


def save_results_json(results: list[ExperimentResult], output_path: Path):
    """Save results to JSON file."""
    data = []
    for r in results:
        d = {k: v for k, v in r.__dict__.items() if k != 'iteration_data'}
        d['iteration_data'] = r.iteration_data  # Include raw data
        data.append(d)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Results saved to: {output_path}")


def print_summary_table(results: list[ExperimentResult]):
    """Print a summary table of results."""
    if not results:
        print("No results to display.")
        return
    
    # Header
    print("\n" + "=" * 140)
    print(f"{'Experiment':<40} {'Status':<10} {'GPUs':>5} {'TP':>3} {'PP':>3} {'CP':>3} {'DP':>3} "
          f"{'Tok/s/GPU':>12} {'TFLOP/s':>10} {'Iter(ms)':>10} {'Samples/s':>10}")
    print("=" * 140)
    
    # Sort by tokens/sec/gpu descending
    sorted_results = sorted(results, key=lambda x: x.tokens_per_sec_per_gpu, reverse=True)
    
    for r in sorted_results:
        status_color = {
            'completed': '\033[92m',  # Green
            'running': '\033[93m',    # Yellow
            'failed': '\033[91m',     # Red
            'error': '\033[91m',
        }.get(r.status, '')
        reset = '\033[0m' if status_color else ''
        
        print(f"{r.exp_name:<40} {status_color}{r.status:<10}{reset} "
              f"{r.num_gpus:>5} {r.tp:>3} {r.pp:>3} {r.cp:>3} {r.dp:>3} "
              f"{r.tokens_per_sec_per_gpu:>12.1f} {r.tflops_per_gpu:>10.1f} "
              f"{r.iteration_time_ms:>10.1f} {r.samples_per_sec:>10.1f}")
    
    print("=" * 140)
    
    # Summary statistics
    completed = [r for r in results if r.status == 'completed']
    if completed:
        avg_tokens = sum(r.tokens_per_sec_per_gpu for r in completed) / len(completed)
        max_tokens = max(r.tokens_per_sec_per_gpu for r in completed)
        best = max(completed, key=lambda x: x.tokens_per_sec_per_gpu)
        
        print(f"\nSummary ({len(completed)} completed experiments):")
        print(f"  Average Tokens/sec/GPU: {avg_tokens:.1f}")
        print(f"  Best Tokens/sec/GPU: {max_tokens:.1f}")
        print(f"  Best configuration: {best.exp_name}")
        print(f"    TP={best.tp}, PP={best.pp}, CP={best.cp}, DP={best.dp}")


def main():
    parser = argparse.ArgumentParser(description='Collect results from Megatron-Bridge experiments')
    parser.add_argument('--sweep-dir', type=Path, help='Sweep directory containing submitted_jobs.txt')
    parser.add_argument('--exp-dirs', type=Path, nargs='+', help='List of experiment directories')
    parser.add_argument('--latest', type=int, help='Collect from latest N experiments')
    parser.add_argument('--base-dir', type=Path, default=Path('./exp_logs/experiments'),
                       help='Base directory for experiments (used with --latest)')
    parser.add_argument('--output-csv', type=Path, help='Output CSV file path')
    parser.add_argument('--output-json', type=Path, help='Output JSON file path')
    parser.add_argument('--no-print', action='store_true', help='Suppress table output')
    
    args = parser.parse_args()
    
    results = []
    
    if args.sweep_dir:
        results = collect_from_sweep(args.sweep_dir)
    elif args.exp_dirs:
        results = collect_from_dirs(args.exp_dirs)
    elif args.latest:
        results = collect_latest(args.base_dir, args.latest)
    else:
        # Default: collect from latest 10 experiments
        results = collect_latest(args.base_dir, 10)
    
    if not results:
        print("No results found.")
        sys.exit(1)
    
    # Print summary table
    if not args.no_print:
        print_summary_table(results)
    
    # Save outputs
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.output_csv:
        save_results_csv(results, args.output_csv)
    elif args.sweep_dir:
        save_results_csv(results, args.sweep_dir / f'results_{timestamp}.csv')
    
    if args.output_json:
        save_results_json(results, args.output_json)
    elif args.sweep_dir:
        save_results_json(results, args.sweep_dir / f'results_{timestamp}.json')


if __name__ == '__main__':
    main()

