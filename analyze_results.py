#!/usr/bin/env python3
"""
Analyze and visualize experiment results from Megatron-Bridge experiments.

Usage:
    python3 analyze_results.py results.csv
    python3 analyze_results.py results.json --plot
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Optional: matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_results(file_path: Path) -> list[dict]:
    """Load results from CSV or JSON file."""
    if file_path.suffix == '.csv':
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    elif file_path.suffix == '.json':
        with open(file_path, 'r') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def analyze_parallelism_impact(results: list[dict]):
    """Analyze the impact of different parallelism strategies."""
    print("\n" + "=" * 80)
    print("PARALLELISM IMPACT ANALYSIS")
    print("=" * 80)
    
    # Group by parallelism configuration
    configs = {}
    for r in results:
        if r.get('status') != 'completed':
            continue
        
        key = f"TP{r['tp']}_PP{r['pp']}_CP{r['cp']}"
        if key not in configs:
            configs[key] = []
        configs[key].append(r)
    
    print(f"\n{'Configuration':<20} {'Experiments':>12} {'Avg Tok/s/GPU':>15} {'Avg TFLOP/s':>12} {'Avg Iter(ms)':>12}")
    print("-" * 80)
    
    for config, exps in sorted(configs.items()):
        n = len(exps)
        avg_tokens = sum(float(e['tokens_per_sec_per_gpu']) for e in exps) / n
        avg_tflops = sum(float(e['tflops_per_gpu']) for e in exps) / n
        avg_time = sum(float(e['iteration_time_ms']) for e in exps) / n
        
        print(f"{config:<20} {n:>12} {avg_tokens:>15.1f} {avg_tflops:>12.1f} {avg_time:>12.1f}")


def analyze_scaling(results: list[dict]):
    """Analyze scaling efficiency across different GPU counts."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)
    
    # Group by GPU count
    gpu_groups = {}
    for r in results:
        if r.get('status') != 'completed':
            continue
        
        num_gpus = int(r['num_gpus'])
        if num_gpus not in gpu_groups:
            gpu_groups[num_gpus] = []
        gpu_groups[num_gpus].append(r)
    
    if len(gpu_groups) < 2:
        print("\nNeed experiments with different GPU counts for scaling analysis.")
        return
    
    print(f"\n{'GPUs':>6} {'Experiments':>12} {'Avg Tok/s/GPU':>15} {'Total Tok/s':>15} {'Efficiency':>12}")
    print("-" * 80)
    
    # Use smallest GPU count as baseline
    baseline_gpus = min(gpu_groups.keys())
    baseline_avg = sum(float(e['tokens_per_sec_per_gpu']) for e in gpu_groups[baseline_gpus]) / len(gpu_groups[baseline_gpus])
    
    for num_gpus, exps in sorted(gpu_groups.items()):
        n = len(exps)
        avg_tokens = sum(float(e['tokens_per_sec_per_gpu']) for e in exps) / n
        total_tokens = avg_tokens * num_gpus
        efficiency = (avg_tokens / baseline_avg) * 100
        
        print(f"{num_gpus:>6} {n:>12} {avg_tokens:>15.1f} {total_tokens:>15.1f} {efficiency:>11.1f}%")


def analyze_model_comparison(results: list[dict]):
    """Compare performance across different models."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    # Group by model
    models = {}
    for r in results:
        if r.get('status') != 'completed':
            continue
        
        key = f"{r['model_name']}_{r['model_size']}"
        if key not in models:
            models[key] = []
        models[key].append(r)
    
    print(f"\n{'Model':<20} {'Experiments':>12} {'Best Tok/s/GPU':>15} {'Best Config':>20}")
    print("-" * 80)
    
    for model, exps in sorted(models.items()):
        best = max(exps, key=lambda x: float(x['tokens_per_sec_per_gpu']))
        best_tokens = float(best['tokens_per_sec_per_gpu'])
        best_config = f"TP{best['tp']}_PP{best['pp']}_CP{best['cp']}"
        
        print(f"{model:<20} {len(exps):>12} {best_tokens:>15.1f} {best_config:>20}")


def find_optimal_config(results: list[dict]):
    """Find the optimal configuration for each model/GPU combination."""
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATIONS")
    print("=" * 80)
    
    # Group by model and GPU count
    groups = {}
    for r in results:
        if r.get('status') != 'completed':
            continue
        
        key = f"{r['model_name']}_{r['model_size']}_{r['num_gpus']}gpus"
        if key not in groups:
            groups[key] = []
        groups[key].append(r)
    
    print(f"\n{'Setup':<35} {'Best Tok/s/GPU':>15} {'TP':>4} {'PP':>4} {'CP':>4} {'DP':>4} {'Iter(ms)':>10}")
    print("-" * 100)
    
    for setup, exps in sorted(groups.items()):
        best = max(exps, key=lambda x: float(x['tokens_per_sec_per_gpu']))
        
        print(f"{setup:<35} {float(best['tokens_per_sec_per_gpu']):>15.1f} "
              f"{best['tp']:>4} {best['pp']:>4} {best['cp']:>4} {best['dp']:>4} "
              f"{float(best['iteration_time_ms']):>10.1f}")


def plot_results(results: list[dict], output_dir: Path):
    """Generate plots for the results."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available. Install with: pip install matplotlib")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    completed = [r for r in results if r.get('status') == 'completed']
    if not completed:
        print("No completed experiments to plot.")
        return
    
    # Plot 1: Tokens/sec/GPU by configuration
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = [f"TP{r['tp']}_PP{r['pp']}" for r in completed]
    tokens = [float(r['tokens_per_sec_per_gpu']) for r in completed]
    
    ax.bar(range(len(configs)), tokens, color='steelblue')
    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(configs, rotation=45, ha='right')
    ax.set_ylabel('Tokens/sec/GPU')
    ax.set_title('Throughput by Parallelism Configuration')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'throughput_by_config.png', dpi=150)
    print(f"Saved: {output_dir / 'throughput_by_config.png'}")
    plt.close()
    
    # Plot 2: TFLOP/s vs Tokens/sec correlation
    fig, ax = plt.subplots(figsize=(8, 6))
    
    tflops = [float(r['tflops_per_gpu']) for r in completed]
    
    ax.scatter(tflops, tokens, c='steelblue', alpha=0.7, s=100)
    ax.set_xlabel('TFLOP/s per GPU')
    ax.set_ylabel('Tokens/sec per GPU')
    ax.set_title('TFLOP/s vs Tokens/sec Correlation')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tflops_vs_tokens.png', dpi=150)
    print(f"Saved: {output_dir / 'tflops_vs_tokens.png'}")
    plt.close()
    
    # Plot 3: Iteration time comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    iter_times = [float(r['iteration_time_ms']) for r in completed]
    exp_names = [r['exp_name'][:30] for r in completed]
    
    colors = ['green' if t < sum(iter_times)/len(iter_times) else 'orange' for t in iter_times]
    ax.barh(range(len(exp_names)), iter_times, color=colors)
    ax.set_yticks(range(len(exp_names)))
    ax.set_yticklabels(exp_names)
    ax.set_xlabel('Iteration Time (ms)')
    ax.set_title('Iteration Time by Experiment')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'iteration_times.png', dpi=150)
    print(f"Saved: {output_dir / 'iteration_times.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze Megatron-Bridge experiment results')
    parser.add_argument('results_file', type=Path, help='Results file (CSV or JSON)')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot-dir', type=Path, default=Path('./exp_logs/plots'),
                       help='Directory for plot outputs')
    
    args = parser.parse_args()
    
    if not args.results_file.exists():
        print(f"Results file not found: {args.results_file}")
        sys.exit(1)
    
    results = load_results(args.results_file)
    print(f"Loaded {len(results)} experiment results from {args.results_file}")
    
    # Run analyses
    analyze_parallelism_impact(results)
    analyze_scaling(results)
    analyze_model_comparison(results)
    find_optimal_config(results)
    
    # Generate plots if requested
    if args.plot:
        plot_results(results, args.plot_dir)


if __name__ == '__main__':
    main()

