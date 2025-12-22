#!/usr/bin/env python3
"""
Configuration Validation Script

Compares your current Megatron-Bridge configurations against the official
GB200 reference configurations to identify any mismatches or missing settings.

Usage:
    python validate_configs.py [--verbose] [--model MODEL_NAME]
"""

import argparse
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class Config:
    """Reference configuration for a model"""
    task: str
    model: str
    size: str
    system: str
    dtype: str
    num_gpus: int
    seq_len: int
    tp_size: int
    pp_size: int
    cp_size: int
    ep_size: int
    vp_size: int
    mbs: int
    gbs: int
    etp_size: int
    cuda_graphs: int
    use_mcore_fsdp: int
    recompute_layers: int
    activation_offload_layers: int
    
    @property
    def dp_size(self) -> int:
        """Calculate DP from GPU count and parallelism"""
        return self.num_gpus // (self.tp_size * self.pp_size * self.cp_size)
    
    @property
    def ga(self) -> int:
        """Calculate gradient accumulation"""
        return self.gbs // (self.mbs * self.dp_size)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for comparison"""
        return {
            'task': self.task,
            'model': self.model,
            'size': self.size,
            'system': self.system,
            'dtype': self.dtype,
            'num_gpus': self.num_gpus,
            'seq_len': self.seq_len,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'cp_size': self.cp_size,
            'ep_size': self.ep_size,
            'vp_size': self.vp_size,
            'mbs': self.mbs,
            'gbs': self.gbs,
            'dp_size': self.dp_size,
            'ga': self.ga,
            'etp_size': self.etp_size,
            'cuda_graphs': self.cuda_graphs,
            'use_mcore_fsdp': self.use_mcore_fsdp,
            'recompute_layers': self.recompute_layers,
            'activation_offload_layers': self.activation_offload_layers,
        }
    
    def __str__(self) -> str:
        return f"{self.model}_{self.size}_{self.dtype}_{self.task}"


# Reference configurations from the provided CSV
REFERENCE_CONFIGS = [
    # LORA Tasks
    Config('lora', 'llama3', '70b', 'gb200', 'fp8', 8, 2048, 1, 4, 1, 1, 20, 1, 64, 0, 1, 0, 0, 0),
    Config('lora', 'llama3', '70b', 'gb200', 'bf16', 8, 2048, 1, 4, 1, 1, 20, 1, 64, 0, 1, 0, 0, 0),
    Config('lora', 'llama3', '8b', 'gb200', 'bf16', 8, 16384, 1, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0),
    Config('lora', 'llama3', '8b', 'gb200', 'fp8', 8, 16384, 1, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0),
    Config('lora', 'llama31', '405b', 'gb200', 'fp8', 32, 2048, 4, 4, 1, 1, 16, 1, 32, 0, 1, 0, 0, 0),
    Config('lora', 'llama31', '405b', 'gb200', 'bf16', 32, 2048, 4, 4, 1, 1, 16, 1, 32, 0, 1, 0, 0, 0),
    
    # Pre-train Tasks
    Config('pre_train', 'gpt3', '175b', 'gb200', 'bf16', 128, 2048, 4, 4, 1, 1, 12, 2, 256, 0, 1, 0, 0, 0),
    Config('pre_train', 'gpt3', '175b', 'gb200', 'fp8', 128, 2048, 4, 4, 1, 1, 12, 2, 256, 0, 1, 0, 0, 0),
    Config('pre_train', 'llama3', '8b', 'gb200', 'bf16', 8, 8192, 1, 1, 1, 1, 1, 2, 128, 0, 1, 0, 0, 0),
    Config('pre_train', 'llama3', '8b', 'gb200', 'fp8', 8, 8192, 1, 1, 1, 1, 1, 4, 128, 0, 1, 0, 0, 0),
    Config('pre_train', 'llama3', '70b', 'gb200', 'fp8', 64, 8192, 1, 1, 1, 1, 1, 1, 128, 0, 0, 1, 0, 0),
    Config('pre_train', 'llama3', '70b', 'gb200', 'bf16', 64, 8192, 1, 1, 1, 1, 1, 1, 128, 0, 0, 1, 20, 0),
    Config('pre_train', 'llama31', '405b', 'gb200', 'bf16', 128, 8192, 4, 8, 2, 1, 8, 1, 64, 0, 0, 0, 0, 0),
    Config('pre_train', 'llama31', '405b', 'gb200', 'fp8', 128, 8192, 2, 1, 1, 1, 1, 1, 256, 0, 0, 1, 0, 95),
    Config('pre_train', 'mixtral', '8x7b', 'gb200', 'bf16', 64, 4096, 1, 1, 1, 8, 1, 2, 256, 0, 1, 0, 0, 0),
    Config('pre_train', 'mixtral', '8x7b', 'gb200', 'fp8', 64, 4096, 1, 1, 1, 8, 1, 2, 256, 0, 1, 0, 0, 0),
    Config('pre_train', 'nemotron4', '15b', 'gb200', 'bf16', 64, 4096, 1, 1, 1, 1, 1, 2, 256, 0, 1, 0, 0, 0),
    Config('pre_train', 'nemotron4', '15b', 'gb200', 'fp8', 64, 4096, 1, 1, 1, 1, 1, 2, 256, 0, 1, 0, 0, 0),
    Config('pre_train', 'nemotron4', '340b', 'gb200', 'bf16', 128, 4096, 4, 8, 1, 1, 12, 1, 32, 0, 0, 0, 0, 0),
    Config('pre_train', 'nemotron4', '340b', 'gb200', 'fp8', 128, 4096, 8, 4, 1, 1, 12, 1, 32, 0, 1, 0, 0, 0),
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 1024, 4096, 2, 4, 1, 64, 4, 1, 8192, 1, 1, 0, 0, 0),
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 256, 4096, 2, 4, 1, 64, 1, 1, 2048, 1, 1, 0, 0, 0),
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 128, 4096, 2, 4, 1, 32, 1, 1, 1024, 1, 1, 0, 0, 0),
    Config('pre_train', 'nemotronh', '8b', 'gb200', 'fp8', 8, 8192, 1, 1, 1, 1, 1, 2, 128, 0, 1, 0, 0, 0),
    Config('pre_train', 'nemotronh', '47b', 'gb200', 'fp8', 64, 8192, 2, 1, 1, 1, 1, 1, 192, 0, 1, 0, 0, 0),
    Config('pre_train', 'nemotronh', '56b', 'gb200', 'fp8', 64, 8192, 2, 1, 1, 1, 1, 1, 192, 0, 1, 0, 0, 0),
    Config('pre_train', 'nemotronh', '56b', 'gb200', 'fp8', 256, 8192, 2, 1, 1, 1, 1, 1, 768, 0, 1, 0, 0, 0),
    Config('pre_train', 'llama4', 'e16', 'gb200', 'bf16', 64, 8192, 1, 1, 1, 16, 1, 1, 1024, 1, 1, 0, 0, 0),
    Config('pre_train', 'llama4', 'e16', 'gb200', 'fp8', 64, 8192, 1, 1, 1, 16, 1, 1, 1024, 1, 1, 0, 0, 0),
    Config('pre_train', 'llama4', 'e128', 'gb200', 'bf16', 128, 8192, 1, 2, 1, 64, 12, 1, 1024, 1, 0, 0, 0, 0),
    Config('pre_train', 'vlm_llama4', 'e16', 'gb200', 'bf16', 64, 8192, 1, 1, 2, 16, 1, 1, 1024, 1, 1, 0, 0, 0),
    Config('pre_train', 'vlm_llama4', 'e128', 'gb200', 'bf16', 128, 8192, 2, 1, 1, 64, 1, 1, 1024, 1, 1, 0, 0, 0),
    Config('pre_train', 'qwen3', '30b_a3b', 'gb200', 'bf16', 8, 4096, 1, 1, 1, 8, 1, 1, 512, 1, 1, 0, 0, 0),
    Config('pre_train', 'qwen3', '235b_a22b', 'gb200', 'bf16', 64, 4096, 2, 1, 1, 64, 1, 1, 1024, 1, 1, 0, 0, 0),
    
    # SFT Tasks
    Config('sft', 'llama3', '8b', 'gb200', 'fp8', 8, 16384, 1, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0),
    Config('sft', 'llama3', '8b', 'gb200', 'bf16', 8, 16384, 1, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0),
    Config('sft', 'llama3', '70b', 'gb200', 'bf16', 32, 4096, 2, 4, 1, 1, 5, 1, 32, 0, 1, 0, 0, 0),
    Config('sft', 'llama3', '70b', 'gb200', 'fp8', 32, 4096, 2, 4, 1, 1, 5, 1, 32, 0, 1, 0, 0, 0),
    
    # Finetune Tasks
    Config('finetune', 'qwen25vl', '7b', 'gb200', 'bf16', 8, 8192, 1, 1, 1, 1, 1, 1, 64, 0, 1, 0, 0, 0),
    Config('finetune', 'qwen25vl', '7b', 'gb200', 'fp8', 8, 8192, 1, 1, 1, 1, 1, 1, 64, 0, 1, 0, 0, 0),
    Config('finetune', 'qwen25vl', '32b', 'gb200', 'bf16', 16, 8192, 4, 1, 1, 1, 1, 1, 64, 0, 1, 0, 0, 0),
]


def parse_current_configs() -> Dict[str, List[Config]]:
    """Parse current configurations from the codebase"""
    # This would parse from your actual config files
    # For now, let's create a placeholder that reads from your sweep scripts
    current = defaultdict(list)
    
    # TODO: Parse from actual config files
    # For demonstration, I'll add what we know exists
    
    return current


def compare_configs(reference: Config, current: Config, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Compare two configs and return (matches, differences)"""
    differences = []
    
    ref_dict = reference.to_dict()
    cur_dict = current.to_dict()
    
    for key in ref_dict:
        if ref_dict[key] != cur_dict.get(key):
            differences.append(f"  {key}: reference={ref_dict[key]}, current={cur_dict.get(key)}")
    
    return len(differences) == 0, differences


def validate_all(filter_model: str = None, verbose: bool = False):
    """Validate all configurations"""
    
    print("=" * 80)
    print("GB200 Configuration Validation Report")
    print("=" * 80)
    print()
    
    # Group by task and model
    by_task_model = defaultdict(list)
    for config in REFERENCE_CONFIGS:
        key = f"{config.task}_{config.model}_{config.size}"
        if filter_model and filter_model.lower() not in key.lower():
            continue
        by_task_model[key].append(config)
    
    # Print reference configurations in a nice table
    for key in sorted(by_task_model.keys()):
        configs = by_task_model[key]
        
        print(f"\n{'─' * 80}")
        print(f"Model: {key}")
        print(f"{'─' * 80}")
        
        for config in configs:
            print(f"\n  Precision: {config.dtype.upper()}")
            print(f"  {'─' * 76}")
            print(f"    GPUs: {config.num_gpus} | Seq Len: {config.seq_len}")
            print(f"    Parallelism:")
            print(f"      TP={config.tp_size}, PP={config.pp_size}, CP={config.cp_size}, DP={config.dp_size}")
            print(f"      EP={config.ep_size}, VP={config.vp_size}, ETP={config.etp_size}")
            print(f"    Batch:")
            print(f"      MBS={config.mbs}, GBS={config.gbs}, GA={config.ga}")
            print(f"    Optimizations:")
            print(f"      FSDP={config.use_mcore_fsdp}, CudaGraphs={config.cuda_graphs}")
            print(f"      Recompute Layers={config.recompute_layers}")
            print(f"      CPU Offload Layers={config.activation_offload_layers}")
            
            if verbose:
                print(f"\n    Full Config: {config.to_dict()}")
    
    print(f"\n{'=' * 80}")
    print(f"Total Reference Configs: {len(REFERENCE_CONFIGS)}")
    print(f"{'=' * 80}\n")
    
    # Print summary by task
    print("\nSummary by Task:")
    print("─" * 80)
    task_summary = defaultdict(lambda: defaultdict(set))
    for config in REFERENCE_CONFIGS:
        task_summary[config.task]['models'].add(f"{config.model}_{config.size}")
        task_summary[config.task]['dtypes'].add(config.dtype)
        task_summary[config.task]['gpu_counts'].add(config.num_gpus)
    
    for task in sorted(task_summary.keys()):
        info = task_summary[task]
        print(f"\n  {task.upper()}:")
        print(f"    Models: {', '.join(sorted(info['models']))}")
        print(f"    Precisions: {', '.join(sorted(info['dtypes']))}")
        print(f"    GPU Counts: {', '.join(map(str, sorted(info['gpu_counts'])))}")
    
    print(f"\n{'=' * 80}\n")
    
    # Check for models you mentioned
    print("\nYour Models Status Check:")
    print("─" * 80)
    
    your_models = [
        ('pre_train', 'llama3', '8b', 'bf16'),
        ('pre_train', 'llama3', '70b', 'bf16'),
        ('pre_train', 'llama31', '405b', 'bf16'),
        ('pre_train', 'deepseek', 'v3', 'bf16'),
        ('pre_train', 'qwen3', '30b_a3b', 'bf16'),
        ('pre_train', 'qwen3', '235b_a22b', 'bf16'),
    ]
    
    for task, model, size, dtype in your_models:
        found = [c for c in REFERENCE_CONFIGS 
                 if c.task == task and c.model == model and c.size == size and c.dtype == dtype]
        
        status = "✓ FOUND" if found else "✗ MISSING"
        print(f"  {status}: {task} | {model} {size} | {dtype}")
        
        if found and verbose:
            for config in found:
                print(f"    → {config.num_gpus} GPUs, SEQ={config.seq_len}, "
                      f"TP={config.tp_size}, PP={config.pp_size}, MBS={config.mbs}, GBS={config.gbs}")
    
    print(f"\n{'=' * 80}\n")


def export_to_json(output_file: str = "reference_configs_gb200.json"):
    """Export reference configs to JSON"""
    data = {
        "description": "GB200 Reference Configurations",
        "source": "Official GB200 Performance Recommendations",
        "date": "2025-12-06",
        "configs": []
    }
    
    for config in REFERENCE_CONFIGS:
        data["configs"].append(config.to_dict())
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported {len(REFERENCE_CONFIGS)} configs to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate Megatron-Bridge configs against GB200 reference"
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Show detailed configuration info")
    parser.add_argument('--model', '-m', type=str,
                        help="Filter by model name (e.g., llama3, qwen3)")
    parser.add_argument('--export', '-e', type=str,
                        help="Export to JSON file")
    
    args = parser.parse_args()
    
    validate_all(filter_model=args.model, verbose=args.verbose)
    
    if args.export:
        export_to_json(args.export)

