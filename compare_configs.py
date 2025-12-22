#!/usr/bin/env python3
"""
Configuration Comparison Tool

Compares your current Megatron-Bridge experimental configurations against
the GB200 reference configurations to identify mismatches.

Usage:
    python compare_configs.py [--verbose] [--model MODEL_NAME]
"""

import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for a model experiment"""
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
    etp_size: int = 0
    cuda_graphs: int = 1
    use_mcore_fsdp: int = 0
    recompute_layers: int = 0
    activation_offload_layers: int = 0
    source: str = "unknown"
    
    @property
    def dp_size(self) -> int:
        """Calculate DP from GPU count and parallelism"""
        return self.num_gpus // (self.tp_size * self.pp_size * self.cp_size)
    
    @property
    def ga(self) -> int:
        """Calculate gradient accumulation"""
        dp = self.dp_size
        if dp == 0:
            return 0
        return self.gbs // (self.mbs * dp)
    
    @property
    def key(self) -> str:
        """Unique key for matching configs"""
        return f"{self.task}_{self.model}_{self.size}_{self.dtype}_{self.num_gpus}gpus"
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
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
            'source': self.source,
        }


# GB200 Reference Configurations
REFERENCE_CONFIGS = [
    # Pre-train: LLAMA3 8B
    Config('pre_train', 'llama3', '8b', 'gb200', 'bf16', 8, 8192, 1, 1, 1, 1, 1, 2, 128, 0, 1, 0, 0, 0, 'gb200_reference'),
    Config('pre_train', 'llama3', '8b', 'gb200', 'fp8', 8, 8192, 1, 1, 1, 1, 1, 4, 128, 0, 1, 0, 0, 0, 'gb200_reference'),
    
    # Pre-train: LLAMA3 70B
    Config('pre_train', 'llama3', '70b', 'gb200', 'bf16', 64, 8192, 1, 1, 1, 1, 1, 1, 128, 0, 0, 1, 20, 0, 'gb200_reference'),
    Config('pre_train', 'llama3', '70b', 'gb200', 'fp8', 64, 8192, 1, 1, 1, 1, 1, 1, 128, 0, 0, 1, 0, 0, 'gb200_reference'),
    
    # Pre-train: LLAMA31 405B
    Config('pre_train', 'llama31', '405b', 'gb200', 'bf16', 128, 8192, 4, 8, 2, 1, 8, 1, 64, 0, 0, 0, 0, 0, 'gb200_reference'),
    Config('pre_train', 'llama31', '405b', 'gb200', 'fp8', 128, 8192, 2, 1, 1, 1, 1, 1, 256, 0, 0, 1, 0, 95, 'gb200_reference'),
    
    # Pre-train: DeepSeek V3
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 1024, 4096, 2, 4, 1, 64, 4, 1, 8192, 1, 1, 0, 0, 0, 'gb200_reference'),
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 256, 4096, 2, 4, 1, 64, 1, 1, 2048, 1, 1, 0, 0, 0, 'gb200_reference'),
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 128, 4096, 2, 4, 1, 32, 1, 1, 1024, 1, 1, 0, 0, 0, 'gb200_reference'),
    
    # Pre-train: Qwen3 30B A3B
    Config('pre_train', 'qwen3', '30b_a3b', 'gb200', 'bf16', 8, 4096, 1, 1, 1, 8, 1, 1, 512, 1, 1, 0, 0, 0, 'gb200_reference'),
    
    # Pre-train: Qwen3 235B A22B
    Config('pre_train', 'qwen3', '235b_a22b', 'gb200', 'bf16', 64, 4096, 2, 1, 1, 64, 1, 1, 1024, 1, 1, 0, 0, 0, 'gb200_reference'),
    
    # SFT: LLAMA3 8B
    Config('sft', 'llama3', '8b', 'gb200', 'bf16', 8, 16384, 1, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0, 'gb200_reference'),
    Config('sft', 'llama3', '8b', 'gb200', 'fp8', 8, 16384, 1, 1, 1, 1, 1, 1, 8, 0, 1, 0, 0, 0, 'gb200_reference'),
    
    # SFT: LLAMA3 70B
    Config('sft', 'llama3', '70b', 'gb200', 'bf16', 32, 4096, 2, 4, 1, 1, 5, 1, 32, 0, 1, 0, 0, 0, 'gb200_reference'),
    Config('sft', 'llama3', '70b', 'gb200', 'fp8', 32, 4096, 2, 4, 1, 1, 5, 1, 32, 0, 1, 0, 0, 0, 'gb200_reference'),
    
    # LORA: LLAMA3 70B
    Config('lora', 'llama3', '70b', 'gb200', 'bf16', 8, 2048, 1, 4, 1, 1, 20, 1, 64, 0, 1, 0, 0, 0, 'gb200_reference'),
    Config('lora', 'llama3', '70b', 'gb200', 'fp8', 8, 2048, 1, 4, 1, 1, 20, 1, 64, 0, 1, 0, 0, 0, 'gb200_reference'),
]

# Your current experimental configurations (from what we've seen)
YOUR_CONFIGS = [
    # From run_reference_sweep_gb200.sh - LLAMA3 8B BF16 (pretrain)
    # Appears to match reference
    
    # From run_reference_sweep_gb200.sh - QWEN3_30B_A3B BF16
    # Reference: 8 GPUs, TP=1, PP=1, CP=1, EP=8, MBS=1, GBS=512
    # Your config in script: 8 GPUs, TP=1, PP=1, CP=1, EP=8, MBS=4, GBS=512
    Config('pre_train', 'qwen3', '30b_a3b', 'gb200', 'bf16', 8, 4096, 1, 1, 1, 8, 1, 4, 512, 1, 1, 0, 0, 0, 'your_run_reference_sweep'),
    
    # From run_nemo2507_validation_sweep.sh - LLAMA3 8B BF16
    Config('pre_train', 'llama3', '8b', 'gb200', 'bf16', 8, 8192, 1, 1, 1, 1, 1, 2, 128, 0, 1, 0, 0, 0, 'your_nemo2507_validation'),
    
    # From run_nemo2507_validation_sweep.sh - LLAMA3 70B BF16
    Config('pre_train', 'llama3', '70b', 'gb200', 'bf16', 64, 8192, 1, 1, 1, 1, 1, 1, 128, 0, 1, 0, 0, 0, 'your_nemo2507_validation'),
    
    # From run_nemo2507_validation_sweep.sh - DeepSeekV3 Large GBS
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 256, 4096, 2, 4, 1, 64, 1, 1, 2048, 1, 1, 0, 0, 0, 'your_nemo2507_validation'),
]


def compare_configs(ref: Config, your: Config) -> Tuple[bool, List[str]]:
    """Compare two configs and return (matches, differences)"""
    differences = []
    
    # Compare key parameters
    important_fields = [
        'num_gpus', 'seq_len', 'tp_size', 'pp_size', 'cp_size', 
        'ep_size', 'vp_size', 'mbs', 'gbs', 'etp_size',
        'use_mcore_fsdp', 'recompute_layers', 'activation_offload_layers'
    ]
    
    for field in important_fields:
        ref_val = getattr(ref, field)
        your_val = getattr(your, field)
        if ref_val != your_val:
            differences.append(f"  {field:30s}: REF={ref_val:6} ≠ YOUR={your_val:6}")
    
    # Check derived values
    if ref.dp_size != your.dp_size:
        differences.append(f"  {'dp_size (calculated)':30s}: REF={ref.dp_size:6} ≠ YOUR={your.dp_size:6}")
    
    if ref.ga != your.ga:
        differences.append(f"  {'ga (calculated)':30s}: REF={ref.ga:6} ≠ YOUR={your.ga:6}")
    
    return len(differences) == 0, differences


def main(filter_model: str = None, verbose: bool = False):
    """Main comparison logic"""
    
    print("=" * 100)
    print("GB200 Configuration Comparison Report")
    print("Comparing your experimental configs against official GB200 reference")
    print("=" * 100)
    print()
    
    # Build lookup for your configs
    your_configs_map = {}
    for config in YOUR_CONFIGS:
        your_configs_map[config.key] = config
    
    # Track results
    matches = []
    mismatches = []
    missing_in_your_setup = []
    
    # Compare each reference config
    for ref_config in REFERENCE_CONFIGS:
        if filter_model and filter_model.lower() not in ref_config.key.lower():
            continue
        
        your_config = your_configs_map.get(ref_config.key)
        
        if your_config is None:
            missing_in_your_setup.append(ref_config)
        else:
            is_match, diffs = compare_configs(ref_config, your_config)
            if is_match:
                matches.append((ref_config, your_config))
            else:
                mismatches.append((ref_config, your_config, diffs))
    
    # Print results
    print(f"\n{'=' * 100}")
    print(f"SUMMARY")
    print(f"{'=' * 100}")
    print(f"  ✓ Matching configs:        {len(matches)}")
    print(f"  ✗ Mismatched configs:      {len(mismatches)}")
    print(f"  ⚠ Missing from your setup: {len(missing_in_your_setup)}")
    print(f"{'=' * 100}\n")
    
    # Print matches
    if matches and verbose:
        print(f"\n{'─' * 100}")
        print("✓ MATCHING CONFIGURATIONS")
        print(f"{'─' * 100}")
        for ref, your in matches:
            print(f"\n  {ref.key}")
            print(f"    ✓ All parameters match between reference and your setup")
            print(f"    Source: {your.source}")
    
    # Print mismatches
    if mismatches:
        print(f"\n{'─' * 100}")
        print("✗ MISMATCHED CONFIGURATIONS")
        print(f"{'─' * 100}")
        for ref, your, diffs in mismatches:
            print(f"\n  {ref.key}")
            print(f"    Source: {your.source}")
            print(f"    Differences:")
            for diff in diffs:
                print(f"    {diff}")
    
    # Print missing
    if missing_in_your_setup:
        print(f"\n{'─' * 100}")
        print("⚠ MISSING FROM YOUR SETUP")
        print(f"{'─' * 100}")
        for ref in missing_in_your_setup:
            print(f"\n  {ref.key}")
            print(f"    GPUs: {ref.num_gpus}, SEQ: {ref.seq_len}")
            print(f"    TP={ref.tp_size}, PP={ref.pp_size}, CP={ref.cp_size}, EP={ref.ep_size}, VP={ref.vp_size}")
            print(f"    MBS={ref.mbs}, GBS={ref.gbs}, FSDP={ref.use_mcore_fsdp}")
            if ref.recompute_layers > 0:
                print(f"    Recompute Layers: {ref.recompute_layers}")
            if ref.activation_offload_layers > 0:
                print(f"    CPU Offload Layers: {ref.activation_offload_layers}")
    
    # Specific issue callouts
    print(f"\n{'=' * 100}")
    print("SPECIFIC ISSUES TO ADDRESS")
    print(f"{'=' * 100}")
    
    issues = []
    
    # Check Qwen3 30B MBS difference
    qwen30b_ref = next((c for c in REFERENCE_CONFIGS if c.model == 'qwen3' and c.size == '30b_a3b' and c.dtype == 'bf16'), None)
    qwen30b_your = next((c for c in YOUR_CONFIGS if c.model == 'qwen3' and c.size == '30b_a3b' and c.dtype == 'bf16'), None)
    
    if qwen30b_ref and qwen30b_your and qwen30b_ref.mbs != qwen30b_your.mbs:
        issues.append({
            'model': 'Qwen3 30B A3B (BF16)',
            'issue': f'MBS mismatch',
            'reference': f'MBS={qwen30b_ref.mbs}',
            'yours': f'MBS={qwen30b_your.mbs}',
            'impact': 'Different batch size may affect performance comparison',
            'action': f'Change MBS from {qwen30b_your.mbs} to {qwen30b_ref.mbs} in run_reference_sweep_gb200.sh'
        })
    
    # Check LLAMA3 70B recompute layers
    llama70b_ref = next((c for c in REFERENCE_CONFIGS if c.model == 'llama3' and c.size == '70b' and c.dtype == 'bf16' and c.task == 'pre_train'), None)
    llama70b_your = next((c for c in YOUR_CONFIGS if c.model == 'llama3' and c.size == '70b' and c.dtype == 'bf16' and c.task == 'pre_train'), None)
    
    if llama70b_ref and llama70b_your and llama70b_ref.recompute_layers != llama70b_your.recompute_layers:
        issues.append({
            'model': 'LLAMA3 70B (BF16)',
            'issue': f'Recompute layers mismatch',
            'reference': f'recompute_layers={llama70b_ref.recompute_layers}',
            'yours': f'recompute_layers={llama70b_your.recompute_layers}',
            'impact': 'Missing activation recomputation affects memory usage and performance',
            'action': f'Enable recompute_layers=20 with FSDP for LLAMA3 70B BF16'
        })
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"\n  Issue #{i}: {issue['model']}")
            print(f"    Problem:    {issue['issue']}")
            print(f"    Reference:  {issue['reference']}")
            print(f"    Your setup: {issue['yours']}")
            print(f"    Impact:     {issue['impact']}")
            print(f"    Action:     {issue['action']}")
    else:
        print("\n  ✓ No critical issues found!")
    
    print(f"\n{'=' * 100}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare your configs against GB200 reference"
    )
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Show detailed information including matches")
    parser.add_argument('--model', '-m', type=str,
                        help="Filter by model name (e.g., llama3, qwen3)")
    
    args = parser.parse_args()
    
    main(filter_model=args.model, verbose=args.verbose)

