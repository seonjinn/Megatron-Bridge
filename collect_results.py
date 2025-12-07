#!/usr/bin/env python3
"""
Collect and summarize results from multiple Megatron-Bridge experiments.

Usage:
    python3 collect_results.py --sweep-dir ./exp_logs/sweeps/20251125_120000
    python3 collect_results.py --exp-dirs ./exp_logs/experiments/llama3_8b_*/
    python3 collect_results.py --latest 5  # Collect from latest 5 experiments
    
Metric Calculation Options:
    # Default: skip first 3 warmup steps, average all remaining
    python3 collect_results.py --scan-all
    
    # NeMo-RL style (RECOMMENDED for NeMo-RL comparison):
    # Exactly matches get_wandb_log_for_nemorl.py calculation
    # at_step=5, average_steps=5 → indices 3,4,5,6,7 → iteration 4,5,6,7,8
    python3 collect_results.py --scan-all --nemorl-style
    python3 collect_results.py --scan-all --match-config nemorl_reference_configs.yaml --use-reference
    
    # NeMo 25.07 style: 50 steps, average steps 35-50
    python3 collect_results.py --scan-all --metric-range 35:50
    python3 collect_results.py --scan-all --match-config gb200_reference_configs.yaml --use-reference
    
    # Use last N iterations only
    python3 collect_results.py --scan-all --use-last-n 15
    
    # Custom warmup steps
    python3 collect_results.py --scan-all --warmup-steps 5
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

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    # Experiment identification
    exp_name: str = ""
    exp_dir: str = ""
    log_file_path: str = ""  # Full path to log file (for clickable links)
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
    fsdp: int = 0  # Megatron FSDP (0 = disabled, 1 = enabled)
    tp: int = 1
    pp: int = 1
    cp: int = 1
    dp: int = 1
    ep: int = 1
    vp: int = 1  # Virtual Pipeline Parallelism
    etp: int = 1  # Expert Tensor Parallelism
    dp_per_ep: int = 1  # DP per expert group (DP / EP) - relevant for MoE models
    
    # Training configuration
    global_batch_size: int = 0
    micro_batch_size: int = 0
    sequence_length: int = 0
    max_steps: int = 0
    precision: str = ""
    task: str = ""  # pretrain, sft, lora, etc.
    
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


def parse_exp_name(exp_name: str) -> dict:
    """Parse experiment name to extract model info.
    
    Format: {model_name}_{model_size}_llm_{task}_{precision}
    Examples:
        llama3_8b_llm_pretrain_bf16 -> model=llama3, size=8b, task=pretrain, precision=bf16
        qwen3_32b_llm_pretrain_bf16 -> model=qwen3, size=32b, task=pretrain, precision=bf16
        qwen3_30b_a3b_llm_pretrain_bf16 -> model=qwen3, size=30b_a3b, task=pretrain, precision=bf16
    """
    info = {'model_name': '', 'model_size': '', 'task': '', 'precision': ''}
    
    # Try to parse the format: model_size_llm_task_precision
    # Handle cases like llama3_8b_llm_pretrain_bf16 or qwen3_30b_a3b_llm_pretrain_bf16
    
    # Check for precision suffix
    for prec in ['bf16', 'fp8_cs', 'fp8_mx', 'fp16', 'nvfp4']:
        if exp_name.endswith(f'_{prec}'):
            info['precision'] = prec
            exp_name = exp_name[:-len(prec)-1]
            break
    
    # Check for task
    for task in ['pretrain', 'sft', 'lora', 'finetune']:
        if f'_llm_{task}' in exp_name:
            info['task'] = task
            # Split by _llm_task to get model part
            parts = exp_name.split(f'_llm_{task}')
            if parts:
                model_part = parts[0]
                # Parse model_name and model_size
                # Common patterns: llama3_8b, qwen3_32b, qwen3_30b_a3b, deepseek_v3
                if '_' in model_part:
                    first_underscore = model_part.find('_')
                    # Check if it looks like modelname_size pattern
                    potential_name = model_part[:first_underscore]
                    potential_size = model_part[first_underscore+1:]
                    info['model_name'] = potential_name
                    info['model_size'] = potential_size
                else:
                    info['model_name'] = model_part
            break
    
    return info


def parse_wandb_exp_name(wandb_name: str) -> dict:
    """Parse wandb_exp_name to extract model info and parallelism.
    
    Formats:
      - rl_{model}_{size}_{precision}_tp{X}pp{Y}cp{Z}ep{W}dp{V}_gbs{G}_seq{S}
      - ref_{model}_{size}_{precision}_tp{X}pp{Y}...
      - {model}-{size}-{gpus}gpus-tp{X}-pp{Y}-... (legacy format)
    Examples:
      - rl_llama3_8b_bf16_tp1pp1cp1ep1dp8_gbs512_seq8192
      - rl_qwen3_32b_bf16_tp4pp1cp1ep1dp4_gbs512
      - ref_llama3_8b_fp8_cs_tp1pp1cp1ep1dp8
      - llama3-8b-16gpus-tp1-pp1-cp1-ep1-dp16
    """
    info = {
        'model_name': '', 'model_size': '', 'precision': '',
        'tp': 0, 'pp': 0, 'cp': 0, 'ep': 0, 'dp': 0,
        'global_batch_size': 0, 'sequence_length': 0
    }
    
    # Check if legacy hyphen-separated format: llama3-8b-16gpus-tp1-pp1-cp1-ep1-dp16
    if '-tp' in wandb_name and '-pp' in wandb_name:
        # Extract parallelism from hyphen format: tp1-pp1-cp1-ep1-dp16
        tp_match = re.search(r'-tp(\d+)', wandb_name)
        pp_match = re.search(r'-pp(\d+)', wandb_name)
        cp_match = re.search(r'-cp(\d+)', wandb_name)
        ep_match = re.search(r'-ep(\d+)', wandb_name)
        dp_match = re.search(r'-dp(\d+)', wandb_name)
        
        if tp_match:
            info['tp'] = int(tp_match.group(1))
        if pp_match:
            info['pp'] = int(pp_match.group(1))
        if cp_match:
            info['cp'] = int(cp_match.group(1))
        if ep_match:
            info['ep'] = int(ep_match.group(1))
        if dp_match:
            info['dp'] = int(dp_match.group(1))
        
        # Extract model name and size: llama3-8b-...
        parts = wandb_name.split('-')
        if len(parts) >= 2:
            info['model_name'] = parts[0]  # llama3
            # Find size - look for pattern like 8b, 70b, 30b_a3b
            for i, part in enumerate(parts[1:], start=1):
                if re.match(r'\d+b', part) or re.match(r'\d+gpus', part):
                    if re.match(r'\d+b', part):
                        info['model_size'] = part
                    break
        return info
    
    # New underscore-separated format
    # Extract parallelism: tp1pp1cp1ep1dp8
    parallelism_match = re.search(r'tp(\d+)pp(\d+)cp(\d+)ep(\d+)dp(\d+)', wandb_name)
    if parallelism_match:
        info['tp'] = int(parallelism_match.group(1))
        info['pp'] = int(parallelism_match.group(2))
        info['cp'] = int(parallelism_match.group(3))
        info['ep'] = int(parallelism_match.group(4))
        info['dp'] = int(parallelism_match.group(5))
    
    # Extract gbs
    gbs_match = re.search(r'gbs(\d+)', wandb_name)
    if gbs_match:
        info['global_batch_size'] = int(gbs_match.group(1))
    
    # Extract seq length
    seq_match = re.search(r'seq(\d+)', wandb_name)
    if seq_match:
        info['sequence_length'] = int(seq_match.group(1))
    
    # Extract precision (bf16, fp8_cs, fp8_mx, fp8, fp16, nvfp4, etc.)
    # Check for fp8_cs or fp8_mx first, then fallback to simpler patterns
    prec_match = re.search(r'_(fp8_cs|fp8_mx|bf16|fp8|fp16|nvfp4)_', wandb_name)
    if prec_match:
        info['precision'] = prec_match.group(1)
    else:
        # Try without trailing underscore for patterns at the end
        prec_match = re.search(r'_(fp8_cs|fp8_mx|bf16|fp8|fp16|nvfp4)(?:_|$)', wandb_name)
        if prec_match:
            info['precision'] = prec_match.group(1)
    
    # Extract model name and size from beginning
    # Format: rl_llama3_8b_bf16_... or ref_llama3_8b_fp8_cs_...
    parts = wandb_name.split('_')
    prefix_idx = 0
    if len(parts) >= 1 and parts[0] in ['rl', 'ref', 'train', 'test']:
        prefix_idx = 1
    
    if len(parts) > prefix_idx:
        info['model_name'] = parts[prefix_idx]  # llama3, qwen3, etc.
        # Find size - it's between model name and precision/parallelism marker
        precision_markers = ['bf16', 'fp8', 'fp16', 'nvfp4']
        for i, part in enumerate(parts[prefix_idx + 1:], start=prefix_idx + 1):
            # Stop at precision marker or parallelism marker
            if part in precision_markers or part.startswith('tp'):
                # Everything between model_name and here is model_size
                size_parts = parts[prefix_idx + 1:i]
                if size_parts:
                    info['model_size'] = '_'.join(size_parts)
                break
    
    return info


def parse_config_yaml(config_path: Path) -> dict:
    """Parse ConfigContainer.yaml to extract experiment configuration.
    
    Returns dict with: model_name, model_size, precision, task, 
                      tp, pp, cp, ep, dp, vp, etp,
                      global_batch_size, micro_batch_size, sequence_length,
                      num_gpus, num_nodes, gpus_per_node, gpu_type
    """
    info = {
        'model_name': '', 'model_size': '', 'precision': '', 'task': '',
        'tp': 0, 'pp': 0, 'cp': 0, 'ep': 0, 'dp': 0, 'vp': 0, 'etp': 0,
        'global_batch_size': 0, 'micro_batch_size': 0, 'sequence_length': 0,
        'num_gpus': 0, 'num_nodes': 0, 'gpus_per_node': 0, 'gpu_type': '', 'fsdp': 0
    }
    
    if not YAML_AVAILABLE:
        return info
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            return info
        
        # Extract model info from model._target_
        model_cfg = config.get('model', {})
        if model_cfg:
            target = model_cfg.get('_target_', '')
            # Example: megatron.bridge.models.llama.llama_provider.LlamaModelProvider
            if 'llama' in target.lower():
                info['model_name'] = 'llama3'
            elif 'qwen' in target.lower():
                info['model_name'] = 'qwen3'
            elif 'deepseek' in target.lower():
                info['model_name'] = 'deepseek'
            elif 'mistral' in target.lower():
                info['model_name'] = 'mistral'
            elif 'nemotron' in target.lower():
                info['model_name'] = 'nemotron'
            
            # Extract parallelism from model config
            info['tp'] = model_cfg.get('tensor_model_parallel_size', 0)
            info['pp'] = model_cfg.get('pipeline_model_parallel_size', 0)
            info['cp'] = model_cfg.get('context_parallel_size', 0)
            info['ep'] = model_cfg.get('expert_model_parallel_size', 0)
            info['vp'] = model_cfg.get('virtual_pipeline_model_parallel_size', 0) or 0
            info['etp'] = model_cfg.get('expert_tensor_parallel_size', 0) or 0
            info['sequence_length'] = model_cfg.get('seq_length', 0)
            
            # Extract precision (check fp8 first, then bf16/fp16)
            fp8_val = model_cfg.get('fp8', None)
            if fp8_val and fp8_val not in [False, 'none', 'None']:
                # fp8 is enabled - check for fp8_cs or fp8_mx in wandb_exp_name later
                info['precision'] = 'fp8'
            elif model_cfg.get('bf16', False):
                info['precision'] = 'bf16'
            elif model_cfg.get('fp16', False):
                info['precision'] = 'fp16'
        
        # Extract training config
        train_cfg = config.get('train', {})
        if train_cfg:
            info['global_batch_size'] = train_cfg.get('global_batch_size', 0)
            info['micro_batch_size'] = train_cfg.get('micro_batch_size', 0)
        
        # Extract DP from comm_overlap config
        comm_cfg = config.get('comm_overlap', {})
        if comm_cfg:
            dp_from_comm = comm_cfg.get('data_parallel_size', 0)
            if dp_from_comm > 0:
                info['dp'] = dp_from_comm
        
        # Extract FSDP from dist config
        dist_cfg = config.get('dist', {})
        if dist_cfg:
            if dist_cfg.get('use_megatron_fsdp', False) or dist_cfg.get('use_torch_fsdp2', False):
                info['fsdp'] = 1
        
        # Extract info from wandb_exp_name (more detailed)
        logger_cfg = config.get('logger', {})
        if logger_cfg:
            wandb_name = logger_cfg.get('wandb_exp_name', '')
            if wandb_name:
                wandb_info = parse_wandb_exp_name(wandb_name)
                # Use wandb info to fill in missing values or get more detailed precision
                if not info['model_name'] and wandb_info['model_name']:
                    info['model_name'] = wandb_info['model_name']
                if not info['model_size'] and wandb_info['model_size']:
                    info['model_size'] = wandb_info['model_size']
                # Use wandb precision if more specific (e.g., fp8_cs vs fp8)
                if wandb_info['precision']:
                    if not info['precision'] or (info['precision'] == 'fp8' and 
                        wandb_info['precision'] in ['fp8_cs', 'fp8_mx']):
                        info['precision'] = wandb_info['precision']
                # Parallelism from wandb_name as fallback
                if info['tp'] == 0 and wandb_info['tp'] > 0:
                    info['tp'] = wandb_info['tp']
                if info['pp'] == 0 and wandb_info['pp'] > 0:
                    info['pp'] = wandb_info['pp']
                if info['cp'] == 0 and wandb_info['cp'] > 0:
                    info['cp'] = wandb_info['cp']
                if info['ep'] == 0 and wandb_info['ep'] > 0:
                    info['ep'] = wandb_info['ep']
                if info['dp'] == 0 and wandb_info['dp'] > 0:
                    info['dp'] = wandb_info['dp']
                if info['global_batch_size'] == 0 and wandb_info['global_batch_size'] > 0:
                    info['global_batch_size'] = wandb_info['global_batch_size']
                if info['sequence_length'] == 0 and wandb_info['sequence_length'] > 0:
                    info['sequence_length'] = wandb_info['sequence_length']
        
        # Calculate num_gpus from parallelism if we have enough info
        # num_gpus = TP * PP * CP * DP (same for both dense and MoE)
        # Note: EP is NOT multiplied because EP is a subdivision within DP, not a separate dimension
        # For MoE: DP = world_size / (TP * PP * CP), and EP divides DP into expert groups
        if info['tp'] > 0 and info['pp'] > 0 and info['dp'] > 0:
            cp = info['cp'] if info['cp'] > 0 else 1
            info['num_gpus'] = info['tp'] * info['pp'] * cp * info['dp']
        
        # Default gpus_per_node assumption (GB200 = 4, others = 8)
        if info['num_gpus'] > 0:
            info['gpus_per_node'] = 4  # Default for GB200
            info['num_nodes'] = info['num_gpus'] // info['gpus_per_node']
            info['gpu_type'] = 'GB200'  # Default assumption
        
        # Extract task and model_size from parent directory path
        # Path format: .../llama3_8b_llm_pretrain_bf16/.../configs/ConfigContainer.yaml
        try:
            # Go up to find the experiment type directory
            exp_type_dir = config_path.parent.parent.parent.parent.name
            for task in ['pretrain', 'sft', 'lora', 'finetune']:
                if f'_llm_{task}_' in exp_type_dir or f'_llm_{task}' in exp_type_dir:
                    info['task'] = task
                    # Also try to extract model_size from directory if not already set
                    if not info['model_size']:
                        # Format: llama3_8b_llm_pretrain_bf16 or qwen3_30b_a3b_llm_pretrain_bf16
                        parts = exp_type_dir.split(f'_llm_{task}')
                        if parts:
                            model_part = parts[0]
                            # Extract size from model_part (e.g., llama3_8b -> 8b)
                            if info['model_name'] and f"{info['model_name']}_" in model_part:
                                size_part = model_part.replace(f"{info['model_name']}_", "")
                                if size_part:
                                    info['model_size'] = size_part
                    break
        except Exception:
            pass
        
    except Exception as e:
        # Silently fail and return empty info
        pass
    
    return info


def parse_log_file(log_path: Path) -> ExperimentResult:
    """Parse a Megatron-Bridge log file and extract metrics."""
    result = ExperimentResult()
    result.exp_dir = str(log_path.parent.parent)
    result.exp_name = log_path.parent.parent.name
    result.log_file_path = str(log_path.resolve())  # Store full absolute path for clickable links
    
    # First, try to parse ConfigContainer.yaml for accurate info
    config_path = log_path.parent / "configs" / "ConfigContainer.yaml"
    if config_path.exists():
        config_info = parse_config_yaml(config_path)
        # Apply config info to result
        if config_info['model_name']:
            result.model_name = config_info['model_name']
        if config_info['model_size']:
            result.model_size = config_info['model_size']
        if config_info['precision']:
            result.precision = config_info['precision']
        if config_info['task']:
            result.task = config_info['task']
        if config_info['tp'] > 0:
            result.tp = config_info['tp']
        if config_info['pp'] > 0:
            result.pp = config_info['pp']
        if config_info['cp'] > 0:
            result.cp = config_info['cp']
        if config_info['ep'] > 0:
            result.ep = config_info['ep']
        if config_info['dp'] > 0:
            result.dp = config_info['dp']
        if config_info['vp'] > 0:
            result.vp = config_info['vp']
        if config_info['etp'] > 0:
            result.etp = config_info['etp']
        if config_info['global_batch_size'] > 0:
            result.global_batch_size = config_info['global_batch_size']
        if config_info['micro_batch_size'] > 0:
            result.micro_batch_size = config_info['micro_batch_size']
        if config_info['sequence_length'] > 0:
            result.sequence_length = config_info['sequence_length']
        if config_info['num_gpus'] > 0:
            result.num_gpus = config_info['num_gpus']
        if config_info['num_nodes'] > 0:
            result.num_nodes = config_info['num_nodes']
        if config_info['gpus_per_node'] > 0:
            result.gpus_per_node = config_info['gpus_per_node']
        if config_info['gpu_type']:
            result.gpu_type = config_info['gpu_type']
        if config_info['fsdp'] > 0:
            result.fsdp = config_info['fsdp']
    
    # Fallback: parse model info from experiment directory name if not found in config
    if not result.model_name or not result.precision:
        exp_type_dir = log_path.parent.parent.parent.name  # e.g., llama3_8b_llm_pretrain_bf16
        exp_info = parse_exp_name(exp_type_dir)
        if not result.model_name and exp_info['model_name']:
            result.model_name = exp_info['model_name']
        if not result.model_size and exp_info['model_size']:
            result.model_size = exp_info['model_size']
        if not result.task and exp_info['task']:
            result.task = exp_info['task']
        if not result.precision and exp_info['precision']:
            result.precision = exp_info['precision']
    
    # Pattern for iteration logs - two formats:
    # Format 1 (older): iteration X/Y | ... throughput per GPU (TFLOP/s/GPU): Z ...
    # Format 2 (newer): iteration X/Y | ... elapsed time per iteration (ms): Z | ... global batch size: N
    iteration_pattern_v1 = re.compile(
        r'iteration\s+(\d+)/\s*(\d+)\s*\|'
        r'.*elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|'
        r'.*throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)\s*\|'
        r'(?:.*Tokens/sec/GPU:\s*([\d.]+)\s*\|)?'
        r'.*global batch size:\s*(\d+)'
    )
    
    # Format 2: Newer logs without inline TFLOP/s - capture iteration, time, and GBS
    iteration_pattern_v2 = re.compile(
        r'iteration\s+(\d+)/\s*(\d+)\s*\|'
        r'.*elapsed time per iteration \(ms\):\s*([\d.]+)\s*\|'
        r'.*global batch size:\s*(\d+)'
    )
    
    # Pattern for Step Time line (contains TFLOP/s info)
    # Format: "Step Time : 27.67s GPU utilization: 970.0MODEL_TFLOP/s/GPU"
    step_time_pattern = re.compile(
        r'Step Time\s*:\s*([\d.]+)s\s+GPU utilization:\s*([\d.]+)MODEL_TFLOP/s/GPU'
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
        'vp': re.compile(r'virtual_pipeline_model_parallel_size[=:\s]+(\d+)'),
        'etp': re.compile(r'expert_tensor_parallel_size[=:\s]+(\d+)'),
        'micro_batch_size': re.compile(r'micro_batch_size[=:\s]+(\d+)'),
        'sequence_length': re.compile(r'(?:seq_length|sequence_length)[=:\s]+(\d+)'),
    }
    
    # Boolean patterns (FSDP)
    fsdp_pattern = re.compile(r'use_megatron_fsdp[=:\s]+(True|true|1)')
    
    # Additional patterns to try for num_gpus (world_size)
    num_gpus_patterns = [
        re.compile(r'world_size[=:\s]+(\d+)', re.IGNORECASE),
        re.compile(r'world size[=:\s]+(\d+)', re.IGNORECASE),
        re.compile(r'Running on (\d+) GPUs', re.IGNORECASE),
        re.compile(r'using (\d+) GPUs', re.IGNORECASE),
        re.compile(r'num_gpus[=:\s]+(\d+)', re.IGNORECASE),
    ]
    
    # GPU type patterns
    gpu_type_patterns = [
        (re.compile(r'GB200', re.IGNORECASE), 'GB200'),
        (re.compile(r'B200', re.IGNORECASE), 'B200'),
        (re.compile(r'H100', re.IGNORECASE), 'H100'),
        (re.compile(r'A100', re.IGNORECASE), 'A100'),
    ]
    
    # Gpus per node patterns (for calculating num_nodes)
    gpus_per_node_patterns = [
        re.compile(r'gpus_per_node[=:\s]+(\d+)', re.IGNORECASE),
        re.compile(r'SLURM_GPUS_PER_NODE[=:\s]+(\d+)', re.IGNORECASE),
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
            
            # Parse FSDP (boolean)
            if fsdp_pattern.search(content):
                result.fsdp = 1
            
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
            
            # Extract GPU type
            for pattern, gpu_name in gpu_type_patterns:
                if pattern.search(content):
                    result.gpu_type = gpu_name
                    break
            
            # Extract gpus_per_node and calculate num_nodes
            for pattern in gpus_per_node_patterns:
                match = pattern.search(content)
                if match:
                    result.gpus_per_node = int(match.group(1))
                    break
            
            # Calculate num_nodes if we have num_gpus and gpus_per_node
            if result.num_gpus > 0 and result.gpus_per_node > 0:
                result.num_nodes = result.num_gpus // result.gpus_per_node
            elif result.num_gpus > 0:
                # Default assumption: 4 GPUs per node for GB200, 8 for others
                if result.gpu_type == 'GB200':
                    result.gpus_per_node = 4
                else:
                    result.gpus_per_node = 8
                result.num_nodes = result.num_gpus // result.gpus_per_node
            
            # Parse memory info
            mem_match = memory_pattern.search(content)
            if mem_match:
                result.memory_allocated_gb = float(mem_match.group(1))
                result.memory_reserved_gb = float(mem_match.group(2))
            
            # Parse Step Time lines first (to get TFLOP/s info for v2 format)
            step_time_data = []
            for match in step_time_pattern.finditer(content):
                step_time_data.append({
                    'step_time': float(match.group(1)),
                    'tflops_per_gpu': float(match.group(2)),
                })
            
            # Parse iteration data - try v1 format first (with inline TFLOP/s)
            for match in iteration_pattern_v1.finditer(content):
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
            
            # If v1 pattern didn't match, try v2 pattern (newer format without inline TFLOP/s)
            if not iterations:
                for match in iteration_pattern_v2.finditer(content):
                    iter_num = int(match.group(1))
                    max_iter = int(match.group(2))
                    iter_time = float(match.group(3))
                    gbs = int(match.group(4))
                    
                    # Match TFLOP/s from step_time_data if available (index-1 because Step Time comes before iteration log)
                    tflops = 0.0
                    idx = len(iterations)
                    if idx < len(step_time_data):
                        tflops = step_time_data[idx].get('tflops_per_gpu', 0.0)
                    
                    iterations.append({
                        'iteration': iter_num,
                        'iteration_time_ms': iter_time,
                        'tflops_per_gpu': tflops,
                        'tokens_per_sec_per_gpu': 0.0,  # Not available in v2 format
                        'global_batch_size': gbs,
                    })
                    
                    result.max_steps = max_iter
                    result.global_batch_size = gbs
            
            # Check for OOM (Out of Memory) errors
            oom_patterns = [
                'out of memory',
                'OutOfMemoryError',
                'CUDA out of memory',
                'OOM',
                'RuntimeError: CUDA error: out of memory',
                'torch.cuda.OutOfMemoryError',
            ]
            has_oom = any(pattern.lower() in content.lower() for pattern in oom_patterns)
            
            # Check for job failure patterns (cancelled, crashed, timeout, etc.)
            failure_patterns = [
                'CANCELLED AT',
                'CANCELLED BY',
                'SIGTERM',
                'SIGKILL',
                'Traceback (most recent call last)',
                'RuntimeError:',
                'Exception raised',
                'Broken pipe',
                'Connection refused',
                'TIMEOUT',
            ]
            has_failure = any(pattern in content for pattern in failure_patterns)
            
            # Check completion status
            # Multiple completion markers:
            # - "Training completed"
            # - "[after training is done]"
            # - Last iteration reached max_steps
            training_done = ('Training completed' in content or 
                           '[after training is done]' in content or
                           (iterations and iterations[-1]['iteration'] >= result.max_steps > 0))
            
            if training_done:
                result.status = 'completed'
            elif has_oom:
                result.status = 'oom'  # Out of Memory
            elif has_failure:
                result.status = 'failed'
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
        tflops_list = [it['tflops_per_gpu'] for it in stable_iterations if it['tflops_per_gpu'] > 0]
        time_list = [it['iteration_time_ms'] for it in stable_iterations if it['iteration_time_ms'] > 0]
        
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
        
        # If tokens_per_sec_per_gpu is 0 but we have iter_time, gbs, seq_len, and num_gpus, calculate it
        if result.tokens_per_sec_per_gpu == 0 and result.iteration_time_ms > 0:
            if result.global_batch_size > 0 and result.sequence_length > 0 and result.num_gpus > 0:
                tokens_per_iter = result.global_batch_size * result.sequence_length
                time_per_iter_sec = result.iteration_time_ms / 1000.0
                result.tokens_per_sec_per_gpu = (tokens_per_iter / time_per_iter_sec) / result.num_gpus
    
    result.tokens_per_sec_total = result.tokens_per_sec_per_gpu * result.num_gpus
    result.total_iterations = len(iterations)
    result.iteration_data = iterations
    
    # Calculate samples per second
    if result.iteration_time_ms > 0 and result.global_batch_size > 0:
        result.samples_per_sec = result.global_batch_size / (result.iteration_time_ms / 1000)
    
    return result


def recalculate_metrics(result: ExperimentResult, 
                        warmup_steps: int = 3, 
                        metric_range: str = None,
                        use_last_n: int = None,
                        nemorl_style: bool = False,
                        at_step: int = 5,
                        average_steps: int = 5) -> ExperimentResult:
    """
    Recalculate metrics with different parameters.
    
    Args:
        result: ExperimentResult with iteration_data
        warmup_steps: Number of warmup steps to skip (default: 3)
        metric_range: Step range like "35:50" (start:end, 1-indexed, inclusive)
        use_last_n: Use only last N iterations
        nemorl_style: Use NeMo-RL style calculation (same as get_wandb_log_for_nemorl.py)
        at_step: Center step for NeMo-RL style (default: 5)
        average_steps: Number of steps to average for NeMo-RL style (default: 5)
        
    Returns:
        Updated ExperimentResult with recalculated metrics
        
    NeMo-RL Style Calculation (from get_wandb_log_for_nemorl.py):
        min_step = at_step - average_steps // 2  # 5 - 2 = 3 (0-indexed)
        max_step = at_step + average_steps // 2 - 1 + (average_steps % 2)  # 5 + 2 - 1 + 1 = 7
        history.loc[min_step:max_step]  # pandas loc (inclusive)
        
        For at_step=5, average_steps=5:
        - min_step = 3, max_step = 7 (0-indexed, inclusive)
        - Selects indices 3, 4, 5, 6, 7 (5 data points)
        - In 1-indexed iteration terms: iteration 4, 5, 6, 7, 8
    """
    iterations = getattr(result, 'iteration_data', [])
    if not iterations:
        return result
    
    # Determine which iterations to use
    if nemorl_style:
        # NeMo-RL style: exactly matches get_wandb_log_for_nemorl.py
        # Calculate step range using same formula as NeMo-RL
        min_step = at_step - average_steps // 2  # 0-indexed start (inclusive)
        max_step = at_step + average_steps // 2 - 1 + (average_steps % 2)  # 0-indexed end (inclusive)
        
        # iterations list is 0-indexed (iterations[0] = iteration 1 in log)
        # So we use slice [min_step:max_step+1] to include max_step
        if len(iterations) > max_step:
            stable_iterations = iterations[min_step:max_step + 1]
        else:
            # Not enough iterations, use all available after min_step
            stable_iterations = iterations[min_step:] if len(iterations) > min_step else iterations
    elif metric_range:
        # Parse range like "35:50" (1-indexed, inclusive)
        try:
            parts = metric_range.split(':')
            start = int(parts[0]) - 1  # Convert to 0-indexed
            end = int(parts[1]) if len(parts) > 1 else len(iterations)
            # Filter iterations by their iteration number
            stable_iterations = [it for it in iterations 
                                if start < it.get('iteration', 0) <= end]
        except (ValueError, IndexError):
            print(f"Warning: Invalid metric-range format: {metric_range}. Using default.")
            stable_iterations = iterations[warmup_steps:] if len(iterations) > warmup_steps else iterations
    elif use_last_n:
        # Use last N iterations
        stable_iterations = iterations[-use_last_n:] if len(iterations) >= use_last_n else iterations
    else:
        # Default: skip warmup steps
        stable_iterations = iterations[warmup_steps:] if len(iterations) > warmup_steps else iterations
    
    if not stable_iterations:
        return result
    
    # Recalculate metrics
    tokens_list = [it['tokens_per_sec_per_gpu'] for it in stable_iterations if it.get('tokens_per_sec_per_gpu', 0) > 0]
    tflops_list = [it['tflops_per_gpu'] for it in stable_iterations if it.get('tflops_per_gpu', 0) > 0]
    time_list = [it['iteration_time_ms'] for it in stable_iterations if it.get('iteration_time_ms', 0) > 0]
    
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
    
    # Recalculate derived metrics
    result.tokens_per_sec_total = result.tokens_per_sec_per_gpu * result.num_gpus
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
            
            # Support multiple formats:
            # Old: JOB_ID|MODEL|SIZE|GPUS|TP|PP|CP|EP|DP|EXP_DIR (10 fields)
            # New: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|EXP_DIR (12 fields)
            # Ref16: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|SEQ_LEN|MBS|GBS|PRECISION|EXP_DIR (16 fields)
            # Ref17: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR (17 fields)
            # Ref18: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR (18 fields)
            # Ref20: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|FSDP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR (20 fields)
            if len(parts) >= 10:
                # Detect format based on number of fields
                if len(parts) >= 20:
                    # New Reference format with VP, ETP, FSDP, and TASK fields
                    # JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|FSDP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
                    exp_dir = Path(parts[19])
                    idx_offset = 2  # Additional fields: NODES, GPU_TYPE
                elif len(parts) >= 18:
                    # Reference format with VP and TASK fields (no ETP, FSDP)
                    # JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
                    exp_dir = Path(parts[17])
                    idx_offset = 2  # Additional fields: NODES, GPU_TYPE
                elif len(parts) >= 17:
                    # Reference format with TASK field (no VP)
                    # JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
                    exp_dir = Path(parts[16])
                    idx_offset = 2  # Additional fields: NODES, GPU_TYPE
                elif len(parts) >= 16:
                    # Reference format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|SEQ_LEN|MBS|GBS|PRECISION|EXP_DIR
                    exp_dir = Path(parts[15])
                    idx_offset = 2  # Additional fields: NODES, GPU_TYPE
                elif len(parts) >= 12:
                    # New format with NODES and GPU_TYPE
                    exp_dir = Path(parts[11])
                    idx_offset = 2  # Additional fields: NODES, GPU_TYPE
                else:
                    # Old format
                    exp_dir = Path(parts[9])
                    idx_offset = 0
                
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
                        
                        # Parse nodes and GPU type based on format
                        if len(parts) >= 20:
                            # 20-field format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|ETP|FSDP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
                            try:
                                result.num_gpus = int(parts[3])
                                result.num_nodes = int(parts[4])
                                result.gpu_type = parts[5]
                                result.vp = int(parts[11])
                                result.etp = int(parts[12])
                                result.fsdp = int(parts[13])
                                result.sequence_length = int(parts[14])
                                result.micro_batch_size = int(parts[15])
                                result.global_batch_size = int(parts[16])
                                result.precision = parts[17]
                                result.task = parts[18]
                                if result.num_nodes > 0:
                                    result.gpus_per_node = result.num_gpus // result.num_nodes
                            except (ValueError, IndexError):
                                pass
                        elif len(parts) >= 18:
                            # 18-field format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|VP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
                            try:
                                result.num_gpus = int(parts[3])
                                result.num_nodes = int(parts[4])
                                result.gpu_type = parts[5]
                                result.vp = int(parts[11])
                                result.sequence_length = int(parts[12])
                                result.micro_batch_size = int(parts[13])
                                result.global_batch_size = int(parts[14])
                                result.precision = parts[15]
                                result.task = parts[16]
                                if result.num_nodes > 0:
                                    result.gpus_per_node = result.num_gpus // result.num_nodes
                            except (ValueError, IndexError):
                                pass
                        elif len(parts) >= 17:
                            # 17-field format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|SEQ_LEN|MBS|GBS|PRECISION|TASK|EXP_DIR
                            try:
                                result.num_gpus = int(parts[3])
                                result.num_nodes = int(parts[4])
                                result.gpu_type = parts[5]
                                result.sequence_length = int(parts[11])
                                result.micro_batch_size = int(parts[12])
                                result.global_batch_size = int(parts[13])
                                result.precision = parts[14]
                                result.task = parts[15]
                                if result.num_nodes > 0:
                                    result.gpus_per_node = result.num_gpus // result.num_nodes
                            except (ValueError, IndexError):
                                pass
                        elif len(parts) >= 16:
                            # 16-field format: JOB_ID|MODEL|SIZE|GPUS|NODES|GPU_TYPE|TP|PP|CP|EP|DP|SEQ_LEN|MBS|GBS|PRECISION|EXP_DIR
                            try:
                                result.num_gpus = int(parts[3])
                                result.num_nodes = int(parts[4])
                                result.gpu_type = parts[5]
                                result.precision = parts[14]
                                if result.num_nodes > 0:
                                    result.gpus_per_node = result.num_gpus // result.num_nodes
                                # Try to infer task from exp_dir name
                                exp_dir_name = str(exp_dir).lower()
                                if 'pretrain' in exp_dir_name:
                                    result.task = 'pretrain'
                                elif 'sft' in exp_dir_name:
                                    result.task = 'sft'
                                elif 'lora' in exp_dir_name:
                                    result.task = 'lora'
                            except (ValueError, IndexError):
                                pass
                        elif len(parts) >= 12:
                            try:
                                result.num_nodes = int(parts[4])
                                result.gpu_type = parts[5]
                                # Calculate gpus_per_node
                                if result.num_nodes > 0:
                                    result.gpus_per_node = result.num_gpus // result.num_nodes
                            except (ValueError, IndexError):
                                pass
                        else:
                            # Old format: assume GB200 with 4 GPUs/node (default for existing experiments)
                            result.gpu_type = "gb200"
                            result.gpus_per_node = 4
                            result.num_nodes = result.num_gpus // result.gpus_per_node if result.num_gpus > 0 else 0
                        
                        # Parse parallelism from submitted_jobs.txt as fallback
                        if result.tp == 1 and result.pp == 1 and result.dp == 1:
                            try:
                                if len(parts) >= 20:
                                    # 20-field format: TP at index 6, VP at index 11, ETP at index 12, FSDP at index 13
                                    result.tp = int(parts[6])
                                    result.pp = int(parts[7])
                                    result.cp = int(parts[8])
                                    result.ep = int(parts[9])
                                    result.dp = int(parts[10])
                                    result.vp = int(parts[11])
                                    result.etp = int(parts[12])
                                    result.fsdp = int(parts[13])
                                elif len(parts) >= 18:
                                    # 18-field format: TP at index 6, VP at index 11
                                    result.tp = int(parts[6])
                                    result.pp = int(parts[7])
                                    result.cp = int(parts[8])
                                    result.ep = int(parts[9])
                                    result.dp = int(parts[10])
                                    result.vp = int(parts[11])
                                elif len(parts) >= 17:
                                    # 17-field format: TP at index 6 (same as 16-field)
                                    result.tp = int(parts[6])
                                    result.pp = int(parts[7])
                                    result.cp = int(parts[8])
                                    result.ep = int(parts[9])
                                    result.dp = int(parts[10])
                                elif len(parts) >= 16:
                                    # 16-field format: TP at index 6
                                    result.tp = int(parts[6])
                                    result.pp = int(parts[7])
                                    result.cp = int(parts[8])
                                    result.ep = int(parts[9])
                                    result.dp = int(parts[10])
                                else:
                                    result.tp = int(parts[4 + idx_offset])
                                    result.pp = int(parts[5 + idx_offset])
                                    result.cp = int(parts[6 + idx_offset])
                                    result.ep = int(parts[7 + idx_offset])
                                    result.dp = int(parts[8 + idx_offset])
                            except (ValueError, IndexError):
                                pass
                        
                        # Recalculate total tokens/sec with correct num_gpus
                        result.tokens_per_sec_total = result.tokens_per_sec_per_gpu * result.num_gpus
                        # Calculate DP per expert group for MoE models
                        if result.ep > 0:
                            result.dp_per_ep = result.dp // result.ep
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
        'exp_name', 'status', 'model_name', 'model_size', 'task', 'precision',
        'gpu_type', 'num_nodes', 'gpus_per_node', 'num_gpus',
        'fsdp', 'tp', 'pp', 'cp', 'dp', 'ep', 'vp', 'etp',
        'micro_batch_size', 'global_batch_size', 'sequence_length', 'dp_per_ep',
        'tokens_per_sec_per_gpu', 'tokens_per_sec_total',
        'tflops_per_gpu', 'iteration_time_ms', 'samples_per_sec',
        'tokens_per_sec_per_gpu_min', 'tokens_per_sec_per_gpu_max', 'tokens_per_sec_per_gpu_std',
        'memory_allocated_gb', 'memory_reserved_gb',
        'total_iterations', 'max_steps', 'exp_dir', 'log_file_path'
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
    
    # Color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'  # Gray/dim text for log paths
    RESET = '\033[0m'
    
    # Status color mapping
    status_colors = {
        'completed': GREEN,
        'running': YELLOW,
        'failed': RED,
        'error': RED,
        'oom': RED,
    }
    
    # Column widths
    COL_MODEL = 18
    COL_TASK = 9  # pretrain, sft, lora
    COL_PREC = 8  # fp8_cs, fp8_mx, nvfp4, bf16
    COL_STATUS = 10
    COL_ITERS = 10  # Iterations (e.g., "100/100")
    COL_GPU = 6
    COL_NODES = 5   # Support up to 9999 nodes
    COL_GPUS = 6    # Support up to 99999 GPUs
    COL_FSDP = 2    # F (FSDP flag)
    COL_PARA = 3    # TP, PP, CP, DP, EP, VP, ETP
    COL_BATCH = 5  # MBS, GBS
    COL_METRIC = 10
    # Directory column - no fixed width, will be appended at the end
    
    # Check if any experiment has VP > 1 or ETP > 1
    has_vp = any(r.vp > 1 for r in results)
    has_etp = any(r.etp > 1 for r in results)
    has_fsdp = any(r.fsdp > 0 for r in results)
    
    # Calculate total width (without directory column - it will extend beyond)
    # Base: Model + Task + Prec + Status + Iters + GPU + N + #GPU + TP + PP + CP + DP + EP + MBS + GBS + 3 metrics
    total_width = COL_MODEL + COL_TASK + COL_PREC + COL_STATUS + COL_ITERS + COL_GPU + COL_NODES + COL_GPUS
    total_width += COL_FSDP if has_fsdp else 0
    total_width += (COL_PARA * 5)  # TP, PP, CP, DP, EP
    total_width += COL_PARA if has_vp else 0  # VP
    total_width += COL_PARA if has_etp else 0  # ETP
    total_width += (COL_BATCH * 2)  # MBS, GBS
    total_width += (COL_METRIC * 3) + 2  # metrics + spacing
    
    # Build header dynamically
    # Order: FSDP → TP → CP → EP → PP → DP → VP → ETP → MBS → GBS
    header = f"{'Model':<{COL_MODEL}}{'Task':<{COL_TASK}}{'Prec':<{COL_PREC}}{'Status':<{COL_STATUS}}{'Iters':<{COL_ITERS}}{'GPU':<{COL_GPU}}"
    header += f"{'N':>{COL_NODES}}{'#GPU':>{COL_GPUS}}"
    if has_fsdp:
        header += f"{'F':>{COL_FSDP}}"  # FSDP flag
    header += f"{'TP':>{COL_PARA}}{'CP':>{COL_PARA}}{'EP':>{COL_PARA}}{'PP':>{COL_PARA}}{'DP':>{COL_PARA}}"
    if has_vp:
        header += f"{'VP':>{COL_PARA}}"
    if has_etp:
        header += f"{'ET':>{COL_PARA}}"  # ETP shortened to ET
    header += f"{'MBS':>{COL_BATCH}}{'GBS':>{COL_BATCH}}"
    header += f"{'Tok/s/GPU':>{COL_METRIC}}{'TFLOP/s':>{COL_METRIC}}{'Iter(ms)':>{COL_METRIC}}"
    # Removed Exp Directory from header - will show log path below each row
    
    # Group results by model (model_name + model_size, avoiding duplicates)
    from collections import defaultdict
    model_groups = defaultdict(list)
    for r in results:
        # Avoid duplicates like "qwen3_qwen3_32b"
        if r.model_size and r.model_name and r.model_size.startswith(f"{r.model_name}_"):
            model_key = r.model_size
        elif r.model_name and r.model_size:
            model_key = f"{r.model_name}_{r.model_size}"
        elif r.model_size:
            model_key = r.model_size
        else:
            model_key = r.model_name or "unknown"
        model_groups[model_key].append(r)
    
    # Sort each group by tokens/sec/gpu descending
    for model_key in model_groups:
        model_groups[model_key] = sorted(
            model_groups[model_key], 
            key=lambda x: x.tokens_per_sec_per_gpu, 
            reverse=True
        )
    
    # Sort model groups by best throughput in each group (descending)
    sorted_model_keys = sorted(
        model_groups.keys(),
        key=lambda k: max(r.tokens_per_sec_per_gpu for r in model_groups[k]),
        reverse=True
    )
    
    # Print header
    print("\n" + "=" * total_width)
    print(header)
    print("=" * total_width)
    
    # Print each model group
    for i, model_key in enumerate(sorted_model_keys):
        group_results = model_groups[model_key]
        
        # Print model group separator (except for first group)
        if i > 0:
            print("-" * total_width)
        
        for r in group_results:
            # Format status display (show "OOM" for out of memory)
            status_display = "OOM" if r.status == 'oom' else r.status
            color = status_colors.get(r.status, '')
            
            # Format model name (combine model_name and model_size, avoiding duplicates)
            # Handle cases like model_name="qwen3" and model_size="qwen3_32b" -> "qwen3_32b" (not "qwen3_qwen3_32b")
            if r.model_size and r.model_name and r.model_size.startswith(f"{r.model_name}_"):
                model_str = r.model_size
            elif r.model_size and r.model_name:
                model_str = f"{r.model_name}_{r.model_size}"
            elif r.model_size:
                model_str = r.model_size
            else:
                model_str = r.model_name or "unknown"
            if len(model_str) > COL_MODEL - 1:
                model_str = model_str[:COL_MODEL - 1]
            
            # Format task (truncate if too long)
            task_str = (r.task[:COL_TASK-1] if r.task else "-")
            
            # Format precision (truncate if too long)
            prec_str = (r.precision[:COL_PREC-1] if r.precision else "-")
            
            # Format GPU type (truncate if too long)
            gpu_str = (r.gpu_type[:COL_GPU-1] if r.gpu_type else "?")
            
            # Format iterations (total_iterations / max_steps)
            if r.max_steps > 0:
                iters_str = f"{r.total_iterations}/{r.max_steps}"
            else:
                iters_str = str(r.total_iterations) if r.total_iterations > 0 else "-"
            
            # Build the row dynamically based on which columns are present
            # Order: FSDP → TP → CP → EP → PP → DP → VP → ETP → MBS → GBS
            row = f"{model_str:<{COL_MODEL}}{task_str:<{COL_TASK}}{prec_str:<{COL_PREC}}{status_display:<{COL_STATUS}}{iters_str:<{COL_ITERS}}{gpu_str:<{COL_GPU}}"
            row += f"{r.num_nodes:>{COL_NODES}}{r.num_gpus:>{COL_GPUS}}"
            if has_fsdp:
                row += f"{r.fsdp:>{COL_FSDP}}"
            row += f"{r.tp:>{COL_PARA}}{r.cp:>{COL_PARA}}{r.ep:>{COL_PARA}}{r.pp:>{COL_PARA}}{r.dp:>{COL_PARA}}"
            if has_vp:
                row += f"{r.vp:>{COL_PARA}}"
            if has_etp:
                row += f"{r.etp:>{COL_PARA}}"
            row += f"{r.micro_batch_size:>{COL_BATCH}}{r.global_batch_size:>{COL_BATCH}}"
            row += f"{r.tokens_per_sec_per_gpu:>{COL_METRIC}.1f}{r.tflops_per_gpu:>{COL_METRIC}.1f}"
            row += f"{r.iteration_time_ms:>{COL_METRIC}.1f}"
            
            # Apply color to status only (replace status in the row)
            status_start = COL_MODEL + COL_TASK + COL_PREC
            status_end = status_start + COL_STATUS
            if color:
                colored_status = f"{color}{status_display:<{COL_STATUS}}{RESET}"
                # Replace the plain status with colored version
                row = row[:status_start] + colored_status + row[status_end:]
            
            print(row)
            
            # Print log file path below in gray (clickable in terminal)
            log_path = r.log_file_path if r.log_file_path else (str(Path(r.exp_dir) / "log-*.out") if r.exp_dir else "")
            if log_path:
                print(f"{GRAY}  └─ {log_path}{RESET}")
    
    print("=" * total_width)
    
    # Summary statistics
    completed = [r for r in results if r.status == 'completed']
    oom_count = sum(1 for r in results if r.status == 'oom')
    failed_count = sum(1 for r in results if r.status in ('failed', 'error'))
    running_count = sum(1 for r in results if r.status == 'running')
    
    print(f"\nSummary:")
    print(f"  Total: {len(results)} | Completed: {len(completed)} | Running: {running_count} | Failed: {failed_count} | \033[91mOOM: {oom_count}\033[0m")
    
    if completed:
        avg_tokens = sum(r.tokens_per_sec_per_gpu for r in completed) / len(completed)
        max_tokens = max(r.tokens_per_sec_per_gpu for r in completed)
        best = max(completed, key=lambda x: x.tokens_per_sec_per_gpu)
        
        print(f"  Average Tokens/sec/GPU: {avg_tokens:.1f}")
        print(f"  Best Tokens/sec/GPU: {max_tokens:.1f}")
        print(f"  Best configuration: {best.exp_name}")
        print(f"    GPU: {best.gpu_type}, Nodes: {best.num_nodes}, GPUs: {best.num_gpus}")
        parallelism_str = f"TP={best.tp}, CP={best.cp}, EP={best.ep}, PP={best.pp}, DP={best.dp}"
        if best.vp > 1:
            parallelism_str += f", VP={best.vp}"
        if best.etp > 1:
            parallelism_str += f", ETP={best.etp}"
        if best.fsdp > 0:
            parallelism_str += ", FSDP=ON"
        print(f"    {parallelism_str}")
        print(f"    MBS={best.micro_batch_size}, GBS={best.global_batch_size}")


def load_config_file(config_path: Path) -> dict:
    """Load a YAML or JSON config file."""
    if not YAML_AVAILABLE and config_path.suffix in ['.yaml', '.yml']:
        print("Warning: PyYAML not installed. Install with: pip install pyyaml")
        return {}
    
    with open(config_path) as f:
        if config_path.suffix in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        else:
            return json.load(f)


def get_target_config(config: dict, target_model: str, config_type: str = 'rl_training_configs') -> dict:
    """Get the configuration for a specific target model."""
    configs = config.get(config_type, {})
    
    if target_model in configs:
        return configs[target_model]
    
    # Try partial match
    for model_name, model_config in configs.items():
        if target_model.lower() in model_name.lower():
            return model_config
    
    return {}


def list_available_configs(config: dict):
    """List all available configurations."""
    print("\nAvailable configurations:")
    
    for config_type in ['rl_training_configs', 'reference_configs']:
        configs = config.get(config_type, {})
        if configs:
            print(f"\n  [{config_type}]")
            for model_name, model_config in configs.items():
                pattern = model_config.get('model_pattern', model_name)
                tp = model_config.get('tp', '?')
                pp = model_config.get('pp', '?')
                cp = model_config.get('cp', '?')
                ep = model_config.get('ep', '?')
                dp = model_config.get('dp', '?')
                gpus = model_config.get('gpus', '?')
                gbs = model_config.get('gbs', '?')
                print(f"    {model_name}: TP={tp} CP={cp} EP={ep} PP={pp} DP={dp} | GPUs={gpus} GBS={gbs}")


def matches_config(result: ExperimentResult, target_config: dict) -> bool:
    """Check if a result matches the target config."""
    # Check model pattern
    model_pattern = target_config.get('model_pattern', '')
    if model_pattern and model_pattern.lower() not in result.exp_dir.lower():
        return False
    
    # Check parallelism
    if target_config.get('tp') is not None and result.tp != target_config['tp']:
        return False
    if target_config.get('pp') is not None and result.pp != target_config['pp']:
        return False
    if target_config.get('cp') is not None and result.cp != target_config['cp']:
        return False
    if target_config.get('ep') is not None and result.ep != target_config['ep']:
        return False
    if target_config.get('dp') is not None and result.dp != target_config['dp']:
        return False
    
    # Check batch and GPU settings
    if target_config.get('gbs') is not None and result.global_batch_size != target_config['gbs']:
        return False
    if target_config.get('gpus') is not None and result.num_gpus != target_config['gpus']:
        return False
    if target_config.get('seq_len') is not None and result.sequence_length != target_config['seq_len']:
        return False
    
    # Check max_steps if specified in config
    if target_config.get('max_steps') is not None and result.max_steps != target_config['max_steps']:
        return False
    
    return True


def filter_results(results: list[ExperimentResult], args) -> list[ExperimentResult]:
    """Filter results based on command line arguments."""
    filtered = results
    
    # Apply config-based filters if provided
    if hasattr(args, 'match_config') and args.match_config:
        config = load_config_file(args.match_config)
        
        # Determine config type
        config_type = 'reference_configs' if args.use_reference else 'rl_training_configs'
        
        if args.target_model:
            # Filter for specific model config only
            target_config = get_target_config(config, args.target_model, config_type)
            
            if not target_config:
                print(f"Warning: No config found for '{args.target_model}' in {config_type}")
                list_available_configs(config)
                return []
            
            print(f"\nUsing config for '{args.target_model}':")
            print(f"  TP={target_config.get('tp')} CP={target_config.get('cp')} EP={target_config.get('ep')} PP={target_config.get('pp')} DP={target_config.get('dp')}")
            print(f"  GPUs={target_config.get('gpus')} GBS={target_config.get('gbs')} SEQ={target_config.get('seq_len')}")
            print()
            
            # Filter using single config
            filtered = [r for r in filtered if matches_config(r, target_config)]
        else:
            # Filter for ANY config in the file (no --target-model specified)
            configs_dict = config.get(config_type, {})
            if not configs_dict:
                print(f"Warning: No configs found in '{config_type}' section")
                list_available_configs(config)
                return []
            
            # Get all configs from the config type
            all_configs = list(configs_dict.values())
            config_names = list(configs_dict.keys())
            
            print(f"\nMatching against {len(all_configs)} configs from '{config_type}':")
            for name in config_names:
                cfg = configs_dict[name]
                print(f"  - {name}: TP={cfg.get('tp')} CP={cfg.get('cp')} EP={cfg.get('ep')} PP={cfg.get('pp')} DP={cfg.get('dp')} GPUs={cfg.get('gpus')}")
            print()
            
            # Filter results that match ANY of the configs
            def matches_any_config(result):
                for cfg in all_configs:
                    if matches_config(result, cfg):
                        return True
                return False
            
            filtered = [r for r in filtered if matches_any_config(r)]
    
    # Apply command-line filters (override or additional)
    if args.filter_tp is not None:
        filtered = [r for r in filtered if r.tp == args.filter_tp]
    if args.filter_pp is not None:
        filtered = [r for r in filtered if r.pp == args.filter_pp]
    if args.filter_cp is not None:
        filtered = [r for r in filtered if r.cp == args.filter_cp]
    if args.filter_ep is not None:
        filtered = [r for r in filtered if r.ep == args.filter_ep]
    if args.filter_dp is not None:
        filtered = [r for r in filtered if r.dp == args.filter_dp]
    if args.filter_gbs is not None:
        filtered = [r for r in filtered if r.global_batch_size == args.filter_gbs]
    if args.filter_seq is not None:
        filtered = [r for r in filtered if r.sequence_length == args.filter_seq]
    if args.filter_gpus is not None:
        filtered = [r for r in filtered if r.num_gpus == args.filter_gpus]
    if args.filter_max_steps is not None:
        filtered = [r for r in filtered if r.max_steps == args.filter_max_steps]
    if args.filter_model:
        filtered = [r for r in filtered if args.filter_model.lower() in r.model_name.lower() 
                   or args.filter_model.lower() in r.model_size.lower()
                   or args.filter_model.lower() in r.exp_dir.lower()]
    if args.filter_precision:
        filtered = [r for r in filtered if r.precision == args.filter_precision]
    if args.filter_task:
        filtered = [r for r in filtered if r.task == args.filter_task]
    if args.filter_status:
        filtered = [r for r in filtered if r.status == args.filter_status]
    
    return filtered


def scan_all_experiments(base_dir: Path) -> list[ExperimentResult]:
    """Scan all experiment directories and collect results."""
    results = []
    
    # Find all model directories (e.g., llama3_8b_llm_pretrain_bf16/)
    for model_dir in base_dir.glob('*/'):
        if not model_dir.is_dir():
            continue
        
        # Find all experiment runs within each model directory
        for exp_dir in model_dir.glob('*/'):
            if not exp_dir.is_dir():
                continue
            
            log_file = find_log_file(exp_dir)
            if log_file:
                result = parse_log_file(log_file)
                results.append(result)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Collect results from Megatron-Bridge experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file to match RL training settings
  python3 collect_results.py --scan-all --match-config rl_training_configs.yaml --target-model qwen3_32b
  python3 collect_results.py --scan-all --match-config rl_training_configs.yaml --target-model qwen3_30b
  python3 collect_results.py --scan-all --match-config rl_training_configs.yaml --target-model llama3_70b
  
  # Use reference configs instead of RL training configs
  python3 collect_results.py --scan-all --match-config rl_training_configs.yaml --target-model qwen3_32b --use-reference
  
  # List available configs
  python3 collect_results.py --match-config rl_training_configs.yaml --list-configs
  
  # Scan all experiments and filter by parallelism (manual)
  python3 collect_results.py --scan-all --filter-tp 4 --filter-pp 1 --filter-dp 4
  
  # Find all Qwen32B experiments with specific settings
  python3 collect_results.py --scan-all --filter-model qwen --filter-gbs 512 --filter-gpus 16
  
  # Find completed experiments only
  python3 collect_results.py --scan-all --filter-status completed
"""
    )
    
    # Source options (mutually exclusive in practice)
    parser.add_argument('--sweep-dir', type=Path, help='Sweep directory containing submitted_jobs.txt')
    parser.add_argument('--exp-dirs', type=Path, nargs='+', help='List of experiment directories')
    parser.add_argument('--latest', type=int, help='Collect from latest N experiments')
    parser.add_argument('--scan-all', action='store_true', 
                       help='Scan ALL experiment directories (can be slow)')
    parser.add_argument('--base-dir', type=Path, default=Path('./exp_logs/experiments'),
                       help='Base directory for experiments')
    
    # Config-based filtering
    config_group = parser.add_argument_group('Config-based filtering')
    config_group.add_argument('--match-config', type=Path,
                             help='YAML/JSON config file with target parallelism settings')
    config_group.add_argument('--target-model', type=str,
                             help='Target model name from config (e.g., qwen3_32b, qwen3_30b, llama3_70b)')
    config_group.add_argument('--use-reference', action='store_true',
                             help='Use reference_configs instead of rl_training_configs')
    config_group.add_argument('--list-configs', action='store_true',
                             help='List available configurations in the config file')
    
    # Filter options
    filter_group = parser.add_argument_group('Filter options')
    filter_group.add_argument('--filter-tp', type=int, help='Filter by Tensor Parallelism')
    filter_group.add_argument('--filter-pp', type=int, help='Filter by Pipeline Parallelism')
    filter_group.add_argument('--filter-cp', type=int, help='Filter by Context Parallelism')
    filter_group.add_argument('--filter-ep', type=int, help='Filter by Expert Parallelism')
    filter_group.add_argument('--filter-dp', type=int, help='Filter by Data Parallelism')
    filter_group.add_argument('--filter-gbs', type=int, help='Filter by Global Batch Size')
    filter_group.add_argument('--filter-seq', type=int, help='Filter by Sequence Length')
    filter_group.add_argument('--filter-gpus', type=int, help='Filter by number of GPUs')
    filter_group.add_argument('--filter-max-steps', type=int, help='Filter by max_steps (training iterations)')
    filter_group.add_argument('--filter-model', type=str, help='Filter by model name (partial match)')
    filter_group.add_argument('--filter-precision', type=str, help='Filter by precision (bf16, fp8_cs, etc.)')
    filter_group.add_argument('--filter-task', type=str, help='Filter by task (pretrain, sft, lora)')
    filter_group.add_argument('--filter-status', type=str, help='Filter by status (completed, failed, oom)')
    
    # Output options
    parser.add_argument('--output-csv', type=Path, help='Output CSV file path')
    parser.add_argument('--output-json', type=Path, help='Output JSON file path')
    parser.add_argument('--no-print', action='store_true', help='Suppress table output')
    
    # Metric calculation options
    metric_group = parser.add_argument_group('Metric calculation options')
    metric_group.add_argument('--warmup-steps', type=int, default=3,
                              help='Number of warmup steps to skip (default: 3)')
    metric_group.add_argument('--metric-range', type=str, default=None,
                              help='Calculate metrics from specific step range, e.g., "35:50" for steps 35-50')
    metric_group.add_argument('--use-last-n', type=int, default=None,
                              help='Use only last N iterations for metric calculation')
    metric_group.add_argument('--nemorl-style', action='store_true',
                              help='Use NeMo-RL style calculation: 5-step moving average around step 5')
    metric_group.add_argument('--at-step', type=int, default=5,
                              help='Center step for NeMo-RL style (default: 5)')
    metric_group.add_argument('--average-steps', type=int, default=5,
                              help='Number of steps to average for NeMo-RL style (default: 5)')
    
    args = parser.parse_args()
    
    # Handle --list-configs option
    if args.list_configs:
        if not args.match_config:
            print("Error: --list-configs requires --match-config")
            sys.exit(1)
        config = load_config_file(args.match_config)
        list_available_configs(config)
        sys.exit(0)
    
    results = []
    
    if args.sweep_dir:
        results = collect_from_sweep(args.sweep_dir)
    elif args.exp_dirs:
        results = collect_from_dirs(args.exp_dirs)
    elif args.scan_all:
        print(f"Scanning all experiments in {args.base_dir}...")
        results = scan_all_experiments(args.base_dir)
        print(f"Found {len(results)} total experiments.")
    elif args.latest:
        results = collect_latest(args.base_dir, args.latest)
    else:
        # Default: collect from latest 10 experiments
        results = collect_latest(args.base_dir, 10)
    
    # Apply filters
    original_count = len(results)
    results = filter_results(results, args)
    
    if original_count > 0 and len(results) < original_count:
        print(f"Filtered: {original_count} -> {len(results)} results")
    
    if not results:
        print("No results found matching the filters.")
        sys.exit(1)
    
    # Load metric calculation settings from config file if available
    config_metric_settings = {}
    if args.match_config:
        config = load_config_file(args.match_config)
        if config and 'metadata' in config and 'metric_calculation' in config.get('metadata', {}):
            config_metric_settings = config['metadata']['metric_calculation']
            print(f"Loaded metric settings from config: {config_metric_settings}")
    
    # Determine final metric settings (CLI args override config file)
    final_warmup = args.warmup_steps
    final_metric_range = args.metric_range
    final_use_last_n = args.use_last_n
    final_nemorl_style = args.nemorl_style
    final_at_step = args.at_step
    final_average_steps = args.average_steps
    
    # Apply config file settings if CLI args not explicitly set
    if config_metric_settings:
        if args.warmup_steps == 3 and 'warmup_steps' in config_metric_settings:
            final_warmup = config_metric_settings.get('warmup_steps', 3)
        if args.metric_range is None and 'metric_range' in config_metric_settings:
            final_metric_range = config_metric_settings.get('metric_range')
        if args.use_last_n is None and 'use_last_n' in config_metric_settings:
            final_use_last_n = config_metric_settings.get('use_last_n')
        # NeMo-RL style settings from config
        if not args.nemorl_style and config_metric_settings.get('nemorl_style'):
            final_nemorl_style = True
        if args.at_step == 5 and 'at_step' in config_metric_settings:
            final_at_step = config_metric_settings.get('at_step', 5)
        if args.average_steps == 5 and 'average_steps' in config_metric_settings:
            final_average_steps = config_metric_settings.get('average_steps', 5)
    
    # Recalculate metrics if custom metric options provided
    need_recalc = (final_metric_range or final_use_last_n or final_warmup != 3 or final_nemorl_style)
    if need_recalc:
        metric_info = []
        if final_nemorl_style:
            # Calculate the actual index range for display
            min_idx = final_at_step - final_average_steps // 2
            max_idx = final_at_step + final_average_steps // 2 - 1 + (final_average_steps % 2)
            metric_info.append(f"nemorl-style(at_step={final_at_step}, average_steps={final_average_steps})")
            metric_info.append(f"→ indices {min_idx}:{max_idx} (iteration {min_idx+1}~{max_idx+1})")
        elif final_metric_range:
            metric_info.append(f"range={final_metric_range}")
        elif final_use_last_n:
            metric_info.append(f"last-{final_use_last_n}")
        else:
            metric_info.append(f"warmup={final_warmup}")
        print(f"Recalculating metrics: {', '.join(metric_info)}")
        
        for result in results:
            recalculate_metrics(
                result, 
                warmup_steps=final_warmup,
                metric_range=final_metric_range,
                use_last_n=final_use_last_n,
                nemorl_style=final_nemorl_style,
                at_step=final_at_step,
                average_steps=final_average_steps
            )
    
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

