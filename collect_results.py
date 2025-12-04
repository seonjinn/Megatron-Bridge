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


def parse_log_file(log_path: Path) -> ExperimentResult:
    """Parse a Megatron-Bridge log file and extract metrics."""
    result = ExperimentResult()
    result.exp_dir = str(log_path.parent.parent)
    result.exp_name = log_path.parent.parent.name
    result.log_file_path = str(log_path.resolve())  # Store full absolute path for clickable links
    
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
            
            # Check completion status
            if 'Training completed' in content or iterations and iterations[-1]['iteration'] >= result.max_steps:
                result.status = 'completed'
            elif has_oom:
                result.status = 'oom'  # Out of Memory
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
    COL_GPU = 6
    COL_NODES = 4
    COL_GPUS = 5
    COL_FSDP = 2  # F (FSDP flag)
    COL_PARA = 3  # TP, PP, CP, DP, EP, VP, ETP
    COL_BATCH = 5  # MBS, GBS
    COL_METRIC = 10
    # Directory column - no fixed width, will be appended at the end
    
    # Check if any experiment has VP > 1 or ETP > 1
    has_vp = any(r.vp > 1 for r in results)
    has_etp = any(r.etp > 1 for r in results)
    has_fsdp = any(r.fsdp > 0 for r in results)
    
    # Calculate total width (without directory column - it will extend beyond)
    # Base: Model + Task + Prec + Status + GPU + N + #GPU + TP + PP + CP + DP + EP + MBS + GBS + 3 metrics
    total_width = COL_MODEL + COL_TASK + COL_PREC + COL_STATUS + COL_GPU + COL_NODES + COL_GPUS
    total_width += COL_FSDP if has_fsdp else 0
    total_width += (COL_PARA * 5)  # TP, PP, CP, DP, EP
    total_width += COL_PARA if has_vp else 0  # VP
    total_width += COL_PARA if has_etp else 0  # ETP
    total_width += (COL_BATCH * 2)  # MBS, GBS
    total_width += (COL_METRIC * 3) + 2  # metrics + spacing
    
    # Build header dynamically
    # Order: FSDP → TP → CP → EP → PP → DP → VP → ETP → MBS → GBS
    header = f"{'Model':<{COL_MODEL}}{'Task':<{COL_TASK}}{'Prec':<{COL_PREC}}{'Status':<{COL_STATUS}}{'GPU':<{COL_GPU}}"
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
    
    # Group results by model (model_name + model_size)
    from collections import defaultdict
    model_groups = defaultdict(list)
    for r in results:
        model_key = f"{r.model_name}_{r.model_size}" if r.model_name else "unknown"
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
            
            # Format model name (combine model_name and model_size)
            model_str = f"{r.model_name}_{r.model_size}"
            if len(model_str) > COL_MODEL - 1:
                model_str = model_str[:COL_MODEL - 1]
            
            # Format task (truncate if too long)
            task_str = (r.task[:COL_TASK-1] if r.task else "-")
            
            # Format precision (truncate if too long)
            prec_str = (r.precision[:COL_PREC-1] if r.precision else "-")
            
            # Format GPU type (truncate if too long)
            gpu_str = (r.gpu_type[:COL_GPU-1] if r.gpu_type else "?")
            
            # Build the row dynamically based on which columns are present
            # Order: FSDP → TP → CP → EP → PP → DP → VP → ETP → MBS → GBS
            row = f"{model_str:<{COL_MODEL}}{task_str:<{COL_TASK}}{prec_str:<{COL_PREC}}{status_display:<{COL_STATUS}}{gpu_str:<{COL_GPU}}"
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
            if color:
                colored_status = f"{color}{status_display:<{COL_STATUS}}{RESET}"
                # Replace the plain status with colored version
                row = row[:status_start] + colored_status + row[status_start + COL_STATUS:]
            
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

