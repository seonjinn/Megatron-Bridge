#!/usr/bin/env python3
"""
Parse GB200 Reference Config YAML for Sweep Scripts

Usage:
    # Get all pretrain_bf16 configs
    python3 scripts/parse_reference_config.py --config gb200_reference_configs.yaml --section pretrain_bf16

    # Get specific model config
    python3 scripts/parse_reference_config.py --config gb200_reference_configs.yaml --section pretrain_bf16 --model llama3_8b

    # List all available models in a section
    python3 scripts/parse_reference_config.py --config gb200_reference_configs.yaml --section pretrain_bf16 --list

    # Output format for bash (space-separated)
    python3 scripts/parse_reference_config.py --config gb200_reference_configs.yaml --section pretrain_bf16 --model llama3_8b --format bash
"""

import argparse
import sys
import json

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def load_config(config_path: str) -> dict:
    """Load YAML or JSON config file."""
    with open(config_path) as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            if not YAML_AVAILABLE:
                print("Error: PyYAML not installed. Run: pip install pyyaml", file=sys.stderr)
                sys.exit(1)
            return yaml.safe_load(f)
        else:
            return json.load(f)


def get_config_for_sweep(config: dict) -> str:
    """Convert config dict to sweep script format.
    
    Format: "MODEL_NAME MODEL_SIZE NUM_GPUS SEQ_LEN TP PP CP EP VP ETP FSDP MBS GBS PRECISION TASK TAG"
    """
    model_name = config.get('model_name', '')
    model_size = config.get('model_size', '')
    gpus = config.get('gpus', 0)
    seq_len = config.get('seq_len', 0)
    tp = config.get('tp', 1)
    pp = config.get('pp', 1)
    cp = config.get('cp', 1)
    ep = config.get('ep', 1)
    vp = config.get('vp', 1)
    etp = config.get('etp', 1)
    fsdp = config.get('fsdp', 0)
    mbs = config.get('mbs', 1)
    gbs = config.get('gbs', 1)
    precision = config.get('precision', 'bf16')
    task = config.get('task', 'pretrain')
    
    # Create tag from model_pattern or generate one
    tag = config.get('model_pattern', f"{model_name}_{model_size}")
    
    return f"{model_name} {model_size} {gpus} {seq_len} {tp} {pp} {cp} {ep} {vp} {etp} {fsdp} {mbs} {gbs} {precision} {task} {tag}"


def main():
    parser = argparse.ArgumentParser(description='Parse GB200 reference config for sweep scripts')
    parser.add_argument('--config', '-c', required=True, help='Path to YAML/JSON config file')
    parser.add_argument('--section', '-s', default='pretrain_bf16', 
                        help='Config section (pretrain_bf16, pretrain_fp8, etc.)')
    parser.add_argument('--model', '-m', help='Specific model name')
    parser.add_argument('--list', '-l', action='store_true', help='List available models')
    parser.add_argument('--format', '-f', choices=['json', 'bash', 'sweep'], default='sweep',
                        help='Output format')
    parser.add_argument('--all', '-a', action='store_true', help='Output all configs in section')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get section
    section = config.get(args.section, {})
    if not section:
        print(f"Error: Section '{args.section}' not found", file=sys.stderr)
        print(f"Available sections: {list(config.keys())}", file=sys.stderr)
        sys.exit(1)
    
    # List mode
    if args.list:
        print(f"Available models in '{args.section}':")
        for model_name, model_config in section.items():
            notes = model_config.get('notes', '')
            gpus = model_config.get('gpus', '?')
            print(f"  {model_name}: {gpus} GPUs - {notes}")
        sys.exit(0)
    
    # All configs mode
    if args.all:
        for model_name, model_config in section.items():
            if args.format == 'sweep':
                print(get_config_for_sweep(model_config))
            elif args.format == 'json':
                print(json.dumps({model_name: model_config}))
            elif args.format == 'bash':
                # Output as bash array element
                sweep_str = get_config_for_sweep(model_config)
                print(f'"{sweep_str}"')
        sys.exit(0)
    
    # Specific model mode
    if args.model:
        if args.model not in section:
            print(f"Error: Model '{args.model}' not found in section '{args.section}'", file=sys.stderr)
            print(f"Available models: {list(section.keys())}", file=sys.stderr)
            sys.exit(1)
        
        model_config = section[args.model]
        
        if args.format == 'sweep':
            print(get_config_for_sweep(model_config))
        elif args.format == 'json':
            print(json.dumps(model_config, indent=2))
        elif args.format == 'bash':
            sweep_str = get_config_for_sweep(model_config)
            print(f'"{sweep_str}"')
        sys.exit(0)
    
    # Default: print all as sweep format
    for model_name, model_config in section.items():
        print(get_config_for_sweep(model_config))


if __name__ == '__main__':
    main()

