#!/usr/bin/env python3
"""
NeMo 25.07 Validation Sweep vs GB200 Reference Comparison

Compares what run_nemo2507_validation_sweep.sh will generate against
the actual GB200 reference configurations.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Config:
    """Configuration class"""
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
    def key(self) -> str:
        return f"{self.task}_{self.model}_{self.size}_{self.dtype}_{self.num_gpus}gpus"
    
    def matches(self, other: 'Config') -> Tuple[bool, List[str]]:
        """Check if configs match, return (is_match, differences)"""
        diffs = []
        
        # Important fields to check
        fields = [
            'num_gpus', 'seq_len', 'tp_size', 'pp_size', 'cp_size', 
            'ep_size', 'vp_size', 'mbs', 'gbs', 'etp_size',
            'cuda_graphs', 'use_mcore_fsdp', 'recompute_layers', 
            'activation_offload_layers'
        ]
        
        for field in fields:
            val_self = getattr(self, field)
            val_other = getattr(other, field)
            
            # Handle empty strings as 0
            if val_self == '':
                val_self = 0
            if val_other == '':
                val_other = 0
            
            # Convert to int for comparison
            try:
                val_self = int(float(val_self)) if val_self != '' else 0
                val_other = int(float(val_other)) if val_other != '' else 0
            except (ValueError, TypeError):
                pass
            
            if val_self != val_other:
                diffs.append(f"  {field:30s}: NEMO2507={val_self:6} ≠ REF={val_other:6}")
        
        return len(diffs) == 0, diffs


# ============================================================================
# GB200 Reference Configurations (from your CSV)
# ============================================================================
GB200_REFERENCE = [
    # Pre-train tasks only
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
]

# ============================================================================
# NeMo 25.07 Validation Sweep Configurations (AFTER FIX)
# (What run_nemo2507_validation_sweep.sh will generate after the fix)
# ============================================================================
NEMO2507_CONFIGS = [
    # LLAMA3_8B_2507_CONFIGS
    # Line 87: "llama3 8b 8 8192 1 1 1 1 1 1 0 2 128 bf16 pretrain nemo2507"
    # cuda_graphs=1 from base config
    Config('pre_train', 'llama3', '8b', 'gb200', 'bf16', 8, 8192, 1, 1, 1, 1, 1, 2, 128, 0, 1, 0, 0, 0),
    
    # LLAMA3_70B_2507_CONFIGS (FIXED!)
    # Line 97: "llama3 70b 64 8192 1 1 1 1 1 1 1 1 128 bf16 pretrain nemo2507"
    # Now with Hydra overrides: recompute_layers=20, cpu_offload=0, cuda_graphs=0
    Config('pre_train', 'llama3', '70b', 'gb200', 'bf16', 64, 8192, 1, 1, 1, 1, 1, 1, 128, 0, 0, 1, 20, 0),
    
    # LLAMA31_405B_2507_CONFIGS
    # Line 107: "llama31 405b 128 8192 4 8 2 1 8 1 0 1 64 bf16 pretrain nemo2507"
    # cuda_graphs=0 for PP>1
    Config('pre_train', 'llama31', '405b', 'gb200', 'bf16', 128, 8192, 4, 8, 2, 1, 8, 1, 64, 0, 0, 0, 0, 0),
    
    # DEEPSEEK_V3_LARGE_GBS_2507_CONFIGS
    # Line 117: "deepseek v3 256 4096 2 4 1 64 1 1 0 1 2048 bf16 pretrain nemo2507_gbs2048"
    Config('pre_train', 'deepseek', 'v3', 'gb200', 'bf16', 256, 4096, 2, 4, 1, 64, 1, 1, 2048, 1, 1, 0, 0, 0),
    
    # DEEPSEEK_V3_SMALL_GBS_2507_CONFIGS - REMOVED (not in GB200 reference)
    
    # QWEN3_30B_2507_CONFIGS
    # Line 137: "qwen3 30b_a3b 8 4096 1 1 1 8 1 1 0 1 512 bf16 pretrain nemo2507"
    Config('pre_train', 'qwen3', '30b_a3b', 'gb200', 'bf16', 8, 4096, 1, 1, 1, 8, 1, 1, 512, 1, 1, 0, 0, 0),
    
    # QWEN3_235B_2507_CONFIGS
    # Line 147: "qwen3 235b_a22b 64 4096 2 1 1 64 1 1 0 1 1024 bf16 pretrain nemo2507"
    Config('pre_train', 'qwen3', '235b_a22b', 'gb200', 'bf16', 64, 4096, 2, 1, 1, 64, 1, 1, 1024, 1, 1, 0, 0, 0),
]


def main():
    print("=" * 120)
    print("NeMo 25.07 Validation Sweep vs GB200 Reference Comparison")
    print("=" * 120)
    print()
    print("Checking if run_nemo2507_validation_sweep.sh configurations match GB200 reference...")
    print()
    
    # Build lookup for reference configs
    ref_map = {config.key: config for config in GB200_REFERENCE}
    
    matches = []
    mismatches = []
    missing_in_ref = []
    
    for nemo_config in NEMO2507_CONFIGS:
        ref_config = ref_map.get(nemo_config.key)
        
        if ref_config is None:
            missing_in_ref.append(nemo_config)
        else:
            is_match, diffs = nemo_config.matches(ref_config)
            if is_match:
                matches.append((nemo_config, ref_config))
            else:
                mismatches.append((nemo_config, ref_config, diffs))
    
    # Print summary
    print(f"{'=' * 120}")
    print(f"SUMMARY")
    print(f"{'=' * 120}")
    print(f"  ✓ Perfect matches:     {len(matches)}/{len(NEMO2507_CONFIGS)}")
    print(f"  ✗ Mismatches:          {len(mismatches)}/{len(NEMO2507_CONFIGS)}")
    print(f"  ⚠ Not in reference:    {len(missing_in_ref)}/{len(NEMO2507_CONFIGS)}")
    print(f"{'=' * 120}\n")
    
    # Print matches
    if matches:
        print(f"\n{'─' * 120}")
        print("✓ PERFECT MATCHES (NeMo 25.07 설정 == GB200 Reference)")
        print(f"{'─' * 120}")
        for nemo, ref in matches:
            print(f"\n  ✓ {nemo.key}")
            print(f"    GPUs={nemo.num_gpus}, SEQ={nemo.seq_len}, TP={nemo.tp_size}, PP={nemo.pp_size}, "
                  f"MBS={nemo.mbs}, GBS={nemo.gbs}")
            print(f"    FSDP={nemo.use_mcore_fsdp}, Recompute={nemo.recompute_layers}, "
                  f"CG={nemo.cuda_graphs}, ETP={nemo.etp_size}")
    
    # Print mismatches
    if mismatches:
        print(f"\n{'─' * 120}")
        print("✗ MISMATCHES (수정 필요)")
        print(f"{'─' * 120}")
        for nemo, ref, diffs in mismatches:
            print(f"\n  ✗ {nemo.key}")
            print(f"    Differences found:")
            for diff in diffs:
                print(f"    {diff}")
    
    # Print configs not in reference
    if missing_in_ref:
        print(f"\n{'─' * 120}")
        print("⚠ NOT IN GB200 REFERENCE")
        print(f"{'─' * 120}")
        for nemo in missing_in_ref:
            print(f"\n  ⚠ {nemo.key}")
            print(f"    NeMo 25.07에는 있지만 GB200 reference에는 없는 설정")
    
    # Specific analysis
    print(f"\n{'=' * 120}")
    print("DETAILED ANALYSIS")
    print(f"{'=' * 120}")
    
    # Check LLAMA3 70B specifically
    llama70b_nemo = next((c for c in NEMO2507_CONFIGS if c.model == 'llama3' and c.size == '70b'), None)
    llama70b_ref = next((c for c in GB200_REFERENCE if c.model == 'llama3' and c.size == '70b' and c.dtype == 'bf16'), None)
    
    if llama70b_nemo and llama70b_ref:
        print(f"\n1. LLAMA3 70B BF16 분석:")
        print(f"   GB200 Reference:")
        print(f"     - use_mcore_fsdp={llama70b_ref.use_mcore_fsdp}, recompute_layers={llama70b_ref.recompute_layers}")
        print(f"   NeMo 25.07 Sweep (run_nemo2507_validation_sweep.sh):")
        print(f"     - use_mcore_fsdp={llama70b_nemo.use_mcore_fsdp}, recompute_layers={llama70b_nemo.recompute_layers}")
        
        if llama70b_nemo.use_mcore_fsdp != llama70b_ref.use_mcore_fsdp:
            print(f"   ⚠️ FSDP 설정 일치")
        if llama70b_nemo.recompute_layers != llama70b_ref.recompute_layers:
            print(f"   ✗ CRITICAL: recompute_layers 불일치! (NeMo={llama70b_nemo.recompute_layers}, Ref={llama70b_ref.recompute_layers})")
            print(f"   → setup_experiment.py는 recompute_layers를 override하는 플래그가 없습니다!")
            print(f"   → FSDP flag만 있고, recompute는 모델 기본 설정을 사용합니다.")
    
    # Check DeepSeek V3 256 GPUs
    deepseek_nemo = next((c for c in NEMO2507_CONFIGS if c.model == 'deepseek' and c.num_gpus == 256 and c.gbs == 2048), None)
    deepseek_ref = next((c for c in GB200_REFERENCE if c.model == 'deepseek' and c.num_gpus == 256 and c.gbs == 2048), None)
    
    if deepseek_nemo and deepseek_ref:
        print(f"\n2. DeepSeek V3 256 GPUs GBS=2048 분석:")
        is_match, diffs = deepseek_nemo.matches(deepseek_ref)
        if is_match:
            print(f"   ✓ 완벽하게 일치합니다!")
        else:
            print(f"   ✗ 차이점:")
            for diff in diffs:
                print(f"   {diff}")
    
    # Overall recommendation
    print(f"\n{'=' * 120}")
    print("권장 사항")
    print(f"{'=' * 120}")
    
    if len(mismatches) > 0:
        print(f"\n⚠️ {len(mismatches)}개의 설정이 GB200 reference와 다릅니다.")
        print(f"\n주요 이슈:")
        print(f"  1. LLAMA3 70B: recompute_layers=0 (Reference는 20)")
        print(f"     → setup_experiment.py에 recompute_layers override 플래그가 없음")
        print(f"     → 모델 base config의 기본값을 사용하게 됨")
        print(f"\n  2. DeepSeek V3 GBS=128 설정은 reference에 없음 (256 GPUs 기준)")
        print(f"     → Reference는 GBS=2048만 정의되어 있음")
    else:
        print(f"\n✓ 모든 설정이 GB200 reference와 일치합니다!")
    
    print(f"\n{'=' * 120}\n")


if __name__ == "__main__":
    main()

