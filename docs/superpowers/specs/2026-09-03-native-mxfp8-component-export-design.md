# Native MXFP8 Component Export for Refit

## Status

Proposed extension to PR #5917.

## Problem

Transformer Engine can keep selected parameters in native MXFP8 storage when
`fp8_param=true`. The storage contains E4M3 value bytes and compact E8M0 scales,
while the logical parameter still follows the Megatron layout.

PR #5917 builds deterministic export tasks for this storage and preserves
grouped expert containers. It does not yet provide one mapping-owned operation
that turns a local task into the physical components needed by a refit
transport. Downstream code therefore has to inspect Transformer Engine metadata
and repeat mapping details. The current downstream implementation covers routed
FFN projections, but not interleaved QKV weights.

## Goals

1. Export native MXFP8 value and scale tensors without dequantization.
2. Avoid TP, EP, and PP collectives while projecting local components.
3. Let Bridge mappings own logical names and layout transforms.
4. Support Q, K, V, O, routed expert FC1, and routed expert FC2.
5. Keep mixed BF16 and native MXFP8 models deterministic under PP and VP.
6. Give downstream refit code a typed contract instead of class-name and
   parameter-name checks.

## Non-goals

- Choosing which modules use MXFP8. Transformer Engine precision matchers own
  that policy.
- Choosing a destination inference backend or its physical weight layout.
- Implementing NCCL Reshard or CUDA IPC transport in Megatron Bridge.
- Adding native MXFP8 support for co-trained MTP experts in this change.
- Quantizing shared experts unless a model already maps them through a supported
  local view.

## Design

### Native storage contract

Add a typed, frozen result in `param_mapping.py`:

```python
@dataclass(frozen=True)
class LocalMXFP8Param:
    name: str
    weight: torch.Tensor
    weight_scale: torch.Tensor
    global_weight_shape: torch.Size
    shard_group: Literal["tp", "etp", "replicated"]
    shard_dim: int | None
```

Value and scale are paired in one result. This prevents a mapping from applying
different row selection or naming to the two physical components.

`extract_native_mxfp8_components()` in `quant_bridge.py` reads Transformer
Engine metadata and returns tensors that share the native storage. It validates
all of the following:

- `rowwise_data` is contiguous `uint8` storage for E4M3 values.
- `rowwise_scale_inv` is a contiguous `uint8` E8M0 scale tensor.
- scales are compact rather than GEMM-swizzled.
- the logical reduction dimension is divisible by 32.
- the scale shape is `(*weight.shape[:-1], weight.shape[-1] // 32)` after
  removing only documented padding.

The returned weight is viewed as `torch.float8_e4m3fn`. Extraction does not
allocate, gather, reorder, or dequantize the parameter.

### Task planning

Keep `build_export_mxfp8_tasks()` as the setup-time planner. Identify routed
experts through the resolved mapping type and its expert classification rather
than an exact Megatron parameter-name suffix. Create grouped quantized members
once during planning and retain them for repeated refits. The hot path must not
recreate grouped members or inspect model-wide parameter names.

### Mapping-owned local projection

Add a native-MXFP8 projection operation separate from `LocalHFParamSpec`.
`LocalHFParamSpec` describes logical tensors that can be requantized and its
current contract explicitly excludes direct physical MXFP8 transfer. The new
operation accepts a paired local value and scale tensor and returns paired
`LocalMXFP8Param` records.

The operation is local by contract. It must not call PP broadcast, TP gather,
EP gather, quantization, or dequantization helpers. Setup-time metadata
broadcasts remain allowed. Identity and equal-split projections remain views. A
mapping such as interleaved QKV may allocate only the local reordered output
that its layout requires. Each result records the full logical shape plus the
local shard group and dimension so downstream code does not infer placement
from class names.

### QKV projection

`QKVMapping` overrides the native projection operation because GQA interleaving
cannot be represented as equal chunks.

1. Read the module's attention configuration.
2. Derive local attention-head and query-group counts from the mapping's TP
   size. Reject non-divisible configurations.
3. Split the E4M3 value tensor with `split_qkv_weights()` and an explicit local
   reduction dimension.
4. Split the E8M0 scale tensor with the same helper and an explicit local
   reduction dimension of `K / 32`. Do not use `split_qkv_weights_scale()`:
   native MXFP8 preserves every output row and compresses only K, while that
   helper also divides the head dimension for two-dimensional block scaling.
5. Return separate `q_proj`, `k_proj`, and `v_proj` views in declared mapping
   order.

The implementation must also preserve `attention_output_gate` behavior already
handled by the shared QKV split helpers. A byte-pattern test verifies that Q, K,
and V are not swapped for MHA or GQA.

### O projection

The output projection is accepted only when `AutoMapping` resolves to
`RowParallelMapping`. Both value and scale remain views of the local source
storage. The result declares a TP shard on dimension 1.

### Routed expert projection

Routed FC1 uses `GatedMLPMapping` or `FusedGatedExpertMapping`:

- split the output dimension into equal gate and up views;
- apply the same split to E4M3 values and E8M0 scales;
- keep local expert order unchanged;
- declare an expert-TP shard on dimension 0.

Routed FC2 uses `FusedExpertMapping` or a compatible one-to-one expert mapping.
Its local value and scale views are unchanged and declare an expert-TP shard on
dimension 1.

For native grouped storage, iterate the existing grouped MXFP8 members in local
expert order and project each member independently. Compute the canonical
global expert ID as `ep_rank * experts_per_rank + local_expert_id`; do not reuse
the expert-0 mapping name for every member. Pass an effective member name ending
in that global expert ID to the mapping so its existing expert-name rules create
the canonical HF name. Do not stack or materialize a full-model expert tensor.
BF16 grouped storage keeps the existing per-expert task expansion from PR
#5917.

### Task-level component export

Add a quantization-bridge helper that accepts the deterministic task sequence
created by PR #5917:

```python
iter_local_native_mxfp8_params(tasks) -> Iterator[LocalMXFP8Param]
```

For each local task, the helper:

1. extracts native value and scale storage once;
2. iterates already-cached grouped members when present;
3. asks the mapping to project each value/scale pair atomically;
4. checks each scale shape against its value shape;
5. yields paired parameters in stable task, local-expert, and mapping order.

Remote PP placeholders yield no parameters. A local parameter with invalid
native metadata raises a `ValueError` that includes the global parameter name.
An unsupported mapping fails closed before transport; direct physical transfer
must never guess a layout. The caller can select the existing BF16 or normal
Bridge fallback before invoking this helper.

## Integration Boundary

NeMo-RL should use the Bridge task and paired-parameter APIs for source
extraction and logical projection. It should not duplicate Transformer Engine
metadata keys, QKV ordering, or expert naming. NeMo-RL still owns transport
meshes, placement metadata, and fallback selection. The vLLM adapter still owns
destination parameter and scale binding because those names and layouts depend
on the vLLM version and selected linear backend.

Direct native transfer is selected only when both conditions hold:

1. Bridge returns validated paired native parameters for the task.
2. The destination adapter confirms compatible MXFP8 value and scale layout.

Otherwise the existing conversion path remains available.

## Error Handling

Fail before transport when any of these invariants is false:

- native metadata is missing, has the wrong dtype, or uses swizzled scales;
- value or scale storage has the wrong number of elements;
- the reduction dimension is not divisible by 32;
- TP does not evenly divide Q heads or query groups;
- FC1 cannot split evenly into gate and up projections;
- a projected scale shape is not the value shape with the final dimension
  divided by 32;
- grouped member count does not match the local expert count;
- `AutoMapping` does not resolve to the required row-parallel mapping;
- a mapping requests transpose or permutation, which would invalidate the
  scale-block direction;
- the parameter uses DTensor/FSDP or belongs to a co-trained MTP module.

Unsupported mapping transforms raise before direct transfer. They do not
silently produce a guessed layout.

## Validation

### Unit tests

- Native metadata extraction preserves storage pointers, trims only allowed
  scale padding, and preserves scale byte zero unchanged.
- O projection is identity for both value and scale and declares TP dimension
  1.
- QKV projection covers MHA and GQA at TP1 and TP2 with distinct byte patterns.
- QKV scale projection preserves every output row, compresses only K, and
  declares TP dimension 0.
- QKV with `attention_output_gate` preserves the existing Q/Z layout.
- Routed FC1 splits gate and up for ordinary and grouped native storage,
  computes global expert IDs at EP greater than one, and declares expert-TP
  dimension 0.
- Routed FC2 preserves per-expert order for ordinary and grouped storage,
  computes global expert IDs at EP greater than one, and declares expert-TP
  dimension 1.
- Mixed BF16/native tasks retain global order and correct PP/VP ownership.
- Remote PP placeholders emit no local parameters.
- Patched payload broadcast/gather/all-gather calls prove that materialization
  is collective-free.
- Mutating native backing storage after task construction is visible on the
  next materialization.
- Grouped members are created during task construction and reused on the hot
  path.
- MTP grouped experts retain the current explicit rejection.
- Invalid metadata, K alignment, TP divisibility, scale shape, grouped count,
  transpose/permutation, DTensor/FSDP, and wrong mapping type each fail with a
  parameter-specific error.

### Downstream proof

After the Bridge unit tests pass, a NeMo-RL integration change should run:

- one QKVO plus routed-MoE native MXFP8 refit smoke;
- one routed-MoE-only native MXFP8 refit smoke;
- a placement/value parity check against the existing conversion path;
- a multi-step Qwen3.5 run to check repeated-refit stability.

Performance claims belong to the downstream refit PR because Bridge only exposes
local views and does not select or run a transport.

## Compatibility

- Existing `LocalHFParamSpec` callers keep their current behavior.
- Existing BF16 and requantized FP8 export paths are unchanged.
- Existing PR #5917 task ordering and grouped BF16 expansion are unchanged.
- New APIs use public task and mapping objects rather than Transformer Engine
  private Python classes, reducing dependency on a specific TE release.
