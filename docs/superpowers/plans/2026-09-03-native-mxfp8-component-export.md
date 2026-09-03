# Native MXFP8 Component Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend PR #5917 with a typed, collective-free Bridge API that exports native MXFP8 QKVO and routed-expert value/scale pairs for downstream refit.

**Architecture:** `quant_bridge.py` owns Transformer Engine storage discovery, validation, grouped-member iteration, and deterministic task traversal. `param_mapping.py` owns atomic projection of an E4M3 value tensor and its E8M0 scale tensor into canonical local HF parameters with explicit shard metadata. Existing logical `LocalHFParamSpec`, BF16 export, and requantized FP8 paths remain unchanged.

**Tech Stack:** Python 3.12, PyTorch, Transformer Engine metadata, Megatron Core process-group metadata, pytest, Ruff, Pyright, pre-commit.

**Spec:** `docs/superpowers/specs/2026-09-03-native-mxfp8-component-export-design.md`

## Global Constraints

- Transformer Engine precision matchers decide which parameters use native MXFP8 storage.
- Direct export must not dequantize, quantize, or execute TP, ETP, EP, or PP payload collectives.
- Setup-time metadata broadcasts used by task planning remain allowed.
- Native values are E4M3 and native rowwise scales are compact E8M0 with 32-value K blocks.
- QKV projection must preserve GQA and `attention_output_gate` row ordering.
- Grouped expert members are created once during task planning and reused during repeated refits.
- Unsupported mapping transforms, DTensor/FSDP, and co-trained MTP fail before transport.
- Do not modify `3rdparty/Megatron-LM`.
- Run targeted tests on Linux; the local macOS arm64 environment cannot resolve `nvidia-resiliency-ext`.

---

### Task 1: Add the paired local MXFP8 mapping contract

**Files:**
- Modify: `src/megatron/bridge/models/conversion/param_mapping.py:45-215`
- Test: `tests/unit_tests/models/test_fp8_param_export.py`

**Interfaces:**
- Consumes: existing `MegatronParamMapping`, `WeightConversionTask`, and mapping process-group properties.
- Produces: `LocalMXFP8Param` and `MegatronParamMapping.local_mxfp8_params(...)`.

- [ ] **Step 1: Write the failing contract tests**

Add imports and tests that require an immutable paired result and fail-closed base behavior:

```python
from megatron.bridge.models.conversion.param_mapping import (
    LocalMXFP8Param,
    MegatronParamMapping,
)


def test_local_mxfp8_param_keeps_value_and_scale_atomic():
    weight = torch.zeros((8, 64), dtype=torch.float8_e4m3fn)
    scale = torch.zeros((8, 2), dtype=torch.uint8)
    result = LocalMXFP8Param(
        name="model.layers.0.self_attn.o_proj.weight",
        weight=weight,
        weight_scale=scale,
        global_weight_shape=torch.Size((8, 128)),
        shard_group="tp",
        shard_dim=1,
    )
    assert result.weight is weight
    assert result.weight_scale is scale
    assert result.shard_group == "tp"
    assert result.shard_dim == 1


def test_base_mapping_rejects_native_mxfp8_projection():
    mapping = _IdentityMapping("hf.weight", "decoder.weight")
    with pytest.raises(ValueError, match="does not support direct native MXFP8 export"):
        mapping.local_mxfp8_params(
            torch.zeros((8, 64), dtype=torch.float8_e4m3fn),
            torch.zeros((8, 2), dtype=torch.uint8),
            global_param_name="decoder.weight",
            megatron_module=SimpleNamespace(),
        )
```

- [ ] **Step 2: Run the tests and confirm the missing API failure**

Run on Linux:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'local_mxfp8_param or base_mapping_rejects_native' -q
```

Expected: collection fails because `LocalMXFP8Param` or `local_mxfp8_params` does not exist.

- [ ] **Step 3: Add the result type and fail-closed base method**

Add the exact public result shape:

```python
MXFP8ShardGroup = Literal["tp", "etp", "replicated"]


@dataclass(frozen=True)
class LocalMXFP8Param:
    name: str
    weight: torch.Tensor
    weight_scale: torch.Tensor
    global_weight_shape: torch.Size
    shard_group: MXFP8ShardGroup
    shard_dim: int | None
```

Add this method to `MegatronParamMapping`:

```python
def local_mxfp8_params(
    self,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    global_param_name: str,
    megatron_module: nn.Module,
) -> tuple[LocalMXFP8Param, ...]:
    raise ValueError(
        f"Mapping {type(self).__name__} for {global_param_name!r} "
        "does not support direct native MXFP8 export."
    )
```

Do not modify `LocalHFParamSpec`; its logical-tensor contract remains separate.

- [ ] **Step 4: Run the focused contract tests**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'local_mxfp8_param or base_mapping_rejects_native' -q
```

Expected: both tests pass.

- [ ] **Step 5: Commit the contract**

```bash
git add src/megatron/bridge/models/conversion/param_mapping.py \
  tests/unit_tests/models/test_fp8_param_export.py
git commit -s -m "feat(conversion): define native MXFP8 export contract"
```

### Task 2: Extract and validate Transformer Engine native MXFP8 storage

**Files:**
- Modify: `src/megatron/bridge/models/conversion/quant_bridge.py:15-35`
- Test: `tests/unit_tests/models/test_fp8_param_export.py`

**Interfaces:**
- Consumes: a local ordinary or cached grouped-member tensor accepted by Megatron Core `is_mxfp8tensor()` and exposing `get_metadata()`.
- Produces: `_NativeMXFP8Storage(weight, weight_scale)` and `_extract_native_mxfp8_storage(param, global_param_name)`.

- [ ] **Step 1: Write failing extraction and validation tests**

Use a small fake metadata tensor whose logical shape is `(5, 64)`, whose data is E4M3 bytes, and whose TE scale storage is padded to `(128, 4)`:

```python
class _FakeNativeMXFP8Tensor:
    shape = torch.Size((5, 64))
    ndim = 2

    def __init__(self):
        self.data_bytes = torch.arange(5 * 64, dtype=torch.uint8).view(5, 64)
        self.scale_bytes = torch.zeros((128, 4), dtype=torch.uint8)

    def get_metadata(self):
        return {
            "rowwise_data": self.data_bytes,
            "rowwise_scale_inv": self.scale_bytes,
            "is_2D_scaled": False,
            "quantizer": SimpleNamespace(block_len=32),
        }


def test_native_mxfp8_storage_crops_only_documented_padding():
    param = _FakeNativeMXFP8Tensor()
    storage = _extract_native_mxfp8_storage(param, "decoder.linear.weight")
    assert storage.weight.dtype == torch.float8_e4m3fn
    assert storage.weight.shape == (5, 64)
    assert storage.weight_scale.dtype == torch.uint8
    assert storage.weight_scale.shape == (5, 2)
    assert storage.weight.untyped_storage().data_ptr() == param.data_bytes.untyped_storage().data_ptr()
    assert storage.weight_scale.untyped_storage().data_ptr() == param.scale_bytes.untyped_storage().data_ptr()


def test_native_mxfp8_storage_preserves_zero_scale_bytes():
    param = _FakeNativeMXFP8Tensor()
    storage = _extract_native_mxfp8_storage(param, "decoder.linear.weight")
    assert torch.count_nonzero(storage.weight_scale) == 0
```

Parameterize malformed cases for missing metadata, wrong dtypes, noncontiguous backing data, `K % 32 != 0`, too-small scale storage, and a swizzled/non-rowwise scale marker. Require every error to contain the global parameter name.

- [ ] **Step 2: Run the extraction tests and verify failure**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'native_mxfp8_storage' -q
```

Expected: failure because `_extract_native_mxfp8_storage` is missing.

- [ ] **Step 3: Implement storage extraction without data conversion**

Import `is_mxfp8tensor` beside the existing grouped-MXFP8 helpers. Add an
internal pair and extraction helper:

```python
@dataclass(frozen=True)
class _NativeMXFP8Storage:
    weight: torch.Tensor
    weight_scale: torch.Tensor


def _extract_native_mxfp8_storage(
    param: torch.Tensor,
    global_param_name: str,
) -> _NativeMXFP8Storage:
    metadata = param.get_metadata()
    rowwise_data = metadata.get("rowwise_data")
    rowwise_scale = metadata.get("rowwise_scale_inv")
    logical_shape = torch.Size(param.shape)
    expected_scale_shape = torch.Size((*logical_shape[:-1], logical_shape[-1] // 32))
    if logical_shape[-1] % 32:
        raise ValueError(f"{global_param_name}: K={logical_shape[-1]} is not divisible by 32")
    if rowwise_data is None or rowwise_data.dtype != torch.uint8 or not rowwise_data.is_contiguous():
        raise ValueError(f"{global_param_name}: invalid native MXFP8 rowwise_data")
    if rowwise_scale is None or rowwise_scale.dtype != torch.uint8 or not rowwise_scale.is_contiguous():
        raise ValueError(f"{global_param_name}: invalid native MXFP8 rowwise_scale_inv")
    if getattr(metadata.get("quantizer"), "block_len", None) != 32:
        raise ValueError(f"{global_param_name}: expected an MXFP8 block length of 32")
    if rowwise_data.ndim != len(logical_shape) or rowwise_scale.ndim != len(expected_scale_shape):
        raise ValueError(f"{global_param_name}: native MXFP8 storage rank mismatch")
    if any(actual < expected for actual, expected in zip(rowwise_data.shape, logical_shape)):
        raise ValueError(f"{global_param_name}: native MXFP8 value storage is too small")
    if any(actual < expected for actual, expected in zip(rowwise_scale.shape, expected_scale_shape)):
        raise ValueError(f"{global_param_name}: native MXFP8 scale storage is too small")
    data_slices = tuple(slice(0, size) for size in logical_shape)
    scale_slices = tuple(slice(0, size) for size in expected_scale_shape)
    return _NativeMXFP8Storage(
        weight=rowwise_data[data_slices].view(torch.float8_e4m3fn),
        weight_scale=rowwise_scale[scale_slices],
    )
```

Implement rank-generic leading-dimension cropping rather than assuming every supported tensor is exactly 2-D. Never call `.contiguous()` on valid native storage because that can allocate and hide an invalid input layout.

- [ ] **Step 4: Run extraction tests**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'native_mxfp8_storage' -q
```

Expected: all extraction and malformed-metadata cases pass.

- [ ] **Step 5: Commit native storage extraction**

```bash
git add src/megatron/bridge/models/conversion/quant_bridge.py \
  tests/unit_tests/models/test_fp8_param_export.py
git commit -s -m "feat(conversion): extract native MXFP8 storage"
```

### Task 3: Project QKV and output projections with explicit shard metadata

**Files:**
- Modify: `src/megatron/bridge/models/conversion/param_mapping.py:1210-1545`
- Modify: `src/megatron/bridge/models/conversion/param_mapping.py:1753-1967`
- Test: `tests/unit_tests/models/test_fp8_param_export.py`

**Interfaces:**
- Consumes: `LocalMXFP8Param` from Task 1 and validated paired tensors from Task 2.
- Produces: local Q/K/V and O parameters with TP shard dimensions 0 and 1.

- [ ] **Step 1: Write failing GQA, gated-QKV, and O tests**

Add deterministic byte-pattern tests. For GQA TP2, use global `H=8`, `G=4`, `D=4`, and local `K=64`. The local packed tensor shape is `(32, 64)` and the local scale shape is `(32, 2)`:

```python
def test_qkv_native_mxfp8_gqa_tp2_preserves_row_order(monkeypatch):
    config = SimpleNamespace(
        num_attention_heads=8,
        num_query_groups=4,
        kv_channels=4,
        hidden_size=128,
        attention_output_gate=False,
    )
    mapping = QKVMapping("decoder.linear_qkv.weight", "hf.q", "hf.k", "hf.v")
    monkeypatch.setattr(type(mapping), "tp_size", PropertyMock(return_value=2))
    weight = torch.arange(32 * 64, dtype=torch.uint8).view(32, 64).view(torch.float8_e4m3fn)
    scale = torch.arange(32 * 2, dtype=torch.uint8).view(32, 2)
    params = mapping.local_mxfp8_params(
        weight,
        scale,
        global_param_name="decoder.linear_qkv.weight",
        megatron_module=SimpleNamespace(config=config),
    )
    assert [p.name for p in params] == ["hf.q", "hf.k", "hf.v"]
    assert [p.weight.shape for p in params] == [(16, 64), (8, 64), (8, 64)]
    assert [p.weight_scale.shape for p in params] == [(16, 2), (8, 2), (8, 2)]
    assert all(p.shard_group == "tp" and p.shard_dim == 0 for p in params)
```

Compare exact Q/K/V rows against `split_qkv_weights(local_config, ..., feature_dim=64)` for values and `feature_dim=2` for scales. Add a second case with `attention_output_gate=True` where local `(48, 64)` produces Q+Z `(32, 64)` and K/V `(8, 64)`. Add an O case where `(64, 32)` plus `(64, 1)` remain pointer-identical and report global shape `(64, 64)` with TP shard dimension 1.

- [ ] **Step 2: Run projection tests and verify failure**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'qkv_native_mxfp8 or output_native_mxfp8' -q
```

Expected: QKV and row-parallel mappings still use the fail-closed base method.

- [ ] **Step 3: Implement the QKV projection**

In `QKVMapping.local_mxfp8_params(...)`:

1. Copy only the scalar attention fields needed by `split_qkv_weights`.
2. Divide `num_attention_heads` and `num_query_groups` by `tp_size`; reject non-divisible values.
3. Call `split_qkv_weights(local_config, weight, feature_dim=weight.shape[-1])`.
4. Call `split_qkv_weights(local_config, weight_scale, feature_dim=weight_scale.shape[-1])`.
5. Zip names, values, and scales into paired `LocalMXFP8Param` records.
6. Compute each global shape by multiplying output dimension 0 by `tp_size`.

Do not call `split_qkv_weights_scale()`. It is for two-dimensional block scaling and divides the head dimension, which is wrong for native MXFP8 rowwise scales.

- [ ] **Step 4: Implement row-parallel O projection and guarded AutoMapping delegation**

Add `RowParallelMapping.local_mxfp8_params(...)` that validates `weight_scale.shape == (*weight.shape[:-1], weight.shape[-1] // 32)`, keeps both tensors as views, multiplies global dimension 1 by `tp_size`, and reports `shard_group="tp", shard_dim=1`.

Add `AutoMapping.local_mxfp8_params(...)` that resolves the concrete mapping from `megatron_module`, accepts only `RowParallelMapping` for this direct path, and delegates to it. Reject column-parallel, replicated, transpose, permutation, and DTensor/FSDP cases with a parameter-specific `ValueError`.

- [ ] **Step 5: Run dense projection tests**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'qkv_native_mxfp8 or output_native_mxfp8' -q
```

Expected: MHA/GQA, gated-QKV, and O tests pass at TP1 and TP2.

- [ ] **Step 6: Commit dense projection support**

```bash
git add src/megatron/bridge/models/conversion/param_mapping.py \
  tests/unit_tests/models/test_fp8_param_export.py
git commit -s -m "feat(conversion): project native MXFP8 QKVO weights"
```

### Task 4: Project ordinary and grouped routed experts

**Files:**
- Modify: `src/megatron/bridge/models/conversion/param_mapping.py:2694-3181`
- Modify: `src/megatron/bridge/models/conversion/quant_bridge.py:35-215`
- Test: `tests/unit_tests/models/test_fp8_param_export.py`

**Interfaces:**
- Consumes: `LocalMXFP8Param`, `_extract_native_mxfp8_storage`, and PR #5917 task planning.
- Produces: canonical gate/up/down expert parameters with global expert IDs and ETP shard metadata.

- [ ] **Step 1: Write failing FC1/FC2 projection tests**

Cover ordinary and grouped storage at EP2/ETP2. On EP rank 1 with four global experts, grouped members 0 and 1 must emit names for global experts 2 and 3:

```python
def test_grouped_fc1_native_mxfp8_uses_global_expert_ids(monkeypatch):
    mapping = FusedGatedExpertMapping(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        "model.layers.0.mlp.experts.gate_up_proj",
    )
    monkeypatch.setattr(type(mapping), "ep_rank", PropertyMock(return_value=1))
    monkeypatch.setattr(type(mapping), "ep_size", PropertyMock(return_value=2))
    monkeypatch.setattr(type(mapping), "etp_size", PropertyMock(return_value=2))
    members = torch.arange(2 * 8 * 64, dtype=torch.uint8).view(2, 8, 64).view(torch.float8_e4m3fn)
    scales = torch.arange(2 * 8 * 2, dtype=torch.uint8).view(2, 8, 2)

    projected = []
    for local_expert_id in range(2):
        global_expert_id = mapping.ep_rank * 2 + local_expert_id
        projected.extend(
            mapping.local_mxfp8_params(
                members[local_expert_id],
                scales[local_expert_id],
                global_param_name=f"decoder.layers.0.mlp.experts.linear_fc1.weight{global_expert_id}",
                megatron_module=SimpleNamespace(),
            )
        )

    assert [p.name for p in projected] == [
        "model.layers.0.mlp.experts.2.gate_proj.weight",
        "model.layers.0.mlp.experts.2.up_proj.weight",
        "model.layers.0.mlp.experts.3.gate_proj.weight",
        "model.layers.0.mlp.experts.3.up_proj.weight",
    ]
    assert all(p.shard_group == "etp" and p.shard_dim == 0 for p in projected)
    assert all(p.weight.shape == (4, 64) and p.weight_scale.shape == (4, 2) for p in projected)
```

For FC2, assert down-projection names for experts 2/3 and `shard_group="etp", shard_dim=1`. Add negative cases for odd FC1 rows, grouped member-count mismatch, transpose/permutation flags, and missing cached grouped members.

- [ ] **Step 2: Run expert tests and verify failure**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'fc1_native_mxfp8 or fc2_native_mxfp8 or grouped_native_mxfp8' -q
```

Expected: expert mappings do not yet implement the native projection contract.

- [ ] **Step 3: Implement FC1 and FC2 mapping projections**

Implement atomic paired projection in `GatedMLPMapping`, `FusedGatedExpertMapping`, and `FusedExpertMapping`:

```python
gate_weight, up_weight = torch.chunk(weight, 2, dim=-2)
gate_scale, up_scale = torch.chunk(weight_scale, 2, dim=-2)
```

For grouped members, compute:

```python
global_expert_id = mapping.ep_rank * experts_per_rank + local_expert_id
```

Call the mapping with `global_param_name=f"{task.global_param_name}{global_expert_id}"` so each resolved mapping's existing canonical expert naming rules emit the right name. Do not reuse the expert-0 name. Report global shape and ETP shard dimension 0 for FC1 and 1 for FC2.

- [ ] **Step 4: Generalize grouped task classification**

Replace exact `grouped_suffixes` recognition in `build_export_mxfp8_tasks()` with a helper that:

1. looks up the normal global mapping;
2. if absent, tries the grouped member form `f"{global_name}0"`;
3. accepts grouped handling only when the resolved mapping reports `is_expert` and is `FusedGatedExpertMapping` or `FusedExpertMapping`;
4. detects native grouped storage with `is_grouped_mxfp8tensor()`;
5. calls `get_grouped_quantized_members(..., create_if_missing=True)` once during planning.

Keep existing global task ordering, PP/VP placeholders, BF16 grouped expansion, and MTP rejection unchanged.

- [ ] **Step 5: Run grouped and existing task-planner tests**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'build_export_mxfp8_tasks or fc1_native_mxfp8 or fc2_native_mxfp8 or grouped_native_mxfp8' -q
```

Expected: new expert tests and all existing PR #5917 task-ordering tests pass.

- [ ] **Step 6: Commit routed-expert support**

```bash
git add src/megatron/bridge/models/conversion/param_mapping.py \
  src/megatron/bridge/models/conversion/quant_bridge.py \
  tests/unit_tests/models/test_fp8_param_export.py
git commit -s -m "feat(conversion): project native MXFP8 routed experts"
```

### Task 5: Add deterministic materialization and complete regression checks

**Files:**
- Modify: `src/megatron/bridge/models/conversion/quant_bridge.py:30-215`
- Test: `tests/unit_tests/models/test_fp8_param_export.py`

**Interfaces:**
- Consumes: `build_export_mxfp8_tasks()`, `_extract_native_mxfp8_storage()`, and mapping `local_mxfp8_params()` implementations.
- Produces: `MegatronQuantizationBridge.iter_local_native_mxfp8_params(tasks) -> Iterator[LocalMXFP8Param]`.

- [ ] **Step 1: Write failing iterator tests**

Add tests for:

- stable task, local-expert, and mapping order;
- remote PP placeholders producing no values;
- mixed BF16/native tasks skipping BF16 without changing order;
- live storage refresh after mutating backing data/scale;
- grouped cached-member reuse without `create_if_missing=True` on the hot path;
- payload collectives patched to raise while local materialization succeeds;
- unsupported mapping, MTP, DTensor/FSDP, bad K alignment, bad scale shape, and bad grouped count failing with the global parameter name.

Use a concrete no-collective assertion:

```python
def _payload_collective_called(*_args, **_kwargs):
    raise AssertionError("native MXFP8 materialization called a payload collective")


def test_native_mxfp8_materialization_uses_no_payload_collectives(monkeypatch):
    monkeypatch.setattr(torch.distributed, "broadcast", _payload_collective_called)
    monkeypatch.setattr(torch.distributed, "all_gather", _payload_collective_called)
    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", _payload_collective_called)
    params = list(bridge.iter_local_native_mxfp8_params(tasks))
    assert params
```

- [ ] **Step 2: Run iterator tests and verify failure**

Run:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py \
  -k 'materialization or live_storage_refresh or no_payload_collectives' -q
```

Expected: failure because `iter_local_native_mxfp8_params` is missing.

- [ ] **Step 3: Implement deterministic local materialization**

Implement the public generator on `MegatronQuantizationBridge`:

```python
def iter_local_native_mxfp8_params(
    self,
    tasks: Iterable[WeightConversionTask],
) -> Iterator[LocalMXFP8Param]:
    for task in tasks:
        if task.param_weight is None:
            continue
        if is_grouped_mxfp8tensor(task.param_weight):
            members = get_grouped_quantized_members(task.param_weight, create_if_missing=False)
            yield from self._iter_grouped_native_mxfp8_params(task, members)
            continue
        if not is_mxfp8tensor(task.param_weight):
            continue
        storage = _extract_native_mxfp8_storage(task.param_weight, task.global_param_name)
        yield from task.mapping.local_mxfp8_params(
            storage.weight,
            storage.weight_scale,
            global_param_name=task.global_param_name,
            megatron_module=task.megatron_module,
        )
```

Keep destination layout qualification outside Bridge. Preserve E8M0 byte zero exactly; downstream code decides whether a backend accepts it.

- [ ] **Step 4: Run the full focused unit file**

Run on a Linux GPU-capable development node or CI container:

```bash
uv run pytest tests/unit_tests/models/test_fp8_param_export.py -q
```

Expected: all tests in the file pass.

- [ ] **Step 5: Run static and repository checks**

Run:

```bash
uv run ruff check \
  src/megatron/bridge/models/conversion/param_mapping.py \
  src/megatron/bridge/models/conversion/quant_bridge.py \
  tests/unit_tests/models/test_fp8_param_export.py
uv run pyright \
  src/megatron/bridge/models/conversion/param_mapping.py \
  src/megatron/bridge/models/conversion/quant_bridge.py
uv run pre-commit run --all-files
git diff --check
```

Expected: all checks exit zero. Do not substitute macOS failures caused by Linux-only dependencies for code validation; run these commands remotely when necessary.

- [ ] **Step 6: Commit the public iterator**

```bash
git add src/megatron/bridge/models/conversion/quant_bridge.py \
  tests/unit_tests/models/test_fp8_param_export.py
git commit -s -m "feat(conversion): materialize local native MXFP8 parameters"
```

- [ ] **Step 7: Verify downstream proof without adding it to this PR**

In the NeMo-RL integration branch, consume `iter_local_native_mxfp8_params()` and run:

1. QKVO plus routed-MoE native MXFP8 refit smoke.
2. Routed-MoE-only native MXFP8 refit smoke.
3. Placement/value parity against the existing conversion path.
4. Multi-step Qwen3.5 repeated-refit validation.

Record run links and measured refit/E2E results in the downstream PR. Do not claim a Bridge performance improvement from unit tests alone.
