#!/usr/bin/env python3
"""Rebuild 3D mRoPE position_ids in maybe_fix_3d_position_ids.

Root cause (confirmed by [PID_DIAG2] tracing): the rollout path builds Qwen3.5
position_ids as a 3D jagged NestedTensor in BATCH-MAJOR layout:

    values  = (batch * mRoPE, seq)   # e.g. (128, 471) for batch=32, mRoPE=4
    offsets = [0, 4, 8, 12, ...]     # diff == mRoPE (4), NOT the seq lengths
    _ragged_idx = 1

i.e. the mRoPE dim (4) was treated as the *ragged* dim and seq (471) as the fixed
inner dim. The engine (chunk_tensordict -> prepare_model_inputs ->
to_padded_tensor(..., output_size=(B, 4, seq))) then hands the model a tensor
whose leading dim is batch*4 -> "size of tensor a (3) must match tensor b (128)".

The original verl workaround only sets _ragged_idx = 2, which does NOT change the
transposed values/offsets. This patch detects the batch-major corruption and
rebuilds a proper (mRoPE, sum_seq) nested tensor using the true per-sample seq
lengths read off input_ids (a 2D jagged tensor that is unaffected).
"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl/verl/utils/tensordict_utils.py"

OLD = """def maybe_fix_3d_position_ids(data: TensorDict):
    # note for tensordict with pickle/unpickle. nested tensor in tensordict after consolidate and pickle/unpickle
    # will incur indexing error for ragged tensor. This only happens when using 3D position ids in VLMs.
    # This is likely a bug in tensordict. As a workaround, we manually set _ragged_index.
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        data["position_ids"]._ragged_idx = 2
"""

NEW = '''def maybe_fix_3d_position_ids(data: TensorDict):
    # note for tensordict with pickle/unpickle. nested tensor in tensordict after consolidate and pickle/unpickle
    # will incur indexing error for ragged tensor. This only happens when using 3D position ids in VLMs.
    # This is likely a bug in tensordict. As a workaround, we manually set _ragged_index.
    if "position_ids" in data.keys() and data["position_ids"].dim() == 3 and data["position_ids"].is_nested:
        pid = data["position_ids"]
        values = pid.values()
        batch = int(data.batch_size[0])
        if values.shape[0] == 4:
            # Correct (mRoPE, sum_seq) layout: only _ragged_idx was clobbered by
            # tensordict consolidate/pickle-unpickle. Restore it.
            pid._ragged_idx = 2
        elif batch > 0 and values.shape[0] == batch * 4 and values.dim() == 2:
            # Corrupted BATCH-MAJOR layout: values = (batch * mRoPE, seq), offsets
            # diff == mRoPE. Rebuild as a proper (mRoPE, sum_seq) nested tensor.
            mrope = 4
            seq = values.shape[1]
            padded = values.reshape(batch, mrope, seq).transpose(0, 1).contiguous()  # (mrope, batch, seq)
            seqlens = None
            if "input_ids" in data.keys() and data["input_ids"].is_nested:
                try:
                    _lens = data["input_ids"].offsets().diff().tolist()
                    if len(_lens) == batch:
                        seqlens = _lens
                except Exception:  # noqa: BLE001
                    seqlens = None
            if seqlens is None:
                seqlens = [seq] * batch
            tensors = [padded[:, i, :l].contiguous() for i, l in enumerate(seqlens)]
            data["position_ids"] = nested_tensor_from_tensor_list(tensors, ragged_idx=2)
            try:
                print(
                    f"[POSID_FIX] rebuilt position_ids: values={list(data['position_ids'].values().shape)} "
                    f"offsets={data['position_ids'].offsets().tolist()[:6]}... seqlens={seqlens[:6]}",
                    flush=True,
                )
            except Exception as _e:  # noqa: BLE001
                print(f"[POSID_FIX] rebuild print failed: {_e}", flush=True)
        else:
            pid._ragged_idx = 2
'''

with open(PATH, encoding="utf-8") as f:
    src = f.read()

if OLD not in src:
    print("ERROR: original maybe_fix_3d_position_ids not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(OLD, NEW, 1)
with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("patched maybe_fix_3d_position_ids to rebuild batch-major position_ids")
