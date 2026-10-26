#!/usr/bin/env python3
"""Patch verl's index_select_tensor_dict so it survives the PyTorch bug that
crashes unbind() on 3D+ jagged NestedTensors (mRoPE position_ids).

Root cause of the "split_with_sizes expects split_sizes to sum exactly to 471,
but got split_sizes=[4, 4, ...]" crash in the FSDP actor update
(engine_workers.train_mini_batch -> make_iterator -> index_select_tensor_dict):

  * Qwen3.5 position_ids are 3D per-sample (4, seq_len) -> batched by verl as a
    3D jagged NestedTensor [B, 4, *(seq_len)] with _values = (4, sum_seq_len).
  * tensordict pickle/unpickle (passing the batch between Ray workers) corrupts
    the nested tensor's internal offsets so that offsets.diff() == [4, 4, ...]
    (the mRoPE dim) instead of the true per-sample sequence lengths.
  * PyTorch unbind(dim=0) then does torch.split(_values, offsets.diff(), dim=1),
    i.e. splits dimension 1 (sum_seq_len) with split_sizes=[4, 4, ...] that sum
    to B*4 != sum_seq_len -> RuntimeError (see pytorch issue #153238).
  * verl already works around this in chunk_tensordict but index_select_tensor_dict
    still calls tensor.unbind() unguarded.

This patch recovers the per-sample (4, seq_len) slices from the still-correct
_values tensor, using the true sequence lengths read off a reliable 2D nested
tensor (input_ids / attention_mask), which are unaffected by the bug.
"""
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else "/workspace/verl/verl/utils/tensordict_utils.py"

HELPER_OLD = """def index_select_tensor_dict(batch: TensorDict, indices: torch.Tensor | list[int]) -> TensorDict:"""

HELPER_NEW = '''def _get_nested_seq_lens(batch: TensorDict) -> list[int] | None:
    """Recover per-sample sequence lengths from a reliable 2D nested tensor.

    Used to rebuild 3D mRoPE position_ids whose internal offsets get corrupted
    by tensordict pickle/unpickle (pytorch issue #153238). 2D jagged tensors such
    as input_ids / attention_mask keep correct offsets and are unaffected.
    """
    for key in ("input_ids", "attention_mask", "response_mask", "loss_mask"):
        val = batch.get(key, None)
        if isinstance(val, torch.Tensor) and val.is_nested and val.dim() == 2:
            try:
                lengths = val.offsets().diff().tolist()
            except Exception:
                continue
            if len(lengths) == len(batch):
                return lengths
    return None


def index_select_tensor_dict(batch: TensorDict, indices: torch.Tensor | list[int]) -> TensorDict:'''

BODY_OLD = """    if batch is not None:
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                data_dict[key] = tensor[indices]
            elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                tensor_lst = tensor.unbind()  # for performance
                selected_tensors = [tensor_lst[idx] for idx in indices]
                data_dict[key] = nested_tensor_from_tensor_list(
                    selected_tensors, ragged_idx=getattr(tensor, "_ragged_idx", tensor.dim() - 1)
                )
            else:"""

BODY_NEW = """    if batch is not None:
        # True per-sample sequence lengths, read off a reliable 2D nested tensor.
        # Needed to rebuild 3D mRoPE position_ids whose offsets are corrupted.
        seq_lens = _get_nested_seq_lens(batch)

        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                data_dict[key] = tensor[indices]
            elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                ragged_idx = getattr(tensor, "_ragged_idx", tensor.dim() - 1)
                try:
                    tensor_lst = tensor.unbind()  # for performance
                    selected_tensors = [tensor_lst[idx] for idx in indices]
                except RuntimeError:
                    # PyTorch unbind(dim=0) crashes on 3D jagged NestedTensors whose
                    # offsets were corrupted by tensordict pickle/unpickle (mRoPE
                    # position_ids): _values is (mRoPE_dim, sum_seq_len) but offsets
                    # describe the mRoPE dim, so split_with_sizes sums to sum_seq_len
                    # instead of B * mRoPE_dim. Recover each sample from _values using
                    # the true sequence lengths above.
                    if tensor.dim() == 3 and seq_lens is not None and len(seq_lens) == len(batch):
                        values = tensor.values()  # (mRoPE_dim, sum_seq_len)
                        offsets = [0]
                        for length in seq_lens:
                            offsets.append(offsets[-1] + length)
                        selected_tensors = [values[:, offsets[idx] : offsets[idx + 1]] for idx in indices.tolist()]
                    else:
                        raise
                data_dict[key] = nested_tensor_from_tensor_list(selected_tensors, ragged_idx=ragged_idx)
            else:"""

with open(PATH, encoding="utf-8") as f:
    src = f.read()

if HELPER_OLD not in src:
    print("ERROR: index_select_tensor_dict signature not found", file=sys.stderr)
    sys.exit(1)
if BODY_OLD not in src:
    print("ERROR: nested-tensor branch in index_select_tensor_dict not found", file=sys.stderr)
    sys.exit(1)

src = src.replace(HELPER_OLD, HELPER_NEW, 1)
src = src.replace(BODY_OLD, BODY_NEW, 1)

with open(PATH, "w", encoding="utf-8") as f:
    f.write(src)

print("patched index_select_tensor_dict to survive corrupted 3D nested position_ids")
