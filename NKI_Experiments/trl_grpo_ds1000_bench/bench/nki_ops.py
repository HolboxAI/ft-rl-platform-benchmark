"""NKI kernels used by the benchmark (Trainium only).

Two uses:
  1. kernels/kernel_microbench.py times these kernels against the stock PyTorch op
     (compiled by neuronx-cc) and against the same PyTorch op on the GPU.
  2. train/grpo_ds1000_bench.py --nki-ops rmsnorm swaps the model's RMSNorm for the NKI
     kernel during rollout (no-grad forward passes: generation + reference/old log-probs).
     These kernels have no backward pass, so the training forward keeps the PyTorch op.
     Comparing a Trainium run with and without --nki-ops shows what the kernel buys inside RL.

Imports are guarded: on a GPU host, or with an NKI version whose API differs, `available()`
returns False with a reason and callers record that instead of crashing.

The NKI API moved between Neuron SDK releases (`neuronxcc.nki` -> standalone `nki`).
Both import paths are tried; check the reason string if `available()` is False.
"""

import math

_IMPORT_ERROR = None
nki = nl = None
try:
    try:
        import neuronxcc.nki as nki                # Neuron SDK 2.x
        import neuronxcc.nki.language as nl
    except Exception:
        import nki                                  # newer standalone package
        import nki.language as nl
except Exception as e:                              # GPU host or missing SDK
    _IMPORT_ERROR = f"{type(e).__name__}: {e}"


def available():
    return (_IMPORT_ERROR is None), (_IMPORT_ERROR or "")


# ---------------------------------------------------------------- RMSNorm kernel
# Follows the RMSNorm tutorial in the NKI docs, plus epsilon. Input is 2-D [rows, hidden];
# rows are processed in tiles of 128 (the partition dimension).
if _IMPORT_ERROR is None:
    @nki.jit
    def _rmsnorm_kernel(a_tensor, g_tensor, eps):
        out_tensor = nl.ndarray(a_tensor.shape, dtype=a_tensor.dtype, buffer=nl.shared_hbm)
        num_rows, hidden = a_tensor.shape
        ix = nl.arange(128)[:, None]
        iw = nl.arange(1)[:, None]
        iy = nl.arange(hidden)[None, :]
        g_tile = nl.load(g_tensor.reshape((1, hidden))[iw, iy])
        for i in nl.affine_range(math.ceil(num_rows / 128)):
            mask = i * 128 + ix < num_rows
            a_tile = nl.load(a_tensor[i * 128 + ix, iy], mask=mask)
            a_f32 = nl.copy(a_tile, dtype=nl.float32)
            sq_sum = nl.sum(nl.square(a_f32), axis=[1])
            rms_recip = nl.rsqrt(sq_sum / hidden + eps)
            out_tile = nl.multiply(a_f32, rms_recip)
            g_bcast = g_tile.broadcast_to((128, hidden))
            out_tile = nl.multiply(out_tile, g_bcast, mask=mask, dtype=a_tensor.dtype)
            nl.store(out_tensor[i * 128 + ix, iy], value=out_tile, mask=mask)
        return out_tensor


def nki_rmsnorm(x, weight, eps=1e-6):
    """RMSNorm over the last dim using the NKI kernel. x: [..., hidden] XLA tensor."""
    shape = x.shape
    out = _rmsnorm_kernel(x.reshape(-1, shape[-1]).contiguous(), weight, eps)
    return out.reshape(shape)


def torch_rmsnorm(x, weight, eps=1e-6):
    import torch
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype) * weight


# ---------------------------------------------------------------- flash attention
def get_flash_fwd():
    """NKI flash-attention forward from the Neuron kernel library, or (None, reason)."""
    for path in ("neuronxcc.nki.kernels.attention", "nki.kernels.attention",
                 "neuronxcc.nki._private_kernels.attention"):
        try:
            mod = __import__(path, fromlist=["flash_fwd"])
            return mod.flash_fwd, ""
        except Exception as e:
            reason = f"{path}: {type(e).__name__}: {e}"
    return None, reason


# ---------------------------------------------------------------- model patching
def patch_rmsnorm_for_rollout(model, counters=None):
    """Use the NKI RMSNorm in every RMSNorm module when autograd is off (rollout passes).

    Returns the number of modules patched. `counters` (dict) counts kernel vs fallback calls,
    so the run can prove the kernel was actually used.
    """
    import torch

    ok, reason = available()
    if not ok:
        raise RuntimeError(f"NKI not available: {reason}")
    patched = 0
    for mod in model.modules():
        if type(mod).__name__.endswith("RMSNorm") and hasattr(mod, "weight"):
            orig = mod.forward
            eps = getattr(mod, "variance_epsilon", getattr(mod, "eps", 1e-6))

            def fwd(hidden_states, _orig=orig, _mod=mod, _eps=eps):
                if torch.is_grad_enabled():
                    if counters is not None:
                        counters["torch"] = counters.get("torch", 0) + 1
                    return _orig(hidden_states)
                if counters is not None:
                    counters["nki"] = counters.get("nki", 0) + 1
                return nki_rmsnorm(hidden_states, _mod.weight, _eps)

            mod.forward = fwd
            patched += 1
    return patched
