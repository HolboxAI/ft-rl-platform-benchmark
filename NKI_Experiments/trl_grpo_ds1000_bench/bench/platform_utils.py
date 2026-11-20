"""Device detection, synchronisation, memory and compiler counters for CUDA and Neuron.

Everything here works on both platforms and degrades to no-ops when a feature is missing,
so the training code never needs `if neuron: ... else: ...` branches.
"""

import importlib
import os
import platform
import socket
import subprocess


def _has(mod):
    try:
        importlib.import_module(mod)
        return True
    except Exception:
        return False


def detect_backend():
    """Return 'cuda', 'neuron' (torch-neuronx / XLA) or 'cpu'."""
    forced = os.environ.get("BENCH_BACKEND")
    if forced:
        return forced
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if _has("torch_neuronx") or _has("torch_xla"):
        return "neuron"
    return "cpu"


def get_device(backend):
    import torch
    if backend == "cuda":
        return torch.device("cuda")
    if backend == "neuron":
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    return torch.device("cpu")


def install_xla_workarounds():
    """Patch ops that the Neuron compiler rejects in stock HF/TRL code paths.

    torch.isin (used by transformers' EOS stopping criteria) lowers to an int64 `dot`, which
    neuronx-cc refuses (NCC_EVRF035). On XLA tensors we use an equivalent broadcast
    comparison instead; other devices keep the original op.
    """
    import torch
    if getattr(torch.isin, "_bench_patched", False):
        return
    orig = torch.isin

    def isin(elements, test_elements, *, assume_unique=False, invert=False):
        if isinstance(elements, torch.Tensor) and elements.device.type == "xla":
            t = test_elements if isinstance(test_elements, torch.Tensor) else torch.tensor(test_elements)
            t = t.to(device=elements.device, dtype=elements.dtype).reshape(-1)
            out = (elements.unsqueeze(-1) == t).any(-1)
            return ~out if invert else out
        return orig(elements, test_elements, assume_unique=assume_unique, invert=invert)

    isin._bench_patched = True
    torch.isin = isin

    # Integer cumsum (position ids from attention_mask.long().cumsum(-1)) lowers to an int64
    # triangular `dot` -> same NCC_EVRF035. Run it in fp32 (exact below 2**24) and cast back.
    _int_types = (torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8, torch.bool)
    orig_cumsum, orig_tcumsum = torch.cumsum, torch.Tensor.cumsum

    def _xla_int(x):
        return isinstance(x, torch.Tensor) and x.device.type == "xla" and x.dtype in _int_types

    def cumsum(input, dim, *, dtype=None, out=None):
        if out is None and _xla_int(input):
            res_dtype = dtype or (torch.int64 if input.dtype in (torch.bool, torch.uint8, torch.int8,
                                                                 torch.int16, torch.int32) else input.dtype)
            if res_dtype in _int_types:
                return orig_cumsum(input.float(), dim).round().to(res_dtype)
        return orig_cumsum(input, dim, dtype=dtype, out=out)

    def tcumsum(self, dim, *, dtype=None):
        return cumsum(self, dim, dtype=dtype) if _xla_int(self) else orig_tcumsum(self, dim, dtype=dtype)

    torch.cumsum = cumsum
    torch.Tensor.cumsum = tcumsum


def synchronize(backend):
    """Block until all queued device work has finished.

    On Neuron (lazy XLA tensors) this also cuts the graph and executes it, which is what makes
    per-phase timing possible -- and is also why phase-level sync changes performance a little.
    """
    if backend == "cuda":
        import torch
        torch.cuda.synchronize()
    elif backend == "neuron":
        try:
            import torch_xla
            torch_xla.sync()
        except Exception:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        import torch_xla.core.xla_model as xm
        xm.wait_device_ops()


def reset_peak_memory(backend):
    if backend == "cuda":
        import torch
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(backend):
    """Peak allocated device memory since the last reset (CUDA). Neuron memory comes from
    the device monitor (neuron-monitor) instead, so this returns None there."""
    if backend == "cuda":
        import torch
        return torch.cuda.max_memory_allocated() / 2**20
    if backend == "neuron":
        try:
            import torch_xla.core.xla_model as xm
            info = xm.get_memory_info(xm.xla_device())
            used = info.get("bytes_used") or (info.get("kb_total", 0) - info.get("kb_free", 0)) * 1024
            return used / 2**20 if used else None
        except Exception:
            return None
    return None


def compiler_counters(backend):
    """Cumulative graph-compile and execute counters (XLA only).

    Returns dict with compile_count, compile_time_s, execute_count, execute_time_s.
    Per-step deltas of these show how much of a Trainium step is spent recompiling.
    """
    out = {"compile_count": None, "compile_time_s": None,
           "execute_count": None, "execute_time_s": None}
    if backend != "neuron":
        return out
    try:
        import torch_xla.debug.metrics as met
        for key, name in (("compile", "CompileTime"), ("execute", "ExecuteTime")):
            data = met.metric_data(name)
            if data:
                out[f"{key}_count"] = data[0]
                out[f"{key}_time_s"] = data[1] / 1e9
    except Exception:
        pass
    return out


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def _imds(path):
    """EC2 instance metadata (IMDSv2). Empty string when not on EC2."""
    tok = _cmd(["curl", "-s", "-m", "2", "-X", "PUT", "http://169.254.169.254/latest/api/token",
                "-H", "X-aws-ec2-metadata-token-ttl-seconds: 60"])
    if not tok:
        return ""
    return _cmd(["curl", "-s", "-m", "2", "-H", f"X-aws-ec2-metadata-token: {tok}",
                 f"http://169.254.169.254/latest/meta-data/{path}"])


def collect_platform_info(backend):
    """Everything needed to reproduce and label a run. Written to run_info.csv."""
    info = {
        "backend": backend,
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "instance_type": os.environ.get("INSTANCE_TYPE") or _imds("instance-type") or "unknown",
        "region": os.environ.get("AWS_REGION") or _imds("placement/region") or "unknown",
        "instance_price_usd_hr": os.environ.get("INSTANCE_PRICE_USD_HR", ""),
        "accelerators_used": os.environ.get("BENCH_NUM_DEVICES", "1"),
    }
    for mod in ("torch", "transformers", "trl", "accelerate", "torch_xla", "torch_neuronx",
                "neuronxcc", "nki"):
        try:
            info[f"ver_{mod}"] = importlib.import_module(mod).__version__
        except Exception:
            info[f"ver_{mod}"] = ""
    if backend == "cuda":
        import torch
        info["device_name"] = torch.cuda.get_device_name(0)
        info["cuda_version"] = torch.version.cuda or ""
        info["driver"] = _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    elif backend == "neuron":
        info["device_name"] = _cmd(["bash", "-c", "neuron-ls --json-output 2>/dev/null | head -c 2000"])[:200] or "neuron"
        info["neuron_rt_visible_cores"] = os.environ.get("NEURON_RT_VISIBLE_CORES", "")
        info["neuron_logical_nc_config"] = os.environ.get("NEURON_LOGICAL_NC_CONFIG", "")
        info["neuron_cc_flags"] = os.environ.get("NEURON_CC_FLAGS", "")
        info["neuron_sdk_packages"] = _cmd(["bash", "-c",
            "pip list 2>/dev/null | grep -i -E 'neuron|torch-xla|torch_xla' | tr -s ' ' | tr '\\n' ';'"])
    return info
