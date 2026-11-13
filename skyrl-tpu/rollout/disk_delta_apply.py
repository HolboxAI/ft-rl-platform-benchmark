"""Receive side of Miles' disk-delta weight sync, for the TPU rollout server (no SGLang).

Miles (``--update-weight-transfer-mode disk-delta``, ``protocols/delta.py`` @8e99947) publishes each weight version
as ``<source_dir>/weight_v<NNNNNN>/``: one or more ``model-*.safetensors`` files whose tensors are zstd-compressed
byte deltas of the changed HF tensors, plus ``model.safetensors.index.json`` whose ``metadata`` names the encoding
(``xor`` | ``overwrite``) and the checksum (``xxh3-128`` | ``blake3`` | ``adler32``). Each file's safetensors
``__metadata__`` maps tensor name -> checksum of the tensor's NEW bytes.

In SGLang the engine's ``/pull_weights`` applies them (``sglang.srt.weight_sync.local_checkpoint``). This module is
the same contract for ``rollout_supervisor.py``:

* ``target_version == 0``: make ``local_dir`` a fresh copy of the base checkpoint (Miles calls this once, before its
  first sync, and diffs every later version against that base).
* ``target_version == v``: apply versions ``current+1 .. v`` in order, in place, checking each tensor's checksum.

Several engines on one host share ``local_dir`` (the v6e-8 layout runs 4). An ``flock`` serialises them and a version
marker makes the apply exactly-once per host: the xor delta is an involution, so applying it twice would revert it.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import struct
import time
import zlib
from contextlib import contextmanager

import numpy as np
import zstandard

VERSION_FILE = ".miles_delta_version"
LOCK_FILE = ".miles_delta.lock"


# ------------------------------------------------------------------ checksums (as miles/utils/disk_delta.py)
def checksum(algorithm: str, buf) -> str:
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128(buf).hexdigest()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3(buf).hexdigest()
    if algorithm == "adler32":
        return f"{zlib.adler32(buf, 1):08x}"
    raise KeyError(f"unknown checksum algorithm {algorithm!r}")


# ------------------------------------------------------------------ safetensors raw access
def _header(path: str) -> tuple[int, dict]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return n, json.loads(f.read(n))


def tensor_locations(ckpt_dir: str) -> dict[str, tuple[str, int, int]]:
    """name -> (file, byte offset, nbytes) for every tensor of the checkpoint in ``ckpt_dir``."""
    out = {}
    for name in sorted(os.listdir(ckpt_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(ckpt_dir, name)
        n, header = _header(path)
        for key, info in header.items():
            if key == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            out[key] = (path, 8 + n + begin, end - begin)
    return out


def read_delta_file(path: str) -> tuple[dict[str, bytes], dict[str, str]]:
    """The compressed delta of each tensor in one published file, and the file's checksum metadata."""
    n, header = _header(path)
    meta = header.pop("__metadata__", {}) or {}
    out = {}
    with open(path, "rb") as f:
        for key, info in header.items():
            begin, end = info["data_offsets"]
            f.seek(8 + n + begin)
            out[key] = f.read(end - begin)
    return out, meta


# ------------------------------------------------------------------ decoding
def overwrite_decode(new_into: np.ndarray, delta: np.ndarray) -> None:
    """Inverse of miles' ``overwrite_encode``: count (u4), positions (u4 each), then the new byte values."""
    count = int(delta[:4].view("<u4")[0])
    pos = delta[4 : 4 + 4 * count].view("<u4").astype(np.int64)
    new_into[pos] = delta[4 + 4 * count : 4 + 5 * count]


def apply_version(local_dir: str, version_dir: str, locations: dict) -> dict:
    with open(os.path.join(version_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    meta = index.get("metadata") or {}
    encoding = meta.get("delta_encoding", "xor")
    algorithm = meta.get("checksum_format", "xxh3-128")
    if meta.get("compression_format", "zstd") != "zstd":
        raise ValueError(f"unsupported compression {meta.get('compression_format')!r}")
    dctx = zstandard.ZstdDecompressor()
    n_tensors = n_bytes = 0
    for fname in sorted(set(index["weight_map"].values())):
        deltas, checksums = read_delta_file(os.path.join(version_dir, fname))
        for name, blob in deltas.items():
            if name not in locations:
                raise KeyError(f"published tensor {name!r} is not in the local checkpoint {local_dir}")
            path, offset, nbytes = locations[name]
            mm = np.memmap(path, dtype=np.uint8, mode="r+", offset=offset, shape=(nbytes,))
            diff = np.frombuffer(dctx.decompress(blob, max_output_size=max(nbytes * 2, 1 << 20)), dtype=np.uint8)
            if encoding == "xor":
                if diff.size != nbytes:
                    raise ValueError(f"{name}: xor delta has {diff.size} bytes, tensor has {nbytes}")
                np.bitwise_xor(mm, diff, out=mm)
            elif encoding == "overwrite":
                overwrite_decode(mm, diff)
            else:
                raise ValueError(f"unknown delta encoding {encoding!r}")
            want = checksums.get(name)
            if want is not None and checksum(algorithm, memoryview(mm)) != want:
                raise ValueError(f"{name}: checksum mismatch after applying {version_dir}")
            mm.flush()
            del mm
            n_tensors += 1
            n_bytes += len(blob)
    return {"tensors": n_tensors, "wire_bytes": n_bytes, "encoding": encoding}


# ------------------------------------------------------------------ the /pull_weights contract
@contextmanager
def _locked(local_dir: str):
    os.makedirs(local_dir, exist_ok=True)
    with open(os.path.join(local_dir, LOCK_FILE), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def local_version(local_dir: str) -> int | None:
    try:
        with open(os.path.join(local_dir, VERSION_FILE)) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _set_version(local_dir: str, version: int) -> None:
    tmp = os.path.join(local_dir, VERSION_FILE + ".tmp")
    with open(tmp, "w") as f:
        f.write(str(version))
    os.replace(tmp, os.path.join(local_dir, VERSION_FILE))


def _materialize_base(base_model: str, local_dir: str) -> None:
    """A real copy (not hard links): deltas are applied to these files in place."""
    for name in os.listdir(local_dir):
        if name not in (LOCK_FILE,):
            path = os.path.join(local_dir, name)
            shutil.rmtree(path) if os.path.isdir(path) else os.remove(path)
    for name in os.listdir(base_model):
        src = os.path.join(base_model, name)
        if os.path.isfile(src):
            shutil.copyfile(src, os.path.join(local_dir, name))


def pull_weights(*, base_model: str, local_dir: str, source_dir: str, target_version: int) -> dict:
    t0 = time.perf_counter()
    with _locked(local_dir):
        current = local_version(local_dir)
        if target_version == 0:
            if current != 0:
                _materialize_base(base_model, local_dir)
                _set_version(local_dir, 0)
            return {"success": True, "message": f"base at {local_dir}", "version": 0,
                    "seconds": round(time.perf_counter() - t0, 2)}
        if current is None:
            raise RuntimeError(f"{local_dir} has no base; pull_weights(target_version=0) must come first")
        if current > target_version:
            raise RuntimeError(f"{local_dir} is at version {current}, past the requested {target_version}")
        applied = []
        locations = tensor_locations(local_dir) if current < target_version else {}
        for v in range(current + 1, target_version + 1):
            stats = apply_version(local_dir, os.path.join(source_dir, f"weight_v{v:06d}"), locations)
            _set_version(local_dir, v)
            applied.append({"version": v, **stats})
        return {"success": True, "message": f"{local_dir} at version {target_version}", "version": target_version,
                "applied": applied, "seconds": round(time.perf_counter() - t0, 2)}
