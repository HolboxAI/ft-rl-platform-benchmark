#!/usr/bin/env python3
"""Owns one rollout server process and swaps its weights by restarting it.

This is the agent side of SkyRL's ``reload`` weight-sync backend
(``skyrl/backends/skyrl_train/weight_sync/reload.py`` in the skyrl-tpu patch).
A TPU chip can be held by only one process, and the rollout engine cannot take
new weights in place, so a weight update is: stop the server, start it on the
new checkpoint, wait until ``/health`` answers.

Agent API (default port = server port + 1):

* ``PUT  /v1/files/{version}/{name}``   stream one checkpoint file into ``<work-dir>/v{version}/``
* ``POST /v1/reload``  ``{"version": v}`` (files uploaded) or ``{"version": v, "model_dir": d}``
  -> restarts the server on that checkpoint; answers once it is healthy
* ``GET  /v1/status``

The trainer only sends weights (``*.safetensors`` + index). Everything else the
server needs (tokenizer, processor and chat template, generation config, and by
default ``config.json``) is copied from the base checkpoint, so the rollout uses
the same template and preprocessing on every step.

    python rollout_supervisor.py --base-model /data/models/cholec-sft-qwen3.5-4b \\
        --server-port 8000 -- --engine vllm --max-model-len 8192
(arguments after ``--`` go to rollout_server.py)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s supervisor: %(message)s")
logger = logging.getLogger("rollout.supervisor")

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_SUFFIXES = (".safetensors", ".safetensors.index.json", ".bin", ".bin.index.json", ".pt")


class Supervisor:
    def __init__(self, args, server_args: List[str]):
        self.args = args
        self.server_args = server_args
        self.base_model = os.path.abspath(args.base_model)
        self.served_name = args.served_model_name or args.base_model
        self.work_dir = args.work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.proc: Optional[subprocess.Popen] = None
        self.model_dir = self.base_model
        self.version = 0
        self.lock = asyncio.Lock()
        self.history: List[dict] = []

    # ------------------------------------------------------------- process
    def _cmd(self, model_dir: str) -> List[str]:
        return [
            sys.executable, os.path.join(HERE, "rollout_server.py"),
            "--model", model_dir,
            "--served-model-name", self.served_name,
            "--weight-version", str(self.version),
            "--port", str(self.args.server_port),
            *self.server_args,
        ]

    def start(self, model_dir: str) -> None:
        log_path = os.path.join(self.args.log_dir, f"rollout_server_v{self.version:06d}.log")
        os.makedirs(self.args.log_dir, exist_ok=True)
        log = open(log_path, "ab")
        cmd = self._cmd(model_dir)
        logger.info(f"starting server v{self.version}: {' '.join(cmd)}  (log: {log_path})")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.model_dir = model_dir

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGTERM)
        try:
            self.proc.wait(timeout=self.args.stop_timeout_s)
        except subprocess.TimeoutExpired:
            logger.warning("server did not exit on SIGTERM; killing")
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        # libtpu releases the chip asynchronously after the process exits.
        time.sleep(self.args.tpu_release_s)

    def wait_healthy(self) -> float:
        t0 = time.perf_counter()
        url = f"http://127.0.0.1:{self.args.server_port}/health"
        while time.perf_counter() - t0 < self.args.start_timeout_s:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"rollout server exited with code {self.proc.returncode} during startup; see logs")
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    if r.status == 200:
                        return time.perf_counter() - t0
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"rollout server not healthy after {self.args.start_timeout_s}s")

    # ------------------------------------------------------------ checkpoint
    def assemble(self, version: int, model_dir: Optional[str]) -> str:
        """A complete HF dir for ``version``: trainer weights + base non-weight files."""
        target = os.path.join(self.work_dir, f"v{version:06d}")
        if model_dir is not None and os.path.abspath(model_dir) != target:
            os.makedirs(target, exist_ok=True)
            for name in os.listdir(model_dir):
                if name.endswith(WEIGHT_SUFFIXES):
                    dst = os.path.join(target, name)
                    if os.path.exists(dst):
                        os.remove(dst)
                    src = os.path.join(model_dir, name)
                    if _same_fs(model_dir, target):
                        os.link(src, dst)
                    else:
                        shutil.copy2(src, dst)
        if not any(n.endswith(".safetensors") for n in os.listdir(target)):
            raise ValueError(f"no *.safetensors in {target}")
        for name in os.listdir(self.base_model):
            src = os.path.join(self.base_model, name)
            if name.endswith(WEIGHT_SUFFIXES) or os.path.isdir(src):
                continue
            if name == "config.json" and self.args.use_trainer_config and os.path.exists(os.path.join(target, name)):
                continue
            shutil.copy2(src, os.path.join(target, name))
        return target

    def prune(self, keep: str) -> None:
        for entry in os.listdir(self.work_dir):
            path = os.path.join(self.work_dir, entry)
            if path != keep and entry.startswith("v") and os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)


def _same_fs(a: str, b: str) -> bool:
    return os.stat(a).st_dev == os.stat(b).st_dev


def create_app(sup: Supervisor) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/status")
    async def status():
        alive = sup.proc is not None and sup.proc.poll() is None
        return {"version": sup.version, "model_dir": sup.model_dir, "server_alive": alive,
                "server_port": sup.args.server_port, "history": sup.history[-20:]}

    @app.put("/v1/files/{version}/{name}")
    async def put_file(version: int, name: str, request: Request):
        if "/" in name or name.startswith("."):
            raise HTTPException(400, "bad file name")
        d = os.path.join(sup.work_dir, f"v{version:06d}")
        os.makedirs(d, exist_ok=True)
        tmp = os.path.join(d, f".{name}.part")
        n = 0
        with open(tmp, "wb") as f:
            async for chunk in request.stream():
                f.write(chunk)
                n += len(chunk)
        os.replace(tmp, os.path.join(d, name))
        return {"name": name, "bytes": n}

    @app.post("/v1/reload")
    async def reload(request: Request):
        body = await request.json()
        version = int(body["version"])
        async with sup.lock:
            loop = asyncio.get_running_loop()
            t0 = time.perf_counter()
            try:
                target = await loop.run_in_executor(None, sup.assemble, version, body.get("model_dir"))
                await loop.run_in_executor(None, sup.stop)
                t_stop = time.perf_counter() - t0
                sup.version = version
                sup.start(target)
                t_start = await loop.run_in_executor(None, sup.wait_healthy)
            except Exception as e:
                logger.exception("reload failed")
                raise HTTPException(500, f"reload to v{version} failed: {e}")
            await loop.run_in_executor(None, sup.prune, target)
            rec = {"version": version, "stop_s": round(t_stop, 1), "start_s": round(t_start, 1),
                   "total_s": round(time.perf_counter() - t0, 1), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            sup.history.append(rec)
            logger.info(f"reloaded: {json.dumps(rec)}")
            return {"ok": True, **rec}

    return app


def main():
    argv = sys.argv[1:]
    server_args: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, server_args = argv[:i], argv[i + 1 :]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True)
    p.add_argument("--served-model-name", default=None)
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--agent-port", type=int, default=None, help="default: server port + 1")
    p.add_argument("--work-dir", default="/tmp/rollout_weights")
    p.add_argument("--log-dir", default=os.path.join(HERE, "..", "logs", "rollout"))
    p.add_argument("--start-timeout-s", type=float, default=3600)
    p.add_argument("--stop-timeout-s", type=float, default=60)
    p.add_argument("--tpu-release-s", type=float, default=5)
    p.add_argument("--use-trainer-config", action="store_true",
                   help="keep the trainer's config.json instead of the base checkpoint's")
    args = p.parse_args(argv)
    args.log_dir = os.path.abspath(args.log_dir)

    sup = Supervisor(args, server_args)
    sup.start(sup.base_model)
    logger.info(f"waiting for the initial server (base checkpoint {sup.base_model})")
    logger.info(f"initial server healthy after {sup.wait_healthy():.1f}s")

    def _shutdown(*_):
        sup.stop()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)
    uvicorn.run(create_app(sup), host="0.0.0.0", port=args.agent_port or args.server_port + 1, log_level="warning")


if __name__ == "__main__":
    main()
