"""Background sampler: accelerator utilisation, memory and power -> device_util.csv.

  GPU     : polls `nvidia-smi` (util %, memory, power, SM clock, temperature)
  Trainium: streams `neuron-monitor` JSON and flattens it to one row per NeuronCore
            (neuroncore utilisation %, runtime device memory). Power is not reported.

Run it alongside training (scripts/run_*.sh do this):
  python -m metrics.device_monitor --backend cuda --out results/<run>/device_util.csv &

Rows carry a unix timestamp, so metrics/summarize_run.py can line them up with the
step start/end times in step_metrics.csv.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics.csv_store import CsvWriter, DEVICE_FIELDS  # noqa: E402

_STOP = False


def _stop(*_):
    global _STOP
    _STOP = True


def _num(x):
    try:
        return float(x)
    except Exception:
        return None


def monitor_cuda(writer, interval_s):
    q = "index,utilization.gpu,memory.used,memory.total,power.draw,clocks.sm,temperature.gpu"
    proc = subprocess.Popen(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits",
         f"-lms", str(int(interval_s * 1000))],
        stdout=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if _STOP:
                break
            p = [s.strip() for s in line.split(",")]
            if len(p) < 7:
                continue
            writer.write({"timestamp_unix": round(time.time(), 3), "backend": "cuda",
                          "device_idx": p[0], "util_pct": _num(p[1]), "mem_used_mb": _num(p[2]),
                          "mem_total_mb": _num(p[3]), "power_w": _num(p[4]),
                          "sm_clock_mhz": _num(p[5]), "temp_c": _num(p[6])})
    finally:
        proc.terminate()


def _neuron_rows(report, ts):
    """Flatten one neuron-monitor JSON report. Field names follow the neuron-monitor docs;
    missing sections are skipped rather than failing."""
    rows = []
    for rt in report.get("neuron_runtime_data", []) or []:
        rep = rt.get("report", {}) or {}
        cores = ((rep.get("neuroncore_counters") or {}).get("neuroncores_in_use") or {})
        mem = ((rep.get("memory_used") or {}).get("neuron_runtime_used_bytes") or {})
        dev_mem = mem.get("neuron_device")
        for idx, c in cores.items():
            rows.append({"timestamp_unix": ts, "backend": "neuron", "device_idx": f"nc{idx}",
                         "util_pct": _num(c.get("neuroncore_utilization")),
                         "mem_used_mb": (dev_mem / 2**20) if dev_mem else None})
    hw = (report.get("system_data") or {}).get("neuron_hw_counters") or {}
    if not rows and hw:
        rows.append({"timestamp_unix": ts, "backend": "neuron", "device_idx": "system"})
    return rows


def monitor_neuron(writer, interval_s):
    cfg = {"period": f"{max(1, int(interval_s))}s",
           "neuron_runtimes": [{"tag_filter": ".*", "metrics": [
               {"type": "neuroncore_counters"}, {"type": "memory_used"},
               {"type": "neuron_runtime_vcpu_usage"}, {"type": "execution_stats"}]}],
           "system_metrics": [{"type": "neuron_hw_counters"}]}
    cfg_path = os.path.join(tempfile.mkdtemp(), "neuron_monitor.json")
    json.dump(cfg, open(cfg_path, "w"))
    proc = subprocess.Popen(["neuron-monitor", "-c", cfg_path], stdout=subprocess.PIPE, text=True)
    try:
        for line in proc.stdout:
            if _STOP:
                break
            try:
                report = json.loads(line)
            except json.JSONDecodeError:
                continue
            for row in _neuron_rows(report, round(time.time(), 3)):
                writer.write(row)
    finally:
        proc.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cuda", "neuron"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=0.5, help="seconds (neuron: min 1s)")
    a = p.parse_args()
    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    w = CsvWriter(a.out, DEVICE_FIELDS)
    try:
        (monitor_cuda if a.backend == "cuda" else monitor_neuron)(w, a.interval)
    finally:
        w.close()


if __name__ == "__main__":
    main()
