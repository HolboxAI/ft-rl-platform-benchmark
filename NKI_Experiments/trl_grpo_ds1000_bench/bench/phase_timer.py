"""Device-synchronised phase timer.

    t = PhaseTimer(backend="cuda", sync=True)
    with t.phase("generation"):
        out = model.generate(...)
    t.events  # list of dicts, one per call -> metrics.csv_store.PhaseEventWriter

Phases can nest; each event records its parent so summaries can compute "share of parent".
With sync=True the timer waits for the device at both edges of every phase, so time is
attributed to the phase that queued the work (needed on Neuron, where work is lazy).
"""

import time
from contextlib import contextmanager

from bench.platform_utils import synchronize


class PhaseTimer:
    def __init__(self, backend, sync=True, on_event=None):
        self.backend = backend
        self.sync = sync
        self.on_event = on_event          # callback(event_dict), e.g. a CSV writer
        self.step = 0                     # set by the trainer callback
        self.events = []
        self._stack = []
        self._call_idx = {}
        self.t0 = time.time()

    @contextmanager
    def phase(self, name, **extra):
        if self.sync:
            synchronize(self.backend)
        parent = self._stack[-1] if self._stack else ""
        self._stack.append(name)
        start_wall = time.time()
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync:
                synchronize(self.backend)
            dur = time.perf_counter() - start
            self._stack.pop()
            idx = self._call_idx.get(name, 0)
            self._call_idx[name] = idx + 1
            ev = {
                "step": self.step,
                "phase": name,
                "parent": parent,
                "call_idx": idx,
                "start_unix": round(start_wall, 6),
                "rel_start_s": round(start_wall - self.t0, 6),
                "duration_s": round(dur, 6),
            }
            ev.update(extra)
            self.events.append(ev)
            if self.on_event:
                self.on_event(ev)

    def wrap(self, fn, name, name_fn=None):
        """Return fn wrapped in a phase. name_fn(*args, **kw) may pick the name per call."""
        timer = self

        def wrapped(*args, **kwargs):
            with timer.phase(name_fn(*args, **kwargs) if name_fn else name):
                return fn(*args, **kwargs)

        wrapped.__wrapped__ = fn
        return wrapped

    def step_totals(self, step):
        """Sum of duration per phase for one step (nested phases are not subtracted)."""
        out = {}
        for e in self.events:
            if e["step"] == step:
                out[e["phase"]] = out.get(e["phase"], 0.0) + e["duration_s"]
        return out
