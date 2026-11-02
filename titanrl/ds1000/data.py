"""DS-1000 dataset for TitanRL (code-completion RL).

Each sample is one DS-1000 problem: a partial Python function (``prompt``) the
model completes, plus the reference solution and the library it belongs to.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

from datasets import load_dataset

from torchtitan.rl.rollout.data import RLDataset


@dataclass(frozen=True, kw_only=True, slots=True)
class Ds1000Sample:
    """One DS-1000 code-completion problem."""

    problem_id: int
    prompt: str
    reference_code: str
    library: str


class Ds1000Dataset(RLDataset):
    """Streams DS-1000 problems as code-completion samples."""

    @dataclass(kw_only=True, slots=True)
    class Config(RLDataset.Config):
        seed: int = 1337
        max_problems: int = 0
        """If > 0, cap the stream at this many problems (0 = all)."""

        hf_dataset: str = "xlangai/DS-1000"
        hf_split: str = "test"

    def __init__(self, config: Config) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._problems = _load_problems(config)
        self._order = list(range(len(self._problems)))
        self._rng.shuffle(self._order)
        self._idx = 0

    def __iter__(self) -> Iterator[Ds1000Sample]:
        return self

    def __next__(self) -> Ds1000Sample:
        if self._idx >= len(self._order):
            self._rng.shuffle(self._order)
            self._idx = 0
        i = self._order[self._idx]
        self._idx += 1
        return self._problems[i]

    def state_dict(self) -> dict:
        return {"rng_state": self._rng.getstate(), "idx": self._idx, "order": self._order}

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._idx = state_dict["idx"]
        self._order = state_dict["order"]


def _load_problems(config: "Ds1000Dataset.Config") -> list[Ds1000Sample]:
    raw = load_dataset(config.hf_dataset, split=config.hf_split)
    rows = []
    for i, r in enumerate(raw):
        meta = r["metadata"]
        prompt = str(r["prompt"])
        # Skip prompts that won't fit the 2048-token context with a 512-token
        # response (~4 chars/token for code; 5000 chars ≈ 1250 tokens).
        if len(prompt) > 5000:
            continue
        rows.append(
            Ds1000Sample(
                problem_id=i,
                prompt=prompt,
                reference_code=str(r.get("reference_code") or ""),
                library=str(meta.get("library") or ""),
            )
        )
    if config.max_problems > 0:
        rows = rows[: config.max_problems]
    return rows
