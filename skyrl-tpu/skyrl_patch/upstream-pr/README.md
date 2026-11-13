# SkyRL TorchTPU change, upstream PR

The TPU change from [`../skyrl-tpu-on-2c54cefc.patch`](../skyrl-tpu-on-2c54cefc.patch) (the version used for the TPU
runs), cleaned and rebased for `NovaSky-AI/SkyRL`. Opened as
[NovaSky-AI/SkyRL#2591](https://github.com/NovaSky-AI/SkyRL/pull/2591) on 2026-10-30, from
`debjyotictr-sudo:torchtpu-fsdp2`; proposed first in issue [#2553](https://github.com/NovaSky-AI/SkyRL/issues/2553).

| File | Content |
|---|---|
| `skyrl-torchtpu-pr2591-on-e4fdd8c1.patch` | The PR's two commits against SkyRL `main` at `e4fdd8c1` (`git format-patch`; 20 files, +1218/−62): device module, TorchTPU workarounds, `reload` weight sync, vLLM-free render decoding, vendored padding helpers, config validation, CPU tests, and the review fixes |
| `PR_DESCRIPTION.md` | The PR description as posted |
| `ci_results/` | Logs of every step of SkyRL's CPU CI workflow on a clean Linux VM (2026-10-28, change applied at `dd8769f8`) |

Changes from the run version: history comments removed, rebased onto `main`, the Qwen3.5 workarounds made to work
with transformers 5.16 and 5.19, `reload` validation and config docs added, and 20 CPU tests added.

## PR commits

| Commit | Content |
|---|---|
| `3edcbf9d` | The change, rebased onto `e4fdd8c1` (applies without conflicts) |
| `923ca875` | Fixes for the automated reviews (Gemini, Greptile, Cursor Bugbot): Ray workers always get `TORCH_DEVICE_BACKEND_AUTOLOAD=0` on TPU; a rank's chip and topology env is applied before TorchTPU loads; `reload` exports to a per-job directory and removes only its own exports, copies the tokenizer and processor files, does not skip the first sync after a checkpoint restore, waits on a CPU group whose timeout covers the reload, runs off the worker event loop, and skips `/pause` and `/resume`; IPv6 agent URLs; clear errors for unreachable agents; `getattr` for private / newer torch APIs; 6 tests |

Before pushing `923ca875`: pre-commit passed; SkyRL's first CPU test step passed locally (2561 passed; the
`flash_attn`-only `test_models.py` cannot import on macOS); and a smoke test on one v6e-8 passed (SkyRL `e4fdd8c1` +
both commits, Qwen3.5-0.8B, 2 GRPO steps, policy on 4 chips): workers started on their chips, two `reload` syncs
(21.6 s, 23.7 s) through the new barrier and worker thread, only the job's latest export kept (`job-<id>/v000002`),
and the export holds the tokenizer and processor files. Both steps had reward 0 (the 0.8B model passed no sample),
so the synced weights were unchanged; the test covers the sync path, not learning.

## CI results (Ubuntu 22.04, e2-standard-4)

Every step of SkyRL's `.github/workflows/cpu_skyrl.yaml`, with its exact commands:

| CI step | Result |
|---|---|
| pre-commit (`--config .pre-commit-config.yaml`) | ruff, black, gitleaks passed |
| `pytest ... -m "not (vllm or megatron)"` (`--extra skyrl-train`) | 2514 passed, 41 skipped |
| `pytest ... -m "vllm and not megatron"` (`--extra fsdp`) | 433 passed, 13 skipped |
| `pytest ... -m "megatron"` (`--extra megatron`) | 113 passed |
| `pytest tests/tinker/skyrl_train/` (`--extra fsdp --extra tinker`) | 93 passed, 22 skipped |
| `pytest --forked -s tests/tinker tests/utils` (`--extra tinker --extra jax`) | 144 passed |
| `skyrl-gym`: `uv run --frozen pytest tests/` | 119 passed |

No test fails. GPU CI (H100) was not run.
