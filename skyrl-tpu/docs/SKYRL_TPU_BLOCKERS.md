# SkyRL on Cloud TPU: blockers and how they were solved

This document records every problem that stopped or slowed the SkyRL CholecT50 GRPO run on Google Cloud TPU,
why each one happened, and what fixed it. It covers the work from 2026-10-23 to 2026-10-26.

## Contents

1. [Context](#1-context)
2. [Summary of all blockers](#2-summary-of-all-blockers)
3. [Cloud access, data and capacity](#3-cloud-access-data-and-capacity)
4. [Software installation](#4-software-installation)
5. [SkyRL itself: written for CUDA only](#5-skyrl-itself-written-for-cuda-only)
6. [TorchTPU gaps](#6-torchtpu-gaps)
7. [The rollout engine](#7-the-rollout-engine)
8. [Memory](#8-memory)
9. [The four training runs](#9-the-four-training-runs)
10. [Not solved: the reward collapse](#10-not-solved-the-reward-collapse)
11. [Lessons](#11-lessons)

---

## 1. Context

**Goal.** Repeat the SkyRL GPU run of CholecT50 GRPO on TPU, with the same configuration:

| Item | Value |
|---|---|
| Framework | SkyRL, commit `2c54cefc` (`RayPPOTrainer`, FSDP2) |
| Model | Qwen3.5-4B vision-language model, Tinker SFT checkpoint (merged) |
| Data | CholecT50, 256 train / 32 val frames (same parquet, SHA-256 checked) |
| Algorithm | GRPO, KL loss 0.04 (`low_var_kl`), 16 prompts × 8 samples, mini-batch 2 (8 optimizer updates per step), 16 steps |
| GPU reference | 1× `g5.12xlarge` (4× A10G), val reward 0.329 at the start, train reward collapsed to 0 at step 14 |
| TPU target | Cloud TPU v6e (Trillium), project `tpu-rl-run`, data in `gs://tpu-rl-run-cholec` |

**Final state.** Run `cholect50-skyrl-tpu-20261024-121826` completed 16 of 16 steps on one spot v6e-8 in
`europe-west4-a`, in 10.3 hours (about 38 min per step). The training worked mechanically, but the reward
collapsed after step 1 (val reward 0.379 → 0.0). The cause of the collapse is still open (section 10).

**Software stack on TPU.**

| Layer | Used |
|---|---|
| PyTorch backend for TPU | TorchTPU, `torch_tpu==0.2.0` (native PyTorch device `"tpu"`, c10d backend `tpu_dist`), with torch 2.14.1 (CPU build) |
| Runtime image | `v2-alpha-tpuv6e` |
| Transformers | 5.16.1 |
| Rollout engine | Own HTTP server with Hugging Face `generate` on TorchTPU (vLLM-TPU was tried; see 7.1) |

---

## 2. Summary of all blockers

| # | Blocker | Category | Effect | Solution | Section |
|---|---|---|---|---|---|
| 1 | `--scopes cloud-platform` rejected | Cloud | VM create failed | Use the full scope URL | 3.1 |
| 2 | gcloud login expires about daily | Cloud | Laptop scripts stopped | User re-runs `gcloud auth login` | 3.1 |
| 3 | Data only in AWS S3, keys must not leave the laptop | Data | No data source for VMs | Stage once into a GCS bucket, keys used as env vars only | 3.2 |
| 4 | Bucket 403 from the VM | Cloud | VM could not read data or write logs | Grant the VM service account `storage.objectAdmin` on the bucket only | 3.2 |
| 5 | No spot v6e capacity | Capacity | Day 1 lost | Polling script across all v6e zones | 3.3 |
| 6 | A create "error" that created a VM | Capacity | Risk of paying for unseen VMs | Check what really exists after every create | 3.3 |
| 7 | Spot VMs preempted after 15–60 min | Capacity | Runs and probes lost | Watcher mirrors state to GCS; one-VM layout | 3.3, 3.4 |
| 8 | Two VMs needed at the same time | Capacity | Rarely both available | Move everything onto one v6e-8 | 3.4 |
| 9 | SkyRL's install pulls CUDA packages | Install | Install fails or breaks torch | Install SkyRL `--no-deps`, add only non-CUDA deps, pin torch | 4.1 |
| 10 | uv index conflict between PyPI and the PyTorch CPU index | Install | Install fails | `--index-strategy unsafe-best-match` | 4.2 |
| 11 | Re-run of setup failed on an existing venv | Install | Setup could not be repeated | `uv venv --clear` | 4.2 |
| 12 | Qwen3.5 processor needs `torchvision` | Install | Processor load failed | Install the CPU `torchvision` that matches torch | 4.2 |
| 13 | `torch.cuda.*` calls everywhere | SkyRL | Crash on the first device call | Device shim (`SKYRL_DEVICE=tpu`) | 5.1 |
| 14 | NCCL process groups | SkyRL | Workers cannot connect | `cpu:gloo,tpu:tpu_dist` | 5.2 |
| 15 | Weight sync only via NCCL / CUDA IPC to vLLM | SkyRL | Engine never gets new weights | New `reload` weight-sync backend | 5.3 |
| 16 | Image features need vLLM's own codec | SkyRL | Trainer cannot rebuild image inputs | Simple safetensors format, decoded without vLLM | 5.4 |
| 17 | Port 8470 already in use | TorchTPU | Process group did not start | Move TorchTPU ports to 9470+ | 6.1 |
| 18 | FSDP2 `set_data` incompatible tensor type | TorchTPU | Crash when sharding the model | `torch.utils.swap_tensors` patch | 6.2 |
| 19 | 2-chip process groups fail `START_SESSION` | TorchTPU | Reference model on 2 chips impossible | Reference model on 1 chip | 6.3 |
| 20 | In-place M-RoPE write aborts (`inversion.cc:112`) | TorchTPU | Batched generate and forward abort | Rewrite with `torch.where` (identical values) | 6.4 |
| 21 | Vision RoPE buffer left on CPU | TorchTPU | `tpu:0 and cpu` device mismatch in the ref worker | Patch applied in every worker | 6.5 |
| 22 | Every torch process opens a chip | TorchTPU | Data loader grabbed chip 1 | `TORCH_DEVICE_BACKEND_AUTOLOAD=0`, explicit load only in workers | 6.6 |
| 23 | vLLM-TPU: about 80 s per frame | Engine | Rollout far too slow | Own server with HF `generate` on TorchTPU | 7.1 |
| 24 | Many distinct shapes → many TPU compiles | Engine | Slow first steps | Fixed batch size and length buckets | 7.2 |
| 25 | Old vLLM process still held chip 6 | Engine | New engine could not open the chip | Stop stale processes before a restart | 7.3 |
| 26 | Reference model out of HBM on 1 chip | Memory | Run 2 failed | Reference model with CPU offload | 8.1 |
| 27 | Optimizer step out of HBM | Memory | Run 3 failed | Policy CPU offload (as on GPU) | 8.2 |
| 28 | Reward 0.292 → 0.0 after step 1 | Result | Result not usable | Not solved; gradient check prepared | 10 |

---

## 3. Cloud access, data and capacity

### 3.1 gcloud details

- **Scopes.** `gcloud compute tpus tpu-vm create --scopes cloud-platform` was rejected. TPU VM creation needs the
  full URL: `--scopes https://www.googleapis.com/auth/cloud-platform`.
- **Login expiry.** The laptop's gcloud login (account `purchase@holbox.ai`) expired roughly once a day. All laptop
  scripts (polling, sync, status) then failed with auth errors. The fix is manual: the user runs `gcloud auth login`.
  Work on the VM was not affected.

### 3.2 Getting the data to the TPU VMs

**Problem.** The checkpoint (about 10 GB) and the parquet files were only in S3 (`s3://stanford-train-data/...`).
The AWS keys were in a CSV file on the laptop, and they must never be copied to a VM or into the repository.

**Solution.**

1. A helper reads the keys from the CSV into environment variables for one command only. Nothing is written to disk
   and nothing is printed.
2. The data was downloaded once to the laptop, then uploaded to `gs://tpu-rl-run-cholec/cholec/` (region us-east5).
3. VMs read only from GCS. Each VM checks the parquet SHA-256 hashes against the copy the GPU run used.

**Follow-up problem.** The first VM got HTTP 403 on the bucket. The default compute service account
(`189277334411-compute@developer.gserviceaccount.com`) had no access to the new bucket.
**Fix:** grant that account `roles/storage.objectAdmin` **on this bucket only**, not on the project.

### 3.3 TPU capacity and preemption

**Problem 1: no capacity.** On 2026-10-23, `tpu-vm create --spot` for v6e-8 failed in 8 zones (us-east5-a/b/c,
us-central1-a/b, us-south1-a, us-east1-d, asia-northeast1-b) with "There is no more capacity in the zone" or
"Insufficient capacity". On-demand v6e was not an option: no capacity, and in us-east1-d the project has no
permission for on-demand v6e.

**Solution.** A polling script tries every v6e zone in a loop (one create per zone, then a 2-minute pause, for up to
24 hours) and stops at the first VM in the `READY` state.

**Problem 2: a failed create that was not a failure.** In europe-west4-a a create call returned an error, but the VM
was in fact being created. If the poller had moved to the next zone, two v6e-8 VMs could have billed at once.
**Fix:** after every create call, the poller reads the real state (`describe`), waits while the state is
`CREATING`, and deletes VMs in `FAILED` / `PREEMPTED` / `TERMINATED` state so they cannot bill.

**Problem 3: preemption.** Granted spot VMs were often reclaimed after 15–60 minutes. A preempted TPU VM cannot
restart; it must be deleted and created again.
**Mitigations:**

- A watcher on the VM writes the run state and metrics to GCS every 2 minutes, so a preemption loses no results.
- Each pipeline stage logs to GCS, so it is always known where the work stopped.
- The setup is idempotent: a new VM repeats setup, data and probes without manual steps.

### 3.4 From two VMs to one

**Problem.** The first layout used two VMs: a v6e-8 for training (policy on chips 0–3, reference model on chips 4–7)
and a v6e-1 for the rollout engine. Two scarce spot VMs in the same zone at the same time were rarely available.

**Solution.** One v6e-8 for everything:

| Chips | Use |
|---|---|
| 0–3 | Policy (FSDP2, 4 ranks) |
| 4 | Reference model (1 rank; see 6.3 for why not 2) |
| 6 | Rollout engine |
| 5, 7 | Free |

The weights now go from the trainer to the engine through `/dev/shm` (host RAM, about 1.4 TB on a v6e-8) instead
of over the network. A switch (`TOPOLOGY=single`) selects this layout.

### 3.5 A hung SSH session

During the gradient check (section 10), the SSH command that started the check on the VM hung for about 3 hours,
while the VM stayed idle and billing. **Fix:** start remote work in the background (`nohup setsid ... &`) and kill
the local SSH process after 2 minutes. The remote job does not depend on it.

---

## 4. Software installation

### 4.1 CUDA packages in SkyRL's dependencies

**Problem.** SkyRL's normal install (`uv sync --extra fsdp`) pulls vLLM 0.31, flash-attn, flashinfer, nixl and a
CUDA build of torch. None of these run on TPU, and a CUDA torch would replace the torch that TorchTPU needs.

**Solution.** The setup script:

1. installs `torch_tpu==0.2.0` first, with the PyTorch CPU index, which brings torch 2.14.1 (CPU build);
2. writes a constraints file that pins that torch version;
3. reads SkyRL's own `pyproject.toml`, removes the CUDA-only packages, and installs the rest under the constraint;
4. installs SkyRL itself with `--no-deps`;
5. checks the result: a tensor on the TPU, and an import of each SkyRL module that the trainer loads.

### 4.2 Smaller install problems

| Problem | Cause | Fix |
|---|---|---|
| Resolver could not combine PyPI and the PyTorch CPU index | uv by default takes a package from the first index that has it | `--index-strategy unsafe-best-match` |
| Second setup run failed | The venv already existed | `uv venv --clear` |
| Qwen3.5 `AutoProcessor` failed to load | It also loads a video processor, which needs `torchvision` | Install the CPU `torchvision` that matches torch 2.14.1 |

---

## 5. SkyRL itself: written for CUDA only

These blockers come from SkyRL's code, not from TPU hardware. They are why SkyRL needs a patch
(`skyrl_patch/skyrl-tpu-on-2c54cefc.patch`) to run on TPU at all.

### 5.1 Direct `torch.cuda` calls

**Problem.** The FSDP workers call `torch.cuda.set_device`, `torch.cuda.synchronize`, `torch.cuda.empty_cache`,
`torch.cuda.mem_get_info`, `torch.cuda.manual_seed_all`, CUDA RNG state functions and others directly. A TPU VM has
no CUDA device, so the first call raises an error.

**Solution: a device shim.** A new module in SkyRL is the single place that knows the accelerator. One environment
variable selects it:

| `SKYRL_DEVICE` | Stack | Process-group backend |
|---|---|---|
| unset or `cuda` | standard PyTorch (unchanged GPU behavior) | `cpu:gloo,cuda:nccl` |
| `tpu` | TorchTPU | `cpu:gloo,tpu:tpu_dist` |

Every CUDA call in the workers now goes through the shim: `synchronize()` calls `torch.accelerator.synchronize()`
on TPU, `empty_cache()` calls `torch.accelerator.empty_cache()`, the device is `torch.device("tpu")`, seeding uses
`torch.manual_seed` (which seeds every registered device), and CUDA-only tools (memory history, IPC, NUMA by PCI bus)
are skipped. With the variable unset, every function does exactly what the original CUDA call did.

### 5.2 NCCL process groups

**Problem.** Workers called `init_process_group` with NCCL.
**Solution.** The shim returns `cpu:gloo,tpu:tpu_dist`: TorchTPU's backend for device tensors, and gloo for the CPU
tensors that FSDP CPU offload uses.

### 5.3 Weight sync to the rollout engine

**Problem.** After each training step SkyRL sends the new weights to vLLM with NCCL broadcast or CUDA IPC. Neither
exists on TPU, so the rollout engine would keep the starting weights forever.

**Solution: a new `reload` weight-sync backend.**

1. The trainer gathers the full bf16 weights and writes them as safetensors to `/dev/shm` (single-VM layout) or
   uploads them over HTTP (two-VM layout).
2. A supervisor process that owns the rollout server stops the server and starts it again on the new checkpoint.
   It copies the tokenizer, processor and config from the base checkpoint, so templates do not change.
3. The supervisor answers once the new server is healthy. Measured: **24–25 s per reload** on the real model.
4. The rollout server reports the weight version, and the trainer's log-probs and the rollout log-probs agree
   (mean absolute difference about 0.07), which shows that the weights arrived.

### 5.4 Image features without vLLM

**Problem.** SkyRL's VLM generator gets the image features from the engine's render route, and decodes them later
with vLLM's own serialization. Without vLLM installed, the trainer cannot rebuild `pixel_values` and
`image_grid_thw` for the training forward pass.
**Solution.** The rollout server encodes the features as `"st:" + base64(safetensors{pixel_values, image_grid_thw})`.
The patched decoder reads this format without vLLM.

---

## 6. TorchTPU gaps

TorchTPU was version 0.2 and was open-sourced in October 2026. These problems appeared one at a time on the
hardware, because each one only shows after the previous one is fixed.

### 6.1 Port 8470

**Problem.** The first multi-process TorchTPU start failed. Port 8470 on a Cloud TPU VM is always held by the TPU
VM's own agent process (`main`), and the first slice-builder address used it.
**Fix.** All TorchTPU slice-builder ports start at 9470, with separate ranges for each process group.

### 6.2 FSDP2 cannot move parameters to TPU

**Problem.** When FSDP2 shards the model, it moves each CPU parameter with `tensor.data = tensor.to(device)`.
TorchTPU tensors are not shallow-copy compatible with CPU tensors, so this fails with:

```
Attempted to call `variable.set_data(tensor)`, but `variable` and `tensor` have incompatible tensor type.
```

**Fix.** A patch for FSDP2's `_move_states_to_device`. For each tensor whose types are not compatible, it uses
`torch.utils.swap_tensors(tensor, moved)`, the same path FSDP2 already uses for wrapper subclasses. Compatible
tensors (all of them on CUDA) take the original path.

### 6.3 Two-chip process groups do not start

**Problem.** With the reference model on 2 chips, its process group failed with `START_SESSION failed`.
Groups of 1 chip and 4 chips worked. This was checked with a dedicated probe (P0).
**Fix.** The reference model runs on 1 chip (chip 4). With only one chip it needed CPU offload (see 8.1).

### 6.4 In-place M-RoPE write aborts

**Problem.** Batched forward passes and batched `generate` aborted inside TorchTPU with `inversion.cc:112`.
The cause is Qwen3.5's `apply_interleaved_mrope`. It writes into `freqs[0]`, which is a view of `freqs`, from other
slices of the same tensor. TorchTPU 0.2 aborts on in-place operations between overlapping views. A `clone()` alone
did not help, because the strided slice assignment itself is the problem.
**Fix.** A replacement with the same values and no in-place writes:

```python
col = torch.arange(d, device=freqs.device)
use_h = (col % 3 == 1) & (col < mrope_section[1] * 3)
use_w = (col % 3 == 2) & (col < mrope_section[2] * 3)
return torch.where(use_h, freqs[1], torch.where(use_w, freqs[2], freqs[0]))
```

It is applied in the trainer workers and in the rollout engine.

### 6.5 Vision RoPE buffer on the wrong device

**Problem.** Run 2 failed in the reference worker's forward pass with a device mismatch between `tpu:0` and `cpu`.
`Qwen3_5VisionRotaryEmbedding.inv_freq` is a non-persistent buffer. FSDP can leave it on the CPU, and the forward
pass multiplies it with position ids on the TPU. The GPU run had the same fix, but only in its entrypoint, which runs
in the driver and not in the Ray workers that run the model.
**Fix.** The patch moves `inv_freq` to the device of the position ids in every worker process.

### 6.6 Every process that imports torch opens a TPU chip

**Problem.** Run 1 failed at start. TorchTPU registers itself through PyTorch's backend autoload, so every process
that imports torch tries to open a chip, including SkyRL's DataLoader worker processes. One of them took chip 1
before the policy worker could (`/dev/vfio/1 ... already opened`).

**Fix.**

- `TORCH_DEVICE_BACKEND_AUTOLOAD=0` for the whole run, so no process opens a chip by accident.
- Only the processes that must own a chip load TorchTPU explicitly, through `torch_tpu._loader.load()`.

**A mistake on the way.** The first version checked `hasattr(torch, "tpu")` to see whether TorchTPU was loaded. A
plain `import torch_tpu` creates `torch.tpu` but leaves the backend unnamed (`privateuseone`), so the check was
wrong. The correct check is `torch._C._get_privateuse1_backend_name() == "tpu"`.

### 6.7 Per-process TPU topology

Each worker needs its own TorchTPU and libtpu environment before the device starts: `TORCH_TPU_TOPOLOGY`,
`TORCH_TPU_SLICEBUILDER_ADDRESSES`, `TPU_VISIBLE_DEVICES`, plus libtpu's `TPU_VISIBLE_CHIPS` and
`TPU_PROCESS_*` values. A helper computes one JSON object per process group (shared values plus one entry per rank).
The patched SkyRL worker applies it at start: topology `1,1,1` for 1 chip, `2,2,1` for 4 chips, and so on.

---

## 7. The rollout engine

### 7.1 vLLM-TPU is too slow for this model

**Problem.** SkyRL's GPU engine (vLLM 0.31 on CUDA) cannot run on TPU. vLLM's TPU build (vLLM-TPU 0.32,
`tpu-inference`) did run Qwen3.5 with images, but its vision tower for Qwen is experimental: **about 80 seconds per
frame.** One training step needs 128 rollouts, so this was unusable.

**Solution: an own rollout server.**

- It serves the HTTP routes that SkyRL's external-server mode calls (`/v1/chat/completions/render`,
  `/inference/v1/generate`, `/pause`, `/resume`, `/get_world_size`, ...). SkyRL needs no engine of its own.
- The engine behind it is Hugging Face `generate` on TorchTPU, with request micro-batching.
- Probe P1 checked it on the 32 real val frames: greedy val reward **0.379** (GPU run: 0.329). One step's 128
  rollouts took about **80 s** in total.

### 7.2 TPU compilation and shapes

**Problem.** TPU compiles a program per tensor shape. Variable batch sizes and prompt lengths would cause a new
compile for most requests.
**Solution.** The engine pads every batch to a fixed size (16) and every prompt to a multiple of 128 tokens, so
only a few shapes exist. Even so, step 1 of training spent 12–18 min on old-policy and reference log-probs, mostly
compiling. Later steps reused the compiled programs.

### 7.3 A stale engine process held the chip

**Problem.** After the switch from vLLM-TPU to the HF engine, the new server could not open chip 6: the old vLLM
process still held it.
**Fix.** Stop all old engine processes, then wait a few seconds (libtpu releases a chip asynchronously) before a
restart. The supervisor also waits 5 s after each stop.

---

## 8. Memory

A v6e chip has 32 GB of HBM. The GPU run used A10G GPUs with 24 GB each, but it ran with CPU offload.

### 8.1 Reference model on one chip

**Problem.** In run 2, the reference model on 1 chip ran out of contiguous HBM in step 1. It held fp32 weights and
the activations of a forward pass over 128 samples of about 2,300 tokens each.
**Fix.** FSDP CPU offload for the reference model. Its weights stay in host RAM and stream to the chip layer by
layer. The math is the same.

### 8.2 Optimizer step

**Problem.** In run 3, the policy update itself worked (187 s for the backward passes). The optimizer step then
failed with a TPU out-of-memory error. AdamW state for a 4B model in fp32 does not fit next to the weights on 4
chips.
**Fix.** Policy `cpu_offload=true`. The GPU run used the same setting, so the TPU run now matches the GPU
configuration more closely.

---

## 9. The four training runs

| Run | Date | Ended at | Cause | Fix before the next run |
|---|---|---|---|---|
| 1 | 2026-10-24 | Start of training | A DataLoader worker opened TPU chip 1 (6.6) | Autoload off; explicit TorchTPU load in workers |
| 2 | 2026-10-24 | Reference forward, step 1 | Vision RoPE device mismatch (6.5), then reference HBM (8.1) | Patch in every worker; reference CPU offload |
| 3 | 2026-10-24 | Optimizer step, step 1 | Out of HBM (8.2) | Policy CPU offload |
| **4** | 2026-10-24/25 | **Completed, 16 of 16 steps** | — | — |

Run 4 timing: eval before training 63 s; rollouts about 80 s per step; weight reload 24–25 s; one full step
26–44 min (mean about 38 min); 10.3 hours in total.

---

## 10. Not solved: the reward collapse

### 10.1 What happened

| Step | Train reward | KL to the reference | Gradient norm | Mean reply length (tokens) |
|---|---|---|---|---|
| 0 (val) | 0.379 | — | — | — |
| 1 | 0.292 | 1.345 | 7.27 | 10.3 |
| 2 | 0.000 | 4.528 | 7.06 | 6.1 |
| 3 | 0.000 | 2.291 | 1.05 | 7.3 |
| 4 | 0.000 | 3.506 | 5.27 | 5.2 |
| 5 | 0.000 | 5.955 | 4.05 | 12.8 |
| 6 | 0.000 | 4.317 | 14.23 | 256.0 |
| 7–16 | 0.000 | about 4 | 0.0 | 256.0 |

Final val reward: 0.0. The GPU run with the same recipe trained for 13 steps (train reward up to 0.306) and then
also collapsed, at step 14.

### 10.2 Explanation so far

1. **Step 1 moved the policy far.** 8 optimizer updates in one step gave a KL of 1.3 to the reference model.
2. **The reward is all or nothing on format.** A reply that does not parse as `instrument-verb-target` triplets
   scores 0. From step 2 on, all 128 replies scored 0.
3. **GRPO then has no signal.** With equal rewards in every group, every advantage is 0. Only the KL term (0.04)
   remains.
4. **The model degenerated.** Replies grew to the 256-token limit from step 6, and gradients went to 0 from step 7.
5. **The weight sync is not the cause.** Trainer and rollout log-probs agree, so the engine served the trained
   weights.

**Open question.** In steps 2–6, the only gradient left was the KL term, which should pull the model back toward the
reference. KL went up instead. That points to a possible TPU gradient error, but it is not proven.

### 10.3 The gradient check (P2)

A probe compares, on one real CholecT50 example, the gradients and the first AdamW step on TPU with the same
computation on CPU:

- A0 and A: plain forward and backward, TPU vs CPU;
- B: FSDP2 with CPU offload and bf16 mixed precision, as in training;
- C: gradient clipping and AdamW, checked against the analytic first step `-lr*g/(|g|+eps) - lr*wd*p`.

It needs about 40 minutes of TPU time. Five attempts on spot v6e-1 did not finish:

| Attempt | Result | Cause | Fixed |
|---|---|---|---|
| 1 | Setup failed in seconds | Missing log folder on the new VM | Yes |
| 2 | Failed at start | 1-chip TorchTPU topology variables missing | Yes |
| 3 | VM preempted | Spot reclaim | — |
| 4 | Out of memory; VM idle about 3 h | Gradient copy kept on the TPU; the hung SSH session (3.5) | Yes: copy to CPU first; background start |
| 5 | Out of memory, then preempted | Gradient checkpointing missing (training uses it) | Yes |

The probe is ready. It has no TPU result yet.

---

## 11. Lessons

1. **Most blockers were integration gaps, not TPU limits.** SkyRL's code, its dependency list and its weight sync
   assume CUDA. Each assumption needed its own replacement.
2. **TorchTPU 0.2 is usable for FSDP2 training with a few patches.** Swap instead of `.data =`, no in-place view
   writes, explicit device loading, 1- or 4-chip groups, and ports away from 8470.
3. **Problems appear one at a time.** Each fix exposes the next blocker, which is why short probes (P0, P1) before
   the full run saved TPU hours.
4. **Spot TPU capacity was the largest time cost.** One VM instead of two, results mirrored to GCS, and idempotent
   setup made the work survive preemptions.
5. **A complete run is not a correct run.** Mechanically the TPU run worked for 16 steps; whether its gradients are
   correct is still open (section 10).
