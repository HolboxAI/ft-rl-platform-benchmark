# HANDOFF — NKI_Experiments (Trainium NKI vs g6e GRPO benchmark)

Written 2026-11-20 ~15:05 UTC for a Claude Code session on a **different machine** that picks up this work.
Read all of this before touching any AWS resource.

## 1. Goal

Compare Trainium (trn1 / trn2 / trn3, with and without NKI kernels) against a g6e GPU
(NVIDIA L40S) on one basic RL job:

- **Job:** stock TRL `GRPOTrainer` (TRL 0.29.1, transformers 5.x) training Qwen2.5-Coder-0.5B-Instruct.
- **Data:** 32 DS-1000 problems, 10 steps.
- **Recorded:** every RL phase (generation, reward, reference-model log-probs, policy forward, backward, optimizer step) goes to CSVs written by separate scripts, for plotting later.

## 2. Where everything is

| What | Where |
|---|---|
| Code (canonical) | GitHub `HolboxAI/ft-rl-platform-benchmark`, branch **`nki-experiments`**, folder `NKI_Experiments/` |
| Code tarball (what EC2 instances download) | `s3://holbox-rl-benchmark-trn2/debjyoti/NKI_Experiments.tar.gz` (bucket region **sa-east-1**) |
| DS-1000 data | `NKI_Experiments/data/ds1000/test.jsonl`. Not in git; the tarball has it. Re-download command is in `NKI_Experiments/trl_grpo_ds1000_bench/README.md`. |
| g6e results (done) | git: `NKI_Experiments/trl_grpo_ds1000_bench/results/` (force-added despite `results/*` in `.gitignore`) |
| trn1 results (in progress) | `s3://holbox-rl-benchmark-trn2/debjyoti/results/trn1.2xlarge/<stage>.tar.gz` |
| trn2.48xlarge results (in progress) | `s3://holbox-rl-benchmark-trn2/debjyoti/results/trn2.48xlarge/<stage>.tar.gz` |

Stage tarballs, uploaded as each stage finishes:
- `0_smoke_2steps` (trn1 only)
- `1_nki-none_sync-phase`
- `2_nki-rmsnorm_sync-phase`
- `3_nki-none_sync-step`
- `4_nki-rmsnorm_sync-step`
- `5_kernels_and_comparison`
- `6_final_all_results`

Each tarball also contains `auto_run.log`.

**Main entry points** (in `NKI_Experiments/trl_grpo_ds1000_bench/`):

| File | Role |
|---|---|
| `train/grpo_ds1000_bench.py` | Training run |
| `train/instrumented_trainer.py` | Phase hooks |
| `bench/xla_generate.py` | Neuron fixed-shape generation |
| `bench/platform_utils.py` | XLA workarounds |
| `bench/static_shapes.py` | Fixed-shape padding |
| `metrics/*.py` | CSV store, summary, comparison |
| `kernels/kernel_microbench.py` | Per-op microbenchmark |
| `scripts/run_g6e.sh` | GPU run |
| `scripts/run_trainium.sh` | Trainium run |
| `scripts/auto_run_unattended.sh` | Unattended Trainium pipeline: uploads to S3, then shuts the instance down |

## 3. AWS access

- **SSO profile:** `trn-sso`, account `992382417943`, Identity Center region `ap-south-1`.
  - Log in with `aws sso login --profile trn-sso`, then use `AWS_PROFILE=trn-sso`.
  - The new machine needs this profile configured in `~/.aws/config`. The user has to do the browser login.
- **Blocked in this account:** IAM listing, EC2 Instance Connect, SSM and Service Quotas. Instances have **no IAM role**.
  - Results therefore leave instances through **presigned S3 PUT URLs** generated on the laptop (boto3 `generate_presigned_url("put_object", ...)`).
  - A presigned URL stops working when the SSO session that signed it expires. The URLs in use now were signed with a session that expires **2026-11-20 17:38 UTC**, so any upload after that returns 403.
- **SSH key:** instances accept `~/.ssh/nki-bench-ap-south-2`. Its private key exists **only on the old Mac**, so a new machine cannot SSH into the current instances.
  - To get in from a new machine, generate a new key and inject it via user-data. For an on-demand instance: stop, set user-data, start. A one-time Spot instance cannot be stopped.
- **Security groups:** allow port 22 only from single `/32` IPs (a wider CIDR was refused before). Add your own IP as a `/32`.
- **Shell gotcha:** zsh does not word-split `$VAR` flags, so wrap multi-flag AWS/SSH commands in `bash <<'EOF' ... EOF`. Inside such a heredoc, always use **`ssh -n`**: without it, ssh reads the rest of the heredoc from stdin and silently swallows the remaining commands.

## 4. Live resources at hand-off (check them first)

| Resource | Region / ID | State | Cost | Auto-stop? |
|---|---|---|---|---|
| trn1.2xlarge (on-demand) | us-east-1f, `i-03da7b445bb8f5bca`, IP 35.175.107.86 | running; 2-step smoke test, then 4 variants, kernels, uploads | ~$1.34/hr | yes: `sudo shutdown` at the end, or a 7 h cap (~21:40 UTC). Shutdown behaviour = **stop**, so EBS data is kept. |
| trn2.48xlarge (**Spot**, one-time) | us-east-2a, `i-060c224f58df3ad3a`, IP 13.184.186.126 | booting; user-data runs setup, then `auto_run_unattended.sh` (see §5) | ~$8.60/hr Spot | yes: end of run or the `MAX_HOURS` cap. Shutdown **terminates** it (one-time Spot): anything not uploaded is lost. AWS can also reclaim it at any time. |
| g6e.2xlarge | ap-south-2b, `i-032cdc20ed25716cc` | **stopped** (results already in git) | EBS only | n/a |
| trn2.3xlarge Capacity Block | sa-east-1b, `cr-0db37dabaca8d868b` | booked for 2026-11-21 11:30 → 2026-11-22 11:30 UTC, **$53.64 already charged, cannot be cancelled** | prepaid | expires on its own. Use it with `NKI_Experiments/infra/launch_trn2_capacity_block.sh` (the `infra/` folder is local-only and not in git; the command is ordinary `run-instances` with `--instance-market-options MarketType=capacity-block` plus `--capacity-reservation-specification`). |
| Snapshot | ap-south-2, `snap-01483d54650879277` | old code + data snapshot | small | n/a |
| Security groups | `nki-bench-ssh` in us-east-2 / ap-south-1 / ap-northeast-1 / sa-east-1; `sg-0670386d5de6c9952` (trn1, us-east-1); `sg-05703ac8515b76cd3` (g6e) | leftovers | free | n/a |

How to check status from any machine (once logged in to SSO):
```bash
export AWS_PROFILE=trn-sso
aws ec2 describe-instances --region us-east-1 --instance-ids i-03da7b445bb8f5bca --query 'Reservations[0].Instances[0].State.Name'
aws ec2 describe-instances --region us-east-2 --instance-ids i-060c224f58df3ad3a --query 'Reservations[0].Instances[0].State.Name'
aws s3 ls s3://holbox-rl-benchmark-trn2/debjyoti/results/ --recursive --region sa-east-1
aws ec2 get-console-output --region us-east-2 --instance-id i-060c224f58df3ad3a --latest --output text | grep 'TRN:'   # bootstrap milestones
```
If an instance is still running long after its cap, or idle, stop it (trn1) or terminate it (Spot). **Ask the user before deleting anything else.**

## 5. What the unattended runs do

`scripts/auto_run_unattended.sh` (trn1 runs an equivalent earlier copy, `~/auto_run.sh`, with `WAIT_FOR_SMOKE=1`):

1. Waits for `~/.setup_ok`, written by the bootstrap user-data after `scripts/setup_trainium.sh` succeeds.
2. Optionally waits for the running smoke test; if it failed, uploads the logs and shuts down.
3. Runs 4 variants, `nki-ops ∈ {none, rmsnorm}` × `sync-mode ∈ {phase, step}`, 10 steps each, with `run_variant` from `scripts/_common.sh`. Uploads a tarball after each.
4. Runs `kernels.kernel_microbench`, then `metrics.compare_runs`, and uploads.
5. Uploads everything, then `sudo shutdown -h now`.

The environment matches `run_trainium.sh`:
- `NEURON_RT_VISIBLE_CORES=0` (one NeuronCore, so results are comparable with one L40S).
- `NEURON_LOGICAL_NC_CONFIG=2` on trn2.
- Compile cache at `/var/tmp/neuron-compile-cache`.

## 6. Trainium problems found and fixed (all in git, commit `1d3fdbb` and later)

1. **NCC_EVRF035, "dot does not support 64-bit integer operands".**
   - Cause: the Neuron compiler rejects `torch.isin` (used by the EOS stopping criteria) and integer `cumsum` (position ids).
   - Fix: patched on XLA in `platform_utils.install_xla_workarounds()`. `isin` becomes a broadcast `==`; `cumsum` runs in fp32 and is cast back.
2. **HF `generate` is unusable on XLA.**
   - Cause: it grows `input_ids` and the mask every token, giving one multi-minute neuronx-cc compile per token. One of those graphs exceeded the 5M-instruction limit (NCC_EBVF030).
   - Fix: `bench/xla_generate.py`, a fixed-shape sampling loop: static cache length P+N, `mark_step` per token, 3 graphs in total. A 12-token test took 219 s cold and **1.1 s** warm.
3. **`NRT_EXEC_OOB` at runtime in the decode graph.**
   - Fix: removed every runtime-indexed scatter/gather. The cache write is a slice for the prompt and a one-hot `torch.where` per decode token, instead of transformers' `index_copy_`. The 4D additive masks are prebuilt and passed in, because transformers returns 4D masks as-is instead of building them with indexing.
4. **TRL post-processes completions with per-row boolean indexing on device** (dynamic shapes; the compile wrapper crashed).
   - Fix: `InstrumentedGRPOTrainer._generate_single_turn` overrides it on Neuron and trims at EOS on CPU.
5. **TRL 0.29 pads prompts itself**, bypassing the tokenizer patch.
   - Fix: `static_shapes.install_trl_static_padding` pads prompts to 512 and completions to 256 on Neuron. If a batch is longer, it falls back to padding to the longest, which is the open issue in §7.
6. **Hang in `accelerator.gather`** when 2 NeuronCores are visible to a single process (the XLA all-gather waits for a missing peer).
   - Fix: identity gather when `num_processes == 1`, plus `NEURON_RT_VISIBLE_CORES=0`.
7. **Kernel microbenchmark timed empty graphs** because lazy outputs were dropped before the sync.
   - Fix: outputs are kept alive. Attention shape s2048 was added, since NKI flash attention needs seqlen divisible by 2048.

GPU (g6e) code paths are unchanged by all of the above. `run_info.csv` has `generation_impl` (`hf_generate` vs `xla_static_loop`), so the comparison shows which generation path each run used.

## 7. Open issues / what to do next

1. **Prompts longer than 512 tokens.**
   - The trn1 smoke test showed a 1187-token sequence (prompt ≈ 931 + 256). Some chat-templated prompts exceed 512, so the padding falls back to longest-in-batch. That changes shape between steps and costs a recompile (minutes) whenever a new longest length appears.
   - Fix options: pad to a fixed bucket, e.g. `max_prompt_length` = 1024 for both platforms, or re-filter the subset by **chat-templated** length (`data/prepare_ds1000_subset.py`).
   - Check `compile_count_delta` in `step_metrics.csv` of the trn1/trn2 runs before deciding. If you change it, re-run g6e too so both platforms do identical work.
2. **Fetch and check results:** `aws s3 cp --recursive s3://holbox-rl-benchmark-trn2/debjyoti/results/ ./results_s3/`, untar, and compare with the g6e CSVs. In g6e, the median step was 11.6 s: reward 45%, generation 39% (~1,410 tok/s), ref log-prob 0.8–1.1 s, backward 0.5–0.7 s.
3. **Expired upload URLs:**
   - If uploads after 17:38 UTC returned 403, the trn1 data is still on its stopped EBS volume. Start the instance, add a new SSH key via user-data (§3), then `scp ~/NKI_Experiments/trl_grpo_ds1000_bench/results`.
   - Spot data that wasn't uploaded is gone; re-run.
4. **trn2.3xlarge Capacity Block (2026-11-21 11:30 UTC, sa-east-1b):** launch into it and run `scripts/auto_run_unattended.sh` with `PLATFORM=trn2.3xlarge`. Generate fresh presigned URLs right before launch.
5. **trn2 / trn3 capacity:**
   - On 2026-11-20, on-demand trn2.48xlarge had **no capacity** in any offering AZ (us-east-2 a/b/c, us-west-2 a/b/c/d, ap-south-1 a/b/c, ap-northeast-1 a/c/d, ap-south-2b).
   - trn2.3xlarge had none in sa-east-1 a/b/c or ap-south-2b, on-demand or Spot.
   - Spot trn2.48xlarge worked in us-east-2a.
   - trn3 has not been attempted.
6. Write the README section with the final comparison (`metrics/compare_runs.py` writes `results/comparison/*.csv`, baseline g6e).

## 8. Rules from the user (keep following them)

- **Git:** commit as **`debjyotictr-sudo <debjyotictr@holbox.ai>`**. **Never add a Claude `Co-Authored-By` trailer.**
- **Push target:** branch `nki-experiments` of `HolboxAI/ft-rl-platform-benchmark`.
- **Never commit:** `NKI_Experiments/data/ds1000/test.jsonl`, `NKI_Experiments/TRAINIUM_VS_G6E_BENCHMARK.md`, the top-level `NKI_Experiments/phase_timer.py`, `NKI_Experiments/infra/`, or any credentials or presigned URLs.
- **Resources:** ask before deleting resources. Stop idle instances. Keep SSH ingress to `/32`s.
- **Credentials:** don't write AWS keys to disk; use the SSO profile.
- **Reporting:** results go into CSVs (one script per kind of number); report numbers to the user plainly, with failures shown as failures.
