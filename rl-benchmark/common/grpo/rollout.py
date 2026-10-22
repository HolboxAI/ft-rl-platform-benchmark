"""Static-shape rollout and log-prob computation.

Why not vLLM / SGLang? Miles, verl and SkyRL all generate rollouts with CUDA
inference engines. Neither runs as-is on Trainium or on TorchTPU, so this
benchmark generates with the policy model itself, in plain PyTorch.

Why static shapes? On XLA backends (torch-neuronx, torch_xla) every new tensor
shape is a new graph compile, which can take minutes on Neuron. A naive
token-by-token loop with a growing sequence would compile once per token. The
`static` generator instead:

  1. left-pads all prompts to the same length P, so every prompt ends at P-1;
  2. allocates one fixed [B, P+T] buffer for the whole generation;
  3. at step t, runs the model over the full buffer (causal attention means the
     not-yet-written positions cannot affect position P+t-1) and reads logits at
     a position held in a *device tensor*, so the index is graph data, not a
     constant baked into the graph.

The result is one compiled graph reused for all T steps and all batches. It is
O(T * (P+T)) compute with no KV cache — a deliberate trade of FLOPs for
compile-friendliness. `hf` (transformers.generate with a static cache) is
provided as the faster alternative to try once the static path works.
"""
from __future__ import annotations

import torch

MODEL_INPUT_KEYS = ("input_ids", "attention_mask")


def _extra(batch: dict, extend_by: int = 0) -> dict:
    """Model inputs other than input_ids / attention_mask (pixel_values, image_grid_thw, ...).

    Per-token extras shaped [B, P] (Qwen3.5's `mm_token_type_ids`: 0 = text, 1 = image) must
    cover the full [B, P + T] sequence the sampler / log-prob pass feeds the model, so they are
    right-padded with zeros (generated tokens are text).
    """
    ids = batch["input_ids"]
    out = {}
    for k, v in batch.items():
        if k in MODEL_INPUT_KEYS:
            continue
        if (extend_by and isinstance(v, torch.Tensor) and v.dim() == 2
                and v.shape == ids.shape and not v.dtype.is_floating_point):
            v = torch.cat([v, torch.zeros((v.shape[0], extend_by), dtype=v.dtype, device=v.device)], dim=1)
        out[k] = v
    return out


def _position_ids(mask: torch.Tensor) -> torch.Tensor:
    # Left padding: real tokens start at 0 regardless of how much padding precedes them.
    return (mask.long().cumsum(-1) - 1).clamp(min=0)


@torch.no_grad()
def generate_static(model, batch, max_new_tokens, temperature, pad_id, backend,
                    greedy=False, pass_position_ids=True, eos_ids=None, early_stop_every=8):
    ids, mask = batch["input_ids"], batch["attention_mask"]
    B, P = ids.shape
    T = max_new_tokens
    dev = ids.device
    buf = torch.cat([ids, torch.full((B, T), pad_id, dtype=ids.dtype, device=dev)], dim=1)
    full_mask = torch.cat([mask, torch.ones((B, T), dtype=mask.dtype, device=dev)], dim=1)
    kw = dict(attention_mask=full_mask, use_cache=False, **_extra(batch, extend_by=T))
    if pass_position_ids:
        kw["position_ids"] = _position_ids(full_mask)
    eos = torch.tensor(sorted(eos_ids or []), device=dev, dtype=ids.dtype)
    # Track EOS from the sampled tokens, not the buffer: the buffer is pre-filled
    # with pad, and for Qwen pad (<|endoftext|>) is itself an EOS id.
    done = torch.zeros(B, dtype=torch.bool, device=dev)

    read_idx = torch.tensor([P - 1], device=dev)  # position whose logits predict the next token
    for t in range(T):
        logits = model(input_ids=buf, logits_to_keep=read_idx, **kw).logits[:, -1, :].float()
        if greedy:
            tok = logits.argmax(-1)
        else:
            # Gumbel-max sampling == multinomial over softmax(logits / T), using
            # only rand/log/argmax, which every backend supports.
            u = torch.rand_like(logits).clamp_(1e-10, 1.0 - 1e-7)
            tok = (logits / temperature - torch.log(-torch.log(u))).argmax(-1)
        buf.index_copy_(1, read_idx + 1, tok.unsqueeze(1).to(buf.dtype))
        read_idx = read_idx + 1
        if eos.numel():
            done = done | torch.isin(tok.to(eos.dtype), eos)
        backend.mark_step()
        # Early exit once every row has produced EOS. This forces a host sync,
        # so only check every few tokens.
        if eos.numel() and early_stop_every and (t + 1) % early_stop_every == 0 and t + 1 < T:
            if bool(done.all().item()):
                break
    return buf[:, P:]


@torch.no_grad()
def generate_hf(model, batch, max_new_tokens, temperature, pad_id, eos_ids, greedy=False):
    P = batch["input_ids"].shape[1]
    out = model.generate(
        **batch,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=temperature if not greedy else None,
        top_p=1.0,
        top_k=0,  # Qwen's generation_config sets top_k/top_p; override to plain sampling like Miles/verl
        pad_token_id=pad_id,
        eos_token_id=list(eos_ids),
        cache_implementation="static",
    )
    resp = out[:, P:]
    if resp.shape[1] < max_new_tokens:
        pad = torch.full((resp.shape[0], max_new_tokens - resp.shape[1]), pad_id,
                         dtype=resp.dtype, device=resp.device)
        resp = torch.cat([resp, pad], dim=1)
    return resp


def postprocess(resp_ids: torch.Tensor, eos_ids, max_new_tokens: int):
    """CPU-side: response mask (tokens up to and including the first EOS),
    lengths, and truncation flags."""
    resp = resp_ids.cpu()
    T = max_new_tokens
    resp = resp[:, :T]
    is_eos = torch.isin(resp, torch.tensor(sorted(eos_ids), dtype=resp.dtype))
    has_eos = is_eos.any(dim=1)
    first = torch.where(has_eos, is_eos.float().argmax(dim=1), torch.full_like(has_eos, T, dtype=torch.long))
    ar = torch.arange(T)
    mask = ar[None, :] <= first[:, None]
    # Rows that stopped early (generate_static break / hf) are padded past EOS; mask handles it.
    lengths = mask.sum(dim=1)
    return resp, mask, lengths, ~has_eos, first


def response_logprobs(model, batch, resp_ids, temperature, pass_position_ids=True):
    """Per-token log-probs of `resp_ids` under `model`, given the prompt batch.

    Logits are divided by the sampling temperature, as verl and Miles do for the
    actor log-probs. Only the last T+1 positions are projected through the LM
    head (`logits_to_keep`), which keeps the [B, T, vocab] tensor small.
    """
    ids, mask = batch["input_ids"], batch["attention_mask"]
    T = resp_ids.shape[1]
    full_ids = torch.cat([ids, resp_ids.to(ids.dtype)], dim=1)
    full_mask = torch.cat([mask, torch.ones_like(resp_ids, dtype=mask.dtype)], dim=1)
    kw = dict(attention_mask=full_mask, use_cache=False, **_extra(batch, extend_by=T))
    if pass_position_ids:
        kw["position_ids"] = _position_ids(full_mask)
    logits = model(input_ids=full_ids, logits_to_keep=T + 1, **kw).logits[:, :-1, :].float()
    logits = logits / temperature
    target = resp_ids.long().unsqueeze(-1)
    return logits.gather(-1, target).squeeze(-1) - torch.logsumexp(logits, dim=-1)
