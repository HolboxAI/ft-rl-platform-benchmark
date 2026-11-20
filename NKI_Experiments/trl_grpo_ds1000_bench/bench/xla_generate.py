"""Fixed-shape sampling loop for Trainium (XLA), used in place of HF `model.generate`.

Why: HF `generate` appends one token per step to `input_ids` / `attention_mask`, so on XLA
every decode step has a new tensor shape -> a new graph -> a fresh multi-minute neuronx-cc
compile for each of the 256 tokens (and some of those graphs exceed the compiler's
instruction limit, NCC_EBVF030). This loop keeps every shape constant:

  * StaticCache of length prompt_len + max_new_tokens, written at `cache_position`
  * a full-length attention mask that is updated with a tensor comparison (no Python-int
    indexing, which would bake a constant into the graph)
  * cache_position / position_ids are device tensors whose *values* change, not shapes
  * one mark_step per token, so there are exactly two graphs: prefill and one decode step

Sampling matches what HF generate does with the benchmark config: EOS suppressed for all
tokens (min_new_tokens == max_new_tokens), temperature, optional top-k / top-p, then a
categorical sample (Gumbel-max, equivalent to torch.multinomial on the softmax).
Returns prompt + completion ids, shape [B, P + max_new_tokens], like `generate`.
"""

import torch


def _eos_ids(gc, extra=None):
    ids = []
    for src in (getattr(gc, "eos_token_id", None), extra):
        if src is None:
            continue
        ids.extend(src if isinstance(src, (list, tuple)) else [src])
    return sorted({int(i) for i in ids})


def _make_sampler(gc, vocab, device, suppress_eos_ids):
    do_sample = getattr(gc, "do_sample", True)
    temperature = float(getattr(gc, "temperature", None) or 1.0)
    top_k = int(getattr(gc, "top_k", None) or 0)
    top_p = float(getattr(gc, "top_p", None) or 1.0)
    bias = torch.zeros(vocab, dtype=torch.float32)
    for i in suppress_eos_ids:
        if i < vocab:
            bias[i] = float("-inf")
    bias = bias.to(device)

    def sample(logits):
        logits = logits.float() + bias
        if not do_sample:
            return logits.argmax(-1)
        logits = logits / temperature
        if 0 < top_k < vocab:
            kth = torch.topk(logits, top_k, dim=-1)[0][:, -1:]   # XLA returns a plain tuple/list
            logits = logits.masked_fill(logits < kth, float("-inf"))
        if top_p < 1.0:
            s, idx = logits.sort(dim=-1, descending=True)
            p = s.softmax(-1)
            drop = (p.cumsum(-1) - p) > top_p
            s = s.masked_fill(drop, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, s)
        u = torch.rand_like(logits).clamp_(min=1e-10, max=1.0)
        return (logits - torch.log(-torch.log(u))).argmax(-1)

    return sample


def _scatter_free_static_cache(config, max_cache_len):
    """StaticCache whose layers write K/V without index_copy_.

    StaticLayer.update writes with `index_copy_` at a runtime position (its own device-side
    `cumulative_length` counter). On Neuron that lowers to an indirect DGE scatter which fails at
    runtime with NRT_EXEC_OOB. Here the prefill write is a static slice (the cache is empty and
    the prompt length is fixed) and each decode write is a one-hot `torch.where` over the cache
    length -- same result, no scatter.
    """
    from transformers import StaticCache

    cache = StaticCache(config=config, max_cache_len=max_cache_len)

    def make_update(layer):
        def update(key_states, value_states, *args, **kwargs):
            if not layer.is_initialized:
                layer.lazy_initialization(key_states, value_states)
                layer._pos = torch.arange(layer.max_cache_len, dtype=torch.long).to(key_states.device)
                layer._len = torch.zeros((), dtype=torch.long).to(key_states.device)
                n = key_states.shape[-2]
                layer.keys = torch.cat([key_states, layer.keys[:, :, n:]], dim=2)
                layer.values = torch.cat([value_states, layer.values[:, :, n:]], dim=2)
                layer._len = layer._len + n
                layer.cumulative_length = layer._len   # keep get_seq_length() consistent
            else:
                if key_states.shape[-2] != 1:
                    raise ValueError("scatter-free cache supports one prefill then 1-token decode steps")
                hit = (layer._pos == layer._len).view(1, 1, -1, 1)
                layer.keys = torch.where(hit, key_states, layer.keys)
                layer.values = torch.where(hit, value_states, layer.values)
                layer._len = layer._len + 1
                layer.cumulative_length = layer._len
            return layer.keys, layer.values
        return update

    for layer in cache.layers:
        layer.update = make_update(layer)
    return cache


@torch.no_grad()
def static_generate(model, input_ids, attention_mask=None, generation_config=None,
                    max_new_tokens=None, min_new_tokens=None, step_fn=None, **_ignored):
    gc = generation_config if generation_config is not None else model.generation_config
    n_new = int(max_new_tokens or gc.max_new_tokens)
    min_new = int(min_new_tokens if min_new_tokens is not None else (getattr(gc, "min_new_tokens", None) or 0))
    dev = input_ids.device
    B, P = input_ids.shape
    L = P + n_new
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    attention_mask = attention_mask.to(torch.long)

    # EOS is only suppressed while fewer than min_new tokens exist; the benchmark sets
    # min_new == n_new (fixed-length generation), which is the case this loop supports.
    if 0 < min_new < n_new:
        raise ValueError("static_generate supports min_new_tokens == 0 or == max_new_tokens only")
    suppress = _eos_ids(gc, getattr(model.generation_config, "eos_token_id", None)) if min_new else []
    vocab = model.get_output_embeddings().weight.shape[0]
    sample = _make_sampler(gc, vocab, dev, suppress)
    if step_fn is None:
        import torch_xla.core.xla_model as xm
        step_fn = xm.mark_step

    cache = _scatter_free_static_cache(model.config, L)
    full_mask = torch.cat([attention_mask, torch.zeros(B, n_new, dtype=torch.long, device=dev)], dim=1)
    positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    arange_l = torch.arange(L, dtype=torch.long).to(dev)

    # Prebuilt additive 4D masks: transformers returns 4D masks as-is, which bypasses the
    # index-based mask construction in masking_utils (another runtime OOB source on Neuron).
    mdtype = next(model.parameters()).dtype
    neg = torch.finfo(mdtype).min
    q_idx = torch.arange(P, dtype=torch.long).to(dev)
    kv = arange_l.view(1, 1, 1, L)
    allowed = (full_mask.bool().view(B, 1, 1, L) & (kv <= q_idx.view(1, 1, P, 1))) | (kv == q_idx.view(1, 1, P, 1))
    prefill_mask = torch.where(allowed, torch.zeros((), dtype=mdtype).to(dev), torch.full((), neg, dtype=mdtype).to(dev))
    zero_m, neg_m = torch.zeros((), dtype=mdtype).to(dev), torch.full((), neg, dtype=mdtype).to(dev)

    out = model(input_ids=input_ids, attention_mask=prefill_mask, position_ids=positions,
                past_key_values=cache, cache_position=torch.arange(P, dtype=torch.long).to(dev),
                use_cache=True, logits_to_keep=1)
    nxt = sample(out.logits[:, -1, :])
    pos = positions[:, -1:]
    toks = [nxt]
    step_fn()

    for i in range(n_new - 1):
        cp = torch.tensor([P + i], dtype=torch.long).to(dev)       # device data, not a constant
        full_mask = torch.where(arange_l.unsqueeze(0) == cp, torch.ones_like(full_mask), full_mask)
        pos = pos + 1
        dec_mask = torch.where(full_mask.bool(), zero_m, neg_m).view(B, 1, 1, L)
        out = model(input_ids=nxt.unsqueeze(-1), attention_mask=dec_mask, position_ids=pos,
                    past_key_values=cache, cache_position=cp, use_cache=True, logits_to_keep=1)
        nxt = sample(out.logits[:, -1, :])
        toks.append(nxt)
        step_fn()

    return torch.cat([input_ids, torch.stack(toks, dim=1)], dim=1)
