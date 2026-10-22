#!/usr/bin/env python3
"""CPU correctness checks for the static-shape rollout and log-probs.

    python -m common.tests.test_rollout --model <hf id or local path>

Any small Qwen3 checkpoint works (a 2-layer random one is enough). Checks:
  1. greedy `generate_static` == greedy `transformers.generate` (token for token)
  2. `response_logprobs` of the greedy continuation == log-softmax of the
     logits the generator saw (same positions, same padding)
  3. early stop does not cut rows that have not produced EOS
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.grpo.device import get_backend
from common.grpo.rollout import generate_static, postprocess, response_logprobs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--new-tokens", type=int, default=24)
    args = p.parse_args()

    backend = get_backend("cpu")
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    model.config._attn_implementation = "eager"
    eos = sorted({tok.eos_token_id, *([model.generation_config.eos_token_id]
                  if isinstance(model.generation_config.eos_token_id, int)
                  else model.generation_config.eos_token_id)})

    texts = ["I love this so much!", "This is the worst day, I am furious and sad about everything."]
    enc = dict(tok(texts, padding="max_length", max_length=24, return_tensors="pt"))
    T = args.new_tokens

    # 1. static greedy == HF greedy. HF without early stopping: suppress EOS so
    # both run the full T tokens and compare every position.
    ours = generate_static(model, enc, T, 1.0, tok.pad_token_id, backend, greedy=True, eos_ids=[])
    pos = (enc["attention_mask"].cumsum(-1) - 1).clamp(min=0)
    ref = model.generate(**enc, max_new_tokens=T, min_new_tokens=T, do_sample=False,
                         pad_token_id=tok.pad_token_id)[:, enc["input_ids"].shape[1]:]
    assert torch.equal(ours, ref), f"static != hf\n{ours}\n{ref}"
    print("OK 1: static greedy matches transformers.generate")

    # 2. log-probs of the greedy continuation under teacher forcing equal the
    # max log-prob at each step (greedy picks argmax) — recompute directly.
    lp = response_logprobs(model, enc, ours, temperature=1.0)
    full = torch.cat([enc["input_ids"], ours], 1)
    fmask = torch.ones_like(full)
    fmask[:, : enc["input_ids"].shape[1]] = enc["attention_mask"]
    fpos = (fmask.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=full, attention_mask=fmask, position_ids=fpos).logits
    P = enc["input_ids"].shape[1]
    direct = torch.log_softmax(logits[:, P - 1:-1].float(), -1).gather(-1, ours.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(lp, direct, atol=1e-4), (lp - direct).abs().max()
    best = torch.log_softmax(logits[:, P - 1:-1].float(), -1).max(-1).values
    assert torch.allclose(lp, best, atol=1e-4), "greedy token is not the argmax"
    print("OK 2: response_logprobs matches direct log-softmax and argmax")

    # 3. sampling with early stop: every row's mask must end at a real EOS or at T.
    torch.manual_seed(0)
    s = generate_static(model, enc, T, 0.7, tok.pad_token_id, backend, eos_ids=eos, early_stop_every=4)
    resp, mask, lengths, truncated, first = postprocess(s, set(eos), T)
    for i in range(resp.shape[0]):
        if not truncated[i]:
            assert int(resp[i, first[i]]) in eos
            # the EOS must be a sampled token, so everything before it is non-pad output
    print(f"OK 3: sampled lengths {lengths.tolist()}, truncated {truncated.tolist()}")


if __name__ == "__main__":
    main()
