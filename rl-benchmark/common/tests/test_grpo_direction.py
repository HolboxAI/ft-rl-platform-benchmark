#!/usr/bin/env python3
"""CPU check that one GRPO update moves probability the right way: samples with
positive advantage gain log-prob, negative-advantage samples lose it.

The GRPO math does not depend on the task, so this uses a tiny text-only stand-in
task and any small Qwen3 text checkpoint (a 2-layer random one is enough):

    python -m common.tests.test_grpo_direction <hf id or local path to a small Qwen3>
"""
import sys

import torch

from common.grpo import train_grpo as tg
from common.grpo.device import get_backend
from common.grpo.rollout import response_logprobs
from common.grpo.tasks import Example, Task


class TextTask(Task):
    """Minimal text task with the same interface as CholecT50Task."""
    name = "text_probe"

    def encode(self, examples):
        texts = [self.tok.apply_chat_template(ex.prompt, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False) for ex in examples]
        return dict(self.tok(texts, padding="max_length", max_length=self.max_prompt_len, return_tensors="pt"))

    def score(self, completion, ex):
        return {"reward": 0.0}


def main(model_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = tg.parse_args(["--backend", "cpu", "--model", model_path, "--out-dir", "/dev/null",
                          "--rollout-batch-size", "2", "--n-samples", "4", "--gen-micro-batch", "8",
                          "--train-micro-batch", "4", "--lr", "1e-3", "--kl-beta", "0.04",
                          "--max-new-tokens", "16", "--lora-rank", "0", "--dtype", "fp32"])
    backend = get_backend("cpu")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.config._attn_implementation = "eager"
    ref = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
    eos = sorted({tok.eos_token_id, *([model.generation_config.eos_token_id]
                  if isinstance(model.generation_config.eos_token_id, int)
                  else model.generation_config.eos_token_id or [])})
    task = TextTask(tok, max_prompt_len=48)
    prompts = [Example([{"role": "user", "content": "Name a surgical instrument."}], None),
               Example([{"role": "user", "content": "Name a surgical target."}], None)]
    examples = [e for e in prompts for _ in range(4)]
    resp_ids = tg.rollout(task, model, examples, args, backend, tok.pad_token_id, eos)
    resp, mask, *_ = tg.score_responses(task, examples, resp_ids, eos, args)
    adv = tg.group_advantages([1, 0, 0, 1, 0, 1, 1, 0], 4)
    enc = task.encode(examples)
    with torch.no_grad():
        before = (response_logprobs(model, enc, resp, 0.7) * mask).sum(1)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):
        stats = tg.train_step(task, model, ref, opt, examples, resp, mask, adv, args, backend)
    with torch.no_grad():
        after = (response_logprobs(model, enc, resp, 0.7) * mask).sum(1)
    d = (after - before).tolist()
    print("adv  ", [round(a, 2) for a in adv])
    print("dlogp", [round(x, 2) for x in d])
    print(stats)
    ok = all((x > 0) == (a > 0) for x, a in zip(d, adv))
    print("DIRECTION OK" if ok else "DIRECTION WRONG")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
