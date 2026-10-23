"""Dump trl 1.16.0 GRPOConfig field names + neuron/compile backend handling."""
import dataclasses

from trl import GRPOConfig

names = [fl.name for fl in dataclasses.fields(GRPOConfig)]
print("ALL_FIELDS_COUNT", len(names))
for n in names:
    print("FIELD", n)
print("---")
print("has max_prompt_length:", "max_prompt_length" in names)
print("has max_completion_length:", "max_completion_length" in names)
print("length fields:", [n for n in names if "length" in n])
print("prompt fields:", [n for n in names if "prompt" in n])
print("compile fields:", [n for n in names if "compile" in n])

import trl
trl_root = trl.__file__.rsplit("/", 1)[0] + "/trainer/grpo_trainer.py"
print("=== grpo_trainer.py neuron/backend refs ===")
try:
    with open(trl_root) as fh:
        hits = [(i, line.rstrip()) for i, line in enumerate(fh, 1)
                if "neuron" in line.lower() or "torch.compile" in line or "torch_compile_backend" in line]
    for i, line in hits:
        print(f"{i}: {line}")
    print("num hits:", len(hits))
except OSError as e:
    print("cannot read", trl_root, e)

print("TRL OK")
