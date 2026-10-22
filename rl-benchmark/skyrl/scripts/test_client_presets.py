#!/usr/bin/env python3
# Usage: python skyrl/scripts/test_client_presets.py skyrl/scripts/cholect50_rl_client.py  (needs torch + pillow only)
"""Load the SkyRL client with stubbed tinker / chz / tinker_cookbook (not installable on this
laptop's Python 3.9) and check preset resolution + the std-normalised advantage patch."""
import sys, types, importlib.util, torch
def stub(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
class Base: pass
stub("chz", chz=lambda c: c)
stub("tinker", ModelInput=Base, types=types.SimpleNamespace(SamplingParams=Base), ServiceClient=Base)
stub("tinker_cookbook", checkpoint_utils=None, renderers=None)
stub("tinker_cookbook.image_processing_utils", get_image_processor=None)
train = stub("tinker_cookbook.rl.train", compute_advantages="ORIGINAL")
stub("tinker_cookbook.rl", train=train)
stub("tinker_cookbook.rl.types", **{n: Base for n in "Action ActionExtra Env EnvGroupBuilder RLDataset RLDatasetBuilder StepResult".split()})
stub("tinker_cookbook.tokenizer_utils", get_tokenizer=None)
stub("tinker_cookbook.renderers"); stub("tinker_cookbook.checkpoint_utils")
spec = importlib.util.spec_from_file_location("client", sys.argv[1]); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)

# presets are complete and consistent with argparse dests
for name, p in c.PRESETS.items():
    assert set(p) == {"loss","norm_adv_by_std","num_substeps","lora_rank","temperature","remove_constant_reward_groups","clip_low","clip_high"}, name
# std patch: compare to SkyRL's formula on a group and a singleton
c.install_std_normalised_advantages()
assert c.train.compute_advantages != "ORIGINAL"
G = types.SimpleNamespace(get_total_rewards=lambda: [0.2, 0.5, 0.5, 0.9])
S = types.SimpleNamespace(get_total_rewards=lambda: [0.4])
a, s = c.train.compute_advantages([G, S])
r = torch.tensor([0.2, 0.5, 0.5, 0.9]); ref = (r - r.mean()) / (torch.std(torch.tensor([r.tolist()])) + 1e-6)
assert torch.allclose(a, ref), (a, ref)
assert torch.allclose(s, torch.tensor([0.4]) / (1 + 1e-6))
print("presets OK; std-normalised advantages match SkyRL:", [round(x, 4) for x in a.tolist()])
