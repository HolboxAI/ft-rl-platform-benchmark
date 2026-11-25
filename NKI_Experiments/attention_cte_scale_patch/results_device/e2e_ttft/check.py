import torch, glob, os
for S in (512, 2048, 4096):
    if not os.path.exists("ref_%d.json.pt" % S): continue
    r = torch.load("ref_%d.json.pt" % S)
    for f in sorted(glob.glob("*_%d_r*.json.pt" % S)):
        a = torch.load(f); x, y = a["prefill_logits"], r["prefill_logits"]
        print(f.replace(".json.pt",""), "max_rel_diff %.4f" % ((x-y).abs().max()/y.abs().max()), "same_first_token", a["tokens"][0]==r["tokens"][0], "top10_overlap", len(set(x.topk(10).indices[0].tolist())&set(y.topk(10).indices[0].tolist())))
