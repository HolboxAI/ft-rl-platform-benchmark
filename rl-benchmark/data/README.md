# Data: CholecT50 only

Every leg in this repo uses the same CholecT50 inputs as the verl, Miles,
SkyRL and NeMo-RL GPU runs. Neither file is committed. Fetch both with:

```bash
bash common/fetch_cholect50.sh      # needs read access to s3://stanford-train-data
```

| Input | Source | Local path |
|---|---|---|
| Frames (verl parquet) | `s3://stanford-train-data/cholec/parquet/` | `data/cholect50/parquet/{train,val}.parquet` |
| SFT checkpoint | `s3://stanford-train-data/models/cholec-sft-qwen3.5-4b` (9.9 GB) | `$SFT_CKPT` (default `~/models/cholec-sft-qwen3.5-4b`) |

| File | Rows | sha256 |
|---|---|---|
| `train.parquet` (122 MB) | 256 frames | `7463bb11bd6518b469f0cfe248218090fe6896b6e10f70721ee7a0910fb1bd10` |
| `val.parquet` (16 MB) | 32 frames | `5c6f3191fedd088b9dbd2c1c8e3e2089e075259cdc2ed01a19010686ef5c4c8c` |

The hashes are from the copy the SkyRL GPU control run downloaded on 2026-10-22.
`fetch_cholect50.sh` checks them.

Each row holds one endoscopic frame (854×480), the prompt (list every
`instrument-verb-target` triplet), and the ground-truth triplets with their
instrument / verb / target components. The reward is in
`common/rewards/cholect50.py`: 0.4 × triplet F1 + 0.6 × mean(instrument, verb, target F1).

`cholect50/build_parquet.py` is a verbatim copy of `verl/scripts/cholec_data.py`,
which rebuilt the parquet from the public CholecT50 zip.
