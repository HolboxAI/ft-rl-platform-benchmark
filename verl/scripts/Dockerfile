FROM verlai/verl:uv.cu130

# Custom verl image for the CholecT50 GRPO (RLVR) run on SageMaker.
# Everything is baked in: the triplet-F1 reward and the launch script.
# The base image ships a 16 GB uv cache so `uv run --frozen --offline` resolves
# with no network access on the training node.

ENV SM_MODEL_DIR=/opt/ml/model \
    SM_CHANNEL_DATA=/opt/ml/input/data/data \
    SM_CHANNEL_MODEL=/opt/ml/input/data/model

COPY cholec_reward.py /workspace/verl/cholec_reward.py
COPY train.sh /workspace/train.sh
COPY patch_qwen35.py /tmp/patch_qwen35.py
RUN python3 /tmp/patch_qwen35.py
COPY patch_tensordict.py /tmp/patch_tensordict.py
RUN python3 /tmp/patch_tensordict.py
COPY patch_fix.py /tmp/patch_fix.py
RUN python3 /tmp/patch_fix.py
RUN chmod +x /workspace/train.sh

WORKDIR /workspace/verl
ENTRYPOINT ["/bin/bash", "/workspace/train.sh"]
