"""Call whichever nkilib attention_cte is on PYTHONPATH (original or patched) directly from torch."""
from nkilib.core.attention.attention_cte import attention_cte
from torch_neuronx.nki_hop import wrap_nki
from torch_neuronx.utils import get_logical_neuron_cores

_fwd = wrap_nki(attention_cte)
LNC = int(get_logical_neuron_cores())  # read once: no Python side effects inside the compiled graph


def attn(q, k, v, scale):
    """q [B,H,L,D], k/v [B,KV,S,D], causal prefill -> [B,H,L,D]"""
    B, H, L, D = q.shape
    _, KV, S, _ = k.shape
    out = _fwd[(LNC,)](q.reshape(B * H, L, D), k.reshape(B * KV, S, D), v.reshape(B * KV, S, D),
                       tp_q=True, tp_k=True, scale=scale, causal_mask=True, cache_softmax=False)
    return out.reshape(B, H, L, D)
