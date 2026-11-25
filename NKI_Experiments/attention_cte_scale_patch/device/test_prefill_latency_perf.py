"""Prefill attention latency, before vs after the scale patch.

Same operating point as test/integration/nkilib/core/attention/test_qwen235b_attention_perf.py
(Qwen3-235B, TP4, CP1, head_dim 128), with two changes: softmax_scale is the model's real
1/sqrt(d) instead of 1.0, and several prompt lengths are run.

Copy into test/integration/nkilib/core/attention/ of BOTH trees (original and patched) and run
the identical file in each. PERF_SCALE=1.0 gives a control run, where no difference is expected.
"""

import math
import os

import nki.language as nl
import pytest

from test.integration.nkilib.core.attention.test_attention_cte import (
    CAUSAL_MASK_DIM_NAME,
    SINK_DIM_NAME,
    TP_K_DIM_NAME,
    TP_OUT_DIM_NAME,
    TP_Q_DIM_NAME,
)
from test.integration.nkilib.core.attention.test_attention_cte import (
    # aliased so pytest does not re-collect the whole CTE suite here
    TestRangedAttentionCTEKernels as _RangedAttentionCTEKernels,
)
from test.integration.nkilib.core.attention.test_attention_cte_model_config import get_attention_config
from test.utils.common_dataclasses import CompilerArgs, Platforms
from test.utils.metrics_collector import IMetricsCollector
from test.utils.pytest_test_metadata import pytest_marks, pytest_test_metadata
from test.utils.test_orchestrator import Orchestrator

_SEQLENS = [512, 1024, 2048, 4096, 8192]


@pytest_test_metadata(name="Prefill latency before/after", tags=["model"])
@pytest_marks(["attention", "cte", "model"])
class TestPrefillLatency:
    @pytest.mark.parametrize("seqlen", _SEQLENS, ids=[f"s{s}" for s in _SEQLENS])
    def test_prefill(
        self,
        test_manager: Orchestrator,
        collector: IMetricsCollector,
        platform_target: Platforms,
        seqlen,
    ):
        conf = get_attention_config("qwen3_235b", tp=4, cp=1, seqlen=seqlen)
        scale = float(os.environ.get("PERF_SCALE") or 1.0 / math.sqrt(conf["d"]))
        _RangedAttentionCTEKernels().run_attention_cte_test(
            test_manager=test_manager,
            compiler_args=CompilerArgs(logical_nc_config=2, platform_target=platform_target),
            collector=collector,
            bool_dims={
                CAUSAL_MASK_DIM_NAME: conf["causal_mask"],
                TP_Q_DIM_NAME: conf["tp_q"],
                TP_K_DIM_NAME: conf["tp_k"],
                TP_OUT_DIM_NAME: conf["tp_out"],
                SINK_DIM_NAME: conf["sink"],
            },
            bs=conf["bs"],
            bs_kv=conf["bs"] // conf["gqa_factor"],
            cache_softmax=False,
            cp_degree=conf["cp_degree"],
            d=conf["d"],
            dtype=nl.bfloat16,
            seqlen_kv=conf["seqlen_kv"],
            seqlen_kv_prior=conf["seqlen_kv_prior"],
            prior_used_len=conf["prior_used_len"],
            seqlen_q=conf["seqlen_kv"] // conf["cp_degree"],
            sliding_window=conf["sliding_window"],
            softmax_scale=scale,
            cp_rank_id=conf["cp_rank_id"],
            use_cp=False,
            cp_strided_q_slicing=False,
        )
