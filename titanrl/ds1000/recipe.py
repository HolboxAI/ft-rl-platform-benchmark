"""TitanRL DS-1000 GRPO recipe (single-GPU friendly).

Place this file at ``torchtitan_recipes/rl/ds1000.py`` in the torchtitan
checkout, and the example files (data.py / env.py / rubric.py) at
``torchtitan/rl/examples/ds1000/``.

Run:
    python -m torchtitan.rl.train \
      --module torchtitan_recipes.rl.ds1000 \
      --config rl_grpo_qwen3_0_6b_ds1000 \
      --output-dir outputs/ds1000
"""

from __future__ import annotations

from renderers import Qwen3RendererConfig

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.components.renderer import from_renderers
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3 import build_model_config
from torchtitan.rl.controller.controller import AsyncLoopConfig, Controller, ValidationConfig
from torchtitan.rl.examples.ds1000.data import Ds1000Dataset
from torchtitan.rl.examples.ds1000.env import Ds1000Env
from torchtitan.rl.examples.ds1000.rubric import RewardDs1000
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.types import SamplingConfig
from torchtitan.rl.observability.metrics import MetricsProcessor
from torchtitan.rl.rollout.data import IterableRolloutDataLoader
from torchtitan.rl.rollout.rollouter import Rollouter, RolloutWorker
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.trainer.losses import GRPOLoss
from torchtitan.rl.trainer.trainer import Trainer

from torchtitan_recipes.rl.model_configs import build_rl_model_configs


def _ds1000_rollouter_config() -> Rollouter.Config:
    return Rollouter.Config(
        training_dataloader=IterableRolloutDataLoader.Config(
            dataset=Ds1000Dataset.Config(seed=42, max_problems=800)
        ),
        validation_dataset=Ds1000Dataset.Config(seed=99, max_problems=200),
        worker=RolloutWorker.Config(
            rubric=Rubric.Config(reward_fns=[RewardDs1000.Config(weight=1.0)]),
            message_env=Ds1000Env.Config(),
        ),
    )


def _qwen3_rl_model_configs(
    flavor: str,
    *,
    seq_len: int,
    attn_backend: str,
    trainer_parallelism: ParallelismConfig,
    generator_parallelism: InferenceParallelismConfig,
):
    return build_rl_model_configs(
        build_model_config,
        flavor,
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
        seq_len=seq_len,
        attn_backend=attn_backend,
    )


def rl_grpo_qwen3_0_6b_ds1000(*, seq_len: int = 4096) -> Controller.Config:
    """GRPO on DS-1000 with Qwen3-0.6B.

    Uses one GPU for the trainer and one for the generator (TP=1, DP=1 each).
    """
    num_samples_per_prompt = 4
    trainer_parallelism = ParallelismConfig(
        data_parallel_shard_degree=1,
        tensor_parallel_degree=1,
    )
    generator_parallelism = InferenceParallelismConfig(
        data_parallel_degree=1,
        tensor_parallel_degree=1,
    )
    trainer_model, generator_model = _qwen3_rl_model_configs(
        "0.6B",
        seq_len=seq_len,
        attn_backend="varlen",
        trainer_parallelism=trainer_parallelism,
        generator_parallelism=generator_parallelism,
    )
    return Controller.Config(
        hf_assets_path="./assets/hf/Qwen3-0.6B",
        async_loop=AsyncLoopConfig(
            num_training_steps=100,
            num_prompts_per_train_step=8,
            num_samples_per_prompt=num_samples_per_prompt,
            validation=ValidationConfig(steps=20),
        ),
        rollouter=_ds1000_rollouter_config(),
        renderer=from_renderers(Qwen3RendererConfig(enable_thinking=False)),
        metrics=MetricsProcessor.Config(enable_wandb=False, enable_tensorboard=False),
        trainer=Trainer.Config(
            model=trainer_model,
            optim=Optim.Config(
                optimizer=OptimizersContainer.Config(
                    optimizers=[AdamW.Config(pattern=r".*", lr=2e-6)]
                ),
                lr_scheduler=LRSchedulersContainer.Config(
                    warmup_steps=2,
                    decay_type="linear",
                ),
            ),
            training=TrainingConfig(
                disable_accelerator_graphs=True,
                num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
                max_context_length=seq_len,
            ),
            parallelism=trainer_parallelism,
            checkpointer=CheckpointManager.Config(
                initial_load_in_hf=True,
                interval=10,
                last_save_model_only=False,
            ),
            loss=ChunkedLossWrapper.Config(
                num_chunks=8,
                loss_fn=GRPOLoss.Config(
                    global_vocab_size=decoder_vocab_size(trainer_model)
                ),
            ),
        ),
        generator=Generator.Config(
            model=generator_model,
            model_dtype="bfloat16",
            parallelism=generator_parallelism,
            checkpointer=None,
            gpu_memory_limit=0.5,
            sampling=SamplingConfig(
                temperature=0.8,
                max_tokens=512,
            ),
        ),
    )
