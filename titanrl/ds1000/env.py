"""Single-turn DS-1000 code-completion environment for TitanRL."""

from __future__ import annotations

from dataclasses import dataclass

from renderers import Message

from torchtitan.rl.examples.ds1000.data import Ds1000Sample
from torchtitan.rl.rollout.message_env import (
    MessageEnv,
    MessageEnvInitOutput,
    MessageEnvStepOutput,
)


class Ds1000Env(MessageEnv):
    """Ask the model to complete a partial Python function, then end the rollout.

    A single-turn environment: ``init`` sends the completion prompt, and ``step``
    terminates after the first (and only) completion.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(MessageEnv.Config):
        pass

    def __init__(self, config: Config, *, env_input: Ds1000Sample) -> None:
        self._env_input = env_input

    async def init(self) -> MessageEnvInitOutput:
        prompt = (
            "Complete the following Python code. Return only the completed code "
            "inside a single ```python code block, with no explanation.\n\n"
            f"```python\n{self._env_input.prompt}\n```"
        )
        return MessageEnvInitOutput(
            init_prompt_messages=[{"role": "user", "content": prompt}]
        )

    async def step(self, completion_message: Message) -> MessageEnvStepOutput:
        del completion_message
        return MessageEnvStepOutput(done=True)
