"""SkyRL entrypoint for GRPO on DS-1000 on Cloud TPU.

Same as ``skyrl_cholect50/entrypoint.py`` (its TPU config check and its metrics capture are reused), with the
``ds1000`` environment registered instead of ``cholec``, and a check that the DS-1000 test venv is set up, since
a missing one would only show as every reward being 0.
"""
import multiprocessing as mp
import os
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register

from entrypoint import check_tpu_setup, install_metrics_jsonl  # skyrl_cholect50/entrypoint.py

mp.set_start_method("spawn", force=True)


def check_ds1000_setup() -> None:
    py = os.environ.get("DS1000_PYTHON", "")
    if not (py and os.access(py, os.X_OK)):
        sys.exit("DS1000_PYTHON is not an executable; run skyrl_ds1000/setup_exec_venv.sh and export it")
    data_home = os.environ.get("DS1000_DATA_HOME", os.path.expanduser("~/.cache/ds1000-data"))
    if not os.path.isdir(os.path.join(data_home, "seaborn")):
        sys.exit(f"no DS-1000 sample data in {data_home}; run skyrl_ds1000/setup_exec_venv.sh")


@ray.remote(num_cpus=1)
def ds1000_entrypoint(cfg: SkyRLTrainConfig):
    install_metrics_jsonl()
    register(id="ds1000", entry_point="ds1000_env:DS1000Env")
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    check_tpu_setup(cfg)
    check_ds1000_setup()
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(ds1000_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
