"""
Driver for evaluation of LLaVa models on Intel Gaudi2 HPU.

References: https://github.com/TRI-ML/vlm-evaluation/blob/main/scripts/evaluate.py
            https://github.com/TRI-ML/vlm-evaluation/blob/main/vlm_eval/models/__init__.py
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import draccus
from accelerate.utils import set_seed

from vlm_eval import VLM
from vlm_eval.conf import DatasetConfig, DatasetRegistry
from vlm_eval.overwatch import initialize_overwatch
from vlm_eval.tasks import get_task_runner

from .llava_gemma_evaluator import LLaVaGemmaGaudi

# === Initializer Dispatch by Family ===
ID2INITIALIZER = {"llava-gemma-v1.5": LLaVaGemmaGaudi}

def load_vlm(
    model_id: str,
    run_dir: Path,
    ocr: Optional[bool] = False,
    load_precision: str = "bf16",
    max_length=128,
    temperature=1.0,
    ddp_backend: str = 'hccl',
    gaudi_config_name: Optional[str] = None,
) -> VLM:
    """Adapted from: https://github.com/TRI-ML/vlm-evaluation/blob/2092905d392e8dbedf01ed4b853df530e3cf9f35/vlm_eval/models/__init__.py#L14"""
    assert model_id in ID2INITIALIZER, f"Model ID `{model_id}` not supported!"
    return ID2INITIALIZER[model_id](
        model_id=model_id,
        run_dir=run_dir,
        load_precision=load_precision,
        max_length=max_length,
        temperature=temperature,
        ocr=ocr,
        ddp_backend=ddp_backend,
        gaudi_config_name=gaudi_config_name,
    )

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger` and `accelerate.PartialState`
overwatch = initialize_overwatch(__name__)


@dataclass
class EvaluationConfig:
    """Adapted from: https://github.com/TRI-ML/vlm-evaluation/blob/2092905d392e8dbedf01ed4b853df530e3cf9f35/scripts/evaluate.py#L32"""
    # fmt: off

    # DatasetConfig from `vlm_eval/conf/datasets.py`; override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.AI2D_FULL.dataset_id)
    )

    # === Model Parameters =>> LLaVa ===
    model_id: str = "llava-gemma-v1.5"
    model_dir: Path = "liuhaotian/llava-v1.5-7b"

    # Inference Parameters
    device_batch_size: int = 1                      # Device Batch Size set to 1 until LLaVa/HF LLaMa fixes bugs!
    num_workers: int = 2                            # Number of Dataloader Workers (on each process)

    # Artifact Parameters
    results_dir: Path = Path(                       # Path to results directory (writing predicted output, metrics)
        "results"
    )

    gaudi_config_name: Path = Path(
        "scripts/gaudi_config.json"
    )

    # Randomness
    seed: int = 21                                  # Random Seed (for reproducibility)

    ddp_backend: Optional[str] = field(
        default="hccl",
        metadata={
            "help": "The backend to be used for distributed evaluation.",
            "choices": ["hccl"],
        },
    )

    def __post_init__(self) -> None:
        self.run_dir = self.model_dir

    # fmt: on


@draccus.wrap()
def evaluate(cfg: EvaluationConfig) -> None:
    """Adapted from: https://github.com/TRI-ML/vlm-evaluation/blob/2092905d392e8dbedf01ed4b853df530e3cf9f35/scripts/evaluate.py#L79"""
    overwatch.info(f"Starting Evaluation for Dataset `{cfg.dataset.dataset_id}` w/ Model `{cfg.model_id}`")
    set_seed(cfg.seed)

    # Short-Circuit (if results/metrics already exist)
    task_results_dir = cfg.results_dir / cfg.dataset.dataset_family / cfg.dataset.dataset_id / cfg.model_id
    if (task_results_dir / "metrics.json").exists():
        overwatch.info(f"Metrics for `{cfg.dataset.dataset_id}` w/ `{cfg.model_id}` exist =>> exiting!")
        return

    # Build the VLM --> Download/Load Pretrained Model from Checkpoint
    overwatch.info("Initializing VLM =>> Bundling Models, Image Processors, and Tokenizer")
    vlm = load_vlm(cfg.model_id, cfg.run_dir, ocr=cfg.dataset.ocr, ddp_backend=cfg.ddp_backend, gaudi_config_name=cfg.gaudi_config_name)

    # Create Task Runner
    overwatch.info(f"Building Evaluation Runner for Dataset `{cfg.dataset.dataset_id}`")
    task_runner = get_task_runner(
        cfg.dataset.dataset_family,
        cfg.dataset.root_dir,
        cfg.dataset.index_file,
        task_results_dir,
        cfg.model_id,
        prompt_fn=vlm.get_prompt_fn(cfg.dataset.dataset_family),
        image_processor=vlm.image_processor,
    )

    # Run Evaluation
    overwatch.info("Starting (Distributed) Evaluation Loop")
    task_runner.evaluate(vlm, cfg.device_batch_size, cfg.num_workers)


if __name__ == "__main__":
    evaluate()