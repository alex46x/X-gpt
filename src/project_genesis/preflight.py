"""Validate a training plan and report capacity facts without allocating model weights."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from project_genesis.datasets import (
    DatasetManifest,
    DatasetSplit,
    is_sha256,
    load_dataset_config,
)
from project_genesis.evaluation import load_evaluation_config
from project_genesis.model import GPTDecoder, load_model_config, parameter_count
from project_genesis.preprocessing import load_preprocessing_config
from project_genesis.tokenizer import load_tokenizer_config
from project_genesis.training import load_training_config


@dataclass(frozen=True, slots=True)
class PreflightReport:
    """Validated source, model, schedule, and device capacity facts."""

    dataset_fingerprint: str
    source_files: int
    source_bytes: int
    training_files: int
    validation_files: int
    configured_vocab_size: int
    parameters: int
    persistent_training_state_bytes: int
    tokens_per_optimizer_step: int
    scheduled_tokens: int
    requested_device: str
    device_available: bool
    device_total_memory_bytes: int | None
    persistent_state_fits: bool | None
    ready: bool

    def __post_init__(self) -> None:
        """Validate report counts and fingerprint."""
        if not is_sha256(self.dataset_fingerprint):
            raise ValueError("dataset fingerprint must be a SHA-256 digest")
        positive = (
            self.source_files,
            self.source_bytes,
            self.training_files,
            self.validation_files,
            self.configured_vocab_size,
            self.parameters,
            self.persistent_training_state_bytes,
            self.tokens_per_optimizer_step,
            self.scheduled_tokens,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("preflight counts must be positive")


def preflight_experiment(
    *,
    dataset_config_path: Path,
    preprocessing_config_path: Path,
    tokenizer_config_path: Path,
    model_config_path: Path,
    training_config_path: Path,
    evaluation_config_path: Path,
    device: str | torch.device = "cpu",
) -> PreflightReport:
    """Validate an experiment and return non-allocating capacity estimates."""
    dataset_config = load_dataset_config(dataset_config_path)
    load_preprocessing_config(preprocessing_config_path)
    tokenizer_config = load_tokenizer_config(tokenizer_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    load_evaluation_config(evaluation_config_path)

    manifest = DatasetManifest.build(
        metadata=dataset_config.metadata,
        root=dataset_config.paths.data,
        sources=dataset_config.sources,
        created_at=dataset_config.metadata.created_at,
    )
    integrity = manifest.verify()
    if not integrity.is_valid:
        raise ValueError("dataset changed during preflight integrity verification")
    training_files = sum(entry.split is DatasetSplit.TRAIN for entry in manifest.entries)
    validation_files = sum(entry.split is DatasetSplit.VALIDATION for entry in manifest.entries)
    training_bytes = sum(
        entry.size_bytes for entry in manifest.entries if entry.split is DatasetSplit.TRAIN
    )
    validation_bytes = sum(
        entry.size_bytes for entry in manifest.entries if entry.split is DatasetSplit.VALIDATION
    )
    if not training_files or not validation_files or not training_bytes or not validation_bytes:
        raise ValueError("dataset must contain non-empty train and validation file sets")
    if tokenizer_config.vocab_size != model_config.vocab_size:
        raise ValueError("tokenizer and model configured vocabulary sizes do not match")
    if training_config.sequence_length > model_config.context_length:
        raise ValueError("training sequence length exceeds model context length")

    # PyTorch meta tensors preserve parameter shapes without allocating their storage.
    with torch.device("meta"):
        parameters = parameter_count(GPTDecoder(model_config))
    persistent_bytes = parameters * 16
    requested_device = torch.device(device)
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError("preflight supports cpu and cuda devices")
    available, total_memory = _device_capacity(requested_device)
    fits = None if total_memory is None else persistent_bytes <= total_memory
    tokens_per_step = (
        training_config.batch_size
        * training_config.sequence_length
        * training_config.gradient_accumulation_steps
    )
    return PreflightReport(
        dataset_fingerprint=manifest.fingerprint,
        source_files=len(manifest.entries),
        source_bytes=sum(entry.size_bytes for entry in manifest.entries),
        training_files=training_files,
        validation_files=validation_files,
        configured_vocab_size=model_config.vocab_size,
        parameters=parameters,
        persistent_training_state_bytes=persistent_bytes,
        tokens_per_optimizer_step=tokens_per_step,
        scheduled_tokens=tokens_per_step * training_config.max_steps,
        requested_device=str(requested_device),
        device_available=available,
        device_total_memory_bytes=total_memory,
        persistent_state_fits=fits,
        ready=available and fits is not False,
    )


def _device_capacity(device: torch.device) -> tuple[bool, int | None]:
    if device.type == "cpu":
        return True, None
    index = 0 if device.index is None else device.index
    available = torch.cuda.is_available() and index < torch.cuda.device_count()
    return (
        (True, torch.cuda.get_device_properties(index).total_memory) if available else (False, None)
    )


def main() -> None:
    """Print a strict JSON preflight report and fail when the device cannot fit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default="configs/dataset/default.yaml")
    parser.add_argument(
        "--preprocessing-config",
        type=Path,
        default="configs/preprocessing/default.yaml",
    )
    parser.add_argument(
        "--tokenizer-config",
        type=Path,
        default="configs/tokenizer/default.yaml",
    )
    parser.add_argument("--model-config", type=Path, default="configs/model/default.yaml")
    parser.add_argument(
        "--training-config",
        type=Path,
        default="configs/training/default.yaml",
    )
    parser.add_argument(
        "--evaluation-config",
        type=Path,
        default="configs/evaluation/default.yaml",
    )
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    report = preflight_experiment(
        dataset_config_path=arguments.dataset_config,
        preprocessing_config_path=arguments.preprocessing_config,
        tokenizer_config_path=arguments.tokenizer_config,
        model_config_path=arguments.model_config,
        training_config_path=arguments.training_config,
        evaluation_config_path=arguments.evaluation_config,
        device=arguments.device,
    )
    print(json.dumps(asdict(report), sort_keys=True, separators=(",", ":")))
    if not report.ready:
        raise SystemExit(1)
