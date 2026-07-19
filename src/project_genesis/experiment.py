"""End-to-end local experiment execution."""

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from project_genesis.datasets import (
    Dataset,
    DatasetManifest,
    DatasetSplit,
    load_dataset_config,
    sha256_file,
)
from project_genesis.evaluation import EvaluationResult, evaluate_model, load_evaluation_config
from project_genesis.inference import BundleProvenance, save_bundle
from project_genesis.model import GPTDecoder, load_model_config
from project_genesis.preprocessing import load_preprocessing_config, preprocess_dataset
from project_genesis.tokenizer import (
    load_tokenizer_config,
    save_tokenizer,
    tokenize_dataset,
    train_tokenizer,
)
from project_genesis.training import (
    TokenBatch,
    Trainer,
    iter_token_batches,
    load_training_config,
    save_checkpoint,
    seed_training,
)
from project_genesis.utilities import atomic_write_text


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Published experiment identity and final evaluation."""

    output: Path
    bundle_fingerprint: str
    dataset_fingerprint: str
    tokenizer_fingerprint: str
    training_steps: int
    evaluation: EvaluationResult


def run_experiment(
    *,
    dataset_config_path: Path,
    preprocessing_config_path: Path,
    tokenizer_config_path: Path,
    model_config_path: Path,
    training_config_path: Path,
    evaluation_config_path: Path,
    output: Path,
    source_revision: str,
    training_run_id: str,
    device: str | torch.device = "cpu",
) -> ExperimentResult:
    """Train, evaluate, and atomically publish one local experiment."""
    destination = output.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"experiment destination already exists: {output}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(dataset_config_path)
    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    tokenizer_config = load_tokenizer_config(tokenizer_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    evaluation_config = load_evaluation_config(evaluation_config_path)

    temporary = Path(tempfile.mkdtemp(dir=destination.parent, prefix=f".{destination.name}-"))
    try:
        input_manifest = DatasetManifest.build(
            metadata=dataset_config.metadata,
            root=dataset_config.paths.data,
            sources=dataset_config.sources,
            created_at=dataset_config.metadata.created_at,
        )
        processed = preprocess_dataset(
            dataset_config,
            preprocessing_config,
            input_manifest,
        )
        provenance = BundleProvenance(
            source_revision,
            training_run_id,
            processed.manifest.output_fingerprint,
        )
        training_dataset = _split(processed.dataset, DatasetSplit.TRAIN)
        validation_dataset = _split(processed.dataset, DatasetSplit.VALIDATION)

        tokenizer_result = train_tokenizer(training_dataset, tokenizer_config)
        tokenizer = tokenizer_result.tokenizer
        if model_config.vocab_size != len(tokenizer.vocabulary):
            raise ValueError(
                "model vocabulary size does not match trained tokenizer: "
                f"{model_config.vocab_size} != {len(tokenizer.vocabulary)}"
            )
        if training_config.sequence_length > model_config.context_length:
            raise ValueError("training sequence length exceeds model context length")

        tokenized_training = tokenize_dataset(training_dataset, tokenizer)
        tokenized_validation = tokenize_dataset(validation_dataset, tokenizer)
        seed_training(training_config.seed)
        model = GPTDecoder(model_config)
        trainer = Trainer(model, training_config, device=device)
        batches = _repeat_batches(
            tokenized_training,
            batch_size=training_config.batch_size,
            sequence_length=training_config.sequence_length,
            separator_token_id=tokenizer.vocabulary.eos_id,
        )
        for _ in range(training_config.max_steps):
            trainer.train_step(
                tuple(next(batches) for _ in range(training_config.gradient_accumulation_steps))
            )

        evaluation = evaluate_model(
            model,
            iter_token_batches(
                tokenized_validation,
                batch_size=training_config.batch_size,
                sequence_length=training_config.sequence_length,
                separator_token_id=tokenizer.vocabulary.eos_id,
                drop_last=False,
            ),
            max_batches=evaluation_config.max_batches,
        )

        atomic_write_text(
            temporary / "input-manifest.json",
            json.dumps(input_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        )
        processed.manifest.write(temporary / "processed-manifest.json")
        save_tokenizer(tokenizer, temporary / "tokenizer.json")
        save_checkpoint(temporary / "checkpoint.pt", trainer)
        bundle_fingerprint = save_bundle(
            temporary / "bundle",
            model,
            tokenizer,
            provenance=provenance,
        )
        config_paths = {
            "dataset": dataset_config_path,
            "preprocessing": preprocessing_config_path,
            "tokenizer": tokenizer_config_path,
            "model": model_config_path,
            "training": training_config_path,
            "evaluation": evaluation_config_path,
        }
        config_checksums: dict[str, str] = {}
        for name, path in config_paths.items():
            source = path.expanduser().resolve()
            target = temporary / "configs" / f"{name}.yaml"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            config_checksums[name] = sha256_file(target)
        result = ExperimentResult(
            output=destination,
            bundle_fingerprint=bundle_fingerprint,
            dataset_fingerprint=processed.manifest.output_fingerprint,
            tokenizer_fingerprint=tokenizer.fingerprint,
            training_steps=trainer.step,
            evaluation=evaluation,
        )
        atomic_write_text(
            temporary / "run.json",
            json.dumps(
                {
                    "source_revision": source_revision,
                    "training_run_id": training_run_id,
                    "bundle_fingerprint": result.bundle_fingerprint,
                    "dataset_fingerprint": result.dataset_fingerprint,
                    "tokenizer_fingerprint": result.tokenizer_fingerprint,
                    "training_steps": result.training_steps,
                    "config_sha256": config_checksums,
                    "evaluation": asdict(result.evaluation),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        os.replace(temporary, destination)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _split(dataset: Dataset, split: DatasetSplit) -> Dataset:
    records = tuple(record for record in dataset if record.metadata.get("split") == split.value)
    if not records:
        raise ValueError(f"processed dataset has no {split.value} records")
    return Dataset(dataset.metadata, records, dataset.schema)


def _repeat_batches(
    dataset: Dataset,
    *,
    batch_size: int,
    sequence_length: int,
    separator_token_id: int,
) -> Iterator[TokenBatch]:
    # ponytail: deterministic epoch replay; add shuffling when training policy defines it.
    while True:
        yielded = False
        for batch in iter_token_batches(
            dataset,
            batch_size=batch_size,
            sequence_length=sequence_length,
            separator_token_id=separator_token_id,
            drop_last=False,
        ):
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("training split does not contain one complete sequence")


def main() -> None:
    """Run an end-to-end local experiment from subsystem configurations."""
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    result = run_experiment(
        dataset_config_path=arguments.dataset_config,
        preprocessing_config_path=arguments.preprocessing_config,
        tokenizer_config_path=arguments.tokenizer_config,
        model_config_path=arguments.model_config,
        training_config_path=arguments.training_config,
        evaluation_config_path=arguments.evaluation_config,
        output=arguments.output,
        source_revision=arguments.source_revision,
        training_run_id=arguments.training_run_id,
        device=arguments.device,
    )
    print(
        json.dumps(
            {
                "output": str(result.output),
                "bundle_fingerprint": result.bundle_fingerprint,
                "evaluation_loss": result.evaluation.loss,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
