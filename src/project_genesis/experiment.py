"""Recoverable end-to-end local experiment execution."""

import argparse
import json
import os
import shutil
import signal
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from types import FrameType

import torch

from project_genesis.datasets import (
    Dataset,
    DatasetManifest,
    DatasetSplit,
    load_dataset_config,
    sha256_file,
)
from project_genesis.evaluation import EvaluationResult, evaluate_model, load_evaluation_config
from project_genesis.inference import BundleProvenance, load_bundle, save_bundle
from project_genesis.model import GPTDecoder, load_model_config
from project_genesis.preprocessing import (
    PreprocessingResult,
    load_preprocessing_config,
    preprocess_dataset,
)
from project_genesis.tokenizer import (
    ByteBPETokenizer,
    load_tokenizer,
    load_tokenizer_config,
    save_tokenizer,
    tokenize_dataset,
    train_tokenizer,
)
from project_genesis.training import (
    Trainer,
    iter_shuffled_token_batches,
    iter_token_batches,
    load_checkpoint,
    load_training_config,
    save_checkpoint,
    seed_training,
)
from project_genesis.utilities import atomic_write_text

RUN_STATE_VERSION = 1
STATE_FILE = "state.json"
METRICS_FILE = "metrics.jsonl"
BEST_FILE = "best-checkpoint.json"


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
    resume: Path | None = None,
    init_bundle: Path | None = None,
) -> ExperimentResult:
    """Train, evaluate, and atomically publish one recoverable local experiment."""
    if resume is not None and init_bundle is not None:
        raise ValueError("init-bundle cannot be combined with resume")
    destination = output.expanduser().resolve()
    staging = destination.parent / f".{destination.name}.in-progress"
    if destination.exists():
        raise FileExistsError(f"experiment destination already exists: {output}")
    if resume is None and staging.exists():
        raise FileExistsError(
            f"unfinished experiment exists: {staging}; pass its latest checkpoint with --resume"
        )
    if resume is not None and not staging.is_dir():
        raise FileNotFoundError(f"experiment staging directory does not exist: {staging}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    dataset_config = load_dataset_config(dataset_config_path)
    preprocessing_config = load_preprocessing_config(preprocessing_config_path)
    tokenizer_config = load_tokenizer_config(tokenizer_config_path)
    model_config = load_model_config(model_config_path)
    training_config = load_training_config(training_config_path)
    evaluation_config = load_evaluation_config(evaluation_config_path)
    config_paths = {
        "dataset": dataset_config_path,
        "preprocessing": preprocessing_config_path,
        "tokenizer": tokenizer_config_path,
        "model": model_config_path,
        "training": training_config_path,
        "evaluation": evaluation_config_path,
    }
    config_checksums = {
        name: sha256_file(path.expanduser().resolve()) for name, path in config_paths.items()
    }

    prepared = False
    new_run = resume is None
    trainer: Trainer | None = None
    initial_model: GPTDecoder | None = None
    initial_bundle_fingerprint: str | None = None
    state: dict[str, object] = {}
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

        if new_run:
            if init_bundle is None:
                tokenizer = train_tokenizer(training_dataset, tokenizer_config).tokenizer
            else:
                initialized = load_bundle(init_bundle)
                if initialized.model.config != model_config:
                    raise ValueError("initial bundle model configuration does not match")
                tokenizer = initialized.tokenizer
                initial_model = initialized.model
                initial_bundle_fingerprint = initialized.fingerprint
        else:
            state = _load_state(staging / STATE_FILE)
            tokenizer = load_tokenizer(staging / "tokenizer.json")
            initial_bundle_fingerprint = _optional_string(state.get("initial_bundle_fingerprint"))
        if model_config.vocab_size != len(tokenizer.vocabulary):
            raise ValueError(
                "model vocabulary size does not match trained tokenizer: "
                f"{model_config.vocab_size} != {len(tokenizer.vocabulary)}"
            )
        if training_config.sequence_length > model_config.context_length:
            raise ValueError("training sequence length exceeds model context length")

        identity = {
            "source_revision": provenance.source_revision,
            "training_run_id": provenance.training_run_id,
            "dataset_fingerprint": provenance.dataset_fingerprint,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "initial_bundle_fingerprint": initial_bundle_fingerprint,
            "config_sha256": config_checksums,
        }
        if new_run:
            staging.mkdir()
            _initialize_staging(
                staging,
                input_manifest=input_manifest,
                processed=processed,
                tokenizer=tokenizer,
                config_paths=config_paths,
            )
            state = _state(identity, status="running", last_step=0)
            _write_state(staging, state)
        else:
            _validate_identity(state, identity)
        prepared = True

        tokenized_training = tokenize_dataset(training_dataset, tokenizer)
        tokenized_validation = tokenize_dataset(validation_dataset, tokenizer)
        seed_training(training_config.seed)
        model = initial_model if initial_model is not None else GPTDecoder(model_config)
        trainer = Trainer(model, training_config, device=device)
        if resume is not None:
            checkpoint = _validated_resume_checkpoint(staging, resume)
            load_checkpoint(checkpoint, trainer)
            _truncate_metrics(staging / METRICS_FILE, trainer.step)

        # ponytail: exact replay avoids sampler state; store epoch offsets only if
        # measured resume startup makes replay materially expensive.
        batches = islice(
            iter_shuffled_token_batches(
                tokenized_training,
                batch_size=training_config.batch_size,
                sequence_length=training_config.sequence_length,
                separator_token_id=tokenizer.vocabulary.eos_id,
                seed=training_config.seed,
            ),
            trainer.microbatches_seen,
            None,
        )
        best_loss = _optional_number(state.get("best_validation_loss"))
        best_checkpoint = _optional_string(state.get("best_checkpoint"))
        last_evaluation: EvaluationResult | None = None
        last_evaluation_step = -1

        previous_sigterm = None
        handles_signals = threading.current_thread() is threading.main_thread()
        if handles_signals:
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _interrupt)
        try:
            while trainer.step < training_config.max_steps:
                previous_step = trainer.step
                loss = trainer.train_step(
                    tuple(next(batches) for _ in range(training_config.gradient_accumulation_steps))
                )
                if trainer.step == previous_step:
                    continue
                if trainer.step % training_config.log_interval_steps == 0:
                    _append_metric(
                        staging / METRICS_FILE,
                        {
                            "type": "train",
                            "step": trainer.step,
                            "loss": loss,
                            "learning_rate": trainer.optimizer.param_groups[0]["lr"],
                        },
                    )

                evaluation_due = trainer.step % training_config.evaluation_interval_steps == 0
                checkpoint_due = trainer.step % training_config.checkpoint_interval_steps == 0
                if evaluation_due:
                    last_evaluation = _evaluate(
                        model,
                        tokenized_validation,
                        tokenizer,
                        training_config.batch_size,
                        training_config.sequence_length,
                        evaluation_config.max_batches,
                    )
                    last_evaluation_step = trainer.step
                    _append_metric(
                        staging / METRICS_FILE,
                        {
                            "type": "validation",
                            "step": trainer.step,
                            **asdict(last_evaluation),
                        },
                    )
                if checkpoint_due or evaluation_due:
                    checkpoint_path = _save_step_checkpoint(staging, trainer)
                    if (
                        last_evaluation is not None
                        and last_evaluation_step == trainer.step
                        and (best_loss is None or last_evaluation.loss < best_loss)
                    ):
                        best_loss = last_evaluation.loss
                        best_checkpoint = checkpoint_path.relative_to(staging).as_posix()
                        _write_best(staging, trainer.step, best_loss, best_checkpoint)
                    _prune_checkpoints(
                        staging,
                        keep_last=training_config.keep_last_checkpoints,
                        best_checkpoint=best_checkpoint,
                    )
                    state = _state(
                        identity,
                        status="running",
                        last_step=trainer.step,
                        best_validation_loss=best_loss,
                        best_checkpoint=best_checkpoint,
                    )
                    _write_state(staging, state)
        except KeyboardInterrupt:
            checkpoint_path = _save_step_checkpoint(staging, trainer)
            _prune_checkpoints(
                staging,
                keep_last=training_config.keep_last_checkpoints,
                best_checkpoint=best_checkpoint,
            )
            state = _state(
                identity,
                status="interrupted",
                last_step=trainer.step,
                best_validation_loss=best_loss,
                best_checkpoint=best_checkpoint,
            )
            _write_state(staging, state)
            print(f"training interrupted; resume from {checkpoint_path}", flush=True)
            raise
        finally:
            if handles_signals:
                signal.signal(signal.SIGTERM, previous_sigterm)

        if last_evaluation_step != trainer.step:
            last_evaluation = _evaluate(
                model,
                tokenized_validation,
                tokenizer,
                training_config.batch_size,
                training_config.sequence_length,
                evaluation_config.max_batches,
            )
            _append_metric(
                staging / METRICS_FILE,
                {
                    "type": "validation",
                    "step": trainer.step,
                    **asdict(last_evaluation),
                },
            )
        if last_evaluation is None:
            raise RuntimeError("final evaluation was not produced")
        final_step_checkpoint = _save_step_checkpoint(staging, trainer)
        if best_loss is None or last_evaluation.loss < best_loss:
            best_loss = last_evaluation.loss
            best_checkpoint = final_step_checkpoint.relative_to(staging).as_posix()
            _write_best(staging, trainer.step, best_loss, best_checkpoint)
        _prune_checkpoints(
            staging,
            keep_last=training_config.keep_last_checkpoints,
            best_checkpoint=best_checkpoint,
        )
        save_checkpoint(staging / "checkpoint.pt", trainer)
        bundle_fingerprint = save_bundle(
            staging / "bundle",
            model,
            tokenizer,
            provenance=provenance,
        )
        result = ExperimentResult(
            output=destination,
            bundle_fingerprint=bundle_fingerprint,
            dataset_fingerprint=processed.manifest.output_fingerprint,
            tokenizer_fingerprint=tokenizer.fingerprint,
            training_steps=trainer.step,
            evaluation=last_evaluation,
        )
        atomic_write_text(
            staging / "run.json",
            json.dumps(
                {
                    "source_revision": source_revision,
                    "training_run_id": training_run_id,
                    "bundle_fingerprint": result.bundle_fingerprint,
                    "dataset_fingerprint": result.dataset_fingerprint,
                    "tokenizer_fingerprint": result.tokenizer_fingerprint,
                    "initial_bundle_fingerprint": initial_bundle_fingerprint,
                    "training_steps": result.training_steps,
                    "best_checkpoint": best_checkpoint,
                    "best_validation_loss": best_loss,
                    "config_sha256": config_checksums,
                    "evaluation": asdict(result.evaluation),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )
        state = _state(
            identity,
            status="running",
            last_step=trainer.step,
            best_validation_loss=best_loss,
            best_checkpoint=best_checkpoint,
        )
        _write_state(staging, state)
        os.replace(staging, destination)
        state["status"] = "completed"
        _write_state(destination, state)
        return result
    except KeyboardInterrupt:
        raise
    except BaseException:
        if prepared and trainer is not None:
            try:
                failed = dict(state)
                failed["status"] = "failed"
                failed["last_step"] = trainer.step
                _write_state(staging, failed)
            except OSError:
                pass
        elif new_run and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def _initialize_staging(
    staging: Path,
    *,
    input_manifest: DatasetManifest,
    processed: PreprocessingResult,
    tokenizer: ByteBPETokenizer,
    config_paths: Mapping[str, Path],
) -> None:
    atomic_write_text(
        staging / "input-manifest.json",
        json.dumps(input_manifest.to_dict(), indent=2, sort_keys=True) + "\n",
    )
    processed.manifest.write(staging / "processed-manifest.json")
    save_tokenizer(tokenizer, staging / "tokenizer.json")
    for name, path in config_paths.items():
        target = staging / "configs" / f"{name}.yaml"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path.expanduser().resolve(), target)


def _split(dataset: Dataset, split: DatasetSplit) -> Dataset:
    records = tuple(record for record in dataset if record.metadata.get("split") == split.value)
    if not records:
        raise ValueError(f"processed dataset has no {split.value} records")
    return Dataset(dataset.metadata, records, dataset.schema)


def _evaluate(
    model: GPTDecoder,
    dataset: Dataset,
    tokenizer: ByteBPETokenizer,
    batch_size: int,
    sequence_length: int,
    max_batches: int,
) -> EvaluationResult:
    return evaluate_model(
        model,
        iter_token_batches(
            dataset,
            batch_size=batch_size,
            sequence_length=sequence_length,
            separator_token_id=tokenizer.vocabulary.eos_id,
            drop_last=False,
        ),
        max_batches=max_batches,
    )


def _save_step_checkpoint(staging: Path, trainer: Trainer) -> Path:
    path = staging / "checkpoints" / f"step-{trainer.step:08d}.pt"
    save_checkpoint(path, trainer)
    return path


def _validated_resume_checkpoint(staging: Path, requested: Path) -> Path:
    checkpoint = requested.expanduser().resolve()
    root = (staging / "checkpoints").resolve()
    if checkpoint.parent != root or not checkpoint.is_file():
        raise ValueError("resume checkpoint must belong to the experiment staging directory")
    latest = max(root.glob("step-*.pt"), default=None)
    if latest is None or checkpoint != latest.resolve():
        raise ValueError("resume requires the latest experiment checkpoint")
    return checkpoint


def _prune_checkpoints(
    staging: Path,
    *,
    keep_last: int,
    best_checkpoint: str | None,
) -> None:
    checkpoints = sorted((staging / "checkpoints").glob("step-*.pt"))
    keep = set(checkpoints[-keep_last:])
    if best_checkpoint is not None:
        keep.add(staging / best_checkpoint)
    for checkpoint in checkpoints:
        if checkpoint not in keep:
            checkpoint.unlink()


def _append_metric(path: Path, metric: Mapping[str, object]) -> None:
    encoded = json.dumps(
        dict(metric),
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{encoded}\n")
        stream.flush()
        os.fsync(stream.fileno())


def _truncate_metrics(path: Path, step: int) -> None:
    if not path.exists():
        return
    retained: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        if not isinstance(value, dict) or not isinstance(value.get("step"), int):
            raise ValueError("training metrics contain an invalid record")
        if value["step"] <= step:
            retained.append(line)
    atomic_write_text(path, "".join(f"{line}\n" for line in retained))


def _write_best(staging: Path, step: int, loss: float, checkpoint: str) -> None:
    atomic_write_text(
        staging / BEST_FILE,
        json.dumps(
            {"checkpoint": checkpoint, "loss": loss, "step": step},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
    )


def _state(
    identity: Mapping[str, object],
    *,
    status: str,
    last_step: int,
    best_validation_loss: float | None = None,
    best_checkpoint: str | None = None,
) -> dict[str, object]:
    return {
        "version": RUN_STATE_VERSION,
        **identity,
        "status": status,
        "last_step": last_step,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint": best_checkpoint,
    }


def _write_state(staging: Path, state: Mapping[str, object]) -> None:
    atomic_write_text(
        staging / STATE_FILE,
        json.dumps(dict(state), indent=2, sort_keys=True) + "\n",
    )


def _load_state(path: Path) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to load experiment state: {error}") from error
    fields = {
        "version",
        "source_revision",
        "training_run_id",
        "dataset_fingerprint",
        "tokenizer_fingerprint",
        "initial_bundle_fingerprint",
        "config_sha256",
        "status",
        "last_step",
        "best_validation_loss",
        "best_checkpoint",
    }
    legacy_fields = fields - {"initial_bundle_fingerprint"}
    if not isinstance(value, dict) or (set(value) != fields and set(value) != legacy_fields):
        raise ValueError("experiment state fields are invalid")
    value.setdefault("initial_bundle_fingerprint", None)
    if value["version"] != RUN_STATE_VERSION:
        raise ValueError("unsupported experiment state version")
    if value["status"] not in {"running", "interrupted", "failed"}:
        raise ValueError("experiment state status is invalid")
    if not isinstance(value["last_step"], int) or value["last_step"] < 0:
        raise ValueError("experiment state step is invalid")
    if any(
        not isinstance(value[field], str) or not value[field]
        for field in (
            "source_revision",
            "training_run_id",
            "dataset_fingerprint",
            "tokenizer_fingerprint",
        )
    ):
        raise ValueError("experiment state identity is invalid")
    checksums = value["config_sha256"]
    if (
        not isinstance(checksums, dict)
        or not checksums
        or not all(
            isinstance(key, str) and key and isinstance(checksum, str) and len(checksum) == 64
            for key, checksum in checksums.items()
        )
    ):
        raise ValueError("experiment state configuration checksums are invalid")
    _optional_number(value["best_validation_loss"])
    _optional_string(value["best_checkpoint"])
    _optional_string(value["initial_bundle_fingerprint"])
    return value


def _validate_identity(
    state: Mapping[str, object],
    identity: Mapping[str, object],
) -> None:
    mismatched = sorted(key for key, value in identity.items() if state.get(key) != value)
    if mismatched:
        raise ValueError(f"resume identity does not match: {', '.join(mismatched)}")


def _optional_number(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError("best validation loss is invalid")
    return float(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError("best checkpoint is invalid")
    return value


def _interrupt(_signum: int, _frame: FrameType | None) -> None:
    raise KeyboardInterrupt


def main() -> None:
    """Run or resume an end-to-end local experiment."""
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
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--init-bundle", type=Path)
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
        resume=arguments.resume,
        init_bundle=arguments.init_bundle,
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
