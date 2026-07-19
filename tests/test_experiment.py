import json
from pathlib import Path

import pytest

from project_genesis.experiment import run_experiment
from project_genesis.inference import load_bundle
from project_genesis.preflight import preflight_experiment


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_complete_experiment_publishes_loadable_artifacts(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _write(data / "train.txt", "hello world " * 20)
    _write(data / "validation.txt", "hello world " * 5)
    configs = tmp_path / "configs"
    dataset = _write(
        configs / "dataset.yaml",
        f"""
environment: test
paths:
  data: {data.as_posix()}
  cache: {(tmp_path / "cache").as_posix()}
  artifacts: {(tmp_path / "artifacts").as_posix()}
dataset:
  name: integration
  version: 1.0.0
  license: MIT
  created_at: "2026-07-19T00:00:00+00:00"
  sources:
    - path: {str((data / "train.txt").as_posix())}
      format: text
      split: train
      language: en
    - path: {str((data / "validation.txt").as_posix())}
      format: text
      split: validation
      language: en
""",
    )
    preprocessing = _write(
        configs / "preprocessing.yaml",
        """
preprocessing:
  unicode_normalization: NFKC
  normalize_newlines: true
  strip_control_characters: true
  collapse_whitespace: false
  trim: true
  min_characters: 1
  max_characters: 10000
  allowed_languages: [en]
  deduplicate: true
  on_error: raise
""",
    )
    tokenizer = _write(
        configs / "tokenizer.yaml",
        """
tokenizer:
  vocab_size: 260
  min_pair_frequency: 1
  special_tokens:
    pad: "<pad>"
    bos: "<bos>"
    eos: "<eos>"
    unk: "<unk>"
  add_bos: false
  add_eos: false
""",
    )
    model = _write(
        configs / "model.yaml",
        """
model:
  vocab_size: 260
  context_length: 8
  d_model: 8
  n_heads: 2
  d_ff: 16
  dropout: 0.0
  bias: true
  layer_norm_epsilon: 0.00001
  n_layers: 1
  initializer_range: 0.02
  tie_embeddings: true
""",
    )
    training = _write(
        configs / "training.yaml",
        """
training:
  batch_size: 1
  sequence_length: 4
  learning_rate: 0.001
  weight_decay: 0.0
  beta1: 0.9
  beta2: 0.95
  epsilon: 0.00000001
  warmup_steps: 0
  max_steps: 2
  min_learning_rate_ratio: 0.1
  gradient_accumulation_steps: 1
  max_gradient_norm: 1.0
  precision: float32
  seed: 17
""",
    )
    evaluation = _write(
        configs / "evaluation.yaml",
        """
evaluation:
  max_batches: 2
  regression:
    max_loss_increase: 0.01
    max_accuracy_drop: 0.005
    min_throughput_ratio: 0.9
""",
    )
    output = tmp_path / "run"

    preflight = preflight_experiment(
        dataset_config_path=dataset,
        preprocessing_config_path=preprocessing,
        tokenizer_config_path=tokenizer,
        model_config_path=model,
        training_config_path=training,
        evaluation_config_path=evaluation,
    )
    result = run_experiment(
        dataset_config_path=dataset,
        preprocessing_config_path=preprocessing,
        tokenizer_config_path=tokenizer,
        model_config_path=model,
        training_config_path=training,
        evaluation_config_path=evaluation,
        output=output,
        source_revision="abc123",
        training_run_id="integration-1",
    )

    loaded = load_bundle(output / "bundle")
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert preflight.ready
    assert preflight.training_files == preflight.validation_files == 1
    assert preflight.parameters > 0
    assert preflight.persistent_training_state_bytes == preflight.parameters * 16
    assert preflight.scheduled_tokens == 8
    assert result.training_steps == 2
    assert loaded.fingerprint == result.bundle_fingerprint
    assert loaded.provenance.training_run_id == "integration-1"
    assert run["dataset_fingerprint"] == result.dataset_fingerprint
    assert set(run["config_sha256"]) == {
        "dataset",
        "preprocessing",
        "tokenizer",
        "model",
        "training",
        "evaluation",
    }
    assert (output / "checkpoint.pt").is_file()
    assert (output / "tokenizer.json").is_file()
    assert (output / "configs" / "dataset.yaml").is_file()
    with pytest.raises(FileExistsError):
        run_experiment(
            dataset_config_path=dataset,
            preprocessing_config_path=preprocessing,
            tokenizer_config_path=tokenizer,
            model_config_path=model,
            training_config_path=training,
            evaluation_config_path=evaluation,
            output=output,
            source_revision="abc123",
            training_run_id="integration-1",
        )

    model.write_text(
        model.read_text(encoding="utf-8").replace("vocab_size: 260", "vocab_size: 261"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="vocabulary sizes do not match"):
        preflight_experiment(
            dataset_config_path=dataset,
            preprocessing_config_path=preprocessing,
            tokenizer_config_path=tokenizer,
            model_config_path=model,
            training_config_path=training,
            evaluation_config_path=evaluation,
        )
