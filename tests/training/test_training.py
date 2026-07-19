from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from project_genesis.datasets import DatasetRecord
from project_genesis.model import GPTDecoder, ModelConfig
from project_genesis.training import (
    Precision,
    Trainer,
    TrainingConfig,
    iter_token_batches,
    load_checkpoint,
    load_training_config,
    save_checkpoint,
    seed_training,
)


def _training_config(
    *,
    accumulation_steps: int = 1,
    warmup_steps: int = 0,
) -> TrainingConfig:
    return TrainingConfig(
        batch_size=1,
        sequence_length=4,
        learning_rate=0.01,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.95,
        epsilon=1e-8,
        warmup_steps=warmup_steps,
        max_steps=6,
        min_learning_rate_ratio=0.1,
        gradient_accumulation_steps=accumulation_steps,
        max_gradient_norm=100.0,
        precision=Precision.FLOAT32,
        seed=17,
    )


def _model(*, dropout: float = 0.1) -> GPTDecoder:
    return GPTDecoder(
        ModelConfig(
            vocab_size=16,
            context_length=4,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=dropout,
            bias=True,
            layer_norm_epsilon=1e-5,
            n_layers=1,
        )
    )


def _record(document_id: str, token_ids: tuple[int, ...] | None) -> DatasetRecord:
    record = DatasetRecord.create(
        text=document_id,
        language="en",
        source="memory",
        license="MIT",
        document_id=document_id,
        created_at=datetime(2026, 7, 19, tzinfo=UTC),
    )
    return replace(record, token_ids=token_ids)


def test_default_training_config_loads_strictly() -> None:
    config = load_training_config(
        Path("configs/training/default.yaml"),
        ["training.batch_size=2", "training.precision=float32"],
    )

    assert config.batch_size == 2
    assert config.precision is Precision.FLOAT32
    assert config.warmup_steps < config.max_steps


def test_token_batching_packs_records_with_explicit_separator() -> None:
    records = (_record("a", (4, 5, 6)), _record("b", (7, 8)))

    inputs, targets = next(
        iter_token_batches(
            records,
            batch_size=2,
            sequence_length=3,
            separator_token_id=2,
        )
    )

    assert inputs.tolist() == [[4, 5, 6], [2, 7, 8]]
    assert targets.tolist() == [[5, 6, 2], [7, 8, 2]]


def test_token_batching_rejects_unencoded_records() -> None:
    with pytest.raises(ValueError, match="has not been tokenized"):
        next(
            iter_token_batches(
                [_record("raw", None)],
                batch_size=1,
                sequence_length=2,
                separator_token_id=2,
            )
        )


def test_token_batching_can_keep_complete_rows_from_the_final_partial_batch() -> None:
    batches = list(
        iter_token_batches(
            [_record("short", (4, 5, 6))],
            batch_size=2,
            sequence_length=3,
            separator_token_id=2,
            drop_last=False,
        )
    )

    assert len(batches) == 1
    assert batches[0][0].tolist() == [[4, 5, 6]]
    assert batches[0][1].tolist() == [[5, 6, 2]]


def test_scheduler_warms_up_before_cosine_decay() -> None:
    trainer = Trainer(_model(), _training_config(warmup_steps=2))
    batch = [(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))]

    assert trainer.optimizer.param_groups[0]["lr"] == 0.005
    trainer.train_step(batch)
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0075
    trainer.train_step(batch)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    trainer.train_step(batch)
    assert trainer.optimizer.param_groups[0]["lr"] < 0.01


def test_accumulated_step_matches_combined_batch() -> None:
    seed_training(4)
    accumulated_model = _model(dropout=0.0)
    combined_model = _model(dropout=0.0)
    combined_model.load_state_dict(accumulated_model.state_dict())

    first = (torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))
    second = (torch.tensor([[5, 6, 7, 8]]), torch.tensor([[6, 7, 8, 9]]))
    accumulated = Trainer(
        accumulated_model,
        _training_config(accumulation_steps=2),
    )
    combined = Trainer(combined_model, _training_config())

    accumulated.train_step([first, second])
    combined.train_step(
        [
            (
                torch.cat((first[0], second[0])),
                torch.cat((first[1], second[1])),
            )
        ]
    )

    for left, right in zip(
        accumulated.model.parameters(),
        combined.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(left, right)


def test_checkpoint_resume_reproduces_the_next_step(tmp_path: Path) -> None:
    seed_training(17)
    trainer = Trainer(_model(), _training_config())
    batch = [(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))]
    trainer.train_step(batch)
    checkpoint = tmp_path / "step.pt"
    save_checkpoint(checkpoint, trainer)

    expected_loss = trainer.train_step(batch)
    expected = {name: value.detach().clone() for name, value in trainer.model.state_dict().items()}

    resumed = Trainer(_model(), _training_config())
    load_checkpoint(checkpoint, resumed)
    actual_loss = resumed.train_step(batch)

    assert resumed.step == 2
    assert resumed.microbatches_seen == 2
    assert actual_loss == expected_loss
    for name, value in resumed.model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_float16_requires_cuda() -> None:
    with pytest.raises(ValueError, match="requires a CUDA"):
        Trainer(
            _model(),
            replace(_training_config(), precision=Precision.FLOAT16),
        )
