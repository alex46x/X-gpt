"""Deterministic next-token batch construction."""

from collections import deque
from collections.abc import Iterable, Iterator

import torch
from torch import Tensor

from project_genesis.datasets import DatasetRecord

type TokenBatch = tuple[Tensor, Tensor]


def iter_token_batches(
    records: Iterable[DatasetRecord],
    *,
    batch_size: int,
    sequence_length: int,
    separator_token_id: int,
    drop_last: bool = True,
) -> Iterator[TokenBatch]:
    """Pack tokenized records and yield shifted input/target batches."""
    if batch_size <= 0 or sequence_length <= 0:
        raise ValueError("batch_size and sequence_length must be positive")
    if separator_token_id < 0:
        raise ValueError("separator_token_id cannot be negative")

    pending: deque[int] = deque()
    tokens_per_batch = batch_size * sequence_length
    for record in records:
        if record.token_ids is None:
            raise ValueError(f"record {record.document_id!r} has not been tokenized")
        pending.extend(record.token_ids)
        pending.append(separator_token_id)
        while len(pending) > tokens_per_batch:
            yield _take_batch(pending, batch_size, sequence_length)

    if not drop_last:
        remaining_rows = min(batch_size, (len(pending) - 1) // sequence_length)
        if remaining_rows:
            yield _take_batch(pending, remaining_rows, sequence_length)


def _take_batch(
    pending: deque[int],
    batch_size: int,
    sequence_length: int,
) -> TokenBatch:
    token_count = batch_size * sequence_length
    values = [pending.popleft() for _ in range(token_count)]
    values.append(pending[0])
    tokens = torch.tensor(values, dtype=torch.long)
    inputs = tokens[:-1].view(batch_size, sequence_length)
    targets = tokens[1:].view(batch_size, sequence_length)
    return inputs, targets
