import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from project_genesis.datasets import Dataset, DatasetMetadata, DatasetRecord
from project_genesis.tokenizer import (
    ByteBPETokenizer,
    SpecialTokens,
    TokenizerConfig,
    load_tokenizer,
    save_tokenizer,
    train_tokenizer,
)


def tokenizer_fixture() -> ByteBPETokenizer:
    created_at = datetime(2026, 7, 19, tzinfo=UTC)
    dataset = Dataset(
        DatasetMetadata("corpus", "1.0.0", "MIT", created_at),
        (
            DatasetRecord.create(
                text="repeat repeat",
                language="en",
                source="source.txt",
                license="MIT",
                document_id="1",
                created_at=created_at,
            ),
        ),
    )
    config = TokenizerConfig(
        266,
        1,
        SpecialTokens("<pad>", "<bos>", "<eos>", "<unk>"),
        False,
        False,
    )
    return train_tokenizer(dataset, config).tokenizer


def test_tokenizer_storage_round_trips(tmp_path: Path) -> None:
    tokenizer = tokenizer_fixture()
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)

    loaded = load_tokenizer(path)

    assert loaded == tokenizer
    assert loaded.fingerprint == tokenizer.fingerprint
    assert loaded.decode(loaded.encode("unseen 🙂")) == "unseen 🙂"


def test_tokenizer_storage_rejects_tampering_and_unknown_fields(tmp_path: Path) -> None:
    tokenizer = tokenizer_fixture()
    path = tmp_path / "tokenizer.json"
    save_tokenizer(tokenizer, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["fingerprint"] = "0" * 64
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_tokenizer(path)

    save_tokenizer(tokenizer, path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["unknown"] = True
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown=\\['unknown'\\]"):
        load_tokenizer(path)
