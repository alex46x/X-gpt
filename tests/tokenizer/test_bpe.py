from datetime import UTC, datetime

import pytest

from project_genesis.datasets import Dataset, DatasetMetadata, DatasetRecord
from project_genesis.tokenizer import (
    SpecialTokens,
    TokenizerConfig,
    evaluate_tokenizer,
    tokenize_dataset,
    train_tokenizer,
)

NOW = datetime(2026, 7, 19, tzinfo=UTC)
SPECIALS = SpecialTokens("<pad>", "<bos>", "<eos>", "<unk>")


def dataset(*texts: str) -> Dataset:
    metadata = DatasetMetadata("corpus", "1.0.0", "MIT", NOW)
    records = tuple(
        DatasetRecord.create(
            text=text,
            language="en",
            source=f"{index}.txt",
            license="MIT",
            document_id=str(index),
            created_at=NOW,
        )
        for index, text in enumerate(texts)
    )
    return Dataset(metadata, records)


def config(
    vocab_size: int = 266, *, add_bos: bool = False, add_eos: bool = False
) -> TokenizerConfig:
    return TokenizerConfig(vocab_size, 1, SPECIALS, add_bos, add_eos)


def test_training_is_deterministic_with_stable_special_and_byte_ids() -> None:
    corpus = dataset("hello hello", "hello world")

    first = train_tokenizer(corpus, config())
    second = train_tokenizer(corpus, config())

    assert first.tokenizer.fingerprint == second.tokenizer.fingerprint
    assert first.report == second.report
    assert first.tokenizer.vocabulary.pad_id == 0
    assert first.tokenizer.vocabulary.bos_id == 1
    assert first.tokenizer.vocabulary.eos_id == 2
    assert first.tokenizer.vocabulary.unk_id == 3
    assert first.tokenizer.vocabulary.bytes_for(4 + ord("A")) == b"A"


def test_pair_frequency_ties_choose_the_lowest_token_ids() -> None:
    corpus = dataset("ab ac", "ab ac")
    trained = train_tokenizer(corpus, config(vocab_size=261)).tokenizer

    assert trained.merges[0].left == 4 + ord("a")
    assert trained.merges[0].right == 4 + ord("b")


def test_encoding_round_trips_unseen_unicode_and_special_literals() -> None:
    tokenizer = train_tokenizer(dataset("training text"), config()).tokenizer
    text = "🙂 café\n<bos>"

    encoded = tokenizer.encode(text)

    assert tokenizer.decode(encoded) == text
    assert tokenizer.vocabulary.bos_id not in encoded
    with pytest.raises(ValueError, match="outside vocabulary"):
        tokenizer.decode((len(tokenizer.vocabulary),))


def test_bos_eos_and_dataset_tokenization_are_explicit() -> None:
    corpus = dataset("hello world")
    tokenizer = train_tokenizer(
        corpus,
        config(add_bos=True, add_eos=True),
    ).tokenizer

    encoded = tokenizer.encode("hello")
    tokenized = tokenize_dataset(corpus, tokenizer)

    assert encoded[0] == tokenizer.vocabulary.bos_id
    assert encoded[-1] == tokenizer.vocabulary.eos_id
    assert tokenizer.decode(encoded) == "hello"
    assert tokenizer.decode(encoded, skip_special_tokens=False).startswith("<bos>")
    assert tokenized[0].text == corpus[0].text
    assert tokenized[0].checksum == corpus[0].checksum
    assert tokenized[0].token_ids == tokenizer.encode(corpus[0].text)


def test_quality_report_measures_round_trip_and_compression() -> None:
    corpus = dataset("hello hello", "hello world")
    tokenizer = train_tokenizer(corpus, config()).tokenizer

    report = evaluate_tokenizer(tokenizer, corpus)

    assert report.documents == 2
    assert report.roundtrip_failures == 0
    assert report.tokens > 0
    assert report.bytes_per_token > 1
