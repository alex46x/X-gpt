"""Tokenizer round-trip and compression quality metrics."""

from dataclasses import dataclass

from project_genesis.datasets import Dataset
from project_genesis.tokenizer.model import ByteBPETokenizer


@dataclass(frozen=True, slots=True)
class TokenizerQualityReport:
    """Corpus-level reversible encoding and compression measurements."""

    documents: int
    utf8_bytes: int
    tokens: int
    roundtrip_failures: int

    @property
    def bytes_per_token(self) -> float:
        """Return average UTF-8 bytes represented by each token."""
        return self.utf8_bytes / self.tokens if self.tokens else 0.0


def evaluate_tokenizer(
    tokenizer: ByteBPETokenizer,
    dataset: Dataset,
) -> TokenizerQualityReport:
    """Measure exact round trips and token compression over a dataset."""
    utf8_bytes = tokens = failures = 0
    for record in dataset:
        encoded = tokenizer.encode(record.text, add_bos=False, add_eos=False)
        utf8_bytes += len(record.text.encode("utf-8"))
        tokens += len(encoded)
        failures += tokenizer.decode(encoded) != record.text
    return TokenizerQualityReport(len(dataset), utf8_bytes, tokens, failures)
