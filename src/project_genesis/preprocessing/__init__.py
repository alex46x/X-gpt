"""Local source parsing, text normalization, filtering, and quality reports."""

from project_genesis.preprocessing.config import (
    ErrorPolicy,
    PreprocessingConfig,
    load_preprocessing_config,
)
from project_genesis.preprocessing.pipeline import (
    PreprocessingError,
    PreprocessingResult,
    ProcessedDatasetManifest,
    QualityReport,
    normalize_text,
    preprocess_dataset,
)
from project_genesis.preprocessing.readers import RawDocument, ReadFailure, read_source

__all__ = [
    "ErrorPolicy",
    "PreprocessingConfig",
    "PreprocessingError",
    "PreprocessingResult",
    "ProcessedDatasetManifest",
    "QualityReport",
    "RawDocument",
    "ReadFailure",
    "load_preprocessing_config",
    "normalize_text",
    "preprocess_dataset",
    "read_source",
]
