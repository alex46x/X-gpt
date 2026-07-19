"""Autoregressive decoder inference."""

from project_genesis.inference.bundle import (
    InferenceBundle,
    load_bundle,
    save_bundle,
)
from project_genesis.inference.config import (
    GenerationConfig,
    load_generation_config,
)
from project_genesis.inference.generation import (
    FinishReason,
    GenerationResult,
    generate,
    sample_next_token,
)

__all__ = [
    "FinishReason",
    "GenerationConfig",
    "GenerationResult",
    "InferenceBundle",
    "generate",
    "load_generation_config",
    "load_bundle",
    "sample_next_token",
    "save_bundle",
]
