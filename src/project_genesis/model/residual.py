"""Residual connection validation and addition."""

from torch import Tensor


def residual_add(inputs: Tensor, update: Tensor) -> Tensor:
    """Add a residual update without broadcasting or device/type promotion."""
    if inputs.shape != update.shape:
        raise ValueError("residual tensors must have identical shapes")
    if inputs.dtype != update.dtype:
        raise TypeError("residual tensors must have identical dtypes")
    if inputs.device != update.device:
        raise ValueError("residual tensors must be on the same device")
    return inputs + update
