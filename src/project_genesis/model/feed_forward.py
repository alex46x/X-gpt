"""Position-wise transformer feed-forward network."""

from torch import Tensor, nn
from torch.nn import functional as functional


class FeedForward(nn.Module):
    """Expand, activate, contract, and regularize hidden states."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        *,
        bias: bool = True,
        activation: str = "gelu",
    ) -> None:
        """Initialize the two learned projections."""
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if activation not in {"gelu", "swiglu"}:
            raise ValueError("activation must be 'gelu' or 'swiglu'")
        self.d_model = d_model
        self.activation = activation
        self.input_projection = nn.Linear(
            d_model,
            d_ff if activation == "gelu" else 2 * d_ff,
            bias=bias,
        )
        self.output_projection = nn.Linear(d_ff, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        """Transform ``(..., d_model)`` hidden states."""
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"final dimension must be {self.d_model}")
        projected = self.input_projection(inputs)
        if self.activation == "swiglu":
            gate, value = projected.chunk(2, dim=-1)
            hidden = functional.silu(gate) * value
        else:
            hidden = functional.gelu(projected, approximate="none")
        output: Tensor = self.dropout(self.output_projection(hidden))
        return output
