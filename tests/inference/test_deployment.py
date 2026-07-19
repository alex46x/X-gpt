from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from torch import Tensor, nn

from project_genesis.inference import (
    GenerationConfig,
    InferenceBundle,
    load_bundle,
    save_bundle,
)
from project_genesis.inference.service import ServiceRuntime, create_app
from project_genesis.model import GPTDecoder, ModelConfig
from project_genesis.tokenizer import ByteBPETokenizer, SpecialTokens, Vocabulary


def _tokenizer() -> ByteBPETokenizer:
    special = SpecialTokens("<pad>", "<bos>", "<eos>", "<unk>")
    return ByteBPETokenizer(
        Vocabulary(
            special,
            (None,) * 4 + tuple(bytes((value,)) for value in range(256)),
        ),
        (),
    )


def _model(*, context_length: int = 16) -> GPTDecoder:
    return GPTDecoder(
        ModelConfig(
            vocab_size=260,
            context_length=context_length,
            d_model=8,
            n_heads=2,
            d_ff=16,
            dropout=0.0,
            bias=True,
            layer_norm_epsilon=1e-5,
            n_layers=1,
        )
    )


class ConstantModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ModelConfig(260, 128, 4, 1, 8, 0.0, True, 1e-5, 1)
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        logits = torch.zeros(
            *inputs.shape,
            self.config.vocab_size,
            device=inputs.device,
        )
        logits[..., 69] = 1
        return logits


def test_bundle_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    model = _model()
    fingerprint = save_bundle(destination, model, _tokenizer())

    loaded = load_bundle(destination)

    assert loaded.fingerprint == fingerprint
    assert loaded.tokenizer.fingerprint == _tokenizer().fingerprint
    for expected, actual in zip(
        model.parameters(),
        loaded.model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(expected, actual)

    with (destination / "model.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum does not match"):
        load_bundle(destination)


def test_bundle_refuses_to_overwrite_existing_artifacts(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    save_bundle(destination, _model(), _tokenizer())

    with pytest.raises(FileExistsError):
        save_bundle(destination, _model(), _tokenizer())


def test_http_health_auth_generation_chat_and_limits() -> None:
    runtime = ServiceRuntime(
        InferenceBundle(  # type: ignore[arg-type]
            ConstantModel(),
            _tokenizer(),
            "a" * 64,
            "0.1.0",
        ),
        GenerationConfig(1, 0.0, 0, 1.0, 1.0, (69,), False),
    )
    application = create_app(
        runtime,
        api_key="secret",
        max_prompt_characters=20,
        max_new_tokens=2,
    )

    with TestClient(application) as client:
        health = client.get("/healthz")
        ready = client.get("/readyz")
        unauthorized = client.post("/v1/generate", json={"prompt": "x"})
        generated = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer secret"},
            json={"prompt": "x"},
        )
        chatted = client.post(
            "/v1/chat",
            headers={"Authorization": "Bearer secret"},
            json={"messages": [{"role": "user", "content": "Write code"}]},
        )
        oversized = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer secret"},
            json={"prompt": "x" * 21},
        )
        excessive_tokens = client.post(
            "/v1/generate",
            headers={"Authorization": "Bearer secret"},
            json={"prompt": "x", "max_new_tokens": 3},
        )

    assert health.json() == {"status": "ok"}
    assert ready.json()["bundle_fingerprint"] == "a" * 64
    assert unauthorized.status_code == 401
    assert generated.status_code == 200
    assert generated.json()["text"] == "A"
    assert generated.headers["x-request-id"]
    assert chatted.status_code == 200
    assert chatted.json()["text"] == "A"
    assert oversized.status_code == 413
    assert excessive_tokens.status_code == 400

    body_limited = create_app(runtime, max_request_bytes=10)
    with TestClient(body_limited) as client:
        assert client.post("/v1/generate", json={"prompt": "payload"}).status_code == 413


def test_deployment_image_is_non_root_and_health_checked() -> None:
    dockerfile = Path("deployment/Dockerfile").read_text(encoding="utf-8")

    assert "USER genesis" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'ENTRYPOINT ["genesis-serve"]' in dockerfile
