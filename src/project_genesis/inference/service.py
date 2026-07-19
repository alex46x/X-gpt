"""Bounded FastAPI service for Project Genesis inference bundles."""

import hmac
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Annotated

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from project_genesis.chat import Conversation, Message, Role, generate_reply
from project_genesis.inference.bundle import InferenceBundle, load_bundle
from project_genesis.inference.config import (
    GenerationConfig,
    load_generation_config,
)
from project_genesis.inference.generation import generate

LOGGER = logging.getLogger("project_genesis.service")


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """Environment-owned artifact, device, authentication, and request limits."""

    bundle_path: Path
    inference_config_path: Path
    device: str
    api_key: str | None
    max_request_bytes: int
    max_prompt_characters: int
    max_new_tokens: int

    @classmethod
    def from_environment(cls) -> "ServiceSettings":
        """Load required deployment settings from process environment."""
        bundle = os.environ.get("GENESIS_BUNDLE")
        if not bundle:
            raise ValueError("GENESIS_BUNDLE must identify an inference bundle")
        api_key = os.environ.get("GENESIS_API_KEY")
        if api_key is not None and not api_key.strip():
            raise ValueError("GENESIS_API_KEY cannot be empty")
        return cls(
            bundle_path=Path(bundle),
            inference_config_path=Path(
                os.environ.get(
                    "GENESIS_INFERENCE_CONFIG",
                    "configs/inference/default.yaml",
                )
            ),
            device=os.environ.get("GENESIS_DEVICE", "cpu"),
            api_key=api_key,
            max_request_bytes=_positive_environment_integer(
                "GENESIS_MAX_REQUEST_BYTES",
                262_144,
            ),
            max_prompt_characters=_positive_environment_integer(
                "GENESIS_MAX_PROMPT_CHARACTERS",
                32_768,
            ),
            max_new_tokens=_positive_environment_integer(
                "GENESIS_MAX_NEW_TOKENS",
                512,
            ),
        )


class ServiceRuntime:
    """Loaded inference artifacts and serialized model execution."""

    def __init__(
        self,
        bundle: InferenceBundle,
        generation_config: GenerationConfig,
    ) -> None:
        """Store immutable artifacts and one process-local inference lock."""
        self.bundle = bundle
        self.generation_config = generation_config
        # ponytail: global lock; use process replicas or continuous batching at saturation.
        self.lock = threading.Lock()


class SamplingOptions(BaseModel):
    """Per-request bounded generation overrides."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_new_tokens: Annotated[int | None, Field(default=None, ge=1)] = None
    temperature: Annotated[float | None, Field(default=None, ge=0)] = None
    top_k: Annotated[int | None, Field(default=None, ge=0)] = None
    top_p: Annotated[float | None, Field(default=None, gt=0, le=1)] = None
    repetition_penalty: Annotated[
        float | None,
        Field(default=None, ge=1),
    ] = None
    seed: Annotated[int | None, Field(default=None, ge=0)] = None


class GenerateRequest(SamplingOptions):
    """Text completion request."""

    prompt: Annotated[str, Field(min_length=1)]


class ChatMessage(BaseModel):
    """One role-tagged service chat message."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Role
    content: Annotated[str, Field(min_length=1)]


class ChatRequest(SamplingOptions):
    """Stateless conversation request ending in a user message."""

    messages: Annotated[list[ChatMessage], Field(min_length=1)]


class InferenceResponse(BaseModel):
    """Generated text, exact token suffix, finish reason, and bundle identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    generated_token_ids: tuple[int, ...]
    finish_reason: str
    bundle_fingerprint: str


def create_app(
    runtime: ServiceRuntime | None = None,
    *,
    api_key: str | None = None,
    max_request_bytes: int = 262_144,
    max_prompt_characters: int = 32_768,
    max_new_tokens: int = 512,
) -> FastAPI:
    """Create the HTTP application, loading environment artifacts at startup."""
    if max_request_bytes <= 0 or max_prompt_characters <= 0 or max_new_tokens <= 0:
        raise ValueError("service request limits must be positive")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if runtime is None:
            settings = ServiceSettings.from_environment()
            application.state.runtime = ServiceRuntime(
                load_bundle(settings.bundle_path, device=settings.device),
                load_generation_config(settings.inference_config_path),
            )
            application.state.api_key = settings.api_key
            application.state.max_request_bytes = settings.max_request_bytes
            application.state.max_prompt_characters = settings.max_prompt_characters
            application.state.max_new_tokens = settings.max_new_tokens
        yield

    application = FastAPI(
        title="Project Genesis",
        version="0.1.0",
        lifespan=lifespan,
    )
    if runtime is not None:
        application.state.runtime = runtime
        application.state.api_key = api_key
        application.state.max_request_bytes = max_request_bytes
        application.state.max_prompt_characters = max_prompt_characters
        application.state.max_new_tokens = max_new_tokens

    @application.middleware("http")
    async def observe(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status_code = 500
        response: Response
        try:
            content_length = request.headers.get("content-length")
            request_limit: int = request.app.state.max_request_bytes
            if content_length is not None and (
                not content_length.isdigit() or int(content_length) > request_limit
            ):
                response = JSONResponse(
                    {"detail": "request body is too large"},
                    status_code=413,
                )
                status_code = response.status_code
                response.headers["X-Request-ID"] = request_id
                return response
            configured_key: str | None = getattr(
                request.app.state,
                "api_key",
                None,
            )
            if configured_key is not None and request.url.path.startswith("/v1/"):
                supplied = request.headers.get("authorization", "")
                expected = f"Bearer {configured_key}"
                if not hmac.compare_digest(supplied, expected):
                    response = JSONResponse(
                        {"detail": "invalid bearer token"},
                        status_code=401,
                    )
                else:
                    response = await call_next(request)
            else:
                response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            LOGGER.info(
                json.dumps(
                    {
                        "event": "http_request",
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status_code,
                        "duration_ms": round(
                            (time.perf_counter() - started) * 1000,
                            3,
                        ),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )

    @application.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz")
    def readiness(request: Request) -> dict[str, str]:
        if not hasattr(request.app.state, "runtime"):
            raise HTTPException(status_code=503, detail="model is not loaded")
        return {
            "status": "ready",
            "bundle_fingerprint": _runtime(request).bundle.fingerprint,
        }

    @application.post("/v1/generate", response_model=InferenceResponse)
    def complete(request: Request, body: GenerateRequest) -> InferenceResponse:
        _validate_character_count(request, len(body.prompt))
        current = _runtime(request)
        config = _request_config(request, current.generation_config, body)
        prompt_token_ids = current.bundle.tokenizer.encode(
            body.prompt,
            add_bos=True,
            add_eos=False,
        )
        try:
            with current.lock:
                result = generate(
                    current.bundle.model,
                    prompt_token_ids,
                    config,
                    generator=_generator(current, body.seed),
                )
            text = current.bundle.tokenizer.decode(result.generated_token_ids)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return InferenceResponse(
            text=text,
            generated_token_ids=result.generated_token_ids,
            finish_reason=result.finish_reason.value,
            bundle_fingerprint=current.bundle.fingerprint,
        )

    @application.post("/v1/chat", response_model=InferenceResponse)
    def chat(request: Request, body: ChatRequest) -> InferenceResponse:
        total_characters = sum(len(message.content) for message in body.messages)
        _validate_character_count(request, total_characters)
        messages = tuple(Message(message.role, message.content) for message in body.messages)
        if messages[-1].role is not Role.USER:
            raise HTTPException(
                status_code=400,
                detail="chat messages must end with a user message",
            )
        try:
            conversation = Conversation(messages[:-1])
            current = _runtime(request)
            config = _request_config(request, current.generation_config, body)
            with current.lock:
                updated, result = generate_reply(
                    current.bundle.model,
                    current.bundle.tokenizer,
                    conversation,
                    messages[-1].content,
                    config,
                    generator=_generator(current, body.seed),
                )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return InferenceResponse(
            text=updated.messages[-1].content,
            generated_token_ids=result.generated_token_ids,
            finish_reason=result.finish_reason.value,
            bundle_fingerprint=current.bundle.fingerprint,
        )

    return application


def _runtime(request: Request) -> ServiceRuntime:
    runtime: object = getattr(request.app.state, "runtime", None)
    if not isinstance(runtime, ServiceRuntime):
        raise HTTPException(status_code=503, detail="model is not loaded")
    return runtime


def _validate_character_count(request: Request, character_count: int) -> None:
    limit: int = request.app.state.max_prompt_characters
    if character_count > limit:
        raise HTTPException(status_code=413, detail="prompt is too large")


def _request_config(
    request: Request,
    base: GenerationConfig,
    options: SamplingOptions,
) -> GenerationConfig:
    max_new_tokens = options.max_new_tokens or base.max_new_tokens
    limit: int = request.app.state.max_new_tokens
    if max_new_tokens > limit:
        raise HTTPException(
            status_code=400,
            detail=f"max_new_tokens cannot exceed {limit}",
        )
    return replace(
        base,
        max_new_tokens=max_new_tokens,
        temperature=(base.temperature if options.temperature is None else options.temperature),
        top_k=base.top_k if options.top_k is None else options.top_k,
        top_p=base.top_p if options.top_p is None else options.top_p,
        repetition_penalty=(
            base.repetition_penalty
            if options.repetition_penalty is None
            else options.repetition_penalty
        ),
    )


def _generator(
    runtime: ServiceRuntime,
    seed: int | None,
) -> torch.Generator | None:
    if seed is None:
        return None
    device = next(runtime.bundle.model.parameters()).device
    return torch.Generator(device=device).manual_seed(seed)


def _positive_environment_integer(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def main() -> None:
    """Run the Project Genesis ASGI service with Uvicorn."""
    logging.basicConfig(
        level=os.environ.get("GENESIS_LOG_LEVEL", "INFO"),
        format="%(message)s",
    )
    uvicorn.run(
        app,
        host=os.environ.get("GENESIS_HOST", "0.0.0.0"),
        port=_positive_environment_integer("GENESIS_PORT", 8000),
        proxy_headers=False,
        server_header=False,
    )


app = create_app()
