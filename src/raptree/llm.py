"""A minimal chat-completion provider interface.

Only two things in this project talk to an LLM: the optional abstractive summariser and the
optional LLM reader. Both go through :class:`ChatProvider`, which has an OpenAI-compatible
HTTP adapter (OpenAI, Ollama, vLLM, LM Studio all serve ``/v1/chat/completions``) and a
deterministic fake for tests and offline runs.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Sequence
from typing import Protocol, TypedDict

import httpx


class ChatMessage(TypedDict):
    role: str
    content: str


class ChatProvider(Protocol):
    name: str

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> str: ...


class ProviderError(RuntimeError):
    pass


class OpenAICompatibleProvider:
    """POSTs to ``{base_url}/chat/completions`` with temperature 0."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), headers=headers, timeout=timeout, transport=transport
        )
        self.model = model
        self.name = f"openai-compatible:{model}"

    @classmethod
    def from_env(cls) -> OpenAICompatibleProvider:
        """Configure from RAPTREE_LLM_BASE_URL, RAPTREE_LLM_MODEL and RAPTREE_LLM_API_KEY."""
        base_url = os.environ.get("RAPTREE_LLM_BASE_URL")
        model = os.environ.get("RAPTREE_LLM_MODEL")
        if not base_url or not model:
            raise ProviderError(
                "set RAPTREE_LLM_BASE_URL (e.g. http://localhost:11434/v1) and RAPTREE_LLM_MODEL"
            )
        return cls(base_url, model, api_key=os.environ.get("RAPTREE_LLM_API_KEY"))

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        try:
            response = self._client.post("/chat/completions", json=payload)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"chat completion failed: {exc}") from exc
        if not isinstance(content, str):
            raise ProviderError("chat completion returned no text")
        return content


Responder = Callable[[Sequence[ChatMessage]], str]


def _echo_responder(messages: Sequence[ChatMessage]) -> str:
    """First 40 words of the last message, prefixed by a digest so outputs are traceable."""
    last = messages[-1]["content"]
    digest = hashlib.sha256(last.encode("utf-8")).hexdigest()[:8]
    return f"[{digest}] " + " ".join(last.split()[:40])


class FakeProvider:
    """Deterministic provider that records every call; no network."""

    def __init__(self, responder: Responder = _echo_responder) -> None:
        self._responder = responder
        self.calls: list[list[ChatMessage]] = []
        self.name = "fake"

    def complete(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> str:
        self.calls.append(list(messages))
        return self._responder(messages)
