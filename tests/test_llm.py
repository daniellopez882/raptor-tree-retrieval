from __future__ import annotations

import json

import httpx
import pytest

from raptree.llm import FakeProvider, OpenAICompatibleProvider, ProviderError


def test_openai_adapter_posts_a_chat_completion() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "B"}}]})

    provider = OpenAICompatibleProvider(
        "http://localhost:11434/v1/",
        "local-model",
        api_key="test-key",
        transport=httpx.MockTransport(handler),
    )
    reply = provider.complete([{"role": "user", "content": "hi"}], max_tokens=5)
    assert reply == "B"
    request = seen[0]
    assert str(request.url) == "http://localhost:11434/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "local-model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, text="boom"),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": [{"message": {"content": None}}]}),
        httpx.Response(200, text="not json"),
    ],
)
def test_openai_adapter_turns_failures_into_provider_errors(response: httpx.Response) -> None:
    provider = OpenAICompatibleProvider(
        "http://localhost:8000/v1", "m", transport=httpx.MockTransport(lambda r: response)
    )
    with pytest.raises(ProviderError):
        provider.complete([{"role": "user", "content": "x"}], max_tokens=3)


def test_from_env_requires_url_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAPTREE_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("RAPTREE_LLM_MODEL", raising=False)
    with pytest.raises(ProviderError, match="RAPTREE_LLM_BASE_URL"):
        OpenAICompatibleProvider.from_env()
    monkeypatch.setenv("RAPTREE_LLM_BASE_URL", "http://localhost:1234/v1")
    monkeypatch.setenv("RAPTREE_LLM_MODEL", "some-model")
    assert OpenAICompatibleProvider.from_env().model == "some-model"


def test_fake_provider_is_deterministic_and_records_calls() -> None:
    fake = FakeProvider()
    messages = [{"role": "user", "content": "one two three"}]
    first = fake.complete(messages, max_tokens=10)
    assert first == fake.complete(messages, max_tokens=10)
    assert first.endswith("one two three")
    assert len(fake.calls) == 2
