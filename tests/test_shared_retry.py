"""Tests for shared retry_with_backoff integration (Ollama).

Verifies that the Ollama provider uses the shared RetryConfig and
retry_with_backoff from amplifier-core instead of its own _retry_with_backoff,
and adopts new error types (AccessDeniedError for 403).
"""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.llm_errors import AccessDeniedError, ProviderUnavailableError
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils.retry import RetryConfig
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

from amplifier_module_provider_ollama import OllamaProvider, _translate_ollama_error


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider(**overrides) -> OllamaProvider:
    provider = OllamaProvider(host="http://localhost:11434", config=overrides)
    provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
    return provider


def _simple_request(**kwargs) -> ChatRequest:
    return ChatRequest(
        messages=[Message(role="user", content="hello")],
        **kwargs,
    )


def _mock_response(content: str = "ok"):
    """Minimal successful Ollama response dict."""
    return {
        "message": {"role": "assistant", "content": content},
        "done": True,
        "model": "llama3.2:3b",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }


# --- Structural: uses shared RetryConfig ---


def test_provider_has_retry_config():
    """Provider should store a RetryConfig instance."""
    provider = OllamaProvider(host="http://localhost:11434")
    assert hasattr(provider, "_retry_config")
    assert isinstance(provider._retry_config, RetryConfig)


def test_retry_config_respects_config_values():
    """RetryConfig should be populated from provider config dict."""
    provider = OllamaProvider(
        host="http://localhost:11434",
        config={
            "max_retries": 5,
            "min_retry_delay": 2.0,
            "max_retry_delay": 120.0,
        },
    )
    assert provider._retry_config.max_retries == 5
    assert provider._retry_config.min_delay == 2.0
    assert provider._retry_config.max_delay == 120.0


def test_no_old_retry_with_backoff_method():
    """Old _retry_with_backoff instance method should be removed."""
    provider = OllamaProvider(host="http://localhost:11434")
    assert not hasattr(provider, "_retry_with_backoff")


def test_no_old_retry_constants():
    """Old MAX_RETRIES and BASE_RETRY_DELAY class constants should be removed."""
    assert not hasattr(OllamaProvider, "MAX_RETRIES")
    assert not hasattr(OllamaProvider, "BASE_RETRY_DELAY")


# --- Error type: 403 -> AccessDeniedError ---


def test_translate_403_becomes_access_denied_error():
    """_translate_ollama_error: 403 -> AccessDeniedError (not AuthenticationError)."""
    err = ResponseError("forbidden")
    err.status_code = 403
    result = _translate_ollama_error(err)
    assert isinstance(result, AccessDeniedError)
    assert result.provider == "ollama"
    assert result.status_code == 403


def test_403_through_complete():
    """complete() should raise AccessDeniedError for 403 ResponseError."""
    provider = _make_provider()
    err = ResponseError("forbidden")
    err.status_code = 403
    provider.client.chat = AsyncMock(side_effect=err)

    with pytest.raises(AccessDeniedError) as exc_info:
        asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.provider == "ollama"
    assert exc_info.value.__cause__ is err


# --- Retry behavior through shared utility ---


def test_connection_error_retried_with_shared_utility():
    """ConnectionError should be retried via shared retry_with_backoff."""
    provider = _make_provider(max_retries=2, min_retry_delay=0.01, max_retry_delay=0.1)

    # Fail twice with ConnectionError, then succeed
    provider.client.chat = AsyncMock(
        side_effect=[
            ConnectionError("refused"),
            ConnectionError("refused"),
            _mock_response(),
        ]
    )

    with patch("asyncio.sleep", new_callable=AsyncMock):
        result = asyncio.run(provider.complete(_simple_request()))

    assert result is not None
    assert provider.client.chat.await_count == 3

    # Should have emitted provider:retry events
    retry_events = [
        e for e in provider.coordinator.hooks.events if e[0] == "provider:retry"
    ]
    assert len(retry_events) == 2
    assert retry_events[0][1]["provider"] == "ollama"


def test_connection_error_exhausts_retries():
    """After exhausting retries, ConnectionError should raise ProviderUnavailableError."""
    provider = _make_provider(max_retries=2, min_retry_delay=0.01, max_retry_delay=0.1)
    provider.client.chat = AsyncMock(side_effect=ConnectionError("refused"))

    with patch("asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

    assert exc_info.value.retryable is True
    # 1 initial + 2 retries = 3 total
    assert provider.client.chat.await_count == 3


def test_response_error_not_retried():
    """ResponseError (API-level errors) should NOT be retried."""
    provider = _make_provider(max_retries=3, min_retry_delay=0.01)
    err = ResponseError("bad request")
    err.status_code = 400
    provider.client.chat = AsyncMock(side_effect=err)

    from amplifier_core.llm_errors import InvalidRequestError

    with pytest.raises(InvalidRequestError):
        asyncio.run(provider.complete(_simple_request()))

    # Should have called API exactly once (no retry for non-retryable API errors)
    assert provider.client.chat.await_count == 1
