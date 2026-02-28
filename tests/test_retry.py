"""Tests for RetryConfig integration and retry behavior in the Ollama provider."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from amplifier_core import (
    InvalidRequestError,
    NotFoundError,
    ProviderUnavailableError,
)
from amplifier_core.message_models import ChatRequest, Message
from amplifier_core.utils.retry import RetryConfig
from amplifier_core import ModuleCoordinator
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

from amplifier_module_provider_ollama import OllamaProvider


# ── Helpers ──────────────────────────────────────────────────────────


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider(**overrides) -> OllamaProvider:
    """Create a provider with fast retry config for testing."""
    defaults = {
        "max_retries": 3,
        "min_retry_delay": 0.01,
        "max_retry_delay": 1.0,
    }
    defaults.update(overrides)
    provider = OllamaProvider(host="http://localhost:11434", config=defaults)
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


# ── TestRetryConfigAttribute ────────────────────────────────────────


class TestRetryConfigAttribute:
    """Verify that OllamaProvider constructs a RetryConfig from config."""

    def test_retry_config_exists_and_is_retry_config(self):
        provider = _make_provider()
        assert hasattr(provider, "_retry_config")
        assert isinstance(provider._retry_config, RetryConfig)

    def test_retry_config_values_from_config(self):
        provider = _make_provider(
            max_retries=7,
            min_retry_delay=2.0,
            max_retry_delay=120.0,
            retry_jitter=0.3,
        )
        assert provider._retry_config.max_retries == 7
        assert provider._retry_config.min_delay == 2.0
        assert provider._retry_config.max_delay == 120.0
        assert provider._retry_config.jitter == 0.3

    def test_retry_config_defaults(self):
        provider = OllamaProvider(host="http://localhost:11434", config={})
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        assert provider._retry_config.max_retries == 3
        assert provider._retry_config.min_delay == 1.0
        assert provider._retry_config.max_delay == 60.0
        assert provider._retry_config.jitter == 0.2

    def test_retry_jitter_bool_backward_compat(self):
        # True maps to 0.2
        p1 = _make_provider(retry_jitter=True)
        assert p1._retry_config.jitter == 0.2

        # False maps to 0.0
        p2 = _make_provider(retry_jitter=False)
        assert p2._retry_config.jitter == 0.0


# ── TestOldRetryCodeRemoved ─────────────────────────────────────────


@pytest.mark.xfail(reason="Old retry code removed in Task 4", strict=True)
class TestOldRetryCodeRemoved:
    """Verify that legacy retry constants and methods are removed."""

    def test_no_max_retries_class_var(self):
        assert not hasattr(OllamaProvider, "MAX_RETRIES")

    def test_no_base_retry_delay_class_var(self):
        assert not hasattr(OllamaProvider, "BASE_RETRY_DELAY")

    def test_no_private_retry_with_backoff_method(self):
        provider = _make_provider()
        assert not hasattr(provider, "_retry_with_backoff")


# ── TestRetryBehavior ───────────────────────────────────────────────


class TestRetryBehavior:
    """Verify retry behavior through the complete() call path."""

    def test_5xx_error_is_retried(self):
        """THE PRIMARY BUG FIX: 500 ResponseError should be retried."""
        provider = _make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, _mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        assert provider.client.chat.await_count == 2

    def test_5xx_exhausted_raises_provider_unavailable(self):
        """After exhausting retries on 500, raises ProviderUnavailableError."""
        provider = _make_provider(max_retries=1)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ProviderUnavailableError) as exc_info:
                asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.retryable is True
        assert exc_info.value.status_code == 500
        assert provider.client.chat.await_count == 2

    def test_400_error_not_retried(self):
        """400 errors should raise immediately without retry."""
        provider = _make_provider(max_retries=3)
        err = ResponseError("bad request")
        err.status_code = 400
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(InvalidRequestError):
                asyncio.run(provider.complete(_simple_request()))

        assert provider.client.chat.await_count == 1

    def test_404_error_not_retried(self):
        """404 errors should raise immediately without retry."""
        provider = _make_provider(max_retries=3)
        err = ResponseError("model not found")
        err.status_code = 404
        provider.client.chat = AsyncMock(side_effect=err)

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(NotFoundError):
                asyncio.run(provider.complete(_simple_request()))

        assert provider.client.chat.await_count == 1

    def test_429_error_is_retried(self):
        """429 (rate limit) should be retried."""
        provider = _make_provider(max_retries=2)
        err = ResponseError("rate limit exceeded")
        err.status_code = 429
        provider.client.chat = AsyncMock(side_effect=[err, _mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        assert provider.client.chat.await_count == 2

    def test_timeout_error_is_retried(self):
        """TimeoutError should be retried."""
        provider = _make_provider(max_retries=2)
        provider.client.chat = AsyncMock(
            side_effect=[asyncio.TimeoutError(), _mock_response()]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        assert provider.client.chat.await_count == 2

    def test_connection_error_is_retried(self):
        """ConnectionError should be retried."""
        provider = _make_provider(max_retries=2)
        provider.client.chat = AsyncMock(
            side_effect=[ConnectionError("refused"), _mock_response()]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = asyncio.run(provider.complete(_simple_request()))

        assert result is not None
        assert provider.client.chat.await_count == 2


# ── TestRetryEventEmission ──────────────────────────────────────────


class TestRetryEventEmission:
    """Verify that provider:retry events are emitted on retry."""

    def test_provider_retry_event_emitted(self):
        """A single retry should emit one provider:retry event."""
        provider = _make_provider(max_retries=2)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, _mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(provider.complete(_simple_request()))

        hooks = provider.coordinator.hooks  # type: ignore[union-attr]
        retry_events = [(n, p) for n, p in hooks.events if n == "provider:retry"]
        assert len(retry_events) == 1

        _, payload = retry_events[0]
        assert payload["provider"] == "ollama"
        assert payload["attempt"] == 1
        assert payload["max_retries"] == 2
        assert "delay" in payload
        assert payload["error_type"] == "ProviderUnavailableError"
        assert "error_message" in payload

    def test_multiple_retry_events(self):
        """Multiple retries should emit incrementing attempt events."""
        provider = _make_provider(max_retries=4)
        err = ResponseError("internal server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=[err, err, err, _mock_response()])

        with patch("asyncio.sleep", new_callable=AsyncMock):
            asyncio.run(provider.complete(_simple_request()))

        hooks = provider.coordinator.hooks  # type: ignore[union-attr]
        retry_events = [(n, p) for n, p in hooks.events if n == "provider:retry"]
        assert len(retry_events) == 3

        attempts = [p["attempt"] for _, p in retry_events]
        assert attempts == [1, 2, 3]
