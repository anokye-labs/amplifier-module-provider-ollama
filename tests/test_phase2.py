"""Tests for Phase 2: error translation, reasoning_effort, and usage notes."""

import asyncio
from typing import cast
from unittest.mock import AsyncMock

import pytest
from amplifier_core import (
    AuthenticationError,
    ContentFilterError,
    ContextLengthError,
    InvalidRequestError,
    LLMError,
    LLMTimeoutError,
    NotFoundError,
    ProviderUnavailableError,
    RateLimitError,
)
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message
from ollama import ResponseError  # pyright: ignore[reportAttributeAccessIssue]

from amplifier_module_provider_ollama import OllamaProvider, _translate_ollama_error


# ── Helpers ──────────────────────────────────────────────────────────────


class FakeHooks:
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    async def emit(self, name: str, payload: dict) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    def __init__(self):
        self.hooks = FakeHooks()


def _make_provider(**overrides) -> OllamaProvider:
    """Create a provider wired to a FakeCoordinator with chat mocked out."""
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


# ── _translate_ollama_error unit tests ───────────────────────────────────


class TestTranslateOllamaError:
    """Unit tests for the standalone error translation helper."""

    def test_response_error_401(self):
        err = ResponseError("unauthorized")
        err.status_code = 401
        result = _translate_ollama_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "ollama"
        assert result.status_code == 401

    def test_response_error_403(self):
        err = ResponseError("forbidden")
        err.status_code = 403
        result = _translate_ollama_error(err)
        assert isinstance(result, AuthenticationError)
        assert result.provider == "ollama"
        assert result.status_code == 403

    def test_response_error_429(self):
        err = ResponseError("rate limit exceeded")
        err.status_code = 429
        result = _translate_ollama_error(err)
        assert isinstance(result, RateLimitError)
        assert result.provider == "ollama"
        assert result.status_code == 429

    def test_response_error_400(self):
        err = ResponseError("bad request")
        err.status_code = 400
        result = _translate_ollama_error(err)
        assert isinstance(result, InvalidRequestError)
        assert result.provider == "ollama"
        assert result.status_code == 400

    def test_response_error_400_context_length(self):
        err = ResponseError("context length exceeded")
        err.status_code = 400
        result = _translate_ollama_error(err)
        assert isinstance(result, ContextLengthError)
        assert result.provider == "ollama"
        assert result.status_code == 400

    def test_response_error_400_content_filter(self):
        err = ResponseError("content blocked by safety filter")
        err.status_code = 400
        result = _translate_ollama_error(err)
        assert isinstance(result, ContentFilterError)
        assert result.provider == "ollama"
        assert result.status_code == 400

    def test_response_error_404(self):
        err = ResponseError("model not found")
        err.status_code = 404
        result = _translate_ollama_error(err)
        assert isinstance(result, NotFoundError)
        assert result.provider == "ollama"
        assert result.status_code == 404
        assert result.retryable is False

    def test_response_error_404_is_not_retryable(self):
        err = ResponseError("not found")
        err.status_code = 404
        result = _translate_ollama_error(err)
        assert result.retryable is False

    def test_response_error_500(self):
        err = ResponseError("internal server error")
        err.status_code = 500
        result = _translate_ollama_error(err)
        assert isinstance(result, ProviderUnavailableError)
        assert result.provider == "ollama"
        assert result.status_code == 500

    def test_response_error_503(self):
        err = ResponseError("service unavailable")
        err.status_code = 503
        result = _translate_ollama_error(err)
        assert isinstance(result, ProviderUnavailableError)
        assert result.status_code == 503

    def test_response_error_other_status(self):
        err = ResponseError("something else")
        err.status_code = 418
        result = _translate_ollama_error(err)
        assert isinstance(result, LLMError)
        assert result.retryable is True
        assert result.provider == "ollama"

    def test_connection_error(self):
        result = _translate_ollama_error(ConnectionError("refused"))
        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is True
        assert result.provider == "ollama"

    def test_os_error(self):
        result = _translate_ollama_error(OSError("network down"))
        assert isinstance(result, ProviderUnavailableError)
        assert result.retryable is True

    def test_timeout_error(self):
        result = _translate_ollama_error(asyncio.TimeoutError())
        assert isinstance(result, LLMTimeoutError)
        assert result.provider == "ollama"

    def test_generic_exception(self):
        result = _translate_ollama_error(RuntimeError("boom"))
        assert isinstance(result, LLMError)
        assert result.retryable is True
        assert result.provider == "ollama"


# ── Error translation integration (through complete()) ──────────────────


class TestErrorTranslationIntegration:
    """Verify that complete() raises translated kernel errors."""

    def test_timeout_raises_llm_timeout_error(self):
        provider = _make_provider()
        provider.client.chat = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(LLMTimeoutError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.provider == "ollama"
        assert exc_info.value.__cause__ is not None

    def test_response_error_401_raises_authentication_error(self):
        provider = _make_provider()
        err = ResponseError("unauthorized")
        err.status_code = 401
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(AuthenticationError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.provider == "ollama"
        assert exc_info.value.__cause__ is err

    def test_response_error_400_raises_invalid_request(self):
        provider = _make_provider()
        err = ResponseError("bad request")
        err.status_code = 400
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(InvalidRequestError):
            asyncio.run(provider.complete(_simple_request()))

    def test_response_error_500_raises_provider_unavailable(self):
        provider = _make_provider()
        err = ResponseError("server error")
        err.status_code = 500
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(ProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.status_code == 500

    def test_response_error_404_raises_not_found(self):
        provider = _make_provider()
        err = ResponseError("model not found")
        err.status_code = 404
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(NotFoundError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.status_code == 404
        assert exc_info.value.retryable is False
        assert exc_info.value.__cause__ is err

    def test_connection_error_after_retry_raises_provider_unavailable(self):
        """ConnectionError is retried by _retry_with_backoff, then translated."""
        provider = _make_provider()
        # _retry_with_backoff retries 3 times; all fail → raises last error
        provider.client.chat = AsyncMock(side_effect=ConnectionError("refused"))

        with pytest.raises(ProviderUnavailableError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.retryable is True

    def test_cause_chain_preserved(self):
        provider = _make_provider()
        original = ResponseError("original")
        original.status_code = 400
        provider.client.chat = AsyncMock(side_effect=original)

        with pytest.raises(InvalidRequestError) as exc_info:
            asyncio.run(provider.complete(_simple_request()))

        assert exc_info.value.__cause__ is original

    def test_streaming_timeout_raises_llm_timeout_error(self):
        provider = _make_provider()
        provider.client.chat = AsyncMock(side_effect=asyncio.TimeoutError())

        with pytest.raises(LLMTimeoutError) as exc_info:
            asyncio.run(provider.complete(_simple_request(stream=True)))

        assert exc_info.value.provider == "ollama"

    def test_streaming_response_error_raises_translated(self):
        provider = _make_provider()
        err = ResponseError("forbidden")
        err.status_code = 403
        provider.client.chat = AsyncMock(side_effect=err)

        with pytest.raises(AuthenticationError):
            asyncio.run(provider.complete(_simple_request(stream=True)))


# ── reasoning_effort support ─────────────────────────────────────────────


class TestReasoningEffort:
    """Verify reasoning_effort on ChatRequest enables thinking."""

    def test_reasoning_effort_enables_thinking_for_thinking_model(self):
        """Non-None reasoning_effort should pass effort level to think param."""
        provider = _make_provider(
            default_model="deepseek-r1:14b", enable_thinking=False
        )
        provider.client.chat = AsyncMock(return_value=_mock_response())

        request = _simple_request(reasoning_effort="high")
        asyncio.run(provider.complete(request, model="deepseek-r1:14b"))

        call_kwargs = provider.client.chat.call_args
        # Ollama v0.9.0+ supports effort levels — value is passed through directly
        assert (
            call_kwargs.kwargs.get("think") == "high"
            or call_kwargs[1].get("think") == "high"
        )

    def test_reasoning_effort_ignored_for_non_thinking_model(self):
        """reasoning_effort should have no effect on non-thinking models."""
        provider = _make_provider(default_model="llama3.2:3b", enable_thinking=False)
        provider.client.chat = AsyncMock(return_value=_mock_response())

        request = _simple_request(reasoning_effort="medium")
        asyncio.run(provider.complete(request, model="llama3.2:3b"))

        call_kwargs = provider.client.chat.call_args
        # think should not be in the params at all
        assert "think" not in (call_kwargs.kwargs or {})

    def test_reasoning_effort_none_falls_through_to_config(self):
        """When reasoning_effort is None, existing config controls thinking."""
        provider = _make_provider(
            default_model="qwen3:8b",
            enable_thinking=True,
            thinking_effort="low",
        )
        provider.client.chat = AsyncMock(return_value=_mock_response())

        request = _simple_request()  # reasoning_effort defaults to None
        asyncio.run(provider.complete(request, model="qwen3:8b"))

        call_kwargs = provider.client.chat.call_args
        # Should use config effort "low" (not True from reasoning_effort)
        assert (
            call_kwargs.kwargs.get("think") == "low"
            or call_kwargs[1].get("think") == "low"
        )

    def test_enable_thinking_takes_precedence_over_reasoning_effort(self):
        """request.enable_thinking (kwargs path) has higher priority."""
        provider = _make_provider(
            default_model="qwen3:8b",
            enable_thinking=False,
            thinking_effort="high",
        )
        provider.client.chat = AsyncMock(return_value=_mock_response())

        # Simulate enable_thinking on request (existing kwargs path)
        request = _simple_request(reasoning_effort="medium")
        # Manually set enable_thinking to test precedence
        request.enable_thinking = True  # type: ignore[attr-defined]
        asyncio.run(provider.complete(request, model="qwen3:8b"))

        call_kwargs = provider.client.chat.call_args
        # Should use config's thinking_effort "high" (from enable_thinking path)
        assert (
            call_kwargs.kwargs.get("think") == "high"
            or call_kwargs[1].get("think") == "high"
        )

    def test_reasoning_effort_low_passes_through(self):
        """'low' reasoning_effort is passed through as effort level."""
        provider = _make_provider(
            default_model="deepseek-r1:14b", enable_thinking=False
        )
        provider.client.chat = AsyncMock(return_value=_mock_response())

        request = _simple_request(reasoning_effort="low")
        asyncio.run(provider.complete(request, model="deepseek-r1:14b"))

        call_kwargs = provider.client.chat.call_args
        # Ollama v0.9.0+ supports effort levels — value is passed through directly
        assert (
            call_kwargs.kwargs.get("think") == "low"
            or call_kwargs[1].get("think") == "low"
        )

    def test_streaming_reasoning_effort_passes_through(self):
        """reasoning_effort should pass effort level through in streaming path."""
        provider = _make_provider(
            default_model="deepseek-r1:14b", enable_thinking=False
        )

        # Create an async iterator for streaming
        async def fake_stream():
            yield {"message": {"content": "hi"}, "done": False}
            yield {
                "message": {"content": ""},
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 2,
                "model": "deepseek-r1:14b",
            }

        provider.client.chat = AsyncMock(return_value=fake_stream())

        request = _simple_request(stream=True, reasoning_effort="high")
        asyncio.run(provider.complete(request, model="deepseek-r1:14b"))

        call_kwargs = provider.client.chat.call_args
        # Ollama v0.9.0+ supports effort levels — value is passed through directly
        assert (
            call_kwargs.kwargs.get("think") == "high"
            or call_kwargs[1].get("think") == "high"
        )
