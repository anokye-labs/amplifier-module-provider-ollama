"""Tests for OllamaProvider.close() method."""

from unittest.mock import MagicMock

import pytest

from amplifier_module_provider_ollama import OllamaProvider


@pytest.mark.asyncio(loop_scope="function")
async def test_close_nils_client_when_initialized():
    """close() should set _client to None when it was previously set."""
    provider = OllamaProvider(host="http://localhost:11434")
    provider._client = MagicMock()
    await provider.close()
    assert provider._client is None


@pytest.mark.asyncio(loop_scope="function")
async def test_close_is_safe_when_client_is_none():
    """close() should not crash when _client is already None."""
    provider = OllamaProvider(host="http://localhost:11434")
    assert provider._client is None  # Confirm starts as None
    await provider.close()  # Should not raise


@pytest.mark.asyncio(loop_scope="function")
async def test_close_can_be_called_twice():
    """close() should be idempotent — calling twice should not crash."""
    provider = OllamaProvider(host="http://localhost:11434")
    provider._client = MagicMock()
    await provider.close()
    await provider.close()
    assert provider._client is None
