"""
Pytest configuration and shared test fixtures for Ollama provider tests.

Behavioral tests use inheritance from amplifier-core base classes.
See tests/test_behavioral.py for the inherited tests.

The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- provider_module, tool_module, etc.: Mounted module instances
"""

from typing import Any, cast

import pytest
from amplifier_core import ModuleCoordinator
from amplifier_core.message_models import ChatRequest, Message

from amplifier_module_provider_ollama import OllamaProvider


class FakeHooks:
    """Minimal hooks implementation that records emitted events."""

    def __init__(self):
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit(self, name: str, payload: dict[str, Any]) -> None:
        self.events.append((name, payload))


class FakeCoordinator:
    """Minimal coordinator wired to FakeHooks."""

    def __init__(self):
        self.hooks = FakeHooks()


@pytest.fixture
def make_provider():
    """Factory fixture: create an OllamaProvider wired to a FakeCoordinator."""

    def _factory(**config_overrides) -> OllamaProvider:
        provider = OllamaProvider(
            host="http://localhost:11434", config=config_overrides
        )
        provider.coordinator = cast(ModuleCoordinator, FakeCoordinator())
        return provider

    return _factory


@pytest.fixture
def simple_request():
    """Factory fixture: create a minimal ChatRequest."""

    def _factory(**kwargs) -> ChatRequest:
        return ChatRequest(
            messages=[Message(role="user", content="hello")],
            **kwargs,
        )

    return _factory


@pytest.fixture
def mock_response():
    """Factory fixture: create a minimal successful Ollama response dict."""

    def _factory(content: str = "ok") -> dict:
        return {
            "message": {"role": "assistant", "content": content},
            "done": True,
            "model": "llama3.2:3b",
            "prompt_eval_count": 10,
            "eval_count": 5,
        }

    return _factory
