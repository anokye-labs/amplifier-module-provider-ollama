"""Tests for the raw_debug guard: raw events require BOTH debug and raw_debug flags."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio(loop_scope="function")
async def test_raw_events_suppressed_when_debug_false(
    make_provider, simple_request, mock_response
):
    """When debug=False and raw_debug=True, no :raw events should fire."""
    provider = make_provider(debug=False, raw_debug=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request())

    raw_events = [
        name for name, _ in provider.coordinator.hooks.events if name.endswith(":raw")
    ]
    assert raw_events == [], f"Expected no :raw events but got {raw_events}"


@pytest.mark.asyncio(loop_scope="function")
async def test_raw_events_suppressed_when_raw_debug_false(
    make_provider, simple_request, mock_response
):
    """When debug=True and raw_debug=False, no :raw events should fire."""
    provider = make_provider(debug=True, raw_debug=False)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request())

    raw_events = [
        name for name, _ in provider.coordinator.hooks.events if name.endswith(":raw")
    ]
    assert raw_events == [], f"Expected no :raw events but got {raw_events}"


@pytest.mark.asyncio(loop_scope="function")
async def test_raw_events_fire_when_both_debug_and_raw_debug(
    make_provider, simple_request, mock_response
):
    """When both debug=True and raw_debug=True, :raw events should fire."""
    provider = make_provider(debug=True, raw_debug=True)
    provider.client.chat = AsyncMock(return_value=mock_response())

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await provider.complete(simple_request())

    raw_events = [
        name for name, _ in provider.coordinator.hooks.events if name.endswith(":raw")
    ]
    assert len(raw_events) >= 2, (
        f"Expected at least 2 :raw events (request + response) but got {raw_events}"
    )
    assert "llm:request:raw" in raw_events
    assert "llm:response:raw" in raw_events
