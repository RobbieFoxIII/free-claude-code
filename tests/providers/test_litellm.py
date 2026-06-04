"""Tests for the LiteLLM proxy provider."""

from unittest.mock import MagicMock, patch

from providers.base import ProviderConfig
from providers.litellm import LITELLM_DEFAULT_BASE, LiteLLMProvider


class MockMessage:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content


class MockRequest:
    model = "cerebras-fast"
    messages = [MockMessage("user", "Hello")]
    max_tokens = 100
    temperature = 0.5
    top_p = 0.9
    system = "System prompt"
    stop_sequences = None
    tools = []
    thinking = MagicMock(enabled=False)


def test_init_uses_configured_proxy_base_url() -> None:
    config = ProviderConfig(
        api_key="test-litellm-key",
        base_url="http://litellm.internal:4000/v1",
    )
    with patch("providers.openai_compat.AsyncOpenAI") as mock_openai:
        provider = LiteLLMProvider(config)

    assert provider._api_key == "test-litellm-key"
    assert provider._base_url == "http://litellm.internal:4000/v1"
    mock_openai.assert_called_once()


def test_default_base_url_is_local_proxy() -> None:
    assert LITELLM_DEFAULT_BASE == "http://127.0.0.1:4000/v1"


def test_build_request_body_preserves_litellm_alias() -> None:
    provider = LiteLLMProvider(
        ProviderConfig(api_key="test-litellm-key", base_url=LITELLM_DEFAULT_BASE)
    )
    body = provider._build_request_body(MockRequest())

    assert body["model"] == "cerebras-fast"
    assert body["messages"][0]["role"] == "system"
