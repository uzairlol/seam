"""Unit tests for OllamaClient with mocked Ollama API calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seam.agents.decoding import OllamaClient, OllamaClientError
from seam.orchestration.config_loader import ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    return ModelConfig(
        model_name="qwen2.5:7b-instruct",
        base_url="http://localhost:11434",
        temperature=0.0,
        seed=42,
        max_tokens=256,
        top_p=1.0,
        request_timeout=10,
        retry_attempts=3,
    )


@patch("ollama.Client")
def test_ollama_client_complete_success(mock_ollama_cls: MagicMock, model_config: ModelConfig) -> None:
    """Test successful text completion using OllamaClient."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.response = "Target move: harvest"
    mock_instance.generate.return_value = mock_response
    mock_ollama_cls.return_value = mock_instance

    client = OllamaClient(model_config)
    text, latency = client.complete("State: [0, 0]. Action?")

    assert text == "Target move: harvest"
    assert latency >= 0
    mock_instance.generate.assert_called_once_with(
        model="qwen2.5:7b-instruct",
        prompt="State: [0, 0]. Action?",
        options={
            "temperature": 0.0,
            "top_p": 1.0,
            "num_predict": 256,
            "seed": 42,
        },
        stream=False,
    )


@patch("ollama.Client")
def test_ollama_client_embed_success(mock_ollama_cls: MagicMock, model_config: ModelConfig) -> None:
    """Test embedding generation using OllamaClient."""
    mock_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.embedding = [0.1, 0.2, 0.3, 0.4]
    mock_instance.embeddings.return_value = mock_response
    mock_ollama_cls.return_value = mock_instance

    client = OllamaClient(model_config)
    vector = client.embed("Sample memory text")

    assert vector == [0.1, 0.2, 0.3, 0.4]
    mock_instance.embeddings.assert_called_once_with(
        model="qwen2.5:7b-instruct",
        prompt="Sample memory text",
    )


@patch("ollama.Client")
def test_ollama_client_is_available(mock_ollama_cls: MagicMock, model_config: ModelConfig) -> None:
    """Test is_available returns True when server responds and False when it fails."""
    mock_instance = MagicMock()
    mock_instance.list.return_value = {"models": []}
    mock_ollama_cls.return_value = mock_instance

    client = OllamaClient(model_config)
    assert client.is_available() is True

    mock_instance.list.side_effect = Exception("Connection refused")
    assert client.is_available() is False


@patch("ollama.Client")
def test_ollama_client_get_model_info(mock_ollama_cls: MagicMock, model_config: ModelConfig) -> None:
    """Test get_model_info returns metadata dict."""
    mock_instance = MagicMock()
    mock_instance.show.return_value = {"details": {"format": "gguf"}}
    mock_ollama_cls.return_value = mock_instance

    client = OllamaClient(model_config)
    info = client.get_model_info()
    assert info == {"details": {"format": "gguf"}}


@patch("ollama.Client")
@patch("time.sleep", return_value=None)
def test_ollama_client_retry_and_exhaustion(
    _mock_sleep: MagicMock,
    mock_ollama_cls: MagicMock,
    model_config: ModelConfig,
) -> None:
    """Test retry mechanism raises OllamaClientError when attempts are exhausted."""
    mock_instance = MagicMock()
    mock_instance.generate.side_effect = Exception("Timeout error")
    mock_ollama_cls.return_value = mock_instance

    client = OllamaClient(model_config)

    with pytest.raises(OllamaClientError) as exc_info:
        client.complete("Test prompt")

    assert "Ollama call failed after 3 attempts" in str(exc_info.value)
    assert mock_instance.generate.call_count == 3
