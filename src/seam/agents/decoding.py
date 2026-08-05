"""Ollama client wrapper with retry logic and structured logging."""

from __future__ import annotations

import logging
import time
from typing import Any

import ollama

from seam.orchestration.config_loader import ModelConfig

logger = logging.getLogger(__name__)


class OllamaClientError(RuntimeError):
    """Raised when all retry attempts for an Ollama call have been exhausted."""


class OllamaClient:
    """Thin wrapper around the Ollama Python client.

    All calls are retried with exponential backoff up to
    ``config.retry_attempts`` times.  Every call is logged with model name,
    prompt length, response length, and wall-clock latency.

    Args:
        config: A :class:`ModelConfig` instance providing model name, URL, and
            generation parameters.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        self._client = ollama.Client(host=config.base_url)
        logger.info(
            "OllamaClient initialised — model=%s base_url=%s",
            config.model_name,
            config.base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def complete(self, prompt: str) -> tuple[str, int]:
        """Send a completion request to Ollama.

        Args:
            prompt: The raw prompt string.

        Returns:
            A ``(response_text, latency_ms)`` tuple.

        Raises:
            OllamaClientError: If all retry attempts fail.
        """
        options: dict[str, Any] = {
            "temperature": self._config.temperature,
            "top_p": self._config.top_p,
            "num_predict": self._config.max_tokens,
        }
        if self._config.seed is not None:
            options["seed"] = self._config.seed

        return self._retry_call(self._do_complete, prompt, options)

    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text*.

        Args:
            text: Text to embed.

        Returns:
            A list of floats representing the embedding.

        Raises:
            OllamaClientError: If all retry attempts fail.
        """
        result, _ = self._retry_call(self._do_embed, text, {})
        return result  # type: ignore[return-value]

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Returns:
            True if the server responds to a list-models call, False otherwise.
        """
        try:
            self._client.list()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama server not reachable: %s", exc)
            return False

    def get_model_info(self) -> dict[str, Any]:
        """Retrieve metadata about the configured model from Ollama.

        Returns:
            A dict with model metadata, or an empty dict on failure.
        """
        try:
            response = self._client.show(self._config.model_name)
            # ollama >=0.3 returns a ShowResponse object; convert to dict
            if hasattr(response, "model_dump"):
                return response.model_dump()  # type: ignore[return-value]
            return dict(response)  # type: ignore[call-overload]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not retrieve model info for %s: %s", self._config.model_name, exc)
            return {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_complete(self, prompt: str, options: dict[str, Any]) -> tuple[str, int]:
        """Execute a single completion call and return (text, latency_ms)."""
        start = time.perf_counter()
        response = self._client.generate(
            model=self._config.model_name,
            prompt=prompt,
            options=options,
            stream=False,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        response_text: str = response.response  # type: ignore[union-attr]
        logger.info(
            "complete model=%s prompt_len=%d response_len=%d latency_ms=%d",
            self._config.model_name,
            len(prompt),
            len(response_text),
            latency_ms,
        )
        return response_text, latency_ms

    def _do_embed(self, text: str, _options: dict[str, Any]) -> tuple[list[float], int]:
        """Execute a single embed call and return (embedding, latency_ms)."""
        start = time.perf_counter()
        response = self._client.embeddings(model=self._config.model_name, prompt=text)
        latency_ms = int((time.perf_counter() - start) * 1000)
        embedding: list[float] = response.embedding  # type: ignore[union-attr]
        logger.info(
            "embed model=%s text_len=%d embedding_dim=%d latency_ms=%d",
            self._config.model_name,
            len(text),
            len(embedding),
            latency_ms,
        )
        return embedding, latency_ms

    def _retry_call(
        self,
        fn: Any,
        arg: str,
        options: dict[str, Any],
    ) -> Any:
        """Call *fn(arg, options)* with exponential backoff.

        Args:
            fn: Callable to invoke.
            arg: First argument (prompt or text).
            options: Second argument (generation options dict).

        Returns:
            The return value of *fn* on success.

        Raises:
            OllamaClientError: If all ``config.retry_attempts`` are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, self._config.retry_attempts + 1):
            try:
                return fn(arg, options)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = 2 ** (attempt - 1)  # 1s, 2s, 4s …
                logger.warning(
                    "Ollama call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt,
                    self._config.retry_attempts,
                    exc,
                    wait,
                )
                if attempt < self._config.retry_attempts:
                    time.sleep(wait)
        raise OllamaClientError(
            f"Ollama call failed after {self._config.retry_attempts} attempts"
        ) from last_exc
