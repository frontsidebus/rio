"""Tests for RIO local-inference config fields and LLM factory routing."""

from __future__ import annotations

import pytest

from orchestrator.config import Settings


# ---------------------------------------------------------------------------
# Task 1.1 — New config field defaults
# ---------------------------------------------------------------------------


class TestLocalInferenceDefaults:
    """Verify default values for RIO local-inference fields."""

    def test_llm_backend_defaults_to_anthropic(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        s = Settings(anthropic_api_key="sk-test")
        assert s.llm_backend == "anthropic"

    def test_llm_api_url_defaults_to_localhost(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("LLM_API_URL", raising=False)
        s = Settings(anthropic_api_key="sk-test")
        assert s.llm_api_url == "http://localhost:8081/v1"

    def test_llm_model_local_defaults_to_qwen(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("LLM_MODEL_LOCAL", raising=False)
        s = Settings(anthropic_api_key="sk-test")
        assert s.llm_model_local == "qwen3.5-35b-a3b-aviation"


# ---------------------------------------------------------------------------
# Task 1.1 — Environment variable population
# ---------------------------------------------------------------------------


class TestLocalInferenceEnvOverrides:
    """Verify env vars populate the new RIO fields."""

    def test_llm_backend_from_env(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_BACKEND", "local")
        s = Settings()
        assert s.llm_backend == "local"

    def test_llm_api_url_from_env(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_API_URL", "http://llm:8000/v1")
        s = Settings()
        assert s.llm_api_url == "http://llm:8000/v1"

    def test_llm_model_local_from_env(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_MODEL_LOCAL", "custom-model")
        s = Settings()
        assert s.llm_model_local == "custom-model"

    def test_llm_backend_set_to_local(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting LLM_BACKEND=local is accepted without error."""
        monkeypatch.setenv("LLM_BACKEND", "local")
        s = Settings()
        assert s.llm_backend == "local"

    def test_llm_backend_case_preserved(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The raw value is stored as-is (factory normalises later)."""
        monkeypatch.setenv("LLM_BACKEND", "LOCAL")
        s = Settings()
        assert s.llm_backend == "LOCAL"


# ---------------------------------------------------------------------------
# Task 1.2 — LLM factory integration
# ---------------------------------------------------------------------------


class TestCreateLLMClient:
    """Verify create_llm_client routes to the correct backend."""

    def test_anthropic_backend_returns_anthropic_client(
        self,
        mock_env_vars: dict[str, str],
    ) -> None:
        from orchestrator.llm import create_llm_client
        from orchestrator.llm.anthropic_client import AnthropicClient

        s = Settings()
        # Settings already has anthropic_api_key from mock_env_vars
        client = create_llm_client(s)
        assert isinstance(client, AnthropicClient)

    def test_local_backend_returns_openai_compat_client(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.llm import create_llm_client
        from orchestrator.llm.openai_compat_client import (
            OpenAICompatClient,
        )

        monkeypatch.setenv("LLM_BACKEND", "local")
        monkeypatch.setenv("LLM_API_URL", "http://llm:8000/v1")
        monkeypatch.setenv("LLM_MODEL_LOCAL", "qwen3.5-35b-a3b-aviation")
        s = Settings()
        client = create_llm_client(s)
        assert isinstance(client, OpenAICompatClient)

    def test_local_backend_passes_url_and_model(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.llm import create_llm_client

        monkeypatch.setenv("LLM_BACKEND", "local")
        monkeypatch.setenv("LLM_API_URL", "http://myhost:9000/v1")
        monkeypatch.setenv("LLM_MODEL_LOCAL", "test-model-7b")
        s = Settings()
        client = create_llm_client(s)
        # OpenAICompatClient stores model as an attribute
        assert client._model == "test-model-7b"

    def test_invalid_backend_raises_value_error(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from orchestrator.llm import create_llm_client

        monkeypatch.setenv("LLM_BACKEND", "invalid")
        s = Settings()
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            create_llm_client(s)

    def test_factory_normalises_backend_case(
        self,
        mock_env_vars: dict[str, str],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Factory lowercases+strips the backend value."""
        from orchestrator.llm import create_llm_client
        from orchestrator.llm.openai_compat_client import (
            OpenAICompatClient,
        )

        monkeypatch.setenv("LLM_BACKEND", "  Local  ")
        s = Settings()
        client = create_llm_client(s)
        assert isinstance(client, OpenAICompatClient)
