"""Validate Docker Compose configs and .env.example for RIO local stack."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOCAL_COMPOSE = ROOT / "docker-compose.local.yml"
LOCAL_OVERRIDE = ROOT / "docker-compose.local.override.yml"
BASE_COMPOSE = ROOT / "docker-compose.yml"
ENV_EXAMPLE = ROOT / ".env.example"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def local_compose() -> dict:
    """Parse docker-compose.local.yml once per module."""
    return yaml.safe_load(LOCAL_COMPOSE.read_text())


@pytest.fixture(scope="module")
def local_override() -> dict:
    """Parse docker-compose.local.override.yml once per module."""
    return yaml.safe_load(LOCAL_OVERRIDE.read_text())


@pytest.fixture(scope="module")
def base_compose() -> dict:
    """Parse docker-compose.yml once per module."""
    return yaml.safe_load(BASE_COMPOSE.read_text())


@pytest.fixture(scope="module")
def env_example_text() -> str:
    """Raw text of .env.example."""
    return ENV_EXAMPLE.read_text()


# ---------------------------------------------------------------------------
# Task 2.3 — docker-compose.local.yml validation
# ---------------------------------------------------------------------------


class TestLocalComposeStructure:
    """Validate docker-compose.local.yml is well-formed YAML."""

    def test_file_exists(self) -> None:
        assert LOCAL_COMPOSE.is_file(), (
            f"{LOCAL_COMPOSE} does not exist"
        )

    def test_parses_as_valid_yaml(self, local_compose: dict) -> None:
        assert isinstance(local_compose, dict)

    def test_has_services_key(self, local_compose: dict) -> None:
        assert "services" in local_compose


class TestLocalComposeServices:
    """Verify required services exist in docker-compose.local.yml."""

    REQUIRED_SERVICES = [
        "llm",
        "whisper",
        "tts",
        "chromadb",
        "orchestrator",
    ]

    @pytest.mark.parametrize("service", REQUIRED_SERVICES)
    def test_required_service_present(
        self, local_compose: dict, service: str
    ) -> None:
        services = local_compose["services"]
        assert service in services, (
            f"Service '{service}' missing from local compose"
        )


class TestLocalComposePorts:
    """Verify services expose the expected ports."""

    def test_llm_exposes_8081(self, local_compose: dict) -> None:
        ports = local_compose["services"]["llm"]["ports"]
        assert any("8081" in str(p) for p in ports)

    def test_whisper_exposes_9090(self, local_compose: dict) -> None:
        ports = local_compose["services"]["whisper"]["ports"]
        assert any("9090" in str(p) for p in ports)

    def test_tts_exposes_9091(self, local_compose: dict) -> None:
        ports = local_compose["services"]["tts"]["ports"]
        assert any("9091" in str(p) for p in ports)


class TestLocalComposeLLM:
    """Verify LLM service has GPU reservation and healthcheck."""

    def test_gpu_reservation(self, local_compose: dict) -> None:
        llm = local_compose["services"]["llm"]
        devices = (
            llm["deploy"]["resources"]["reservations"]["devices"]
        )
        gpu_devs = [
            d for d in devices
            if "gpu" in d.get("capabilities", [])
        ]
        assert gpu_devs, "LLM service must reserve a GPU device"

    def test_gpu_driver_is_nvidia(self, local_compose: dict) -> None:
        llm = local_compose["services"]["llm"]
        devices = (
            llm["deploy"]["resources"]["reservations"]["devices"]
        )
        drivers = [d.get("driver") for d in devices]
        assert "nvidia" in drivers

    def test_healthcheck_configured(self, local_compose: dict) -> None:
        llm = local_compose["services"]["llm"]
        hc = llm.get("healthcheck", {})
        assert "test" in hc, "LLM healthcheck must define a test"
        assert "interval" in hc
        assert "retries" in hc

    def test_healthcheck_uses_health_endpoint(
        self, local_compose: dict
    ) -> None:
        hc = local_compose["services"]["llm"]["healthcheck"]
        test_cmd = " ".join(hc["test"])
        assert "/health" in test_cmd


class TestLocalComposeVolumes:
    """Verify volume mounts reference the models directory."""

    def test_llm_mounts_models_dir(self, local_compose: dict) -> None:
        volumes = local_compose["services"]["llm"].get("volumes", [])
        assert any("./models/" in str(v) for v in volumes), (
            "LLM service should mount ./models/ directory"
        )

    def test_tts_mounts_models_dir(self, local_compose: dict) -> None:
        volumes = local_compose["services"]["tts"].get("volumes", [])
        assert any("./models/" in str(v) for v in volumes), (
            "TTS service should mount ./models/ directory"
        )


class TestLocalComposeOrchestratorEnv:
    """Verify orchestrator env vars reference local backend URLs."""

    def _get_env_dict(self, compose: dict) -> dict[str, str]:
        """Parse the environment list into a dict."""
        env_list = compose["services"]["orchestrator"].get(
            "environment", []
        )
        result: dict[str, str] = {}
        for entry in env_list:
            if "=" in entry:
                k, v = entry.split("=", 1)
                result[k] = v
        return result

    def test_llm_backend_is_local(self, local_compose: dict) -> None:
        env = self._get_env_dict(local_compose)
        assert env.get("LLM_BACKEND") == "local"

    def test_llm_api_url_points_to_llm_service(
        self, local_compose: dict
    ) -> None:
        env = self._get_env_dict(local_compose)
        assert "llm" in env.get("LLM_API_URL", ""), (
            "LLM_API_URL should reference the llm service"
        )

    def test_tts_backend_is_local(self, local_compose: dict) -> None:
        env = self._get_env_dict(local_compose)
        assert env.get("TTS_BACKEND") == "local"

    def test_tts_url_points_to_tts_service(
        self, local_compose: dict
    ) -> None:
        env = self._get_env_dict(local_compose)
        assert "tts" in env.get("TTS_LOCAL_URL", ""), (
            "TTS_LOCAL_URL should reference the tts service"
        )

    def test_whisper_url_points_to_whisper_service(
        self, local_compose: dict
    ) -> None:
        env = self._get_env_dict(local_compose)
        assert "whisper" in env.get("WHISPER_URL", ""), (
            "WHISPER_URL should reference the whisper service"
        )


# ---------------------------------------------------------------------------
# Task 2.4 — docker-compose.local.override.yml validation
# ---------------------------------------------------------------------------


class TestLocalOverride:
    """Validate docker-compose.local.override.yml."""

    def test_file_exists(self) -> None:
        assert LOCAL_OVERRIDE.is_file()

    def test_parses_as_valid_yaml(
        self, local_override: dict
    ) -> None:
        assert isinstance(local_override, dict)

    def test_has_services_key(self, local_override: dict) -> None:
        assert "services" in local_override

    def test_overrides_are_subset_of_base(
        self, local_compose: dict, local_override: dict
    ) -> None:
        """Every service in the override must exist in the base."""
        base_services = set(local_compose["services"].keys())
        override_services = set(
            local_override["services"].keys()
        )
        extra = override_services - base_services
        assert not extra, (
            f"Override defines services not in base: {extra}"
        )

    def test_override_llm_reduces_context(
        self, local_override: dict
    ) -> None:
        """Dev override should use a smaller max-model-len."""
        llm = local_override["services"].get("llm", {})
        cmd = llm.get("command", "")
        assert "4096" in cmd, (
            "Dev override should reduce --max-model-len"
        )

    def test_override_whisper_uses_small_model(
        self, local_override: dict
    ) -> None:
        whisper = local_override["services"].get("whisper", {})
        env_list = whisper.get("environment", [])
        model_entries = [
            e for e in env_list if "WHISPER__MODEL" in e
        ]
        assert model_entries, "Override should set WHISPER__MODEL"
        assert any("small" in e for e in model_entries)


# ---------------------------------------------------------------------------
# Task 2.5 — .env.example completeness
# ---------------------------------------------------------------------------


class TestEnvExampleCompleteness:
    """Verify .env.example documents all RIO variables."""

    RIO_VARIABLES = [
        "LLM_BACKEND",
        "LLM_API_URL",
        "LLM_MODEL_LOCAL",
        "TTS_BACKEND",
        "TTS_LOCAL_URL",
        "TTS_VOICE_ID_LOCAL",
    ]

    @pytest.mark.parametrize("var", RIO_VARIABLES)
    def test_rio_variable_present(
        self, env_example_text: str, var: str
    ) -> None:
        assert var in env_example_text, (
            f"{var} missing from .env.example"
        )

    @pytest.mark.parametrize("var", RIO_VARIABLES)
    def test_rio_variable_has_comment(
        self, env_example_text: str, var: str
    ) -> None:
        """Each RIO variable should have a comment line nearby."""
        lines = env_example_text.splitlines()
        for i, line in enumerate(lines):
            if line.startswith(var) or line.startswith(f"# {var}"):
                # Check preceding lines for a comment
                preceding = lines[max(0, i - 3): i]
                has_comment = any(
                    ln.strip().startswith("#") and len(ln.strip()) > 2
                    for ln in preceding
                )
                # Or the line itself is a commented-out assignment
                if line.startswith("#"):
                    has_comment = True
                assert has_comment, (
                    f"No descriptive comment found near {var} "
                    f"in .env.example"
                )
                return
        pytest.fail(f"{var} not found in .env.example")

    def test_env_example_file_exists(self) -> None:
        assert ENV_EXAMPLE.is_file()
