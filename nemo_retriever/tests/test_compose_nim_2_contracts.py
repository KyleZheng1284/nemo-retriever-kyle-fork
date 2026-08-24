# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose contracts for hosted endpoints, extraction NIMs, and agentic service mode."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "dev/compose/service-mode.compose.yaml"
AGENTIC_OVERLAY = ROOT / "dev/compose/service-mode.agentic.compose.yaml"
CORE_PRESET = ROOT / "dev/compose/presets/nims-core.env"
LOCAL_PRESET = ROOT / "dev/compose/presets/local-models.env"


def _compose() -> dict:
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))


def _agentic_overlay() -> dict:
    return yaml.safe_load(AGENTIC_OVERLAY.read_text(encoding="utf-8"))


def _render_compose(*files: Path, values: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is not installed")
    version = subprocess.run(
        [docker, "compose", "version"],
        check=False,
        capture_output=True,
        text=True,
    )
    if version.returncode != 0:
        pytest.skip("Docker Compose is not available")

    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith(("AGENTIC_", "NRL_")):
            environment.pop(key)
    environment.update(
        {
            "NEMO_RETRIEVER_IMAGE": "nemo-retriever-service:compose-contract",
            "NVIDIA_API_KEY": "validation-key",
            **(values or {}),
        }
    )
    command = [docker, "compose", "--env-file", os.devnull]
    for path in files:
        command.extend(("-f", str(path)))
    command.extend(("config",))
    return subprocess.run(
        command,
        cwd=ROOT.parent,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_extraction_nims_use_distinct_native_models_and_supported_model_paths() -> None:
    services = _compose()["services"]
    expected = {
        "nim-page-elements": (
            "nvidia/nemotron-page-elements-v3",
            "page-elements",
            "NIM_PAGE_ELEMENTS_CACHE_PATH",
        ),
        "nim-table-structure": (
            "nvidia/nemotron-table-structure-v1",
            "table-structure",
            "NIM_TABLE_STRUCTURE_CACHE_PATH",
        ),
        "nim-ocr": ("nvidia/nemotron-ocr-v2", "ocr", "NIM_OCR_CACHE_PATH"),
    }

    for service_name, (model_name, model_dir, path_variable) in expected.items():
        service = services[service_name]
        env = service["environment"]
        assert env["NIM_ENGINE_MODEL_DOWNLOAD_PROVIDER"] == "ngc"
        assert env["NIM_ENGINE_MODEL_NAME"] == model_name
        assert env["NIM_ENGINE_MODEL_PATH"] == f"${{{path_variable}:-/model}}/{model_dir}"
        assert service["volumes"] == [
            f"nim_{service_name.removeprefix('nim-').replace('-', '_')}_cache:${{{path_variable}:-/model}}"
        ]
        assert service["healthcheck"]["test"] == [
            "CMD",
            "curl",
            "--fail",
            "--silent",
            "http://localhost:8000/v1/health/ready",
        ]
        assert not any(name.startswith("NIM_TRITON_") for name in env)

    assert services["nim-ocr"]["environment"]["NIM_ENGINE_MODEL_VARIANT"] == "multilingual"
    assert "/model-store" not in COMPOSE.read_text(encoding="utf-8")


def test_caption_nim_uses_the_helm_aligned_omni_image_tag() -> None:
    caption_image = _compose()["services"]["nim-caption"]["image"]
    assert caption_image == (
        "${NIM_CAPTION_IMAGE:-nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning}:"
        "${NIM_CAPTION_TAG:-2.0.4-variant}"
    )


def test_zero_profile_defaults_to_hosted_endpoints() -> None:
    compose = _compose()
    config = compose["configs"]["retriever_service_config"]["content"]

    hosted_endpoints = (
        "${NIM_PAGE_ELEMENTS_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3}",
        "${NIM_TABLE_STRUCTURE_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1}",
        "${NIM_OCR_URL-https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v2}",
    )
    for endpoint in hosted_endpoints:
        assert endpoint in config


def test_zero_profile_keeps_agentic_retrieval_and_auth_disabled() -> None:
    compose = _compose()
    config = compose["configs"]["retriever_service_config"]["content"]
    command = compose["services"]["vectordb"]["command"]

    assert "agentic:\n  enabled: ${AGENTIC_ENABLED:-false}" in config
    assert "llm_model: ${AGENTIC_LLM_MODEL:-null}" in config
    assert "invoke_url: ${AGENTIC_INVOKE_URL:-null}" in config
    assert "max_tokens: ${AGENTIC_MAX_TOKENS:-1024}" in config
    assert "enabled: ${NRL_AUTH_ENABLED:-false}" in config
    assert 'default_scope: "${NRL_SCOPE:-default}"' in config
    assert "allow_unscoped_dev: ${NRL_ALLOW_UNSCOPED_DEV:-true}" in config
    assert "--agentic" not in command


def test_agentic_overlay_requires_remote_model_and_wires_vectordb_flags() -> None:
    overlay = _agentic_overlay()
    assert set(overlay["services"]) == {"vectordb"}
    assert set(overlay["services"]["vectordb"]) == {"command"}
    command = overlay["services"]["vectordb"]["command"]

    assert "--agentic" in command
    expected_values = {
        "--embed-endpoint": "${NIM_EMBED_URL:-https://integrate.api.nvidia.com/v1/embeddings}",
        "--embed-model": "${NIM_EMBED_MODEL:-nvidia/llama-nemotron-embed-vl-1b-v2}",
        "--agentic-llm-model": "${AGENTIC_LLM_MODEL:?Set AGENTIC_LLM_MODEL for agentic retrieval}",
        "--agentic-invoke-url": "${AGENTIC_INVOKE_URL:?Set AGENTIC_INVOKE_URL for agentic retrieval}",
        "--agentic-reasoning-effort": "${AGENTIC_REASONING_EFFORT:-high}",
        "--agentic-backend-top-k": "${AGENTIC_BACKEND_TOP_K:-20}",
        "--agentic-react-max-steps": "${AGENTIC_REACT_MAX_STEPS:-50}",
        "--agentic-text-truncation": "${AGENTIC_TEXT_TRUNCATION:-0}",
        "--agentic-temperature": "${AGENTIC_TEMPERATURE:-0}",
        "--agentic-max-tokens": "${AGENTIC_MAX_TOKENS:-1024}",
    }
    for flag, expected_value in expected_values.items():
        assert command[command.index(flag) + 1] == expected_value
    assert "--agentic-request-timeout" not in command


def test_compose_render_keeps_base_classic_and_enables_scoped_agentic_overlay() -> None:
    base = _render_compose(COMPOSE)
    assert base.returncode == 0, base.stderr
    base_config = yaml.safe_load(base.stdout)
    base_gateway = yaml.safe_load(base_config["configs"]["retriever_service_config"]["content"])
    assert base_gateway["agentic"]["enabled"] is False
    assert base_gateway["auth"] == {
        "enabled": False,
        "api_token": "",
        "default_scope": "default",
        "allow_unscoped_dev": True,
    }
    assert "--agentic" not in base_config["services"]["vectordb"]["command"]

    values = {
        "AGENTIC_ENABLED": "true",
        "AGENTIC_LLM_MODEL": "agent-model",
        "AGENTIC_INVOKE_URL": "https://agent.example/v1/chat/completions",
        "AGENTIC_MAX_TOKENS": "2048",
        "AGENTIC_REQUEST_TIMEOUT_S": "1777",
        "NRL_AUTH_ENABLED": "true",
        "NRL_API_TOKEN": "public-validation-token",
        "NRL_INTERNAL_VDB_TOKEN": "internal-validation-token",
        "NRL_SCOPE": "aiq-agentic-poc",
        "NRL_ALLOW_UNSCOPED_DEV": "false",
    }
    agentic = _render_compose(COMPOSE, AGENTIC_OVERLAY, values=values)
    assert agentic.returncode == 0, agentic.stderr
    agentic_config = yaml.safe_load(agentic.stdout)
    agentic_gateway = yaml.safe_load(agentic_config["configs"]["retriever_service_config"]["content"])
    assert agentic_gateway["agentic"]["enabled"] is True
    assert agentic_gateway["agentic"]["max_tokens"] == 2048
    assert agentic_gateway["agentic"]["request_timeout_s"] == 1777
    assert agentic_gateway["auth"] == {
        "enabled": True,
        "api_token": "public-validation-token",
        "default_scope": "aiq-agentic-poc",
        "allow_unscoped_dev": False,
    }
    assert "--agentic" in agentic_config["services"]["vectordb"]["command"]
    agentic_command = agentic_config["services"]["vectordb"]["command"]
    assert agentic_command[agentic_command.index("--agentic-max-tokens") + 1] == "2048"
    assert "--agentic-request-timeout" not in agentic_command
    assert agentic_config["services"]["retriever"]["environment"]["NRL_API_TOKEN"] == "public-validation-token"
    assert (
        agentic_config["services"]["retriever"]["environment"]["NRL_INTERNAL_VDB_TOKEN"] == "internal-validation-token"
    )
    assert "NRL_API_TOKEN" not in agentic_config["services"]["vectordb"]["environment"]
    assert (
        agentic_config["services"]["vectordb"]["environment"]["NRL_INTERNAL_VDB_TOKEN"] == "internal-validation-token"
    )


@pytest.mark.parametrize("missing", ["AGENTIC_LLM_MODEL", "AGENTIC_INVOKE_URL"])
def test_agentic_compose_render_requires_model_and_invoke_url(missing: str) -> None:
    values = {
        "AGENTIC_LLM_MODEL": "agent-model",
        "AGENTIC_INVOKE_URL": "https://agent.example/v1/chat/completions",
    }
    values.pop(missing)

    rendered = _render_compose(COMPOSE, AGENTIC_OVERLAY, values=values)

    assert rendered.returncode != 0
    assert missing in rendered.stderr


def test_nims_core_preset_uses_internal_endpoints() -> None:
    core_preset = set(CORE_PRESET.read_text(encoding="utf-8").splitlines())
    internal_endpoints = {
        "NIM_PAGE_ELEMENTS_URL=http://nim-page-elements:8000/v1/page-elements",
        "NIM_TABLE_STRUCTURE_URL=http://nim-table-structure:8000/v1/table-structure",
        "NIM_OCR_URL=http://nim-ocr:8000/v1/ocr",
        "NIM_EMBED_URL=http://nim-embedding:8000/v1/embeddings",
    }
    assert internal_endpoints <= core_preset
    assert "/v1/infer" not in CORE_PRESET.read_text(encoding="utf-8")


def test_local_models_preset_disables_remote_endpoints() -> None:
    local_preset = set(LOCAL_PRESET.read_text(encoding="utf-8").splitlines())
    disabled_endpoints = {
        "NIM_PAGE_ELEMENTS_URL=",
        "NIM_TABLE_STRUCTURE_URL=",
        "NIM_OCR_URL=",
        "NIM_EMBED_URL=",
    }
    assert disabled_endpoints <= local_preset
