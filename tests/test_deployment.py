"""
Deployment Configuration and Production Readiness Tests.
Verifies container specifications, environment variable templates,
platform deployment files (Procfile, render.yaml, .dockerignore),
Streamlit production config, and model downloader utilities.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_env_example_completeness_and_safety():
    """Verify .env.example contains necessary production variables without leaking secrets."""
    env_file = PROJECT_ROOT / ".env.example"
    assert env_file.is_file(), ".env.example file not found"

    content = env_file.read_text(encoding="utf-8")
    required_keys = [
        "PORT",
        "DEVICE",
        "API_URL",
        "STREAMLIT_SERVER_PORT",
        "CONF_THRESHOLD",
        "MAX_UPLOAD_SIZE_MB",
    ]
    for key in required_keys:
        assert f"{key}=" in content, f"Missing required environment key: {key}"

    # Verify no real secrets or passwords leaked
    forbidden_tokens = ["sk-", "password123", "secret_key_123", "api_key_live"]
    for token in forbidden_tokens:
        assert token not in content.lower()


def test_dockerfile_production_directives():
    """Verify Dockerfile contains essential production directives and security best practices."""
    dockerfile = PROJECT_ROOT / "Dockerfile"
    assert dockerfile.is_file(), "Dockerfile not found"

    lines = dockerfile.read_text(encoding="utf-8")
    assert "FROM python:3.11-slim" in lines
    assert "USER appuser" in lines  # Non-root user
    assert "HEALTHCHECK" in lines   # Health check
    assert "EXPOSE" in lines
    assert "streamlit run" in lines or "uvicorn" in lines


def test_docker_compose_validity():
    """Verify docker-compose.yml exists and contains valid YAML structure."""
    compose_file = PROJECT_ROOT / "docker-compose.yml"
    assert compose_file.is_file(), "docker-compose.yml not found"

    data = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    assert "services" in data
    assert "app" in data["services"]
    assert "api" in data["services"]


def test_streamlit_config_toml():
    """Verify .streamlit/config.toml exists and contains valid server and theme settings."""
    config_file = PROJECT_ROOT / ".streamlit" / "config.toml"
    assert config_file.is_file(), ".streamlit/config.toml not found"

    content = config_file.read_text(encoding="utf-8")
    assert "[server]" in content
    assert "port = 8501" in content
    assert "headless = true" in content
    assert "maxUploadSize = 25" in content
    assert "[theme]" in content


def test_paas_deployment_files():
    """Verify Procfile, render.yaml, and .dockerignore exist and are non-empty."""
    procfile = PROJECT_ROOT / "Procfile"
    render_yaml = PROJECT_ROOT / "render.yaml"
    dockerignore = PROJECT_ROOT / ".dockerignore"

    assert procfile.is_file() and procfile.stat().st_size > 0
    assert render_yaml.is_file() and render_yaml.stat().st_size > 0
    assert dockerignore.is_file() and dockerignore.stat().st_size > 0

    render_data = yaml.safe_load(render_yaml.read_text(encoding="utf-8"))
    assert "services" in render_data


def test_deployment_guide_documentation():
    """Verify docs/DEPLOYMENT_GUIDE.md is present and covers required operational instructions."""
    guide_path = PROJECT_ROOT / "docs" / "DEPLOYMENT_GUIDE.md"
    assert guide_path.is_file(), "DEPLOYMENT_GUIDE.md not found"

    content = guide_path.read_text(encoding="utf-8")
    assert "Streamlit Community Cloud" in content
    assert "Docker" in content
    assert "Environment Variables" in content
    assert "Startup Commands" in content
    assert "Known Production Limitations" in content


def test_model_downloader_script():
    """Verify scripts/download_models.py functions with --check-only."""
    from scripts.download_models import MODEL_CATALOG, compute_file_sha256

    assert "yolov8n_detector" in MODEL_CATALOG
    # Test SHA256 computation on an existing file
    readme = PROJECT_ROOT / "README.md"
    assert len(compute_file_sha256(readme)) == 64


def test_streamlit_cloud_deployment_configuration():
    """Verify Streamlit Community Cloud CPU deployment files, python version, and wheels."""
    python_ver_file = PROJECT_ROOT / ".python-version"
    assert python_ver_file.is_file(), ".python-version file is required for Streamlit Cloud"
    py_ver = python_ver_file.read_text(encoding="utf-8").strip()
    assert py_ver.startswith("3.12"), f"Expected Python 3.12, got: {py_ver}"

    packages_file = PROJECT_ROOT / "packages.txt"
    assert packages_file.is_file(), "packages.txt is required for Debian system dependencies"
    packages = packages_file.read_text(encoding="utf-8")
    assert "libgl1" in packages
    assert "libglib2.0-0" in packages

    req_file = PROJECT_ROOT / "requirements.txt"
    req_content = req_file.read_text(encoding="utf-8")
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in req_content
    assert "torch==2.5.1+cpu" in req_content
    assert "torchvision==0.20.1+cpu" in req_content
    assert "opencv-python-headless" in req_content


