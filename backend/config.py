"""
Central configuration module for environment variables and service settings.
Loads variables securely from .env.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Locate root directory containing .env
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)

# OpenRouter LLM Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
AGENT1_MODEL = os.getenv("AGENT1_MODEL", "qwen/qwen3-8b:free").strip()
AGENT2_MODEL = os.getenv("AGENT2_MODEL", "cohere/north-mini-code:free").strip()
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()


def get_openrouter_api_key() -> str:
    """Returns the current OPENROUTER_API_KEY from environment or reloads from .env if updated."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key and ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return key


def get_agent1_model() -> str:
    """Returns the configured model ID for Agent 1 (default: qwen/qwen3-8b:free)."""
    return os.getenv("AGENT1_MODEL", AGENT1_MODEL).strip() or "qwen/qwen3-8b:free"


def get_agent2_model() -> str:
    """Returns the configured model ID for Agent 2 (default: cohere/north-mini-code:free)."""
    return os.getenv("AGENT2_MODEL", AGENT2_MODEL).strip() or "cohere/north-mini-code:free"
