"""Config: read API key and model name from .env, or take them explicitly.

[What this file does]
Central place for "which LLM, which key". Two sources are supported:
  1. CLI usage: load_settings() reads .env automatically.
  2. Streamlit deploy: the user types their key in the sidebar, we call
     build_settings(key=...) to construct a Settings object explicitly.

[Beginner concepts cheat sheet]
- Environment variable: an OS-level key=value pair. `.env` is just a file
  that bundles them.
- dataclass: Python's "lightweight class" — auto-generates __init__, __repr__, etc.
- frozen=True: makes the object immutable (so you can't accidentally overwrite a key).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openai_model: str


def build_settings(openai_api_key: str, openai_model: str = DEFAULT_MODEL) -> Settings:
    """Explicit constructor (used by Streamlit after the user enters a key)."""
    key = (openai_api_key or "").strip()
    if not key or not key.startswith("sk-"):
        raise ValueError("OpenAI key looks invalid (should start with 'sk-')")
    return Settings(openai_api_key=key, openai_model=(openai_model or DEFAULT_MODEL).strip())


def load_settings() -> Settings:
    """Read from .env (CLI mode). Fail fast on missing key — silent bugs are worse."""
    load_dotenv(PROJECT_ROOT / ".env")

    key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL).strip()

    if not key or key.startswith("sk-...") or key == "your_openai_key_here":
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Copy .env.example to .env and fill in a real key.\n"
            "Get one at: https://platform.openai.com/api-keys"
        )

    return Settings(openai_api_key=key, openai_model=model)


if __name__ == "__main__":
    s = load_settings()
    print(f"OK. model = {s.openai_model}, key prefix = {s.openai_api_key[:8]}...")
