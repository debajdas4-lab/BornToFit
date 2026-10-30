"""Typed settings: environment / .env first, then Streamlit secrets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get(key: str, default: str = "") -> str:
    v = os.environ.get(key)
    if v:
        return v
    try:  # Streamlit Cloud "Secrets"
        import streamlit as st

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:
        pass
    return default


def _bool(key: str, default: bool) -> bool:
    v = _get(key, "")
    return default if v == "" else v.strip().lower() in {"1", "true", "yes", "on"}


def _real(v: str) -> str:
    """Treat empty / placeholder values as 'not set'."""
    v = (v or "").strip()
    return "" if (not v or v.lower().startswith(("your_", "sk-your", "<", "changeme"))) else v


@dataclass
class Settings:
    llm_provider: str = "auto"          # auto | none | openai | anthropic
    llm_planner: bool = True
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = ""
    anthropic_api_key: str = ""
    anthropic_model: str = ""
    tavily_api_key: str = ""
    live_search: bool = True
    medline_enabled: bool = True
    contact_email: str = ""
    auto_confirm_writes: bool = True
    parallel: bool = True
    max_workers: int = 4
    simulated_latency_ms: int = 0       # only for the parallel-vs-sequential benchmark
    fixed_now: str = ""
    seed: int = 42
    embeddings: str = "hash"            # hash | remote
    http_timeout: float = 6.0
    tz_offset: str = "+05:30"
    db_path: str = ":memory:"

    @property
    def resolved_provider(self) -> str:
        p = self.llm_provider.lower()
        has_oa = bool(self.openai_api_key and self.openai_model)
        has_an = bool(self.anthropic_api_key and self.anthropic_model)
        if p == "none":
            return "none"
        if p == "openai":
            return "openai" if has_oa else "none"
        if p == "anthropic":
            return "anthropic" if has_an else "none"
        return "openai" if has_oa else "anthropic" if has_an else "none"

    def now(self) -> datetime:
        if self.fixed_now:
            return datetime.fromisoformat(self.fixed_now)
        return datetime.now().replace(second=0, microsecond=0)


def get_settings() -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    return Settings(
        llm_provider=_get("BORNTOFIT_LLM_PROVIDER", "auto"),
        llm_planner=_bool("BORNTOFIT_LLM_PLANNER", True),
        openai_api_key=_real(_get("OPENAI_API_KEY")),
        openai_base_url=_get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
        openai_model=_get("OPENAI_MODEL", ""),
        anthropic_api_key=_real(_get("ANTHROPIC_API_KEY")),
        anthropic_model=_get("ANTHROPIC_MODEL", ""),
        tavily_api_key=_real(_get("TAVILY_API_KEY")),
        live_search=_bool("BORNTOFIT_LIVE_SEARCH", True),
        contact_email=_get("BORNTOFIT_CONTACT_EMAIL", ""),
        auto_confirm_writes=_bool("BORNTOFIT_AUTO_CONFIRM_WRITES", True),
        parallel=_bool("BORNTOFIT_PARALLEL", True),
        fixed_now=_get("BORNTOFIT_FIXED_NOW", ""),
        embeddings=_get("BORNTOFIT_EMBEDDINGS", "hash"),
        tz_offset=_get("BORNTOFIT_TZ_OFFSET", "+05:30"),
        db_path=_get("BORNTOFIT_DB_PATH", "") or ":memory:",
    )
