"""Single entry point that wires every layer together (like InsightForge's build_app)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agent.agent import HealthAgent
from .agent.llm_factory import LLMClient, build_llm
from .config import Settings, get_settings
from .data.db import Database
from .data.generate_sample_data import seed
from .memory.memory import MemoryManager
from .tools.medical_search import MedicalSearch
from .tools.registry import ToolRegistry, default_registry


@dataclass
class BornToFitApp:
    settings: Settings
    db: Database
    memory: MemoryManager
    registry: ToolRegistry
    search: MedicalSearch
    agent: HealthAgent
    llm: LLMClient | None

    @property
    def mode(self) -> str:
        return "Smart Mode (LLM)" if self.llm else "Live Mode (offline logic)"


def build_app(settings: Settings | None = None, llm: Any = "auto", db_path: str | None = None) -> BornToFitApp:
    settings = settings or get_settings()
    db = Database(db_path or settings.db_path)
    if not db.is_seeded():                       # a shared SQLite file is seeded once, then reused
        seed(db, settings.now(), settings.seed)
    memory = MemoryManager(db, settings)
    memory.index_all()
    registry = default_registry()
    search = MedicalSearch(settings)
    llm_client = build_llm(settings) if llm == "auto" else llm
    return BornToFitApp(settings, db, memory, registry, search, HealthAgent(db, memory, registry, search, settings, llm_client), llm_client)
