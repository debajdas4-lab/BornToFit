"""Tool registry: every tool has a name, JSON-schema params, role policy, side-effect flag and a callable."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from ..security.rbac import Principal


class TransientToolError(Exception):
    """Network-style failure worth one automatic retry."""


@dataclass
class ToolContext:
    db: Any
    memory: Any
    search: Any
    settings: Any
    session: Any = None
    run_id: str = ""
    principal: Principal | None = None

    def now(self) -> datetime:
        return self.settings.now()

    def can_see(self, patient_id: str | None) -> bool:
        return self.principal is None or self.principal.sees(patient_id)


@dataclass
class ToolSpec:
    name: str
    description: str
    params: dict                      # JSON-schema "properties"
    fn: Callable[..., dict]
    side_effect: bool = False
    required: list[str] = field(default_factory=list)
    destructive: bool = False         # MCP annotation hint


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "params": t.params, "required": t.required,
                 "side_effect": t.side_effect} for t in (self._tools[n] for n in self.names())]


def default_registry() -> ToolRegistry:
    from . import interop, medical_search, patients, scheduling
    reg = ToolRegistry()
    for mod in (patients, scheduling, medical_search, interop):
        for spec in mod.SPECS:
            reg.register(spec)
    return reg
