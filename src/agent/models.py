"""Plan / Step / RunResult models (pydantic) + the per-user Session."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

StepStatus = Literal["pending", "running", "ok", "failed", "skipped", "needs_confirmation"]


class Step(BaseModel):
    id: str
    goal: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    status: StepStatus = "pending"
    result: Any = None
    error: str | None = None
    resolved_args: dict[str, Any] = Field(default_factory=dict)
    attempts: int = 0
    start_ms: float = 0.0
    end_ms: float = 0.0

    @property
    def latency_ms(self) -> float:
        return round(max(self.end_ms - self.start_ms, 0.0), 1)


class Plan(BaseModel):
    intents: list[str] = Field(default_factory=list)
    steps: list[Step] = Field(default_factory=list)
    planner: str = "rules"
    clarification: str | None = None
    notes: list[str] = Field(default_factory=list)

    def tools(self) -> list[str]:
        return [s.tool for s in self.steps]

    def critical_path(self) -> int:
        depth: dict[str, int] = {}
        for s in self.steps:
            depth[s.id] = 1 + max([depth.get(d, 0) for d in s.depends_on], default=0)
        return max(depth.values(), default=0)


class RunResult(BaseModel):
    run_id: str
    query: str
    plan: Plan
    answer: str
    ok: bool
    latency_ms: float
    mode: str
    principal: str = ""
    emergency: bool = False
    needs_clarification: bool = False
    sources: list[dict] = Field(default_factory=list)
    patient_id: str | None = None
    wall_ms: float = 0.0
    sequential_estimate_ms: float = 0.0
    tokens: int = 0
    faithfulness: float = 1.0
    quarantined: int = 0
    pending_approvals: list[str] = Field(default_factory=list)
    checkpoint_version: int = 0
    spans: list[dict] = Field(default_factory=list)


@dataclass
class Session:
    """Working memory for one user session."""
    active_patient_id: str | None = None
    last_patient_id: str | None = None
    last_condition: str | None = None
    last_appointment_id: int | None = None
    turns: list[dict] = field(default_factory=list)

    def remember_turn(self, role: str, text: str, keep: int = 8) -> None:
        self.turns.append({"role": role, "text": text[:600]})
        del self.turns[:-keep * 2]
