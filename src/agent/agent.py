"""HealthAgent: safety triage -> plan -> checkpoint -> execute (parallel DAG, RBAC) -> compose -> checkpoint -> log.
Writes that need approval are held in a persisted checkpoint and finished later by `resume()`."""
from __future__ import annotations

import time
import uuid
from dataclasses import replace

from ..observability.tracing import RunTrace
from ..security import guardrails
from ..security.rbac import DEFAULT_PRINCIPAL, Principal
from ..tools.medical_terms import detect_conditions
from ..tools.registry import ToolContext
from .composer import compose
from .executor import Executor
from .models import Plan, RunResult, Session, Step
from .planner import LLMPlanner, RulePlanner


class HealthAgent:
    def __init__(self, db, memory, registry, search, settings, llm=None):
        self.db, self.memory, self.registry, self.search = db, memory, registry, search
        self.settings, self.llm = settings, llm
        self.rules = RulePlanner(db, settings.now)
        self.llm_planner = LLMPlanner(llm, registry, self.rules) if llm else None

    def _ctx(self, session: Session, run_id: str, principal: Principal) -> ToolContext:
        return ToolContext(self.db, self.memory, self.search, self.settings, session, run_id, principal)

    @staticmethod
    def _effective_session(session: Session, principal: Principal) -> Session:
        """A family member with access to exactly one patient does not need to name them."""
        if principal.role == "patient" and not session.active_patient_id and len(principal.patient_ids) == 1:
            return replace(session, active_patient_id=next(iter(principal.patient_ids)))
        return session

    # ------------------------------------------------------------------ run
    def run(self, text: str, session: Session, principal: Principal | None = None, parallel: bool | None = None,
            simulated_latency_ms: int | None = None, use_llm: bool = True) -> RunResult:
        principal = principal or DEFAULT_PRINCIPAL
        t0 = time.perf_counter()
        run_id = uuid.uuid4().hex[:8]
        trace = RunTrace(run_id)
        session.remember_turn("user", text)
        with trace.span("agent.run", "agent", None, principal=principal.short(), query_chars=len(text)) as root:
            res = self._body(text, session, principal, run_id, trace, root, self.llm if use_llm else None, parallel, simulated_latency_ms)
            root["attrs"].update(ok=res.ok, mode=res.mode)
        return self._seal(res, trace, principal, text, time.perf_counter() - t0)

    def _body(self, text, session, principal, run_id, trace, root, llm, parallel, sim) -> RunResult:
        with trace.span("guardrails", "guardrail", root["span_id"]) as g:
            flags = guardrails.classify(text)
            g["attrs"].update(emergency=flags.emergency, clinical_advice=flags.clinical_advice)
        if flags.emergency:
            plan = Plan(intents=["emergency_triage"], planner="safety", notes=flags.notes)
            self.memory.trace("plan", "emergency language -> no actions taken", None, run_id)
            return self._result(run_id, principal, text, plan, guardrails.EMERGENCY_MESSAGE, True, session, [], emergency=True)
        redactor = guardrails.Redactor([p["name"] for p in self.db.list_patients()])
        eff = self._effective_session(session, principal)
        with trace.span("plan", "plan", root["span_id"]) as ps:
            if llm is not None and self.llm_planner is not None and self.settings.llm_planner:
                self.llm_planner.llm = llm
                plan = self.llm_planner.plan(text, eff, redactor, trace, ps["span_id"])
            else:
                plan = self.rules.plan(text, eff)
            if flags.clinical_advice and not plan.steps:
                cond = (detect_conditions(text) or [None])[0]
                if cond:
                    plan = Plan(intents=["medical_info"], planner=plan.planner, notes=plan.notes, steps=[
                        Step(id="s1", goal=f"General information on {cond} (no diagnosis or dosing)", tool="search_medical_info",
                             args={"query": f"{cond} treatment options"})])
                else:
                    plan.clarification = guardrails.CLINICAL_ADVICE_NOTE
            ps["attrs"].update(planner=plan.planner, intents=plan.intents, steps=len(plan.steps), critical_path=plan.critical_path())
        self.memory.trace("plan", f"{plan.planner}: intents={plan.intents}; steps={[s.tool for s in plan.steps]}", None, run_id)
        stub = RunResult(run_id=run_id, query=text, plan=plan, answer="", ok=False, latency_ms=0, mode="", principal=principal.short())
        self.db.save_checkpoint(run_id, {"result": stub.model_dump(mode="json"), "principal": principal.short()}, "planned")
        wall = 0.0
        if plan.steps:
            with trace.span("execute", "execute", root["span_id"]) as es:
                ex = Executor(self.registry, self._ctx(session, run_id, principal),
                              parallel=self.settings.parallel if parallel is None else parallel, max_workers=self.settings.max_workers,
                              simulated_latency_ms=self.settings.simulated_latency_ms if sim is None else sim,
                              trace=trace, parent_span=es["span_id"])
                wall = ex.run(plan, run_id, redactor)
                es["attrs"]["wall_ms"] = round(wall, 1)
        name = self._patient_name(plan)
        with trace.span("compose", "compose", root["span_id"]) as cs:
            if plan.steps:
                cr = compose(plan, name, flags.clinical_advice, llm, trace, cs["span_id"])
                plan.notes.extend(cr.notes)
                answer, sources, used_llm, faith, quar = cr.answer, cr.sources, cr.used_llm, cr.faithfulness, cr.quarantined
            else:
                answer, sources, used_llm, faith, quar = (plan.clarification or "I wasn't able to understand that request."), [], False, 1.0, 0
            cs["attrs"].update(used_llm=used_llm, faithfulness=round(faith, 2), quarantined=quar)
        return self._result(run_id, principal, text, plan, answer, self._ok(plan), session, sources, mode_llm=used_llm or plan.planner == "llm",
                            wall_ms=wall, faithfulness=faith, quarantined=quar,
                            clarification=bool(plan.clarification) or any(isinstance(s.result, dict) and s.result.get("needs_clarification") for s in plan.steps))

    # ------------------------------------------------------------------ approval (human-in-the-loop)
    def resume(self, run_id: str, step_id: str, approve: bool, session: Session, principal: Principal | None = None) -> RunResult:
        """Finish a held write from its persisted checkpoint. Works across process restarts (state lives in the DB)."""
        principal = principal or DEFAULT_PRINCIPAL
        cp = self.db.load_checkpoint(run_id)
        if not cp:
            raise ValueError(f"no checkpoint for run {run_id}")
        res = RunResult.model_validate(cp["state"]["result"])
        step = next((s for s in res.plan.steps if s.id == step_id), None)
        if step is None or step.status != "needs_confirmation":
            raise ValueError(f"step {step_id} of run {run_id} is not awaiting approval")
        trace = RunTrace(f"{run_id}-r{cp['version']}")
        with trace.span("agent.resume", "agent", None, approve=approve, principal=principal.short()) as root:
            if approve:
                Executor(self.registry, self._ctx(session, run_id, principal), self.settings.parallel, trace=trace,
                         parent_span=root["span_id"]).confirm(step, run_id)
                if step.status == "ok" and isinstance(step.result, dict) and step.result.get("status") == "booked":
                    session.last_appointment_id = step.result["appointment"]["appt_id"]
            else:
                step.status, step.error = "skipped", "declined by the user"
            cr = compose(res.plan, self._patient_name(res.plan), False, None)
            res.answer, res.sources = cr.answer, cr.sources or res.sources
            res.ok = self._ok(res.plan)
            res.pending_approvals = [s.id for s in res.plan.steps if s.status == "needs_confirmation"]
        res.spans = res.spans + trace.spans
        self.db.log_spans(trace.spans)
        res.checkpoint_version = self.db.save_checkpoint(run_id, {"result": res.model_dump(mode="json"), "principal": principal.short()},
                                                         "approved" if approve else "declined")
        return res

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ok(plan: Plan) -> bool:
        return bool(plan.steps) and all(s.status in ("ok", "needs_confirmation") for s in plan.steps)

    @staticmethod
    def _patient_name(plan: Plan) -> str | None:
        for s in plan.steps:
            if s.tool == "find_patient" and isinstance(s.result, dict) and s.result.get("ok"):
                return s.result["patient"]["name"]
        return None

    def _result(self, run_id, principal, text, plan, answer, ok, session, sources, emergency=False, mode_llm=False, wall_ms=0.0,
                clarification=False, faithfulness=1.0, quarantined=0) -> RunResult:
        pid = None
        for s in plan.steps:
            if s.tool == "find_patient" and isinstance(s.result, dict) and s.result.get("ok"):
                pid = session.last_patient_id = s.result["patient"]["patient_id"]
            if s.tool == "book_appointment" and isinstance(s.result, dict) and s.result.get("status") == "booked":
                session.last_appointment_id = s.result["appointment"]["appt_id"]
            if s.tool == "search_medical_info" and isinstance(s.result, dict) and s.result.get("ok"):
                session.last_condition = (detect_conditions(s.result.get("query_sent", "")) or [None])[0]
        session.remember_turn("assistant", answer)
        return RunResult(run_id=run_id, query=text, plan=plan, answer=answer, ok=ok, latency_ms=0.0,
                         mode="Smart Mode (LLM)" if mode_llm else "Live Mode (offline logic)", principal=principal.short(), emergency=emergency,
                         needs_clarification=clarification, sources=sources, patient_id=pid, wall_ms=round(wall_ms, 1),
                         sequential_estimate_ms=round(sum(s.latency_ms for s in plan.steps), 1), faithfulness=round(faithfulness, 3),
                         quarantined=quarantined, pending_approvals=[s.id for s in plan.steps if s.status == "needs_confirmation"])

    def _seal(self, res: RunResult, trace: RunTrace, principal: Principal, text: str, elapsed_s: float) -> RunResult:
        """Called after the root span has closed, so the persisted trace is complete."""
        res.latency_ms = round(elapsed_s * 1000, 1)
        res.tokens = trace.tokens()
        res.spans = list(trace.spans)
        self.db.log_spans(trace.spans)
        self.db.log_run(res.run_id, principal.short(), text, res.plan.intents, len(res.plan.steps), res.ok, res.latency_ms,
                        res.plan.planner, res.mode, res.tokens)
        res.checkpoint_version = self.db.save_checkpoint(res.run_id, {"result": res.model_dump(mode="json"), "principal": principal.short()},
                                                         "awaiting_approval" if res.pending_approvals else "completed")
        return res
