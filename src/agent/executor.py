"""Dependency-aware executor: runs independent steps in parallel, resolves $step.path references, enforces RBAC,
audits every call, retries transient failures once, records spans, and never lets the plan set `confirm`."""
from __future__ import annotations

import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any

from ..observability.tracing import RunTrace
from ..security.rbac import PermissionDenied, authorize
from ..tools.registry import ToolContext, ToolRegistry, TransientToolError
from .models import Plan, Step

_REF = re.compile(r"^\$(s\d+)\.?(.*)$")


class UnresolvedRef(Exception):
    pass


def resolve(value: Any, results: dict[str, Any], redactor=None) -> Any:
    if isinstance(value, str):
        if m := _REF.match(value):
            sid, path = m.groups()
            if sid not in results:
                raise UnresolvedRef(f"step {sid} has no result")
            cur: Any = results[sid]
            for part in [p for p in path.split(".") if p]:
                try:
                    cur = cur[int(part)] if isinstance(cur, list) else cur[part]
                except (KeyError, IndexError, ValueError, TypeError):
                    raise UnresolvedRef(f"cannot resolve {value}") from None
            return cur
        return redactor.restore(value) if redactor else value
    if isinstance(value, dict):
        return {k: resolve(v, results, redactor) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, results, redactor) for v in value]
    return value


def invoke_tool(ctx: ToolContext, registry: ToolRegistry, name: str, args: dict, allow_confirm: bool = False) -> dict:
    """The single guarded entry point for every tool call (agent executor and MCP server both use it)."""
    spec = registry.get(name)
    clean = {k: v for k, v in args.items() if v is not None and (allow_confirm or k != "confirm")}
    ok, reason = authorize(ctx.principal, name, clean)
    p = ctx.principal
    ctx.db.log_audit(ctx.run_id, p.id if p else "system", p.role if p else "system", name, clean.get("patient_id"),
                     "allow" if ok else "deny", reason)
    if not ok:
        raise PermissionDenied(reason)
    if unknown := set(clean) - set(spec.params):
        raise ValueError(f"unknown argument(s) {sorted(unknown)} for {name}")
    if missing := [r for r in spec.required if r not in clean]:
        raise ValueError(f"missing required argument(s) {missing} for {name}")
    return spec.fn(ctx, **clean)


def _log_args(tool: str, args: dict) -> dict:
    a = dict(args)
    if tool in {"add_patient_record", "update_patient_record"}:
        for k in ("details", "title"):                    # free text is clinical content: log its size only
            if k in a:
                a[k] = f"<{len(str(a[k]))} chars>"
    return a


class Executor:
    def __init__(self, registry: ToolRegistry, ctx: ToolContext, parallel: bool = True, max_workers: int = 4,
                 simulated_latency_ms: int = 0, trace: RunTrace | None = None, parent_span: str | None = None):
        self.registry, self.ctx, self.parallel, self.max_workers = registry, ctx, parallel, max_workers
        self.sim, self.trace, self.parent = simulated_latency_ms, trace, parent_span

    def _run_step(self, step: Step, results: dict[str, Any], run_id: str, t0: float, redactor) -> None:
        spec = self.registry.get(step.tool)
        step.start_ms = (time.perf_counter() - t0) * 1000
        step.status = "running"
        span_cm = self.trace.span(f"tool:{step.tool}", "tool", self.parent, step=step.id, goal=step.goal) if self.trace else None
        sp = span_cm.__enter__() if span_cm else None
        try:
            step.resolved_args = resolve(step.args, results, redactor)
            if self.sim:
                time.sleep(self.sim / 1000)
            for attempt in (1, 2):
                step.attempts = attempt
                try:
                    res = invoke_tool(self.ctx, self.registry, step.tool, step.resolved_args)
                    break
                except TransientToolError:
                    if attempt == 2:
                        raise
                    time.sleep(0.2)
            step.result = res
            if res.get("status") == "pending_confirmation":
                step.status = "needs_confirmation"
            elif res.get("ok", True):
                step.status = "ok"
            else:
                step.status, step.error = "failed", res.get("message", "tool reported failure")
        except PermissionDenied as e:
            step.status, step.error = "failed", f"permission denied: {e}"
        except UnresolvedRef as e:
            step.status, step.error = "failed", f"missing input: {e}"
        except Exception as e:  # tool bug / bad args: record, never crash the run
            step.status, step.error = "failed", f"{type(e).__name__}: {e}"
        step.end_ms = (time.perf_counter() - t0) * 1000
        if sp is not None:
            sp["attrs"].update(status=step.status, side_effect=spec.side_effect)
            sp["status"] = "ok" if step.status in ("ok", "needs_confirmation") else "error"
            span_cm.__exit__(None, None, None)
        self.ctx.db.log_tool(run_id, step.id, step.tool, step.status, step.latency_ms,
                             _log_args(step.tool, step.resolved_args or step.args), step.error, spec.side_effect)

    def run(self, plan: Plan, run_id: str, redactor=None) -> float:
        """Execute the plan in place. Returns wall-clock ms."""
        t0 = time.perf_counter()
        steps = {s.id: s for s in plan.steps}
        pending, running = set(steps), {}
        limit = self.max_workers if self.parallel else 1
        with ThreadPoolExecutor(max_workers=limit) as pool:
            while pending or running:
                for sid in sorted(pending):
                    s = steps[sid]
                    deps = [steps[d].status for d in s.depends_on if d in steps]
                    if any(d in ("failed", "skipped", "needs_confirmation") for d in deps):
                        s.status, s.error = "skipped", "an earlier step did not succeed"
                        pending.discard(sid)
                    elif all(d == "ok" for d in deps) and len(running) < limit:
                        pending.discard(sid)
                        # A worker sets `result` BEFORE `status="ok"`, so "all deps ok" guarantees their results exist.
                        snap = {k: v.result for k, v in steps.items() if v.status == "ok"}
                        running[pool.submit(self._run_step, s, snap, run_id, t0, redactor)] = sid
                if running:
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for f in done:
                        running.pop(f)
                        f.result()
        return (time.perf_counter() - t0) * 1000

    def confirm(self, step: Step, run_id: str) -> Step:
        """Execute a write that was held for approval. Authorization is re-checked for the approver."""
        spec = self.registry.get(step.tool)
        t0 = time.perf_counter()
        try:
            res = invoke_tool(self.ctx, self.registry, step.tool, {**step.resolved_args, "confirm": True}, allow_confirm=True)
            step.result = res
            ok = res.get("ok", True) and res.get("status") != "pending_confirmation"
            step.status, step.error = ("ok", None) if ok else ("failed", res.get("message"))
        except PermissionDenied as e:
            step.status, step.error = "failed", f"permission denied: {e}"
        step.end_ms = step.start_ms + (time.perf_counter() - t0) * 1000
        self.ctx.db.log_tool(run_id, step.id + "-approved", step.tool, step.status, step.latency_ms,
                             _log_args(step.tool, step.resolved_args), step.error, spec.side_effect)
        return step
