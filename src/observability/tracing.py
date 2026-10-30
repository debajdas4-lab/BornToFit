"""OpenTelemetry-style tracing (trace/span ids, parent links, attributes, status) stored in SQLite.
Not an OTLP exporter: spans are plain dicts so they can be shipped to any collector later."""
from __future__ import annotations

import threading
import time
import uuid
from contextlib import contextmanager


class RunTrace:
    def __init__(self, trace_id: str):
        self.id = trace_id
        self.t0 = time.perf_counter()
        self.spans: list[dict] = []
        self._lock = threading.Lock()

    def now_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1000

    @contextmanager
    def span(self, name: str, kind: str = "internal", parent: str | None = None, **attrs):
        sp = {"trace_id": self.id, "span_id": uuid.uuid4().hex[:8], "parent_id": parent, "name": name, "kind": kind,
              "start_ms": self.now_ms(), "end_ms": None, "status": "ok", "attrs": dict(attrs)}
        try:
            yield sp
        except Exception as e:
            sp["status"] = "error"
            sp["attrs"]["error"] = f"{type(e).__name__}: {str(e)[:160]}"
            raise
        finally:
            sp["end_ms"] = self.now_ms()
            with self._lock:
                self.spans.append(sp)

    def tokens(self) -> int:
        return int(sum((s["attrs"].get("tokens_in", 0) or 0) + (s["attrs"].get("tokens_out", 0) or 0) for s in self.spans))


def traced_chat(trace: RunTrace | None, parent: str | None, llm, label: str, system: str, user: str, **kw) -> str:
    """Call an LLM client inside a span that records latency, model and token usage."""
    if trace is None:
        return llm.chat(system, user, **kw)
    with trace.span(f"llm:{label}", "llm", parent, model=getattr(llm, "name", "?")) as sp:
        out = llm.chat(system, user, **kw)
        u = getattr(llm, "last_usage", None) or {}
        sp["attrs"].update(tokens_in=u.get("in", 0), tokens_out=u.get("out", 0), chars_in=len(system) + len(user), chars_out=len(out))
        return out
