import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent.models import Session  # noqa: E402
from src.config import Settings  # noqa: E402
from src.pipeline import build_app  # noqa: E402
from src.security.rbac import DEMO_PRINCIPALS  # noqa: E402

NOW = "2026-10-05T10:00"   # a Monday
CAPSTONE = ("My 70-year-old father has chronic kidney disease. I want to book a nephrologist for him. "
            "Also, can you summarize latest treatment methods?")


@pytest.fixture
def settings():
    return Settings(fixed_now=NOW, live_search=False)


@pytest.fixture
def app(settings):
    return build_app(settings, llm=None)


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def P():
    return DEMO_PRINCIPALS


class FakeLLM:
    """Scripted LLM: returns queued replies in order; records every prompt and reports token usage."""
    name = "fake"

    def __init__(self, *replies):
        self.replies, self.calls, self.last_usage = list(replies), [], None

    def chat(self, system, user, json_mode=False, json_schema=None, temperature=0.1, max_tokens=900):
        self.calls.append({"system": system, "user": user, "json_schema": json_schema})
        self.last_usage = {"in": 11, "out": 7}
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return r
