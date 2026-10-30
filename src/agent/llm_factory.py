"""Model-agnostic LLM client (OpenAI-compatible and Anthropic) over plain HTTPS. No SDK needed.
Records token usage on `last_usage`. Supports JSON-schema structured output with automatic fallback."""
from __future__ import annotations

import json
import re

import requests


class LLMError(Exception):
    pass


class LLMClient:
    name = "none"
    last_usage: dict | None = None

    def chat(self, system: str, user: str, json_mode: bool = False, json_schema: dict | None = None,
             temperature: float = 0.1, max_tokens: int = 900) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class OpenAICompatClient(LLMClient):
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout
        self.name = f"openai-compatible:{model}"

    def _post(self, body: dict) -> requests.Response:
        return requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                             headers={"Authorization": f"Bearer {self.api_key}"})

    def chat(self, system, user, json_mode=False, json_schema=None, temperature=0.1, max_tokens=900):
        body = {"model": self.model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        try:
            if json_schema:   # structured outputs; fall back to plain JSON mode if the endpoint rejects the schema
                r = self._post({**body, "response_format": {"type": "json_schema", "json_schema": {"name": "plan", "schema": json_schema}}})
                if r.status_code in (400, 404, 422):
                    r = self._post({**body, "response_format": {"type": "json_object"}})
            else:
                r = self._post({**body, **({"response_format": {"type": "json_object"}} if json_mode else {})})
            r.raise_for_status()
            data = r.json()
            u = data.get("usage") or {}
            self.last_usage = {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, timeout: float = 30.0):
        self.api_key, self.model, self.timeout = api_key, model, timeout
        self.name = f"anthropic:{model}"

    def chat(self, system, user, json_mode=False, json_schema=None, temperature=0.1, max_tokens=900):
        sys_ = system + ("\nReply with a single JSON object only." if (json_mode or json_schema) else "")
        body = {"model": self.model, "max_tokens": max_tokens, "temperature": temperature, "system": sys_,
                "messages": [{"role": "user", "content": user}]}
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", json=body, timeout=self.timeout,
                              headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"})
            r.raise_for_status()
            data = r.json()
            u = data.get("usage") or {}
            self.last_usage = {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
            return "".join(b.get("text", "") for b in data["content"])
        except Exception as e:
            raise LLMError(f"{type(e).__name__}: {e}") from e


def build_llm(settings) -> LLMClient | None:
    p = settings.resolved_provider
    if p == "openai":
        return OpenAICompatClient(settings.openai_base_url, settings.openai_api_key, settings.openai_model)
    if p == "anthropic":
        return AnthropicClient(settings.anthropic_api_key, settings.anthropic_model)
    return None


def extract_json(text: str) -> dict:
    """Tolerant JSON extraction (handles code fences / leading prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise LLMError("no JSON object in model output")
        return json.loads(m.group(0))
