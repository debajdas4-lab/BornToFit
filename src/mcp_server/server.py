"""MCP server (stdio, newline-delimited JSON-RPC 2.0) exposing BornToFit's guarded tools to any MCP client.

Protocol notes (verified against the published specs, not against an official conformance suite):
- 2026-07-28 is stateless: no initialize handshake, every request carries its version in `_meta`, and a
  `server/discover` call is optional. This server therefore answers tools/list and tools/call WITHOUT any
  prior handshake, and also answers `initialize` so older clients (2025-11-25 / 2025-06-18) still work.
- The exact result shape of `server/discover` and the `ttlMs` cache hint are best-effort and UNVERIFIED.
  For production use the official SDK for your language and keep this module as the tool layer behind it.

Security: the principal (role + patient scope) is fixed at launch, discovery only lists tools that role may use,
every call goes through RBAC + audit, and instruction-like text in returned data is neutralised for the client LLM.
Approval of write tools is the MCP client's responsibility (annotations mark them); the server still enforces RBAC."""
from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from ..agent.executor import invoke_tool
from ..security.injection import neutralize_obj
from ..security.rbac import TOOL_ROLES, PermissionDenied, Principal
from ..tools.registry import ToolContext

SUPPORTED_VERSIONS = ["2026-07-28", "2025-11-25", "2025-06-18"]
SERVER_INFO = {"name": "borntofit-healthcare", "title": "BornToFit Healthcare Agent tools", "version": "1.0.0"}
INSTRUCTIONS = ("Synthetic healthcare demo. Tools book appointments, read and update records, search trusted medical sources and "
                "export FHIR R4 bundles. No diagnosis, prescribing or dosing. Writes are audited; ask the user before calling write tools.")
READ_SANITISED = {"get_patient_history", "search_medical_info"}
MAX_TEXT = 20_000


class MCPServer:
    def __init__(self, app, principal: Principal):
        self.app, self.principal = app, principal
        self._n = 0

    # ---- helpers ---------------------------------------------------------------
    @staticmethod
    def _err(rid: Any, code: int, msg: str) -> dict:
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": msg}}

    def _visible_tools(self):
        reg = self.app.registry
        return [reg.get(n) for n in reg.names() if self.principal.role in TOOL_ROLES.get(n, set())]

    def _tool_json(self, spec) -> dict:
        return {"name": spec.name, "title": spec.name.replace("_", " ").title(), "description": spec.description,
                "inputSchema": {"type": "object", "properties": spec.params, "required": spec.required, "additionalProperties": False},
                "annotations": {"readOnlyHint": not spec.side_effect, "destructiveHint": bool(spec.destructive),
                                "openWorldHint": spec.name == "search_medical_info"}}

    # ---- dispatch ----------------------------------------------------------------
    def handle(self, msg: Any) -> dict | None:
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self._err(None, -32600, "Invalid Request")
        rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
        is_notification = "id" not in msg
        if not isinstance(method, str):
            return self._err(rid, -32600, "Invalid Request: missing method")
        if method.startswith("notifications/"):
            return None
        if is_notification:
            return None
        if method == "initialize":
            asked = params.get("protocolVersion")
            ver = asked if asked in SUPPORTED_VERSIONS else SUPPORTED_VERSIONS[0]
            return {"jsonrpc": "2.0", "id": rid, "result": {"protocolVersion": ver, "capabilities": {"tools": {"listChanged": False}},
                                                              "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": rid, "result": {}}
        if method == "server/discover":
            return {"jsonrpc": "2.0", "id": rid, "result": {"supportedVersions": SUPPORTED_VERSIONS, "capabilities": {"tools": {}},
                                                              "serverInfo": SERVER_INFO, "instructions": INSTRUCTIONS}}
        if method == "tools/list":
            tools = sorted((self._tool_json(s) for s in self._visible_tools()), key=lambda t: t["name"])   # deterministic order
            return {"jsonrpc": "2.0", "id": rid, "result": {"tools": tools, "ttlMs": 60_000}}
        if method == "tools/call":
            return self._call(rid, params)
        return self._err(rid, -32601, f"Method not found: {method}")

    def _call(self, rid: Any, params: dict) -> dict:
        name, args = params.get("name"), params.get("arguments") or {}
        if name not in self.app.registry or not isinstance(args, dict):
            return self._err(rid, -32602, f"Unknown tool or bad arguments: {name!r}")
        self._n += 1
        ctx = ToolContext(self.app.db, self.app.memory, self.app.search, self.app.settings, None, f"mcp-{self._n}", self.principal)
        try:
            result = invoke_tool(ctx, self.app.registry, name, args, allow_confirm=True)
            is_error = not result.get("ok", True) and result.get("status") != "pending_confirmation"
        except PermissionDenied as e:
            result, is_error = {"ok": False, "message": f"permission denied: {e}"}, True
        except (ValueError, TypeError) as e:
            result, is_error = {"ok": False, "message": f"invalid arguments: {e}"}, True
        except Exception as e:  # tool bug: report as a tool error, never crash the server
            result, is_error = {"ok": False, "message": f"{type(e).__name__}: {e}"}, True
        if name in READ_SANITISED:
            result = neutralize_obj(result)
        text = json.dumps(result, default=str, separators=(",", ":"))
        if len(text) > MAX_TEXT:
            text = json.dumps({"ok": result.get("ok"), "message": f"Result too large for text ({len(text)} chars); see structuredContent."})
        return {"jsonrpc": "2.0", "id": rid, "result": {"content": [{"type": "text", "text": text}], "structuredContent": result,
                                                         "isError": bool(is_error)}}


def serve(app, principal: Principal, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    """One JSON-RPC message per line in, one per line out. Nothing else is ever written to stdout."""
    server = MCPServer(app, principal)
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out = MCPServer._err(None, -32700, "Parse error")
        else:
            out = server.handle(msg)
        if out is not None:
            stdout.write(json.dumps(out, default=str, separators=(",", ":")) + "\n")
            stdout.flush()
