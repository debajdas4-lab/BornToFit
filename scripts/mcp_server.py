"""Run BornToFit as an MCP server over stdio.

  python scripts/mcp_server.py --role doctor
  python scripts/mcp_server.py --role patient --patients P1001
  python scripts/mcp_server.py --role attendant --db borntofit.db      # shared, persistent SQLite file

Claude Desktop / any MCP client config:
  {"mcpServers": {"borntofit": {"command": "python", "args": ["scripts/mcp_server.py", "--role", "doctor"]}}}
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import get_settings  # noqa: E402
from src.mcp_server.server import serve  # noqa: E402
from src.pipeline import build_app  # noqa: E402
from src.security.rbac import Principal  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--role", choices=["patient", "attendant", "doctor", "admin"], default="attendant")
ap.add_argument("--patients", default="", help="comma-separated patient IDs (patient role only)")
ap.add_argument("--db", default="", help="SQLite file to share with the UI (default: private in-memory demo data)")
ap.add_argument("--live-search", action="store_true", help="allow live MedlinePlus/Tavily lookups")
a = ap.parse_args()

st = get_settings()
st.live_search = bool(a.live_search)
if a.db:
    st.db_path = a.db
app = build_app(st, llm=None)
principal = Principal(f"mcp-{a.role}", f"MCP {a.role}", a.role, frozenset(x.strip() for x in a.patients.split(",") if x.strip()))
print(f"[borntofit-mcp] ready as {principal.short()} patients={sorted(principal.patient_ids)}", file=sys.stderr, flush=True)
serve(app, principal)
