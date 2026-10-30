"""Role-based access control with patient scoping. Deny by default; every decision is audited.

IMPORTANT: the principal is a DEMO identity chosen in the UI (or MCP server flag). It is not authentication.
In production, derive it from your identity provider (SSO / OIDC claims)."""
from __future__ import annotations

from dataclasses import dataclass


class PermissionDenied(Exception):
    pass


@dataclass(frozen=True)
class Principal:
    id: str
    label: str
    role: str                                   # patient | attendant | doctor | admin
    patient_ids: frozenset = frozenset()        # patient role only: the records this person may access

    def sees(self, patient_id: str | None) -> bool:
        if patient_id is None:
            return True
        return patient_id in self.patient_ids if self.role == "patient" else True

    def short(self) -> str:
        return f"{self.role}:{self.id}"


ALL = {"patient", "attendant", "doctor", "admin"}
CLINICAL_STAFF = {"attendant", "doctor", "admin"}

TOOL_ROLES: dict[str, set[str]] = {
    "find_patient": ALL, "get_patient_history": ALL, "find_slots": ALL, "book_appointment": ALL,
    "cancel_appointment": ALL, "search_medical_info": ALL, "export_fhir_bundle": ALL,
    "add_patient_record": CLINICAL_STAFF, "update_patient_record": CLINICAL_STAFF,
}

DEMO_PRINCIPALS: dict[str, Principal] = {
    "attendant": Principal("att-01", "Front-desk attendant", "attendant"),
    "guardian_p1001": Principal("fam-p1001", "Anita Kumar (family of P1001)", "patient", frozenset({"P1001"})),
    "guardian_p1003": Principal("fam-p1003", "Ravi Pillai (family of P1003)", "patient", frozenset({"P1003"})),
    "doctor": Principal("doc-D01", "Dr. Anjali Rao (nephrologist)", "doctor"),
    "admin": Principal("adm-01", "Platform admin", "admin"),
}
DEFAULT_PRINCIPAL = DEMO_PRINCIPALS["attendant"]


def authorize(principal: Principal | None, tool: str, args: dict) -> tuple[bool, str]:
    """Return (allowed, reason). `None` principal means an internal/system call (still audited)."""
    if principal is None:
        return True, "system call"
    allowed_roles = TOOL_ROLES.get(tool)
    if allowed_roles is None:
        return False, f"tool '{tool}' has no access policy (deny by default)"
    if principal.role not in allowed_roles:
        return False, f"role '{principal.role}' may not call {tool}"
    pid = args.get("patient_id")
    if pid is not None and not principal.sees(str(pid)):
        return False, f"role '{principal.role}' has no access to patient {pid}"
    return True, f"role '{principal.role}' allowed"
