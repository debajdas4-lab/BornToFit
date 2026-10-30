"""Trusted medical information with CORRECTIVE RAG:
retrieve (MedlinePlus live / WHO+MedlinePlus via Tavily live / offline corpus) -> GRADE relevance ->
rewrite and retry if nothing relevant -> fall back -> SCAN for prompt injection and quarantine -> cite.
Every result carries source, freshness (live/offline) and retrieval time."""
from __future__ import annotations

import html
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime

from ..memory.vector_store import HybridIndex, tokenize
from ..security.guardrails import MEDICAL_DISCLAIMER, redact_for_external
from ..security.injection import scan
from .medical_terms import canonical_condition, count_condition_mentions, detect_conditions
from .registry import ToolContext, ToolSpec, TransientToolError

OFFLINE_LABEL = "Offline reference corpus (paraphrased general education; not a live Medline/WHO result)"
GENERIC = {"treatment", "option", "latest", "method", "therapy", "management", "information", "overview", "symptom", "cause",
           "prevention", "guideline", "tell", "summarize", "summary", "new", "diet", "exercise", "lifestyle", "nutrition", "activity"}
KEEP_THRESHOLD = 0.5

OFFLINE_CORPUS = [
    {"title": "Chronic kidney disease: treatment overview", "topic": "chronic kidney disease", "text": (
        "Treatment of chronic kidney disease (CKD) aims to slow progression and reduce complications. Core steps are controlling blood pressure (often with an ACE inhibitor or ARB when appropriate), controlling blood sugar in people with diabetes, and managing cholesterol and cardiovascular risk. SGLT2 inhibitors have been shown to slow kidney function decline in many adults with CKD and are considered by clinicians when suitable. Lifestyle measures include limiting sodium, individualised protein, potassium and phosphorus guidance from a renal dietitian, regular activity, not smoking, and avoiding NSAID painkillers unless a doctor approves. Kidney function (eGFR) and urine albumin are monitored regularly, and anaemia and bone-mineral problems are treated as they arise. In kidney failure the options are haemodialysis, peritoneal dialysis, kidney transplant, or conservative care.")},
    {"title": "Chronic kidney disease: staging, monitoring and warning signs", "topic": "chronic kidney disease", "text": (
        "CKD is staged by eGFR (G1 at 90 or above down to G5 below 15 mL/min/1.73 m2) and by the amount of albumin in urine. Early CKD often has no symptoms, so routine blood and urine tests matter, especially with diabetes, hypertension or a family history of kidney disease. Signs that may appear as it advances include swelling of the legs or around the eyes, tiredness, changes in urination, nausea and itching. Seek prompt care for marked swelling, very little urine, confusion, chest pain or shortness of breath.")},
    {"title": "Chronic kidney disease: diet and physical activity", "topic": "chronic kidney disease", "text": (
        "Diet and exercise are part of kidney care. A renal dietitian can tailor advice on sodium, protein, potassium and phosphorus to the stage of disease and to other conditions such as diabetes. Most people with CKD are encouraged to stay physically active, for example regular walking or other moderate activity as tolerated, and to build up gradually; the right amount depends on blood pressure, heart health and fluid status, so it should be agreed with the care team. Avoid over-the-counter NSAID painkillers and herbal supplements unless the kidney doctor approves, and keep follow-up blood and urine tests.")},
    {"title": "Type 2 diabetes: management overview", "topic": "type 2 diabetes", "text": (
        "Management of type 2 diabetes combines healthy eating, regular physical activity, weight management and medicines. Metformin is commonly the first-line medicine unless it is unsuitable. SGLT2 inhibitors and GLP-1 receptor agonists are widely used, especially when heart or kidney disease is present, because they can reduce those complications. HbA1c targets are individualised, often around 7% for many adults. Blood pressure and cholesterol control, plus regular eye, foot and kidney screening, are part of routine care. People on insulin or sulfonylureas should know how to recognise and treat low blood sugar.")},
    {"title": "Type 2 diabetes: diet and physical activity", "topic": "type 2 diabetes", "text": (
        "Lifestyle is central to type 2 diabetes care. Eating patterns that emphasise vegetables, whole grains, legumes and limited refined sugar help glucose control, and a dietitian can individualise the plan. Regular physical activity improves insulin sensitivity; many guidelines suggest building towards about 150 minutes of moderate activity a week, spread across the week, with strength training on some days. People who take insulin or sulfonylureas should check with their care team about timing meals and exercise to avoid low blood sugar.")},
    {"title": "Hypertension: diagnosis and treatment overview", "topic": "hypertension", "text": (
        "High blood pressure is confirmed with repeated clinic readings and often home or ambulatory monitoring. Lifestyle measures include reducing salt, a DASH-style diet, weight loss, regular exercise, limiting alcohol and stopping smoking. Medicine classes commonly used are ACE inhibitors, ARBs, calcium-channel blockers and thiazide-type diuretics, and many people need a combination. Targets are individualised; many guidelines aim for below 130/80 mmHg when tolerated. Regular follow-up checks blood pressure, kidney function and potassium.")},
    {"title": "Heart failure: treatment overview", "topic": "heart failure", "text": (
        "Heart failure causes breathlessness, fatigue and fluid retention. For reduced ejection fraction, guideline therapy typically combines an ARNI (or ACE inhibitor/ARB), a beta-blocker, a mineralocorticoid receptor antagonist and an SGLT2 inhibitor, with diuretics for congestion. Daily weights, sodium and fluid guidance, vaccinations and cardiac rehabilitation help, and some people benefit from implanted devices. Rapid weight gain, worsening breathlessness or swelling should prompt a clinical review.")},
    {"title": "COPD: treatment overview", "topic": "COPD", "text": (
        "Chronic obstructive pulmonary disease (COPD) management starts with stopping smoking and avoiding irritants. Long-acting bronchodilators (LAMA and/or LABA) are the mainstay, inhaled corticosteroids are added for selected patients, and short-acting inhalers relieve symptoms. Pulmonary rehabilitation, vaccinations and a written action plan reduce flare-ups. Long-term oxygen is used when blood oxygen is persistently low. Seek care promptly for worsening breathlessness, fever or a change in sputum.")},
    {"title": "Asthma: treatment overview", "topic": "asthma", "text": (
        "Asthma control relies on regular inhaled corticosteroid controller treatment plus a reliever inhaler, sometimes a combined inhaled corticosteroid-formoterol. Identifying and avoiding triggers, checking inhaler technique, and a written action plan are central. Needing a reliever more than twice a week suggests poor control and warrants review. Severe breathlessness, difficulty speaking or no relief from the reliever needs urgent medical care.")},
    {"title": "Osteoarthritis: treatment overview", "topic": "osteoarthritis", "text": (
        "Osteoarthritis care centres on exercise and physiotherapy, weight management and pain relief. Paracetamol and topical NSAIDs are often tried first; oral NSAIDs need caution in people with kidney, heart or stomach problems. Joint injections can help some people, and joint replacement is considered when pain and function are severely affected despite other treatment.")},
    {"title": "Hypothyroidism: treatment overview", "topic": "hypothyroidism", "text": (
        "Hypothyroidism is treated with daily levothyroxine, taken consistently, usually on an empty stomach. The dose is adjusted using TSH blood tests, first every few weeks to months and then about yearly once stable. Calcium and iron supplements can reduce absorption, so they are usually taken a few hours apart.")},
]


def _strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html.unescape(s or ""))).strip()


def _trim(s: str, n: int = 650) -> str:
    if len(s) <= n:
        return s
    cut = s[:n]
    k = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:k + 1] if k > 200 else cut.rstrip()) + " ..."


def parse_medlineplus(xml_text: str) -> list[dict]:
    out = []
    for doc in ET.fromstring(xml_text).iter("document"):
        fields = {c.get("name"): "".join(c.itertext()) for c in doc.findall("content")}
        title, body = _strip_html(fields.get("title", "")), _strip_html(fields.get("FullSummary") or fields.get("snippet") or "")
        if title and body:
            out.append({"title": title, "snippet": _trim(body), "url": doc.get("url")})
    return out


def grade(query: str, doc: dict) -> float:
    """Relevance grader (deterministic): fraction of the query's content words found in the document, penalised when
    the document is about a different condition than the one asked about."""
    q = [t for t in tokenize(query) if t not in GENERIC]
    if not q:
        return 1.0
    text = set(tokenize(doc["title"] + " " + doc["snippet"]))
    cov = sum(t in text for t in q) / len(q)
    cond = canonical_condition(query)
    # the asked-about condition must be what the document is ABOUT: in its title, or the dominant topic of its text
    cond_ok = cond is None or cond in detect_conditions(doc["title"]) or count_condition_mentions(doc["snippet"], cond) >= 2
    return round(cov * (1.0 if cond_ok else 0.4), 3)


def rewrite_query(q: str) -> str:
    cond = canonical_condition(q)
    base = cond or " ".join(t for t in tokenize(q) if t not in GENERIC) or q
    return f"{base} treatment overview"


class OfflineCorpus:
    def __init__(self):
        self.index = HybridIndex()
        for i, d in enumerate(OFFLINE_CORPUS):
            self.index.add(f"off{i}", f"{d['title']}. {d['text']}", {"i": i, "topic": d["topic"]})

    def search(self, query: str, k: int = 3) -> list[dict]:
        cond = canonical_condition(query)
        where = (lambda m: m["topic"] == cond) if cond else None
        life = bool(re.search(r"\b(diet|exercise|lifestyle|nutrition|physical activity|activity)\b", query, re.I))
        out = []
        for h in self.index.search(query, k=max(k, 5), where=where):
            d = OFFLINE_CORPUS[h.meta["i"]]
            is_life = "diet" in d["title"].lower()
            is_core = bool(re.search(r"treatment|management|overview", d["title"], re.I))
            bonus = 0.05 if (life and is_life) or (not life and is_core and not is_life) else 0.0
            out.append({"title": d["title"], "snippet": d["text"], "source": OFFLINE_LABEL, "url": None, "freshness": "offline",
                        "retrieved_at": None, "score": h.score + bonus})
        return sorted(out, key=lambda x: -x["score"])[:k]


class MedicalSearch:
    TTL = 12 * 3600   # NLM recommends caching Web-service results for 12-24 h

    def __init__(self, settings):
        self.settings = settings
        self.offline = OfflineCorpus()
        self._cache: dict[tuple, tuple[float, list[dict]]] = {}

    def status(self) -> dict:
        s = self.settings
        return {"medlineplus_live": bool(s.live_search and s.medline_enabled),
                "tavily_who_medline": bool(s.live_search and s.tavily_api_key), "offline_corpus": True}

    def _cached(self, key: tuple, fn):
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self.TTL:
            return hit[1]
        val = fn()
        self._cache[key] = (time.time(), val)
        return val

    def _medlineplus(self, query: str, n: int) -> list[dict]:
        import requests

        def call():
            params = {"db": "healthTopics", "term": query, "retmax": n, "tool": "borntofit"}
            if self.settings.contact_email:
                params["email"] = self.settings.contact_email
            try:
                r = requests.get("https://wsearch.nlm.nih.gov/ws/query", params=params, timeout=self.settings.http_timeout)
                r.raise_for_status()
            except requests.RequestException as e:
                raise TransientToolError(f"MedlinePlus: {type(e).__name__}") from e
            return parse_medlineplus(r.text)

        now = datetime.now().isoformat(timespec="seconds")
        return [{**d, "source": "MedlinePlus.gov (U.S. National Library of Medicine)", "freshness": "live", "retrieved_at": now}
                for d in self._cached(("medline", query, n), call)]

    def _tavily(self, query: str, n: int) -> list[dict]:
        import requests

        def call():
            try:
                r = requests.post("https://api.tavily.com/search", timeout=self.settings.http_timeout,
                                  headers={"Authorization": f"Bearer {self.settings.tavily_api_key}"},
                                  json={"query": query, "max_results": n, "search_depth": "basic",
                                        "include_domains": ["who.int", "medlineplus.gov"]})
                r.raise_for_status()
            except requests.RequestException as e:
                raise TransientToolError(f"Tavily: {type(e).__name__}") from e
            return [{"title": x.get("title", ""), "snippet": _trim(x.get("content", "")), "url": x.get("url")}
                    for x in r.json().get("results", []) if x.get("content")]

        now = datetime.now().isoformat(timespec="seconds")
        return [{**d, "source": "WHO (who.int)" if "who.int" in (d["url"] or "") else "MedlinePlus.gov", "freshness": "live", "retrieved_at": now}
                for d in self._cached(("tavily", query, n), call)]

    def _live(self, q: str, n: int, errors: list[str]) -> list[dict]:
        live: list[dict] = []
        if not self.settings.live_search:
            return live
        for enabled, fn in ((bool(self.settings.tavily_api_key), self._tavily), (self.settings.medline_enabled, self._medlineplus)):
            if enabled and len(live) < n:
                try:
                    live += fn(q, n)
                except Exception as e:
                    errors.append(str(e))
        seen, uniq = set(), []
        for d in live:
            if d["title"] not in seen:
                seen.add(d["title"])
                uniq.append(d)
        return uniq

    def _graded(self, q: str, n: int, crag: dict, errors: list[str], allow_live: bool = True) -> tuple[list[dict], str]:
        if allow_live:
            docs = self._live(q, n, errors)
            kept = [{**d, "relevance": g} for d in docs if (g := grade(q, d)) >= KEEP_THRESHOLD]
            crag["attempts"].append({"query": q, "source": "live", "retrieved": len(docs), "kept": len(kept)})
            if kept:
                return kept[:n], "live"
        docs = self.offline.search(q, k=min(3, n))
        kept = [{**d, "relevance": g} for d in docs if (g := grade(q, d)) >= KEEP_THRESHOLD]
        crag["attempts"].append({"query": q, "source": "offline", "retrieved": len(docs), "kept": len(kept)})
        return kept[:n], "offline"

    def search(self, query: str, max_results: int = 4, known_names: list[str] | None = None) -> dict:
        safe_q = redact_for_external(query, known_names)
        crag: dict = {"attempts": [], "rewritten": False, "quarantined_fallback": False}
        errors: list[str] = []
        results, mode = self._graded(safe_q, max_results, crag, errors)
        if not results:
            crag["rewritten"] = True
            results, mode = self._graded(rewrite_query(safe_q), max_results, crag, errors)
        clean, quarantined = [], []
        for d in results:
            f = scan(d["title"] + " " + d["snippet"])
            (quarantined if f.suspicious else clean).append({**d, "injection_reasons": f.reasons} if f.suspicious else d)
        if not clean and mode == "live":                      # every live result was poisoned: use the offline corpus
            crag["quarantined_fallback"] = True
            clean, mode = self._graded(safe_q, max_results, crag, errors, allow_live=False)
        return {"query_sent": safe_q, "mode": mode, "results": clean, "errors": errors, "crag": crag,
                "quarantined": [{"title": q_["title"], "source": q_["source"], "reasons": q_["injection_reasons"]} for q_ in quarantined],
                "disclaimer": MEDICAL_DISCLAIMER}


def search_medical_info(ctx: ToolContext, query: str, condition: str | None = None, max_results: int = 4) -> dict:
    q = " ".join(x for x in [condition if condition and condition.lower() not in query.lower() else "", query] if x)
    res = ctx.search.search(q, max_results=max_results, known_names=[p["name"] for p in ctx.db.list_patients()])
    ctx.memory.trace("read", f"medical search ({res['mode']}; {len(res['crag']['attempts'])} attempt(s); {len(res['quarantined'])} quarantined): "
                             f"'{res['query_sent'][:60]}' -> {len(res['results'])} result(s)", None, ctx.run_id)
    if not res["results"]:
        return {"ok": False, "message": "No reliable medical information found.", **res}
    return {"ok": True, **res}


SPECS = [ToolSpec("search_medical_info", "Search trusted sources (MedlinePlus, WHO) for disease information with relevance grading, query rewriting and injection screening; falls back to an offline corpus. Never include patient identifiers in the query.",
                  {"query": {"type": "string"}, "condition": {"type": "string"}, "max_results": {"type": "integer"}},
                  search_medical_info, required=["query"])]
