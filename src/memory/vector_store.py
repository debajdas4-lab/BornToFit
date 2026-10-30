"""Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion (RRF).
Dense search uses FAISS when installed, otherwise NumPy (same interface)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

try:  # optional
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

_STOP = set("a an the of and or to in on for with is are was were be been it this that as at by from his her their my i you we they he she what which who how can do does about me".split())


def tokenize(text: str) -> list[str]:
    out = []
    for t in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if t in _STOP:
            continue
        if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


class HashEmbedder:
    """Offline, deterministic embeddings (hashed word + bigram counts, L2-normalised). Lexical, not semantic."""
    name = "hashing-768"

    def __init__(self, dim: int = 768):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._v = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2", ngram_range=(1, 2), stop_words="english")

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._v.transform(texts).toarray().astype("float32")


class RemoteEmbedder:
    """OpenAI-compatible /embeddings endpoint (optional; the index falls back to hashing on any error)."""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 8.0):
        self.base_url, self.api_key, self.model, self.timeout = base_url, api_key, model, timeout
        self.name = f"remote:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        import requests
        r = requests.post(f"{self.base_url}/embeddings", timeout=self.timeout, headers={"Authorization": f"Bearer {self.api_key}"},
                          json={"model": self.model, "input": texts})
        r.raise_for_status()
        arr = np.array([d["embedding"] for d in r.json()["data"]], dtype="float32")
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b, self.docs, self.df, self.avgdl = k1, b, [], {}, 0.0

    def fit(self, docs: list[list[str]]) -> None:
        self.docs, self.df = docs, {}
        for d in docs:
            for t in set(d):
                self.df[t] = self.df.get(t, 0) + 1
        self.avgdl = (sum(len(d) for d in docs) / len(docs)) if docs else 0.0

    def scores(self, q: list[str]) -> np.ndarray:
        n, out = len(self.docs), np.zeros(len(self.docs), dtype="float32")
        for t in set(q):
            df = self.df.get(t, 0)
            if not df:
                continue
            idf = np.log(1 + (n - df + 0.5) / (df + 0.5))
            for i, d in enumerate(self.docs):
                f = d.count(t)
                if f:
                    out[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(d) / (self.avgdl or 1)))
        return out


@dataclass
class Hit:
    id: str
    text: str
    meta: dict
    score: float
    bm25: float
    dense: float
    matched_by: str


class HybridIndex:
    def __init__(self, embedder: Any = None):
        self.embedder = embedder or HashEmbedder()
        self.docs: list[dict] = []
        self._bm25 = BM25()
        self._emb: np.ndarray | None = None
        self._faiss = None
        self._dirty = True

    def add(self, id: str, text: str, meta: dict | None = None) -> None:
        self.docs = [d for d in self.docs if d["id"] != id]
        self.docs.append({"id": id, "text": text, "meta": meta or {}})
        self._dirty = True

    def remove_where(self, fn: Callable[[dict], bool]) -> int:
        before = len(self.docs)
        self.docs = [d for d in self.docs if not fn(d["meta"])]
        self._dirty = True
        return before - len(self.docs)

    def _embed(self, texts: list[str]) -> np.ndarray:
        try:
            return self.embedder.embed(texts)
        except Exception:
            self.embedder = HashEmbedder()
            return self.embedder.embed(texts)

    def _ensure(self) -> None:
        if not self._dirty:
            return
        texts = [d["text"] for d in self.docs]
        self._bm25.fit([tokenize(t) for t in texts])
        if texts:
            self._emb = self._embed(texts)
            if faiss is not None:
                self._faiss = faiss.IndexFlatIP(self._emb.shape[1])
                self._faiss.add(self._emb)
        else:
            self._emb, self._faiss = None, None
        self._dirty = False

    @property
    def backend(self) -> str:
        self._ensure()
        return ("FAISS" if faiss is not None else "NumPy") + f" + BM25 ({self.embedder.name})"

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, k: int = 5, where: Callable[[dict], bool] | None = None) -> list[Hit]:
        self._ensure()
        n = len(self.docs)
        if n == 0:
            return []
        mask = np.array([bool(where(d["meta"])) if where else True for d in self.docs])
        if not mask.any():
            return []
        bm = self._bm25.scores(tokenize(query))
        q = self._embed([query])[0]
        if self._faiss is not None:
            D, I = self._faiss.search(q[None, :], n)
            dense = np.zeros(n, dtype="float32")
            dense[I[0]] = D[0]
        else:
            dense = (self._emb @ q).astype("float32")  # type: ignore[operator]
        fused: dict[int, float] = {}
        for scores in (bm, dense):
            idx = [i for i in np.argsort(-scores) if mask[i] and scores[i] > 1e-6]
            for rank, i in enumerate(idx):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank + 1)
        hits = []
        for i, s in sorted(fused.items(), key=lambda kv: -kv[1])[:k]:
            by = "+".join(n_ for n_, sc in (("bm25", bm[i]), ("dense", dense[i])) if sc > 1e-6)
            hits.append(Hit(self.docs[i]["id"], self.docs[i]["text"], self.docs[i]["meta"], round(float(s), 5),
                            round(float(bm[i]), 3), round(float(dense[i]), 3), by))
        return hits
