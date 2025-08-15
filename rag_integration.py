"""
Lightweight integration for Streamlit diary app with cost-control and optional source display.

Usage:
from rag_integration import get_rag_helper

rag = get_rag_helper(artifacts_dir="artifacts")
rag_insight, source_docs = rag.suggest(
    user_log="I had chest pain in the cold",
    k=5,
    model="gpt-4o-mini",
    show_sources=True,
    use_llm=True,
    max_chunk_chars=1200
)

Returns:
- rag_insight: str
- source_docs: list[str] (if show_sources=True)
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Any, List, Optional
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
    _openai_available = True
except Exception:
    _openai_available = False

PILOT_LLM = os.getenv("RAG_PILOT_LLM", "0") == "1"

@dataclass
class RAGSettings:
    artifacts_dir: str = "artifacts"
    embed_model: str = "all-MiniLM-L6-v2"
    default_llm: str = "gpt-4o-mini"
    top_k: int = 5
    system_prompt: str = (
        "You are a careful, supportive health assistant for Sickle Cell Disease. "
        "Use the provided context verbatim for factual claims. Cite sources as "
        "[source: filename p.##] inline. If information is not in context, say so."
    )

class RAGHelper:
    def __init__(self, settings: Optional[RAGSettings] = None):
        self.s = settings or RAGSettings()
        self._index = None
        self._metadata = None
        self._id2text = None
        self._embedder = None
        self._client = OpenAI() if (_openai_available and os.getenv("OPENAI_API_KEY")) else None

    def _ensure_loaded(self):
        if self._index is not None:
            return

        index_path = os.path.join(self.s.artifacts_dir, "faiss.index")
        meta_path = os.path.join(self.s.artifacts_dir, "metadata.json")
        chunks_path = os.path.join(self.s.artifacts_dir, "chunks_tokenized.jsonl")

        if not os.path.exists(index_path) or not os.path.exists(meta_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Artifacts not found in {self.s.artifacts_dir}. Run rag_build_pipeline.py first."
            )

        self._index = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self._metadata = json.load(f)

        self._id2text = {}
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                self._id2text[rec["id"]] = rec["text"]

        self._embedder = SentenceTransformer(self.s.embed_model)

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        kk = k or self.s.top_k
        qv = self._embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self._index.search(qv, kk)

        results = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            meta = dict(self._metadata[i])
            meta["score"] = float(score)
            meta["text"] = self._id2text.get(meta["id"], "")
            meta["filename"] = os.path.basename(meta["source"])
            results.append(meta)
        return results

    def _build_messages(self, user_log: str, ctx: List[Dict[str, Any]], max_chunk_chars: int) -> list:
        ctx_lines = []
        for c in ctx:
            cite = f"[source: {c.get('filename','?')} p.{c.get('page','?')}]"
            text = c['text'][:max_chunk_chars] + ("..." if len(c['text']) > max_chunk_chars else "")
            ctx_lines.append(f"{text}\n{cite}")
        context_block = "\n\n---\n\n".join(ctx_lines)

        user_content = (
            f"Context (verbatim chunks with citations):\n{context_block}\n\n"
            f"User Log:\n{user_log}\n\n"
            "Instructions:\n"
            "- Ground your answer in the context above; do not invent clinical claims.\n"
            "- Offer gentle, supportive suggestions framed for SCD patients/caregivers.\n"
            "- Include inline citations where relevant.\n"
            "- If a key fact is missing, say what additional info would be needed.\n"
        )
        return [
            {"role": "system", "content": self.s.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def suggest(self, user_log: str, k: Optional[int] = None, model: Optional[str] = None,
                temperature: float = 0.2, show_sources: bool = True,
                use_llm: Optional[bool] = None, max_chunk_chars: int = 1200):
        ctx = self.retrieve(user_log, k=k)
        sources = [c["text"] for c in ctx] if show_sources else []

        if not (self._client and (use_llm if use_llm is not None else PILOT_LLM)):
            return ("(Retrieval-only mode — LLM disabled for cost control.)", sources)

        try:
            m = model or self.s.default_llm
            resp = self._client.chat.completions.create(
                model=m,
                messages=self._build_messages(user_log, ctx, max_chunk_chars),
                temperature=temperature,
            )
            answer = resp.choices[0].message.content
            return (answer, sources)
        except Exception:
            return ("(Fallback) Could not call LLM. Showing retrieved context only.", sources)

@lru_cache(maxsize=4)
def get_rag_helper(artifacts_dir: str = "artifacts", embed_model: str = "all-MiniLM-L6-v2") -> RAGHelper:
    return RAGHelper(RAGSettings(artifacts_dir=artifacts_dir, embed_model=embed_model))