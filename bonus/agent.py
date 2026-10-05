"""HybridMemoryAgent — Personal AI Assistant Memory System for Vietnamese Users.

Combines:
  1. Episodic Memory: Qdrant Vector Store + BM25 Lexical with RRF (Filtered by user_id)
  2. Semantic Profile & Real-time Activity: Feast Online Feature Store (SQLite/Redis)
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, MatchValue,
                                  PointStruct, VectorParams)
from rank_bm25 import BM25Okapi

try:
    from feast import FeatureStore
except ImportError:
    FeatureStore = None


@dataclass
class MemoryChunk:
    chunk_id: str
    user_id: str
    text: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)


class HybridMemoryAgent:
    """Agent blending Episodic Long-Term Memory (Qdrant + BM25)

    with Real-time User Characteristics (Feast Feature Store).
    """

    COLLECTION_NAME = "ai_episodic_memory"

    def __init__(
        self,
        qdrant_client: QdrantClient | None = None,
        feature_store_path: str | Path | None = None,
        embed_model: str = "BAAI/bge-small-en-v1.5",
        vector_dim: int = 384,
    ) -> None:
        self.vector_dim = vector_dim
        self.embedder = TextEmbedding(model_name=embed_model)

        # 1. Vector Store (Qdrant)
        if qdrant_client is not None:
            self.client = qdrant_client
        else:
            qdrant_mode = os.getenv("QDRANT_MODE", "memory")
            if qdrant_mode == "server":
                url = os.getenv("QDRANT_URL", "http://localhost:6333")
                self.client = QdrantClient(url=url)
            else:
                self.client = QdrantClient(":memory:")

        self._ensure_collection()

        # 2. Local in-memory store for BM25 indexing
        self.memories: list[MemoryChunk] = []
        self.bm25: BM25Okapi | None = None
        self._point_id_seq = 0

        # 3. Feature Store (Feast)
        self.fs: FeatureStore | None = None
        if feature_store_path is not None:
            fs_path = Path(feature_store_path).resolve()
            if (fs_path / "feature_store.yaml").exists() and FeatureStore is not None:
                try:
                    self.fs = FeatureStore(repo_path=str(fs_path))
                except Exception as e:
                    print(f"[Warning] Failed to load Feast repository: {e}")

    def _ensure_collection(self) -> None:
        cols = {c.name for c in self.client.get_collections().collections}
        if self.COLLECTION_NAME not in cols:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
            )

    # ── 1. EPISODIC MEMORY: REMEMBER ─────────────────────────────────────────
    def remember(
        self,
        text: str,
        user_id: str = "u_001",
        metadata: dict[str, Any] | None = None,
    ) -> list[str]:
        """Chunks incoming conversational/reading text, computes embeddings,

        and indexes into both Qdrant (with user_id filter) and BM25.
        """
        # Simple paragraph/sentence boundary chunking
        raw_chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if not raw_chunks:
            raw_chunks = [text.strip()]

        created_ids: list[str] = []
        points: list[PointStruct] = []
        now_iso = datetime.now(timezone.utc).isoformat()

        for chunk_text in raw_chunks:
            self._point_id_seq += 1
            chunk_id = f"mem_{user_id}_{self._point_id_seq:05d}"
            mem = MemoryChunk(
                chunk_id=chunk_id,
                user_id=user_id,
                text=chunk_text,
                timestamp=now_iso,
                metadata=metadata or {},
            )
            self.memories.append(mem)
            created_ids.append(chunk_id)

            # Vector embedding
            vec = next(self.embedder.embed([chunk_text])).tolist()
            points.append(
                PointStruct(
                    id=self._point_id_seq,
                    vector=vec,
                    payload={
                        "chunk_id": chunk_id,
                        "user_id": user_id,
                        "text": chunk_text,
                        "timestamp": now_iso,
                        **(metadata or {}),
                    },
                )
            )

        self.client.upsert(collection_name=self.COLLECTION_NAME, points=points)

        # Re-build / update BM25
        tokenized_corpus = [m.text.lower().split() for m in self.memories]
        self.bm25 = BM25Okapi(tokenized_corpus)

        return created_ids

    # ── 2. EPISODIC + STABLE RECALL ──────────────────────────────────────────
    def recall(
        self,
        query: str,
        user_id: str = "u_001",
        top_k: int = 3,
        rrf_k: int = 60,
    ) -> str:
        """Retrieves user profile from Feast + performs hybrid search on Qdrant,

        returning an assembled prompt-ready context string.
        """
        # A. Fetch Profile & Recent Activity from Feast
        profile_data = self._fetch_user_features(user_id)

        # B. Hybrid Search on User's Episodic Memory
        top_memories = self._search_memories(query, user_id=user_id, top_k=top_k, rrf_k=rrf_k)

        # C. Assemble Context String for LLM
        return self._assemble_context(query, user_id, profile_data, top_memories)

    def _fetch_user_features(self, user_id: str) -> dict[str, Any]:
        """Queries Feast online store for tabular and streaming velocity features."""
        features = {
            "reading_speed_wpm": 200,
            "preferred_language": "vi",
            "topic_affinity": "general",
            "queries_last_hour": 1,
            "distinct_topics_24h": 1,
        }
        if self.fs is not None:
            try:
                out = self.fs.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:preferred_language",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour",
                        "query_velocity_features:distinct_topics_24h",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()
                for k in features:
                    val = out.get(k, [None])[0]
                    if val is not None:
                        features[k] = val
            except Exception:
                pass
        return features

    def _search_memories(
        self, query: str, user_id: str, top_k: int, rrf_k: int
    ) -> list[dict[str, Any]]:
        """Hybrid Search (BM25 + Qdrant) restricted strictly to user_id."""
        depth = max(top_k * 5, 20)

        # 1. Lexical BM25 (filter by user_id)
        user_mems = [(i, m) for i, m in enumerate(self.memories) if m.user_id == user_id]
        kw_ranked_ids: list[str] = []
        if user_mems and self.bm25:
            scores = self.bm25.get_scores(query.lower().split())
            user_scored = [(scores[idx], mem.chunk_id) for idx, mem in user_mems]
            user_scored.sort(key=lambda x: -x[0])
            kw_ranked_ids = [cid for _, cid in user_scored[:depth]]

        # 2. Vector ANN (Qdrant payload filter: user_id)
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        )
        q_res = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=q_vec,
            query_filter=user_filter,
            limit=depth,
        ).points
        sem_ranked_ids = [p.payload["chunk_id"] for p in q_res]

        # 3. Reciprocal Rank Fusion (RRF, 1-based rank)
        rrf_scores: dict[str, float] = {}
        for rank, cid in enumerate(kw_ranked_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        for rank, cid in enumerate(sem_ranked_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)

        sorted_cids = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]

        # Map to chunk payload
        chunk_map = {m.chunk_id: m for m in self.memories}
        results = []
        for cid, score in sorted_cids:
            chunk = chunk_map.get(cid)
            if chunk:
                results.append({
                    "chunk_id": cid,
                    "text": chunk.text,
                    "score": score,
                    "timestamp": chunk.timestamp,
                })
        return results

    def _assemble_context(
        self,
        query: str,
        user_id: str,
        profile: dict[str, Any],
        memories: list[dict[str, Any]],
    ) -> str:
        """Assembles structured prompt context for downstream LLM generation."""
        mem_lines = []
        if memories:
            for i, m in enumerate(memories, 1):
                mem_lines.append(f"  [{i}] (score: {m['score']:.4f}) {m['text']}")
        else:
            mem_lines.append("  (Chưa có ghi nhớ liên quan trong kho ký ức)")

        return (
            f"=== [ASSISTANT CONTEXT FRAMEWORK] ===\n"
            f"• User ID            : {user_id}\n"
            f"• Query              : '{query}'\n"
            f"• Ngôn ngữ ưu tiên   : {profile['preferred_language'].upper()}\n"
            f"• Tốc độ đọc ước tính: {profile['reading_speed_wpm']} wpm\n"
            f"• Lĩnh vực quan tâm  : {profile['topic_affinity']} (affinity score high)\n"
            f"• Tần suất hoạt động : {profile['queries_last_hour']} truy vấn/giờ qua ({profile['distinct_topics_24h']} chủ đề)\n"
            f"\n[Episodic Memories (Ký ức liên quan)]:\n"
            + "\n".join(mem_lines)
            + f"\n======================================"
        )
