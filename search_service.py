# search_service.py
import os
import json
import threading
from typing import List, Dict

import numpy as np
import faiss
from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
OS_HOST = {"host": "localhost", "port": 9200}
OS_INDEX = "chunks"

FAISS_INDEX_PATH = "output/faiss_ivfpq.index"
META_PATH = "output/embeddings_meta.jsonl"

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BM25_TOP_K = 100
ANN_TOP_K = 100
FAISS_NPROBE = 16   # IMPORTANT for IVF recall


# ------------------------------------------------------------
# Language code mapping (ISO-639-1 → ISO-639-3)
# ------------------------------------------------------------
LANG_MAP = {
    "ta": "tam",
    "kn": "kan",
    "hi": "hin",
    "te": "tel",
    "ml": "mal",
    "bn": "ben",
    "gu": "guj",
    "mr": "mar",
    "pa": "pan",
    "or": "ori",
    "ur": "urd",
    "en": "eng",
}


# ------------------------------------------------------------
# Initialize clients & models
# ------------------------------------------------------------
print("[INIT] Loading OpenSearch client...")
os_client = OpenSearch([OS_HOST])

print("[INIT] Loading embedding model...")
embed_model = SentenceTransformer(MODEL_NAME)

print("[INIT] Loading FAISS index...")
faiss_index = faiss.read_index(FAISS_INDEX_PATH)
faiss_index.nprobe = FAISS_NPROBE

print("[INIT] Loading embedding metadata...")
meta_index: List[Dict] = []
with open(META_PATH, "r", encoding="utf-8") as f:
    for line in f:
        meta_index.append(json.loads(line))

assert len(meta_index) == faiss_index.ntotal, (
    "Meta index size does not match FAISS index size"
)

print(f"[INIT] Ready | FAISS vectors: {faiss_index.ntotal}")


# ------------------------------------------------------------
# BM25 search (language-aware)  ✅ FIXED
# ------------------------------------------------------------
def bm25_search(query: str, lang: str = None, top_k: int = BM25_TOP_K) -> List[Dict]:
    must = [{"match": {"text": query}}]
    filters = []

    if lang:
        lang = LANG_MAP.get(lang, lang)   # 🔥 CRITICAL FIX
        filters.append({"term": {"language": lang}})

    body = {
        "size": top_k,
        "_source": ["chunk_id", "doc_id", "text", "page_no"],
        "query": {
            "bool": {
                "must": must,
                "filter": filters
            }
        }
    }

    res = os_client.search(index=OS_INDEX, body=body)

    hits = []
    for h in res["hits"]["hits"]:
        hits.append({
            "chunk_id": h["_source"]["chunk_id"],
            "doc_id": h["_source"]["doc_id"],
            "bm25_score": h["_score"],
            "text": h["_source"]["text"],
            "page_no": h["_source"]["page_no"],
        })

    return hits


# ------------------------------------------------------------
# ANN search (FAISS IVF-PQ)
# ------------------------------------------------------------
def ann_search(query: str, top_k: int = ANN_TOP_K) -> List[Dict]:
    qv = embed_model.encode(
        query,
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(qv.reshape(1, -1))

    hits = []
    D, I = faiss_index.search(qv.reshape(1, -1), top_k)

    for rank, idx in enumerate(I[0]):
        if idx < 0:
            continue
        hits.append({
            "chunk_index": int(idx),
            "vector_score": 1.0 / (rank + 1)
        })

    return hits


# ------------------------------------------------------------
# Union + document-level aggregation
# ------------------------------------------------------------
def union_and_aggregate(
    bm25_hits: List[Dict],
    ann_hits: List[Dict],
    meta_index: List[Dict]
) -> List[Dict]:

    doc_features = {}
    seen_chunks = set()

    # BM25 hits
    for h in bm25_hits:
        cid = h["chunk_id"]
        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)

        doc_id = h["doc_id"]
        f = doc_features.setdefault(
            doc_id, {"bm25": [], "vec": [], "chunks": []}
        )

        f["bm25"].append(h["bm25_score"])
        f["chunks"].append({
            "score": h["bm25_score"],
            "text": h["text"],
            "page_no": h["page_no"],
        })

    # ANN hits
    for h in ann_hits:
        meta = meta_index[h["chunk_index"]]
        cid = meta["chunk_id"]

        if cid in seen_chunks:
            continue
        seen_chunks.add(cid)

        doc_id = meta["doc_id"]
        f = doc_features.setdefault(
            doc_id, {"bm25": [], "vec": [], "chunks": []}
        )

        f["vec"].append(h["vector_score"])

    # Final scoring
    results = []
    for doc_id, f in doc_features.items():
        max_bm25 = max(f["bm25"]) if f["bm25"] else 0.0
        max_vec = max(f["vec"]) if f["vec"] else 0.0

        score = 0.5 * max_bm25 + 0.5 * max_vec

        snippet = (
            max(f["chunks"], key=lambda x: x["score"])
            if f["chunks"] else None
        )

        results.append({
            "doc_id": doc_id,
            "score": score,
            "snippet": snippet
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)


# ------------------------------------------------------------
# Public search API
# ------------------------------------------------------------
def search(query: str, lang: str, top_k: int = 10) -> List[Dict]:
    bm25_hits: List[Dict] = []
    ann_hits: List[Dict] = []

    t1 = threading.Thread(
        target=lambda: bm25_hits.extend(
            bm25_search(query, lang)
        )
    )
    t2 = threading.Thread(
        target=lambda: ann_hits.extend(
            ann_search(query)
        )
    )

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    print("[DEBUG] BM25 hits:", len(bm25_hits))
    print("[DEBUG] ANN hits:", len(ann_hits))

    results = union_and_aggregate(bm25_hits, ann_hits, meta_index)
    return results[:top_k]


# ------------------------------------------------------------
# CLI test
# ------------------------------------------------------------
if __name__ == "__main__":
    q = "தமிழ்நாடு அரசு புதிய திட்டம்"
    lang = "ta"

    res = search(q, lang)
    for r in res:
        print("\nDOC:", r["doc_id"])
        print("SCORE:", round(r["score"], 4))
        if r["snippet"]:
            print("PAGE:", r["snippet"]["page_no"])
            print("TEXT:", r["snippet"]["text"][:300])
