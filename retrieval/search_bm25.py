from typing import List, Dict, NamedTuple

class DocHit(NamedTuple):
    id: str
    score: float
    section: str
    title: str
    text: str
    citations: list
    len: int

# Module-level caches (filled by loader)
_CACHE = {
    "manifest_etag": None,
    "meta": None,
    "docs": {},            # id -> dict(meta)
    "terms": {},           # term -> { "df": int, "postings": [{"id": str, "tf": int}] }
}

def load_bm25_index(s3_client, bucket: str, manifest_key: str, ttl_seconds: int = 600) -> None:
    """
    Lazily load/refresh BM25 index from S3 into _CACHE.
    TODO: implement: get manifest (ETag), compare with _CACHE['manifest_etag'], download meta/docs/terms as needed.
    """
    # placeholder no-op for now
    return None

def bm25_search(query: str, sections: List[str], top_k: int = 50,
                max_postings_per_term: int = 500) -> List[DocHit]:
    """
    Tokenize query; fetch postings; compute BM25 using _CACHE['meta'] (N, avgdl).
    Filter by sections; return top_k hits.
    TODO: implement tokenizer, scoring, and term-shard fetch.
    """
    return []

def debug_bm25_breakdown(query: str, doc_id: str) -> Dict:
    """
    Return per-term scoring contributions for a given doc_id (audit).
    TODO: implement once bm25_search is ready.
    """
    return {"doc_id": doc_id, "query": query, "terms": []}
