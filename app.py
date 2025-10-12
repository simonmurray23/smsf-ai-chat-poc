# app.py  (Python 3.13, AWS Lambda)
# Demo-safe, free-tier-friendly, educational-only SMSF chat backend
# Mini-RAG v1 integrated: S3 rag/chunks.json, Jaccard overlap, top-k context → Titan Text Lite

import os
import json
import logging
import re
from typing import Dict, Any, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from retrieval.router import route_sections, explain_routing
from retrieval.search_bm25 import load_bm25_index, bm25_search
from retrieval.re_rank import re_rank_with_embeddings
from retrieval.oracle import assess_coverage, compose_prompt, render_refusal

# ---------- Logging ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------- Env / Config ----------
# Support BOTH the new names (preferred) and your existing ones.
CORPUS_BUCKET = (os.getenv("CORPUS_BUCKET") or os.getenv("S3_BUCKET") or "").strip()
FAQ_PREFIX    = os.getenv("S3_FAQ_PREFIX", "faq/").strip()
INDEX_KEY     = (os.getenv("INDEX_KEY") or os.getenv("FAQ_INDEX_KEY") or f"{FAQ_PREFIX}index.json").strip()

# ---------- RAG v2 Config ----------
RAG_PIPELINE = os.getenv("RAG_PIPELINE", "baseline")  # baseline | rerank | auto
BM25_MANIFEST_KEY = os.getenv("BM25_MANIFEST_KEY", "rag/v2/bm25/current.manifest.json")
RERANK_MAX_EMBED = int(os.getenv("RERANK_MAX_EMBED", "30"))
RERANK_BUDGET_MS = int(os.getenv("RERANK_BUDGET_MS", "1500"))
RAG_TOP_N = int(os.getenv("RAG_TOP_N", "8"))

# RAG
RAG_PREFIX    = os.getenv("RAG_PREFIX", "rag/").strip() or "rag/"
RAG_INDEX_KEY = (os.getenv("RAG_INDEX_KEY") or f"{RAG_PREFIX}chunks.json").strip()
RAG_TOP_K     = int(os.getenv("RAG_TOP_K", "3"))

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", os.getenv("MODEL_ID", "amazon.titan-text-lite-v1"))
AWS_REGION       = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "ap-southeast-2"))

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*").strip() or "*"

DISCLAIMER = "Educational information only — not financial advice."

# ---------- AWS Clients ----------
s3 = boto3.client("s3")
# Lazy init for Bedrock so we don't require it on corpus path
_bedrock_client = None
def _bedrock():
    global _bedrock_client
    if _bedrock_client is None:
        _bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock_client

# ---------- CORS / HTTP helpers ----------
def _cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Requested-With",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
        "Access-Control-Max-Age": "3600",
    }

def _http(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": status, "headers": _cors_headers(), "body": json.dumps(body, ensure_ascii=False)}

# ---------- Frozen Contract ----------
def _contract(
    *,
    source: str,                   # "corpus" | "fallback"
    answer: str,                   # markdown or plain text
    citations: List[Dict[str, str]],
    suggestions: List[Dict[str, str]],
) -> Dict[str, Any]:
    return {
        "source": "corpus" if source == "corpus" else "fallback",
        "answer": answer or "",
        "citations": citations if isinstance(citations, list) else [],
        "suggestions": suggestions if isinstance(suggestions, list) else [],
        "disclaimer": DISCLAIMER,
    }

# ---------- Body parsing ----------
def _parse_body(event: Dict[str, Any]) -> Dict[str, Any]:
    body_raw = event.get("body")
    if body_raw is None:
        return {}
    if event.get("isBase64Encoded"):
        import base64
        try:
            body_raw = base64.b64decode(body_raw).decode("utf-8", errors="replace")
        except Exception:
            pass
    try:
        parsed = json.loads(body_raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        # Treat raw as prompt text
        return {"prompt": str(body_raw)}

# ---------- S3 helpers ----------
def _s3_read_text(key: str) -> str:
    obj = s3.get_object(Bucket=CORPUS_BUCKET, Key=key)
    data = obj["Body"].read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")

def _strip_front_matter(md: str) -> str:
    if not md:
        return md
    text = md.lstrip()
    if not text.startswith('---'):
        return md
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            end = i
            break
    if end is None:
        return md
    return "\n".join(lines[end + 1:]).lstrip("\n")

# ---------- index.json cache & access ----------
_INDEX: Optional[Dict[str, Dict[str, Any]]] = None

def _normalize_entry(fid: str, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized index entry: {id, title, key, url, suggestions:[...]}"""
    entry = entry or {}
    key = entry.get("key") or entry.get("s3_key") or entry.get("path") or f"{FAQ_PREFIX}{fid}.md"
    title = entry.get("title") or fid
    url = entry.get("url") or entry.get("link") or ""
    suggestions = entry.get("suggestions") or entry.get("followups") or []
    # Only keep string ids in suggestions
    norm_sugg = [s for s in suggestions if isinstance(s, str)]
    return {"id": fid, "title": title, "key": key, "url": url, "suggestions": norm_sugg}

def _load_index() -> Dict[str, Dict[str, Any]]:
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    if not CORPUS_BUCKET or not INDEX_KEY:
        raise RuntimeError("CORPUS_BUCKET and INDEX_KEY environment variables are required")

    raw = json.loads(_s3_read_text(INDEX_KEY))
    norm: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        # { "faq.id": { ... }, ... } OR { "items": [ {...}, ... ] }
        if "items" in raw and isinstance(raw["items"], list):
            for it in raw["items"]:
                if isinstance(it, dict) and it.get("id"):
                    fid = str(it["id"])
                    norm[fid] = _normalize_entry(fid, it)
        else:
            for fid, meta in raw.items():
                fid = str(fid)
                if isinstance(meta, dict):
                    norm[fid] = _normalize_entry(fid, meta)
    elif isinstance(raw, list):
        for it in raw:
            if isinstance(it, dict) and it.get("id"):
                fid = str(it["id"])
                norm[fid] = _normalize_entry(fid, it)

    _INDEX = norm
    logger.info("index.json loaded: %d entries", len(_INDEX))
    return _INDEX

def _get_entry(faq_id: str) -> Optional[Dict[str, Any]]:
    idx = _load_index()
    return idx.get(str(faq_id))

def _build_suggestions(ids: List[str], current_id: str) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    seen = {current_id}
    idx = _load_index()
    for sid in ids or []:
        if sid in seen:
            continue
        ent = idx.get(sid)
        if not ent:
            continue
        out.append({"id": ent["id"], "title": ent["title"]})
        seen.add(sid)
    return out

# ---------- RAG helpers (Mini-RAG v1) ----------
_RAG_CHUNKS: Optional[List[Dict[str, Any]]] = None

def _load_rag_chunks() -> List[Dict[str, Any]]:
    """Load & cache s3://<bucket>/<RAG_INDEX_KEY> which is a list of {file,key,title,text,tokens}."""
    global _RAG_CHUNKS
    if _RAG_CHUNKS is not None:
        return _RAG_CHUNKS
    try:
        raw = _s3_read_text(RAG_INDEX_KEY)
        data = json.loads(raw)
        chunks: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for c in data:
                if not isinstance(c, dict):
                    continue
                file = c.get("file") or ""
                key  = c.get("key") or (f"{RAG_PREFIX}{file}" if file else "")
                title = c.get("title") or (file or key or "RAG")
                text  = c.get("text") or ""
                tokens = c.get("tokens") or 0
                chunks.append({"file": file, "key": key, "title": title, "text": str(text), "tokens": tokens})
        _RAG_CHUNKS = chunks
        logger.info("RAG chunks loaded: %d", len(_RAG_CHUNKS))
        return _RAG_CHUNKS
    except Exception as e:
        logger.warning("Failed to load RAG index (%s): %s", RAG_INDEX_KEY, e)
        _RAG_CHUNKS = []
        return _RAG_CHUNKS

_token_rx = re.compile(r"[a-z0-9]+", re.I)

def _token_set(text: str) -> set:
    return set(_token_rx.findall((text or "").lower()))

def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    union = a | b
    return len(inter) / len(union)

def _select_top_chunks(prompt: str, chunks: List[Dict[str, Any]], k: int) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    q = _token_set(prompt)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for c in chunks:
        s = _jaccard(q, _token_set(c.get("text", "")))
        if s > 0:
            scored.append((s, c))
    if not scored:
        return [], None
    scored.sort(key=lambda x: x[0], reverse=True)
    top = [c for _, c in scored[: max(1, k)]]
    return top, top[0] if top else None

def _truncate(s: str, n: int) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[: max(0, n-3)] + "..."

def _build_context_block(top_chunks: List[Dict[str, Any]]) -> str:
    # Keep context concise to fit Titan Lite comfortably
    parts = []
    for i, c in enumerate(top_chunks, 1):
        title = c.get("title") or f"Chunk {i}"
        text  = _truncate(c.get("text", ""), 2600)  # conservative cap
        parts.append(f"### {title}\n{text}")
    return "\n\n".join(parts)

def _compose_prompt(user_prompt: str, context_block: str) -> str:
    ctx = ""
    if context_block:
        ctx = (
            "Use the following context to answer. If the answer is not in the context, respond generally without "
            "giving personal advice.\n\n" + context_block + "\n\n"
        )
    return (
        "You are an educational assistant for SMSF (Self-Managed Super Funds) topics in Australia.\n"
        "Answer concisely in plain English. Use bullet points where it helps. Do NOT provide financial advice.\n\n"
        f"{ctx}"
        f"User question:\n{user_prompt}\n"
    )

# ---------- Bedrock (fallback path only) ----------
def _titan_generate(prompt: str, temperature: float = 0.2, max_tokens: int = 512) -> str:
    """Minimal Titan Text Lite call; return plain string. Graceful on errors."""
    try:
        body = {
            "inputText": str(prompt),
            "textGenerationConfig": {
                "maxTokenCount": int(max_tokens),
                "temperature": float(temperature),
                "topP": 0.9
            }
        }
        resp = _bedrock().invoke_model(
            modelId=BEDROCK_MODEL_ID,
            accept="application/json",
            contentType="application/json",
            body=json.dumps(body).encode("utf-8"),
        )
        payload = json.loads(resp.get("body", b"{}").read().decode("utf-8", errors="replace"))
        results = payload.get("results") or []
        if results and isinstance(results, list):
            txt = (results[0].get("outputText") or "").strip()
            return txt or "No content returned by the model."
        return (payload.get("outputText") or "").strip() or "No content returned by the model."
    except Exception as e:
        logger.warning("Bedrock invocation failed: %s", e)
        return "General educational overview unavailable right now. Please try again later."

# ---------- Lambda Handler ----------
def handler(event, context):
    # CORS preflight
    if (event.get("httpMethod") or "").upper() == "OPTIONS":
        return {
            "statusCode": 204,
            "headers": _cors_headers(),
            "body": ""
        }

    if (event.get("httpMethod") or "").upper() != "POST":
        return _http(405, _contract(
            source="fallback",
            answer="Method not allowed. Use POST.",
            citations=[],
            suggestions=[],
        ))

    try:
        body = _parse_body(event)
        faq_id = (body.get("faq_id") or "").strip() if isinstance(body, dict) else ""
        prompt = (body.get("prompt") or "").strip() if isinstance(body, dict) else ""

        # ---------- STRICT SHORT-CIRCUIT (faq_id present) ----------
        if faq_id:
            try:
                entry = _get_entry(faq_id)
                if not entry:
                    # Unknown id → graceful fallback (still contract)
                    return _http(200, _contract(
                        source="fallback",
                        answer=f"Sorry — I couldn’t find a snippet for **{faq_id}**.",
                        citations=[],
                        suggestions=[],
                    ))
                # Load snippet markdown from S3
                md_raw = _s3_read_text(entry["key"])
                md = _strip_front_matter(md_raw)
                # Build suggestions from index (de-duped)
                sugg = _build_suggestions(entry.get("suggestions", []), entry["id"])
                # Build single citation
                citations = [{
                    "title": entry.get("title", entry["id"]),
                    "key": entry["key"],
                    "url": entry.get("url", "")
                }]
                return _http(200, _contract(
                    source="corpus",
                    answer=md,
                    citations=citations,
                    suggestions=sugg,
                ))
            except (ClientError, BotoCoreError) as e:
                # S3 read failures → fallback contract
                logger.warning("FAQ load failed: %s", e)
                return _http(200, _contract(
                    source="fallback",
                    answer=f"Failed to load snippet for **{faq_id}**.",
                    citations=[],
                    suggestions=[],
                ))

        # ---------- Free-prompt path (no faq_id) ----------
        if prompt:
            try:
                # --- RAG v2 baseline: Router → BM25 ---
                sections, routing_why = explain_routing(prompt)
                logger.info("router.sections=%s why=%s", sections, routing_why)

                # Load/refresh BM25 index from S3 (ETag/TTL handled inside)
                load_bm25_index(_s3_client, CORPUS_BUCKET, BM25_MANIFEST_KEY)

                # Retrieve lexical candidates
                candidates = bm25_search(prompt, sections, top_k=50)

                # If nothing came back, fall through to v1
                if not candidates:
                    raise RuntimeError("bm25_empty")

                # --- Optional embeddings re-rank (learning step) ---
                use_rerank = RAG_PIPELINE in ("rerank", "auto")
                if use_rerank:
                    try:
                        hits = re_rank_with_embeddings(
                            prompt,
                            candidates,
                            top_n=RAG_TOP_N,
                            max_embed=RERANK_MAX_EMBED,
                            budget_ms=RERANK_BUDGET_MS,
                        )
                    except Exception as e:
                        logger.warning("re_rank failed, using BM25 only: %s", e)
                        hits = candidates[:RAG_TOP_N]
                else:
                    hits = candidates[:RAG_TOP_N]

                # --- Oracle: check coverage from snippets-only ---
                snippets = []
                for h in hits:
                    # Expect BM25 docs to include S3 'key' for citation; if your builder
                    # doesn’t yet include it, add it to docs.jsonl.gz records.
                    snippets.append({
                        "id": h.id,
                        "section": h.section,
                        "title": h.title,
                        "text": h.text,
                        "citations": h.citations,
                        "key": getattr(h, "key", None),  # safe if absent
                    })

                ok, why = assess_coverage([s["text"] for s in snippets], min_chars=600, min_sources=2)
                logger.info("oracle.coverage ok=%s why=%s", ok, why)

                if not ok:
                    refusal = render_refusal(
                        prompt,
                        suggestions=[
                            "Narrow the question (e.g., specify the fee or rule).",
                            "Ask about one step (setup/fees/trustees/compliance).",
                        ],
                    )
                    return _http(200, _contract(
                        source="fallback",
                        answer=refusal["message"],
                        citations=[],
                        suggestions=refusal["suggestions"],
                    ))

                # --- Compose strict snippets-only prompt for Titan ---
                titan_prompt = compose_prompt(prompt, snippets)

                text = _titan_generate(
                    titan_prompt,
                    float(body.get("temperature", 0.2)),
                    int(body.get("max_tokens", 512))
                )

                # Build citations (top 3 unique sources)
                citations: List[Dict[str, str]] = []
                seen = set()
                for s in snippets[:3]:
                    key = s.get("key")
                    title = s.get("title") or s.get("id")
                    if title in seen:
                        continue
                    seen.add(title)
                    citations.append({
                        "title": title,
                        "key": key or "",
                        "url": f"s3://{CORPUS_BUCKET}/{key}" if key else ""
                    })

                return _http(200, _contract(
                    source="corpus",           # snippets-only answer path
                    answer=text,
                    citations=citations,
                    suggestions=[],            # keep your existing suggestions hook if needed
                ))

            except Exception as e:
                # Any failure in v2 path → fall back to your existing Mini-RAG v1
                logger.warning("RAG v2 pipeline failed, falling back to v1: %s", e)

                chunks = _load_rag_chunks()
                top_chunks, top_chunk = _select_top_chunks(prompt, chunks, RAG_TOP_K)
                context_block = _build_context_block(top_chunks) if top_chunks else ""
                titan_prompt = _compose_prompt(prompt, context_block)

                text = _titan_generate(
                    titan_prompt,
                    float(body.get("temperature", 0.2)),
                    int(body.get("max_tokens", 512))
                )

                citations: List[Dict[str, str]] = []
                if top_chunk and top_chunk.get("key"):
                    citations.append({
                        "title": top_chunk.get("title", "RAG context"),
                        "key": top_chunk["key"],
                        "url": f"s3://{CORPUS_BUCKET}/{top_chunk['key']}"
                    })

                return _http(200, _contract(
                    source="fallback",  # keep as per your current acceptance criteria
                    answer=text,
                    citations=citations,
                    suggestions=[],
                ))

        # ---------- Neither present ----------
        return _http(400, _contract(
            source="fallback",
            answer="Invalid request. Provide either 'faq_id' or 'prompt'.",
            citations=[],
            suggestions=[],
        ))

    except Exception as e:
        logger.exception("Unhandled error")
        return _http(500, _contract(
            source="fallback",
            answer=f"Unexpected error: {e}",
            citations=[],
            suggestions=[],
        ))

# --- entrypoint shim (lets old 'app.lambda_handler' keep working) ---
def lambda_handler(event, context):  # keep legacy handler name alive
    return handler(event, context)
