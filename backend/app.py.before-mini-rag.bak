# app.py  (Python 3.13, AWS Lambda)
# Demo-safe, free-tier-friendly, educational-only SMSF chat backend

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple

import boto3
from botocore.exceptions import ClientError, BotoCoreError

# ---------- Logging ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------- Env / Config ----------
# Support BOTH the new names (preferred) and your existing ones.
CORPUS_BUCKET = (os.getenv("CORPUS_BUCKET") or os.getenv("S3_BUCKET") or "").strip()
FAQ_PREFIX    = os.getenv("S3_FAQ_PREFIX", "faq/").strip()
INDEX_KEY     = (os.getenv("INDEX_KEY") or os.getenv("FAQ_INDEX_KEY") or f"{FAQ_PREFIX}index.json").strip()

BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", os.getenv("MODEL_ID", "amazon.titan-text-lite-v1"))
AWS_REGION       = os.getenv("BEDROCK_REGION", os.getenv("AWS_REGION", "ap-southeast-2"))

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*").strip() or "*"

DISCLAIMER = "Educational information only — not financial advice."

# ---------- AWS Clients ----------
s3 = boto3.client("s3")
# Lazy init for Bedrock so we don't require it on corpus path
_bedrock = None

def _bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    return _bedrock

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
                return _http(200, _contract(
                    source="fallback",
                    answer=f"Failed to load snippet for **{faq_id}**.",
                    citations=[],
                    suggestions=[],
                ))

        # ---------- Free-prompt path (no faq_id) ----------
        if prompt:
            text = _titan_generate(prompt, float(body.get("temperature", 0.2)), int(body.get("max_tokens", 512)))
            return _http(200, _contract(
                source="fallback",
                answer=text,
                citations=[],
                suggestions=[],  # keep minimal moving parts
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
