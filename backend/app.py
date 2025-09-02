# app.py  (Python 3.13, AWS Lambda)
# Demo-safe, free-tier-friendly, educational-only SMSF chat backend

print("Lambda edited version running!")

import os
import json
import re
import logging
import traceback
from typing import Dict, Any, List, Tuple, Optional

import boto3
from botocore.exceptions import ClientError

# ---------- Logging / Boot ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)

FUNCTION_VERSION = os.getenv("AWS_LAMBDA_FUNCTION_VERSION", "$LATEST")
AWS_REGION = os.getenv("AWS_REGION", "ap-southeast-2")
logger.info(
    "Boot: version=%s region=%s model_id=%s",
    FUNCTION_VERSION,
    AWS_REGION,
    os.getenv("MODEL_ID", "amazon.titan-text-lite-v1"),
)

# ---------- Config / Env ----------
S3_BUCKET = (os.getenv("S3_BUCKET") or "").strip()
FAQ_PREFIX = os.getenv("S3_FAQ_PREFIX", "faq/").strip()
FAQ_INDEX_KEY = os.getenv("FAQ_INDEX_KEY", f"{FAQ_PREFIX}index.json")
MODEL_ID = os.getenv("MODEL_ID", "amazon.titan-text-lite-v1")

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

MAX_TOKENS = _int_env("MAX_TOKENS", 384)
TEMPERATURE = _float_env("TEMPERATURE", 0.3)
TOP_P = _float_env("TOP_P", 0.9)

ALLOWED_ORIGIN = os.getenv("ALLOWED_ORIGIN", "*")
DEBUG = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes")

DISCLAIMER = "Educational information only — not financial advice."

ADVICE_RE = re.compile(
    r"(financial advice|give.*advice|should I|what should I|invest|buy|sell|which fund|"
    r"specific product|recommend.*product|tax advice|personal circumstances)",
    re.IGNORECASE,
)

# ---------- AWS Clients ----------
S3 = boto3.client("s3")
BR = boto3.client("bedrock-runtime", region_name=AWS_REGION)

# ---------- In-memory Index Cache ----------
INDEX_CACHE: Dict[str, Dict[str, Any]] = {}
INDEX_LOADED = False

# ---------- HTTP Helpers / CORS ----------
def cors_headers() -> Dict[str, str]:
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN or "*",
        "Access-Control-Allow-Headers": "Content-Type,Authorization,X-Requested-With",
        "Access-Control-Allow-Methods": "OPTIONS,POST",
        "Access-Control-Max-Age": "3600",
    }

def _ok(status: int, body: Dict[str, Any]) -> Dict[str, Any]:
    return {"statusCode": status, "headers": cors_headers(), "body": json.dumps(body)}

def _err(status: int, message: str, detail: str = "", debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = {
        "error": {"code": status, "message": message, **({"detail": detail} if detail else {})},
        "text": "",
        "citations": [],
        "suggestions": [],
        "disclaimer": DISCLAIMER,
    }
    if DEBUG and debug:
        payload["debug"] = debug
    return {"statusCode": status, "headers": cors_headers(), "body": json.dumps(payload)}

# ---------- Tiny utilities ----------
def _safe_print(label: str, value: Any) -> None:
    try:
        logger.info("%s=%s", label, value)
    except Exception:
        logger.info("%s=<unprintable>", label)

def append_disclaimer(text: str) -> str:
    if DISCLAIMER.lower() not in (text or "").lower():
        return (text or "").rstrip() + "\n\n" + DISCLAIMER
    return text

def is_advice_seeking(text: str) -> bool:
    return bool(ADVICE_RE.search(text or ""))

# ---------- Flexible prompt extractor ----------
def _extract_prompt(obj: Any) -> str:
    """
    Try hard to find a user prompt in flexible shapes:
    - {"prompt": "..."} (preferred)
    - {"message"|"input"|"text"|"q"|"query": "..."}
    - {"data":{"prompt":"..."}}, {"body":{"prompt":"..."}}, {"detail":{"payload":{"prompt":"..."}}}, etc.
    - plain string body
    """
    if isinstance(obj, str):
        return obj.strip()

    if isinstance(obj, dict):
        for k in ("prompt", "message", "input", "text", "q", "query"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        for k in ("data", "body", "detail", "payload"):
            inner = obj.get(k)
            if isinstance(inner, dict):
                p = _extract_prompt(inner)
                if p:
                    return p
    return ""

# ---------- Body Parsing (v1/v2 + base64 + JSON + form + double-encoded) ----------
def parse_event_body(event: Dict[str, Any]) -> Dict[str, Any]:
    import base64
    headers = { (k or ""): (v or "") for k, v in (event.get("headers") or {}).items() }
    ctype = headers.get("content-type") or headers.get("Content-Type") or ""
    body_raw = event.get("body")
    if body_raw is None:
        return {}

    if event.get("isBase64Encoded"):
        try:
            body_raw = base64.b64decode(body_raw).decode("utf-8", errors="replace")
        except Exception:
            pass

    if "application/x-www-form-urlencoded" in ctype:
        from urllib.parse import parse_qs
        qs = parse_qs(body_raw, keep_blank_values=True)
        return {k: (v[0] if v else "") for k, v in qs.items()}

    candidate = body_raw
    for _ in range(2):  # handle possible double-encoding
        try:
            parsed = json.loads(candidate or "{}")
            if isinstance(parsed, str):
                candidate = parsed
                continue
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            break

    # Fallback: treat raw as a plain-text prompt
    return {"prompt": str(body_raw)}

# ---------- S3 Helpers ----------
def s3_read_text(key: str) -> str:
    obj = S3.get_object(Bucket=S3_BUCKET, Key=key)
    data = obj["Body"].read()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")

# ---------- Front-matter Parsing (robust, list-aware) ----------
def parse_front_matter(md_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Forgiving YAML-like front matter parser.
    - Supports keys with scalar values: key: value
    - Supports empty list headers followed by dash items:
        key:
          - a
          - b
    - Ignores comments/blank lines.
    - Returns (meta_dict, body_without_front_matter)
    """
    if not md_text:
        return {}, ""
    text = md_text.lstrip()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, md_text

    # Find closing '---' line
    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        # No closing delimiter; treat as no front matter
        return {}, md_text

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:])

    meta: Dict[str, Any] = {}
    current_key: Optional[str] = None

    for raw in fm_lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        # key: value  (value may be empty)
        m = re.match(r'^\s*([A-Za-z0-9_.-]+)\s*:\s*(.*)$', line)
        if m:
            key = m.group(1).strip()
            val = (m.group(2) or "").strip()
            current_key = key

            if val == "":
                # Start of a list (expect following "- item" lines)
                meta[key] = []
            else:
                # Scalar normalization
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                low = val.lower()
                if low == "true":
                    meta[key] = True
                elif low == "false":
                    meta[key] = False
                elif val.startswith("[") and val.endswith("]"):
                    # Try JSON list inline: key: ["a","b"]
                    try:
                        meta[key] = json.loads(val)
                    except Exception:
                        meta[key] = val
                else:
                    meta[key] = val
            continue

        # - list item (under the most recent key)
        li = re.match(r'^\s*-\s*(.+)$', line)
        if li and current_key:
            if not isinstance(meta.get(current_key), list):
                # Convert any prior scalar to a list (preserve if non-empty)
                prev = meta.get(current_key)
                meta[current_key] = [] if prev in (None, "",) else [str(prev)]
            meta[current_key].append(li.group(1).strip())

    # Normalize common list-y fields
    for f in ("tags", "source_urls", "followups"):
        if f not in meta:
            meta[f] = []
        elif isinstance(meta[f], str):
            meta[f] = [meta[f]]

    return meta, body.lstrip("\n\r")

# ---------- Index Loading & Normalisation ----------
def load_index() -> None:
    global INDEX_CACHE, INDEX_LOADED
    try:
        if not S3_BUCKET:
            raise RuntimeError("S3_BUCKET env var is required")
        data = s3_read_text(FAQ_INDEX_KEY)
        raw = json.loads(data)

        normalized: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw, dict):
            for fid, meta in raw.items():
                meta = meta or {}
                meta.setdefault("id", str(fid))
                if not meta.get("key"):
                    meta["key"] = meta.get("s3_key") or meta.get("path") or f"{FAQ_PREFIX}{fid}.md"
                normalized[str(fid)] = meta
        elif isinstance(raw, list):
            for item in raw:
                if not isinstance(item, dict):
                    continue
                fid = str(item.get("id") or item.get("slug") or item.get("name") or "")
                if not fid:
                    continue
                meta = dict(item)
                if not meta.get("key"):
                    meta["key"] = meta.get("s3_key") or meta.get("path") or f"{FAQ_PREFIX}{fid}.md"
                meta["id"] = fid
                normalized[fid] = meta
        else:
            normalized = {}

        INDEX_CACHE = normalized
        INDEX_LOADED = True
        logger.info("FAQ index loaded: %d entries", len(INDEX_CACHE))
    except Exception as e:
        logger.error("Failed to load FAQ index: %s", e)
        INDEX_CACHE = {}
        INDEX_LOADED = False

# Cold start attempt
load_index()

def top_suggestions(n: int = 5) -> List[Dict[str, str]]:
    if not INDEX_LOADED:
        load_index()
    out: List[Dict[str, str]] = []
    for fid, meta in list(INDEX_CACHE.items())[:n]:
        out.append({"id": str(fid), "title": meta.get("title", str(fid))})
    return out

def get_faq_doc(faq_id: str) -> Tuple[str, str, Dict[str, str]]:
    if not INDEX_LOADED:
        load_index()
    meta = INDEX_CACHE.get(str(faq_id))
    if meta:
        key = meta.get("key") or meta.get("s3_key") or meta.get("path") or f"{FAQ_PREFIX}{faq_id}.md"
        title_guess = meta.get("title") or str(faq_id)
    else:
        key = f"{FAQ_PREFIX}{faq_id}.md"
        title_guess = str(faq_id)

    try:
        md = s3_read_text(key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            raise FileNotFoundError(key)
        raise

    fm, content = parse_front_matter(md)
    title = fm.get("title") or title_guess
    citation = {"id": str(faq_id), "title": title, "key": key}
    return title, content, citation

# ---------- Titan Helpers (schema-correct) ----------
def titan_payload(input_text: str) -> Dict[str, Any]:
    return {
        "inputText": str(input_text if input_text is not None else ""),
        "textGenerationConfig": {
            "maxTokenCount": int(MAX_TOKENS),
            "temperature": float(TEMPERATURE),
            "topP": float(TOP_P),
            "stopSequences": [],
        },
    }

def call_titan(payload: Dict[str, Any]) -> str:
    # Emit single-line payload (with input redacted for logs)
    redacted = {
        "inputText": "<redacted>",
        "textGenerationConfig": payload.get("textGenerationConfig", {}),
    }
    logger.info("titan.payload=%s", json.dumps(redacted, separators=(",", ":"), ensure_ascii=False))
    body_json = json.dumps(payload, separators=(",", ":"))
    resp = BR.invoke_model(
        modelId=MODEL_ID,
        body=body_json,
        contentType="application/json",
        accept="application/json",
    )
    raw = resp["body"].read().decode("utf-8", errors="replace")
    parsed = json.loads(raw)
    results = parsed.get("results") or []
    if not results:
        raise RuntimeError("Empty Titan results")
    return (results[0].get("outputText") or "").strip()

# ---------- Lambda Handler ----------
def lambda_handler(event, context):
    try:
        if not S3_BUCKET:
            return _err(500, "Configuration error", "S3_BUCKET env var is required")

        # Method/Path/QS (v1+v2)
        method = (event.get("httpMethod") or (event.get("requestContext", {}).get("http", {}) or {}).get("method") or "").upper()
        path = event.get("path") or event.get("rawPath") or ""
        qsp = event.get("queryStringParameters") or {}
        echo = (qsp.get("echo") or "").lower()

        # CORS preflight
        if method == "OPTIONS":
            return _ok(200, {"ok": True})

        # Enforce POST /prompt route
        if method != "POST" or not path.endswith("/prompt"):
            return _err(404, "Route not found")

        # Echo short-circuits (no Bedrock call)
        if echo == "schema":
            payload = titan_payload("Hello Titan Lite")
            dbg = {"version": context.function_version, "payload": payload}
            return _ok(200, {"text": "Echo: Titan request schema", "citations": [], "suggestions": [], "disclaimer": DISCLAIMER, "debug": dbg})

        if echo == "probe":
            dbg = {
                "function_version": context.function_version,
                "invoked_arn": context.invoked_function_arn,
                "model_id": MODEL_ID,
                "bucket": S3_BUCKET,
                "route": {"method": method, "path": path},
            }
            return _ok(200, {"text": "Echo: probe OK", "citations": [], "suggestions": top_suggestions(), "disclaimer": DISCLAIMER, "debug": dbg})

        # Parse body
        body = parse_event_body(event)
        try:
            sample = (json.dumps(body)[:160] if isinstance(body, dict) else str(body)[:160]).replace("\n", " ")
        except Exception:
            sample = "<unprintable>"
        logger.info("req: ver=%s method=%s path=%s sample=%s", context.function_version, method, path, sample)

        # Flexible prompt + faq_id
        user_q = _extract_prompt(body)
        faq_id = (body.get("faq_id") or "").strip() if isinstance(body, dict) else ""

        if not user_q and not faq_id:
            dbg = {}
            if DEBUG:
                try:
                    headers = { (k or ""): (v or "") for k, v in (event.get("headers") or {}).items() }
                    dbg = {
                        "parsed_body_type": type(body).__name__,
                        "parsed_body_keys": list(body.keys())[:10] if isinstance(body, dict) else [],
                        "content_type": headers.get("content-type") or headers.get("Content-Type"),
                    }
                except Exception:
                    pass
            return _err(400, "Missing 'prompt' or 'faq_id' in request body.", debug=dbg)

        # Fast-path regex deflection to save model cost
        if is_advice_seeking(user_q):
            msg = ("I can’t provide financial, legal, or tax advice. "
                   "Here’s general, educational information instead.")
            return _ok(200, {"text": append_disclaimer(msg), "citations": [], "suggestions": top_suggestions(), "disclaimer": DISCLAIMER})

        # Suggestions (best-effort)
        try:
            suggestions = top_suggestions()
        except Exception:
            suggestions = []

        citations: List[Dict[str, str]] = []
        context_text = ""

        if faq_id:
            try:
                title, content, cite = get_faq_doc(faq_id)
                context_text = f"FAQ: {title}\n\n{content}"
                citations.append(cite)
            except FileNotFoundError as e:
                return _err(404, "S3 key not found", str(e), {"faq_id": faq_id})
            except KeyError:
                return _err(404, "Unknown FAQ id", faq_id)
            except Exception as e:
                return _err(500, "Failed to load FAQ", str(e))

        # Compose prompt for Titan
        preface = (
            "You are an educational assistant for Australian SMSFs. "
            "Answer concisely in plain English. Use provided FAQ context if available. "
            "Do not provide personal financial, legal, or tax advice."
        )
        if context_text:
            composed = f"{preface}\n\nUse this context:\n\n{context_text}\n\nUser: {user_q}\n\nAnswer:"
        else:
            composed = f"{preface}\n\nUser: {user_q}\n\nAnswer based on general SMSF educational information."

        payload = titan_payload(composed)

        try:
            model_out = call_titan(payload)
        except ClientError as e:
            return _err(502, "Bedrock invoke failed", str(e))
        except Exception as e:
            return _err(502, "Bedrock error", str(e))

        text = append_disclaimer(model_out or "")
        return _ok(200, {"text": text, "citations": citations, "suggestions": suggestions, "disclaimer": DISCLAIMER})

    except ClientError as e:
        logger.error("AWS ClientError: %s", e, exc_info=True)
        return _err(502, "Upstream AWS error", str(e))
    except Exception as e:
        logger.error("Unhandled error: %s\n%s", e, traceback.format_exc())
        return _err(500, "Internal error", str(e))
