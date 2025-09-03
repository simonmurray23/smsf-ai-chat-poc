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

# --- New: matching & prompts config ---
MAX_SNIPPETS = _int_env("MAX_SNIPPETS", 3)

# Prompt templates
CORPUS_INSTRUCTIONS = (
    "You are an educational assistant for Australian SMSFs.\n"
    "You will be given APPROVED SOURCE EXCERPTS and a USER QUESTION.\n"
    "Use ONLY the excerpts as factual ground truth; if something is not covered, say so.\n"
    "Avoid financial advice or recommendations. Do not include citations inline.\n"
    "Write concise markdown."
)

FALLBACK_INSTRUCTIONS = (
    "You are an educational assistant for Australian SMSFs.\n"
    "No approved excerpts are available.\n"
    "Provide neutral, general educational information only. Avoid financial advice.\n"
    "Do not include a disclaimer (the system appends it). No citations. Concise markdown."
)

DEFLECTION_LEAD = (
    "I can’t provide personal financial advice. Here’s general information to help you understand the topic."
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
    - nested dicts like {"data":{"prompt":"..."}}, etc.
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

# ---------- Body Parsing ----------
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

# ---------- Front-matter Parsing ----------
def parse_front_matter(md_text: str) -> Tuple[Dict[str, Any], str]:
    """
    Forgiving YAML-like front matter parser.
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
        return {}, md_text

    fm_lines = lines[1:end_idx]
    body = "\n".join(lines[end_idx + 1:])

    meta: Dict[str, Any] = {}
    current_key: Optional[str] = None

    for raw in fm_lines:
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        m = re.match(r'^\s*([A-Za-z0-9_.-]+)\s*:\s*(.*)$', line)
        if m:
            key = m.group(1).strip()
            val = (m.group(2) or "").strip()
            current_key = key

            if val == "":
                meta[key] = []
            else:
                if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                    val = val[1:-1]
                low = val.lower()
                if low == "true":
                    meta[key] = True
                elif low == "false":
                    meta[key] = False
                elif val.startswith("[") and val.endswith("]"):
                    try:
                        meta[key] = json.loads(val)
                    except Exception:
                        meta[key] = val
                else:
                    meta[key] = val
            continue

        li = re.match(r'^\s*-\s*(.+)$', line)
        if li and current_key:
            if not isinstance(meta.get(current_key), list):
                prev = meta.get(current_key)
                meta[current_key] = [] if prev in (None, "",) else [str(prev)]
            meta[current_key].append(li.group(1).strip())

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

# ---------- New: matching utilities ----------
def _score_match(question: str, title: str, tags: List[str]) -> int:
    ql = (question or "").lower()
    score = 0
    if title and title.lower() in ql:
        score += 5
    for t in (tags or []):
        if (t or "").lower() in ql:
            score += 1
    if title:
        q_tokens = set(re.findall(r"[a-z0-9]+", ql))
        t_tokens = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
        score += len(q_tokens & t_tokens)
    return score

def find_matches(question: str, index: Dict[str, Dict[str, Any]], max_snippets: int = MAX_SNIPPETS) -> List[Tuple[str, Dict[str, Any]]]:
    scored: List[Tuple[int, str, Dict[str, Any]]] = []
    for sid, meta in (index or {}).items():
        sc = _score_match(question, meta.get("title", ""), meta.get("tags", []))
        if sc > 0:
            scored.append((sc, sid, meta))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [(sid, meta) for _, sid, meta in scored[:max_snippets]]

def guess_path_from_id(snippet_id: str) -> str:
    return f"{FAQ_PREFIX}{snippet_id}.md".replace(" ", "_")

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

# ---------- New: prompt builders + response shaper ----------
def build_corpus_prompt(question: str, excerpts: List[str]) -> str:
    joined = "\n\n---\n\n".join(excerpts)
    return (
        f"{CORPUS_INSTRUCTIONS}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"APPROVED SOURCE EXCERPTS:\n{joined}\n"
    )

def build_fallback_prompt(question: str) -> str:
    return f"{FALLBACK_INSTRUCTIONS}\n\nUSER QUESTION:\n{question}\n"

def apply_deflection_wrapping(text: str, add_neutral_preamble: bool = False, topic_title: Optional[str] = None) -> str:
    pre = DEFLECTION_LEAD
    if add_neutral_preamble and topic_title:
        pre += f" Below is neutral, educational context about **{topic_title}**."
    return f"{pre}\n\n{text}"

def normalize_response(answer_md: str,
                       citations: List[Dict[str, Any]],
                       suggestions: List[Dict[str, str]],
                       source: str) -> Dict[str, Any]:
    with_disc = append_disclaimer(answer_md or "")
    return {
        "answer": with_disc,          # new
        "text": with_disc,            # backward-compat for current UI
        "citations": citations,
        "suggestions": suggestions,
        "source": source,
        "disclaimer": DISCLAIMER,
    }

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

        # Advice-seeking detection (used later)
        advice_detected = is_advice_seeking(user_q)

        # If clearly advice-seeking and no faq_id context, deflect early to save model cost
        if advice_detected and not faq_id:
            msg = apply_deflection_wrapping("Here’s a general overview.")
            return _ok(200, normalize_response(msg, [], top_suggestions(), "titan_fallback"))

        # Suggestions (best-effort)
        try:
            suggestions = top_suggestions()
        except Exception:
            suggestions = []

        citations: List[Dict[str, Any]] = []
        context_text = ""
        source = "titan_fallback"

        if faq_id:
            try:
                title, content, cite = get_faq_doc(faq_id)
                context_text = f"[{title}]\n\n{content}"
                citations.append(cite)
                meta = INDEX_CACHE.get(str(faq_id)) or {}
                snippet_deflection_required = bool(meta.get("deflection_required"))
                topic_title = meta.get("title") or title
                source = "corpus"
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                    return _err(404, "S3 key not found", str(e), {"faq_id": faq_id})
                return _err(500, "Failed to load FAQ", str(e))
            except KeyError:
                return _err(404, "Unknown FAQ id", faq_id)
            except Exception as e:
                return _err(500, "Failed to load FAQ", str(e))
        else:
            # No explicit faq_id → try to match from index
            snippet_deflection_required = False
            topic_title = None
            excerpts: List[str] = []
            if not INDEX_LOADED:
                load_index()
            matches = find_matches(user_q, INDEX_CACHE)
            if matches:
                for sid, meta in matches:
                    s3_key = meta.get("key") or meta.get("s3_key") or meta.get("path") or guess_path_from_id(sid)
                    try:
                        md = s3_read_text(s3_key)
                        fm, body_md = parse_front_matter(md)
                        title = fm.get("title") or meta.get("title") or sid
                        excerpts.append(f"[{title}]\n\n{body_md.strip()}")
                        citations.append({"id": str(sid), "title": title, "key": s3_key})
                        fups = meta.get("followups") or fm.get("followups") or []
                        if fups and isinstance(fups, list):
                            for f_id in fups:
                                f_meta = INDEX_CACHE.get(str(f_id), {})
                                suggestions.append({"id": str(f_id), "title": f_meta.get("title", str(f_id))})
                        if (meta.get("deflection_required") is True) or (fm.get("deflection_required") is True):
                            snippet_deflection_required = True
                            topic_title = topic_title or title
                    except Exception:
                        continue

                if excerpts:
                    # Build a corpus-constrained prompt
                    composed = build_corpus_prompt(user_q, excerpts)
                    payload = titan_payload(composed)
                    try:
                        model_out = call_titan(payload)
                    except ClientError as e:
                        return _err(502, "Bedrock invoke failed", str(e))
                    except Exception as e:
                        return _err(502, "Bedrock error", str(e))

                    # Guardrails
                    if advice_detected or snippet_deflection_required:
                        model_out = apply_deflection_wrapping(
                            model_out, add_neutral_preamble=True, topic_title=topic_title
                        )
                    return _ok(200, normalize_response(model_out, citations, suggestions, "corpus"))
                # else: no readable excerpts → continue to fallback

        # Compose prompt for Titan (faq_id path OR pure fallback)
        if context_text:
            # faq_id path with explicit context
            preface = CORPUS_INSTRUCTIONS
            composed = (
                f"{preface}\n\nUSER QUESTION:\n{user_q}\n\n"
                f"APPROVED SOURCE EXCERPTS:\n{context_text}\n"
            )
            source = "corpus"
        else:
            # fallback (no snippets)
            composed = build_fallback_prompt(user_q)
            source = "titan_fallback"

        payload = titan_payload(composed)

        try:
            model_out = call_titan(payload)
        except ClientError as e:
            return _err(502, "Bedrock invoke failed", str(e))
        except Exception as e:
            return _err(502, "Bedrock error", str(e))

        # Guardrails for faq_id route as well
        if advice_detected and source == "corpus":
            topic_title = (citations[0]["title"] if citations else None)
            model_out = apply_deflection_wrapping(model_out, add_neutral_preamble=True, topic_title=topic_title)

        return _ok(200, normalize_response(model_out, citations, suggestions, source))

    except ClientError as e:
        logger.error("AWS ClientError: %s", e, exc_info=True)
        return _err(502, "Upstream AWS error", str(e))
    except Exception as e:
        logger.error("Unhandled error: %s\n%s", e, traceback.format_exc())
        return _err(500, "Internal error", str(e))
