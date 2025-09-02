import os
import json
import boto3
import datetime
import re
import uuid

S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_KEY = os.getenv("S3_KEY", "sample_corpus.json")
TABLE_NAME = os.getenv("TABLE_NAME")
ALLOW_BEDROCK = os.getenv("ALLOW_BEDROCK", "false").lower() == "true"
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "")

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb") if TABLE_NAME else None
bedrock = boto3.client("bedrock-runtime") if ALLOW_BEDROCK and BEDROCK_MODEL_ID else None

DISCLAIMER = (
    "⚠️ Educational only — Not financial advice. "
    "This chat provides general information about SMSFs and links to approved resources. "
    "It does not consider your personal objectives, financial situation or needs."
)

# Load corpus at init (cold start)
def load_corpus():
    if not S3_BUCKET:
        return []
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    data = json.loads(obj["Body"].read().decode("utf-8"))
    # Expect list of {id, title, topic, content, url}
    return data

CORPUS = []
try:
    CORPUS = load_corpus()
except Exception as e:
    # Fall back to empty corpus; log to CW
    print(f"Failed to load corpus: {e}")


ADVICE_PATTERNS = [
    r"\bshould I\b",
    r"\bwhat should I do\b",
    r"\bis (an )?smsf right for me\b",
    r"\brecommend\b",
    r"\bwhich option\b",
    r"\bpersonal advice\b",
    r"\bcan you tell me if\b",
    r"\bfinancial product advice\b",
    r"\bdo you think I should\b"
]

def is_advice_seeking(text: str) -> bool:
    t = text.lower()
    for pat in ADVICE_PATTERNS:
        if re.search(pat, t):
            return True
    return False

def simple_retrieve(query: str, k: int = 3):
    """Naive retrieval: rank by overlap count of unique words (stopwords removed)."""
    if not CORPUS:
        return []
    stop = set(["the","and","a","an","of","to","in","is","it","that","for","on","with","as","by","are","be","or","at","from"])
    q_terms = {w for w in re.findall(r"[a-z0-9]+", query.lower()) if w not in stop}
    scored = []
    for doc in CORPUS:
        content = f"{doc.get('title','')} {doc.get('topic','')} {doc.get('content','')}"
        terms = {w for w in re.findall(r"[a-z0-9]+", content.lower()) if w not in stop}
        score = len(q_terms & terms)
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in scored[:k]]

def build_prompt(query: str, snippets: list):
    sources_block = "\n\n".join([f"- {s.get('title','Untitled')} ({s.get('url','')})\nExcerpt: {s.get('content','')[:400]}" for s in snippets])
    system = (
        "You are an educational-only SMSF assistant for Australian consumers.\n"
        "STRICT RULES:\n"
        "1) Do NOT provide financial advice or recommendations. No 'should', no personal tailoring.\n"
        "2) Use ONLY the provided sources. If insufficient, say so and suggest reading the linked resources.\n"
        "3) Keep answers concise, plain-English, and cite sources by title with links.\n"
        "4) If the user asks for advice, provide a compliant deflection message.\n"
    )
    user = f"User question: {query}\n\nApproved sources:\n{sources_block}\n\nWrite an educational summary with 2–4 concise paragraphs, then list the sources."
    return system, user

def bedrock_answer(query: str, snippets: list):
    if not bedrock:
        return None
    system, user = build_prompt(query, snippets)
    try:
        body = {
            "messages": [
                {"role":"system","content":[{"type":"text","text":system}]},
                {"role":"user","content":[{"type":"text","text":user}]}
            ],
            "max_tokens": 500,
            "temperature": 0.3
        }
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            body=json.dumps(body)
        )
        payload = json.loads(resp["body"].read())
        # Anthropic-format response (Bedrock messages API) shape may vary; handle common structures:
        txt = ""
        if "output" in payload and "message" in payload["output"]:
            parts = payload["output"]["message"].get("content", [])
            txt = " ".join([p.get("text","") for p in parts if p.get("type")=="text"])
        elif "content" in payload:
            parts = payload.get("content", [])
            txt = " ".join([p.get("text","") for p in parts if p.get("type")=="text"])
        return txt or None
    except Exception as e:
        print(f"Bedrock error: {e}")
        return None

def extractive_answer(query: str, snippets: list):
    """Fallback: stitch together the top snippets with a simple summary-like structure."""
    if not snippets:
        return "I couldn't find an exact match in the approved materials. Try browsing the fundamentals or ask a broader question."
    lines = ["Here’s an educational overview based on approved resources:\n"]
    for s in snippets:
        lines.append(f"• {s.get('title','Untitled')}: {s.get('content','')[:350]}")
    lines.append("\nSources:")
    for s in snippets:
        lines.append(f"- {s.get('title','Untitled')} – {s.get('url','')}")
    return "\n".join(lines)

def log_event(session_id: str, query: str, response: str, advice_flag: bool, snippets: list):
    print(json.dumps({
        "event": "chat_turn",
        "session_id": session_id,
        "ts": datetime.datetime.utcnow().isoformat(),
        "advice_flag": advice_flag,
        "query": query,
        "response_preview": response[:200],
        "sources": [s.get("id") for s in snippets]
    }))
    if dynamodb:
        table = dynamodb.Table(TABLE_NAME)
        table.put_item(Item={
            "session_id": session_id,
            "ts": datetime.datetime.utcnow().isoformat(),
            "advice_flag": advice_flag,
            "query": query,
            "response": response,
            "sources": [s.get("id") for s in snippets]
        })

def compliant_deflection():
    return (
        "I can’t provide personal financial advice or recommendations. "
        "If you’re considering an SMSF, it may help to review the educational resources on suitability, "
        "trustee responsibilities, and costs, then speak with a licensed financial adviser for guidance tailored to you."
    )

def lambda_handler(event, context):
    import os, logging
    logger = logging.getLogger()
    logger.setLevel("INFO")
    logger.info({
        "marker": "version_probe",
        "function_version": os.environ.get("AWS_LAMBDA_FUNCTION_VERSION"),
        "invoked_arn": getattr(context, "invoked_function_arn", None),
        "bucket_env": os.environ.get("BUCKET_NAME")  # just to confirm the alias sees it
    })

    try:
        body = event.get("body") or "{}"
        if isinstance(body, str):
            payload = json.loads(body)
        else:
            payload = body
        message = (payload.get("message") or "").strip()
        session_id = payload.get("session_id") or str(uuid.uuid4())

        if not message:
            return {"statusCode": 400, "headers": {"Content-Type":"application/json", "Access-Control-Allow-Origin":"*"}, "body": json.dumps({"error":"Missing 'message'"})}

        advice_flag = is_advice_seeking(message)

        if advice_flag:
            answer = f"{DISCLAIMER}\n\n{compliant_deflection()}"
            log_event(session_id, message, answer, advice_flag, [])
            return {"statusCode": 200, "headers": {"Content-Type":"application/json", "Access-Control-Allow-Origin":"*"}, "body": json.dumps({"session_id": session_id, "answer": answer, "sources": []})}

        snippets = simple_retrieve(message, k=3)

        # Try Bedrock first (if allowed & configured), else fallback
        answer_txt = None
        if bedrock:
            answer_txt = bedrock_answer(message, snippets)
        if not answer_txt:
            answer_txt = extractive_answer(message, snippets)

        answer = f"{DISCLAIMER}\n\n{answer_txt}"
        log_event(session_id, message, answer, advice_flag, snippets)

        sources = [{"id": s.get("id"), "title": s.get("title"), "url": s.get("url")} for s in snippets]
        return {"statusCode": 200, "headers": {"Content-Type":"application/json", "Access-Control-Allow-Origin":"*"}, "body": json.dumps({"session_id": session_id, "answer": answer, "sources": sources})}

    except Exception as e:
        print(f"Error: {e}")
        return {"statusCode": 500, "headers": {"Content-Type":"application/json", "Access-Control-Allow-Origin":"*"}, "body": json.dumps({"error":"Internal error"})}
