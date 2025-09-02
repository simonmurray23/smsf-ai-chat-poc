# SMSF Educational Chat – Lightweight AWS Prototype

This starter kit lets you demo a **compliant, educational-only** SMSF chat using AWS Free Tier–friendly services.

## What you get
- **Lambda (Python)**: Chat orchestrator with guardrails (advice detection), simple retrieval over a small corpus in S3, optional Bedrock LLM generation, and logging.
- **Static Web UI**: Minimal HTML/JS chat that calls your API Gateway endpoint.
- **Sample Corpus**: Placeholder JSON snippets to replace with approved content.

---

## High-Level Architecture (MVP)
**Web (S3/CloudFront or any static host)** → **API Gateway (HTTP API)** → **Lambda (Python)** →
- S3 (read-only) for the approved content corpus (`sample_corpus.json`)
- DynamoDB (optional) for chat logs (`smsf_chat_logs` table)
- Bedrock Runtime (optional) to generate friendly summaries **from** your approved content only (RAG-style)

> This demo is **educational-only** by design. The Lambda enforces guardrails and deflects advice-seeking queries.

---

## Deploy – Quick Start (console-friendly)

### 0) Prereqs
- AWS account (Free Tier), Region with Bedrock **optional** (if not enabled, the Lambda will fall back to extractive summaries without an LLM).
- IAM permissions to create Lambda, API Gateway, S3 objects, and (optional) DynamoDB table.

### 1) Create S3 bucket & upload corpus
1. Create a bucket, e.g. `smsf-edu-corpus-<yourid>`
2. Upload `content/sample_corpus.json` to the bucket root.

### 2) (Optional) Create DynamoDB table
- Name: `smsf_chat_logs`
- Partition key: `session_id` (String)
- Sort key: `ts` (String) – ISO timestamp

### 3) Create the Lambda function
- Runtime: **Python 3.12**
- Handler: `app.lambda_handler`
- Upload `lambda/app.py` as the code (or zip and upload).
- **Environment variables**:
  - `S3_BUCKET` = your bucket name
  - `S3_KEY` = `sample_corpus.json`
  - `TABLE_NAME` (optional) = `smsf_chat_logs`
  - `BEDROCK_MODEL_ID` (optional) = e.g. `anthropic.claude-3-haiku-20240307-v1:0` (Region permitting)
  - `ALLOW_BEDROCK` = `true` or `false` (default `false`)
- **Execution role** must allow:
  - `s3:GetObject` on the corpus object
  - `dynamodb:PutItem` on the table (if used)
  - `bedrock:InvokeModel` (only if using Bedrock)

### 4) Create an API Gateway (HTTP API) for the Lambda
- Route: `POST /chat`
- CORS: enable for your web origin

### 5) Host the web UI
- Open `web/index.html` and set `API_URL` to your API invoke URL.
- Host `index.html` on S3 static website hosting, CloudFront, or Replit.

### 6) Try it
- Open the web UI
- Ask: “What are trustee responsibilities in an SMSF?”
- Try advice-seeking: “Should I start an SMSF?” → You should see a **compliant deflection**.

---

## Compliance by Design (in this demo)
- **Hard disclaimer** prefixed to every response.
- **Advice detector** (rules + phrases) blocks personal recommendations and redirects to educational resources.
- **Narrow retrieval**: Only returns content from your approved S3 corpus; no internet access.
- **Audit logging**: Request/response with flags to CloudWatch; optional DynamoDB for query trails.

---

## Files
- `lambda/app.py` – Lambda code
- `web/index.html` – Minimal chat UI (edit `API_URL` before deploying)
- `content/sample_corpus.json` – Replace with approved content (same JSON schema)

---

## Next Steps
- Replace the sample corpus with **approved SMSF snippets**.
- Expand the advice detector phrase list with your Compliance team.
- Add quizzes/progress in the front end.
- Add a Compliance dashboard (Athena/QuickSight over DynamoDB logs).

–––
*Generated on 2025-08-18*
