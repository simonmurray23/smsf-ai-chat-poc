#!/usr/bin/env bash
# Mini-RAG aware smoke test for SMSF AI Chat
# Usage:
#   bash tests/smoke.sh "<POST endpoint>"
# Example:
#   bash tests/smoke.sh "https://XXXX.execute-api.ap-southeast-2.amazonaws.com/Prod/chat"

set -u

ENDPOINT="${1:-}"
if [ -z "$ENDPOINT" ]; then
  echo "Usage: bash tests/smoke.sh \"<POST endpoint>\""
  echo "Tip: Use /Prod/chat (new) not legacy /prompt"
  exit 2
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "ERROR: 'jq' is required for this smoke test."
  echo "On Windows Git Bash: choco install jq   (or)  winget install jqlang.jq"
  exit 2
fi

disclaimer="Educational information only — not financial advice."

pass=0
fail=0

post_json () {
  local payload="$1"
  curl -sS -X POST "$ENDPOINT" \
    -H "Content-Type: application/json" \
    --data "$payload"
}

check_common_contract () {
  local json="$1"
  local ok=1

  # Ensure required top-level fields exist with correct types
  jq -e '.answer | type=="string" and (.answer|length>0)' >/dev/null 2>&1 <<<"$json" || ok=0
  jq -e '.source | type=="string"' >/dev/null 2>&1 <<<"$json" || ok=0
  jq -e '.citations | type=="array"' >/dev/null 2>&1 <<<"$json" || ok=0
  jq -e '.suggestions | type=="array"' >/dev/null 2>&1 <<<"$json" || ok=0
  jq -e --arg d "$disclaimer" '.disclaimer == $d' >/dev/null 2>&1 <<<"$json" || ok=0

  [ $ok -eq 1 ]
}

test_faq () {
  local id="$1" ; local label="$2"
  echo "→ $label"
  local req
  req="$(jq -nc --arg id "$id" '{faq_id:$id}')"
  local res
  res="$(post_json "$req")" || true

  local ok=1

  check_common_contract "$res" || ok=0
  jq -e '.source=="corpus"' >/dev/null 2>&1 <<<"$res" || ok=0
  # At least one FAQ citation with key starting "faq/"
  jq -e '[.citations[]?.key|tostring|startswith("faq/")]|any' >/dev/null 2>&1 <<<"$res" || ok=0

  if [ $ok -eq 1 ]; then
    echo "  PASS"
    pass=$((pass+1))
  else
    echo "  FAIL: expected source=corpus, faq/* citation, and full contract"
    echo "  Response: $(jq -c . <<<"$res")"
    fail=$((fail+1))
  fi
}

test_free_prompt_rag () {
  echo "→ free prompt (Mini-RAG)"
  local prompt="Briefly compare trustees vs members in an SMSF."
  local req
  req="$(jq -nc --arg p "$prompt" '{prompt:$p}')"
  local res
  res="$(post_json "$req")" || true

  local ok=1

  check_common_contract "$res" || ok=0
  # By design, free prompt remains "fallback"
  jq -e '.source=="fallback"' >/dev/null 2>&1 <<<"$res" || ok=0
  # Exactly one citation and its key starts with rag/
  jq -e '(.citations|length)==1 and (.citations[0].key|startswith("rag/"))' >/dev/null 2>&1 <<<"$res" || ok=0

  if [ $ok -eq 1 ]; then
    echo "  PASS"
    pass=$((pass+1))
  else
    echo "  FAIL: expected source=fallback, single rag/* citation, and full contract"
    echo "  Response: $(jq -c . <<<"$res")"
    fail=$((fail+1))
  fi
}

echo "Running: $0"
test_faq "faq.what.is.smsf"        "faq.what.is.smsf"
test_faq "faq.smsf.setup"          "faq.smsf.setup"
test_faq "faq.smsf.differences"    "faq.smsf.differences"
test_free_prompt_rag

echo
if [ $fail -eq 0 ]; then
  echo "✅ All good ($pass tests passed)"
  exit 0
else
  echo "❌ Smoke tests failed ($fail failed, $pass passed)"
  exit 1
fi
