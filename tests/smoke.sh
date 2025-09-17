#!/usr/bin/env bash
set -euo pipefail

echo "Running: $(realpath "$0")"
API="${1:?Usage: ./smoke.sh <api-url>}"
command -v jq >/dev/null || { echo "jq is required"; exit 1; }

pass=true

check() {
  local id="$1"
  local expected="$2"
  echo "→ $id"
  out="$(curl -s -X POST "$API" -H "Content-Type: application/json" -d "{\"faq_id\":\"$id\"}")"
  src="$(jq -r '.source // "MISSING"' <<<"$out")"
  key="$(jq -r '.citations[0].key // "MISSING"' <<<"$out")"
  if [[ "$src" != "corpus" ]]; then
    echo "  FAIL: source=$src (want corpus)"
    pass=false
  fi
  if [[ "$key" != "$expected" ]]; then
    echo "  FAIL: citations[0].key=$key (want $expected)"
    pass=false
  fi
}

# Match your actual index.json mappings:
check "faq.what.is.smsf"       "faq/faq_what_is_smsf.md"
check "faq.smsf.setup"         "faq/faq_smsf_setup.md"
check "faq.smsf.differences"   "faq/faq_smsf_differences.md"

# Free-prompt path sanity check
echo "→ free prompt"
out="$(curl -s -X POST "$API" -H "Content-Type: application/json" \
  -d '{"prompt":"What is an SMSF?","temperature":0.2,"max_tokens":256}')"
src="$(jq -r '.source // "MISSING"' <<<"$out")"
ans="$(jq -r '.answer // ""' <<<"$out")"
if [[ -z "$ans" ]]; then
  echo "  FAIL: missing answer text"
  pass=false
fi
if [[ "$src" != "fallback" && "$src" != "corpus" ]]; then
  echo "  FAIL: source=$src (want fallback or corpus)"
  pass=false
fi

$pass && echo "✅ All good" || { echo "❌ Smoke tests failed"; exit 1; }
