# tools/build_chunks.py
# Mini-RAG v1 chunk builder: 1 chunk per Markdown file
# Usage:
#   python tools/build_chunks.py --src corpus_local --out rag/chunks.json
# Defaults:
#   --src ./corpus_local
#   --out ./rag/chunks.json

import argparse
import json
import os
import re
from pathlib import Path

def approx_tokens(s: str) -> int:
    # Very rough heuristic: ~4 chars per token
    return max(1, int(len(s) / 4))

def md_title_from_text(text: str, fallback: str) -> str:
    # First Markdown heading (# ... ) wins; else filename sans extension
    for line in text.splitlines():
        m = re.match(r'^\s*#\s+(.*)\s*$', line)
        if m:
            return m.group(1).strip()
    return Path(fallback).stem.replace('_', ' ').replace('-', ' ').strip()

def load_md(path: Path) -> str:
    # Read as UTF-8; trim excessive whitespace
    text = path.read_text(encoding='utf-8', errors='ignore')
    # Keep it compact but don’t mutilate content; strip trailing spaces
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE).strip()
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="corpus_local", help="Directory containing .md files")
    parser.add_argument("--out", default="rag/chunks.json", help="Output chunks.json path")
    args = parser.parse_args()

    src_dir = Path(args.src).resolve()
    out_path = Path(args.out)

    if not src_dir.exists():
        raise SystemExit(f"Source folder not found: {src_dir}")

    records = []
    md_files = sorted([p for p in src_dir.rglob("*.md") if p.is_file()])
    if not md_files:
        print(f"Warning: no .md files found under {src_dir}")

    for md in md_files:
        text = load_md(md)
        title = md_title_from_text(text, md.name)
        basename = md.name  # e.g., faq_smsf_differences.md
        rec = {
            "file": str(md.relative_to(src_dir)).replace("\\", "/"),  # keep a human hint to the original location
            "key": f"rag/{basename}",  # <- aligns with Lambda's citation key expectation
            "title": title,
            "text": text,
            "tokens": approx_tokens(text)
        }
        records.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Built {len(records)} chunk(s) → {out_path}")

if __name__ == "__main__":
    main()
