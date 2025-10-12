#!/usr/bin/env python3
# Free-tier-friendly BM25 builder for SMSF AI Chat
import argparse, os, re, json, gzip, hashlib, time, pathlib, sys, io
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Iterable, Optional

# Optional deps for front-matter + S3 publish
try:
    import yaml  # PyYAML
except ImportError:
    yaml = None
try:
    import boto3
    from botocore.exceptions import ClientError, BotoCoreError
except Exception:
    boto3 = None

# ---------- Tokenizer ----------
_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_STOP = set(["the","a","an","and","or","of","to","in","for","on","at","by","with","as","is","are","was","were","be","been","it","this","that"])

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOP]

# ---------- Front-matter ----------
_FM_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n?", re.S)

def parse_front_matter(md: str) -> Tuple[Dict, str]:
    """
    Returns (front_matter_dict, markdown_without_front_matter)
    """
    if not yaml:
        return {}, md
    m = _FM_RE.match(md)
    if not m:
        return {}, md
    try:
        fm = yaml.safe_load(m.group(1)) or {}
    except Exception:
        fm = {}
    body = md[m.end():]
    return fm, body

# ---------- Chunking (heading aware, simple) ----------
_HEADING_RE = re.compile(r"^#{2,6}\s+(.+)$", re.M)

def split_into_snippets(md_body: str) -> List[Tuple[str, str]]:
    """
    Returns list of (anchor, snippet_text) for each H2+/section.
    If no headings exist, returns one pseudo-section.
    """
    positions = [(m.start(), m.group(1).strip()) for m in _HEADING_RE.finditer(md_body)]
    if not positions:
        return [("root", md_body.strip())]
    snippets = []
    for i, (pos, title) in enumerate(positions):
        start = pos
        end = positions[i+1][0] if i+1 < len(positions) else len(md_body)
        chunk = md_body[start:end].strip()
        anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        snippets.append((anchor, chunk))
    return snippets

# ---------- IO helpers ----------
def write_jsonl_gz(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json_gz(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def term_shard(term: str, shards: int) -> int:
    h = hashlib.md5(term.encode("utf-8")).hexdigest()
    return int(h[:2], 16) % shards

# ---------- Builder core ----------
def derive_section_from_path(rel_path: str, fm: Dict) -> str:
    # Prefer FM 'category' if present; else infer from path segment
    sec = (fm.get("category") or "").strip().lower()
    if sec in {"setup","fees","trustees","compliance","faq","guides","laws"}:
        # map broader groups to our router buckets if needed
        mapping = {"faq":"other","guides":"other","laws":"compliance"}
        return mapping.get(sec, sec)
    # infer from path
    parts = rel_path.split(os.sep)
    for p in parts:
        pl = p.lower()
        if pl in {"setup","fees","trustees","compliance"}:
            return pl
    return "other"

def build_docs_and_postings(corpus_dir: str, content_prefix_for_s3: str) -> Tuple[List[dict], Dict[str, Dict]]:
    """
    Returns (docs, postings_by_term) where postings_by_term[term] = {"df": int, "postings":[{"id", "tf"}]}
    """
    docs: List[dict] = []
    tf_by_doc: Dict[str, Counter] = {}
    lengths: Dict[str, int] = {}

    # iterate markdown files
    for path in pathlib.Path(corpus_dir).rglob("*.md"):
        abs_path = str(path)
        rel_path = os.path.relpath(abs_path, corpus_dir)  # e.g., faq/fees.md
        s3_key = os.path.join(content_prefix_for_s3, rel_path).replace("\\", "/").lstrip("/")
        md_raw = read_text(abs_path)
        fm, body = parse_front_matter(md_raw)
        section = derive_section_from_path(rel_path, fm)
        title = (fm.get("title") or path.stem).strip()

        snippets = split_into_snippets(body)
        for anchor, text in snippets:
            snippet_text = text.strip()
            if not snippet_text:
                continue
            doc_id = f"{rel_path}#{anchor}"
            tokens = tokenize(snippet_text)
            if not tokens:
                continue
            c = Counter(tokens)
            tf_by_doc[doc_id] = c
            lengths[doc_id] = len(tokens)

            rec = {
                "id": doc_id,
                "section": section,
                "title": title,
                "text": snippet_text,
                "citations": fm.get("citations") or [],
                "effective_date": fm.get("effective_date") or "",
                "last_reviewed": fm.get("last_reviewed") or "",
                "reviewer": fm.get("reviewer") or "",
                "len": len(tokens),
                "key": s3_key,
                "anchor": anchor
            }
            docs.append(rec)

    # build postings & df
    postings_by_term: Dict[str, Dict] = {}
    df = Counter()
    for doc_id, tfc in tf_by_doc.items():
        for term, tf in tfc.items():
            df[term] += 1

    for term, docfreq in df.items():
        postings = []
        for doc_id, tfc in tf_by_doc.items():
            if term in tfc:
                postings.append({"id": doc_id, "tf": int(tfc[term])})
        postings_by_term[term] = {"df": int(docfreq), "postings": postings}

    # meta stats
    avgdl = (sum(lengths.values()) / max(1, len(lengths)))
    return docs, postings_by_term, len(lengths), avgdl

def write_sharded_terms(terms_dir: str, postings_by_term: Dict[str, Dict], shards: int = 16) -> None:
    buckets: List[Dict[str, List[Tuple[str, Dict]]]] = [defaultdict(list) for _ in range(shards)]
    for term, data in postings_by_term.items():
        idx = term_shard(term, shards)
        buckets[idx][term] = data

    os.makedirs(terms_dir, exist_ok=True)
    for i in range(shards):
        shard_path = os.path.join(terms_dir, f"shard_{i:02d}.json.gz")
        # Store compact dict: term -> {df, postings}
        payload = {t: v for t, v in buckets[i].items()}
        write_json_gz(shard_path, payload)

def main():
    ap = argparse.ArgumentParser(description="Build BM25 index from ./corpus into ./rag/v2/bm25/builds/<ts>")
    ap.add_argument("--corpus", default="./corpus", help="Path to corpus root")
    ap.add_argument("--out", required=True, help="Output build directory (e.g., ./rag/v2/bm25/builds/20251007T090000Z)")
    ap.add_argument("--content-prefix", default="", help="Prefix for S3 content keys (e.g., '' or 'content/')")
    ap.add_argument("--k1", type=float, default=1.5)
    ap.add_argument("--b", type=float, default=0.75)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--publish-s3", action="store_true", help="Upload build + update manifest in S3")
    ap.add_argument("--bucket", default=os.environ.get("CORPUS_BUCKET",""), help="S3 bucket for rag/")
    ap.add_argument("--bm25-prefix", default="rag/v2/bm25", help="S3 prefix where bm25 data lives")
    ap.add_argument("--write-manifest", action="store_true", help="Also write/overwrite current.manifest.json")
    args = ap.parse_args()

    # Build paths
    build_dir = args.out
    docs_path = os.path.join(build_dir, "docs.jsonl.gz")
    terms_dir = os.path.join(build_dir, "terms")
    meta_path = os.path.join(build_dir, "meta.json")

    os.makedirs(build_dir, exist_ok=True)

    print(f"[1/4] Scanning corpus: {args.corpus}")
    docs, postings_by_term, N, avgdl = build_docs_and_postings(args.corpus, args.content_prefix)
    if not docs:
        print("No documents found. Aborting.", file=sys.stderr); sys.exit(1)

    meta = {
        "version_id": os.path.basename(build_dir.rstrip("/")),
        "num_docs": len(docs),
        "N": N,
        "avgdl": round(avgdl, 2),
        "tokenizer": {"lang":"en","lower":True,"strip_punct":True,"stopwords":"minimal"},
        "bm25": {"k1": args.k1, "b": args.b},
        "section_names": sorted(list({d["section"] for d in docs})),
        "builder_version": "bm25-v0.1.0",
        "provenance": {"built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    }

    print(f"[2/4] Writing docs.jsonl.gz ({len(docs)} snippets)")
    write_jsonl_gz(docs_path, docs)

    print(f"[3/4] Writing sharded terms ({args.shards} shards)")
    write_sharded_terms(terms_dir, postings_by_term, args.shards)

    print(f"[4/4] Writing meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.publish_s3:
        if not boto3:
            print("boto3 not installed; cannot publish to S3.", file=sys.stderr); sys.exit(2)
        if not args.bucket:
            print("--bucket is required for --publish-s3", file=sys.stderr); sys.exit(2)

        s3 = boto3.client("s3")
        # Upload build folder recursively
        def upload_file(local_path, bucket_key):
            s3.upload_file(local_path, args.bucket, bucket_key)

        # Upload docs, meta, terms
        base_key = f"{args.bm25-prefix}/builds/{meta['version_id']}".replace("//","/")
        print(f"[publish] Uploading to s3://{args.bucket}/{base_key}/ ...")
        # docs
        upload_file(docs_path, f"{base_key}/docs.jsonl.gz")
        # meta
        upload_file(meta_path, f"{base_key}/meta.json")
        # terms
        for p in pathlib.Path(terms_dir).glob("*.json.gz"):
            upload_file(str(p), f"{base_key}/terms/{p.name}")

        if args.write_manifest:
            manifest_key = f"{args.bm25-prefix}/current.manifest.json".replace("//","/")
            manifest = {
                "version_id": meta["version_id"],
                "build_prefix": f"{args.bm25-prefix}/builds/{meta['version_id']}/".replace("//","/"),
                "files": {
                    "meta": f"{args.bm25-prefix}/builds/{meta['version_id']}/meta.json".replace("//","/"),
                    "docs": f"{args.bm25-prefix}/builds/{meta['version_id']}/docs.jsonl.gz".replace("//","/"),
                    "terms_prefix": f"{args.bm25-prefix}/builds/{meta['version_id']}/terms/".replace("//","/")
                }
            }
            print(f"[publish] Writing manifest s3://{args.bucket}/{manifest_key}")
            s3.put_object(
                Bucket=args.bucket,
                Key=manifest_key,
                Body=json.dumps(manifest).encode("utf-8"),
                ContentType="application/json"
            )

    print("Done.")

if __name__ == "__main__":
    main()
