import sys, os, json, re, pathlib
def title_of(txt): m=re.search(r'^\s*#\s+(.+)$', txt, re.M); return m.group(1).strip() if m else None
def tokens(txt): return len(re.findall(r"\w+", txt))
src_dir, out_json = sys.argv[1], sys.argv[2]
os.makedirs(os.path.dirname(out_json), exist_ok=True)
items=[]
for p in pathlib.Path(src_dir).glob("*.md"):
    t=p.read_text(encoding="utf-8", errors="ignore")
    items.append({"file":p.name,"key":f"rag/{p.name}","title":title_of(t) or p.stem,"text":t,"tokens":tokens(t)})
with open(out_json,"w",encoding="utf-8") as f: json.dump(items,f,ensure_ascii=False,indent=2)
print(f"Wrote {len(items)} chunks -> {out_json}")
