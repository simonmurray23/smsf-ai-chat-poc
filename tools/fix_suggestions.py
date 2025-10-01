import re, sys, io, os

p = sys.argv[1] if len(sys.argv) > 1 else "frontend/index.html"
src = io.open(p, "r", encoding="utf-8").read()

changed = False

# A) Remove calls to bindSuggestionButtons();
new = re.sub(r'[ \t]*bindSuggestionButtons\(\);\s*(?://.*)?\n', '', src)
changed |= (new != src)
src = new

# B) Remove the bindSuggestionButtons function entirely
pattern_func = re.compile(
    r'\s*function\s+bindSuggestionButtons\s*\(\s*\)\s*\{.*?\n\s*\}\s*',
    re.S
)
new = pattern_func.sub('\n', src)
changed |= (new != src)
src = new

# C) Replace renderSuggestions with controlled re-render
pattern_render = re.compile(
    r'function\s+renderSuggestions\s*\(\s*sugs\s*\)\s*\{.*?\n\s*\}',
    re.S
)
replacement_render = """function renderSuggestions(sugs){
  if(!Array.isArray(sugs) || sugs.length === 0){
    els.suggestions.replaceChildren(); // controlled clear
    return;
  }
  const frag = document.createDocumentFragment();
  for (const s of sugs){
    const btn = document.createElement('button');
    btn.className = 'px-3 py-1 rounded-full text-sm border border-slate-300 hover:bg-slate-50';
    btn.type = 'button';
    btn.dataset.faq = (s && typeof s === 'object') ? (s.id || s.faq_id || '') : String(s);
    btn.textContent = (s && typeof s === 'object') ? (s.title || s.id || s.faq_id || '') : String(s);
    frag.appendChild(btn);
  }
  els.suggestions.replaceChildren(frag);
}"""
if pattern_render.search(src):
    new = pattern_render.sub(replacement_render, src)
    changed |= (new != src)
    src = new

# D) Strip any per-button onclick for suggestions (we rely on delegation)
# (comment them if they target askFaq to be safe)
src = re.sub(
    r'(\bbtn\.onclick\s*=\s*\(\)\s*=>\s*askFaq\([^)]*\)\s*;)',
    r'/* delegated: \1 */',
    src
)

# E) Ensure delegated listener is present (insert once before </script>)
if 'els.suggestions.dataset.delegated' not in src:
    delegation = """
    (function attachSuggestionDelegationOnce(){
      if (typeof els !== 'undefined' && els && els.suggestions && !els.suggestions.dataset.delegated){
        els.suggestions.addEventListener('click', (e) => {
          const btn = e.target.closest('button[data-faq]');
          if (!btn) return;
          const faqId = btn.dataset.faq || '';
          if (faqId && typeof askFaq === 'function'){
            askFaq(faqId);
          } else {
            console.warn('Suggestion click ignored (missing askFaq or faqId)', { faqId });
          }
        }, { passive: true });
        els.suggestions.dataset.delegated = '1';
      }
    })();
"""
    src = src.replace('</script>', delegation + '\n</script>')
    changed = True

# Write back
io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("OK: frontend/index.html updated." if changed else "No changes needed.")
