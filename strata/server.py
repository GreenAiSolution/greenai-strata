"""A tiny stdlib web UI that shows *why* each result ranked where it did.

No framework, no build step, no CDN — one file, one port. The interesting part
is not the search box; it is that every row exposes its BM25 score with the
contributing terms, its cosine similarity, the fused value with each leg's
share, and the re-ranker's verdict. You can drag alpha and watch the ordering
move.
"""

from __future__ import annotations

import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .pipeline import SearchEngine
from .rerank import ClaudeReranker, LocalCrossEncoder

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>STRATA — layered retrieval</title>
<style>
:root{--bg:#0d1014;--panel:#141a21;--line:#232d38;--fg:#e6edf3;--dim:#8b98a5;
--lex:#f0a868;--vec:#6aa9f0;--rr:#7bd88f;--accent:#7bd88f}
@media(prefers-color-scheme:light){:root{--bg:#f7f8fa;--panel:#fff;--line:#e3e7ec;
--fg:#12181f;--dim:#5b6672}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:18px 22px;border-bottom:1px solid var(--line);display:flex;
gap:14px;align-items:baseline;flex-wrap:wrap}
h1{font-size:15px;margin:0;letter-spacing:.14em}
.sub{color:var(--dim);font-size:12px}
main{max-width:1080px;margin:0 auto;padding:20px 22px 60px}
form{display:flex;gap:8px;margin-bottom:12px}
input[type=text]{flex:1;background:var(--panel);border:1px solid var(--line);
color:var(--fg);padding:11px 13px;border-radius:7px;font:inherit}
input[type=text]:focus{outline:none;border-color:var(--accent)}
button{background:var(--accent);color:#06120b;border:0;padding:0 18px;
border-radius:7px;font:inherit;font-weight:700;cursor:pointer}
.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;
color:var(--dim);font-size:12px;margin-bottom:16px}
select,input[type=range]{background:var(--panel);border:1px solid var(--line);
color:var(--fg);border-radius:6px;font:inherit;font-size:12px;padding:4px 6px}
input[type=range]{padding:0;width:130px;vertical-align:middle}
.trace{color:var(--dim);font-size:11.5px;margin-bottom:14px;
border-left:2px solid var(--line);padding-left:10px}
.hit{background:var(--panel);border:1px solid var(--line);border-radius:9px;
padding:13px 15px;margin-bottom:10px}
.hit h3{margin:0 0 2px;font-size:13.5px;font-weight:600}
.path{color:var(--dim);font-size:11.5px;margin-bottom:9px;word-break:break-all}
.bars{display:grid;grid-template-columns:58px 1fr 62px;gap:7px;
align-items:center;font-size:11px;color:var(--dim);margin-bottom:3px}
.track{height:7px;background:var(--line);border-radius:4px;overflow:hidden}
.fill{height:100%;border-radius:4px}
.snippet{margin-top:9px;color:var(--fg);opacity:.85;font-size:12.5px;
font-family:ui-sans-serif,system-ui}
.terms{margin-top:7px;font-size:11px;color:var(--dim)}
.chip{display:inline-block;border:1px solid var(--line);border-radius:11px;
padding:1px 8px;margin:2px 3px 0 0}
.why{margin-top:6px;font-size:11.5px;color:var(--rr)}
.rank{float:right;color:var(--dim);font-size:11px}
.empty{color:var(--dim);padding:30px 0;text-align:center}
</style></head><body>
<header>
  <h1>S T R A T A</h1>
  <span class="sub" id="meta">loading…</span>
</header>
<main>
  <form id="f" autocomplete="off">
    <input type="text" id="q" placeholder="ask the corpus something…" autofocus>
    <button>search</button>
  </form>
  <div class="controls">
    <label>mode
      <select id="mode">
        <option value="rrf" selected>rrf</option>
        <option value="hybrid">hybrid (weighted)</option>
        <option value="bm25">bm25 only</option>
        <option value="vector">vectors only</option>
      </select></label>
    <label>alpha <input type="range" id="alpha" min="0" max="1" step="0.05" value="0.35">
      <span id="av">0.35</span> <span style="opacity:.6">0=lexical 1=semantic</span></label>
    <label>rerank
      <select id="rerank">
        <option value="none">off</option>
        <option value="local">local cross-encoder</option>
        <option value="claude">claude judge</option>
      </select></label>
  </div>
  <div class="trace" id="trace"></div>
  <div id="out"></div>
</main>
<script>
const $ = s => document.querySelector(s);
$('#alpha').oninput = e => { $('#av').textContent = (+e.target.value).toFixed(2); };
fetch('/api/meta').then(r=>r.json()).then(m=>{
  $('#meta').textContent =
    `${m.chunks.toLocaleString()} chunks · ${m.docs} files · ${m.embedder} d=${m.dim}` +
    ` · ${m.has_ann ? 'hnsw' : 'exact'}`;
});
function bar(cls,val,max){
  const pct = Math.max(0, Math.min(100, max ? val/max*100 : 0));
  return `<div class="track"><div class="fill" style="width:${pct}%;background:var(--${cls})"></div></div>`;
}
$('#f').onsubmit = async e => {
  e.preventDefault();
  const q = $('#q').value.trim(); if(!q) return;
  $('#out').innerHTML = '<div class="empty">searching…</div>';
  const p = new URLSearchParams({q, mode:$('#mode').value,
    alpha:$('#alpha').value, rerank:$('#rerank').value});
  const r = await fetch('/api/search?'+p);
  const d = await r.json();
  if(d.error){ $('#out').innerHTML = `<div class="empty">${d.error}</div>`; return; }
  const t = d.trace;
  $('#trace').textContent = `${t.mode}${t.mode==='hybrid'?' α='+t.alpha:''}` +
    `${t.reranker? ' → '+t.reranker : ''} · ${t.candidates} candidates · ` +
    Object.entries(t.timings_ms).map(([k,v])=>`${k} ${v}ms`).join('  ');
  const maxB = Math.max(...d.hits.map(h=>h.bm25), 1e-6);
  const maxV = Math.max(...d.hits.map(h=>h.vector), 1e-6);
  $('#out').innerHTML = d.hits.map((h,i)=>`
    <div class="hit">
      <span class="rank">#${i+1}</span>
      <h3>${esc(h.title)}</h3>
      <div class="path">${esc(h.doc)} · chunk ${h.doc_id}</div>
      <div class="bars"><span>bm25</span>${bar('lex',h.bm25,maxB)}<span>${h.bm25.toFixed(2)}</span></div>
      <div class="bars"><span>cosine</span>${bar('vec',h.vector,1)}<span>${h.vector.toFixed(3)}</span></div>
      ${h.reranked!==null?`<div class="bars"><span>rerank</span>${bar('rr',h.reranked,1)}<span>${h.reranked.toFixed(3)}</span></div>`:''}
      ${h.terms.length?`<div class="terms">${h.terms.map(t=>`<span class="chip">${esc(t[0])} ${t[1]}</span>`).join('')}</div>`:''}
      ${h.rationale?`<div class="why">▸ ${esc(h.rationale)}</div>`:''}
      <div class="snippet">${esc(h.snippet)}</div>
    </div>`).join('') || '<div class="empty">no matches</div>';
};
function esc(s){return String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));}
</script></body></html>"""


def serve(index_dir: str, host: str = "127.0.0.1", port: int = 8105) -> None:
    engine = SearchEngine.load(index_dir)
    idf = {t: float(engine.bm25.idf[i]) for t, i in engine.bm25.vocab.items()}
    rerankers = {"local": LocalCrossEncoder(idf=idf), "none": None}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_):  # keep the console clean
            pass

        def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urllib.parse.urlparse(self.path)
            params = urllib.parse.parse_qs(parsed.query)

            if parsed.path in ("/", "/index.html"):
                return self._send(PAGE.encode(), "text/html; charset=utf-8")

            if parsed.path == "/api/meta":
                meta = {
                    "chunks": len(engine.corpus),
                    "docs": len({c.doc_id for c in engine.corpus.chunks}),
                    "embedder": engine.embedder.name,
                    "dim": int(engine.vectors.shape[1]),
                    "has_ann": engine.ann is not None,
                }
                return self._send(json.dumps(meta).encode(), "application/json")

            if parsed.path == "/api/search":
                query = (params.get("q") or [""])[0]
                if not query.strip():
                    return self._send(b'{"error":"empty query"}', "application/json", 400)
                name = (params.get("rerank") or ["none"])[0]
                if name == "claude" and "claude" not in rerankers:
                    try:
                        rerankers["claude"] = ClaudeReranker()
                    except Exception as exc:  # no key / no sdk — say so plainly
                        return self._send(
                            json.dumps({"error": f"claude judge unavailable: {exc}"}).encode(),
                            "application/json", 200,
                        )
                try:
                    hits, trace = engine.search(
                        query,
                        k=int((params.get("k") or ["10"])[0]),
                        mode=(params.get("mode") or ["rrf"])[0],
                        alpha=float((params.get("alpha") or ["0.35"])[0]),
                        reranker=rerankers.get(name),
                    )
                except Exception as exc:
                    return self._send(json.dumps({"error": str(exc)}).encode(),
                                      "application/json", 200)
                payload = {"trace": trace.__dict__,
                           "hits": [h.to_dict() for h in hits]}
                return self._send(json.dumps(payload).encode(), "application/json")

            self._send(b"not found", "text/plain", 404)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"STRATA on http://{host}:{port}  "
          f"({len(engine.corpus):,} chunks, {Path(index_dir).name})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
