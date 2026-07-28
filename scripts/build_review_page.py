"""Build a local, self-contained HTML review page for a precision-adjudication worklist.

Why local HTML rather than a hosted artifact: the crops are large (a P&ID region at 360dpi is
~0.5-4MB each), so inlining 40+ of them as data URIs would make a page tens of MB. A local file
referencing local PNGs opens instantly and has no size ceiling. Verdicts are exported as JSON
via a download button, so nothing needs a server.
"""
from __future__ import annotations
import html, json, os, sys

def build(worklist_json: str, out_html: str) -> int:
    blob = json.load(open(worklist_json))
    items, summary = blob["items"], blob["summary"]
    rows = []
    for it in items:
        crop = it.get("crop_path") or ""
        rel = os.path.basename(crop)
        flags = []
        if it.get("series_disagreement"):
            flags.append('<span class="f warn">ISA-series mismatch</span>')
        src = it.get("source") or ""
        flags.append(f'<span class="f">{"traversal" if src.startswith("path_through_") else "direct segment"}</span>')
        flags.append(f'<span class="f">{html.escape(it.get("edge_class") or "")}</span>')
        if it.get("gap_px") is not None:
            flags.append(f'<span class="f">gap {it["gap_px"]:.0f}px</span>')
        pre = it.get("verdict")
        rows.append(f"""
<article class="card" data-i="{it['index']}" data-pre="{pre or ''}">
  <header>
    <span class="n">{it['index']}/{len(items)}</span>
    <b>{html.escape(str(it.get('a_text') or it['a_id']))}</b>
    <span class="ar">&harr;</span>
    <b>{html.escape(str(it.get('b_text') or it['b_id']))}</b>
    <span class="flags">{''.join(flags)}</span>
  </header>
  <div class="imgwrap"><img loading="lazy" src="{html.escape(rel)}" alt="claim {it['index']}"></div>
  <footer>
    <div class="btns">
      <button data-v="real">1 &middot; real</button>
      <button data-v="not_real">2 &middot; not real</button>
      <button data-v="unsure">3 &middot; unsure</button>
    </div>
    <input class="note" placeholder="note (optional)">
    <span class="state"></span>
  </footer>
</article>""")

    doc = f"""<!doctype html><meta charset=utf-8>
<title>Precision adjudication &mdash; {html.escape(summary.get('sheet_id',''))}</title>
<style>
:root{{--bg:#eef1f3;--card:#f8fafb;--ink:#16232b;--soft:#3f4f57;--mute:#71828a;--rule:#c7d2d6;
--ok:#1f8f5f;--bad:#ad3f3f;--warn:#8a7a2f;--acc:#2b6e8f;
--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace}}
@media(prefers-color-scheme:dark){{:root{{--bg:#0d161c;--card:#131f27;--ink:#e7eef1;--soft:#b9c8cd;
--mute:#7f939b;--rule:#263840;--ok:#5ed19a;--bad:#e2867e;--warn:#d8c26e;--acc:#6fb6d4}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
.top{{position:sticky;top:0;z-index:9;background:var(--card);border-bottom:1px solid var(--rule);
padding:12px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}}
.top h1{{font-size:15px;margin:0}}
.prog{{font-family:var(--mono);font-size:12px;color:var(--soft)}}
.prog b{{color:var(--ink)}}
button{{font:inherit;padding:7px 13px;border:1px solid var(--rule);border-radius:5px;
background:var(--bg);color:var(--ink);cursor:pointer}}
button:hover{{border-color:var(--acc)}}
#export{{background:var(--acc);color:#fff;border-color:var(--acc);font-weight:600}}
.wrap{{max-width:1500px;margin:0 auto;padding:18px 20px 90px;display:flex;flex-direction:column;gap:16px}}
.card{{background:var(--card);border:1px solid var(--rule);border-radius:8px;overflow:hidden}}
.card.done{{opacity:.55}}
.card header{{padding:10px 14px;border-bottom:1px solid var(--rule);display:flex;gap:10px;
align-items:baseline;flex-wrap:wrap}}
.card header b{{font-family:var(--mono);font-size:13px}}
.n{{font-family:var(--mono);font-size:11px;color:var(--mute);min-width:52px}}
.ar{{color:var(--mute)}}
.flags{{margin-left:auto;display:flex;gap:6px;flex-wrap:wrap}}
.f{{font-family:var(--mono);font-size:10px;padding:1px 7px;border-radius:99px;background:var(--bg);
color:var(--mute);border:1px solid var(--rule)}}
.f.warn{{color:var(--warn);border-color:var(--warn)}}
.imgwrap{{overflow:auto;max-height:70vh;background:#fff}}
.imgwrap img{{display:block;max-width:100%}}
.card footer{{padding:10px 14px;border-top:1px solid var(--rule);display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
.btns{{display:flex;gap:8px}}
.note{{flex:1;min-width:180px;padding:6px 9px;border:1px solid var(--rule);border-radius:5px;
background:var(--bg);color:var(--ink);font:inherit}}
.state{{font-family:var(--mono);font-size:12px;font-weight:700}}
.state.real{{color:var(--ok)}} .state.not_real{{color:var(--bad)}} .state.unsure{{color:var(--warn)}}
.card.sel{{outline:2px solid var(--acc);outline-offset:-2px}}
</style>
<div class="top">
  <h1>Precision adjudication &mdash; {html.escape(summary.get('sheet_id',''))}</h1>
  <span class="prog">RED = endpoint A &nbsp; BLUE = endpoint B &nbsp;|&nbsp;
    <b id="done">0</b>/{len(items)} judged &nbsp; real <b id="cr">0</b> &nbsp;
    not-real <b id="cn">0</b> &nbsp; unsure <b id="cu">0</b> &nbsp;
    precision <b id="pp">&ndash;</b></span>
  <button id="export">Download verdicts JSON</button>
</div>
<div class="wrap">{''.join(rows)}</div>
<script>
const V={{}};
const cards=[...document.querySelectorAll('.card')];
cards.forEach(c=>{{
  const pre=c.dataset.pre; if(pre) set(c,pre,true);
  c.querySelectorAll('button[data-v]').forEach(b=>b.onclick=()=>set(c,b.dataset.v));
}});
function set(c,v,quiet){{
  const i=c.dataset.i; V[i]={{verdict:v,note:c.querySelector('.note').value||null}};
  const s=c.querySelector('.state'); s.textContent=v.replace('_',' '); s.className='state '+v;
  c.classList.add('done'); if(!quiet) tally();
  else tally();
}}
function tally(){{
  const vs=Object.values(V); const r=vs.filter(x=>x.verdict==='real').length;
  const n=vs.filter(x=>x.verdict==='not_real').length; const u=vs.filter(x=>x.verdict==='unsure').length;
  done.textContent=vs.length; cr.textContent=r; cn.textContent=n; cu.textContent=u;
  pp.textContent=(r+n)?((r/(r+n))*100).toFixed(1)+'%':'\\u2013';
}}
let cur=0;
function focus(i){{cards.forEach(c=>c.classList.remove('sel'));
  cur=Math.max(0,Math.min(cards.length-1,i)); const c=cards[cur];
  c.classList.add('sel'); c.scrollIntoView({{block:'center',behavior:'smooth'}});}}
focus(0);
document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='1'){{set(cards[cur],'real');focus(cur+1);}}
  if(e.key==='2'){{set(cards[cur],'not_real');focus(cur+1);}}
  if(e.key==='3'){{set(cards[cur],'unsure');focus(cur+1);}}
  if(e.key==='j'||e.key==='ArrowDown')focus(cur+1);
  if(e.key==='k'||e.key==='ArrowUp')focus(cur-1);
}});
document.getElementById('export').onclick=()=>{{
  const out={{sheet_id:{json.dumps(summary.get('sheet_id',''))},
    adjudicated_by:'human-review',verdicts:V}};
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{{type:'application/json'}}));
  a.download='verdicts_{html.escape(summary.get("sheet_id","sheet"))}.json'; a.click();
}};
</script>"""
    open(out_html, "w").write(doc)
    return len(items)

if __name__ == "__main__":
    n = build(sys.argv[1], sys.argv[2])
    print(f"wrote {sys.argv[2]} with {n} claims")
