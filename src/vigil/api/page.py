"""The desk page, as one self-contained HTML string (§8).

**No build step, no CDN, no framework.** §11.1 cuts the Next.js dashboard by
default: the submission asks for a working URL, and on the day a page that cannot
fail beats a prettier one with a deploy pipeline. Everything here is inline, so
the page renders with no network beyond this service — a judge behind a corporate
proxy that blocks unpkg still sees the desk.

A Python string rather than a template file because there is exactly one page and
no template inheritance to gain; a Jinja dependency for this would be the kind of
thing §12 exists to refuse.
"""

from __future__ import annotations

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Vigil</title>
<style>
  :root {
    --bg:#0b0e14; --panel:#131822; --line:#232b3a; --text:#d7dce5;
    --dim:#7d879c; --ok:#4ec9a7; --bad:#e56a6a; --warn:#d7a55b; --accent:#5b9dd9;
  }
  * { box-sizing:border-box; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font:14px/1.5 ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }
  header {
    display:flex; align-items:baseline; gap:16px; flex-wrap:wrap;
    padding:16px 20px; border-bottom:1px solid var(--line);
  }
  h1 { margin:0; font-size:18px; letter-spacing:.14em; text-transform:uppercase; }
  .sub { color:var(--dim); }
  .pill {
    padding:2px 8px; border-radius:10px; border:1px solid var(--line);
    font-size:12px; color:var(--dim);
  }
  .pill.on  { color:var(--bad);  border-color:var(--bad); }
  .pill.live{ color:var(--ok);   border-color:var(--ok); }
  main { padding:20px; display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
  h2 { margin:0 0 10px; font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--dim); }
  .big { font-size:26px; }
  .row { display:flex; justify-content:space-between; gap:12px; padding:3px 0; }
  .row span:first-child { color:var(--dim); }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:5px 8px 5px 0; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--dim); font-weight:400; font-size:11px; text-transform:uppercase; letter-spacing:.1em; }
  .scroll { overflow-x:auto; }
  .pos { color:var(--ok); } .neg { color:var(--bad); } .muted { color:var(--dim); }
  .bar { height:5px; background:var(--line); border-radius:3px; overflow:hidden; }
  .bar > i { display:block; height:100%; background:var(--accent); }
  #feed { max-height:280px; overflow-y:auto; }
  footer { padding:0 20px 24px; color:var(--dim); font-size:12px; }
  code { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>Vigil</h1>
  <span class="sub" id="acct">connecting…</span>
  <span class="pill" id="stream">stream: offline</span>
  <span class="pill" id="halt">halt: —</span>
  <span class="pill" id="flat">flatten: —</span>
</header>

<main>
  <section>
    <h2>Equity</h2>
    <div class="big" id="equity">—</div>
    <div class="row"><span>day P&amp;L</span><span id="pnl">—</span></div>
    <div class="row"><span>as of</span><span id="asof" class="muted">—</span></div>
  </section>

  <section>
    <h2>Open risk</h2>
    <div class="big" id="risk">—</div>
    <div class="bar"><i id="riskbar" style="width:0%"></i></div>
    <div class="row"><span>net dollar delta</span><span id="delta">—</span></div>
    <div class="row"><span>open structures</span><span id="count">—</span></div>
  </section>

  <section style="grid-column:1/-1">
    <h2>Book</h2>
    <div class="scroll"><table id="book">
      <thead><tr><th>underlying</th><th>structure</th><th>expiry</th><th>qty</th>
        <th>credit</th><th>max loss</th><th>resting target</th></tr></thead>
      <tbody><tr><td colspan="7" class="muted">flat</td></tr></tbody>
    </table></div>
  </section>

  <section style="grid-column:1/-1">
    <h2>Gate verdicts — passes and rejections</h2>
    <div class="scroll"><table id="gates">
      <thead><tr><th>#</th><th>gate</th><th>passed</th><th>failed</th><th>pass rate</th></tr></thead>
      <tbody><tr><td colspan="5" class="muted">no proposals evaluated yet</td></tr></tbody>
    </table></div>
  </section>

  <section style="grid-column:1/-1">
    <h2>Decision feed</h2>
    <div class="scroll" id="feed"><table>
      <thead><tr><th>cycle</th><th>kind</th><th>regime</th><th>started</th><th>notes</th></tr></thead>
      <tbody><tr><td colspan="5" class="muted">waiting…</td></tr></tbody>
    </table></div>
  </section>
</main>

<footer>
  Read-only. This service never places an order — the worker does, and only after
  every gate in the risk kernel. Controls live at <code>POST /api/control/*</code>
  and require a bearer token.
</footer>

<script>
const money = v => v === null || v === undefined ? "—"
  : Number(v).toLocaleString("en-US", {style:"currency", currency:"USD", maximumFractionDigits:0});
const signed = v => {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  return `<span class="${n > 0 ? "pos" : n < 0 ? "neg" : "muted"}">${n > 0 ? "+" : ""}${money(n)}</span>`;
};
const clock = t => t ? new Date(t).toLocaleTimeString("en-US", {timeZone:"America/New_York", hour12:false}) : "—";
const cell = t => { const d = document.createElement("td"); d.textContent = t; return d; };

function flag(el, name, on) {
  el.textContent = `${name}: ${on ? "ACTIVE" : "clear"}`;
  el.className = on ? "pill on" : "pill";
}

async function refresh() {
  try {
    const [s, g, c] = await Promise.all([
      fetch("/api/state").then(r => r.json()),
      fetch("/api/gates/stats").then(r => r.json()),
      fetch("/api/cycles?limit=25").then(r => r.json()),
    ]);
    document.getElementById("acct").textContent =
      s.account_id ? `${s.account_id.slice(0,8)}… · ${s.trading_date ?? "no session"}` : "no account yet";
    document.getElementById("equity").textContent = money(s.equity);
    document.getElementById("pnl").innerHTML = signed(s.day_pnl);
    document.getElementById("asof").textContent = s.as_of ? clock(s.as_of) + " ET" : "—";
    document.getElementById("risk").textContent = money(s.open_risk);
    document.getElementById("delta").innerHTML = signed(s.net_dollar_delta);
    document.getElementById("count").textContent = s.open_structures.length;
    flag(document.getElementById("halt"), "halt", s.halted);
    flag(document.getElementById("flat"), "flatten", s.flatten_requested);

    // Gate 2 caps a single trade at 2% of equity and Gate 5 allows six open
    // structures, so 12% of equity is the most the book can hold at once. The
    // meter is drawn against that ceiling rather than against equity, because
    // "6% of equity at risk" means half the budget, not a rounding error.
    if (s.equity) {
      const pct = Math.min(100, (Number(s.open_risk) / (Number(s.equity) * 0.12)) * 100);
      document.getElementById("riskbar").style.width = pct.toFixed(1) + "%";
    }

    const book = document.querySelector("#book tbody");
    book.innerHTML = "";
    if (!s.open_structures.length) {
      book.innerHTML = '<tr><td colspan="7" class="muted">flat</td></tr>';
    } else for (const st of s.open_structures) {
      const tr = document.createElement("tr");
      [st.underlying, st.structure_type ?? "—", st.expiry, st.contracts,
       money(st.net_credit), money(st.max_loss)].forEach(v => tr.appendChild(cell(v)));
      const t = document.createElement("td");
      // §2.6: an open structure with no resting GTC target is a reconciliation
      // defect, so it is called one here rather than shown as a blank cell.
      t.innerHTML = st.has_resting_target
        ? '<span class="pos">resting</span>'
        : '<span class="neg">MISSING — §2.6 defect</span>';
      tr.appendChild(t);
      book.appendChild(tr);
    }

    const gt = document.querySelector("#gates tbody");
    gt.innerHTML = "";
    if (!g.length) {
      gt.innerHTML = '<tr><td colspan="5" class="muted">no proposals evaluated yet</td></tr>';
    } else for (const row of g) {
      const total = row.passed + row.failed;
      const tr = document.createElement("tr");
      [row.gate_no, row.name, row.passed, row.failed].forEach(v => tr.appendChild(cell(v)));
      const r = document.createElement("td");
      // A gate that never passes is as broken as one that never fires (§5.2),
      // so 0% is flagged in the same colour as a failure, not left to read as
      // diligence.
      const rate = total ? (row.passed / total) * 100 : 0;
      r.innerHTML = `<span class="${rate === 0 ? "neg" : rate === 100 ? "muted" : "pos"}">${rate.toFixed(0)}%</span>`;
      tr.appendChild(r);
      gt.appendChild(tr);
    }

    renderCycles(c);
  } catch (e) {
    document.getElementById("acct").textContent = "api unreachable";
  }
}

function renderCycles(list) {
  const tb = document.querySelector("#feed tbody");
  tb.innerHTML = "";
  if (!list.length) { tb.innerHTML = '<tr><td colspan="5" class="muted">waiting…</td></tr>'; return; }
  for (const c of list) {
    const tr = document.createElement("tr");
    [c.id, c.kind, c.regime ?? "—", clock(c.started_at)].forEach(v => tr.appendChild(cell(v)));
    const n = document.createElement("td");
    // finished_at NULL is the crash signal the two-transaction cycle write
    // exists to preserve — surfaced, not hidden behind an empty notes column.
    n.innerHTML = c.finished_at === null
      ? '<span class="neg">did not finish</span>'
      : `${c.cold_start ? '<span class="muted">[cold start] </span>' : ""}${c.notes ?? ""}`;
    tr.appendChild(n);
    tb.appendChild(tr);
  }
}

const es = new EventSource("/api/stream");
es.onopen = () => { const p = document.getElementById("stream");
  p.textContent = "stream: live"; p.className = "pill live"; };
es.onerror = () => { const p = document.getElementById("stream");
  p.textContent = "stream: reconnecting"; p.className = "pill"; };
// A new cycle landed, so every panel is stale, not just the feed — refetch
// rather than splice the one row in.
es.addEventListener("cycle", refresh);

refresh();
// A slow backstop under the SSE push: if the stream is blocked by a proxy the
// page still updates, just less promptly.
setInterval(refresh, 30000);
</script>
</body>
</html>
"""
