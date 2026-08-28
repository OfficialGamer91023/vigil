"""The desk page, as one self-contained HTML string (§8).

**No build step, no CDN, no framework.** §11.1 cuts the Next.js dashboard by
default: the submission asks for a working URL, and on the day a page that cannot
fail beats a prettier one with a deploy pipeline. Everything here is inline, so
the page renders with no network beyond this service — a judge behind a corporate
proxy that blocks unpkg still sees the desk.

A Python string rather than a template file because there is exactly one page and
no template inheritance to gain; a Jinja dependency for this would be the kind of
thing §12 exists to refuse. The equity curve is hand-drawn SVG for the same
reason: a charting library is twenty lines of path arithmetic wearing 90KB.

**What the page is for.** Not a status board — an argument. The kernel evaluates
twelve gates and persists every verdict including passes (§3, §5.2), and the
whole point of that record is that "was this trade allowed, and by what?" is
answerable. So the decision feed is **expandable**: clicking a cycle fetches
`/api/cycles/{id}` and renders each proposal with all twelve verdicts and the
reason each rejection gave. Aggregate pass-rates alone cannot make that argument.
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
  .pill.warn{ color:var(--warn); border-color:var(--warn); }
  main { padding:20px; display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(320px,1fr)); }
  section { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px 16px; }
  .wide { grid-column:1/-1; }
  h2 { margin:0 0 10px; font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--dim); }
  h2 .note { float:right; text-transform:none; letter-spacing:0; font-size:11px; }
  .big { font-size:26px; }
  .row { display:flex; justify-content:space-between; gap:12px; padding:3px 0; }
  .row span:first-child { color:var(--dim); }
  table { width:100%; border-collapse:collapse; }
  th, td { text-align:left; padding:5px 8px 5px 0; border-bottom:1px solid var(--line); white-space:nowrap; }
  th { color:var(--dim); font-weight:400; font-size:11px; text-transform:uppercase; letter-spacing:.1em; }
  .scroll { overflow-x:auto; }
  .pos { color:var(--ok); } .neg { color:var(--bad); } .muted { color:var(--dim); }
  .warnc { color:var(--warn); }
  .bar { height:5px; background:var(--line); border-radius:3px; overflow:hidden; }
  .bar > i { display:block; height:100%; background:var(--accent); }
  #feed { max-height:420px; overflow-y:auto; }
  #orders { max-height:240px; overflow-y:auto; }
  footer { padding:0 20px 24px; color:var(--dim); font-size:12px; }
  code { color:var(--accent); }
  svg#curve { width:100%; height:88px; display:block; margin:6px 0 2px; }
  /* A row that opens a drawer has to look like one. */
  tr.clickable { cursor:pointer; }
  tr.clickable:hover td { background:#1a2130; }
  tr.open td { background:#1a2130; }
  .caret { color:var(--dim); display:inline-block; width:12px; }
  /* The drill-down drawer: a full-width cell inside the same table, so it
     scrolls with the feed instead of floating over it. */
  td.drawer { white-space:normal; padding:10px 12px 14px; background:#0f141d; }
  .prop { border:1px solid var(--line); border-radius:6px; padding:10px 12px; margin-bottom:10px; }
  .prop:last-child { margin-bottom:0; }
  .prop h3 { margin:0 0 6px; font-size:13px; font-weight:400; }
  .rationale { color:var(--dim); margin:0 0 8px; font-size:12px; }
  .verdicts { display:flex; flex-wrap:wrap; gap:6px; }
  .v {
    font-size:11px; padding:2px 7px; border-radius:4px;
    border:1px solid var(--line); color:var(--dim);
  }
  .v.pass { border-color:#245c4c; color:var(--ok); }
  .v.fail { border-color:#6b2f2f; color:var(--bad); }
  .v .why { color:var(--text); }
  .badge { font-size:11px; padding:1px 7px; border-radius:4px; border:1px solid var(--line); }
  .badge.ok  { color:var(--ok);  border-color:#245c4c; }
  .badge.no  { color:var(--bad); border-color:#6b2f2f; }
</style>
</head>
<body>
<header>
  <h1>Vigil</h1>
  <span class="sub" id="acct">connecting&hellip;</span>
  <span class="pill" id="stream">stream: offline</span>
  <span class="pill" id="age">last cycle: &mdash;</span>
  <span class="pill" id="halt">halt: &mdash;</span>
  <span class="pill" id="flat">flatten: &mdash;</span>
</header>

<main>
  <section>
    <h2>Equity</h2>
    <div class="big" id="equity">&mdash;</div>
    <svg id="curve" viewBox="0 0 600 88" preserveAspectRatio="none" aria-label="equity curve"></svg>
    <div class="row"><span>day P&amp;L</span><span id="pnl">&mdash;</span></div>
    <div class="row"><span>session change</span><span id="curvedelta">&mdash;</span></div>
    <div class="row"><span>as of</span><span id="asof" class="muted">&mdash;</span></div>
  </section>

  <section>
    <h2>Open risk</h2>
    <div class="big" id="risk">&mdash;</div>
    <div class="bar"><i id="riskbar" style="width:0%"></i></div>
    <div class="row"><span>of 12% ceiling</span><span id="riskpct" class="muted">&mdash;</span></div>
    <div class="row"><span>net dollar delta</span><span id="delta">&mdash;</span></div>
    <div class="row"><span>open structures</span><span id="count">&mdash;</span></div>
  </section>

  <section class="wide">
    <h2>Market &mdash; what the router last saw
      <span class="note muted">signals come from the underlying (&sect;1.2)</span></h2>
    <div class="scroll"><table id="market">
      <thead><tr><th>underlying</th><th>spot</th><th>trend</th><th>ATM IV</th>
        <th>realized vol</th><th>VRP %ile</th><th>IV %ile</th><th>cycle</th></tr></thead>
      <tbody><tr><td colspan="8" class="muted">no market read yet</td></tr></tbody>
    </table></div>
  </section>

  <section class="wide">
    <h2>Book</h2>
    <div class="scroll"><table id="book">
      <thead><tr><th>underlying</th><th>structure</th><th>expiry</th><th>qty</th>
        <th>credit</th><th>max loss</th><th>resting target</th></tr></thead>
      <tbody><tr><td colspan="7" class="muted">flat</td></tr></tbody>
    </table></div>
  </section>

  <section class="wide">
    <h2>Gate verdicts &mdash; passes and rejections
      <span class="note muted">every verdict persisted, passes included (&sect;5.2)</span></h2>
    <div class="scroll"><table id="gates">
      <thead><tr><th>#</th><th>gate</th><th>passed</th><th>failed</th><th>pass rate</th></tr></thead>
      <tbody><tr><td colspan="5" class="muted">no proposals evaluated yet</td></tr></tbody>
    </table></div>
  </section>

  <section class="wide">
    <h2>Decision feed <span class="note muted">click a cycle for its proposals and all twelve verdicts</span></h2>
    <div class="scroll" id="feed"><table>
      <thead><tr><th></th><th>cycle</th><th>kind</th><th>regime</th><th>started</th><th>notes</th></tr></thead>
      <tbody><tr><td colspan="6" class="muted">waiting&hellip;</td></tr></tbody>
    </table></div>
  </section>

  <section class="wide">
    <h2>Orders <span class="note muted">open &middot; target &middot; close &mdash; every ticket the router sent</span></h2>
    <div class="scroll" id="orders"><table>
      <thead><tr><th>submitted</th><th>intent</th><th>limit</th><th>qty</th>
        <th>rung</th><th>status</th><th>client_order_id</th></tr></thead>
      <tbody><tr><td colspan="7" class="muted">no orders yet</td></tr></tbody>
    </table></div>
  </section>
</main>

<footer>
  Read-only. This service never places an order &mdash; the worker does, and only after
  every gate in the risk kernel. Controls live at <code>POST /api/control/*</code>
  and require a bearer token.
</footer>

<script>
const money = v => v === null || v === undefined ? "—"
  : Number(v).toLocaleString("en-US", {style:"currency", currency:"USD", maximumFractionDigits:0});
const money2 = v => v === null || v === undefined ? "—"
  : Number(v).toLocaleString("en-US", {style:"currency", currency:"USD", minimumFractionDigits:2});
const signed = v => {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  return `<span class="${n > 0 ? "pos" : n < 0 ? "neg" : "muted"}">${n > 0 ? "+" : ""}${money(n)}</span>`;
};
// Percentages and greeks are NOT money, and rendering a null as 0 would turn an
// unmeasurable session into a confident reading — the exact inversion
// MarketSnapshot.rv_annual is typed `float | None` to prevent.
const pct = (v, digits = 1) => v === null || v === undefined
  ? '<span class="muted">—</span>' : (Number(v) * 100).toFixed(digits) + "%";
const signedPct = v => {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  const n = Number(v);
  return `<span class="${n > 0 ? "pos" : n < 0 ? "neg" : "muted"}">${n > 0 ? "+" : ""}${(n*100).toFixed(2)}%</span>`;
};
const num = (v, digits = 2) => v === null || v === undefined
  ? '<span class="muted">—</span>' : Number(v).toFixed(digits);
const clock = t => t ? new Date(t).toLocaleTimeString("en-US", {timeZone:"America/New_York", hour12:false}) : "—";
const cell = t => { const d = document.createElement("td"); d.textContent = t; return d; };
const htmlCell = h => { const d = document.createElement("td"); d.innerHTML = h; return d; };

function flag(el, name, on) {
  el.textContent = `${name}: ${on ? "ACTIVE" : "clear"}`;
  el.className = on ? "pill on" : "pill";
}

// --------------------------------------------------------------------------
// The equity curve. Hand-drawn rather than charted: §12 rejects a dependency
// that can be written in twenty lines, and a CDN would break the property this
// page is built on — that it renders with no network beyond this service.
//
// `preserveAspectRatio="none"` lets a fixed 600x88 viewBox stretch to whatever
// width the panel is, and `vector-effect="non-scaling-stroke"` is what stops
// that stretch from smearing the stroke into a wedge.
// --------------------------------------------------------------------------
function drawCurve(points) {
  const svg = document.getElementById("curve");
  const out = document.getElementById("curvedelta");
  svg.innerHTML = "";
  if (!points || points.length < 2) {
    out.innerHTML = '<span class="muted">—</span>';
    return;
  }
  const W = 600, H = 88, PAD = 6;
  const ys = points.map(p => Number(p.equity));
  const first = ys[0], last = ys[ys.length - 1];
  const lo = Math.min(...ys), hi = Math.max(...ys);
  // A flat series has zero span; dividing by it would put every point at NaN.
  const span = (hi - lo) || 1;
  const x = i => PAD + (i / (ys.length - 1)) * (W - 2 * PAD);
  const y = v => H - PAD - ((v - lo) / span) * (H - 2 * PAD);

  const line = ys.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const stroke = last > first ? "#4ec9a7" : last < first ? "#e56a6a" : "#7d879c";
  const base = y(first).toFixed(1);

  // Built as markup rather than with `createElementNS`. Assigning `innerHTML` on
  // an element that is already SVG parses its children in the SVG namespace, so
  // the namespace URI is not needed — which also keeps every absolute-URL scheme
  // out of the page, the property `test_the_desk_page_is_self_contained` guards.
  //
  // The dashed line is opening equity, so "up on the day" reads as geometry
  // rather than only as a number underneath. The fill is the same colour as the
  // stroke at low opacity: one decision, not two that can disagree.
  svg.innerHTML =
      `<line x1="${PAD}" x2="${W - PAD}" y1="${base}" y2="${base}"`
    + ` stroke="#232b3a" stroke-dasharray="3 4" vector-effect="non-scaling-stroke"/>`
    + `<path d="${line} L${x(ys.length-1).toFixed(1)},${H-PAD} L${PAD},${H-PAD} Z"`
    + ` fill="${stroke}" opacity="0.10"/>`
    + `<path d="${line}" fill="none" stroke="${stroke}" stroke-width="1.5"`
    + ` vector-effect="non-scaling-stroke"/>`;

  out.innerHTML = signed(last - first) + ` <span class="muted">over ${ys.length} points</span>`;
}

// --------------------------------------------------------------------------
// The drill-down. `openCycle` lives outside refresh() on purpose: an SSE tick
// re-renders the feed, and a drawer that collapsed on every tick would be
// unreadable during exactly the cycles worth reading.
// --------------------------------------------------------------------------
let openCycle = null;
let openDetail = null;

async function loadDetail(id) {
  try {
    openDetail = await fetch(`/api/cycles/${id}`).then(r => r.ok ? r.json() : null);
  } catch (e) { openDetail = null; }
}

function drawerHtml(detail) {
  if (!detail) return '<span class="muted">could not load this cycle</span>';
  if (!detail.proposals.length) {
    return '<span class="muted">no proposals were built this cycle — '
         + 'the router stood down, or no candidate could be constructed</span>';
  }
  return detail.proposals.map(p => {
    const verdicts = p.verdicts
      .slice()
      .sort((a, b) => a.gate_no - b.gate_no)
      .map(v => `<span class="v ${v.passed ? "pass" : "fail"}">${v.gate_no} ${v.name}`
              + (v.passed ? "" : ` <span class="why">— ${escapeHtml(v.reason ?? "")}</span>`)
              + `</span>`)
      .join("");
    const legs = p.legs
      .slice()
      .sort((a, b) => Number(a.strike) - Number(b.strike))
      .map(l => `${l.is_short ? "-" : "+"}${l.ratio_qty} ${Number(l.strike)}${l.is_put ? "P" : "C"}`
              + ` <span class="muted">Δ${Number(l.delta).toFixed(3)} oi ${l.open_interest}</span>`)
      .join(" &nbsp; ");
    return `<div class="prop">
      <h3>${escapeHtml(p.structure_type)} ${escapeHtml(p.underlying)} ${p.expiry} &times;${p.contracts}
        <span class="badge ${p.approved ? "ok" : "no"}">${p.approved ? "APPROVED" : "REJECTED"}</span>
      </h3>
      <p class="rationale">${escapeHtml(p.rationale ?? "")}</p>
      <div class="row"><span>credit / width / max loss</span>
        <span>${money2(p.net_credit)} / ${Number(p.width)} / ${money(p.max_loss)}</span></div>
      <div class="row"><span>max profit / dollar delta</span>
        <span>${p.max_profit === null ? '<span class="muted">unbounded</span>' : money(p.max_profit)}
              / ${signed(p.dollar_delta)}</span></div>
      <div class="row"><span>legs</span><span>${legs}</span></div>
      <div class="verdicts" style="margin-top:8px">${verdicts}</div>
    </div>`;
  }).join("");
}

// The verdict reasons and rationales are model- and config-derived strings that
// land in innerHTML, so they are escaped rather than trusted. Cheap, and the
// alternative is a page that breaks on a `<` in a rejection message.
function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

async function refresh() {
  try {
    const [s, g, c, h, m, o] = await Promise.all([
      fetch("/api/state").then(r => r.json()),
      fetch("/api/gates/stats").then(r => r.json()),
      fetch("/api/cycles?limit=40").then(r => r.json()),
      fetch("/health").then(r => r.json()),
      fetch("/api/market").then(r => r.json()),
      fetch("/api/orders?limit=40").then(r => r.json()),
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

    // /health reports the age of the last cycle rather than merely resolving —
    // past ~20 minutes during a session the loop is not turning, and that is the
    // one number that distinguishes "quiet market" from "dead agent".
    const ageEl = document.getElementById("age");
    if (h.last_cycle_age_seconds === null || h.last_cycle_age_seconds === undefined) {
      ageEl.textContent = "last cycle: never"; ageEl.className = "pill warn";
    } else {
      const mins = h.last_cycle_age_seconds / 60;
      ageEl.textContent = `last cycle: ${mins < 1 ? "just now" : mins.toFixed(0) + "m ago"}`
                        + (h.last_cycle_kind ? ` (${h.last_cycle_kind})` : "");
      ageEl.className = mins > 20 ? "pill on" : "pill live";
    }

    // Gate 2 caps a single trade at 2% of equity and Gate 5 allows six open
    // structures, so 12% of equity is the most the book can hold at once. The
    // meter is drawn against that ceiling rather than against equity, because
    // "6% of equity at risk" means half the budget, not a rounding error.
    if (s.equity) {
      const share = (Number(s.open_risk) / (Number(s.equity) * 0.12)) * 100;
      document.getElementById("riskbar").style.width = Math.min(100, share).toFixed(1) + "%";
      document.getElementById("riskpct").textContent = share.toFixed(0) + "%";
    }

    renderMarket(m);
    renderBook(s.open_structures);
    renderGates(g);
    renderCycles(c);
    renderOrders(o);

    if (s.account_id) {
      const eq = await fetch("/api/equity?limit=500").then(r => r.json());
      drawCurve(eq);
    }
  } catch (e) {
    document.getElementById("acct").textContent = "api unreachable";
  }
}

function renderMarket(rows) {
  const tb = document.querySelector("#market tbody");
  tb.innerHTML = "";
  if (!rows.length) {
    tb.innerHTML = '<tr><td colspan="8" class="muted">no market read yet</td></tr>';
    return;
  }
  for (const r of rows) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(r.underlying));
    tr.appendChild(cell(Number(r.spot).toFixed(2)));
    tr.appendChild(htmlCell(signedPct(r.trend)));
    tr.appendChild(htmlCell(pct(r.iv_atm, 1)));
    tr.appendChild(htmlCell(pct(r.rv_annual, 1)));
    // The VRP percentile is the sell decision: §4.3 stands down below the 40%
    // floor and calls STRESS in the bottom decile, so those thresholds are
    // coloured rather than left for the reader to remember.
    const v = r.vrp_pct === null || r.vrp_pct === undefined ? null : Number(r.vrp_pct);
    tr.appendChild(htmlCell(
      v === null ? '<span class="muted">—</span>'
        : `<span class="${v <= 0.10 ? "neg" : v < 0.40 ? "warnc" : "pos"}">${(v*100).toFixed(0)}%</span>`));
    tr.appendChild(htmlCell(pct(r.iv_pct, 0)));
    tr.appendChild(cell(r.cycle_id));
    tb.appendChild(tr);
  }
}

function renderBook(structures) {
  const book = document.querySelector("#book tbody");
  book.innerHTML = "";
  if (!structures.length) {
    book.innerHTML = '<tr><td colspan="7" class="muted">flat</td></tr>';
    return;
  }
  for (const st of structures) {
    const tr = document.createElement("tr");
    [st.underlying, st.structure_type ?? "—", st.expiry, st.contracts,
     money2(st.net_credit), money(st.max_loss)].forEach(v => tr.appendChild(cell(v)));
    // §2.6: an open structure with no resting GTC target is a reconciliation
    // defect, so it is called one here rather than shown as a blank cell.
    tr.appendChild(htmlCell(st.has_resting_target
      ? '<span class="pos">resting</span>'
      : '<span class="neg">MISSING — §2.6 defect</span>'));
    book.appendChild(tr);
  }
}

function renderGates(g) {
  const gt = document.querySelector("#gates tbody");
  gt.innerHTML = "";
  if (!g.length) {
    gt.innerHTML = '<tr><td colspan="5" class="muted">no proposals evaluated yet</td></tr>';
    return;
  }
  for (const row of g) {
    const total = row.passed + row.failed;
    const tr = document.createElement("tr");
    [row.gate_no, row.name, row.passed, row.failed].forEach(v => tr.appendChild(cell(v)));
    // A gate that never passes is as broken as one that never fires (§5.2),
    // so 0% is flagged in the same colour as a failure, not left to read as
    // diligence.
    const rate = total ? (row.passed / total) * 100 : 0;
    tr.appendChild(htmlCell(
      `<span class="${rate === 0 ? "neg" : rate === 100 ? "muted" : "pos"}">${rate.toFixed(0)}%</span>`));
    gt.appendChild(tr);
  }
}

function renderCycles(list) {
  const tb = document.querySelector("#feed tbody");
  tb.innerHTML = "";
  if (!list.length) { tb.innerHTML = '<tr><td colspan="6" class="muted">waiting…</td></tr>'; return; }
  for (const c of list) {
    const expanded = c.id === openCycle;
    const tr = document.createElement("tr");
    tr.className = expanded ? "clickable open" : "clickable";
    // The id rides on the row rather than in a closure so the delegated handler
    // below can read it off whatever row was actually clicked.
    tr.dataset.cycle = c.id;
    tr.appendChild(htmlCell(`<span class="caret">${expanded ? "▾" : "▸"}</span>`));
    [c.id, c.kind, c.regime ?? "—", clock(c.started_at)].forEach(v => tr.appendChild(cell(v)));
    // finished_at NULL is the crash signal the two-transaction cycle write
    // exists to preserve — surfaced, not hidden behind an empty notes column.
    tr.appendChild(htmlCell(c.finished_at === null
      ? '<span class="neg">did not finish</span>'
      : `${c.cold_start ? '<span class="muted">[cold start] </span>' : ""}${escapeHtml(c.notes ?? "")}`));
    tb.appendChild(tr);

    if (expanded) {
      const drawer = document.createElement("tr");
      const td = document.createElement("td");
      td.className = "drawer";
      td.colSpan = 6;
      td.innerHTML = drawerHtml(openDetail);
      drawer.appendChild(td);
      tb.appendChild(drawer);
    }
  }
}

function renderOrders(list) {
  const tb = document.querySelector("#orders tbody");
  tb.innerHTML = "";
  if (!list.length) { tb.innerHTML = '<tr><td colspan="7" class="muted">no orders yet</td></tr>'; return; }
  for (const o of list) {
    const tr = document.createElement("tr");
    tr.appendChild(cell(clock(o.submitted_at)));
    // `target` is the §2.6 resting exit — the one order whose presence is a
    // safety property, so it is coloured rather than listed as another word.
    tr.appendChild(htmlCell(
      `<span class="${o.intent === "target" ? "pos" : o.intent === "close" ? "warnc" : ""}">${o.intent}</span>`));
    tr.appendChild(cell(money2(o.limit_price)));
    tr.appendChild(cell(o.qty));
    tr.appendChild(cell(o.rung ?? "—"));
    tr.appendChild(cell(o.status));
    tr.appendChild(cell(o.client_order_id));
    tb.appendChild(tr);
  }
}

// **Event delegation, one listener for the whole feed.** Rows are rebuilt on
// every SSE tick, so a listener attached per row would be re-created (and its
// predecessors orphaned) several times a minute. The tbody itself survives
// `innerHTML = ""`, so one handler on it outlives every re-render.
document.querySelector("#feed tbody").addEventListener("click", async (ev) => {
  const tr = ev.target.closest("tr[data-cycle]");
  if (!tr) return;
  const id = Number(tr.dataset.cycle);
  if (openCycle === id) {
    openCycle = null; openDetail = null;
  } else {
    openCycle = id;
    await loadDetail(id);
  }
  const list = await fetch("/api/cycles?limit=40").then(r => r.json());
  renderCycles(list);
});

const es = new EventSource("/api/stream");
es.onopen = () => { const p = document.getElementById("stream");
  p.textContent = "stream: live"; p.className = "pill live"; };
es.onerror = () => { const p = document.getElementById("stream");
  p.textContent = "stream: reconnecting"; p.className = "pill"; };
// A new cycle landed, so every panel is stale, not just the feed — refetch
// rather than splice the one row in. An open drawer is reloaded too, so a cycle
// being read while it is still running stays current.
es.addEventListener("cycle", async () => {
  if (openCycle !== null) await loadDetail(openCycle);
  refresh();
});

refresh();
// A slow backstop under the SSE push: if the stream is blocked by a proxy the
// page still updates, just less promptly.
setInterval(refresh, 30000);
</script>
</body>
</html>
"""
