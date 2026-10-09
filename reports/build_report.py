"""Build reports/boost_report.html: can we subtract paid to get organic, and which posts are paid.

Charts first, one-line findings with hover detail, and an FAQ for the long answers.
Reads aggregate result files only (no row-level data, no client or person names):
  results/reconciliation.json, results/reconciliation_tiktok.json, results/classification_metrics.json,
  results/organic_coverage.json, results/weighted_recall.json, results/llm_vs_ml*.csv, reconcile/organic_curve_*.csv
Run from repo root (after reconcile.analyze, reconcile.analyze_tiktok, detector.evaluate):
  python3 -m reports.build_report
"""
import csv
import datetime
import html
import json
import math
import os
import re

OUT = "reports/boost_report.html"
REPO = "https://github.com/jcuthbertson-hue/boosted_or_organic_detector/tree/claude/funny-franklin-5w06tt"
PLATFORMS = [("Instagram", "Instagram"), ("Tiktok", "TikTok")]
DEFAULT_PF = "Instagram"
HAND_RULE = "Hand rule: views > followers AND engagement < 1%"
GLOSSARY = [
    ("Public views (Nimble)", "The count anyone can see on the post, read by our scraper (Nimble). Organic plus paid. On Instagram it is the Instagram part only, "
                              "not the views of the same post on Facebook."),
    ("Total seen", "All views of the post on the platform: Instagram plus Facebook (SocAPI total plays)."),
    ("Boosted", "An ad record (ad link or post-ID tag) or a model flag says the post had paid views."),
    ("Pre-boost read", "The last public view count before the boost started. It holds organic views only."),
    ("Organic curve", "How unboosted posts normally grow with age (share of their day-120 views on each day). We grow the pre-boost read along it."),
    ("View jump", "The first day the daily views grow far faster than the organic curve. With no ad date, we use it as the boost start."),
    ("Dark post", "An ad that runs as its own post, not as a boost of the creator's post. The creator's post then keeps organic views only."),
    ("Opt-in views", "The creator's own account numbers, shared with consent. On Instagram they count organic views only, so they are the "
                     "organic truth. On TikTok they include Spark Ad views."),
    ("SocAPI", "A separate feed of play counts per post (total, Instagram, Facebook). Used only as an independent check, never as a model input."),
    ("Large boost", "Opt-in shows 40% or more of public views are paid, or an ad link or post-ID tag with the boost inside the model window. "
                    "Every TikTok boost counts as large."),
    ("Flags really paid (precision)", "Of the posts the model flags as paid, the share that are paid."),
    ("Paid caught (recall)", "Of the posts that are paid, the share the model flags."),
    ("Overall score (F1)", "One number that balances the two above. 1.000 is perfect."),
    ("Scores match reality (calibration)", "A score of 80% should mean about 80% of such posts are paid. The gap is shown in percentage points (pts)."),
    ("Locked test", "Posts published 2026-07-01 to 2026-09-09, kept out of training and scored once per model version."),
    ("Day-14, day-30, day-60 model", "The same task, scored with the public reads a post has by that age."),
    ("Typical", "The median: half the posts are above, half below."),
    ("95% range", "The range the true value very likely sits in, given the number of posts."),
]


# ---------------------------------------------------------------- formatting
def esc(s):
    return html.escape(str(s), quote=True)


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def pct(x, d=0):
    return "n/a" if isnan(x) else f"{100 * x:.{d}f}%"


def f2(x):
    return "n/a" if isnan(x) else f"{x:.2f}"


def f3(x):
    return "n/a" if isnan(x) else f"{x:.3f}"


def times(x):
    return f"{x:.2f}×" if x < 10 else f"{x:.0f}×"


def load():
    j = lambda p: json.load(open(p))
    curves = {}
    for p, f in [("Instagram", "reconcile/organic_curve_ig.csv"), ("Tiktok", "reconcile/organic_curve_tiktok.csv")]:
        curves[p] = [(int(r["age_days"]), float(r["median_share_of_day120_views"]), int(r["posts"])) for r in csv.DictReader(open(f))]
    return dict(R=j("results/reconciliation.json"), T=j("results/reconciliation_tiktok.json"),
                C=j("results/classification_metrics.json"), V=j("results/organic_coverage.json"),
                W=j("results/weighted_recall.json"), curves=curves,
                llm=list(csv.DictReader(open("results/llm_vs_ml.csv"))),
                paired=list(csv.DictReader(open("results/llm_vs_ml_paired_auc.csv"))),
                C2=(j("results/classification_metrics_v2_2.json") if os.path.exists("results/classification_metrics_v2_2.json") else
                    j("results/classification_metrics_v2_1.json") if os.path.exists("results/classification_metrics_v2_1.json") else
                    j("results/classification_metrics_v2.json") if os.path.exists("results/classification_metrics_v2.json") else None),
                TG=(j("results/model_v2_2_targets.json") if os.path.exists("results/model_v2_2_targets.json") else
                    j("results/model_v2_1_targets.json") if os.path.exists("results/model_v2_1_targets.json") else
                    j("results/model_v2_targets.json") if os.path.exists("results/model_v2_targets.json") else None),
                TG0=j("results/model_v2_targets.json") if os.path.exists("results/model_v2_targets.json") else None,
                TG1=j("results/model_v2_1_targets.json") if os.path.exists("results/model_v2_1_targets.json") else None,
                RL=j("results/model_v2_relabel.json") if os.path.exists("results/model_v2_relabel.json") else None,
                ST=j("results/optin_staleness.json") if os.path.exists("results/optin_staleness.json") else None,
                MF=j("results/month_folds.json") if os.path.exists("results/month_folds.json") else None,
                AU=j("results/ig_miss_audit.json") if os.path.exists("results/ig_miss_audit.json") else None,
                EQ=j("results/vn_boost_equation.json") if os.path.exists("results/vn_boost_equation.json") else None,
                PM=j("results/predict_missing.json") if os.path.exists("results/predict_missing.json") else None,
                BS=j("results/boost_start_eval.json") if os.path.exists("results/boost_start_eval.json") else None,
                MR=j("results/missing_research.json") if os.path.exists("results/missing_research.json") else None)


# ---------------------------------------------------------------- small components
def info(tip):
    return f'<button class="info" type="button" data-tip="{esc(tip)}" aria-label="{esc(tip)}">i</button>'


def tipped(text, tip):
    """text with an info button that never wraps onto a line of its own"""
    head, _, last = str(text).rpartition(" ")
    return (esc(head) + " " if head else "") + f'<span class="nw">{esc(last)}{info(tip)}</span>' if tip else esc(text)


def num(v, tip):
    """a number that explains itself on hover, tap or keyboard focus"""
    return f'<b class="num" tabindex="0" data-tip="{esc(tip)}">{v}</b>'


def status(good):
    return f'<span class="st {"ok" if good else "no"}">{"Met" if good else "Missed"}</span>'


def day(s, year=False):
    d = datetime.date.fromisoformat(s)
    return f"{d:%b} {d.day}" + (f", {d.year}" if year else "")


def table(head, rows, hl=None, tips=None):
    th = "".join(f'<th class="{"r" if i else ""}">{esc(h)}{info(tips[i]) if tips and tips[i] else ""}</th>' for i, h in enumerate(head))
    tr = "".join(f'<tr class="{"hl" if hl and hl(r) else ""}">' + "".join(f'<td class="{"r" if i else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def card(title, body, *, cls="", sub="", tip="", data=None, attrs=""):
    head = (f'<div class="card-head"><h3>{tipped(title, tip)}</h3>'
            + (f'<p class="card-sub">{sub}</p>' if sub else "") + "</div>")
    more = (f'<details class="more"><summary aria-label="Show the data as a table"><span aria-hidden="true">···</span></summary>'
            f'<div class="more-body">{data}</div></details>') if data else ""
    return f'<article class="card {cls}" {attrs}>{head}{body}{more}</article>'


def legend(items):
    return '<div class="legend">' + "".join(f'<span class="key"><i class="sw {c}"></i>{esc(n)}</span>' for c, n in items) + "</div>"


def pf_panels(fn):
    """one panel per platform; only the selected platform is visible"""
    return "".join(f'<div data-pf="{p}"{"" if p == DEFAULT_PF else " hidden"}>{fn(p, lab)}</div>' for p, lab in PLATFORMS)


# ---------------------------------------------------------------- SVG charts
def lin(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) * (r1 - r0) / (d1 - d0)


DEFS = """<svg class="defs" width="0" height="0" aria-hidden="true" focusable="false"><defs>
<linearGradient id="wash-a" x1="0" y1="0" x2="0" y2="1"><stop offset="0" class="st-a" stop-opacity=".2"/><stop offset="1" class="st-a" stop-opacity="0"/></linearGradient>
<pattern id="hatch-a" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="hb-bg-a"/><rect width="2" height="6" class="hb-a"/></pattern>
<pattern id="hatch-n" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="hb-bg-n"/><rect width="1.6" height="6" class="hb-n"/></pattern>
<pattern id="vhatch-n" width="5" height="5" patternUnits="userSpaceOnUse"><rect width="1" height="5" class="hb-vn"/></pattern>
</defs></svg>"""


def xy_chart(series, *, label, xdom=(0, 1), ydom=(0, 1), ticks=(0, .5, 1), xlabel="", ylabel="", diag=False, vline=None,
             w=360, h=250, fmt=lambda v: f"{v:g}"):
    m = dict(t=12, r=14, b=40, l=40)
    X = lin(xdom[0], xdom[1], m["l"], w - m["r"])
    Y = lin(ydom[0], ydom[1], h - m["b"], m["t"])
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(label)}">']
    for t in ticks:
        s.append(f'<line class="grid" x1="{m["l"]}" x2="{w - m["r"]}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{m["l"] - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{esc(fmt(t))}</text>'
                 f'<text class="tick" x="{X(t):.1f}" y="{h - m["b"] + 17}" text-anchor="middle">{esc(fmt(t))}</text>')
    s.append(f'<text class="axlab" x="{(m["l"] + w - m["r"]) / 2:.1f}" y="{h - 6}" text-anchor="middle">{esc(xlabel)}</text>'
             f'<text class="axlab" transform="translate(11 {(m["t"] + h - m["b"]) / 2:.1f}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    if diag:
        s.append(f'<line class="ref" x1="{X(xdom[0]):.1f}" y1="{Y(ydom[0]):.1f}" x2="{X(xdom[1]):.1f}" y2="{Y(ydom[1]):.1f}"/>')
    if vline is not None:
        v, txt = vline
        s.append(f'<line class="ref" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{m["t"]}" y2="{h - m["b"]}"/>'
                 f'<text class="tick" x="{X(v) + 5:.1f}" y="{h - m["b"] - 6}">{esc(txt)}</text>')
    for se in series:
        pts = se["pts"]
        d = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        if se.get("area"):
            s.append(f'<path class="area" d="{d} L{X(pts[-1][0]):.1f},{Y(ydom[0]):.1f} L{X(pts[0][0]):.1f},{Y(ydom[0]):.1f} Z"/>')
        s.append(f'<path class="ln {se["cls"]}" d="{d}"/>')
        if se.get("dots"):
            s.extend(f'<circle class="dot {se["cls"]}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5"/>' for x, y in pts)
    for se in series:
        if not se.get("tip"):
            continue
        pts = se["pts"]
        step = max(1, len(pts) // 40)
        for i, (x, y) in enumerate(pts):
            if i % step == 0 or i == len(pts) - 1:
                s.append(f'<circle class="hit" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="9" data-tip="{esc(se["tip"](x, y, i))}"/>')
    s.append("</svg>")
    return "".join(s)


def round_bar(x, y_base, y_end, w, r=4):
    up, hgt = y_end < y_base, abs(y_base - y_end)
    if hgt < .5:
        return ""
    r = min(r, hgt, w / 2)
    e = y_end + r if up else y_end - r
    return (f"M{x:.1f},{y_base:.1f} V{e:.1f} Q{x:.1f},{y_end:.1f} {x + r:.1f},{y_end:.1f} H{x + w - r:.1f} "
            f"Q{x + w:.1f},{y_end:.1f} {x + w:.1f},{e:.1f} V{y_base:.1f} Z")


def mirror_hist(hist, thr, lab, w=360, h=250):
    e, pos, neg = hist["edges"], hist["boosted"], hist["organic"]
    fp = [v / sum(pos) for v in pos]
    fn = [v / sum(neg) for v in neg]
    top = math.ceil(max(fp + fn) * 10) / 10
    m = dict(t=12, r=14, b=40, l=40)
    X = lin(0, 1, m["l"], w - m["r"])
    mid = (m["t"] + h - m["b"]) / 2
    Yu, Yd = lin(0, top, mid, m["t"]), lin(0, top, mid, h - m["b"])
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(lab)} score distribution">']
    for t in (top, top / 2):
        for yy in (Yu(t), Yd(t)):
            s.append(f'<line class="grid" x1="{m["l"]}" x2="{w - m["r"]}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                     f'<text class="tick" x="{m["l"] - 8}" y="{yy + 4:.1f}" text-anchor="end">{pct(t)}</text>')
    bw = X(e[1]) - X(e[0]) - 2
    for i in range(len(pos)):
        rng = f"score {e[i]:.2f}–{e[i + 1]:.2f}"
        for d, cls, txt in [(round_bar(X(e[i]) + 1, mid, Yu(fp[i]), bw), "bar-a", f"Paid posts, {rng}: {pos[i]} ({pct(fp[i])} of paid)"),
                            (round_bar(X(e[i]) + 1, mid, Yd(fn[i]), bw), "bar-n", f"Organic posts, {rng}: {neg[i]} ({pct(fn[i])} of organic)")]:
            if d:
                s.append(f'<path class="{cls}" d="{d}" data-tip="{esc(txt)}"/>')
    s.append(f'<line class="axis" x1="{m["l"]}" x2="{w - m["r"]}" y1="{mid}" y2="{mid}"/>'
             f'<line class="ref" x1="{X(thr):.1f}" x2="{X(thr):.1f}" y1="{m["t"]}" y2="{h - m["b"]}"/>'
             f'<text class="tick" x="{X(thr) + 5:.1f}" y="{m["t"] + 10}">threshold {thr:.2f}</text>')
    for t in (0, .5, 1):
        s.append(f'<text class="tick" x="{X(t):.1f}" y="{h - m["b"] + 17}" text-anchor="middle">{t:g}</text>')
    s.append(f'<text class="axlab" x="{(m["l"] + w - m["r"]) / 2:.1f}" y="{h - 6}" text-anchor="middle">Model score</text>'
             f'<text class="tick" x="{w - m["r"]}" y="{m["t"] + 10}" text-anchor="end">paid ↑</text>'
             f'<text class="tick" x="{w - m["r"]}" y="{h - m["b"] - 5}" text-anchor="end">organic ↓</text></svg>')
    return "".join(s)


def curve_chart(curves, w=360, h=230):
    m = dict(t=14, r=14, b=36, l=40)
    X = lin(0, 120, m["l"], w - m["r"])
    Y = lin(0, 1, h - m["b"], m["t"])
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Organic growth curve: share of day-120 views by post age">']
    for t in (0, .5, 1):
        s.append(f'<line class="grid" x1="{m["l"]}" x2="{w - m["r"]}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{m["l"] - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{pct(t)}</text>')
    for t in (0, 30, 60, 90, 120):
        s.append(f'<text class="tick" x="{X(t):.1f}" y="{h - m["b"] + 16}" text-anchor="middle">day {t}</text>')

    def step(pts):
        d = f"M{X(pts[0][0]):.1f},{Y(pts[0][1]):.1f}"
        for a, b in zip(pts, pts[1:]):
            d += f" H{X(b[0]):.1f} V{Y(b[1]):.1f}"
        return d
    ig = [(a, v) for a, v, _ in curves["Instagram"] if a <= 120]
    tt = [(a, v) for a, v, _ in curves["Tiktok"] if a <= 120]
    s.append(f'<path class="vfill" d="{step(ig)} H{X(120):.1f} V{Y(0):.1f} H{X(0):.1f} Z"/>')
    s.append(f'<path class="ln ln-n4" d="{step(tt)}"/><path class="ln ln-acc" d="{step(ig)}"/>')
    for lab, pts, cls in [("Instagram", ig, "ln-acc"), ("TikTok", tt, "ln-n4")]:
        for a, v in pts[::3]:
            s.append(f'<circle class="hit" cx="{X(a):.1f}" cy="{Y(v):.1f}" r="8" data-tip="{esc(f"{lab}, day {a}: {pct(v)} of day-120 views")}"/>')
    s.append("</svg>")
    pills = ""
    for lab, pts, dy in [("Instagram", ig, -34), ("TikTok", tt, 10)]:
        a, v = next((a, v) for a, v in pts if a == 7)
        pills += (f'<span class="pill pin" style="left:{100 * X(a) / w:.1f}%;top:calc({100 * Y(v) / h:.1f}% + {dy}px)">'
                  f'{lab} day 7: <b>{pct(v)}</b></span>')
    return f'<div class="plot-wrap">{"".join(s)}{pills}</div>'


# ---------------------------------------------------------------- hero cards
def hero_columns(R):
    eb = R["error_by_paid_share"]
    vmax = math.ceil(max(r["median_abs_error"] for r in eb))
    hl = max(range(len(eb)), key=lambda i: eb[i]["posts"])
    heads, cells = ['<div class="cy"></div>'], ['<div class="cy">' + "".join(
        f'<span class="cy-t" style="bottom:{100 * t / vmax:.1f}%">{t * 100:.0f}%</span>' for t in range(0, vmax + 1)) + "</div>"]
    for i, r in enumerate(eb):
        on = " on" if i == hl else ""
        heads.append(f'<div class="ch{on}"><span class="cl">Paid {esc(r["paid_share"])}</span><span class="cv">{pct(r["median_abs_error"])}</span>'
                     f'<span class="cn">{r["posts"]} posts</span></div>')
        tip = f'Paid {r["paid_share"]} of public views: typical organic error {pct(r["median_abs_error"])}, {r["posts"]} posts'
        cells.append(f'<div class="cc{on}"><div class="cbar {"solid" if on else "hatch"}" style="height:{max(1.2, 100 * r["median_abs_error"] / vmax):.1f}%" '
                     f'tabindex="0" data-tip="{esc(tip)}"></div></div>')
    grid = "".join(f'<span class="gl" style="bottom:{100 * t / vmax:.1f}%"></span>' for t in range(0, vmax + 1))
    r = eb[hl]
    pill = (f'<span class="pill float"><b>{r["posts"]}</b> of {R["postflight_posts"]} posts <i></i> paid <b>{esc(r["paid_share"])}</b> of public '
            f'<i></i> organic error <b>{pct(r["median_abs_error"])}</b></span>')
    return (f'<div class="cols"><div class="cols-row heads">{"".join(heads)}</div>'
            f'<div class="cols-row plot">{grid}{"".join(cells)}{pill}</div>{pill.replace("pill float", "pill float mobile")}</div>')


def progress(rows, fmt=pct, goal=None):
    mark = f'<span class="pb-goal" style="left:{100 * goal:.1f}%"></span>' if goal is not None else ""
    return '<div class="pbars">' + "".join(
        f'<div class="pb"><div class="pb-top"><span>{esc(lab)}</span><b>{fmt(v)}</b></div>'
        f'<div class="pb-track"><div class="pb-fill {cls}" style="width:{100 * v:.1f}%" tabindex="0" data-tip="{esc(tip)}"></div>{mark}</div></div>'
        for lab, v, cls, tip in rows) + "</div>"


def dots(V):
    seg = [("measured_optin", "Measured (opt-in)", "m"), ("optin_frozen", "Opt-in until it froze, then estimate", "mf"),
           ("est_high", "Estimate, high", "h"), ("est_medium", "Estimate, medium", "md"),
           ("est_low", "Estimate, low", "l"), ("not_separable", "Not separable", "x")]
    out = []
    for p, lab in PLATFORMS:
        t = V["totals"][p]
        counts = [round(100 * t.get(k, 0) / t["posts"]) for k, _, _ in seg]
        counts[0] += 100 - sum(counts)
        cells = []
        for (k, name, cls), n in zip(seg, counts):
            tip = f"{lab}: {name}: {t.get(k, 0):,} of {t['posts']:,} paid posts ({pct(t.get(k, 0) / t['posts'])})"
            cells += [f'<i class="dt {cls}" data-tip="{esc(tip)}"></i>'] * n
        out.append(f'<div class="dm"><span class="dm-lab">{lab}<small>{t["posts"]:,} with an ad record</small></span><div class="dm-grid">{"".join(cells)}</div></div>')
    shown = [s for s in seg if any(V["totals"][p].get(s[0], 0) for p, _ in PLATFORMS)]
    return "".join(out) + legend([(c, n) for _, n, c in shown])


def dot_range(R):
    ig, fb = R["metric_match"]["instagram_side"], R["metric_match"]["facebook_side"]
    lo, hi = .5, 1000
    pos = lambda v: 100 * (math.log10(min(max(v, lo), hi)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
    ticks = [.5, 1, 2, 5, 10, 100, 1000]
    closest = lambda side: min((k for k, r in side.items() if r.get("n")), key=lambda k: abs(math.log10(side[k]["median"])))
    best = {"t": closest(ig), "b": closest(fb)}
    rows = []
    for name in ig:
        marks = ""
        for r, cls, side, where in [(ig[name], "k-ig", "Instagram placement", "t"), (fb.get(name, {}), "k-fb", "Facebook placement", "b")]:
            if not r.get("n"):
                continue
            cls += " best" if best[where] == name else ""
            tip = f"{name}, {side}: {times(r['median'])} (middle half {times(r['q25'])}–{times(r['q75'])}), {r['n']} reads"
            marks += (f'<span class="dr-w {cls} {where}" style="left:{pos(r["q25"]):.2f}%;width:{max(pos(r["q75"]) - pos(r["q25"]), .4):.2f}%"></span>'
                      f'<span class="dr-d {cls} {where}" style="left:{pos(r["median"]):.2f}%" tabindex="0" data-tip="{esc(tip)}"></span>')
        lcls = " best" if name in best.values() else ""
        rows.append(f'<div class="dr-row"><span class="dr-l{lcls}">{esc(name)}</span><div class="dr-t">'
                    + "".join(f'<span class="dr-g" style="left:{pos(t):.2f}%"></span>' for t in ticks)
                    + f'<span class="dr-one" style="left:{pos(1):.2f}%"></span>{marks}</div></div>')
    axis = '<div class="dr-row ax"><span class="dr-l"></span><div class="dr-t">' + "".join(
        f'<span class="dr-x" style="left:{pos(t):.2f}%">{t:g}×</span>' for t in ticks) + "</div></div>"
    return legend([("c-n5", "Instagram side"), ("c-n3", "Facebook side"), ("c-acc", "Closest to 1×")]) + f'<div class="dr">{"".join(rows)}{axis}</div>'


# ---------------------------------------------------------------- page sections
def overview(D):
    R, T, C, V = D["R"], D["T"], D["C"], D["V"]
    rec = R["recipes"]
    best, curve = rec["public - paid IG Impressions"], rec["pre-boost read x organic curve (no fitting on these posts)"]
    soc = rec["SocAPI total - FB cross-post - IG impressions - FB video plays (best mix)"]
    ig_cov = V["totals"]["Instagram"]
    P = (D["C2"] or C)["platforms"]

    a = card("Subtracting paid breaks down as paid grows",
             hero_columns(R)
             + '<div class="ask"><div class="ask-q"><span class="spark" aria-hidden="true">✦</span>What would make the numbers better?</div>'
               '<div class="ask-a">Tag every boosted ad with its post ID, wait 7–14 days before a boost, and run one '
               '<span class="chip" tabindex="0" data-tip="Run the ad as a separate dark post, not as a boost of the creator\'s own post. '
               'The creator\'s post then shows organic views only, which gives TikTok an organic number to check against.">TikTok dark-post test</span> '
               'to check TikTok organic.</div></div>',
             cls="span2 hero", sub="Paid Instagram posts: organic error of “public − paid impressions”, by paid share of public views",
             tip=f"Instagram, {R['postflight_posts']} paid posts, one read per post after the ads ended. Truth = opt-in views (organic only). Error = median absolute error.",
             data=table(["Paid share of public", "Posts", "Typical organic error"],
                        [[esc(r["paid_share"]), r["posts"], pct(r["median_abs_error"])] for r in R["error_by_paid_share"]]))
    b = card("Use the views before the boost",
             f'<div class="big">{pct(curve["median_abs_error"], 1)}</div><p class="big-sub">typical organic error, never negative</p>'
             + '<p class="mini">Posts within ±25% of true organic</p>'
             + progress([(f"Pre-boost views × organic growth ({curve['posts']} posts)", curve["within_25pct"], "solid",
                          f"{pct(curve['within_25pct'])} of {curve['posts']} posts within ±25%; typical error {pct(curve['median_abs_error'], 1)}"),
                         (f"Public − paid impressions ({best['posts']} posts)", best["within_25pct"], "hatch-n",
                          f"{pct(best['within_25pct'])} of {best['posts']} posts within ±25%; typical error {pct(best['median_abs_error'])}; {pct(best['negative_organic'])} negative"),
                         (f"SocAPI total − paid ({soc['posts']} posts)", soc["within_25pct"], "hatch-n",
                          f"{pct(soc['within_25pct'])} of {soc['posts']} posts within ±25%; typical error {pct(soc['median_abs_error'])}")]),
             sub="Instagram, checked against opt-in",
             tip=f"Take the last public read before the first ad day, then grow it by the median curve of unpaid posts. Checked against opt-in views on {curve['posts']} paid Instagram posts.",
             data=table(["Method", "Posts", "Typical error", "Within ±25%", "Negative"],
                        [[esc(n), rec[k]["posts"], pct(rec[k]["median_abs_error"]), pct(rec[k]["within_25pct"]), pct(rec[k]["negative_organic"])]
                         for k, n in [("pre-boost read x organic curve (no fitting on these posts)", "Pre-boost × curve"),
                                      ("public - paid IG Impressions", "Public − paid impressions"),
                                      ("SocAPI total - FB cross-post - IG impressions - FB video plays (best mix)", "SocAPI total − paid")]]))
    c = card("Normal organic growth", curve_chart(D["curves"]) + legend([("c-acc", "Instagram"), ("c-n4", "TikTok")]),
             sub="Share of day-120 views by post age, unpaid posts",
             tip="The method grows the pre-boost views along this curve. Built from unpaid posts only: up to "
                 + " and ".join(f'{max(n for _, _, n in D["curves"][p]):,} {lab}' for p, lab in PLATFORMS) + " posts per day of age.",
             data=table(["Day", "Instagram", "TikTok"],
                        [[a, pct(v, 1), pct(D["curves"]["Tiktok"][i][1], 1)] for i, (a, v, _) in enumerate(D["curves"]["Instagram"]) if a in (0, 1, 3, 7, 14, 30, 60, 90, 120)]))
    d = card("Organic we can measure",
             f'<div class="big sm">{pct(ig_cov["measured_optin"] / ig_cov["posts"])}</div><p class="big-sub">of paid Instagram posts have opt-in that gives true organic views. '
             + (f'Another {pct(ig_cov["optin_frozen"] / ig_cov["posts"])} have opt-in until it stopped updating.' if ig_cov.get("optin_frozen") else "")
             + '</p>' + dots(V),
             tip="Each dot is 1% of posts with an ad record in the daily table (all publish dates). Model-flagged posts are not in this chart. Estimate = pre-boost public read × organic curve; its "
                 "confidence depends on how late that read is: day 14+ high, day 7–13 medium, before day 7 low. Opt-in that stopped updating is "
                 "used up to the freeze, then grown along the organic curve.",
             data=table(["Platform", "Measured", "Opt-in, then estimate", "High", "Medium", "Low", "Not separable"],
                        [[lab] + [f'{V["totals"][p].get(k, 0):,}' for k in ("measured_optin", "optin_frozen", "est_high", "est_medium", "est_low", "not_separable")]
                         for p, lab in PLATFORMS]))
    tt, im = P["Tiktok"]["test_metrics"], P["Instagram"]["test_metrics"]
    run, C2_ = (D["C2"] or {}).get("run"), D["C2"] or {}
    tip = " ".join(f'{lab}: {m["tp"]} of {m["tp"] + m["fp"]} flags are paid ({pct(m["precision"], 1)}).' for lab, m in (("TikTok", tt), ("Instagram", im)))
    lead = ("of the posts the model flags as paid really are paid, on both TikTok and Instagram."
            if pct(tt["precision"]) == pct(im["precision"]) else f'of TikTok flags are really paid. Instagram: {pct(im["precision"])}.')
    e = (f'<article class="card inv"><span class="pill glass">Paid-post model</span>'
         f'<div class="inv-num" tabindex="0" data-tip="{esc(tip)}">{pct(tt["precision"])}</div>'
         f'<p class="inv-sub">{lead}<span class="inv-foot">{"Day-30 model v2.2" if run == "v22" else "Day-30 model"}, test posts '
         f'{day(C2_.get("test_cutoff", "2026-07-01"))} – {day(C2_.get("test_end", "2026-09-09"), True)}.</span></p></article>')
    f = card("Which paid metric matches the platform count", dot_range(R), cls="span3",
             sub="Extra views on the platform ÷ paid metric. 1× is a perfect match.",
             tip="Instagram side: (public − opt-in) ÷ paid Instagram-placement metric, 146 posts. Facebook side: SocAPI Facebook-paid plays ÷ paid Facebook-placement metric, 20 posts. Dot = median, bar = middle half. Log scale.",
             data=table(["Paid metric", "Instagram side", "Facebook side"],
                        [[esc(n), times(R["metric_match"]["instagram_side"][n]["median"]),
                          times(R["metric_match"]["facebook_side"][n]["median"]) if R["metric_match"]["facebook_side"][n]["n"] else "no data"]
                         for n in R["metric_match"]["instagram_side"]]))
    return f'<section id="overview" class="grid">{a}{b}{c}{d}{e}{f}</section>'


def millions(x):
    return f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}K"


def term(cls, text):
    """a term in its own color, the same color as its bar (one color per quantity, everywhere in the section)"""
    return f'<span class="t {cls}">{esc(text)}</span>'


def claim_card(kicker, claim, body, proof, *, cls="", attrs=""):
    """claim first (one sentence), then the chart that proves it, then where every number comes from"""
    return (f'<article class="card claim-card anim {cls}" {attrs}><div class="claim-head"><span class="kicker">{kicker}</span>'
            f'<h3 class="claim">{claim}</h3></div>{body}<p class="proof"><b>Proof.</b> {proof}</p></article>')


def dot_plot(rows, d):
    """one dot per post, stacked in 2.5-point bins of (organic + boosted) ÷ total; the band is ±10%"""
    n = len(d["instagram_side"]["counts"])
    lo, w = d["bin_from"], d["bin_width"]
    hi = lo + n * w
    x = lambda v: 100 * (v - lo) / (hi - lo)
    inband = lambda i: .9 - 1e-9 <= lo + i * w and lo + (i + 1) * w <= 1.1 + 1e-9
    out = ""
    for lab, sub, key, stat in rows:
        cols = "".join(f'<span class="dp-col">' + "".join(f'<i class="pop{" in" if inband(i) else ""}" style="--d:{.02 * i:.2f}s"></i>' for _ in range(c)) + "</span>"
                       for i, c in enumerate(d[key]["counts"]))
        out += (f'<div class="dp-row"><div class="dp-lab"><b>{lab}</b><span>{sub}</span></div>'
                f'<div class="dp-plot" role="img" aria-label="{esc(lab)}: {sum(d[key]["counts"])} posts, one dot each. {esc(re.sub("<[^>]+>", "", stat))}.">'
                f'<i class="dp-band" style="left:{x(.9):.1f}%;width:{x(1.1) - x(.9):.1f}%"></i>'
                f'<i class="dp-one" style="left:{x(1):.1f}%"></i><div class="dp-cols" style="--n:{n}">{cols}</div></div>'
                f'<div class="dp-stat">{stat}</div></div>')
    ticks = "".join(f'<span style="left:{x(v):.1f}%">{t}</span>' for v, t in ((lo, f"≤{lo:.0%}"), (.75, "75%"), (1, "100%"), (1.25, "125%"), (hi, f"≥{hi:.0%}")))
    return f'<div class="dp">{out}<div class="dp-axis"><span class="dp-gap"></span><div class="dp-ticks">{ticks}</div><span class="dp-gap"></span></div></div>'


def range_rows(groups, top=1.2):
    """median (dot) and middle half (bar) of (organic + boosted) ÷ total, on one 0–120% line with 100% marked"""
    x = lambda v: 100 * min(v, top) / top
    out = ""
    for g, cls, rows in groups:
        out += f'<div class="rr-g">{g}</div>'
        for lab, c, best in rows:
            tip = (f"Typical post: {pct(c['median'])}. Middle half of posts: {pct(c['q25'])} to {pct(c['q75'])}. "
                   f"Within 10%: {pct(c['within_10pct'])} of {c['posts']} posts.")
            out += (f'<div class="rr {cls}{" best" if best else ""}"><span class="rr-l">{lab}</span><div class="rr-track"><i class="rr-one" style="left:{x(1):.1f}%"></i>'
                    f'<i class="rr-iqr grow" style="left:{x(c["q25"]):.1f}%;width:{x(c["q75"]) - x(c["q25"]):.1f}%"></i>'
                    f'<i class="rr-dot" style="left:{x(c["median"]):.1f}%" tabindex="0" data-tip="{esc(tip)}"></i></div><b>{pct(c["median"])}</b></div>')
    return f'<div class="rr-wrap">{out}<div class="rr rr-axis"><span></span><div class="rr-ticks">' + "".join(
        f'<span style="left:{x(v):.1f}%">{v:.0%}</span>' for v in (0, .5, 1)) + '</div><b></b></div></div>'


def waffle(rows):
    """100 squares = all paid impressions; largest-remainder rounding so the squares add to 100"""
    tot = sum(r["impressions"] for r in rows)
    raw = [100 * r["impressions"] / tot for r in rows]
    n = [int(v) for v in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - n[i], reverse=True)[:100 - sum(n)]:
        n[i] += 1
    cls = {"Facebook": "f", "Instagram": "i"}
    cells = "".join(f'<i class="{cls.get(r["placement"], "x")} pop" style="--d:{.006 * k:.3f}s"></i>' for r, c in zip(rows, n) for k in range(c))
    return f'<div class="waffle" role="img" aria-label="Paid impressions by placement: {", ".join(f"{r["placement"]} {pct(v / 100, 2)}" for r, v in zip(rows, raw))}">{cells}</div>'


def mbars(rows, top=1.0, fmt=pct):
    """small horizontal bars on one 0-to-top scale: (label, value, color class); the value is printed at the end"""
    return ('<div class="mb">' + "".join(
        f'<div class="mb-r {c}"><span class="mb-l">{esc(lab)}</span><div class="mb-t"><i class="grow" style="width:{100 * min(v, top) / top:.1f}%;--d:{.1 * i:.1f}s"></i></div>'
        f'<b>{fmt(v)}</b></div>' for i, (lab, v, c) in enumerate(rows)) + '</div>')


def post100(ps, gap):
    """one typical boosted post as 100 views: paid, organic, and the size of a typical paid miss (hatched)"""
    return (f'<div class="p100" role="img" aria-label="A typical boosted post: {round(100 * ps)} of 100 views are paid; the paid count misses by about {round(100 * gap)}.">'
            f'<i class="sg i" style="width:{100 * ps:.1f}%"><span>{round(100 * ps)} paid</span></i><i class="sg o" style="width:{100 * (1 - ps):.1f}%"></i>'
            f'<i class="lim-err" style="left:{100 * (ps - gap):.1f}%;width:{100 * 2 * gap:.1f}%"></i></div>'
            f'<div class="p100-l"><span>100 views</span><span>{round(100 * (1 - ps))} {term("o", "organic")}</span></div>')


def reach100(a, b):
    """how close the parts come to the total: solid to a, light to b, a line at 100%"""
    return (f'<div class="r100" role="img" aria-label="Parts add up to {pct(a)} of the total, {pct(b)} with the Facebook correction."><i class="r100-a" style="width:{100 * a:.1f}%"></i>'
            f'<i class="r100-b" style="left:{100 * a:.1f}%;width:{100 * max(b - a, 0):.1f}%"></i><i class="r100-one"></i></div>'
            f'<div class="r100-l"><span>{pct(a)} parts</span><span>+ Facebook correction = {pct(b)}</span><span>100% total</span></div>')


def dots100(hit, lab):
    """100 squares; the first `hit` are marked"""
    return (f'<div class="d100" role="img" aria-label="{hit} of 100 {esc(lab)}">' + "".join(f'<i class="{"h" if i < hit else ""}"></i>' for i in range(100))
            + f'</div><p class="bt-cap"><b>{hit}</b> of 100 {esc(lab)}</p>')


def ratio_chips(rows):
    """paid metric chips: extra views ÷ metric, the best one marked"""
    return ('<div class="rc">' + "".join(f'<span class="rc-i{" on" if on else ""}"><b>{times(v) if v < 10 else f"{v:.0f}×"}</b>{esc(n)}</span>' for n, v, on in rows)
            + '</div><p class="bt-cap">Extra views ÷ paid metric. 1× = exact match.</p>')


def stat2(rows):
    return '<div class="s2">' + "".join(f'<div><b>{v}</b><span>{esc(n)}</span></div>' for v, n in rows) + '</div>'


def split2(a, la, lb):
    """one bar split in two: a and 1 - a"""
    return (f'<div class="fbx-bar"><i class="sg ip" style="width:{100 * a:.1f}%"><span>{pct(a)}</span></i><i class="sg fp" style="width:{100 * (1 - a):.1f}%"><span>{pct(1 - a)}</span></i></div>'
            f'<div class="fbx-lab"><span style="width:{100 * a:.1f}%">{esc(la)}</span><span>{esc(lb)}</span></div>')


def cols(rows):
    """small columns on a 0-100% scale"""
    return ('<div class="c5">' + "".join(f'<div class="c5-c"><b>{pct(v)}</b><div class="c5-t"><i style="height:{100 * v:.1f}%"></i></div><span>{esc(n)}</span></div>'
                                         for n, v in rows) + '</div>')


def start(D):
    """Start here: the goal, the ask, the outcome, and the four steps from a post to its true organic and paid views."""
    E, M, P = D.get("EQ"), D.get("PM"), (D["C2"] or D["C"])["platforms"]
    if not (E and M and M.get("coverage")):
        return ""
    cv, pd_, fb = M["coverage"], M["paid_from_organic_estimate"], M["facebook_factor"]
    ig, tt = cv["Instagram"], cv["Tiktok"]
    best = E["full_platform"]["with paid Facebook video plays"]
    eqf = fb["equation_with_factor"]
    curve = D["R"]["recipes"]["pre-boost read x organic curve (no fitting on these posts)"]
    sub_ = D["R"]["recipes"]["public - paid IG Impressions"]
    im, tm = P["Instagram"]["test_metrics"], P["Tiktok"]["test_metrics"]
    rule = P["Instagram"]["rules_test"][HAND_RULE]
    boosted = lambda c: c["boosted_by_record"] + c["boosted_by_model"]
    share_model = lambda c: c["boosted_by_model"] / boosted(c)
    k = fb["k_median_all_posts"]
    bs = (D.get("BS") or {}).get("test", {}).get("organic_instagram")

    R = D["R"]
    mm_ig, mm_fb = R["metric_match"]["instagram_side"], R["metric_match"]["facebook_side"]
    view_types = [v["median"] for n_, v in mm_ig.items() if n_ not in ("Impressions", "Reach", "Video plays (starts)") and v.get("n")]
    tags = {r["platform"]: r["posts"] for r in D["V"]["by_evidence"] if r["evidence"] == "CONFIRMED_PAID_TAG"}
    links = {r["platform"]: r["posts"] for r in D["V"]["by_evidence"] if r["evidence"] == "CONFIRMED_AD_LINK"}
    split = lambda c: 1 - c["organic_from"]["missing"] / boosted(c)
    mr = "missing_by_reason"
    early = lambda c: c[mr]["boosted_before_first_read"] + c[mr]["jump_before_day_3"]
    MR = D.get("MR") or {}
    cr = MR.get("creator", {})
    cr_err = [cr[k]["median_abs_error"] for k in ("boosted_tune", "boosted_test") if cr.get(k)]
    so = E["socapi"]
    other = sum(r["impressions"] for r in E["placements"]["rows"][2:]) / sum(r["impressions"] for r in E["placements"]["rows"])

    # bottom line: three answers as numbers with a small chart each; the detail sits in a tooltip
    ps, gap = R["paid_share_of_public"]["median"], E["instagram_side"]["Impressions"]["median_abs_gap"]
    tg = R["post_id_tag"]
    conf = R["production_function_by_confidence"]
    pl = E["placements"]["rows"]
    ptot = sum(r["impressions"] for r in pl)
    tiles = [("w", "Total − a paid metric = organic?", "No, not for one post.", f'{pct(sub_["median_abs_error"])} off',
              f'for a typical post. Organic goes below zero on {pct(sub_["negative_organic"])} of posts.',
              post100(ps, gap) + f'<p class="bt-cap">A typical boosted post: {round(100 * ps)} of 100 views are paid. The paid count misses by about {round(100 * gap)} (hatched), which is most of organic.</p>',
              f'Best paid metric (impressions), {sub_["posts"]} boosted Instagram posts, checked against creator data (opt-in). One read per post, 2 or more days after the last ad day.'),
             ("o", "A repeatable way to find organic?", "Yes: the views just before the boost.", f'{pct(curve["median_abs_error"], 1)} off',
              "for a typical post. Never below zero.",
              mbars([("Total − paid impressions", sub_["median_abs_error"], "w"), ("Views before the boost × normal growth", curve["median_abs_error"], "o")], top=1,
                    fmt=lambda v: pct(v, 1) if v < .2 else pct(v)),
              f'Take the last view count before the boost and grow it at the normal organic rate. {curve["posts"]} boosted Instagram posts, checked against creator data; '
              f'{pct(curve["within_25pct"])} are within 25%.'),
             ("i", "Organic + boosted = total seen?", "Yes, once Facebook is counted.", pct(eqf["median"]),
              f'of the total for a typical post, with a small Facebook correction ({pct(best["median"])} without).',
              reach100(best["median"], eqf["median"]) + f'<p class="bt-cap"><span class="pill warn">{best["posts"]} posts</span> In-app check: '
              f'<b class="ac-left">{best["posts"]}</b> of {best["posts"]} still to do.</p>',
              f'Organic (creator data) + paid Instagram impressions + paid Facebook plays, against the SocAPI total (Instagram + Facebook). '
              f'Correction: paid Facebook plays × {k:.1f}, fitted on the other posts each time.')]
    tile = lambda i, t: (f'<div class="bt {t[0]}"><span class="bt-k">{esc(t[1])}{info(t[6])}</span><b class="bt-a">{esc(t[2])}</b>'
                         f'<div class="bt-num"><b class="bt-v">{t[3]}</b><span>{esc(t[4])}</span></div>{t[5]}</div>')
    asks = [("Tag every boosted ad with the post ID or URL.", "Gives the exact start date and the paid numbers for each post."),
            ("Boost on day 3 or later, best day 7–14. Or use a dark post.", "A boost in the first days leaves no organic-only views to grow forward."),
            ("Run one TikTok post as a dark post.", "TikTok creator data also counts Spark Ad views, so today we cannot check TikTok organic.")]
    bottom = ('<article class="card span3 bluf"><span class="kicker">Bottom line</span><div class="bt3">'
              + "".join(tile(i, t) for i, t in enumerate(tiles)) + '</div>'
              + '<div class="ask3"><span class="bl-ak">The math works when Paid Media does this</span><ol>'
              + "".join(f'<li><span class="a3-n">{i}</span><span>{tipped(t, why)}</span></li>' for i, (t, why) in enumerate(asks, 1)) + '</ol></div>'
              + f'<p class="bl-cav"><span class="pill warn">What could change this</span><span>The Facebook sum rests on {best["posts"]} posts, and no one has checked the totals in the app yet.</span></p></article>')

    # every question: the short answer and one number in the row; open it for a small chart and the proof
    n_early = early(ig) + early(tt)
    qs = [("Does total − a paid metric = organic?", "No", "no", f'{pct(sub_["median_abs_error"])} off', post100(ps, gap),
           f'Best paid metric (impressions): {pct(sub_["median_abs_error"])} off for a typical post. Organic goes below zero on {pct(sub_["negative_organic"])} of posts '
           f'({sub_["posts"]} Instagram posts). Paid is {pct(ps)} of a boosted post\'s views, so a small paid miss is most of organic.', "#overview"),
          ("Which paid metric matches the views?", "Impressions · plays", "info", f'{times(mm_ig["Impressions"]["median"])} · {times(mm_fb["Video plays (starts)"]["median"])}',
           ratio_chips([("Impressions", mm_ig["Impressions"]["median"], True), ("Video plays", mm_ig["Video plays (starts)"]["median"], False),
                        ("3-second views", mm_ig["3-second video views"]["median"], False), ("25% watched", mm_ig["25% watched"]["median"], False),
                        ("ThruPlays", mm_ig["ThruPlays"]["median"], False), ("100% watched", mm_ig["100% watched"]["median"], False)]),
           f'Instagram: extra views = {times(mm_ig["Impressions"]["median"])} paid impressions ({mm_ig["Impressions"]["n"]} posts). Facebook: '
           f'{times(mm_fb["Video plays (starts)"]["median"])} paid video plays ({best["posts"]} posts). The other view types are {min(view_types):.0f} to {max(view_types):.0f} times too small.', "#eq-types"),
          ("We started with the post-ID tagged posts. What do they show?", "Same: it fails", "no", f'{pct(tg["public - paid IG impressions"]["negative_organic"])} below zero',
           dots100(round(100 * tg["public - paid IG impressions"]["negative_organic"]), "days with negative organic"),
           f'{tg["posts"]} tagged Instagram posts have daily creator data. On {round(tg["post_days"] * tg["public - paid IG impressions"]["negative_organic"])} of '
           f'{tg["post_days"]} days while the ads ran, total − paid impressions gave negative organic. That is too few posts alone, so we added '
           f'{R["panel"]["posts"] - tg["posts"]} boosted posts matched by the post key in the ad name ({R["panel"]["posts"]} in total, reads from {day(R["panel"]["obs_from"], True)}).', "#faq"),
          ("Is there a repeatable way to find organic?", "Yes", "yes", f'{pct(curve["median_abs_error"], 1)} off',
           mbars([("Read on day 14+", conf["high"]["median_abs_error"], "o"), ("Read on day 7–13", conf["medium"]["median_abs_error"], "o"),
                  ("Read before day 7", conf["low"]["median_abs_error"], "w")], top=.5, fmt=lambda v: pct(v, 1)),
           f'The last view count before the boost, grown at the normal organic rate: {pct(curve["median_abs_error"], 1)} off for a typical post, '
           f'{pct(curve["within_25pct"])} within 25%, never below zero ({curve["posts"]} posts). The later the read, the better.'
           + (f' With no ad date, the start comes from the jump in daily views: {pct(bs["median_abs_error"], 1)} off ({bs["posts"]} posts).' if bs else ""), "#findings"),
          ("Do we still need dark posts?", "Before day 3", "part", f'{n_early:,} posts',
           stat2([(f"{early(ig):,}", "Instagram"), (f"{early(tt):,}", "TikTok")]),
           "These boosted posts were boosted before day 3 or before our first read, so there are no organic-only views to grow. "
           + (f'Every other signal we tried is {min(cr_err):.0%}–100% off.' if cr_err else ""), "#tried"),
          ("For posts VN boosted: organic + boosted = total seen?", "Yes, with Facebook", "yes", f'{pct(best["median"])} → {pct(eqf["median"])}',
           reach100(best["median"], eqf["median"]),
           f'Organic + paid Instagram impressions + paid Facebook plays = {pct(best["median"])} of the SocAPI total for a typical post; all {best["posts"]} posts are within 25%. '
           f'With paid Facebook plays × {k:.1f}: {pct(eqf["median"])}, and {round(eqf["within_10pct"] * eqf["posts"])} of {eqf["posts"]} posts are within 10%.', "#equation"),
          ("Callout 1: how much of the total is Facebook?", "Most of it", "info", pct(so["facebook_share_of_total_median"]),
           split2(1 - so["facebook_share_of_total_median"], "Instagram (Nimble)", "Facebook"),
           f'Nimble shows only the Instagram part. SocAPI\'s Instagram plays equal Nimble ({so["instagram_plays_over_nimble_median"]:.2f}×); '
           f'the full total is {so["total_over_nimble_median"]:.2f}× Nimble for a typical post ({so["posts"]} posts).', "#eq-fb"),
          ("Callout 2: do other placements matter?", "No", "yes", f'{pct(other, 2)}',
           f'<div class="wf-mini">{waffle(pl)}</div>',
           f'Audience Network, Messenger and unknown placements get {pct(other, 2)} of paid impressions. Facebook gets {pct(pl[0]["impressions"] / ptot, 1)}, '
           f'Instagram {pct(pl[1]["impressions"] / ptot, 1)} ({ptot:,} paid impressions on ads linked to Instagram campaign posts).', "#eq-types"),
          ("Callout 3: is our total the total in the app?", "Not checked yet", "wait", f'<span class="ac-left">{best["posts"]}</span> left',
           f'<div class="acp"><div class="acp-t"><i class="ac-prog" style="width:0%"></i></div><span><b class="ac-done">0</b> of {best["posts"]} checked</span></div>',
           f'Nimble counts Instagram only. Open each of the {best["posts"]} posts in the app and type the view count you see. The list is shared, and the page says which total each count is closer to.', "#appcheck"),
          ("The simple rule (ER < 1%, views > followers) vs the model?", "Model finds more", "info", f'{pct(rule["recall"])} → {pct(im["recall"])}',
           mbars([("Rule: boosts found", rule["recall"], "n"), ("Model: boosts found", im["recall"], "i"),
                  ("Rule: flags right", rule["precision"], "n"), ("Model: flags right", im["precision"], "i")], top=1),
           f'Instagram test posts, {im["tp"] + im["fn"]} boosted. The rule flags {pct(rule["fpr"])} of unboosted posts. TikTok: the rule finds '
           f'{pct(P["Tiktok"]["rules_test"][HAND_RULE]["recall"])}, the model {pct(tm["recall"])}. Posts VN boosted have ad records, so the equation needs neither.', "#need")]
    rows_q = "".join(f'<details class="qx"><summary><span class="qx-n">{i}</span><span class="qx-q">{esc(q)}</span><span class="qx-v {vc}">{esc(v)}</span>'
                     f'<b class="qx-k">{key}</b><span class="qx-c" aria-hidden="true"></span></summary>'
                     f'<div class="qx-b"><div class="qx-viz">{viz}</div><div class="qx-t"><p>{txt}</p><a href="{h}">See the proof →</a></div></div></details>'
                     for i, (q, v, vc, key, viz, txt, h) in enumerate(qs, 1))
    answers = ('<article class="card span3" id="questions"><div class="card-head qx-head"><div><h3>Paid Media\'s questions, answered</h3>'
               '<p class="card-sub">Short answer and one number per row. Open a row for its chart and proof.</p></div>'
               '<button type="button" class="qx-all" data-open="0">Open all</button></div><div class="qx-list">' + rows_q + '</div>'
               f'<p class="proof"><b>Basis.</b> Organic truth = creator account data (opt-in), Instagram only. Typical = the median post. '
               f'Post-ID tags so far: {tags.get("Instagram", 0) + tags.get("Tiktok", 0)} posts; ads matched by the post key in the ad name: '
               f'{links.get("Instagram", 0):,} Instagram and {links.get("Tiktok", 0):,} TikTok posts.</p></article>')

    def step(n, q, big, small, tip, proof, href):
        return (f'<div class="step"><span class="step-n">{n}</span><h3>{q}</h3><div class="step-v"><b>{big}</b><span>{small}{info(tip)}</span></div>'
                f'<a class="step-p" href="{href}">{proof} →</a></div>')
    steps = ('<div class="steps">'
             + step(1, "Was it boosted?", f'{boosted(ig) + boosted(tt):,}', "boosted posts: ad record or model flag",
                    f'Ad records: {ig["boosted_by_record"]:,} Instagram and {tt["boosted_by_record"]:,} TikTok posts. With no record, the model reads the view pattern: '
                    f'{ig["boosted_by_model"]:,} more on Instagram and {tt["boosted_by_model"]:,} more on TikTok.',
                    f'{round(100 * im["precision"])} in 100 model flags are right.', "#model")
             + step(2, f"How many views are {term('o', 'organic')}?", f'{pct(curve["median_abs_error"], 1)} off', "views before the boost × normal growth",
                    'Creator account data when we have it. If not, the views just before the boost, grown at the normal rate. With no start date, the start comes from the jump in daily views.',
                    f'{curve["posts"]} posts vs creator data.', "#gaps")
             + step(3, f"How many are {term('i', 'paid')}?", f'{pct(pd_["public_minus_organic_estimate"]["median_abs_error"], 1)} off', "total − the organic estimate",
                    f'Paid is most of the views, so a small organic miss is a tiny paid miss. Paid ad impressions alone: {pct(pd_["paid_instagram_impressions"]["median_abs_error"])} off.',
                    f'{pd_["public_minus_organic_estimate"]["posts"]} posts vs creator data.', "#findings")
             + step(4, "What was the total seen?", f'{pct(fb["total_from_nimble_plus_k_fb_plays"]["median_abs_error"], 1)} off', f"Nimble + {k:.1f} × paid Facebook plays",
                    'Nimble shows the Instagram part only. The Facebook part is about 1.1 × the paid Facebook plays.',
                    f'Only {fb["total_from_nimble_plus_k_fb_plays"]["posts"]} posts so far.', "#equation")
             + '</div>')
    flow = claim_card("How it works", "Four questions take a post from public views to its true organic and paid views.",
                      steps, f"Counts: daily table run 2026-10-08 (ad records read 2026-10-07). Errors: typical post vs creator data or SocAPI, one read per post 2 or more days after the last ad day. "
                             f"Model test: posts published {day(D['C2'].get('test_cutoff', '2026-07-01'))} – {day(D['C2'].get('test_end', '2026-09-09'), True)}.",
                      cls="span3")

    def cov_bar(c, lab):
        n = boosted(c)
        o = c["organic_from"]
        segs = [("cd", o["creator_data"], "creator data"), ("es", o["estimated_known_start"], "estimated from the known start date"),
                ("ej", o["estimated_view_jump"], "estimated, start found from the view jump"), ("ms", o["missing"], "missing")]
        bar = "".join(f'<i class="sg {k_} grow" style="width:{100 * v / n:.2f}%;--d:{.25 * i}s" tabindex="0" data-tip="{esc(f"{v:,} of {n:,} boosted {lab} posts: {t}.")}">'
                      + (f'<span>{pct(v / n)}</span>' if v / n >= .08 else "") + "</i>" for i, (k_, v, t) in enumerate(segs) if v)
        return f'<div class="cov-row"><span class="cov-l"><b>{lab}</b>{n:,} boosted posts</span><div class="cov-bar">{bar}</div></div>'
    cov = claim_card("Where we stand today",
                     f"We can split {pct(split(ig))} of boosted Instagram posts into organic and paid, and {pct(split(tt))} of boosted TikTok posts.",
                     cov_bar(ig, "Instagram") + cov_bar(tt, "TikTok")
                     + legend([("cov-k cd", "Creator data (measured)"), ("cov-k es", "Known start date"),
                               ("cov-k ej", "Start found from the view jump"), ("cov-k ms", "Missing: no split yet")])
                     + '<details class="more-d"><summary>Show the counts</summary>'
                     + table(["Boosted posts", "Creator data", "Known start", "View jump", "Missing"],
                             [[lab] + [f'{c["organic_from"][k_]:,}' for k_ in ("creator_data", "estimated_known_start", "estimated_view_jump", "missing")]
                              for c, lab in ((ig, "Instagram"), (tt, "TikTok"))]) + '</details>',
                     f"Boosted = ad record or model flag. Missing = no clean views before the boost (<a href=\"#gaps\">fixes</a>). Daily table: evidence read 2026-10-07, daily views pulled 2026-10-08. "
                     f"The model cannot score {ig['not_scored']:,} Instagram and {tt['not_scored']:,} TikTok posts, mostly because we have no public read in their first 60 days.",
                     cls="span2")
    need = claim_card("Do we need the model?",
                      "Not for posts VN boosted. Yes, to find boosts with no ad record.",
                      f'<div class="facts"><div><b class="t i">{pct(share_model(ig))}</b><span>of boosted Instagram posts have no ad record. Only the model finds them.</span></div>'
                      f'<div><b class="t i">{pct(share_model(tt))}</b><span>of boosted TikTok posts have no ad record.</span></div>'
                      f'<div><b>{pct(im["recall"])}</b><span>of Instagram boosts found by the model, {pct(rule["recall"])} by the simple rule.</span></div></div>',
                      f"Posts VN boosted have ad records, so the equation needs no model. Model vs rule: Instagram test posts, {im['tp'] + im['fn']} boosted. "
                      f"The rule is right on {pct(rule['precision'])} of its flags, the model on {pct(im['precision'])}.", attrs='id="need"')
    nj = MR.get("no_clear_jump_instagram_optin")
    gaps = [(f"{early(ig):,} Instagram and {early(tt):,} TikTok posts boosted before day 3",
             "Boosted before day 3 or before our first read. No organic-only views exist before the boost, so there is nothing to grow forward."
             + (f" Every other signal we tried is {min(cr_err):.0%}–100% off." if cr_err else ""),
             "Boost on day 3 or later, or use a dark post.", "Paid Media"),
            (f"{ig[mr]['no_clear_jump']:,} Instagram and {tt[mr]['no_clear_jump']:,} TikTok posts with no clear jump",
             "The daily views never jump, so we cannot see when the boost started."
             + (f" Most are small boosts: on {nj['posts']} such Instagram posts with creator data, a typical {pct(nj['paid_share_median'])} of views are paid "
                f"(the {R['paid_share_of_public']['n']} boosted posts in the subtraction test: {pct(R['paid_share_of_public']['median'])})." if nj else ""),
             "Match them to ad records.", "BI · next"),
            ("TikTok organic after a boost", "TikTok creator data also counts Spark Ad views, so we cannot check the estimate.",
             "Run one TikTok dark-post test.", "Paid Media"),
            (f"The Facebook correction ({k:.1f}×)", f"It rests on {fb['total_from_nimble_plus_k_fb_plays']['posts']} posts with SocAPI data.",
             f"Check the {best['posts']} posts in the app. Collect more SocAPI reads.", "BI + Paid Media")]
    rows = "".join(f'<li><span class="gap-n">{i}</span><div><b>{esc(w)}{info(why)}</b></div><div><span class="gap-fix">{esc(fix)}</span>'
                   f'<span class="gap-o">{esc(o)}</span></div></li>' for i, (w, why, fix, o) in enumerate(gaps, 1))
    miss = claim_card("What is still missing", f"{len(gaps)} gaps remain, and each has a fix.",
                      f'<ol class="gaps">{rows}</ol>',
                      f"Daily table, data read 2026-10-07; daily views pulled 2026-10-08. View-jump test: {(D.get('BS') or {}).get('posts', {}).get('ig_boosted', 0) + (D.get('BS') or {}).get('posts', {}).get('tt_boosted', 0):,} boosted posts, "
                      "tuned on half and scored once on the other half. Paid can be predicted wherever organic can (step 3).",
                      cls="span3", attrs='id="gaps"')
    tried = ""
    if MR and bs:
        wg, sr, lk = MR["window_growth_organic"]["Instagram day 7-21"], MR["slow_ramp"], MR["likes"]["organic_from_likes_test"]
        su, aa = MR["speed_up"], MR["after_the_ads_panel"]
        rows_ = [("Views before the boost, ad start date", curve, True, "the method in use"),
                 ("Views before the boost, start found from the view jump", bs, True, "in use since 2026-10-08"),
                 ("The creator's usual organic views", cr["boosted_test"], False, "median of the creator's unboosted posts"),
                 ("Slow ramp: growth over 5 days", sr["instagram_organic_vs_optin"], False, f"finds {sr['found']} of {sr['boosted_no_jump_posts']} no-jump posts"),
                 ("Growth in a later window (day 7–21)", wg, False, "best case: tested on unboosted posts"),
                 ("Total − paid impressions", sub_, False, "the subtraction Paid Media asked about"),
                 ("Likes per view", lk, False, "paid views get far fewer likes")]
        tr = "".join(f'<div class="tr{" on" if on else ""}"><span class="tr-l">{esc(n_)}<small>{esc(why)} · {e["posts"]:,} posts</small></span>'
                     f'<div class="tr-track"><i class="tr-fill grow" style="width:{100 * min(e["median_abs_error"], 1):.1f}%;--d:{.08 * i:.2f}s"></i></div>'
                     f'<b>{pct(e["median_abs_error"], 1) if e["median_abs_error"] < .2 else pct(e["median_abs_error"])}</b></div>'
                     for i, (n_, e, on, why) in enumerate(rows_))
        tr += '<div class="tr tr-ax"><span></span><div class="rr-ticks"><span style="left:0%">0%</span><span style="left:25%">25%</span><span style="left:50%">50%</span><span style="left:100%">100% off</span></div><b></b></div>'
        g = lambda p, ph: su[p][ph]["share_up_25pct"]
        speed = mbars([("Instagram, first week of ads", g("Instagram", "first 7 days of ads"), "i"), ("Instagram, unboosted", g("Instagram", "organic posts"), "n"),
                       ("TikTok, first week of ads", g("Tiktok", "first 7 days of ads"), "i"), ("TikTok, unboosted", g("Tiktok", "organic posts"), "n")], top=.5)
        after = cols([("Before", aa["before the ads"]["paid_share_of_daily_gain_median"]), ("During", aa["during the ads"]["paid_share_of_daily_gain_median"]),
                      ("+2 days", aa["2 days after"]["paid_share_of_daily_gain_median"]), ("+3 days", aa["3 days after"]["paid_share_of_daily_gain_median"]),
                      ("Days 4–30", aa["4-30 days after"]["paid_share_of_daily_gain_median"])])
        tried = claim_card("Can we predict the posts we still miss?",
                           f"Only from the views before the boost. Every other signal we tried is {min(cr_err):.0%}–100% off for a typical post.",
                           f'<div class="tried" role="img" aria-label="Typical organic error by method">{tr}</div>'
                           + legend([("tr-k on", "In use"), ("tr-k", "Tested, not used")])
                           + f'<div class="two-mini"><div><h4>Does a boost show as faster growth?</h4>{speed}'
                             '<p class="bt-cap">Share of reads where daily views rise 25% or more. Boosts speed up only a little more often than unboosted posts, so we look for one large jump.</p></div>'
                             f'<div><h4>Do paid views stop when the ads stop?</h4>{after}'
                             '<p class="bt-cap">Paid share of the daily gain in public views. Paid views fade slowly, so growth after the ads is not organic.</p></div></div>',
                           f"Error = typical gap to creator data (opt-in), Instagram boosted posts unless stated. Fitted parts were tuned on half the posts and scored once on the other half. "
                           f"Speed-up: {su['Instagram']['first 7 days of ads']['intervals']:,} and {su['Instagram']['organic posts']['intervals']:,} Instagram read-to-read steps after day 3. "
                           f"After the ads: daily reads of {aa['during the ads']['posts']} boosted posts with creator data. Data read 2026-10-07 and 2026-10-08.",
                           cls="span3", attrs='id="tried"')
    head = ('<div class="block-head"><div><h2>Answers first</h2><p class="card-sub">The bottom line, then each question with its answer. Open a row for the chart and proof.</p></div></div>')
    return f'<section id="start" class="block">{head}<div class="grid">{bottom}{answers}{flow}{cov}{need}{miss}{tried}</div></section>'

def equation(D):
    """Posts VN boosted itself: does organic + boosted = the total seen on the platform? Claims first, each proved by one chart."""
    E, R = D.get("EQ"), D["R"]
    if not E:
        return ""
    ig, fp, so, po, pl = E["instagram_side"], E["full_platform"], E["socapi"], E["pooled_full_platform"], E["placements"]["rows"]
    best, ig_best = fp["with paid Facebook video plays"], ig["Impressions"]
    parts = po["organic"] + po["paid_instagram_impressions"] + po["paid_facebook_plays"] + po["paid_other_plays"]
    total, nim = po["socapi_total"], po["nimble"]
    fb_total = total - nim
    ptot = sum(r["impressions"] for r in pl)
    fb_share = pl[0]["impressions"] / ptot
    other = sum(r["impressions"] for r in pl[2:]) / ptot
    window = f"{day(so['reads_from'])} – {day(so['reads_to'], True)}"
    o, i_, f_ = term("o", "organic"), term("i", "paid Instagram"), term("f", "paid Facebook")
    n_in10 = round(best["within_10pct"] * best["posts"])
    ps = R["paid_share_of_public"]["median"]
    sub_ = R["recipes"]["public - paid IG Impressions"]
    gap = ig_best["median_abs_gap"]
    FF = (D.get("PM") or {}).get("facebook_factor")

    # the answer, in four lines an executive can repeat
    strip = [("1", "It adds up", pct(best["median"]), f"of the total, for a typical post, once Facebook is counted. All {best['posts']} posts are within 25%.", "#eq-sum"),
             ("2", "Facebook", pct(so["facebook_share_of_total_median"]), "of the total is on Facebook. Nimble shows the Instagram part only.", "#eq-fb"),
             ("3", "Placements", pct(other, 2), "of paid impressions ran outside Facebook and Instagram. Too small to matter.", "#eq-types"),
             ("4", "Still to do", f'<span class="ac-left">{best["posts"]}</span>', "posts to check by hand in the Instagram app before we quote a total.", "#appcheck")]
    tiles = "".join(f'<a class="ans" href="{h}"><span class="ans-k"><i>{k}</i>{esc(t)}</span><b>{v}</b><span class="ans-s">{esc(sx)}</span></a>' for k, t, v, sx, h in strip)
    head = (f'<div class="block-head"><div><h2>Organic + boosted = total seen?</h2>'
            f'<p class="card-sub">Paid Media\'s ask, for posts VN boosted itself. One read per post, 2 or more days after the last ad day. Each claim below links to the chart that proves it.</p></div></div>'
            f'<div class="answers">{tiles}</div>')

    # claim 1: the sum, built up term by term on one scale
    scale = max(total, parts)
    seg = lambda v, cls, d, lab="", short="": (f'<i class="sg {cls} grow" style="width:{100 * v / scale:.2f}%;--d:{d}s">'
                                               + (f'<span class="lg">{lab}</span><span class="sm">{short}</span>' if lab else "") + "</i>")
    sumtile = lambda cls, lab, v, tip: f'<div class="eqn-t {cls}"><span>{lab}</span><b class="num" tabindex="0" data-tip="{esc(tip)}">{millions(v)}</b></div>'
    opx = lambda c: f'<span class="eqn-op" aria-hidden="true">{c}</span>'
    eqn = ('<div class="eqn">'
           + sumtile("o", "organic", po["organic"], "Creator account data (opt-in). It counts Instagram organic views only.")
           + opx("+") + sumtile("i", "paid Instagram", po["paid_instagram_impressions"], "Paid impressions on Instagram placements, from our Meta ad data.")
           + opx("+") + sumtile("f", "paid Facebook", po["paid_facebook_plays"], "Paid video plays on Facebook placements, from our Meta ad data.")
           + opx("≈") + sumtile("tot", "total seen", total, f"SocAPI total plays, Instagram + Facebook. Nimble shows {millions(nim)} for the same posts: Instagram only.")
           + "</div>")
    bars = (f'<div class="eqb"><div class="eqb-row"><span class="eqb-l">Our parts</span><div class="eqb-track">'
            f'{seg(po["organic"], "o", 0)}{seg(po["paid_instagram_impressions"], "i", .25, "paid Instagram", "IG")}{seg(po["paid_facebook_plays"] + po["paid_other_plays"], "f", .5, "paid Facebook", "paid FB")}'
            f'</div><b class="eqb-r">{pct(parts / total)}</b></div>'
            f'<div class="eqb-row"><span class="eqb-l">Total seen</span><div class="eqb-track">{seg(nim, "ip", .85, "Instagram part (Nimble)", "IG")}{seg(fb_total, "fp", 1.05, "Facebook part", "FB part")}'
            f'</div><b class="eqb-r">100%</b></div>'
            f'<div class="eqb-row"><span class="eqb-l"></span><div class="eqb-brk"><span class="brk" style="width:{100 * nim / scale:.2f}%">'
            f'{o} + {term("i", "paid IG")} = {pct((po["organic"] + po["paid_instagram_impressions"]) / nim)} of it</span>'
            f'<span class="brk" style="width:{100 * fb_total / scale:.2f}%">{term("f", "paid Facebook")} = {pct(po["paid_facebook_plays"] / fb_total)} of it</span></div><b></b></div></div>')
    c1 = claim_card("Claim 1 · the answer",
                    f"Yes. {o} + {i_} + {f_} adds up to {pct(parts / total)} of the total seen.",
                    eqn + bars + f'<p class="claim-note">The {pct(1 - parts / total)} gap sits mostly in the Facebook part: paid Facebook plays count a little below what SocAPI shows.'
                    + (f' Multiply paid Facebook plays by {FF["k_median_all_posts"]:.1f} and the sum comes to {pct(FF["equation_with_factor"]["median"])} for a typical post, '
                       f'{round(FF["equation_with_factor"]["within_10pct"] * FF["equation_with_factor"]["posts"])} of {FF["equation_with_factor"]["posts"]} posts within 10% '
                       f'(factor fitted on the other posts each time).' if FF else "") + '</p>',
                    f"{best['posts']} posts VN boosted, with SocAPI data, all added together (reads {window}). Organic = creator account data (Instagram only). "
                    f"Paid = our Meta ad data by placement: impressions on Instagram, video plays on Facebook. Total = SocAPI plays, Instagram + Facebook.",
                    cls="span3", attrs='id="eq-sum"')

    # claim 1, post by post
    dp = dot_plot([("Instagram only", f"{ig_best['posts']} posts. {term('o', 'Organic')} + {term('i', 'paid Instagram')}, compared with Nimble.",
                    "instagram_side", f"<b>{pct(ig_best['within_10pct'])}</b> within 10%"),
                   ("Instagram + Facebook", f"{best['posts']} posts. {term('o', 'Organic')} + {term('i', 'paid IG')} + {term('f', 'paid FB')}, compared with SocAPI.",
                    "full_platform", f"<b>{n_in10} of {best['posts']}</b> within 10%")], E["dots"])
    c1b = claim_card("Claim 1 · post by post", "It is not an average trick. Post by post, the sum lands close to 100%.",
                     '<p class="claim-sub">Each dot is one post. 100% means organic + paid equals the total exactly. The shaded band is within 10%.</p>' + dp + legend([("dp-k in", "Within 10% (shaded band)"), ("dp-k", "Outside 10%")]),
                     f"Each dot is one post: (organic + paid) ÷ total. Typical post: {pct(ig_best['median'])} on Instagram only ({ig_best['posts']} posts), "
                     f"{pct(best['median'])} with Facebook ({best['posts']} posts). Same read rule as above.", cls="span3")

    # callout 1: Facebook cross-posting
    igp = 1 - so["facebook_share_of_total_median"]
    c2 = claim_card("Callout 1 · Facebook cross-posting",
                    f"Nimble shows the Instagram part only. The full total is {so['total_over_nimble_median']:.2f}× bigger.",
                    f'<div class="fbx"><div class="fbx-bar"><i class="sg ip grow" style="width:{100 * igp:.1f}%"><span>{pct(igp)}</span></i>'
                    f'<i class="sg fp grow" style="width:{100 * (1 - igp):.1f}%;--d:.3s"><span>{pct(1 - igp)}</span></i></div>'
                    f'<div class="fbx-lab"><span style="width:{100 * igp:.1f}%">Instagram: what Nimble shows</span><span>Facebook: what Nimble misses</span></div></div>'
                    f'<div class="facts"><div><b>{so["instagram_plays_over_nimble_median"]:.2f}×</b><span>SocAPI\'s Instagram number ÷ Nimble. The Instagram parts agree, so the extra is Facebook.</span></div>'
                    f'<div><b class="t f">{pct(fb_share, 1)}</b><span>of paid impressions on these campaigns ran on Facebook. The Facebook part is mostly paid.</span></div></div>',
                    f"Typical post of {so['posts']} (reads {window}). SocAPI's Instagram plays equal Nimble ({so['instagram_plays_over_nimble_median']:.2f}×), so the extra is Facebook. "
                    f"SocAPI's Facebook-only field is empty, so Facebook = total − Instagram. Paid split: Snowflake EDW Meta ad tables, all dates.",
                    cls="eq-fbcard", attrs='id="eq-fb"')

    # callout 2: which paid view type, and which placements
    lab = {"Impressions": "impressions", "Video plays (starts)": "video plays", "3-second video views": "3-second views", "ThruPlays": "ThruPlays"}
    rr = range_rows([(f"Instagram only · {ig_best['posts']} posts", "i", [(f"+ {term('i', 'paid IG ' + lab[k])}", ig[k], k == "Impressions") for k in lab]),
                     (f"Instagram + Facebook · {best['posts']} posts", "f", [(f"+ {term('f', 'paid FB video plays')}", best, True),
                                                                         (f"+ {term('f', 'paid FB impressions')}", fp["with paid Facebook impressions"], False)])])
    wf = (f'<div class="wf-box">{waffle(pl)}<div class="wf-key">{legend([("wf-f", f"Facebook {pct(fb_share, 1)}"), ("wf-i", "Instagram " + pct(pl[1]["impressions"] / ptot, 1)), ("wf-x", "Other " + pct(other, 2))])}'
          f'<p>1 square = 1% of paid impressions. Audience Network, Messenger and unknown placements do not fill one square.</p></div></div>')
    c3 = claim_card("Callout 2 · view types and placements",
                    f"Count {term('i', 'impressions on Instagram')} and {term('f', 'plays on Facebook')}. Other view types miss most views.",
                    f'<div class="c3-grid"><div><p class="claim-sub">(Organic + paid) ÷ total, by paid metric. Dot = typical post, bar = middle half.</p>{rr}</div>'
                    f'<div><p class="claim-sub">Where paid impressions ran</p>{wf}</div></div>',
                    f"Same posts and reads as Claim 1. On Facebook, plays give the tightest spread; impressions land closer on average but spread wider. "
                    f"Placements: {ptot:,} paid impressions on Meta ads linked to Instagram campaign posts (Snowflake EDW, all dates).",
                    cls="span2", attrs='id="eq-types"')

    # callout 3: the in-app check (rows live in this artifact's database, never in the page source)
    c4 = claim_card("Callout 3 · check the total in the app",
                    "Before we quote a total, a person checks it in the Instagram app.",
                    '<p class="claim-sub">Our warehouse may not read the total cleanly. Open each post in the app and type the view count you see. '
                    'The page says which total it is closer to. Everyone with this page sees the same list.</p>'
                    '<p class="ac-sum" aria-live="polite">The list loads when this page is open in claude.ai.</p>'
                    '<div class="table-wrap"><table class="ac"><thead><tr><th>Post</th><th class="r">Instagram only (Nimble)</th><th class="r">Instagram + Facebook</th>'
                    '<th class="r">Views in app</th><th>Cross-posted to Facebook?</th><th>Closer to</th></tr></thead><tbody></tbody></table></div>',
                    "If most app counts are closer to Instagram + Facebook, the full-platform sum is the one to quote. "
                    "Nimble and SocAPI numbers are the latest warehouse reads for each post.",
                    cls="span3", attrs='id="appcheck"')

    # limit: why it cannot run backwards for one post
    org = 1 - ps
    lim = claim_card("Limit · one post",
                     f"It adds up for a total. It does not run backwards to give one post's {o}.",
                     f'<div class="lim"><div class="lim-bar"><i class="sg i grow" style="width:{100 * ps:.1f}%"><span>{round(100 * ps)} paid</span></i>'
                     f'<i class="sg o grow" style="width:{100 * org:.1f}%;--d:.3s"></i>'
                     f'<i class="lim-err" style="left:{100 * (ps - gap):.1f}%;width:{100 * 2 * gap:.1f}%"></i></div>'
                     f'<div class="lim-lab"><span>A typical boosted post: 100 views</span><span class="lim-o">{round(100 * org)} {term("o", "organic")}</span></div>'
                     f'<p class="claim-note">The paid count is typically about {round(100 * gap)} views in 100 off (hatched). That miss is almost as big as the whole {o} part. '
                     f'So {o} = total − paid is {pct(sub_["median_abs_error"])} off for a typical post, and {pct(sub_["negative_organic"])} of posts go below zero. '
                     f'For one post, use the pre-boost method in Findings.</p></div>',
                     f"{sub_['posts']} boosted Instagram posts, checked against creator organic. Paid share = typical post. Typical miss = the median gap between "
                     f"organic + paid impressions and Nimble ({pct(gap, 1)}). TikTok cannot be tested: its creator data counts Spark Ad views.",
                     cls="span3 lim-card")
    return f'<section id="equation" class="block">{head}<div class="grid">{c1}{c1b}{c2}{c3}{c4}{lim}</div></section>'


def findings(D):
    R, T, C, V = D["R"], D["T"], D["C"], D["V"]
    rec = R["recipes"]
    best, curve = rec["public - paid IG Impressions"], rec["pre-boost read x organic curve (no fitting on these posts)"]
    ig = R["metric_match"]["instagram_side"]
    fb = R["metric_match"]["facebook_side"]
    g = C["tag_gold"]
    P = C["platforms"]
    tt, im = P["Tiktok"]["test_metrics"], P["Instagram"]["test_metrics"]
    cl = R["campaign_level"]
    items = [
        ("Paid impressions are the paid number that matches the extra views on the platform.",
         f"Instagram: public − opt-in = {times(ig['Impressions']['median'])} paid Instagram impressions ({ig['Impressions']['n']} posts). "
         f"Facebook: SocAPI extra plays = {times(fb['Video plays (starts)']['median'])} paid Facebook plays (20 posts). On the Instagram side, view-type metrics are 5–110× too small."),
        ("Subtracting paid views from total views does not give a reliable organic number for one post.",
         f"On paid Instagram posts, paid is a median {pct(R['paid_share_of_public']['median'])} of public views, so a small paid error becomes a large organic error. "
         f"Best case: {pct(best['median_abs_error'])} typical error, {pct(best['negative_organic'])} of posts negative."),
        ("What works: take the views just before the boost and grow them at the normal organic rate.",
         f"{pct(curve['median_abs_error'], 1)} typical error, {pct(curve['within_25pct'])} of posts within ±25%, never negative ({curve['posts']} posts vs opt-in). "
         f"10-post campaign totals: {pct(cl['pre-boost read x organic curve']['within_25pct'])} within ±25%."
         + (f" With no start date, the start is found from the jump in daily views: {pct(D['BS']['test']['organic_instagram']['median_abs_error'], 1)} typical error "
            f"({D['BS']['test']['organic_instagram']['posts']} posts)." if D.get("BS") else "")),
        ("The post-ID tag in the paid table is the best proof that a post is paid.",
         f"Our Meta ad-name rule finds {g['Instagram']['found_by_link_rule_all']} of {g['Instagram']['tagged']} tagged Instagram posts. "
         f"The TikTok Spark link finds {g['Tiktok']['found_by_link_rule_all']} of {g['Tiktok']['tagged']} tagged TikTok posts, and "
         f"{g['Tiktok']['found_by_link_rule_tracked']} of the {g['Tiktok']['tracked_in_bira']} that are tracked campaign posts."),
        ("Where no paid record exists, the model flags paid posts well.",
         f"Locked test: TikTok F1 {f2(tt['f1'])}, AUC {f2(tt['roc_auc'])}. Instagram F1 {f2(im['f1'])}, AUC {f2(im['roc_auc'])}, precision {f2(im['precision'])}."),
    ]
    if D["C2"] and D["TG"]:
        P2, T2 = D["C2"]["platforms"], D["TG"]["targets"]
        run = D["C2"].get("run")
        name = {"v22": "v2.2", "v21": "v2.1"}.get(run, "v2")
        met = sum(goal_met(T2, k) for k in GOALS)
        items[-1] = (f"Where no paid record exists, the model finds paid posts well: it meets {met} of {len(GOALS)} goals set before the test.",
                     f"Locked test, day-30 model {name}. " + " ".join(
                         f"{lab}: F1 {f3(P2[p]['test_metrics']['f1'])}; large boosts caught {pct(T2['C1'][p]['material_recall'], 1)}; "
                         f"flags really paid {pct(T2['C1'][p]['precision'], 1)}." for p, lab in PLATFORMS)
                     + (" Labels corrected for frozen opt-in." if run in ("v21", "v22") else ""))
    lis = "".join(f'<li><span class="n">{i + 1}</span><span>{tipped(t, tip)}</span></li>' for i, (t, tip) in enumerate(items))
    CV, M_ = (D.get("PM") or {}).get("coverage"), "missing_by_reason"
    nxt = [("Tag every boosted ad with the post ID.", "The tag gives an exact match on every platform."),
           ("Wait 7–14 days after publish before a boost.",
            f"Instagram, checked against opt-in: a pre-boost read on day 14+ gives {pct(R['production_function_by_confidence']['high']['median_abs_error'], 1)} typical error; "
            f"before day 7, {pct(R['production_function_by_confidence']['low']['median_abs_error'])}."),
           ("If a boost starts before day 3, use a dark post or get the creator to opt in.",
            (lambda c: f"Today {c['Instagram'][M_]['boosted_before_first_read'] + c['Instagram'][M_]['jump_before_day_3']:,} boosted Instagram posts and "
                       f"{c['Tiktok'][M_]['boosted_before_first_read'] + c['Tiktok'][M_]['jump_before_day_3']:,} boosted TikTok posts have no clean views before the boost.")(CV)
            if CV else
            f"Today {pct(V['totals']['Instagram']['not_separable'] / V['totals']['Instagram']['posts'])} of paid Instagram posts and "
            f"{pct(V['totals']['Tiktok']['not_separable'] / V['totals']['Tiktok']['posts'])} of paid TikTok posts cannot be separated."),
           (f"Check the {D['EQ']['full_platform']['with paid Facebook video plays']['posts'] if D.get('EQ') else 20} totals in the Instagram app.",
            "Shows whether the app counts Instagram only or Instagram + Facebook, so we know which total to quote. The shared list is in the Equation section.")]
    nx = "".join(f'<li><span class="chk" aria-hidden="true">→</span><span>{tipped(t, tip)}</span></li>' for t, tip in nxt)
    caveat = (f'<div class="caveat"><span class="pill warn">Caveat</span><span>TikTok organic after a boost is an estimate, not verified.'
              f'{info("TikTok opt-in views include Spark Ad views, so there is no organic truth after a boost. The method passes a back-test on unpaid TikTok posts only.")}</span></div>')
    return (f'<section id="findings" class="grid"><article class="card span2"><div class="card-head"><h3>What we found</h3></div>'
            f'<ol class="finds">{lis}</ol>{caveat}</article>'
            f'<article class="card"><div class="card-head"><h3>What would make it better</h3></div><ul class="finds next">{nx}</ul></article></section>')


GOALS = ("C1", "C2", "C3", "C4", "C5", "C6")


def goal_met(T, k):
    """a goal is met when it holds on every platform"""
    v = T[k]
    return bool(v["pass"]) if "pass" in v else all(v[p]["pass"] for p, _ in PLATFORMS if p in v)


def scorecard(D):
    """Model against the goals written before the test (docs/IMPROVEMENT_PLAN.md): summary + action on the left, scorecard on the right."""
    TG, C2 = D["TG"], D["C2"]
    T, P2 = TG["targets"], C2["platforms"]
    c1, c2, c3, c4, c5, c6 = (T[k] for k in GOALS)
    p1 = lambda x: pct(x, 1)
    rng = lambda ci, f=p1: f"{f(ci[0])}–{f(ci[1])}"
    v22 = C2.get("run") == "v22"
    t0, t1 = C2.get("test_cutoff", "2026-07-01"), C2.get("test_end", "2026-09-09")
    f0, f1 = (C2.get("fresh") or ["2026-09-10", "2026-09-24"])[:2]
    tm = {p: P2[p]["test_metrics"] for p, _ in PLATFORMS}

    def cell(val, tip, good, goal):
        return f'<td><div class="sc-v">{num(val, tip)}{status(good)}</div><div class="sc-goal">{goal}</div></td>'

    def row(name, help_, tip, cells):
        return (f'<tr><th scope="row"><span class="sc-name">{tipped(name, tip)}</span>'
                f'<span class="sc-help">{esc(help_)}</span></th>{"".join(cells)}</tr>')

    def grp(label, tip):
        return f'<tr class="grp"><th colspan="3" scope="colgroup">{esc(label)}{info(tip)}</th></tr>'

    # 1. locked test, day-30 model
    rows = [grp(f"Test posts, {day(t0)} – {day(t1)} · day-30 model",
                f"Locked test: {tm['Instagram']['n']:,} Instagram posts ({tm['Instagram']['n_pos']} paid) and {tm['Tiktok']['n']:,} TikTok posts "
                f"({tm['Tiktok']['n_pos']} paid), published {t0} to {t1}. No test post was used to train or tune the model. "
                + ("This is the fourth look at the test (plan amendment 5), so it cannot separate small differences." if v22 else ""))]
    cells = []
    for p, lab in PLATFORMS:
        v, n = c1[p], P2[p]["material"]["test_material_posts"]
        cells.append(cell(p1(v["material_recall"]),
                          f"{lab}: the model flags {round(v['material_recall'] * n)} of {n} large boosts ({p1(v['material_recall'])}). "
                          f"95% range {rng(v['ci95']['material_recall'])}.", v["material_recall"] >= 0.90, "goal 90%+"))
    rows.append(row("Large boosts caught", "Share of big paid pushes the model flags",
                    "Goal C1, part 1. Large boost: opt-in shows 40% or more of public views are paid, or an ad link or post-ID tag "
                    "with the boost inside the model window. Every TikTok boost counts as large.", cells))
    cells = []
    for p, lab in PLATFORMS:
        v, m = c1[p], tm[p]
        cells.append(cell(p1(v["precision"]),
                          f"{lab}: {m['tp']} of {m['tp'] + m['fp']} posts flagged as paid are paid ({p1(v['precision'])}). "
                          f"95% range {rng(v['ci95']['precision'])}.", v["precision"] >= 0.90, "goal 90%+"))
    rows.append(row("Flags that are really paid", "Of the posts flagged paid, the share that are",
                    "Goal C1, part 2 (precision). Both parts of C1 must reach 90%.", cells))
    cells = []
    for p, lab in PLATFORMS:
        v, m = c2[p], tm[p]
        short = f" Short of the goal by {v['goal'] - v['f1']:.3f}." if not v["pass"] else ""
        cells.append(cell(f3(v["f1"]),
                          f"{lab}: F1 {f3(v['f1'])} over all {m['n_pos']} paid test posts, large and small. It balances flags that are "
                          f"really paid ({p1(m['precision'])}) and paid posts caught ({p1(m['recall'])}). 95% range {rng(v['ci95'], f3)}.{short}",
                          v["pass"], f"goal {f2(v['goal'])}+"))
    rows.append(row("Overall score (F1)", "One number for all boosts: 1.000 is perfect",
                    "Goal C2. F1 combines the two numbers above, counting every paid post, not only large boosts. "
                    "Goals: Instagram 0.80 (v1 had 0.77), TikTok 0.93 (keep the v1 level).", cells))
    cells = []
    for p, lab in PLATFORMS:
        v = c3[p]
        cells.append(cell(f"{100 * v['ece']:.1f} pts",
                          f"{lab}: calibration error {f3(v['ece'])}. On average, a score is {100 * v['ece']:.1f} percentage points away "
                          "from the real share of paid posts at that score.", v["pass"], "goal 5 pts max"))
    rows.append(row("Scores match reality", "A score of 80% should mean about 80% are paid",
                    "Goal C3: calibration error 0.05 or less. Shown in percentage points (pts): 0.028 = 2.8 pts.", cells))

    # 2. day 14 vs day 30 on the same posts
    sp = c4["same_posts"]
    rows.append(grp("Same test posts · day-14 vs day-30 model",
                    f"Only the test posts that both models score: {sp['Instagram']['posts']} Instagram and {sp['Tiktok']['posts']} TikTok. "
                    "This is a smaller post set than the rows above, so the day-30 F1 here is not the same number."))
    cells = []
    for p, lab in PLATFORMS:
        v = sp[p]
        prov = " Day-14 “organic” calls on Instagram are marked provisional until the day-30 score." if p == "Instagram" and v["gap"] > 0.05 else ""
        cells.append(cell(f3(v["gap"]),
                          f"{lab}, {v['posts']} posts: F1 {f3(v['f1_day14'])} at day 14 and {f3(v['f1_day30'])} at day 30. Gap {f3(v['gap'])}.{prov}",
                          v["gap"] <= 0.05, "gap · goal 0.05 max"))
    rows.append(row("Day 14 nearly as good as day 30", "F1 lost by scoring 16 days earlier",
                    "Goal C4, part 2: the day-14 model's F1 is within 0.05 of the day-30 model on the same posts.", cells))

    # 3. fresh posts, day-14 model
    rows.append(grp(f"New posts, {day(f0)} – {day(f1)} · day-14 model · small sample",
                    "Posts no one had looked at: " + " and ".join(f"{c5[p]['n']} {lab} ({c5[p]['n_pos']} paid)" for p, lab in PLATFORMS if "n" in c5[p])
                    + ". Goal C5: the 90% line falls inside the 95% range. With so few paid posts the ranges are wide, so this check is weak."
                    + (" This is the third look at these posts." if v22 else "")))
    for key, name, help_, what in (("material_recall", "Large boosts caught", "Same measure as above, on new posts", "large boosts caught"),
                                   ("precision", "Flags that are really paid", "Same measure as above, on new posts", "flags that are really paid")):
        cells = []
        for p, lab in PLATFORMS:
            v = c5[p]
            if key not in v:
                cells.append(f'<td><div class="sc-v">{status(False)}</div><div class="sc-goal">no labels yet</div></td>')
                continue
            ci = v["ci95"][key]
            cells.append(cell(p1(v[key]), f"{lab}, posts published {f0} to {f1}: {what} {p1(v[key])}, 95% range {rng(ci)}, "
                                          f"from {v['n_pos']} paid posts. The goal is met when the range reaches 90%.",
                              ci[1] >= 0.90, f"range {pct(ci[0])[:-1]}–{pct(ci[1])}"))
        rows.append(row(name, help_, "Goal C5: the same C1 measures on posts published after the test window.", cells))

    # 4. both platforms
    cov = c4["coverage"]
    rows.append(grp("All posts", "Checks that cover both platforms together."))
    rows.append(row("Posts that get a score", "Posts 14+ days old, first read by day 7",
                    "Goal C4, part 1: 85% or more of these posts get a day-14 or day-30 score. Posts first read after day 7 are a "
                    "tracking gap, so no public-data model can score them early.",
                    [f'<td colspan="2"><div class="sc-v">' + num(p1(cov["all"]),
                     f"{p1(cov['all'])} of {c4['posts_in_base']:,} posts get a score (Instagram {p1(cov['Instagram'])}, TikTok {p1(cov['Tiktok'])}).")
                     + status(cov["all"] >= 0.85) + '</div><div class="sc-goal">goal 85%+</div></td>']))
    bp = c6["by_platform"]
    rows.append(row("Ad-tagged paid posts flagged", "Ads that carry the post ID: the surest proof of paid",
                    "Goal C6: every post whose ad carries its post ID, with the boost inside the model window, is flagged.",
                    [f'<td colspan="2"><div class="sc-v">' + num(f"{c6['posts_flagged_at_every_horizon']} of {c6['distinct_posts']}",
                     f"{bp['Instagram']['posts']} Instagram and {bp['Tiktok']['posts']} TikTok post(s). All {c6['scores']} scores "
                     "(one per model day that applies) are flagged.")
                     + status(c6["pass"]) + '</div><div class="sc-goal">goal: all of them</div></td>']))
    sc_table = (f'<table class="sct"><thead><tr><th scope="col"><span class="vh">Goal</span></th>'
                + "".join(f'<th scope="col">{lab}</th>' for _, lab in PLATFORMS) + f'</tr></thead><tbody>{"".join(rows)}</tbody></table>')

    # summary and what to do
    met = [k for k in GOALS if goal_met(T, k)]
    fails = {k: [lab for p, lab in PLATFORMS if p in T[k] and not T[k][p].get("pass", True)] for k in GOALS if k not in met}
    if "C4" in fails:
        fails["C4"] = [lab for p, lab in PLATFORMS if sp[p]["gap"] > 0.05] + (["both"] if cov["all"] < 0.85 else [])
    one_each = all(len(v) == 1 for v in fails.values())
    lead = ("All goals are met." if not fails else
            f"The other {len(fails)} {'are' if len(fails) > 1 else 'is'} missed on one platform{' each' if len(fails) > 1 else ''}."
            if one_each else f"{len(fails)} goals are missed.")
    acts = []
    mf = (D.get("MF") or {}).get("summary", {})
    for p, lab in PLATFORMS:
        if not c2[p]["pass"]:
            swing = (f" The {c2[p]['goal'] - c2[p]['f1']:.3f} gap is well inside the normal month-to-month range "
                     f"({f2(mf[p + '_h30']['f1_min'])}–{f2(mf[p + '_h30']['f1_max'])}).") if p + "_h30" in mf else ""
            acts.append(("no", f"{lab} overall score: {f3(c2[p]['f1'])}, goal {f2(c2[p]['goal'])}",
                         f"Use {lab} flags as they are.{swing}",
                         "Month-by-month check inside the training period (train on earlier posts, score the next month, Mar–Jun 2026). "
                         "More training data did not raise F1. Source: results/month_folds.json."))
    for p, lab in PLATFORMS:
        if sp[p]["gap"] > 0.05:
            acts.append(("no", f"{lab} day 14 vs day 30: gap {f3(sp[p]['gap'])}, goal 0.05",
                         f"Treat a day-14 “organic” call on {lab} as provisional. The daily table marks it, and the day-30 score replaces it.",
                         (f"Month-by-month, {lab} day-14 F1 ranged {f2(mf[p + '_h14']['f1_min'])}–{f2(mf[p + '_h14']['f1_max'])} and day-30 "
                          f"{f2(mf[p + '_h30']['f1_min'])}–{f2(mf[p + '_h30']['f1_max'])}. A post 14 days old has less of its history, "
                          "so the early model is weaker by design. Source: results/month_folds.json.") if p + "_h14" in mf else ""))
    if all("n_pos" in c5[p] for p, _ in PLATFORMS):
        acts.append(("note", "New-post check: met, but on few posts",
                     f"Only {c5['Instagram']['n_pos']} Instagram and {c5['Tiktok']['n_pos']} TikTok paid posts. The next clean test, on posts "
                     "published after Sep 24, can run from about Oct 21.",
                     "Plan amendment 6: score v2.2 and the v2.3 candidate once on the new posts; the one that meets more goals is used "
                     "(a tie keeps v2.2)."))
    summ = (f'<div class="sc-sum"><div class="sc-score"><span class="sc-big">{len(met)}</span>'
            f'<span class="sc-of">of {len(GOALS)} goals met<br>on both platforms</span></div><p class="sc-lead">{esc(lead)}</p>'
            + '<ul class="sc-acts">' + "".join(
                f'<li class="{c}"><span class="sc-what">{tipped(w, tip)}</span><span class="sc-act">{esc(a)}</span></li>'
                for c, w, a, tip in acts) + "</ul></div>")
    score = card("Did the model hit its goals?", f'<div class="sc">{summ}<div class="sc-wrap">{sc_table}</div></div>', cls="span3",
                 sub=("Model v2.2. " if v22 else "") + "Six goals, written down before testing. Hover or tap a number to see what it means.",
                 tip="Goals C1–C6 and their rules are in docs/IMPROVEMENT_PLAN.md. Labels are corrected for frozen opt-in counts.",
                 data=table(["Goal", "Platform", "Metric", "Value", "95% range", "Met"],
                            [[k, lab, ("large boosts caught" if "material_recall" in T[k][p] else "F1" if "f1" in T[k][p] else "calibration error"),
                              f3(T[k][p].get("material_recall", T[k][p].get("f1", T[k][p].get("ece", float("nan"))))),
                              "–".join(f3(x) for x in T[k][p]["ci95"]["material_recall"]) if isinstance(T[k][p].get("ci95"), dict) else
                              ("–".join(f3(x) for x in T[k][p]["ci95"]) if isinstance(T[k][p].get("ci95"), list) else ""),
                              "yes" if T[k][p]["pass"] else "no"]
                             for k in ("C1", "C2", "C3", "C5") for p, lab in PLATFORMS if p in T[k] and "pass" in T[k][p]]))
    return score


def targets(D):
    TG, C2 = D["TG"], D["C2"]
    if not (TG and C2):
        return ""
    T = TG["targets"]
    t0, t1 = C2.get("test_cutoff", "2026-07-01"), C2.get("test_end", "2026-09-09")
    vv = C2["v1_vs_v2"]
    prod = C2.get("run") in ("v21", "v22")
    if prod and D.get("TG0") and D.get("RL") and D.get("ST"):
        a0 = D["TG0"]["targets"]["C1"]["Instagram"]
        a1 = D["RL"]["models"]["Instagram_h30_test"]
        a2 = T["C1"]["Instagram"]
        st = D["ST"]
        stress = a1.get("stress_socapi_paid_added_back", {})
        sc = st["socapi_check_day30"]["locked_test"]
        v22 = C2.get("run") == "v22"
        bars = progress([("v2 · labels as first pulled", a0["material_recall"], "hatch-n",
                          f'First result: {pct(a0["material_recall"], 1)} of large boosts caught, {pct(a0["precision"], 1)} of flags really paid. Goal missed.'),
                         ("v2 · same model, labels fixed", a1["material_recall"], "hatch-n",
                          f'Same model, labels fixed: {pct(a1["material_recall"], 1)} caught, {pct(a1["precision"], 1)} of flags really paid.'),
                         *([("v2.1 · retrained on fixed labels", D["TG1"]["targets"]["C1"]["Instagram"]["material_recall"], "hatch-n",
                             f'Retrained on fixed labels: {pct(D["TG1"]["targets"]["C1"]["Instagram"]["material_recall"], 1)} caught, '
                             f'{pct(D["TG1"]["targets"]["C1"]["Instagram"]["precision"], 1)} of flags really paid.')] if v22 and D.get("TG1") else []),
                         (("v2.2 · in use, follower count fixed" if v22 else "v2.1 · in use, retrained on fixed labels"), a2["material_recall"], "solid",
                          f'In use: {pct(a2["material_recall"], 1)} caught (95% range {pct(a2["ci95"]["material_recall"][0])}–{pct(a2["ci95"]["material_recall"][1])}), '
                          f'{pct(a2["precision"], 1)} of flags really paid.'
                          + (" v2 and v2.1 used the follower count from the data pull, which a live score would not know yet. "
                             "v2.2 uses the count on the scoring day, so its result is the honest one." if v22 else ""))],
                       fmt=lambda x: pct(x, 1), goal=0.90)
        comp = card("Why Instagram improved: cleaner labels", bars
                    + f'<p class="mini">On {st["stale_at_latest_read"]:,} of {st["instagram_posts_2025_with_optin"]:,} Instagram posts with opt-in '
                      f'(published 2025 or later), the opt-in count stopped updating while public views kept growing. '
                      f'That gap looked like paid views, so some organic posts were labelled paid.</p>',
                    sub="Large boosts caught, Instagram day-30 model, same test posts. Line = 90% goal.",
                    tip=(f'Fix (plan amendment 4, chosen on training posts only): a paid label that appeared only after the opt-in count froze is removed '
                         f'(day 30: {st["label_changes_instagram"]["day_30"]["paid_to_no_label"]} posts). Independent check with SocAPI: paid plays on '
                         f'{sc["dropped"]["socapi_paid"]} of {sc["dropped"]["with_socapi"]} removed test posts, vs {sc["kept_paid"]["socapi_paid"]} of '
                         f'{sc["kept_paid"]["with_socapi"]} kept paid posts. '
                         + (f'If the removed posts that SocAPI shows as paid were counted as paid, v2 would catch {pct(stress["material_recall"], 1)}. ' if stress else "")
                         + ("v2.2 also uses the follower count known on the scoring day (amendment 5). " if v22 else "")),
                    data=table(["Platform", "Metric", "v1 (labels as pulled)", f'{"v2.2" if v22 else "v2.1"} (fixed labels)'],
                               [[lab, k.replace("_", " "), f3(vv[p][k]["v1"]), f3(vv[p][k]["v2"])] for p, lab in PLATFORMS
                                for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "ece")]))
    else:
        bars = progress([(f"{lab} {v}", vv[p]["f1"][v], "solid" if v == "v2" else "hatch-n", f"{lab}, model {v}: F1 {vv[p]['f1'][v]:.3f} on the locked test")
                         for p, lab in PLATFORMS for v in ("v1", "v2")], fmt=f3)
        comp = card("v1 → v2", bars, sub="F1 on the locked test",
                    tip="v2 uses more public reads (views and likes at many ages, the creator's own norms) and adds post-ID tags to the labels, so the label sets differ a little.",
                    data=table(["Platform", "Metric", "v1", "v2"],
                               [[lab, k.replace("_", " "), f3(vv[p][k]["v1"]), f3(vv[p][k]["v2"])] for p, lab in PLATFORMS for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "ece")]))
    hz, ig14 = [], None
    for blk, d in (("day14", 14), ("platforms", 30), ("day60", 60)):
        for p, lab in PLATFORMS:
            if p in C2.get(blk, {}):
                t = C2[blk][p]["test_metrics"]
                ig14 = t["recall"] if (p, d) == ("Instagram", 14) else ig14
                hz.append([f"Day {d}, {lab}", f'{t["n"]:,} ({t["n_pos"]:,})', pct(t["precision"]), pct(t["recall"]), f3(t["f1"])])
    note = ((f'Instagram at day 14 catches {pct(ig14)} of paid posts, so its “organic” calls stay provisional until day 30. ' if ig14 is not None else "")
            + "Day 60 sees boosts that start late; it was added after the goals were set, so it has no goal.")
    horiz = card("Score early, check later",
                 table(["Model", "Test posts (paid)", "Flags really paid", "Paid caught", "Overall (F1)"], hz,
                       tips=[None, "Test posts the model can score at that day; in brackets, those labelled paid.",
                             "Precision: of the posts flagged as paid, the share that are paid.",
                             "Recall: of the paid posts (large and small), the share the model flags.",
                             "F1: one number that balances the two columns before it. 1.000 is perfect."])
                 + f'<p class="mini">{esc(note)}</p>', cls="span2",
                 sub=f"{'Model v2.2. ' if C2.get('run') == 'v22' else ''}Each post gets the longest model its data allows. Test posts {day(t0)} – {day(t1)}.",
                 tip="Day 14 needs a public read on day 12–14 and a first read by day 10. Day 30 needs a read on day 28–30. "
                     "Day 60 needs a read on day 55 or later.")
    return f'<section id="targets" class="grid">{scorecard(D)}{comp}{horiz}</section>'


def model(D):
    C, W = D["C2"] or D["C"], D["W"]
    P = C["platforms"]
    vname = {"v22": "v2.2", "v21": "v2.1"}.get(C.get("run"), "v2" if D["C2"] else "v1")
    seg = ('<div class="seg-ctl" role="group" aria-label="Platform">' + "".join(
        f'<button type="button" data-pf-btn="{p}" aria-pressed="{"true" if p == DEFAULT_PF else "false"}">{lab}</button>' for p, lab in PLATFORMS) + "</div>")

    def tiles(p, lab):
        t, ci = P[p]["test_metrics"], P[p]["test_ci95"]
        spec = [("Ranking (ROC AUC)", "roc_auc", f3, "Chance that a random paid post scores higher than a random organic post. 0.5 = coin flip, 1.000 = perfect."),
                ("Paid ranking (PR AUC)", "pr_auc", f3, f"Average share of flags that are paid, over every catch rate. Fairer than ROC AUC when most posts "
                                                         f"are organic. A random guess scores {pct(t['prevalence'], 1)}, the share of paid test posts."),
                ("Overall (F1)", "f1", f3, "One number that balances the two tiles to the right. 1.000 = perfect."),
                ("Flags really paid", "precision", lambda x: pct(x, 1), f"Precision: {t['tp']} of {t['tp'] + t['fp']} posts flagged as paid are paid."),
                ("Paid posts caught", "recall", lambda x: pct(x, 1), f"Recall: the model flags {t['tp']} of {t['tp'] + t['fn']} paid posts."),
                ("Agreement (MCC)", "mcc", f3, "Agreement between flags and labels, counting all four outcomes in the matrix below. 0 = chance, 1.000 = perfect.")]
        return '<div class="tiles">' + "".join(
            f'<div class="tile"><span class="tl">{n}{info(tip)}</span><span class="tv">{fm(t[k])}</span>'
            f'<span class="tc">95% range {fm(ci[k][0])}–{fm(ci[k][1])}</span></div>' for n, k, fm, tip in spec) + "</div>"

    def cmx(p, lab):
        m = P[p]["test_metrics"]
        pos, neg = m["tp"] + m["fn"], m["fp"] + m["tn"]

        def cell(v, tot, kind, of, tech):
            q = {"paid, flagged": "cm-a", "organic, cleared": "cm-k"}.get(kind, "cm-l")
            return (f'<div class="cm {q}" tabindex="0" data-tip="{esc(f"{lab}: {v:,} of {tot:,} {of} ({pct(v / tot, 1)}). Also called a {tech}.")}">'
                    f'<b>{v:,}</b><span>{kind}</span></div>')
        return (f'<div class="cmx"><span></span><span class="cmh">Model says paid</span><span class="cmh">Model says organic</span>'
                f'<span class="cmr">Really paid ({pos})</span>{cell(m["tp"], pos, "paid, flagged", "paid posts", "true positive")}'
                f'{cell(m["fn"], pos, "paid, missed", "paid posts", "false negative")}'
                f'<span class="cmr">Really organic ({neg})</span>{cell(m["fp"], neg, "flagged by mistake", "organic posts", "false positive")}'
                f'{cell(m["tn"], neg, "organic, cleared", "organic posts", "true negative")}</div>')

    def roc(p, lab):
        c, ct = P[p]["curves_test"]["roc"], P[p]["curves_train_cv"]["roc"]
        return xy_chart([{"cls": "ln-n3", "pts": list(zip(ct["fpr"], ct["tpr"]))},
                         {"cls": "ln-acc", "area": True, "pts": list(zip(c["fpr"], c["tpr"])),
                          "tip": lambda x, y, i: f"{lab} test: false positive rate {pct(x, 1)}, true positive rate {pct(y, 1)}"}],
                        label=f"{lab} ROC curve", diag=True, xlabel="False positive rate", ylabel="True positive rate")

    def pr(p, lab):
        c, ct = P[p]["curves_test"]["pr"], P[p]["curves_train_cv"]["pr"]
        base = P[p]["test_metrics"]["prevalence"]
        return xy_chart([{"cls": "ln-n3", "pts": list(zip(ct["recall"], ct["precision"]))},
                         {"cls": "ln-acc", "area": True, "pts": list(zip(c["recall"], c["precision"])),
                          "tip": lambda x, y, i: f"{lab} test: recall {pct(x, 1)}, precision {pct(y, 1)}"},
                         {"cls": "ln-ref", "pts": [(0, base), (1, base)]}],
                        label=f"{lab} precision-recall curve", xlabel="Recall", ylabel="Precision")

    def cal(p, lab):
        c = P[p]["curves_test"]["calibration"]
        return xy_chart([{"cls": "ln-acc", "dots": True, "pts": [(r["mean_pred"], r["share_boosted"]) for r in c],
                          "tip": lambda x, y, i: f"{lab}: mean score {pct(x, 1)}, actually paid {pct(y, 1)} ({c[i]['n']} posts)"}],
                        label=f"{lab} calibration", diag=True, xlabel="Mean model score", ylabel="Share actually paid")

    def sweep(p, lab):
        s, thr = P[p]["threshold_sweep_test"], P[p]["threshold_from_train"]
        mk = lambda k: [(r["threshold"], r[k]) for r in s]
        tipf = lambda nm: (lambda x, y, i: f"{lab}, threshold {x:.2f}: {nm} {pct(y, 1)}")
        return xy_chart([{"cls": "ln-n5", "pts": mk("precision"), "tip": tipf("precision")},
                         {"cls": "ln-n3", "pts": mk("recall"), "tip": tipf("recall")},
                         {"cls": "ln-acc", "pts": mk("f1"), "tip": tipf("F1")}],
                        label=f"{lab} threshold sweep", vline=(thr, f"train pick {thr:.2f}"), xlabel="Threshold", ylabel="Test metric")

    def gains(p, lab):
        g = P[p]["curves_test"]["gains"]
        return xy_chart([{"cls": "ln-acc", "area": True, "pts": [(0, 0)] + [(r["top_share"], r["boosted_captured"]) for r in g],
                          "tip": lambda x, y, i: f"{lab}: check the top {pct(x)} of scores, find {pct(y, 1)} of paid posts"}],
                        label=f"{lab} cumulative gains", diag=True, xlabel="Share of posts checked, top score first", ylabel="Paid posts found")

    def baselines(p, lab):
        t, rr = P[p]["test_metrics"], P[p]["rules_test"]
        llm = [r for r in D["llm"] if r["platform"] == p and not r["detector"].startswith("ML")]
        ml = next(r for r in D["llm"] if r["platform"] == p and r["detector"].startswith("ML"))
        best = max(llm, key=lambda r: float(r["f1"]))
        same = f"the same {t['n']:,} test posts"
        a = progress([(f"This model ({vname})", t["f1"], "solid", f"{lab}: model {vname}, F1 {f3(t['f1'])} on {same} ({t['n_pos']} paid)."),
                      ("Hand rule: views > followers and engagement < 1%", rr[HAND_RULE]["f1"], "hatch-n",
                       f"{lab}: F1 {f3(rr[HAND_RULE]['f1'])} on {same}."),
                      ("Views > followers", rr["views > followers"]["f1"], "hatch-n", f"{lab}: F1 {f3(rr['views > followers']['f1'])} on {same}.")], fmt=f3)
        n = int(ml["n"])
        b = progress([("Older v1 model", float(ml["f1"]), "solid", f"{lab}: v1 model, F1 {f3(float(ml['f1']))} on {n} sample posts ({ml['n_pos']} paid)."),
                      (f"Best LLM ({best['detector'].replace('LLM ', '')})", float(best["f1"]), "hatch-n",
                       f"{lab}: F1 {f3(float(best['f1']))} on the same {n} posts, from the same public numbers, threshold 0.5.")], fmt=f3)
        return a + f'<p class="pb-h">{n}-post sample, older v1 model and labels</p>' + b

    names = {"late_share_d7": "Views gained after day 7", "front_load_d1": "Views already there on day 1", "log_views30": "Views at day 30",
             "log_accel_gap": "View jump minus like jump", "log_er30": "Engagement rate", "log_vtf30": "Views ÷ followers",
             "late_share_d14": "Views gained after day 14", "lpv_dilution": "Likes per view, early vs late", "log_accel": "Biggest daily view jump",
             "comments_per_like": "Comments per like", "log_followers": "Followers",
             "log_views": "Views at the horizon", "log_vtf": "Views ÷ followers", "er": "Engagement rate", "like_dilution": "Likes per view, early vs late",
             "creator_rel_reach": "Views vs the creator's earlier posts", "creator_rel_lpv": "Likes per view vs the creator's norm",
             "creator_rel_vtf": "Reach vs the creator's norm", "creator_rel_share_v3": "Day-3 share vs the creator's norm",
             "creator_prior_posts": "Creator's earlier posts", "jump_lpv_ratio": "Likes per new view in the biggest jump",
             "min_new_lpv_ratio": "Lowest likes per new view", "jump_lpv": "Likes per new view in the biggest jump", "jump_age": "Day of the biggest jump",
             "late_new_lpv_ratio": "Likes per new view after day 21", "jump_late_vs_early": "Late jump vs early jump",
             "shares_per_view": "Shares per view", "share_v25": "Share of views by day 25", "share_v28": "Share of views by day 28"}
    for a in (1, 2, 3, 5, 7, 10, 14, 21, 30, 45):
        names.setdefault(f"share_v{a}", f"Share of views by day {a}")
        names.setdefault(f"lpv{a}", f"Likes per view, day {a}")
    for k in ("r7", "r14", "r30", "r60"):
        names.setdefault(f"jump_{k}", f"Biggest daily jump, days {dict(r7='2–7', r14='8–14', r30='15–30', r60='31–60')[k]}")

    def inputs(p, lab):
        top = list(P[p]["permutation_importance_test_pr_auc"].items())[:5]
        mx = max(v for _, v in top)
        return progress([(names.get(k, k), max(v, 0) / mx, "solid" if i == 0 else "hatch-n", f"{lab}: PR AUC drops {v:.3f} when this input is shuffled")
                         for i, (k, v) in enumerate(top)])

    both = lambda fn: pf_panels(fn)
    sweep_rows = [[lab, f'{r["threshold"]:.2f}', pct(r["precision"], 1), pct(r["recall"], 1), f3(r["f1"])] for p, lab in PLATFORMS for r in P[p]["threshold_sweep_test"]]
    cards = [
        card("Confusion matrix", both(cmx), sub="Locked test, threshold chosen on train",
             tip="Test posts published 2026-07-01 to 2026-09-09, scored once. Rows = true label, columns = model flag.",
             data=table(["Platform", "TP", "FP", "FN", "TN"], [[lab] + [P[p]["test_metrics"][k] for k in ("tp", "fp", "fn", "tn")] for p, lab in PLATFORMS])),
        card("ROC curve", both(roc) + legend([("c-acc", "Test"), ("c-n3", "Train, cross-validated")]),
             sub="Test vs training (cross-validated): " + ", ".join(f'{lab} {f3(P[p]["test_metrics"]["roc_auc"])} vs {f3(P[p]["train_cv_metrics"]["roc_auc"])}' for p, lab in PLATFORMS),
             tip="Dashed line = random guess. Train curve uses 5-fold cross-validation grouped by creator.",
             data=table(["Platform", "Test AUC", "Train CV AUC", "Train in-sample AUC"],
                        [[lab, f3(P[p]["test_metrics"]["roc_auc"]), f3(P[p]["train_cv_metrics"]["roc_auc"]), f3(P[p]["train_in_sample_metrics"]["roc_auc"])] for p, lab in PLATFORMS])),
        card("Precision-recall", both(pr) + legend([("c-acc", "Test"), ("c-n3", "Train, cross-validated"), ("c-ref", "Random guess")]),
             sub="The fair view when most posts are organic", tip="Random guess = share of paid posts in the test set.",
             data=table(["Platform", "Test PR AUC", "Train CV PR AUC", "Share paid (test)"],
                        [[lab, f3(P[p]["test_metrics"]["pr_auc"]), f3(P[p]["train_cv_metrics"]["pr_auc"]), pct(P[p]["test_metrics"]["prevalence"], 1)] for p, lab in PLATFORMS])),
        card("Calibration", both(cal), sub="Does a score of 0.8 mean 80% paid?",
             tip=f"Dots on the dashed line = well calibrated. Expected calibration error: TikTok {f3(P['Tiktok']['test_metrics']['ece'])}, Instagram {f3(P['Instagram']['test_metrics']['ece'])}.",
             data=table(["Platform", "Mean score", "Share paid", "Posts"],
                        [[lab, pct(r["mean_pred"], 1), pct(r["share_boosted"], 1), r["n"]] for p, lab in PLATFORMS for r in P[p]["curves_test"]["calibration"]])),
        card("Score distribution", both(lambda p, lab: mirror_hist(P[p]["curves_test"]["histogram"], P[p]["threshold_from_train"], lab))
             + legend([("c-acc", "Paid (label)"), ("c-n3", "Organic (label)")]), sub="Paid and organic posts pull apart",
             tip="Each bar is the share of that label group in a score bin, so both groups use the same scale.",
             data=table(["Platform", "Score bin", "Paid", "Organic"],
                        [[lab, f'{h["edges"][i]:.2f}–{h["edges"][i + 1]:.2f}', h["boosted"][i], h["organic"][i]]
                         for p, lab in PLATFORMS for h in [P[p]["curves_test"]["histogram"]] for i in range(len(h["boosted"]))])),
        card("Threshold choice", both(sweep) + legend([("c-n5", "Precision"), ("c-n3", "Recall"), ("c-acc", "F1")]), sub="How results move with the cut-off",
             tip="The threshold was chosen on train. " + " ".join(
                 f'{lab}: the cut-off picked for 95% precision on train ({o["threshold"]:.2f}) gives {pct(o["test"]["precision"], 1)} precision at {pct(o["test"]["recall"], 1)} recall on test.'
                 for p, lab in PLATFORMS for o in P[p]["operating_points"] if o["rule"].startswith("precision >=")),
             data=table(["Platform", "Threshold", "Precision", "Recall", "F1"], sweep_rows)),
        card("Cumulative gains", both(gains), sub="Check the highest scores first",
             tip="Dashed line = random order.",
             data=table(["Platform", "Top share", "Paid found"], [[lab, pct(r["top_share"]), pct(r["boosted_captured"], 1)] for p, lab in PLATFORMS for r in P[p]["curves_test"]["gains"]])),
        card("Against rules and LLMs", both(baselines), sub="Overall score (F1). Each group is compared on the same posts.",
             tip="LLMs: GPT-5, Claude Sonnet 4.5 and Llama 3.1 70B in Snowflake Cortex, given the same public numbers, threshold 0.5, "
                 "150 test posts per platform. They were run once, next to the older v1 model, so they are compared with v1, not with the current model.",
             data=table(["Platform", "Detector", "AUC", "Precision", "Recall", "F1"],
                        [[("Instagram" if r["platform"] == "Instagram" else "TikTok"),
                          esc(r["detector"].replace("LLM ", "").replace("ML model (same posts)", "v1 model (same posts)")), f3(float(r["roc_auc"])),
                          pct(float(r["precision"]), 1), pct(float(r["recall"]), 1), f3(float(r["f1"]))] for r in D["llm"]])),
        card("What the model looks at", both(inputs), sub="Top 5 inputs, public data only. Top input = 100%.",
             tip="Importance = drop in test PR AUC when the input is shuffled. No paid date, ad, spend, opt-in or SocAPI field is an input."),
    ]
    if D["C2"]:
        mr = P["Instagram"]["material"]
        note = (f'<p class="note">Instagram large boosts (40%+ paid, or an ad link) caught: {pct(mr["test_material_recall"], 1)} of {mr["test_material_posts"]} test posts.'
                f'{info("Day-30 model, locked test. Large = opt-in shows 40% or more of public views are paid, or an ad link or post-ID tag with the boost inside the window.")}</p>')
    else:
        note = (f'<p class="note">Instagram misses are mostly small boosts: flagged posts hold {pct(W["Instagram"]["paid_view_weighted_recall"], 1)} of paid views on opt-in posts.'
                f'{info("Weighted by paid views measured with opt-in, 186 test posts.")}</p>')
    return (f'<section id="model" class="block"><div class="block-head"><div><h2>Paid-post model</h2>'
            f'<p class="card-sub">Day-30 model {vname}. Used only for posts with no paid record. Locked test: posts published '
            f'{day(C.get("test_cutoff", "2026-07-01"))} – {day(C.get("test_end", "2026-09-09"), True)}. Hover or tap the i for what each number means.</p></div>{seg}</div>'
            f'{pf_panels(tiles)}<div class="grid">{"".join(cards)}</div>{note}</section>')


def jump_faq(D):
    """FAQ: how the view jump finds a boost start, and why some boosted posts still have no organic number"""
    B, M = D.get("BS"), (D.get("PM") or {}).get("coverage")
    if not (B and M):
        return []
    t, p = B["test"], B["params"]
    o, bc, pd_, st = t["organic_instagram"], t["organic_instagram_by_confidence"], t["paid_instagram"], t["start_tiktok"]
    fa_i, fa_t = t["false_alarm_instagram"], t["false_alarm_tiktok"]
    rows = ([["Instagram organic vs creator data", o["posts"], pct(o["median_abs_error"], 1), pct(o["within_25pct"])]]
            + [[f"&nbsp;&nbsp;· pre-boost read on {nm}", bc[k]["posts"], pct(bc[k]["median_abs_error"], 1), pct(bc[k]["within_25pct"])]
               for k, nm in (("high (day 14+)", "day 14 or later"), ("medium (day 7-13)", "day 7–13"), ("low (day 3-6)", "day 3–6")) if k in bc]
            + [["Instagram paid (total − organic estimate)", pd_["posts"], pct(pd_["median_abs_error"], 1), pct(pd_["within_25pct"])],
               ["TikTok organic vs the ad-date estimate", t["tiktok_vs_known_date"]["posts"], pct(t["tiktok_vs_known_date"]["median_abs_error"], 1),
                pct(t["tiktok_vs_known_date"]["within_25pct"])]])
    q1 = ("How do we find the boost start when there is no ad date?",
          "<p>We read the public views every day from day 0 to day 90 and compare each day's growth with the organic curve. The boost start is the first step where "
          f"growth is far above normal: {p['t_early']:g}× by day 1, {p['t_mid']:g}× by day 3, {p['t_late']:g}× later, and large enough to matter. "
          "The read just before it is the pre-boost read. If the first jump comes before day 3, we do not guess.</p>"
          + table(["Test (held-out half of the posts)", "Posts", "Typical error", "Within ±25%"], rows)
          + f"<p>False alarms on unboosted posts: Instagram {fa_i['found']} of {fa_i['posts']} ({pct(fa_i['rate'])}), TikTok {fa_t['found']} of {fa_t['posts']} ({pct(fa_t['rate'])}). "
            f"TikTok posts with an ad date: the jump gives a clean start for {st['usable']} of {st['found']}, a late one for {st['bad']}, and {st['found_without_clean_read']} had no clean read at all.</p>")
    lab = [("boosted_before_first_read", "Boosted before our first read"), ("jump_before_day_3", "First jump before day 3"),
           ("no_clear_jump", "No clear jump in 90 days"), ("too_few_reads", "Fewer than 2 public reads")]
    q2 = ("Why do some boosted posts still have no organic number?",
          "<p>There are no clean, organic-only views before the boost. We do not fill these with a guess.</p>"
          + table(["Reason", "Instagram", "TikTok"],
                  [[n, f"{M['Instagram']['missing_by_reason'][k]:,}", f"{M['Tiktok']['missing_by_reason'][k]:,}"] for k, n in lab]
                  + [["<b>All missing</b>", f"<b>{M['Instagram']['organic_from']['missing']:,}</b>", f"<b>{M['Tiktok']['organic_from']['missing']:,}</b>"]])
          + '<p>What we tried for these posts is in <a href="#tried">Answers → Can we predict the posts we still miss?</a></p>')
    return [q1, q2]


def faq(D):
    R, T, C, V = D["R"], D["T"], D["C"], D["V"]
    g = C["tag_gold"]                      # link-rule check (independent of the model version)
    C = D["C2"] or C
    P = C["platforms"]
    rec = R["recipes"]
    a = R["post_id_tag"]
    pf_ = R["production_function_by_confidence"]
    bt = T["curve_backtest_unboosted_creator_split"]
    tv = T["tiktok_subtraction_vs_curve"]["subtraction_over_curve"]
    tiers = V["evidence_tiers_all_posts"]
    order = [("public - paid IG Impressions", "Public − paid impressions"),
             ("public - k x paid IG impressions (k fitted on other posts)", "Public − 1.07 × paid impressions"),
             ("public - paid IG Video plays (starts)", "Public − paid video plays"),
             ("public - paid IG 3-second video views", "Public − paid 3-second views"),
             ("public - paid IG ThruPlays", "Public − paid ThruPlays"),
             ("SocAPI total - FB cross-post - IG impressions - FB video plays (best mix)", "SocAPI total − paid"),
             ("pre-boost public read (no adjustment)", "Pre-boost read, no growth"),
             ("pre-boost read x flat growth (fitted on other posts)", "Pre-boost read × one factor"),
             ("pre-boost read x organic curve (no fitting on these posts)", "Pre-boost read × organic curve")]
    tier_lab = {"CONFIRMED_PAID_TAG": "Post-ID tag in the paid table", "CONFIRMED_AD_LINK": "Ad link (ad name, Spark item id)",
                "MEASURED_OPTIN_GAP": "Opt-in below 80% of public (opt-in still updating)", "MEASURED_SOCAPI_GAP": "SocAPI shows paid Facebook plays",
                "LOGGED_PAID_DATE_ONLY": "Manual paid date only", "NO_PAID_EVIDENCE": "No paid record (model scores it)"}

    def full(p, lab):
        a_, b, c, ci = P[p]["train_in_sample_metrics"], P[p]["train_cv_metrics"], P[p]["test_metrics"], P[p]["test_ci95"]
        keys = [("roc_auc", "ROC AUC"), ("pr_auc", "PR AUC"), ("f1", "F1"), ("f2", "F2"), ("precision", "Precision"), ("recall", "Recall"),
                ("specificity", "Specificity"), ("npv", "Negative predictive value"), ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                ("mcc", "MCC"), ("kappa", "Cohen's kappa"), ("ks", "KS"), ("brier", "Brier score"), ("log_loss", "Log loss"), ("ece", "Calibration error")]
        return table([f"{lab} metric", "Train in-sample", "Train CV", "Test", "Test 95% CI"],
                     [[n, f3(a_[k]), f3(b[k]), f"<b>{f3(c[k])}</b>", (f"{f3(ci[k][0])}–{f3(ci[k][1])}" if k in ci else "")] for k, n in keys])

    qa = [
        ("How did we test “total − paid = organic”?",
         f"<p>We matched {R['panel']['posts']} paid Instagram posts to their Meta ads and lined up, day by day, public views, opt-in views (organic only), SocAPI plays and every paid metric by placement. "
         f"{R['panel']['rows']:,} post-days, {R['panel']['obs_from']} to {R['panel']['obs_to']}.</p>"
         + table(["Method", "Posts", "Typical error", "Within ±25%", "Negative"],
                 [[n, rec[k]["posts"], pct(rec[k]["median_abs_error"]), pct(rec[k]["within_25pct"]), pct(rec[k]["negative_organic"])] for k, n in order],
                 hl=lambda r: r[0].startswith("Pre-boost read × organic"))),
        ("Why does subtraction fail?",
         f"<p>Organic is the small number left after you remove a large paid number. Paid is a median {pct(R['paid_share_of_public']['median'])} of public views. "
         f"On the {a['posts']} tagged posts while ads ran, {pct(a['public - paid IG impressions']['negative_organic'])} of days give negative organic.</p>"),
        ("How sure is the pre-boost method?",
         "<p>It depends on how late the pre-boost read is (paid Instagram posts, checked against opt-in).</p>"
         + table(["Pre-boost read", "Posts", "Typical error", "Within ±25%"],
                 [[n, pf_[k]["posts"], pct(pf_[k]["median_abs_error"], 1), pct(pf_[k]["within_25pct"])] for k, n in
                  [("high", "Day 14 or later"), ("medium", "Day 7–13"), ("low", "Before day 7")]])
         + "<p>Back-test on unpaid posts, creator split (predict day 30):</p>"
         + table(["Read", "Posts", "Typical error", "Within ±25%"],
                 [[f"{lab}, day {d}", f'{bt[p][f"read on day {d} -> predict day 30"]["posts"]:,}', pct(bt[p][f"read on day {d} -> predict day 30"]["median_abs_error"], 1),
                   pct(bt[p][f"read on day {d} -> predict day 30"]["within_25pct"])] for p, lab in [("Instagram", "Instagram"), ("Tiktok", "TikTok")] for d in (3, 7, 14)])),
    ] + jump_faq(D) + [
        ("What about TikTok and YouTube?",
         f"<p>TikTok public views rise {times(T['tiktok_metric_match']['Video plays (starts)']['median'])} the paid plays ({T['tiktok_posts']} posts). "
         f"But TikTok opt-in equals public (ratio {f3(T['tiktok_optin_equals_public']['median'])}), so there is no organic truth after a boost. "
         f"Subtraction gives {times(tv['median'])} the curve estimate, and we cannot check which is right.</p>"
         "<p>YouTube has no ad-to-video link in the warehouse, so it cannot be tested.</p>"),
        ("How do we know a post is paid?",
         "<p>Each post gets its strongest proof. The model only scores posts with none.</p>"
         + table(["Proof (strongest first)", "Instagram", "TikTok"],
                 [[tier_lab[k], f'{tiers["Instagram"].get(k, 0):,}', (f'{tiers["Tiktok"][k]:,}' if k in tiers["Tiktok"] else "n/a")] for k in tier_lab])
         + f"<p>Gold check: our Meta ad-name rule finds {g['Instagram']['found_by_link_rule_all']} of {g['Instagram']['tagged']} tagged Instagram posts. "
           f"The TikTok Spark link finds {g['Tiktok']['found_by_link_rule_all']} of {g['Tiktok']['tagged']} tagged TikTok posts, and "
           f"{g['Tiktok']['found_by_link_rule_tracked']} of the {g['Tiktok']['tracked_in_bira']} that are tracked campaign posts, so the tag goes first. "
           "Since v2 the labels count a post-ID tag as paid.</p>"),
        ("How was the model kept honest?",
         ("<ul><li>No paid data in the inputs: only public views, likes, comments, shares and followers up to the model day (14, 30 or 60), "
          "and the same creator's earlier posts.</li>"
          "<li>Targets written down before any test (docs/IMPROVEMENT_PLAN.md). Every change was chosen on train only and logged there first.</li>"
          f"<li>Locked test (posts {C['test_cutoff']} to {C.get('test_end', '2026-09-09')}): no model choice was made on it. "
          + ("v2.2 is the fourth look at it and the third at the fresh posts "
             if C.get("run") == "v22" else "v2 is the second look at it, and the fresh posts ")
          + f"({C['fresh'][0]} to {C['fresh'][1]}) were scored once per version. Posts published after 2026-09-24 are the next clean test.</li>" if D["C2"] else
          "<ul><li>No paid data in the inputs: only public views, likes, comments, shares and followers from days 0–30.</li>"
          f"<li>Locked test: trained on posts before {C['test_cutoff']}, tested once on later posts. Re-running the code gives the same scores.</li>")
         + "<li>Model and threshold chosen on train only, with cross-validation grouped by creator.</li>"
         f"<li>95% intervals resample whole creators ({C['n_boot']:,} draws).</li>"
         "<li>Labels miss some paid posts, so test precision is a floor.</li></ul>"),
        ("All model metrics", "".join(full(p, lab) for p, lab in PLATFORMS)),
        ("Thresholds for a stricter or looser flag",
         table(["Platform", "Rule (chosen on train)", "Threshold", "Precision", "Recall", "F1"],
               [[lab, esc(o["rule"]), f'{o["threshold"]:.2f}', pct(o["test"]["precision"], 1), pct(o["test"]["recall"], 1), f3(o["test"]["f1"])]
                for p, lab in PLATFORMS for o in P[p]["operating_points"]])),
        ("Slices and model choice",
         table(["Platform", "Slice", "Posts (paid)", "AUC", "F1"],
               [[lab, esc(k), f'{v["n"]} ({v["n_pos"]})', f3(v["roc_auc"]), f3(v["f1"])] for p, lab in PLATFORMS for k, v in P[p]["slices_test"].items()]
               + [[lab, "Client holdout (train CV)", f'{P[p]["train"]["n"]:,}', f3(P[p]["overfit_check"]["train_cv_by_client_roc_auc"]), ""] for p, lab in PLATFORMS])
         + ("<p>Candidates from the model search (train cross-validation, before the follower fix). v2.2 refits the highlighted pick.</p>"
            if C.get("run") == "v22" else "")
         + table(["Platform", "Candidate (train CV)", "AUC", "PR AUC"],
                 [[lab, esc(r["model"].replace("_", " ")), f3(r["cv_roc_auc"]), f3(r["cv_pr_auc"])] for p, lab in PLATFORMS for r in P[p]["selection_table"]],
                 hl=lambda r: any(r[1] == f'{P[p]["selected_model"].split(" + isotonic")[0]} [{P[p].get("feature_set", "")}]'.replace("_", " ")
                                  or r[1] == P[p]["selected_model"].replace("_", " ") for p, _ in PLATFORMS))),
        ("What does the daily table do?",
         "<p>One row per post: proof tier, paid status, model score (only with no proof), organic views with method and confidence, and, for boosted posts with no "
         "ad date, the boost start found from the view jump (BOOST_START_DETECTED_DATE, PREBOOST_SOURCE). "
         + (f"The latest test file covers {sum(D['PM']['coverage'][p]['posts'] for p in ('Instagram', 'Tiktok')):,} posts (run 2026-10-08). " if D.get("PM") else "")
         + "The pipeline and its tests are in the repo. It is not scheduled yet: it needs a Snowflake service account and a schema to write to.</p>"),
        ("Glossary: sources and terms",
         '<dl class="gloss">' + "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in GLOSSARY) + "</dl>"
         + "<p>Organic truth: BIRA observation time series (public = Nimble, opt-in = private API). Paid: unified paid table and EDW Meta and TikTok ad tables. "
         "Data read 2026-10-07. Middle half = 25th–75th percentile. Error = (estimate − opt-in) ÷ opt-in.</p>"),
    ]
    items = "".join(f'<details class="qa"><summary>{esc(q)}<span class="plus" aria-hidden="true"></span></summary><div class="qa-a">{a_}</div></details>' for q, a_ in qa)
    return f'<section id="faq" class="block"><div class="block-head"><h2>FAQ</h2></div><div class="faq">{items}</div></section>'


CSS = """
:root {
  --backdrop: #e4e5e8; --shell: #f4f4f5; --card: #fcfcfc; --raise: #ffffff;
  --ink: #111214; --ink-2: #3f4147; --muted: #686b72; --faint: #8b8e95; --rule: rgba(17,18,20,.08); --rule-2: rgba(17,18,20,.05);
  --shadow: 0 1px 1px rgba(17,18,20,.03), 0 12px 32px -14px rgba(17,18,20,.14);
  --accent: #2b5cf2; --accent-deep: #1a3dd8; --accent-soft: #e8eefe; --accent-ink: #1f48d6;
  --n1: #eceef1; --n2: #d5d7dc; --n3: #8f929a; --n4: #6d7078; --n5: #2c2e33;
  --inv: #141518; --inv-ink: #ffffff; --inv-2: rgba(255,255,255,.72);
  --glass: rgba(255,255,255,.8); --glass-line: rgba(17,18,20,.08); --warn: #8a5300; --warn-bg: #fbf3e4;
  --nav-on: #17181b; --nav-on-ink: #ffffff;
  --c-o: #0f9e78; --c-i: #2b5cf2; --c-f: #7646e8; --c-o-ink: #067556; --c-i-ink: #1f48d6; --c-f-ink: #5f2fd6; --on-c: #ffffff;
  --font: "Geist", system-ui, -apple-system, "SF Pro Display", "Segoe UI", Roboto, sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --backdrop: #050506; --shell: #0f1013; --card: #17181c; --raise: #1d1e23;
    --ink: #f2f3f5; --ink-2: #c3c5cc; --muted: #9497a0; --faint: #80838c; --rule: rgba(255,255,255,.09); --rule-2: rgba(255,255,255,.05);
    --shadow: 0 1px 1px rgba(0,0,0,.4), 0 14px 34px -16px rgba(0,0,0,.7);
    --accent: #5f8bff; --accent-deep: #3a66f0; --accent-soft: #1d2747; --accent-ink: #a9c0ff;
    --n1: #24262c; --n2: #383b42; --n3: #6b6f79; --n4: #9a9ea7; --n5: #e3e4e8;
    --inv: #26272d; --inv-ink: #ffffff; --inv-2: rgba(255,255,255,.7);
    --glass: rgba(29,30,35,.84); --glass-line: rgba(255,255,255,.1); --warn: #f2b766; --warn-bg: #2a2216;
    --nav-on: #f2f3f5; --nav-on-ink: #111214; color-scheme: dark;
    --c-o: #34d399; --c-i: #6d93ff; --c-f: #a78bfa; --c-o-ink: #6ee7b7; --c-i-ink: #a9c0ff; --c-f-ink: #c9b6ff; --on-c: #0d0e12;
  }
}
:root[data-theme="dark"] {
  --backdrop: #050506; --shell: #0f1013; --card: #17181c; --raise: #1d1e23;
  --ink: #f2f3f5; --ink-2: #c3c5cc; --muted: #9497a0; --faint: #80838c; --rule: rgba(255,255,255,.09); --rule-2: rgba(255,255,255,.05);
  --shadow: 0 1px 1px rgba(0,0,0,.4), 0 14px 34px -16px rgba(0,0,0,.7);
  --accent: #5f8bff; --accent-deep: #3a66f0; --accent-soft: #1d2747; --accent-ink: #a9c0ff;
  --n1: #24262c; --n2: #383b42; --n3: #6b6f79; --n4: #9a9ea7; --n5: #e3e4e8;
  --inv: #26272d; --inv-ink: #ffffff; --inv-2: rgba(255,255,255,.7);
  --glass: rgba(29,30,35,.84); --glass-line: rgba(255,255,255,.1); --warn: #f2b766; --warn-bg: #2a2216;
  --nav-on: #f2f3f5; --nav-on-ink: #111214; color-scheme: dark;
    --c-o: #34d399; --c-i: #6d93ff; --c-f: #a78bfa; --c-o-ink: #6ee7b7; --c-i-ink: #a9c0ff; --c-f-ink: #c9b6ff; --on-c: #0d0e12;
}
* { box-sizing: border-box; }
body { background: var(--backdrop); color: var(--ink); font-family: var(--font); font-size: 15px; line-height: 1.5; -webkit-font-smoothing: antialiased; }
.shell { max-width: 1280px; margin: 0 auto; background: var(--shell); border-radius: 40px; padding-inline: 40px; padding-block: 28px 48px; display: grid; grid-template-columns: minmax(0, 1fr); gap: 28px; }
.page-wrap { padding-inline: 16px; padding-block: 16px; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 8px; }
h1, h2, h3 { margin: 0; font-weight: 500; letter-spacing: -.02em; text-wrap: balance; }
p { margin: 0; }
a { color: var(--accent-ink); }
/* top bar */
.top { display: flex; align-items: center; justify-content: space-between; gap: 16px 20px; flex-wrap: wrap; }
.top-right { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; min-width: 0; max-width: 100%; }
.brand { display: flex; align-items: center; gap: 12px; font-size: 26px; font-weight: 500; letter-spacing: -.03em; }
.mark { width: 34px; height: 34px; border-radius: 9px; background: var(--nav-on); display: grid; place-items: center; box-shadow: 0 6px 14px -8px rgba(0,0,0,.5); }
.mark svg { width: 20px; height: 20px; } .mk { fill: var(--nav-on-ink); opacity: .55; } .mk.a { fill: var(--accent); opacity: 1; }
nav.pills { display: flex; gap: 4px; overflow-x: auto; max-width: 100%; scrollbar-width: none; }
nav.pills a { color: var(--ink); text-decoration: none; font-size: 15px; padding: 10px 18px; border-radius: 14px; white-space: nowrap; }
nav.pills a:hover { background: var(--rule-2); }
nav.pills a.on { background: var(--nav-on); color: var(--nav-on-ink); box-shadow: 0 6px 14px -8px rgba(0,0,0,.6), inset 0 1px 0 rgba(255,255,255,.12); }
/* title row */
.title-row { display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; flex-wrap: wrap; }
h1 { font-size: clamp(40px, 7vw, 76px); line-height: 1; letter-spacing: -.045em; }
.dates { display: flex; align-items: center; background: var(--card); border: 1px solid var(--rule); border-radius: 16px; overflow: hidden; box-shadow: var(--shadow); flex-wrap: wrap; }
.dates span { display: inline-flex; align-items: center; gap: 8px; padding: 10px 16px; font-size: 14px; white-space: nowrap; }
.dates .vs { color: var(--muted); background: var(--rule-2); }
.dates svg, .seg-ctl svg { width: 16px; height: 16px; stroke: currentColor; fill: none; flex: none; }
/* cards + grid */
.grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 20px; }
.span2 { grid-column: span 2; } .span3 { grid-column: 1 / -1; }
.card { position: relative; background: var(--card); border: 1px solid var(--rule-2); border-radius: 28px; box-shadow: var(--shadow); padding: 26px 26px 24px; display: flex; flex-direction: column; gap: 16px; min-width: 0; }
.card-head { display: grid; gap: 4px; padding-right: 52px; }
.card-head h3 { font-size: 21px; }
.card-sub { color: var(--muted); font-size: 13.5px; }
details.more > summary { position: absolute; top: 20px; right: 20px; width: 44px; height: 44px; border-radius: 50%; border: 1px solid var(--rule); background: var(--raise);
  display: grid; place-items: center; list-style: none; cursor: pointer; color: var(--ink); font-size: 18px; letter-spacing: 1px; line-height: 1; }
details.more > summary::-webkit-details-marker { display: none; }
details.more > summary:hover { background: var(--n1); }
details.more[open] > summary { background: var(--nav-on); color: var(--nav-on-ink); }
.more-body { display: grid; gap: 10px; }
.info { position: relative; width: 18px; height: 18px; border-radius: 50%; border: 1px solid var(--rule); background: var(--raise); color: var(--muted); font: 600 11px/1 var(--font);
  display: inline-grid; place-items: center; cursor: help; padding: 0; margin-left: 6px; vertical-align: middle; flex: none; }
.info::after { content: ""; position: absolute; inset: -6px; }
.info:hover { color: var(--ink); border-color: var(--n3); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 12.5px; color: var(--ink-2); }
.key { display: inline-flex; align-items: center; gap: 6px; }
.sw { width: 10px; height: 10px; border-radius: 3px; display: inline-block; }
.c-acc { background: var(--accent); } .c-n2 { background: var(--n2); } .c-n3 { background: var(--n3); } .c-n4 { background: var(--n4); } .c-n5 { background: var(--n5); }
.c-ref { background: none; border-top: 2px dashed var(--n3); height: 0; border-radius: 0; width: 14px; }
/* fills: neutral hatch by default, accent only for emphasis */
.hatch-n, .cbar.hatch { background: repeating-linear-gradient(135deg, var(--n3) 0 1.5px, transparent 1.5px 6px), var(--n1); }
.solid, .cbar.solid { background: linear-gradient(180deg, var(--accent-deep), var(--accent) 60%, color-mix(in srgb, var(--accent) 78%, #ffffff)); box-shadow: inset 0 1px 0 rgba(255,255,255,.25); }
/* hero columns */
.cols { display: grid; gap: 0; }
.cols-row { display: grid; grid-template-columns: 46px repeat(5, minmax(0, 1fr)); }
.ch, .cc { border-left: 1px solid var(--rule); padding-inline: 12px; }
.ch { display: grid; gap: 2px; padding-bottom: 14px; align-content: start; }
.cl { font-size: 12.5px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.cv { font-size: clamp(22px, 2.6vw, 34px); color: var(--faint); letter-spacing: -.03em; line-height: 1.1; }
.cn { font-size: 12px; color: var(--muted); }
.ch.on .cl, .ch.on .cv { color: var(--ink); }
.ch.on, .cc.on { background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 9%, transparent), color-mix(in srgb, var(--accent) 2%, transparent)); }
.cols-row.plot { position: relative; height: 250px; }
.cc { position: relative; display: flex; align-items: flex-end; padding-inline: 0; }
.cbar { width: 100%; border-radius: 4px 4px 0 0; }
.cbar.solid { border-radius: 6px 6px 0 0; }
.cy { position: relative; }
.cy-t { position: absolute; left: 0; transform: translateY(50%); font-size: 12px; color: var(--muted); }
.gl { position: absolute; left: 46px; right: 0; height: 1px; background: var(--rule-2); }
.pill { display: inline-flex; align-items: center; gap: 8px; background: var(--glass); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
  border: 1px solid var(--glass-line); border-radius: 999px; padding: 7px 14px; font-size: 13.5px; color: var(--ink-2); box-shadow: 0 8px 20px -10px rgba(17,18,20,.25); white-space: nowrap; }
.pill b { color: var(--ink); font-weight: 600; }
.pill i { width: 1px; height: 14px; background: var(--rule); }
.pill.float { position: absolute; right: calc(20% - 40px); top: 38%; transform: translateX(-20%); }
.pill.pin { position: absolute; transform: translate(-6%, 0); padding: 4px 10px; font-size: 12px; }
.ask { margin-top: 2px; border-radius: 20px; padding: 14px 18px 16px; display: grid; gap: 10px; background: var(--n1); }
.ask-q { display: flex; align-items: center; gap: 10px; font-weight: 500; color: var(--ink-2); }
.spark { color: var(--accent); font-size: 18px; }
.ask-a { background: var(--raise); border-radius: 14px; padding: 12px 16px; box-shadow: 0 4px 16px -8px rgba(17,18,20,.18); }
.chip { display: inline-block; background: var(--accent-soft); color: var(--accent-ink); border-radius: 8px; padding: 1px 8px; font-weight: 500; }
/* stat + progress */
.big { font-size: clamp(56px, 7vw, 84px); line-height: .95; letter-spacing: -.05em; font-weight: 500; }
.big.sm { font-size: clamp(44px, 5vw, 60px); }
.big-sub { color: var(--muted); font-size: 13.5px; margin-top: -6px; }
/* scorecard: summary + actions left, goals table right */
.sc { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 2fr); gap: 32px; align-items: start; }
.sc-sum { display: grid; gap: 14px; }
.sc-score { display: flex; align-items: center; gap: 14px; }
.sc-big { font-size: 76px; line-height: .9; letter-spacing: -.05em; font-weight: 500; }
.sc-of { font-size: 15px; color: var(--ink-2); line-height: 1.35; }
.sc-lead { color: var(--ink-2); font-size: 14.5px; }
.sc-acts { list-style: none; margin: 0; padding: 0; display: grid; gap: 10px; }
.sc-acts li { display: grid; gap: 4px; padding: 12px 14px; border-radius: 16px; background: var(--warn-bg); font-size: 14px; line-height: 1.45; }
.sc-acts li.note { background: var(--n1); }
.sc-what { font-weight: 500; color: var(--ink); }
.sc-act { color: var(--ink-2); }
table.sct { font-size: 14px; table-layout: fixed; }
table.sct th, table.sct td { padding: 11px 12px; border-bottom: 1px solid var(--rule-2); vertical-align: top; text-align: left; }
table.sct thead th { font-size: 13px; color: var(--ink); font-weight: 500; padding-top: 0; border-bottom: 1px solid var(--rule); }
table.sct thead th:first-child { width: 44%; }
table.sct tbody th { font-weight: 400; }
table.sct tr.grp th { padding: 18px 12px 8px; font-size: 12px; color: var(--muted); font-weight: 500; border-bottom: 1px solid var(--rule); }
table.sct tr.grp:first-child th { padding-top: 6px; }
table.sct tr:last-child th, table.sct tr:last-child td { border-bottom: 0; }
.sc-name { display: block; color: var(--ink); font-weight: 500; }
.sc-help { display: block; color: var(--muted); font-size: 12.5px; margin-top: 2px; line-height: 1.4; }
.sc-v { display: flex; align-items: center; gap: 6px 8px; flex-wrap: wrap; }
.sc-v .num { font-size: 18px; font-weight: 500; letter-spacing: -.01em; }
.sc-goal { color: var(--muted); font-size: 12px; margin-top: 3px; }
.num { cursor: help; font-variant-numeric: tabular-nums; text-decoration: underline dotted var(--n3); text-decoration-thickness: 1px; text-underline-offset: 5px; }
.st { display: inline-flex; align-items: center; gap: 5px; font-size: 11.5px; font-weight: 500; padding: 2px 8px 2px 7px; border-radius: 999px; line-height: 1.4; white-space: nowrap; }
.st::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.st.ok { background: var(--accent-soft); color: var(--accent-ink); }
.st.no { background: var(--warn-bg); color: var(--warn); box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--warn) 30%, transparent); }
.nw { white-space: nowrap; }
.vh { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.mini { font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
.pbars { display: grid; gap: 14px; }
.pb { display: grid; gap: 6px; }
.pb-top { display: flex; justify-content: space-between; gap: 10px; font-size: 14px; color: var(--ink-2); }
.pb-top b { color: var(--ink); font-weight: 500; font-variant-numeric: tabular-nums; }
.pb-h { font-size: 12px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 12px; margin-top: 2px; }
.pb-track { position: relative; height: 14px; border-radius: 999px; background: var(--rule-2); overflow: hidden; }
.pb-goal { position: absolute; top: 0; bottom: 0; width: 2px; margin-left: -1px; background: var(--ink); opacity: .55; }
.pb-fill { height: 100%; border-radius: 999px; min-width: 6px; }
/* dot matrix: accent = measured, neutral steps = estimate confidence, ring = not separable */
.dm { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 12px; align-items: center; }
.dm-lab { font-size: 14px; font-weight: 500; display: grid; } .dm-lab small { color: var(--muted); font-weight: 400; font-size: 12px; }
.dm-grid { display: grid; grid-template-columns: repeat(20, minmax(0, 1fr)); gap: 3px; max-width: 260px; }
.dt { aspect-ratio: 1; border-radius: 50%; display: block; }
.m { background: var(--accent); } .mf { background: color-mix(in srgb, var(--accent) 38%, var(--card)); } .h { background: var(--n5); } .md { background: var(--n4); } .l { background: var(--n2); }
.x { background: transparent; box-shadow: inset 0 0 0 1.5px var(--n3); }
.sw.m, .sw.mf, .sw.h, .sw.md, .sw.l, .sw.x { border-radius: 50%; }
/* inverse card */
.inv { justify-content: space-between; min-height: 300px; background: var(--inv); color: var(--inv-ink); border: 0; overflow: hidden; isolation: isolate; }
.inv::after { content: ""; position: absolute; inset: 0; z-index: -1; opacity: .18; mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.inv::before { content: ""; position: absolute; right: -60px; bottom: -60px; width: 220px; height: 220px; border-radius: 50%; z-index: -1;
  background: radial-gradient(circle, color-mix(in srgb, var(--accent) 45%, transparent), transparent 70%); }
.pill.glass { color: var(--inv-ink); background: rgba(255,255,255,.1); border-color: rgba(255,255,255,.2); align-self: flex-start; box-shadow: none; }
.inv-num { font-size: clamp(72px, 9vw, 112px); line-height: .9; letter-spacing: -.05em; font-weight: 500; }
.inv-sub { font-size: 14px; color: var(--inv-2); max-width: 30ch; }
.inv-foot { display: block; margin-top: 8px; font-size: 12.5px; opacity: .8; }
/* dot-range: dark = Instagram side, light = Facebook side, accent = closest to 1x */
.dr { display: grid; gap: 2px; }
.dr-row { display: grid; grid-template-columns: minmax(0, 150px) minmax(0, 1fr); gap: 14px; align-items: center; }
.dr-l { font-size: 13.5px; color: var(--ink-2); } .dr-l.best { color: var(--ink); font-weight: 600; }
.dr-t { position: relative; height: 30px; }
.ax .dr-t { height: 18px; }
.dr-g { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--rule-2); }
.dr-one { position: absolute; top: -2px; bottom: -2px; border-left: 1.5px dashed var(--n3); }
.dr-x { position: absolute; transform: translateX(-50%); font-size: 12px; color: var(--muted); }
.dr-w { position: absolute; height: 4px; border-radius: 2px; opacity: .35; }
.dr-w.t { top: 7px; } .dr-w.b { top: 19px; }
.dr-d { position: absolute; width: 11px; height: 11px; border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--card); }
.dr-d::after { content: ""; position: absolute; inset: -7px; }
.dr-d.t { top: 9px; } .dr-d.b { top: 21px; }
.k-ig { background: var(--n5); } .k-fb { background: var(--n3); }
.dr-d.best { background: var(--accent); width: 13px; height: 13px; } .dr-w.best { background: var(--accent); opacity: .45; }
/* findings */
.finds { list-style: none; margin: 0; padding: 0; display: grid; gap: 12px; }
.finds li { display: grid; grid-template-columns: 28px minmax(0, 1fr); gap: 12px; align-items: start; font-size: 16px; }
.finds .n { width: 26px; height: 26px; border-radius: 50%; background: var(--n1); color: var(--ink); display: grid; place-items: center; font-size: 13px; font-weight: 500; }
.finds li:nth-child(3) .n { background: var(--accent); color: #ffffff; }
.finds .chk { color: var(--muted); font-size: 16px; text-align: center; }
.caveat { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; background: var(--warn-bg); border-radius: 16px; padding: 12px 14px; font-size: 14.5px; }
.pill.ok { background: transparent; border-color: color-mix(in srgb, var(--accent) 40%, transparent); color: var(--accent-ink); box-shadow: none; padding: 1px 8px; font-weight: 500; font-size: 12px; }
td .pill.warn { padding: 1px 8px; font-size: 12px; }
.pill.warn { background: transparent; border-color: color-mix(in srgb, var(--warn) 40%, transparent); color: var(--warn); box-shadow: none; padding: 3px 10px; font-weight: 500; }
/* model block */
.block { display: grid; gap: 18px; }
.block-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
h2 { font-size: clamp(30px, 4vw, 44px); letter-spacing: -.035em; }
.seg-ctl { display: inline-flex; background: var(--card); border: 1px solid var(--rule); border-radius: 14px; padding: 4px; box-shadow: var(--shadow); }
.seg-ctl button { font: inherit; font-size: 14px; border: 0; background: none; color: var(--ink-2); padding: 8px 16px; border-radius: 10px; cursor: pointer; display: inline-flex; align-items: center; gap: 6px; min-height: 36px; }
.seg-ctl button:hover { color: var(--ink); }
.seg-ctl button[aria-pressed="true"] { background: var(--nav-on); color: var(--nav-on-ink); }
.tiles { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 12px; }
.tile { background: var(--card); border: 1px solid var(--rule-2); border-radius: 20px; box-shadow: var(--shadow); padding: 14px 16px; display: grid; gap: 2px; }
.tl { font-size: 12.5px; color: var(--muted); }
.tv { font-size: 34px; letter-spacing: -.04em; line-height: 1.1; }
.tc { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.cmx { display: grid; grid-template-columns: auto minmax(0, 1fr) minmax(0, 1fr); gap: 6px; align-items: stretch; }
.cmh { font-size: 12px; color: var(--muted); text-align: center; align-self: end; }
.cmr { font-size: 12px; color: var(--muted); align-self: center; max-width: 9em; }
.cm { border-radius: 16px; padding: 16px 8px; text-align: center; display: grid; gap: 2px; }
.cm b { font-size: 30px; font-weight: 500; letter-spacing: -.03em; line-height: 1.1; }
.cm span { font-size: 12px; opacity: .85; }
.cm-a { background: linear-gradient(160deg, var(--accent-deep), var(--accent)); color: #ffffff; }
.cm-k { background: var(--n5); color: var(--card); }
.cm-l { background: var(--n1); color: var(--ink); }
.note { color: var(--muted); font-size: 13.5px; display: flex; align-items: center; }
/* svg */
svg.defs { position: absolute; width: 0; height: 0; overflow: hidden; }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
.plot-wrap { position: relative; }
svg .grid { stroke: var(--rule); stroke-width: 1; }
svg .axis { stroke: var(--n3); stroke-width: 1; }
svg .ref, svg .ln.ln-ref { stroke: var(--n3); stroke-width: 1.5; stroke-dasharray: 4 4; fill: none; }
svg .tick { fill: var(--muted); font-size: 11px; font-family: var(--font); }
svg .axlab { fill: var(--ink-2); font-size: 11.5px; font-family: var(--font); }
svg .ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg .ln-acc { stroke: var(--accent); stroke-width: 2.5; } svg .ln-n3 { stroke: var(--n3); } svg .ln-n4 { stroke: var(--n4); } svg .ln-n5 { stroke: var(--n5); }
svg .area { fill: url(#wash-a); }
svg .vfill { fill: url(#vhatch-n); }
svg .dot { stroke: var(--card); stroke-width: 2; } svg .dot.ln-acc { fill: var(--accent); }
svg .bar-a { fill: url(#hatch-a); } svg .bar-n { fill: url(#hatch-n); }
svg .hit { fill: transparent; cursor: crosshair; } svg .hit:hover { fill: var(--ink); fill-opacity: .12; }
.st-a { stop-color: var(--accent); }
.hb-a { fill: var(--accent); } .hb-bg-a { fill: var(--accent); fill-opacity: .12; }
.hb-n { fill: var(--n3); } .hb-bg-n { fill: var(--n3); fill-opacity: .1; }
.hb-vn { fill: var(--n3); fill-opacity: .45; }
/* tables + faq */
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 16px; background: var(--raise); min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }
th, td { padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--rule-2); vertical-align: top; }
th { font-size: 12px; color: var(--muted); font-weight: 500; }
td.r, th.r { text-align: right; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
tr.hl td { background: var(--accent-soft); }
.faq { display: grid; gap: 10px; }
details.qa { background: var(--card); border: 1px solid var(--rule-2); border-radius: 22px; box-shadow: var(--shadow); }
details.qa > summary { list-style: none; cursor: pointer; padding: 18px 22px; font-size: 17px; font-weight: 500; display: flex; justify-content: space-between; align-items: center; gap: 16px; }
details.qa > summary::-webkit-details-marker { display: none; }
.plus { width: 28px; height: 28px; border-radius: 50%; border: 1px solid var(--rule); position: relative; flex: none; }
.plus::before, .plus::after { content: ""; position: absolute; left: 50%; top: 50%; width: 11px; height: 1.5px; background: var(--ink); transform: translate(-50%, -50%); }
.plus::after { transform: translate(-50%, -50%) rotate(90deg); transition: transform .2s; }
details.qa[open] .plus::after { transform: translate(-50%, -50%) rotate(0deg); }
.qa-a { padding: 0 22px 20px; display: grid; gap: 12px; color: var(--ink-2); max-width: 980px; }
.qa-a p, .qa-a ul { max-width: 72ch; }
.gloss { display: grid; grid-template-columns: minmax(0, 220px) minmax(0, 1fr); gap: 8px 18px; margin: 0; max-width: 900px; }
.gloss dt { font-weight: 500; color: var(--ink); } .gloss dd { margin: 0; } .qa-a ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
footer { color: var(--muted); font-size: 13px; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; border-top: 1px solid var(--rule); padding-top: 18px; }
#tip { position: fixed; z-index: 20; max-width: 320px; background: var(--glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-line); color: var(--ink); font-size: 13px; line-height: 1.45; padding: 8px 12px; border-radius: 14px;
  box-shadow: 0 12px 28px -12px rgba(17,18,20,.35); pointer-events: none; }
/* equation: organic + paid = total */
.brief { gap: 0; padding-block: 10px; }
.br-row { display: grid; grid-template-columns: 130px minmax(0, 1fr); gap: 18px; padding: 16px 0; border-top: 1px solid var(--rule); }
.br-row:first-child { border-top: 0; }
.br-k { font-size: 13px; font-weight: 600; color: var(--muted); padding-top: 3px; }
.br-row p { margin: 0; font-size: 17px; line-height: 1.5; color: var(--ink-2); max-width: 82ch; } .br-row p b { color: var(--ink); font-weight: 600; }
.br-row a, .claim-note a { color: var(--accent-ink); white-space: nowrap; }
.steps { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; counter-reset: s; }
.step { position: relative; display: grid; align-content: start; gap: 8px; padding: 18px 18px 16px; border-radius: 20px; background: var(--n1); color: inherit; text-decoration: none; transition: transform .15s ease; }

.step:not(:last-child)::after { content: "→"; position: absolute; right: -13px; top: 22px; font-size: 16px; color: var(--faint); }
.step-n { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; background: var(--ink); color: var(--card); font-size: 12px; font-weight: 600; }
.step h3 { font-size: 18px; }
.step p { margin: 0; font-size: 14px; color: var(--ink-2); } .step p b { color: var(--ink); font-weight: 600; font-variant-numeric: tabular-nums; }
.step-p { margin-top: auto; padding-top: 8px; border-top: 1px solid var(--rule); font-size: 12.5px; color: var(--muted); }
.cov-row { display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 14px; align-items: center; }
.cov-l { display: grid; font-size: 12.5px; color: var(--muted); } .cov-l b { font-size: 14.5px; color: var(--ink); font-weight: 500; }
.cov-bar { display: flex; gap: 2px; height: 40px; }
.sg.cd, .sw.cov-k.cd { background: var(--c-o); } .sg.cd span { color: var(--on-c); }
.sg.es, .sw.cov-k.es { background: color-mix(in srgb, var(--c-o) 35%, var(--card)); } .sg.es span { color: var(--c-o-ink); }
.sg.ej, .sw.cov-k.ej { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--c-o) 45%, transparent) 0 2px, transparent 2px 6px), color-mix(in srgb, var(--c-o) 18%, var(--card)); } .sg.ej span { color: var(--c-o-ink); }
.sg.ms, .sw.cov-k.ms { background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--warn) 40%, transparent) 0 2px, transparent 2px 7px), var(--warn-bg); }
.sg.ms span { color: var(--warn); }
.gaps { list-style: none; margin: 0; padding: 0; display: grid; }
.gaps li { display: grid; grid-template-columns: 28px minmax(0, 1.25fr) minmax(0, 1fr); gap: 16px; padding: 14px 0; border-top: 1px solid var(--rule); align-items: start; }
.gaps li:first-child { border-top: 0; padding-top: 4px; }
.gap-n { width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center; background: var(--warn-bg); color: var(--warn); font-size: 12px; font-weight: 600; }
.gaps li > div { display: grid; gap: 3px; font-size: 13.5px; } .gaps b { font-weight: 500; font-size: 14.5px; } .gaps li > div > span { color: var(--muted); }
.gap-fix { color: var(--ink-2); } .gaps .gap-o { font-size: 12px; font-weight: 500; color: var(--accent-ink); }
.t { font-weight: 500; } .t.o { color: var(--c-o-ink); } .t.i { color: var(--c-i-ink); } .t.f { color: var(--c-f-ink); }
.answers { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.ans { display: grid; align-content: start; gap: 6px; padding: 18px 18px 16px; border-radius: 22px; background: var(--card); border: 1px solid var(--rule-2); box-shadow: var(--shadow); color: inherit; text-decoration: none; transition: transform .15s ease; }
.ans:hover { transform: translateY(-2px); } .ans:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.ans-k { display: flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 500; color: var(--muted); }
.ans-k i { font-style: normal; width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center; background: var(--ink); color: var(--card); font-size: 11px; font-weight: 600; }
.ans > b { font-size: 40px; font-weight: 500; letter-spacing: -.045em; line-height: 1.05; font-variant-numeric: tabular-nums; }
.ans:nth-child(2) > b { color: var(--c-f-ink); } .ans:nth-child(3) > b { color: var(--n4); } .ans:nth-child(4) > b { color: var(--warn); }
.ans-s { font-size: 13.5px; color: var(--ink-2); }
.claim-head { display: grid; gap: 6px; }
.kicker { font-size: 12.5px; font-weight: 500; color: var(--muted); }
.claim { font-size: clamp(20px, 2.1vw, 25px); line-height: 1.25; letter-spacing: -.02em; }
.claim-sub { font-size: 13.5px; color: var(--muted); margin: 0 0 10px; }
.claim-note { font-size: 14.5px; color: var(--ink-2); margin: 0; max-width: 78ch; }
.proof { margin: auto 0 0; font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 12px; }
.proof b { color: var(--ink-2); font-weight: 600; }
.eqn { display: flex; align-items: stretch; gap: 10px; }
.eqn-t { flex: 1 1 0; min-width: 0; display: grid; gap: 2px; padding: 14px 16px 12px; border-radius: 18px; background: var(--n1); }
.eqn-t span { font-size: 13px; color: var(--muted); }
.eqn-t b { font-size: clamp(26px, 3.2vw, 40px); font-weight: 500; letter-spacing: -.045em; line-height: 1.1; }
.eqn-t.o { box-shadow: inset 0 4px 0 var(--c-o); } .eqn-t.o b { color: var(--c-o-ink); }
.eqn-t.i { box-shadow: inset 0 4px 0 var(--c-i); } .eqn-t.i b { color: var(--c-i-ink); }
.eqn-t.f { box-shadow: inset 0 4px 0 var(--c-f); } .eqn-t.f b { color: var(--c-f-ink); }
.eqn-t.tot { background: var(--inv); } .eqn-t.tot b { color: var(--inv-ink); } .eqn-t.tot span { color: var(--inv-2); }
.eqn-op { align-self: center; font-size: 28px; font-weight: 300; color: var(--faint); }
.eqb { display: grid; gap: 10px; margin-top: 4px; }
.eqb-row { display: grid; grid-template-columns: 92px minmax(0, 1fr) 52px; gap: 14px; align-items: center; }
.eqb-l { font-size: 13.5px; color: var(--ink-2); }
.eqb-r { text-align: right; font-size: 15px; font-weight: 500; font-variant-numeric: tabular-nums; }
.eqb-track, .fbx-bar, .lim-bar { display: flex; gap: 2px; height: 36px; }
.sg { display: flex; align-items: center; height: 100%; min-width: 3px; border-radius: 8px; overflow: hidden; font-style: normal; transform-origin: left center; }
.sg span { padding: 0 10px; font-size: 12.5px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sg.o { background: var(--c-o); } .sg.i { background: var(--c-i); } .sg.f { background: var(--c-f); }
.sg.o span, .sg.i span, .sg.f span { color: var(--on-c); }
.sg.ip { background: color-mix(in srgb, var(--c-i) 20%, var(--card)); } .sg.ip span { color: var(--c-i-ink); }
.sg.fp { background: color-mix(in srgb, var(--c-f) 20%, var(--card)); } .sg.fp span { color: var(--c-f-ink); }
.eqb-brk { display: flex; gap: 2px; }
.brk { position: relative; min-width: 0; padding: 14px 6px 0; font-size: 13px; color: var(--ink-2); text-align: center; }
.brk::before { content: ""; position: absolute; left: 3px; right: 3px; top: 0; height: 8px; border: 1.5px solid var(--n3); border-top: 0; border-radius: 0 0 7px 7px; }
.dp { display: grid; gap: 22px; }
.dp-row, .dp-axis { display: grid; grid-template-columns: 210px minmax(0, 1fr) 104px; gap: 18px; }
.dp-row { align-items: end; }
.dp-lab { display: grid; gap: 2px; font-size: 13px; color: var(--muted); align-self: center; } .dp-lab b { color: var(--ink); font-weight: 500; font-size: 14.5px; }
.dp-plot { position: relative; padding-top: 8px; border-bottom: 1px solid var(--n2); }
.dp-band { position: absolute; top: 0; bottom: 0; background: color-mix(in srgb, var(--ink) 6%, transparent); border-radius: 8px 8px 0 0; }
.dp-one { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1.5px dashed var(--n3); }
.dp-cols { position: relative; display: grid; grid-template-columns: repeat(var(--n), minmax(0, 1fr)); align-items: end; }
.dp-col { display: flex; flex-direction: column-reverse; align-items: center; gap: 1px; padding-bottom: 2px; }
.dp-col i { width: min(8px, 85%); aspect-ratio: 1; border-radius: 50%; background: var(--n3); }
.dp-col i.in, .sw.dp-k.in { background: var(--ink); } .sw.dp-k { background: var(--n3); border-radius: 50%; }
.dp-stat { font-size: 13px; color: var(--muted); align-self: center; } .dp-stat b { display: block; font-size: 22px; line-height: 1.15; color: var(--ink); font-weight: 500; }
.dp-ticks, .rr-ticks { position: relative; height: 18px; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }
.dp-ticks span, .rr-ticks span { position: absolute; top: 0; transform: translateX(-50%); white-space: nowrap; }
.dp-ticks span:first-child, .rr-ticks span:first-child { transform: none; } .dp-ticks span:last-child { transform: translateX(-100%); }
.c3-grid { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr); gap: 32px; }
.rr-wrap { display: grid; gap: 8px; }
.rr-g { font-size: 12.5px; font-weight: 500; color: var(--muted); margin-top: 6px; }
.rr { display: grid; grid-template-columns: minmax(0, 178px) minmax(0, 1fr) 44px; gap: 12px; align-items: center; font-size: 13.5px; }
.rr > b { text-align: right; font-weight: 500; font-variant-numeric: tabular-nums; }
.rr:not(.best) > b, .rr:not(.best) .rr-l { color: var(--muted); }
.rr-track { position: relative; height: 22px; }
.rr-track::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; border-top: 1px solid var(--rule); }
.rr-one { position: absolute; top: -5px; bottom: -5px; border-left: 1.5px dashed var(--n3); }
.rr-iqr { position: absolute; top: 6px; height: 10px; border-radius: 999px; background: var(--n2); transform-origin: left center; }
.rr-dot { position: absolute; top: 50%; width: 13px; height: 13px; margin: -6.5px 0 0 -6.5px; border-radius: 50%; background: var(--n4); border: 2px solid var(--card); cursor: help; }
.rr.best.i .rr-iqr { background: color-mix(in srgb, var(--c-i) 32%, transparent); } .rr.best.i .rr-dot { background: var(--c-i); }
.rr.best.f .rr-iqr { background: color-mix(in srgb, var(--c-f) 32%, transparent); } .rr.best.f .rr-dot { background: var(--c-f); }
.rr-axis { margin-top: -2px; }
.wf-box { display: grid; gap: 14px; max-width: 260px; }
.sg .sm { display: none; }
.facts { display: grid; gap: 14px; }
.facts > div { display: grid; grid-template-columns: 76px minmax(0, 1fr); gap: 12px; align-items: baseline; }
.facts b { font-size: 26px; font-weight: 500; letter-spacing: -.035em; font-variant-numeric: tabular-nums; }
.facts span { font-size: 13.5px; color: var(--ink-2); }
.waffle { display: grid; grid-template-columns: repeat(10, minmax(0, 1fr)); gap: 3px; }
.waffle i { aspect-ratio: 1; border-radius: 3px; }
.waffle i.f, .sw.wf-f { background: var(--c-f); } .waffle i.i, .sw.wf-i { background: var(--c-i); } .waffle i.x, .sw.wf-x { background: var(--n3); }
.wf-key { display: grid; gap: 8px; } .wf-key .legend { gap: 6px 14px; font-size: 13px; }
.wf-key p { font-size: 12.5px; color: var(--muted); margin: 0; }
.fbx { display: grid; gap: 8px; }
.fbx-bar { height: 48px; } .fbx-bar .sg span { font-size: 16px; }
.fbx-lab { display: flex; gap: 2px; font-size: 12.5px; color: var(--muted); } .fbx-lab span { min-width: 0; flex: none; padding-right: 8px; } .fbx-lab span:last-child { flex: 1; }
.lim { display: grid; gap: 12px; }
.lim-bar { position: relative; height: 40px; margin-top: 6px; }
.lim-err { position: absolute; top: -6px; bottom: -6px; border-radius: 9px; border: 1.5px solid var(--warn);
  background: repeating-linear-gradient(135deg, color-mix(in srgb, var(--warn) 45%, transparent) 0 2px, transparent 2px 7px); }
.lim-lab { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; color: var(--muted); }
.eq-anim .anim .grow { transform: scaleX(0); transition: transform .8s cubic-bezier(.2, .8, .2, 1) var(--d, 0s); }
.eq-anim .anim.in .grow { transform: none; }
.eq-anim .anim .pop { opacity: 0; transform: translateY(6px); transition: opacity .3s ease var(--d, 0s), transform .45s cubic-bezier(.2, .8, .2, 1) var(--d, 0s); }
.eq-anim .anim.in .pop { opacity: 1; transform: none; }
.ac-sum { font-size: 14px; color: var(--ink-2); }
.ac-sum b { color: var(--ink); font-weight: 500; }
table.ac td { vertical-align: middle; }
.ac input, .ac select { font: inherit; font-size: 13.5px; color: var(--ink); background: var(--raise); border: 1px solid var(--rule); border-radius: 8px; padding: 6px 8px; }
.ac input { width: 120px; text-align: right; font-variant-numeric: tabular-nums; }
.ac input:focus, .ac select:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.ac .match-fb { color: var(--c-f-ink); font-weight: 500; } .ac .match-ig { color: var(--c-i-ink); font-weight: 500; }
/* answers first: bottom-line tiles, question dropdowns, small charts */
.card.bluf { gap: 18px; padding: 28px 28px 24px; background: linear-gradient(180deg, color-mix(in srgb, var(--accent) 5%, var(--card)), var(--card) 60%); }
.bluf > .kicker { font-size: 13px; font-weight: 600; color: var(--accent-ink); }
.bt3 { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; }
.bt { display: grid; align-content: start; gap: 10px; padding: 18px 18px 16px; border-radius: 20px; background: var(--raise); border: 1px solid var(--rule); min-width: 0; }
.bt-k { display: flex; align-items: center; justify-content: space-between; gap: 8px; font-size: 13px; font-weight: 500; color: var(--muted); }
.bt-a { font-size: 17px; font-weight: 600; line-height: 1.3; color: var(--ink); }
.bt-num { display: grid; gap: 2px; }
.bt-v { font-size: clamp(36px, 4vw, 48px); font-weight: 500; letter-spacing: -.045em; line-height: 1; font-variant-numeric: tabular-nums; }
.bt.w .bt-v { color: var(--warn); } .bt.o .bt-v { color: var(--c-o-ink); } .bt.i .bt-v { color: var(--c-i-ink); }
.bt-num span { font-size: 13px; color: var(--ink-2); }
.bt-cap { font-size: 12.5px; color: var(--muted); margin: 0; line-height: 1.45; } .bt-cap b { color: var(--ink); }
.bt-cap .pill.warn { padding: 1px 8px; font-size: 11.5px; margin-right: 4px; }
.bl-ak { font-size: 12.5px; font-weight: 600; color: var(--muted); }
.ask3 { display: grid; gap: 10px; }
.ask3 ol { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
.ask3 li { display: flex; align-items: flex-start; gap: 10px; padding: 12px 14px; border-radius: 14px; background: var(--accent-soft); font-size: 14px; font-weight: 500; color: var(--ink); line-height: 1.4; }
.a3-n { flex: none; width: 22px; height: 22px; border-radius: 7px; background: var(--card); color: var(--accent-ink); display: grid; place-items: center; font-size: 12px; font-weight: 600; }
.bl-cav { display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap; font-size: 13.5px; color: var(--ink-2); border-top: 1px solid var(--rule); padding-top: 14px; margin: 0; }
.bl-cav .pill.warn { padding: 2px 10px; font-size: 12px; }
.mb { display: grid; gap: 8px; }
.mb-r { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr) 46px; gap: 10px; align-items: center; font-size: 12.5px; }
.mb-l { color: var(--ink-2); line-height: 1.3; }
.mb-t { position: relative; height: 10px; border-radius: 999px; background: var(--rule-2); overflow: hidden; }
.mb-t i { position: absolute; left: 0; top: 0; bottom: 0; min-width: 3px; border-radius: 999px; background: var(--n3); transform-origin: left center; }
.mb-r.o .mb-t i { background: var(--c-o); } .mb-r.w .mb-t i { background: var(--warn); } .mb-r.i .mb-t i { background: var(--c-i); }
.mb-r > b { text-align: right; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--ink); }
.p100 { position: relative; display: flex; gap: 2px; height: 30px; margin-top: 6px; }
.p100 .sg span { font-size: 12px; }
.p100-l, .r100-l { display: flex; justify-content: space-between; gap: 8px; flex-wrap: wrap; font-size: 12px; color: var(--muted); }
.r100 { position: relative; height: 16px; border-radius: 999px; background: var(--rule-2); margin-top: 6px; }
.r100-a { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 999px 0 0 999px; background: var(--c-i); }
.r100-b { position: absolute; top: 0; bottom: 0; background: var(--c-f); opacity: .55; }
.r100-one { position: absolute; right: 0; top: -5px; bottom: -5px; border-left: 2px dashed var(--ink); opacity: .55; }
.d100 { display: grid; grid-template-columns: repeat(20, minmax(0, 1fr)); gap: 3px; max-width: 280px; }
.d100 i { aspect-ratio: 1; border-radius: 2px; background: var(--n2); } .d100 i.h { background: var(--warn); }
.rc { display: flex; flex-wrap: wrap; gap: 6px; }
.rc-i { display: inline-grid; padding: 6px 10px; border-radius: 10px; background: var(--n1); font-size: 11.5px; color: var(--muted); }
.rc-i b { font-size: 16px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.rc-i.on { background: color-mix(in srgb, var(--c-i) 16%, var(--card)); box-shadow: inset 0 0 0 1.5px var(--c-i); } .rc-i.on b { color: var(--c-i-ink); }
.s2 { display: flex; gap: 28px; } .s2 > div { display: grid; }
.s2 b { font-size: 32px; font-weight: 500; letter-spacing: -.035em; line-height: 1.1; font-variant-numeric: tabular-nums; color: var(--warn); } .s2 span { font-size: 12.5px; color: var(--muted); }
.c5 { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; align-items: end; }
.c5-c { display: grid; gap: 4px; justify-items: center; font-size: 11.5px; color: var(--muted); text-align: center; }
.c5-c b { font-size: 13px; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.c5-t { position: relative; width: 100%; height: 90px; border-radius: 8px; background: var(--rule-2); overflow: hidden; }
.c5-t i { position: absolute; left: 0; right: 0; bottom: 0; background: var(--c-i); border-radius: 8px 8px 0 0; }
.wf-mini { max-width: 180px; }
.acp { display: grid; gap: 6px; max-width: 320px; } .acp-t { height: 12px; border-radius: 999px; background: var(--rule-2); overflow: hidden; }
.acp-t i { display: block; height: 100%; border-radius: 999px; background: var(--c-o); } .acp span { font-size: 12.5px; color: var(--muted); } .acp b { color: var(--ink); }
.two-mini { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px; border-top: 1px solid var(--rule); padding-top: 16px; }
.two-mini > div { display: grid; gap: 10px; align-content: start; min-width: 0; }
.two-mini h4 { margin: 0; font-size: 14.5px; font-weight: 500; }
.qx-head { display: flex; justify-content: space-between; align-items: flex-end; gap: 12px; flex-wrap: wrap; padding-right: 0; }
.qx-all { font: inherit; font-size: 13px; border: 1px solid var(--rule); background: var(--raise); color: var(--ink); border-radius: 10px; padding: 7px 12px; cursor: pointer; min-height: 36px; }
.qx-all:hover { background: var(--n1); }
.qx-list { display: grid; }
details.qx { border-top: 1px solid var(--rule-2); } details.qx:first-child { border-top: 0; }
.qx > summary { list-style: none; cursor: pointer; display: grid; grid-template-columns: 26px minmax(0, 1fr) auto minmax(132px, auto) 20px; grid-template-areas: "n q v k c";
  gap: 14px; align-items: center; padding: 12px 10px; margin: 0 -10px; border-radius: 12px; }
.qx > summary::-webkit-details-marker { display: none; }
.qx > summary:hover { background: var(--rule-2); }
.qx-n { grid-area: n; width: 24px; height: 24px; border-radius: 50%; background: var(--n1); display: grid; place-items: center; font-size: 12px; font-weight: 600; color: var(--ink-2); }
.qx-q { grid-area: q; font-size: 15px; font-weight: 500; color: var(--ink); line-height: 1.35; }
.qx-v { grid-area: v; justify-self: start; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 999px; white-space: nowrap; }
.qx-v.no, .qx-v.part { background: var(--warn-bg); color: var(--warn); }
.qx-v.yes { background: color-mix(in srgb, var(--c-o) 16%, var(--card)); color: var(--c-o-ink); }
.qx-v.info { background: color-mix(in srgb, var(--c-i) 14%, var(--card)); color: var(--c-i-ink); }
.qx-v.wait { background: var(--n1); color: var(--ink-2); box-shadow: inset 0 0 0 1px var(--n2); }
.qx-k { grid-area: k; text-align: right; font-size: 16px; font-weight: 600; font-variant-numeric: tabular-nums; color: var(--ink); white-space: nowrap; }
.qx-c { grid-area: c; position: relative; width: 20px; height: 20px; }
.qx-c::before { content: ""; position: absolute; left: 6px; top: 3px; width: 7px; height: 7px; border-right: 1.5px solid var(--muted); border-bottom: 1.5px solid var(--muted); transform: rotate(45deg); transition: transform .2s; }
.qx[open] .qx-c::before { transform: translateY(5px) rotate(-135deg); }
.qx-b { display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1.15fr); gap: 26px; padding: 6px 0 20px 40px; align-items: start; }
.qx-viz { display: grid; gap: 8px; min-width: 0; }
.qx-t { display: grid; gap: 8px; min-width: 0; font-size: 14px; color: var(--ink-2); line-height: 1.5; } .qx-t p { margin: 0; max-width: 62ch; } .qx-t a { font-size: 13px; font-weight: 500; }
.step-v { display: grid; gap: 2px; } .step-v b { font-size: 28px; font-weight: 500; letter-spacing: -.035em; line-height: 1.1; font-variant-numeric: tabular-nums; }
.step-v span { display: flex; align-items: center; font-size: 13px; color: var(--ink-2); }
a.step-p { text-decoration: none; color: var(--accent-ink); }
details.more-d > summary { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; list-style: none; font-size: 13px; font-weight: 500; color: var(--accent-ink); }
details.more-d > summary::-webkit-details-marker { display: none; }
details.more-d > summary::after { content: "+"; } details.more-d[open] > summary::after { content: "−"; } details.more-d[open] > summary { margin-bottom: 10px; }
.tried { display: grid; gap: 12px; }
.tr { display: grid; grid-template-columns: minmax(0, 300px) minmax(0, 1fr) 56px; gap: 16px; align-items: center; }
.tr-l { display: grid; gap: 1px; font-size: 14px; color: var(--ink-2); } .tr-l small { font-size: 12px; color: var(--muted); }
.tr.on .tr-l { color: var(--ink); font-weight: 500; } .tr.on .tr-l small { font-weight: 400; }
.tr-track { position: relative; height: 14px; border-radius: 999px; background: var(--rule-2); }
.tr-fill { position: absolute; left: 0; top: 0; bottom: 0; min-width: 4px; border-radius: 999px; background: var(--n3); transform-origin: left center; }
.tr.on .tr-fill, .sw.tr-k.on { background: var(--c-o); } .sw.tr-k { background: var(--n3); }
.tr > b { text-align: right; font-size: 15px; font-weight: 500; font-variant-numeric: tabular-nums; color: var(--muted); }
.tr.on > b { color: var(--c-o-ink); }
.tr-ax { margin-top: -4px; } .tr-ax .rr-ticks span:last-child { transform: translateX(-100%); }
.lede { font-size: clamp(16px, 1.6vw, 19px); color: var(--ink-2); max-width: 60ch; margin-top: 12px; line-height: 1.45; }
@media (max-width: 1080px) {
  .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .span2 { grid-column: 1 / -1; }
  .sc { grid-template-columns: minmax(0, 1fr); gap: 22px; }
  .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .answers { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .eq-fbcard { grid-column: 1 / -1; }
  .steps { grid-template-columns: repeat(2, minmax(0, 1fr)); } .step::after { display: none; }
}
@media (max-width: 820px) {
  .bt3, .ask3 ol, .two-mini, .qx-b { grid-template-columns: minmax(0, 1fr); }
  .qx-b { padding-left: 0; gap: 14px; }
}
@media (max-width: 720px) {
  table.sct th, table.sct td { padding: 10px 6px; }
  table.sct thead th:first-child { width: 40%; }
  .sc-v .num { font-size: 16px; }
  .sc-help { display: none; }
  .sc-big { font-size: 60px; }
  .page-wrap { padding: 8px; }
  .shell { border-radius: 26px; padding-inline: 16px; padding-block: 18px 32px; gap: 22px; }
  .grid { grid-template-columns: minmax(0, 1fr); gap: 14px; }
  .card { border-radius: 22px; padding: 20px 18px; }
  details.more > summary { top: 14px; right: 14px; width: 38px; height: 38px; }
  .tiles { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .cols-row { grid-template-columns: 34px repeat(5, minmax(0, 1fr)); }
  .gl { left: 34px; }
  .ch { padding-inline: 6px; } .cl { font-size: 11px; white-space: normal; }
  .cols-row.plot { height: 200px; }
  .cols-row.plot .pill.float { display: none; }
  .pill.float.mobile { display: block; position: static; transform: none; margin-top: 10px; border-radius: 16px; white-space: normal; line-height: 1.6; }
  .pill.mobile i { display: inline-block; margin: 0 6px; vertical-align: middle; }
  .dr-row { grid-template-columns: minmax(0, 1fr); gap: 0; }
  .ax .dr-l { display: none; }
  .dm { grid-template-columns: minmax(0, 1fr); }
  .ans { padding: 16px; } .ans > b { font-size: 32px; }
  .br-row { grid-template-columns: minmax(0, 1fr); gap: 4px; } .br-row p { font-size: 15.5px; }
  .steps { grid-template-columns: minmax(0, 1fr); }
  .cov-row { grid-template-columns: minmax(0, 1fr); gap: 6px; }
  .gaps li { grid-template-columns: 28px minmax(0, 1fr); gap: 6px 12px; } .gaps li > div:last-child { grid-column: 2; }
  .eqn { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; } .eqn-op { display: none; }
  .eqb-row { grid-template-columns: 64px minmax(0, 1fr) 40px; gap: 8px; } .eqb-l { font-size: 12.5px; }
  .brk { font-size: 12px; padding-inline: 2px; }
  .dp-row, .dp-axis { grid-template-columns: minmax(0, 1fr); gap: 8px; } .dp-axis .dp-gap { display: none; }
  .dp-stat b { display: inline; font-size: 15px; margin-right: 4px; }
  .c3-grid { grid-template-columns: minmax(0, 1fr); gap: 22px; }
  .rr { grid-template-columns: minmax(0, 128px) minmax(0, 1fr) 40px; gap: 8px; font-size: 12.5px; }
  .sg .lg { display: none; } .sg .sm { display: inline; } .sg span { padding: 0 6px; font-size: 11.5px; }
  .gloss { grid-template-columns: minmax(0, 1fr); gap: 2px; } .gloss dd { margin-bottom: 8px; }
  .card.bluf { padding: 22px 16px 18px; gap: 16px; }
  .bt { padding: 16px; }
  .qx > summary { grid-template-columns: 24px minmax(0, 1fr) auto 16px; grid-template-areas: "n q q c" ". v k ."; gap: 6px 10px; padding: 12px 8px; margin: 0 -8px; }
  .qx-k { font-size: 15px; }
  .tr { grid-template-columns: minmax(0, 1fr) 48px; gap: 4px 12px; } .tr-l { grid-column: 1 / -1; } .tr-ax > span:first-child { display: none; }
  .brand { font-size: 22px; }
  .top-right { width: 100%; justify-content: space-between; flex-wrap: nowrap; }
  .top { min-width: 0; width: 100%; }
  nav.pills { min-width: 0; flex: 1 1 0; }
  nav.pills a { padding: 8px 11px; font-size: 14px; }
  .seg-ctl.theme button span { display: none; }
}
@media (min-width: 721px) { .pill.mobile { display: none; } }
@media (max-width: 520px) { .answers { grid-template-columns: minmax(0, 1fr); } }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = """
(function () {
  var tip = document.getElementById('tip');
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth;
    var left = Math.min(Math.max(8, x + 14), vw - w - 8), top = y - h - 14;
    if (top < 8) top = y + 18;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function show(el, x, y) { tip.textContent = el.getAttribute('data-tip'); tip.hidden = false; place(x, y); }
  function find(t) { return t && t.closest ? t.closest('[data-tip]') : null; }
  document.addEventListener('pointerover', function (e) { var el = find(e.target); if (el) show(el, e.clientX, e.clientY); else tip.hidden = true; });
  document.addEventListener('pointermove', function (e) { if (!tip.hidden) place(e.clientX, e.clientY); });
  document.addEventListener('focusin', function (e) { var el = find(e.target); if (!el) return; var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); });
  document.addEventListener('focusout', function () { tip.hidden = true; });
  document.addEventListener('click', function (e) { var el = find(e.target); if (el && el.classList.contains('info')) { var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top); } });
  window.addEventListener('scroll', function () { tip.hidden = true; }, { passive: true });
  document.querySelectorAll('[data-pf-btn]').forEach(function (b) {
    b.addEventListener('click', function () {
      var p = b.getAttribute('data-pf-btn');
      document.querySelectorAll('[data-pf-btn]').forEach(function (x) { x.setAttribute('aria-pressed', x === b ? 'true' : 'false'); });
      document.querySelectorAll('[data-pf]').forEach(function (el) { el.hidden = el.getAttribute('data-pf') !== p; });
    });
  });
  var root = document.documentElement, tbtn = document.querySelectorAll('[data-theme-btn]');
  var mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  function effective() { var t = root.getAttribute('data-theme'); return t === 'dark' || t === 'light' ? t : (mq && mq.matches ? 'dark' : 'light'); }
  function sync() { var e = effective(); tbtn.forEach(function (b) { b.setAttribute('aria-pressed', b.getAttribute('data-theme-btn') === e ? 'true' : 'false'); }); }
  try { var saved = localStorage.getItem('po-theme'); if (saved === 'light' || saved === 'dark') root.setAttribute('data-theme', saved); } catch (err) {}
  tbtn.forEach(function (b) {
    b.addEventListener('click', function () {
      var t = b.getAttribute('data-theme-btn'); root.setAttribute('data-theme', t);
      try { localStorage.setItem('po-theme', t); } catch (err) {}
      sync();
    });
  });
  if (mq && mq.addEventListener) mq.addEventListener('change', sync);
  sync();
  document.querySelectorAll('.qx-all').forEach(function (b) {
    b.addEventListener('click', function () {
      var open = b.getAttribute('data-open') !== '1';
      document.querySelectorAll('details.qx').forEach(function (d) { d.open = open; });
      b.setAttribute('data-open', open ? '1' : '0'); b.textContent = open ? 'Close all' : 'Open all';
    });
  });
  var links = document.querySelectorAll('nav.pills a');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (!en.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle('on', a.getAttribute('href') === '#' + en.target.id); });
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    ['start', 'overview', 'findings', 'equation', 'targets', 'model', 'faq'].forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }
})();

(function () {
  // Start and Equation sections: bars grow and dots appear once each chart scrolls into view (skipped when the viewer prefers less motion)
  var secs = ['start', 'equation'].map(function (id) { return document.getElementById(id); }).filter(Boolean);
  if (!secs.length || !('IntersectionObserver' in window) || (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches)) return;
  secs.forEach(function (sec) { sec.classList.add('eq-anim'); });
  var io = new IntersectionObserver(function (es) {
    es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
  }, { threshold: 0.15 });
  secs.forEach(function (sec) { sec.querySelectorAll('.anim').forEach(function (c) { io.observe(c); }); });
})();

(function () {
  // Callout 3: shared in-app check list, read from and written to this artifact's database (rows are never in the page source)
  var box = document.getElementById('appcheck');
  if (!box || !window.claude || typeof window.claude.use !== 'function') return;
  var sum = box.querySelector('.ac-sum'), body = box.querySelector('tbody');
  var fmt = function (n) { return typeof n === 'number' && isFinite(n) ? Math.round(n).toLocaleString('en-US') : ''; };
  var today = function () { var d = new Date(); return d.getFullYear() + '-' + String(d.getMonth() + 1).padStart(2, '0') + '-' + String(d.getDate()).padStart(2, '0'); };
  function closer(r) {
    var app = r.app_views, ig = r.nimble, both = r.nimble + r.fb_part;
    if (typeof app !== 'number' || !(app > 0) || !(ig > 0)) return '';
    return Math.abs(app / ig - 1) <= Math.abs(app / both - 1) ? 'Instagram only' : 'Instagram + Facebook';
  }
  window.claude.use('db').then(function (db) {
    if (!db) { sum.textContent = 'The check list is not available in this view.'; return; }
    var col = db.collection('app_checks');
    col.orderBy('order').onSnapshot(function (snap) {
      body.textContent = '';
      var done = 0, both = 0, igOnly = 0;
      snap.docs.forEach(function (doc) {
        var r = doc.data() || {}, tr = document.createElement('tr');
        var td = function (cls) { var c = document.createElement('td'); if (cls) c.className = cls; tr.appendChild(c); return c; };
        var a = document.createElement('a'), url = String(r.url || '');
        if (/^https:\\/\\/www\\.instagram\\.com\\/(p|reel|reels)\\/[A-Za-z0-9_-]+\\/?$/.test(url)) { a.href = url; a.target = '_blank'; a.rel = 'noopener'; }
        a.textContent = 'Post ' + (r.order || '') + (r.published ? ' (' + r.published + ')' : '');
        td().appendChild(a);
        td('r').textContent = fmt(r.nimble);
        td('r').textContent = fmt(r.nimble + r.fb_part);
        var inp = document.createElement('input'); inp.type = 'number'; inp.min = '0'; inp.inputMode = 'numeric';
        inp.setAttribute('aria-label', 'Views shown in the app for post ' + (r.order || ''));
        if (typeof r.app_views === 'number') inp.value = String(r.app_views);
        inp.addEventListener('change', function () {
          var v = inp.value === '' ? null : Number(inp.value);
          if (v !== null && !(v >= 0)) return;
          col.doc(doc.id).update({ app_views: v, checked_on: v === null ? null : today() }).catch(function () { sum.textContent = 'Could not save. Try again.'; });
        });
        td('r').appendChild(inp);
        var sel = document.createElement('select');
        [['', '—'], ['Y', 'Yes'], ['N', 'No']].forEach(function (o) { var op = document.createElement('option'); op.value = o[0]; op.textContent = o[1]; sel.appendChild(op); });
        sel.value = r.cross_posted || '';
        sel.setAttribute('aria-label', 'Cross-posted to Facebook, post ' + (r.order || ''));
        sel.addEventListener('change', function () { col.doc(doc.id).update({ cross_posted: sel.value || null }).catch(function () { sum.textContent = 'Could not save. Try again.'; }); });
        td().appendChild(sel);
        var c = closer(r), out = td(c === 'Instagram only' ? 'match-ig' : c ? 'match-fb' : '');
        out.textContent = c;
        if (c) { done++; if (c === 'Instagram only') igOnly++; else both++; }
        body.appendChild(tr);
      });
      sum.textContent = '';
      var b = function (s) { var e = document.createElement('b'); e.textContent = s; return e; };
      if (!snap.size) { sum.textContent = 'No posts in the list yet.'; return; }
      sum.append(b(done + ' of ' + snap.size), ' checked. ', b(String(both)), ' closer to Instagram + Facebook, ', b(String(igOnly)), ' closer to Instagram only.');
      document.querySelectorAll('.ac-left').forEach(function (el) { el.textContent = String(snap.size - done); });
      document.querySelectorAll('.ac-done').forEach(function (el) { el.textContent = String(done); });
      document.querySelectorAll('.ac-prog').forEach(function (el) { el.style.width = (snap.size ? 100 * done / snap.size : 0) + '%'; });
    }, function () { sum.textContent = 'The check list could not load in this view.'; });
  });
})();
"""

CAL = '<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3.5" y="5" width="17" height="15" rx="3"/><path d="M3.5 10h17M8 3v4M16 3v4"/></svg>'
MARK = ('<svg viewBox="0 0 20 20" aria-hidden="true"><rect class="mk" x="3" y="9" width="4" height="8" rx="1.2"/>'
        '<rect class="mk" x="8" y="5" width="4" height="12" rx="1.2"/><rect class="mk a" x="13" y="2" width="4" height="15" rx="1.2"/></svg>')
SUN = ('<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/>'
       '<path d="M12 2.5v2M12 19.5v2M4.6 4.6l1.4 1.4M18 18l1.4 1.4M2.5 12h2M19.5 12h2M4.6 19.4L6 18M18 6l1.4-1.4"/></svg>')
MOON = '<svg viewBox="0 0 24 24" stroke-width="1.8" stroke-linejoin="round" aria-hidden="true"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg>'


def build():
    D = load()
    R = D["R"]
    t0, t1 = (D["C2"] or {}).get("test_cutoff", "2026-07-01"), (D["C2"] or {}).get("test_end", "2026-09-09")
    body = f"""<title>Paid vs Organic</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&display=swap">
<style>/* Layout: one rounded app shell; 3-column card grid (charts first), one-line findings, model cards with a platform toggle, FAQ accordion. */{CSS}</style>
{DEFS}
<div class="page-wrap"><div class="shell">
  <div class="top">
    <div class="brand"><span class="mark">{MARK}</span>paid/organic</div>
    <div class="top-right">
      <nav class="pills" aria-label="Sections">{'<a class="on" href="#start">Answers</a>' if D.get("PM") else ""}<a{'' if D.get("PM") else ' class="on"'} href="#overview">Overview</a><a href="#findings">Findings</a>{'<a href="#equation">Equation</a>' if D.get("EQ") else ""}{'<a href="#targets">Targets</a>' if D["TG"] else ""}<a href="#model">Model</a><a href="#faq">FAQ</a></nav>
      <div class="seg-ctl theme" role="group" aria-label="Color theme"><button type="button" data-theme-btn="light" aria-pressed="true" aria-label="Light mode">{SUN}<span>Light</span></button><button type="button" data-theme-btn="dark" aria-pressed="false" aria-label="Dark mode">{MOON}<span>Dark</span></button></div>
    </div>
  </div>
  <div class="title-row">
    <div><h1>Paid vs Organic</h1>{'<p class="lede">Can we split a boosted post&#8217;s views into organic and paid? Answers first, proof below.</p>' if D.get("PM") else ""}</div>
    <div class="dates"><span class="vs">Paid posts tracked</span><span tabindex="0" data-tip="{esc(f"Daily reads of {R['panel']['posts']} paid Instagram posts, used to test whether total minus paid equals organic.")}">{CAL}{day(R['panel']['obs_from'], True)} – {day(R['panel']['obs_to'], True)}</span><span class="vs">Model test posts</span><span tabindex="0" data-tip="Posts published in this window were kept out of training and scored once per model version.">{CAL}{day(t0)} – {day(t1, True)}</span></div>
  </div>
  {start(D)}
  {overview(D)}
  {findings(D)}
  {equation(D)}
  {targets(D)}
  {model(D)}
  {faq(D)}
  <footer><span>Aggregates only. Data read 2026-10-07. Typical = median.</span><a href="{REPO}">Code and SQL on GitHub</a></footer>
</div></div>
<div id="tip" role="tooltip" hidden></div>
<script>{JS}</script>
"""
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w").write(body)
    print("wrote", OUT, f"{len(body) / 1024:.0f} KB")


if __name__ == "__main__":
    build()
