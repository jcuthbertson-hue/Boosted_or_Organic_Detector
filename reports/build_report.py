"""Build reports/boost_report.html: can we subtract paid to get organic, and which posts are paid.

Charts first, one-line findings with hover detail, and an FAQ for the long answers.
Reads aggregate result files only (no row-level data, no client or person names):
  results/reconciliation.json, results/reconciliation_tiktok.json, results/classification_metrics.json,
  results/organic_coverage.json, results/weighted_recall.json, results/llm_vs_ml*.csv, reconcile/organic_curve_*.csv
Run from repo root (after reconcile.analyze, reconcile.analyze_tiktok, detector.evaluate):
  python3 -m reports.build_report
"""
import csv
import html
import json
import math
import os

OUT = "reports/boost_report.html"
REPO = "https://github.com/jcuthbertson-hue/boosted_or_organic_detector/tree/claude/funny-franklin-5w06tt"
PLATFORMS = [("Instagram", "Instagram"), ("Tiktok", "TikTok")]
DEFAULT_PF = "Instagram"
HAND_RULE = "Hand rule: views > followers AND engagement < 1%"


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
                AU=j("results/ig_miss_audit.json") if os.path.exists("results/ig_miss_audit.json") else None)


# ---------------------------------------------------------------- small components
def info(tip):
    return f'<button class="info" type="button" data-tip="{esc(tip)}" aria-label="{esc(tip)}">i</button>'


def table(head, rows, hl=None):
    th = "".join(f'<th class="{"r" if i else ""}">{esc(h)}</th>' for i, h in enumerate(head))
    tr = "".join(f'<tr class="{"hl" if hl and hl(r) else ""}">' + "".join(f'<td class="{"r" if i else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def card(title, body, *, cls="", sub="", tip="", data=None, attrs=""):
    head = (f'<div class="card-head"><h3>{esc(title)}{info(tip) if tip else ""}</h3>'
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


def progress(rows, fmt=pct):
    return '<div class="pbars">' + "".join(
        f'<div class="pb"><div class="pb-top"><span>{esc(lab)}</span><b>{fmt(v)}</b></div>'
        f'<div class="pb-track"><div class="pb-fill {cls}" style="width:{100 * v:.1f}%" tabindex="0" data-tip="{esc(tip)}"></div></div></div>'
        for lab, v, cls, tip in rows) + "</div>"


def dots(V):
    seg = [("measured_optin", "Measured (opt-in)", "m"), ("est_high", "Estimate, high", "h"), ("est_medium", "Estimate, medium", "md"),
           ("est_low", "Estimate, low", "l"), ("not_separable", "Not separable", "x")]
    out = []
    for p, lab in PLATFORMS:
        t = V["totals"][p]
        counts = [round(100 * t[k] / t["posts"]) for k, _, _ in seg]
        counts[0] += 100 - sum(counts)
        cells = []
        for (k, name, cls), n in zip(seg, counts):
            tip = f"{lab}: {name}: {t[k]:,} of {t['posts']:,} paid posts ({pct(t[k] / t['posts'])})"
            cells += [f'<i class="dt {cls}" data-tip="{esc(tip)}"></i>'] * n
        out.append(f'<div class="dm"><span class="dm-lab">{lab}<small>{t["posts"]:,} paid posts</small></span><div class="dm-grid">{"".join(cells)}</div></div>')
    return "".join(out) + legend([(c, n) for _, n, c in seg])


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
               '<span class="chip">/tiktok dark-post test</span> to check TikTok organic.</div></div>',
             cls="span2 hero", sub="Organic error of “public − paid impressions”, by paid share of public views",
             tip=f"Instagram, {R['postflight_posts']} paid posts, one read per post after the ads ended. Truth = opt-in views (organic only). Error = median absolute error.",
             data=table(["Paid share of public", "Posts", "Typical organic error"],
                        [[esc(r["paid_share"]), r["posts"], pct(r["median_abs_error"])] for r in R["error_by_paid_share"]]))
    b = card("Use the views before the boost",
             f'<div class="big">{pct(curve["median_abs_error"], 1)}</div><p class="big-sub">typical organic error, never negative</p>'
             + '<p class="mini">Posts within ±25% of true organic</p>'
             + progress([("Pre-boost views × organic growth", curve["within_25pct"], "solid",
                          f"{pct(curve['within_25pct'])} of {curve['posts']} posts within ±25%; typical error {pct(curve['median_abs_error'], 1)}"),
                         ("Public − paid impressions", best["within_25pct"], "hatch-n",
                          f"{pct(best['within_25pct'])} of {best['posts']} posts within ±25%; typical error {pct(best['median_abs_error'])}; {pct(best['negative_organic'])} negative"),
                         ("SocAPI total − paid", soc["within_25pct"], "hatch-n",
                          f"{pct(soc['within_25pct'])} of {soc['posts']} posts within ±25%; typical error {pct(soc['median_abs_error'])}")]),
             tip="Take the last public read before the first ad day, then grow it by the median curve of unpaid posts. Checked against opt-in views on 117 paid Instagram posts.",
             data=table(["Method", "Posts", "Typical error", "Within ±25%", "Negative"],
                        [[esc(n), rec[k]["posts"], pct(rec[k]["median_abs_error"]), pct(rec[k]["within_25pct"]), pct(rec[k]["negative_organic"])]
                         for k, n in [("pre-boost read x organic curve (no fitting on these posts)", "Pre-boost × curve"),
                                      ("public - paid IG Impressions", "Public − paid impressions"),
                                      ("SocAPI total - FB cross-post - IG impressions - FB video plays (best mix)", "SocAPI total − paid")]]))
    c = card("Normal organic growth", curve_chart(D["curves"]) + legend([("c-acc", "Instagram"), ("c-n4", "TikTok")]),
             sub="Share of day-120 views by post age, unpaid posts",
             tip="The method grows the pre-boost views along this curve. Built from unpaid posts only (Instagram 1,505, TikTok 608).",
             data=table(["Day", "Instagram", "TikTok"],
                        [[a, pct(v, 1), pct(D["curves"]["Tiktok"][i][1], 1)] for i, (a, v, _) in enumerate(D["curves"]["Instagram"]) if a in (0, 1, 3, 7, 14, 30, 60, 90, 120)]))
    d = card("Organic we can measure",
             f'<div class="big sm">{pct(ig_cov["measured_optin"] / ig_cov["posts"])}</div><p class="big-sub">of paid Instagram posts: opt-in gives true organic</p>' + dots(V),
             tip="Each dot is 1% of paid posts in BIRA (all dates). Confidence of the estimate depends on how late the pre-boost read is: day 14+ high, day 7–13 medium, before day 7 low.",
             data=table(["Platform", "Measured", "High", "Medium", "Low", "Not separable"],
                        [[lab] + [f'{V["totals"][p][k]:,}' for k in ("measured_optin", "est_high", "est_medium", "est_low", "not_separable")] for p, lab in PLATFORMS]))
    e = (f'<article class="card inv"><span class="pill glass">Paid-post model</span>'
         f'<div class="inv-num">{pct(P["Tiktok"]["test_metrics"]["precision"])}</div>'
         f'<p class="inv-sub">of TikTok flags are real paid posts. Instagram: {pct(P["Instagram"]["test_metrics"]["precision"])}.</p></article>')
    f = card("Which paid metric matches the platform count", dot_range(R), cls="span3",
             sub="Extra views on the platform ÷ paid metric. 1× is a perfect match.",
             tip="Instagram side: (public − opt-in) ÷ paid Instagram-placement metric, 146 posts. Facebook side: SocAPI Facebook-paid plays ÷ paid Facebook-placement metric, 20 posts. Dot = median, bar = middle half. Log scale.",
             data=table(["Paid metric", "Instagram side", "Facebook side"],
                        [[esc(n), times(R["metric_match"]["instagram_side"][n]["median"]),
                          times(R["metric_match"]["facebook_side"][n]["median"]) if R["metric_match"]["facebook_side"][n]["n"] else "no data"]
                         for n in R["metric_match"]["instagram_side"]]))
    return f'<section id="overview" class="grid">{a}{b}{c}{d}{e}{f}</section>'


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
        ("Impressions is the paid metric that matches the platform count.",
         f"Instagram: public − opt-in = {times(ig['Impressions']['median'])} paid Instagram impressions ({ig['Impressions']['n']} posts). "
         f"Facebook: SocAPI extra plays = {times(fb['Video plays (starts)']['median'])} paid Facebook plays (20 posts). View-type metrics are 5–110× too small."),
        ("Subtracting paid from the total does not work for one post.",
         f"Paid is a median {pct(R['paid_share_of_public']['median'])} of public views, so a small paid error becomes a large organic error. "
         f"Best case: {pct(best['median_abs_error'])} typical error, {pct(best['negative_organic'])} of posts negative."),
        ("The views before the boost, grown at the normal organic rate, do work.",
         f"{pct(curve['median_abs_error'], 1)} typical error, {pct(curve['within_25pct'])} of posts within ±25%, never negative ({curve['posts']} posts vs opt-in). "
         f"10-post campaign totals: {pct(cl['pre-boost read x organic curve']['within_25pct'])} within ±25%."),
        ("The post-ID tag in the paid table is the best proof that a post is paid.",
         f"Our Meta ad-name rule finds {g['Instagram']['found_by_link_rule_all']} of {g['Instagram']['tagged']} tagged posts. "
         f"The TikTok Spark link finds {g['Tiktok']['found_by_link_rule_tracked']} of {g['Tiktok']['tracked_in_bira']} tagged campaign posts."),
        ("Where no paid record exists, the model flags paid posts well.",
         f"Locked test: TikTok F1 {f2(tt['f1'])}, AUC {f2(tt['roc_auc'])}. Instagram F1 {f2(im['f1'])}, AUC {f2(im['roc_auc'])}, precision {f2(im['precision'])}."),
    ]
    if D["C2"] and D["TG"]:
        P2, T2 = D["C2"]["platforms"], D["TG"]["targets"]
        items[-1] = (("Both platforms now sit on the target lines; Instagram day 14 is the weak spot."
                      if D["C2"].get("run") == "v22" else
                      "The model meets every target on TikTok. Instagram reaches the 90% line once frozen opt-in labels are fixed."
                      if D["C2"].get("run") == "v21" else
                      "The model meets every target on TikTok. On Instagram, a flag is reliable but an organic call is not proof."),
                     f"Locked test, day-30 model v2. TikTok: F1 {f2(P2['Tiktok']['test_metrics']['f1'])}, large boosts caught "
                     f"{pct(T2['C1']['Tiktok']['material_recall'])} at {pct(T2['C1']['Tiktok']['precision'])} precision. "
                     f"Instagram: {pct(T2['C1']['Instagram']['precision'])} of flags are paid, and it catches {pct(T2['C1']['Instagram']['material_recall'], 1)} "
                     "of large boosts (target 90%)." + (" Day-30 model v2.2, labels corrected for frozen opt-in; fourth look at the locked test."
                                                        if D["C2"].get("run") == "v22" else
                                                        " Day-30 model v2.1, labels corrected for frozen opt-in; third look at the locked test."
                                                        if D["C2"].get("run") == "v21" else ""))
    lis = "".join(f'<li><span class="n">{i + 1}</span><span>{esc(t)}{info(tip)}</span></li>' for i, (t, tip) in enumerate(items))
    nxt = [("Tag every boosted ad with the post ID.", "The tag gives an exact match on every platform."),
           ("Wait 7–14 days after publish before a boost.",
            f"A pre-boost read on day 14+ gives {pct(R['production_function_by_confidence']['high']['median_abs_error'], 1)} typical error; before day 7, {pct(R['production_function_by_confidence']['low']['median_abs_error'])}."),
           ("Dark post, or get the creator to opt in, if a boost starts on day 0.",
            f"Today {pct(V['totals']['Instagram']['not_separable'] / V['totals']['Instagram']['posts'])} of paid Instagram posts and "
            f"{pct(V['totals']['Tiktok']['not_separable'] / V['totals']['Tiktok']['posts'])} of paid TikTok posts cannot be separated.")]
    nx = "".join(f'<li><span class="chk" aria-hidden="true">→</span><span>{esc(t)}{info(tip)}</span></li>' for t, tip in nxt)
    caveat = (f'<div class="caveat"><span class="pill warn">Caveat</span><span>TikTok organic after a boost is an estimate, not verified.'
              f'{info("TikTok opt-in views include Spark Ad views, so there is no organic truth after a boost. The method passes a back-test on unpaid TikTok posts only.")}</span></div>')
    return (f'<section id="findings" class="grid"><article class="card span2"><div class="card-head"><h3>What we found</h3></div>'
            f'<ol class="finds">{lis}</ol>{caveat}</article>'
            f'<article class="card"><div class="card-head"><h3>What would make it better</h3></div><ul class="finds next">{nx}</ul></article></section>')


def targets(D):
    """Model v2 against the targets written before the test (docs/IMPROVEMENT_PLAN.md)."""
    TG, C2, C = D["TG"], D["C2"], D["C"]
    if not (TG and C2):
        return ""
    T = TG["targets"]
    ok = lambda b: '<span class="pill ok">met</span>' if b else '<span class="pill warn">missed</span>'
    pfv = lambda k, f: [f(T[k][p]) for p, _ in PLATFORMS]
    c4 = T["C4"]
    c6 = T["C6"]
    rows = [
        ["90%+ of large boosts caught, at 90%+ precision"] + pfv("C1", lambda v: f'{pct(v["material_recall"])} caught, {pct(v["precision"])} precise {ok(v["pass"])}'),
        ["F1 (Instagram 0.80, TikTok 0.93)"] + pfv("C2", lambda v: f'{f3(v["f1"])} {ok(v["pass"])}'),
        ["Calibration error 0.05 or less"] + pfv("C3", lambda v: f'{f3(v["ece"])} {ok(v["pass"])}'),
        ["Fresh posts (Sep 10–24), day-14 model"] + pfv("C5", lambda v: (f'{pct(v["material_recall"])} caught, {pct(v["precision"])} precise '
                                                                       f'({v["n_pos"]} paid posts) '
                                                                       + ('<span class="pill ok">met, small sample</span>' if v["pass"] else ok(False))
                                                                      if "precision" in v else f'no labels {ok(False)}')),
        ["Day-14 F1 within 0.05 of day 30"] + [f'{f2(c4["same_posts"][p]["f1_day14"])} vs {f2(c4["same_posts"][p]["f1_day30"])} '
                                               f'{ok(c4["same_posts"][p]["gap"] <= 0.05)}' for p, _ in PLATFORMS],
    ]
    lst = '<div class="tgts">' + "".join(
        f'<div class="tgt"><div class="tgt-name">{esc(r[0])}</div>'
        + "".join(f'<div class="tgt-v"><span class="tgt-pf">{lab}</span><span>{v}</span></div>' for (p, lab), v in zip(PLATFORMS, r[1:]))
        + "</div>" for r in rows) + "</div>"
    why = ""
    if D.get("AU") and C2.get("run") not in ("v21", "v22"):
        a = D["AU"]["splits"]["locked test"]; b = D["AU"]["splits"]["train (out-of-fold)"]
        mi, ca, og = a["large boost, missed"], a["large boost, caught"], a["organic"]
        why = (f'<div class="caveat"><span class="pill warn">Why Instagram misses</span><span>The {mi["posts"]} missed large boosts look organic: '
               f'{mi["views_per_follower_median"]:.2f} views per follower (organic {og["views_per_follower_median"]:.2f}, caught boosts {ca["views_per_follower_median"]:.1f}). '
               f'Only opt-in says paid.' + info(
                   f'Locked test: SocAPI shows paid plays on {mi["socapi_paid_over_25pct"]} of {mi["with_socapi"]} missed boosts, '
                   f'{ca["socapi_paid_over_25pct"]} of {ca["with_socapi"]} caught boosts and {og["socapi_paid_over_25pct"]} of {og["with_socapi"]} organic posts. '
                   f'Train: {b["large boost, missed"]["socapi_paid_over_25pct"]} of {b["large boost, missed"]["with_socapi"]} missed boosts. '
                   "So some misses are real boosts that public data cannot see, and some opt-in labels are doubtful. Source: results/ig_miss_audit.json.")
               + '</span></div>')
    prod = C2.get("run") in ("v21", "v22")
    swing = ""
    if D.get("MF"):
        mf = D["MF"]["summary"]
        swing = (f'<div class="caveat"><span class="pill warn">Read the two misses</span><span>'
                 f'TikTok F1 is 0.003 under its line, while it swings {f2(mf["Tiktok_h30"]["f1_min"])}–{f2(mf["Tiktok_h30"]["f1_max"])} '
                 f'from month to month. Instagram day 14 is weaker by design: its “organic” calls are marked provisional until day 30.'
                 + info(f'Month-by-month check inside the training period (train on earlier posts, score the next month, Mar–Jun 2026): '
                        f'Instagram day 14 F1 {f2(mf["Instagram_h14"]["f1_min"])}–{f2(mf["Instagram_h14"]["f1_max"])}, day 30 '
                        f'{f2(mf["Instagram_h30"]["f1_min"])}–{f2(mf["Instagram_h30"]["f1_max"])}; TikTok day 14 '
                        f'{f2(mf["Tiktok_h14"]["f1_min"])}–{f2(mf["Tiktok_h14"]["f1_max"])}. More training data did not raise F1. '
                        "Next clean test: posts published after 2026-09-24, from about 2026-10-21. Source: results/month_folds.json.")
                 + '</span></div>')
    score = card("Did the model reach its targets?",
                 lst + why + swing + f'<p class="mini">Coverage: {pct(c4["coverage"]["all"], 1)} of posts at least 14 days old and tracked by day 7 get a score {ok(c4["coverage"]["all"] >= 0.85)}. '
                       f'Tagged paid posts flagged: {c6["posts_flagged_at_every_horizon"]} of {c6["distinct_posts"]} {ok(c6["pass"])}.</p>',
                 cls="span2", sub=(("Model v2.2" if C2.get("run") == "v22" else "TikTok v2, Instagram v2.1")
                                   + ", labels corrected for frozen opt-in. Locked test (posts 2026-07-01 to 09-09) and fresh posts"
                                   if prod else "Locked test (posts published 2026-07-01 to 09-09) and fresh posts, each scored once"),
                 tip="Large boost = opt-in shows 40% or more paid, or an ad link or post-ID tag inside the model window (all TikTok boosts). Targets and rules: docs/IMPROVEMENT_PLAN.md.",
                 data=table(["Target", "Platform", "Value", "95% interval"],
                            [[k, lab, f3(T[k][p].get("material_recall", T[k][p].get("f1", T[k][p].get("ece", float("nan"))))),
                              "–".join(f3(x) for x in T[k][p]["ci95"]["material_recall"]) if isinstance(T[k][p].get("ci95"), dict) else
                              ("–".join(f3(x) for x in T[k][p]["ci95"]) if isinstance(T[k][p].get("ci95"), list) else "")]
                             for k in ("C1", "C2", "C3", "C5") for p, lab in PLATFORMS if p in T[k]]))
    vv = C2["v1_vs_v2"]
    if prod and D.get("TG0") and D.get("RL") and D.get("ST"):
        a0 = D["TG0"]["targets"]["C1"]["Instagram"]
        a1 = D["RL"]["models"]["Instagram_h30_test"]
        a2 = T["C1"]["Instagram"]
        st = D["ST"]
        stress = a1.get("stress_socapi_paid_added_back", {})
        bars = progress([("v2, labels as pulled", a0["material_recall"], "hatch-n",
                          f'Instagram day 30, labels as pulled: {pct(a0["material_recall"], 1)} caught at {pct(a0["precision"], 1)} precision (first look, target missed)'),
                         ("v2, corrected labels", a1["material_recall"], "hatch-n",
                          f'Same model, labels corrected: {pct(a1["material_recall"], 1)} caught at {pct(a1["precision"], 1)} precision'),
                         *([("v2.1, corrected labels", D["TG1"]["targets"]["C1"]["Instagram"]["material_recall"], "hatch-n",
                             f'Retrained on corrected labels: {pct(D["TG1"]["targets"]["C1"]["Instagram"]["material_recall"], 1)} caught at '
                             f'{pct(D["TG1"]["targets"]["C1"]["Instagram"]["precision"], 1)} precision')] if C2.get("run") == "v22" and D.get("TG1") else []),
                         (("v2.2, followers fixed (used)" if C2.get("run") == "v22" else "v2.1, corrected labels (used)"), a2["material_recall"], "solid",
                          f'In use: {pct(a2["material_recall"], 1)} caught at {pct(a2["precision"], 1)} precision; '
                          f'95% interval {pct(a2["ci95"]["material_recall"][0])}–{pct(a2["ci95"]["material_recall"][1])}')])
        comp = card("Instagram: the labels were the gap", bars
                    + f'<p class="mini">Opt-in stopped updating on {st["stale_at_latest_read"]:,} of {st["instagram_posts_2025_with_optin"]:,} Instagram posts with opt-in. '
                      f'Public views kept growing, so the post looked paid.</p>',
                    sub="Large boosts caught, day-30 model, locked test",
                    tip=(f'Posts whose paid label came only after opt-in froze get no label (day 30: {st["label_changes_instagram"]["day_30"]["paid_to_no_label"]} posts). '
                         f'SocAPI shows paid plays on {st["socapi_check_day30"]["locked_test"]["dropped"]["socapi_paid"]} of '
                         f'{st["socapi_check_day30"]["locked_test"]["dropped"]["with_socapi"]} of them on the test set, vs '
                         f'{st["socapi_check_day30"]["locked_test"]["kept_paid"]["socapi_paid"]} of {st["socapi_check_day30"]["locked_test"]["kept_paid"]["with_socapi"]} kept paid posts. '
                         + (f'If those SocAPI-paid posts count as paid, v2 catches {pct(stress["material_recall"], 1)}. ' if stress else "")
                         + "Rule chosen on train (plan amendment 4). v2.2 also uses follower counts known at the horizon (amendment 5); "
                         + "fourth look at the locked test."),
                    data=table(["Platform", "Metric", "v1 (labels as pulled)", "v2.1 / v2 (corrected labels)"],
                               [[lab, k.replace("_", " "), f3(vv[p][k]["v1"]), f3(vv[p][k]["v2"])] for p, lab in PLATFORMS
                                for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "ece")]))
    else:
        bars = progress([(f"{lab} {v}", vv[p]["f1"][v], "solid" if v == "v2" else "hatch-n", f"{lab}, model {v}: F1 {vv[p]['f1'][v]:.3f} on the locked test")
                         for p, lab in PLATFORMS for v in ("v1", "v2")], fmt=f2)
        comp = card("v1 → v2", bars, sub="F1 on the locked test",
                    tip="v2 uses more public reads (views and likes at many ages, the creator's own norms) and adds post-ID tags to the labels, so the label sets differ a little.",
                    data=table(["Platform", "Metric", "v1", "v2"],
                               [[lab, k.replace("_", " "), f3(vv[p][k]["v1"]), f3(vv[p][k]["v2"])] for p, lab in PLATFORMS for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "ece")]))
    hz = []
    for blk, name in (("day14", "Day 14"), ("platforms", "Day 30"), ("day60", "Day 60")):
        for p, lab in PLATFORMS:
            if p in C2.get(blk, {}):
                t = C2[blk][p]["test_metrics"]
                hz.append([f"{name}, {lab}", f'{t["n"]:,} ({t["n_pos"]})', f2(t["precision"]), f2(t["recall"]), f2(t["f1"])])
    horiz = card("Score early, check later", table(["Model", "Posts (paid)", "Precision", "Recall", "F1"], hz), cls="span3",
                 sub="A post gets the longest model its data allows",
                 tip="Day 14: needs a public read on day 12–14 and a first read by day 10. Day 30: a read on day 28 or later. Day 60: a read on day 55 or later; it sees boosts that start late. Day 60 was added after the targets were set, so it has no target.")
    return f'<section id="targets" class="grid">{score}{comp}{horiz}</section>'


def model(D):
    C, W = D["C2"] or D["C"], D["W"]
    P = C["platforms"]
    seg = ('<div class="seg-ctl" role="group" aria-label="Platform">' + "".join(
        f'<button type="button" data-pf-btn="{p}" aria-pressed="{"true" if p == DEFAULT_PF else "false"}">{lab}</button>' for p, lab in PLATFORMS) + "</div>")

    def tiles(p, lab):
        t, ci = P[p]["test_metrics"], P[p]["test_ci95"]
        return '<div class="tiles">' + "".join(
            f'<div class="tile"><span class="tl">{n}</span><span class="tv">{t[k]:.2f}</span><span class="tc">95% CI {ci[k][0]:.2f}–{ci[k][1]:.2f}</span></div>'
            for n, k in [("ROC AUC", "roc_auc"), ("PR AUC", "pr_auc"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"), ("MCC", "mcc")]) + "</div>"

    def cmx(p, lab):
        m = P[p]["test_metrics"]
        pos, neg = m["tp"] + m["fn"], m["fp"] + m["tn"]

        def cell(v, tot, kind, of):
            q = {"true positive": "cm-a", "true negative": "cm-k"}.get(kind, "cm-l")
            return f'<div class="cm {q}" tabindex="0" data-tip="{esc(f"{lab}: {kind} {v} = {pct(v / tot, 1)} of {of}")}"><b>{v:,}</b><span>{kind}</span></div>'
        return (f'<div class="cmx"><span></span><span class="cmh">Model: paid</span><span class="cmh">Model: organic</span>'
                f'<span class="cmr">Label: paid ({pos})</span>{cell(m["tp"], pos, "true positive", "paid posts")}{cell(m["fn"], pos, "false negative", "paid posts")}'
                f'<span class="cmr">Label: organic ({neg})</span>{cell(m["fp"], neg, "false positive", "organic posts")}{cell(m["tn"], neg, "true negative", "organic posts")}</div>')

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
        t = P[p]["test_metrics"]
        llm = [r for r in D["llm"] if r["platform"] == p and not r["detector"].startswith("ML")]
        bestllm = max(llm, key=lambda r: float(r["f1"]))
        rows = [("This model", t["f1"], "solid"), ("Hand rule: views > followers, engagement < 1%", P[p]["rules_test"][HAND_RULE]["f1"], "hatch-n"),
                ("Views > followers", P[p]["rules_test"]["views > followers"]["f1"], "hatch-n"),
                (f"Best LLM ({bestllm['detector'].replace('LLM ', '')}, 150 posts)", float(bestllm["f1"]), "hatch-n")]
        return progress([(n, v, c, f"{lab}: {n}: F1 {v:.2f}") for n, v, c in rows], fmt=f2)

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
    sweep_rows = [[lab, f'{r["threshold"]:.2f}', pct(r["precision"], 1), pct(r["recall"], 1), pct(r["f1"], 1)] for p, lab in PLATFORMS for r in P[p]["threshold_sweep_test"]]
    cards = [
        card("Confusion matrix", both(cmx), sub="Locked test, threshold chosen on train",
             tip="Test posts published 2026-07-01 to 2026-09-09, scored once. Rows = true label, columns = model flag.",
             data=table(["Platform", "TP", "FP", "FN", "TN"], [[lab] + [P[p]["test_metrics"][k] for k in ("tp", "fp", "fn", "tn")] for p, lab in PLATFORMS])),
        card("ROC curve", both(roc) + legend([("c-acc", "Test"), ("c-n3", "Train, cross-validated")]),
             sub="Test AUC vs train cross-validated: " + ", ".join(f'{lab} {f2(P[p]["test_metrics"]["roc_auc"])} vs {f2(P[p]["train_cv_metrics"]["roc_auc"])}' for p, lab in PLATFORMS),
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
                 f'{lab}: the stricter cut-off {o["threshold"]:.2f} (precision 95% on train) gives {pct(o["test"]["precision"])} precision at {pct(o["test"]["recall"])} recall on test.'
                 for p, lab in PLATFORMS for o in P[p]["operating_points"] if o["rule"].startswith("precision >=")),
             data=table(["Platform", "Threshold", "Precision", "Recall", "F1"], sweep_rows)),
        card("Cumulative gains", both(gains), sub="Check the highest scores first",
             tip="Dashed line = random order.",
             data=table(["Platform", "Top share", "Paid found"], [[lab, pct(r["top_share"]), pct(r["boosted_captured"], 1)] for p, lab in PLATFORMS for r in P[p]["curves_test"]["gains"]])),
        card("Against rules and LLMs", both(baselines), sub="F1 on the locked test",
             tip="LLMs: GPT-5, Claude Sonnet 4.5 and Llama 3.1 70B in Snowflake Cortex, same public numbers, threshold 0.5, 150 test posts per platform.",
             data=table(["Platform", "Detector", "AUC", "Precision", "Recall", "F1"],
                        [[("Instagram" if r["platform"] == "Instagram" else "TikTok"), esc(r["detector"].replace("LLM ", "")), f3(float(r["roc_auc"])),
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
            f'<p class="card-sub">{("Day-30 model v2.2. " if D["C2"].get("run") == "v22" else "Day-30 model v2. ") if D["C2"] else ""}Only for posts with no paid record. Locked test of later posts.</p></div>{seg}</div>'
            f'{pf_panels(tiles)}<div class="grid">{"".join(cards)}</div>{note}</section>')


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
                "MEASURED_OPTIN_GAP": "Opt-in below 80% of public", "MEASURED_SOCAPI_GAP": "SocAPI shows paid Facebook plays",
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
         "<p>It depends on how late the pre-boost read is.</p>"
         + table(["Pre-boost read", "Posts", "Typical error", "Within ±25%"],
                 [[n, pf_[k]["posts"], pct(pf_[k]["median_abs_error"], 1), pct(pf_[k]["within_25pct"])] for k, n in
                  [("high", "Day 14 or later"), ("medium", "Day 7–13"), ("low", "Before day 7")]])
         + "<p>Back-test on unpaid posts, creator split (predict day 30):</p>"
         + table(["Read", "Posts", "Typical error", "Within ±25%"],
                 [[f"{lab}, day {d}", f'{bt[p][f"read on day {d} -> predict day 30"]["posts"]:,}', pct(bt[p][f"read on day {d} -> predict day 30"]["median_abs_error"], 1),
                   pct(bt[p][f"read on day {d} -> predict day 30"]["within_25pct"])] for p, lab in [("Instagram", "Instagram"), ("Tiktok", "TikTok")] for d in (3, 7, 14)])),
        ("What about TikTok and YouTube?",
         f"<p>TikTok public views rise {times(T['tiktok_metric_match']['Video plays (starts)']['median'])} the paid plays ({T['tiktok_posts']} posts). "
         f"But TikTok opt-in equals public (ratio {f3(T['tiktok_optin_equals_public']['median'])}), so there is no organic truth after a boost. "
         f"Subtraction gives {times(tv['median'])} the curve estimate, and we cannot check which is right.</p>"
         "<p>YouTube has no ad-to-video link in the warehouse (0 of 34,687 posts), so it cannot be tested.</p>"),
        ("How do we know a post is paid?",
         "<p>Each post gets its strongest proof. The model only scores posts with none.</p>"
         + table(["Proof (strongest first)", "Instagram", "TikTok"],
                 [[tier_lab[k], f'{tiers["Instagram"].get(k, 0):,}', (f'{tiers["Tiktok"][k]:,}' if k in tiers["Tiktok"] else "n/a")] for k in tier_lab])
         + f"<p>Gold check: our Meta ad-name rule finds {g['Instagram']['found_by_link_rule_all']} of {g['Instagram']['tagged']} tagged posts. "
           f"The TikTok Spark link finds {g['Tiktok']['found_by_link_rule_tracked']} of {g['Tiktok']['tracked_in_bira']} tagged campaign posts, so the tag goes first. "
           "One TikTok test “false positive” is a tagged paid post: the label was wrong, the model was right.</p>"),
        ("How was the model kept honest?",
         ("<ul><li>No paid data in the inputs: only public views, likes, comments, shares and followers up to the model day (14, 30 or 60), "
          "and the same creator's earlier posts.</li>"
          "<li>Targets written down before any test (docs/IMPROVEMENT_PLAN.md). Every change was chosen on train only and logged there first.</li>"
          f"<li>Locked test (posts {C['test_cutoff']} to {C.get('test_end', '2026-09-09')}): v1 was scored on it once, v2 once more (second look). "
          f"Fresh posts ({C['fresh'][0]} to {C['fresh'][1]}) were scored once.</li>" if D["C2"] else
          "<ul><li>No paid data in the inputs: only public views, likes, comments, shares and followers from days 0–30.</li>"
          f"<li>Locked test: trained on posts before {C['test_cutoff']}, tested once on later posts. Re-running the code gives the same scores.</li>")
         + "<li>Model and threshold chosen on train only, with cross-validation grouped by creator.</li>"
         f"<li>95% intervals resample whole creators ({C['n_boot']:,} draws).</li>"
         "<li>Labels miss some paid posts, so test precision is a floor.</li></ul>"),
        ("All model metrics", "".join(full(p, lab) for p, lab in PLATFORMS)),
        ("Thresholds for a stricter or looser flag",
         table(["Platform", "Rule (chosen on train)", "Threshold", "Precision", "Recall", "F1"],
               [[lab, esc(o["rule"]), f'{o["threshold"]:.2f}', pct(o["test"]["precision"], 1), pct(o["test"]["recall"], 1), pct(o["test"]["f1"], 1)]
                for p, lab in PLATFORMS for o in P[p]["operating_points"]])),
        ("Slices and model choice",
         table(["Platform", "Slice", "Posts (paid)", "AUC", "F1"],
               [[lab, esc(k), f'{v["n"]} ({v["n_pos"]})', f3(v["roc_auc"]), f3(v["f1"])] for p, lab in PLATFORMS for k, v in P[p]["slices_test"].items()]
               + [[lab, "Client holdout (train CV)", f'{P[p]["train"]["n"]:,}', f3(P[p]["overfit_check"]["train_cv_by_client_roc_auc"]), ""] for p, lab in PLATFORMS])
         + table(["Platform", "Candidate (train CV)", "AUC", "PR AUC"],
                 [[lab, esc(r["model"].replace("_", " ")), f3(r["cv_roc_auc"]), f3(r["cv_pr_auc"])] for p, lab in PLATFORMS for r in P[p]["selection_table"]],
                 hl=lambda r: any(r[1] == f'{P[p]["selected_model"].split(" + isotonic")[0]} [{P[p].get("feature_set", "")}]'.replace("_", " ")
                                  or r[1] == P[p]["selected_model"].replace("_", " ") for p, _ in PLATFORMS))),
        ("What does the daily table do?",
         "<p>One row per post: proof tier, paid status, model score (only with no proof), and organic views with method and confidence. "
         "The pipeline and its tests are in the repo. It is not scheduled yet: it needs a Snowflake service account and a schema to write to.</p>"),
        ("Sources and terms",
         "<p>Organic truth: BIRA observation time series (public = Nimble, opt-in = private API). Paid: unified paid table and EDW Meta and TikTok ad tables. "
         "Data read 2026-10-07. Typical = median. Middle half = 25th–75th percentile. Error = (estimate − opt-in) ÷ opt-in.</p>"),
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
.card-head h3 { font-size: 21px; display: flex; align-items: center; gap: 8px; }
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
.tgts { display: grid; gap: 0; }
.tgt { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr) minmax(0, 1fr); gap: 12px; align-items: baseline; padding: 11px 0; border-bottom: 1px solid var(--rule-2); font-size: 14px; }
.tgt:last-child { border-bottom: 0; }
.tgt-name { color: var(--ink-2); }
.tgt-v { display: flex; flex-wrap: wrap; gap: 4px 8px; align-items: baseline; font-variant-numeric: tabular-nums; }
.tgt-pf { color: var(--faint); font-size: 12px; min-width: 64px; }
.mini { font-size: 12.5px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
.pbars { display: grid; gap: 14px; }
.pb { display: grid; gap: 6px; }
.pb-top { display: flex; justify-content: space-between; gap: 10px; font-size: 14px; color: var(--ink-2); }
.pb-top b { color: var(--ink); font-weight: 500; font-variant-numeric: tabular-nums; }
.pb-track { height: 14px; border-radius: 999px; background: var(--rule-2); overflow: hidden; }
.pb-fill { height: 100%; border-radius: 999px; min-width: 6px; }
/* dot matrix: accent = measured, neutral steps = estimate confidence, ring = not separable */
.dm { display: grid; grid-template-columns: 92px minmax(0, 1fr); gap: 12px; align-items: center; }
.dm-lab { font-size: 14px; font-weight: 500; display: grid; } .dm-lab small { color: var(--muted); font-weight: 400; font-size: 12px; }
.dm-grid { display: grid; grid-template-columns: repeat(20, minmax(0, 1fr)); gap: 3px; max-width: 260px; }
.dt { aspect-ratio: 1; border-radius: 50%; display: block; }
.m { background: var(--accent); } .h { background: var(--n5); } .md { background: var(--n4); } .l { background: var(--n2); }
.x { background: transparent; box-shadow: inset 0 0 0 1.5px var(--n3); }
.sw.m, .sw.h, .sw.md, .sw.l, .sw.x { border-radius: 50%; }
/* inverse card */
.inv { justify-content: space-between; min-height: 300px; background: var(--inv); color: var(--inv-ink); border: 0; overflow: hidden; isolation: isolate; }
.inv::after { content: ""; position: absolute; inset: 0; z-index: -1; opacity: .18; mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E"); }
.inv::before { content: ""; position: absolute; right: -60px; bottom: -60px; width: 220px; height: 220px; border-radius: 50%; z-index: -1;
  background: radial-gradient(circle, color-mix(in srgb, var(--accent) 45%, transparent), transparent 70%); }
.pill.glass { color: var(--inv-ink); background: rgba(255,255,255,.1); border-color: rgba(255,255,255,.2); align-self: flex-start; box-shadow: none; }
.inv-num { font-size: clamp(72px, 9vw, 112px); line-height: .9; letter-spacing: -.05em; font-weight: 500; }
.inv-sub { font-size: 14px; color: var(--inv-2); max-width: 30ch; }
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
.qa-a p, .qa-a ul { max-width: 72ch; } .qa-a ul { margin: 0; padding-left: 20px; display: grid; gap: 6px; }
footer { color: var(--muted); font-size: 13px; display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap; border-top: 1px solid var(--rule); padding-top: 18px; }
#tip { position: fixed; z-index: 20; max-width: 320px; background: var(--glass); backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--glass-line); color: var(--ink); font-size: 13px; line-height: 1.45; padding: 8px 12px; border-radius: 14px;
  box-shadow: 0 12px 28px -12px rgba(17,18,20,.35); pointer-events: none; }
@media (max-width: 1080px) {
  .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .span2 { grid-column: 1 / -1; }
  .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); }
}
@media (max-width: 720px) {
  .tgt { grid-template-columns: minmax(0, 1fr); gap: 4px; }
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
  .brand { font-size: 22px; }
  .top-right { width: 100%; justify-content: space-between; flex-wrap: nowrap; }
  .top { min-width: 0; width: 100%; }
  nav.pills { min-width: 0; flex: 1 1 0; }
  nav.pills a { padding: 8px 11px; font-size: 14px; }
  .seg-ctl.theme button span { display: none; }
}
@media (min-width: 721px) { .pill.mobile { display: none; } }
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
  var links = document.querySelectorAll('nav.pills a');
  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (en) {
        if (!en.isIntersecting) return;
        links.forEach(function (a) { a.classList.toggle('on', a.getAttribute('href') === '#' + en.target.id); });
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    ['overview', 'findings', 'targets', 'model', 'faq'].forEach(function (id) { var s = document.getElementById(id); if (s) io.observe(s); });
  }
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
      <nav class="pills" aria-label="Sections"><a class="on" href="#overview">Overview</a><a href="#findings">Findings</a>{'<a href="#targets">Targets</a>' if D["TG"] else ""}<a href="#model">Model</a><a href="#faq">FAQ</a></nav>
      <div class="seg-ctl theme" role="group" aria-label="Color theme"><button type="button" data-theme-btn="light" aria-pressed="true" aria-label="Light mode">{SUN}<span>Light</span></button><button type="button" data-theme-btn="dark" aria-pressed="false" aria-label="Dark mode">{MOON}<span>Dark</span></button></div>
    </div>
  </div>
  <div class="title-row">
    <h1>Paid vs Organic</h1>
    <div class="dates"><span>{CAL}{R['panel']['obs_from']} – {R['panel']['obs_to']}</span><span class="vs">model test</span><span>{CAL}2026-07-01 – 2026-09-09</span></div>
  </div>
  {overview(D)}
  {findings(D)}
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
