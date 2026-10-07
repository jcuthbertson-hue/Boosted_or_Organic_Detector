"""Build reports/boost_report.html: Tom's subtraction question + every standard classification output.

Reads aggregate result files only (no row-level data, no client names):
  results/reconciliation.json, results/reconciliation_tiktok.json, results/classification_metrics.json,
  results/organic_coverage.json, results/llm_vs_ml.csv, results/llm_vs_ml_paired_auc.csv
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
PLATFORMS = [("Tiktok", "TikTok"), ("Instagram", "Instagram")]


def esc(s):
    return html.escape(str(s), quote=True)


def pct(x, d=0):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def num(x):
    return f"{x:,.0f}"


def times(x):
    return f"{x:.2f}×" if x < 10 else f"{x:.0f}×"


def load():
    j = lambda p: json.load(open(p))
    llm = list(csv.DictReader(open("results/llm_vs_ml.csv")))
    paired = list(csv.DictReader(open("results/llm_vs_ml_paired_auc.csv")))
    return (j("results/reconciliation.json"), j("results/reconciliation_tiktok.json"),
            j("results/classification_metrics.json"), j("results/organic_coverage.json"), llm, paired)


# ---------------------------------------------------------------- chart helpers
def lin(d0, d1, r0, r1):
    return lambda v: r0 + (v - d0) * (r1 - r0) / (d1 - d0)


def logpos(v, lo, hi):
    """position 0-100 (%) of v on a log axis lo..hi"""
    v = min(max(v, lo), hi)
    return 100 * (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))


def legend(items):
    return ('<div class="legend">' + "".join(
        f'<span class="key"><i class="sw {cls}"></i>{esc(name)}</span>' for cls, name in items) + "</div>")


def table_view(head, rows, caption="Show data as a table"):
    th = "".join(f'<th class="{"r" if i else ""}">{esc(h)}</th>' for i, h in enumerate(head))
    tr = "".join("<tr>" + "".join(f'<td class="{"r" if i else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return (f'<details class="tv"><summary>{esc(caption)}</summary><div class="table-wrap"><table>'
            f"<thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div></details>")


def xy_chart(series, *, label, xdom=(0, 1), ydom=(0, 1), xticks=None, yticks=None, xfmt=f2, yfmt=f2,
             xlabel="", ylabel="", diag=False, vline=None, w=440, h=300):
    m = dict(t=14, r=18, b=46, l=50)
    X = lin(xdom[0], xdom[1], m["l"], w - m["r"])
    Y = lin(ydom[0], ydom[1], h - m["b"], m["t"])
    xticks = xticks or [0, .25, .5, .75, 1]
    yticks = yticks or [0, .25, .5, .75, 1]
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(label)}">']
    for t in yticks:
        s.append(f'<line class="grid" x1="{m["l"]}" x2="{w - m["r"]}" y1="{Y(t):.1f}" y2="{Y(t):.1f}"/>'
                 f'<text class="tick" x="{m["l"] - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{esc(yfmt(t))}</text>')
    for t in xticks:
        s.append(f'<text class="tick" x="{X(t):.1f}" y="{h - m["b"] + 18}" text-anchor="middle">{esc(xfmt(t))}</text>')
    s.append(f'<line class="axis" x1="{m["l"]}" x2="{w - m["r"]}" y1="{h - m["b"]}" y2="{h - m["b"]}"/>')
    s.append(f'<text class="axlab" x="{(m["l"] + w - m["r"]) / 2:.1f}" y="{h - 8}" text-anchor="middle">{esc(xlabel)}</text>')
    s.append(f'<text class="axlab" transform="translate(13 {(m["t"] + h - m["b"]) / 2:.1f}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>')
    if diag:
        s.append(f'<line class="ref" x1="{X(xdom[0]):.1f}" y1="{Y(ydom[0]):.1f}" x2="{X(xdom[1]):.1f}" y2="{Y(ydom[1]):.1f}"/>')
    if vline is not None:
        v, txt = vline
        s.append(f'<line class="ref" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{m["t"]}" y2="{h - m["b"]}"/>'
                 f'<text class="tick" x="{X(v) + 5:.1f}" y="{m["t"] + 11}">{esc(txt)}</text>')
    for se in series:
        pts = se["pts"]
        d = "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
        s.append(f'<path class="ln {se["cls"]}" d="{d}"/>')
        if se.get("dots"):
            for x, y in pts:
                s.append(f'<circle class="dot {se["cls"]}" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4.5"/>')
    for se in series:   # hit targets on top
        pts, tip = se["pts"], se.get("tip")
        if not tip:
            continue
        step = max(1, len(pts) // 45)
        for i, (x, y) in enumerate(pts):
            if i % step and i != len(pts) - 1:
                continue
            s.append(f'<circle class="hit" cx="{X(x):.1f}" cy="{Y(y):.1f}" r="8" data-tip="{esc(tip(x, y, i))}"/>')
    s.append("</svg>")
    return "".join(s)


def rounded_bar(x, y_base, y_end, w, r=4):
    """vertical bar from y_base to y_end; 4px round at the data end, square at the baseline"""
    up = y_end < y_base
    hgt = abs(y_base - y_end)
    r = min(r, hgt, w / 2)
    if hgt < 0.5:
        return ""
    if up:
        return (f"M{x:.1f},{y_base:.1f} V{y_end + r:.1f} Q{x:.1f},{y_end:.1f} {x + r:.1f},{y_end:.1f} "
                f"H{x + w - r:.1f} Q{x + w:.1f},{y_end:.1f} {x + w:.1f},{y_end + r:.1f} V{y_base:.1f} Z")
    return (f"M{x:.1f},{y_base:.1f} V{y_end - r:.1f} Q{x:.1f},{y_end:.1f} {x + r:.1f},{y_end:.1f} "
            f"H{x + w - r:.1f} Q{x + w:.1f},{y_end:.1f} {x + w:.1f},{y_end - r:.1f} V{y_base:.1f} Z")


def mirror_hist(hist, thr, label, w=440, h=300):
    edges, pos, neg = hist["edges"], hist["boosted"], hist["organic"]
    sp, sn = sum(pos), sum(neg)
    fp = [v / sp for v in pos]
    fn = [v / sn for v in neg]
    top = max(max(fp), max(fn))
    top = math.ceil(top * 10) / 10
    m = dict(t=14, r=18, b=46, l=50)
    X = lin(0, 1, m["l"], w - m["r"])
    mid = (m["t"] + h - m["b"]) / 2
    Y = lin(0, top, mid, m["t"])
    Yd = lin(0, top, mid, h - m["b"])
    s = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="{esc(label)}">']
    for t in [top, top / 2]:
        for yy in (Y(t), Yd(t)):
            s.append(f'<line class="grid" x1="{m["l"]}" x2="{w - m["r"]}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        s.append(f'<text class="tick" x="{m["l"] - 8}" y="{Y(t) + 4:.1f}" text-anchor="end">{pct(t)}</text>')
        s.append(f'<text class="tick" x="{m["l"] - 8}" y="{Yd(t) + 4:.1f}" text-anchor="end">{pct(t)}</text>')
    bw = (X(edges[1]) - X(edges[0])) - 2
    for i in range(len(pos)):
        x0 = X(edges[i]) + 1
        rng = f"score {edges[i]:.2f}–{edges[i + 1]:.2f}"
        dp = rounded_bar(x0, mid, Y(fp[i]), bw)
        dn = rounded_bar(x0, mid, Yd(fn[i]), bw)
        if dp:
            s.append(f'<path class="bar s-blue" d="{dp}" data-tip="Boosted posts, {rng}: {pos[i]} ({pct(fp[i])} of boosted)"/>')
        if dn:
            s.append(f'<path class="bar s-aqua" d="{dn}" data-tip="Organic posts, {rng}: {neg[i]} ({pct(fn[i])} of organic)"/>')
    s.append(f'<line class="axis" x1="{m["l"]}" x2="{w - m["r"]}" y1="{mid}" y2="{mid}"/>')
    s.append(f'<line class="ref" x1="{X(thr):.1f}" x2="{X(thr):.1f}" y1="{m["t"]}" y2="{h - m["b"]}"/>'
             f'<text class="tick" x="{X(thr) + 5:.1f}" y="{m["t"] + 11}">threshold {thr:.2f}</text>')
    for t in [0, .25, .5, .75, 1]:
        s.append(f'<text class="tick" x="{X(t):.1f}" y="{h - m["b"] + 18}" text-anchor="middle">{t:.2f}</text>')
    s.append(f'<text class="axlab" x="{(m["l"] + w - m["r"]) / 2:.1f}" y="{h - 8}" text-anchor="middle">Model score (0 = organic, 1 = boosted)</text>')
    s.append(f'<text class="axlab" transform="translate(13 {mid:.1f}) rotate(-90)" text-anchor="middle">Share of class</text>')
    s.append(f'<text class="tick" x="{w - m["r"]}" y="{m["t"] + 11}" text-anchor="end">boosted ↑</text>'
             f'<text class="tick" x="{w - m["r"]}" y="{h - m["b"] - 6}" text-anchor="end">organic ↓</text>')
    s.append("</svg>")
    return "".join(s)


def hbars(rows, *, vmax, fmt, cls="s-blue"):
    """rows: (label, value, tip, highlight). HTML bars so text stays readable at phone width."""
    out = ['<div class="hb">']
    for label, v, tip, hl in rows:
        w = max(0.6, 100 * min(v, vmax) / vmax)
        c = cls if hl is None else ("s-blue" if hl else "s-mute")
        out.append(f'<div class="hb-row"><div class="hb-label">{esc(label)}</div>'
                   f'<div class="hb-track"><div class="hb-bar {c}" style="width:{w:.1f}%" tabindex="0" data-tip="{esc(tip)}"></div></div>'
                   f'<div class="hb-val num">{esc(fmt(v))}</div></div>')
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------- sections
def section_math(R, T, V):
    mm = R["metric_match"]
    ig, fb = mm["instagram_side"], mm["facebook_side"]
    rec = R["recipes"]
    lo, hi = 0.5, 1000
    ticks = [0.5, 1, 2, 5, 10, 100, 1000]
    rows = []
    for name in ig:
        a, b = ig[name], fb.get(name, {"n": 0})

        def mark(r, cls, side, where):
            if not r.get("n"):
                return ""
            l, m_, h = logpos(r["q25"], lo, hi), logpos(r["median"], lo, hi), logpos(r["q75"], lo, hi)
            tip = (f"{name}, {side}: typical {times(r['median'])} (middle half {times(r['q25'])}–{times(r['q75'])}), "
                   f"{r['n']} reads")
            return (f'<span class="dr-whisk {where}" style="left:{l:.2f}%;width:{max(h - l, .3):.2f}%"></span>'
                    f'<span class="dr-dot {cls} {where}" style="left:{m_:.2f}%" tabindex="0" data-tip="{esc(tip)}"></span>')
        rows.append(f'<div class="dr-row"><div class="dr-label">{esc(name)}</div><div class="dr-track">'
                    + "".join(f'<span class="dr-grid" style="left:{logpos(t, lo, hi):.2f}%"></span>' for t in ticks)
                    + f'<span class="dr-ref" style="left:{logpos(1, lo, hi):.2f}%"></span>'
                    + mark(a, "s-blue", "Instagram placement", "top") + mark(b, "s-orange", "Facebook placement", "bot")
                    + "</div></div>")
    axis = ('<div class="dr-row dr-axis"><div class="dr-label"></div><div class="dr-track">'
            + "".join(f'<span class="dr-tick" style="left:{logpos(t, lo, hi):.2f}%">{t:g}×</span>' for t in ticks) + "</div></div>")
    dot_tbl = table_view(["Paid metric", "Instagram side: typical (middle half)", "reads", "Facebook side: typical (middle half)", "reads"],
                         [[esc(n), f"{times(ig[n]['median'])} ({times(ig[n]['q25'])}–{times(ig[n]['q75'])})", ig[n]["n"],
                           (f"{times(fb[n]['median'])} ({times(fb[n]['q25'])}–{times(fb[n]['q75'])})" if fb[n]["n"] else "no data"), fb[n]["n"]]
                          for n in ig])

    best = rec["public - paid IG Impressions"]
    curve = rec["pre-boost read x organic curve (no fitting on these posts)"]
    eb = R["error_by_paid_share"]
    eb_rows = [(f"Paid {r['paid_share']} of public ({r['posts']} posts)", r["median_abs_error"],
                f"Paid share {r['paid_share']}: typical organic error {pct(r['median_abs_error'])}, {r['posts']} posts", None) for r in eb]

    order = [("public - paid IG Impressions", "Public − paid IG impressions"),
             ("public - k x paid IG impressions (k fitted on other posts)", "Public − 1.07 × paid IG impressions (factor fitted on other posts)"),
             ("public - paid IG Video plays (starts)", "Public − paid IG video plays"),
             ("public - paid IG 3-second video views", "Public − paid IG 3-second views"),
             ("public - paid IG ThruPlays", "Public − paid IG ThruPlays"),
             ("SocAPI total - FB cross-post - IG impressions - FB video plays (best mix)", "SocAPI total − FB cross-post − IG impressions − FB plays"),
             ("pre-boost public read (no adjustment)", "Pre-boost public read, no growth"),
             ("pre-boost read x flat growth (fitted on other posts)", "Pre-boost read × one growth factor (fitted on other posts)"),
             ("pre-boost read x organic curve (no fitting on these posts)", "Pre-boost read × organic curve (recommended)")]
    w25 = [(lab, rec[k]["within_25pct"], f"{lab}: {pct(rec[k]['within_25pct'])} of {rec[k]['posts']} posts within ±25%; typical error {pct(rec[k]['median_abs_error'])}",
            k.startswith("pre-boost read x organic")) for k, lab in order]
    recipe_tbl = "".join(
        f'<tr class="{"hl" if k.startswith("pre-boost read x organic") else ""}"><td>{esc(lab)}</td><td class="r">{rec[k]["posts"]}</td>'
        f'<td class="r">{pct(rec[k]["median_abs_error"])}</td><td class="r">{pct(rec[k]["within_25pct"])}</td>'
        f'<td class="r">{pct(rec[k]["within_50pct"])}</td><td class="r">{pct(rec[k]["negative_organic"])}</td></tr>' for k, lab in order)

    cl = R["campaign_level"]
    a = R["andrew_confirmed"]
    pf = R["production_function_by_confidence"]
    bt = T["curve_backtest_unboosted_creator_split"]
    curve_posts = max(int(r["posts"]) for r in csv.DictReader(open("reconcile/organic_curve_ig.csv")))
    conf_rows = "".join(
        f'<tr><td>{lab}</td><td class="r">{pf[c]["posts"]}</td><td class="r">{pct(pf[c]["median_abs_error"], 1)}</td>'
        f'<td class="r">{pct(pf[c]["within_25pct"])}</td></tr>'
        for c, lab in [("high", "High: pre-boost read on day 14 or later"), ("medium", "Medium: read on day 7–13"), ("low", "Low: read before day 7")])
    bt_rows = "".join(
        f'<tr><td>{p_lab}, read on day {d}</td><td class="r">{bt[p][f"read on day {d} -> predict day 30"]["posts"]:,}</td>'
        f'<td class="r">{pct(bt[p][f"read on day {d} -> predict day 30"]["median_abs_error"], 1)}</td>'
        f'<td class="r">{pct(bt[p][f"read on day {d} -> predict day 30"]["within_25pct"])}</td></tr>'
        for p, p_lab in [("Instagram", "Instagram"), ("Tiktok", "TikTok")] for d in (3, 7, 14))

    tm = T["tiktok_metric_match"]
    tv = T["tiktok_subtraction_vs_curve"]["subtraction_over_curve"]

    cov = V["totals"]
    seg_def = [("measured_optin", "Measured: opt-in views", "q4"), ("est_high", "Estimate, high confidence", "q3"),
               ("est_medium", "Estimate, medium", "q2"), ("est_low", "Estimate, low", "q1"), ("not_separable", "Not separable", "qn")]
    cov_rows = []
    for p, p_lab in [("Instagram", "Instagram"), ("Tiktok", "TikTok")]:
        t = cov[p]
        segs = "".join(
            f'<span class="seg {cls}" style="flex:{t[k]} 1 0" tabindex="0" data-tip="{esc(f"{p_lab}: {lab}: {t[k]:,} posts ({pct(t[k] / t["posts"])})")}"></span>'
            for k, lab, cls in seg_def if t[k])
        cov_rows.append(f'<div class="st-row"><div class="st-label">{p_lab}<span class="sub">{t["posts"]:,} boosted posts</span></div>'
                        f'<div class="st-bar">{segs}</div></div>')
    cov_tbl = table_view(["Platform"] + [lab for _, lab, _ in seg_def],
                         [[p_lab] + [f"{cov[p][k]:,} ({pct(cov[p][k] / cov[p]['posts'])})" for k, _, _ in seg_def]
                          for p, p_lab in [("Instagram", "Instagram"), ("Tiktok", "TikTok")]])
    ig_cov, tt_cov = cov["Instagram"], cov["Tiktok"]

    return f"""
<section id="math">
  <div class="eyebrow">Part 1 · Tom's question</div>
  <h2>Platform total − one paid metric = organic?</h2>
  <p>We matched {R['panel']['posts']} boosted Instagram posts to their Meta ads ({R['panel']['posts_by_tier']['andrew_confirmed']} from Andrew's post tag, the rest from the ad-name taxonomy, going back in time as you asked).
  For each day we put the public view count, the opt-in count (organic only) and SocAPI plays next to the paid totals by placement.
  The panel has {R['panel']['rows']:,} post-days from {R['panel']['obs_from']} to {R['panel']['obs_to']}.</p>

  <h3>1. Which paid metric matches what the platform counts</h3>
  <p>A ratio of 1× means the paid metric explains the extra views exactly. Impressions are closest on the Instagram side. Video plays are closest on the Facebook side. The video-view counts are far too small.</p>
  <div class="viz">
    <div class="viz-title">Extra views on the platform ÷ paid metric, after the ads ended</div>
    {legend([("s-blue", "Instagram side: (public − opt-in) ÷ paid Instagram placement"), ("s-orange", "Facebook side: SocAPI Facebook-paid plays ÷ paid Facebook placement")])}
    <div class="dr">{''.join(rows)}{axis}</div>
    <p class="proof">Dot = typical (median) ratio; line = middle half of posts. Log scale. Instagram side: {ig['Impressions']['n']} posts, one read per post at least 2 days after the last ad day.
    Facebook side: {R['socapi_postflight']['rows']} SocAPI reads from {R['socapi_postflight']['posts']} posts (SocAPI daily reads exist only on a few days). 2-second plays exist for 20 Instagram posts only.</p>
    {dot_tbl}
  </div>

  <h3>2. Why the subtraction still fails for one post</h3>
  <p>On a boosted post, paid is typically <b>{pct(R['paid_share_of_public']['median'])}</b> of public views (middle half {pct(R['paid_share_of_public']['q25'])}–{pct(R['paid_share_of_public']['q75'])}).
  Organic is the small number left over, so a small error in the paid number becomes a large error in organic. The error grows with the paid share:</p>
  <div class="viz">
    <div class="viz-title">Typical organic error of "public − paid IG impressions", by paid share of public views</div>
    {hbars(eb_rows, vmax=max(r[1] for r in eb_rows), fmt=pct)}
    <p class="proof">{R['postflight_posts']} boosted Instagram posts, truth = opt-in views. Typical = median absolute error.</p>
  </div>
  <p>Best subtraction (public − paid IG impressions): typical error <b>{pct(best['median_abs_error'])}</b>, only {pct(best['within_25pct'])} of posts within ±25%, and {pct(best['negative_organic'])} of posts come out negative.
  On Andrew's {a['posts']} tagged posts while the ads ran ({a['post_days']} post-days), typical error is {pct(a['public - paid IG impressions']['median_abs_error'])} and <b>{pct(a['public - paid IG impressions']['negative_organic'])} of days give negative organic</b> (paid share {pct(a['paid_share_of_public']['median'])}).
  SocAPI-based subtraction is worse ({rec['SocAPI total - all paid video plays (naive)']['posts']} posts, typical error 270–380%).</p>

  <h3>3. The method that works: pre-boost read × organic growth</h3>
  <p>Take the last public read before the first ad day. Before a boost, public views equal opt-in views (typical ratio {f3(R['preboost_public_equals_optin']['median'])}, {R['preboost_public_equals_optin']['n']} posts), so this read is organic.
  Then grow it with the normal organic curve of unboosted posts, from the age at that read to today. The curve was built from {curve_posts:,} unboosted Instagram posts, not from these boosted posts.</p>
  <div class="viz">
    <div class="viz-title">Share of boosted posts with organic within ±25% of opt-in truth</div>
    {hbars(w25, vmax=1, fmt=pct, cls=None)}
    <p class="proof">Subtraction rows: {best['posts']} posts. Pre-boost rows: {curve['posts']} posts (posts with a read before the first ad day). Truth = opt-in views on the same day. Blue = recommended.</p>
  </div>
  <div class="table-wrap"><table>
    <thead><tr><th>Method (Instagram, per post)</th><th class="r">Posts</th><th class="r">Typical error</th><th class="r">Within ±25%</th><th class="r">Within ±50%</th><th class="r">Negative organic</th></tr></thead>
    <tbody>{recipe_tbl}</tbody></table></div>
  <p><b>Campaign totals are better still.</b> Summed over random 10-post groups, the curve method lands within ±25% for {pct(cl['pre-boost read x organic curve']['within_25pct'])} of groups (middle half {pct(cl['pre-boost read x organic curve']['q25'])} to +{pct(cl['pre-boost read x organic curve']['q75'])}). Subtraction: {pct(cl['subtraction: public - k x IG impressions']['within_25pct'])}.</p>

  <div class="two">
    <div>
      <h4>Confidence depends on how late the pre-boost read is</h4>
      <div class="table-wrap"><table><thead><tr><th>Confidence (boosted Instagram)</th><th class="r">Posts</th><th class="r">Typical error</th><th class="r">Within ±25%</th></tr></thead><tbody>{conf_rows}</tbody></table></div>
      <p class="proof">Production function <span class="num">reconcile/estimate.py</span> on the same {curve['posts']} posts, truth = opt-in.</p>
    </div>
    <div>
      <h4>Back-test on unboosted posts (predict day 30)</h4>
      <div class="table-wrap"><table><thead><tr><th>Read</th><th class="r">Posts</th><th class="r">Typical error</th><th class="r">Within ±25%</th></tr></thead><tbody>{bt_rows}</tbody></table></div>
      <p class="proof">Growth factor fitted on half the creators, tested on the other half (no creator in both).</p>
    </div>
  </div>

  <h3>4. TikTok and YouTube</h3>
  <ul>
    <li><b>TikTok:</b> the public lift after a boost is {times(tm['Video plays (starts)']['median'])} the paid video plays (middle half {times(tm['Video plays (starts)']['q25'])}–{times(tm['Video plays (starts)']['q75'])}; impressions {times(tm['Impressions']['median'])}; {T['tiktok_posts']} posts). Paid is typically {pct(T['tiktok_paid_share_of_public']['median'])} of public views.</li>
    <li>TikTok opt-in views equal public views (typical ratio {f3(T['tiktok_optin_equals_public']['median'])}, {T['tiktok_optin_equals_public']['n']} posts), so opt-in includes Spark Ad views. <b>There is no organic truth on TikTok after a boost.</b></li>
    <li>On TikTok, subtraction gives a typical {times(tv['median'])} the curve estimate (middle half {times(tv['q25'])}–{times(tv['q75'])}). We cannot check which is right on TikTok. On Instagram, where we can check, the curve wins.</li>
    <li><b>YouTube:</b> the warehouse has no ad-to-video link (0 of 34,687 posts match), so the math cannot be tested.</li>
  </ul>

  <h3>5. How many boosted posts can get an organic number today</h3>
  <div class="viz">
    <div class="viz-title">Boosted posts by organic method and confidence</div>
    {legend([(c, lab) for _, lab, c in seg_def])}
    <div class="st">{''.join(cov_rows)}</div>
    <p class="proof">All in-feed campaign posts in BIRA with any paid evidence (all publish dates). Source: <span class="num">sql/05_boost_flags.sql</span>, run 2026-10-07.
    Instagram: {pct(ig_cov['measured_optin'] / ig_cov['posts'])} measured by opt-in, {pct((ig_cov['est_high'] + ig_cov['est_medium']) / ig_cov['posts'])} high or medium estimate, {pct(ig_cov['not_separable'] / ig_cov['posts'])} not separable.
    TikTok: {pct((tt_cov['est_high'] + tt_cov['est_medium']) / tt_cov['posts'])} high or medium, {pct(tt_cov['not_separable'] / tt_cov['posts'])} not separable (boosted before the first read).</p>
    {cov_tbl}
  </div>
</section>"""


def confusion(m, name):
    tp, fn, fp, tn = m["tp"], m["fn"], m["fp"], m["tn"]
    pos, neg = tp + fn, fp + tn

    def cell(v, tot, kind, desc):
        share = v / tot if tot else 0
        q = "q4" if share >= .75 else "q3" if share >= .4 else "q2" if share >= .1 else "q1"
        return (f'<td class="cm {q}" tabindex="0" data-tip="{esc(f"{name}: {kind} = {v} ({pct(share, 1)} of {desc})")}">'
                f'<b>{v:,}</b><span>{kind}</span><span>{pct(share, 1)} of {desc}</span></td>')
    return (f'<div class="table-wrap"><table class="cmx"><thead><tr><th></th><th>Model: boosted</th><th>Model: organic</th></tr></thead><tbody>'
            f'<tr><th>Label: boosted ({pos})</th>{cell(tp, pos, "true positive", "boosted")}{cell(fn, pos, "false negative", "boosted")}</tr>'
            f'<tr><th>Label: organic ({neg})</th>{cell(fp, neg, "false positive", "organic")}{cell(tn, neg, "true negative", "organic")}</tr>'
            f'</tbody></table></div>')


METRIC_ROWS = [("roc_auc", "ROC AUC", f3), ("pr_auc", "PR AUC (average precision)", f3), ("f1", "F1", f3), ("f2", "F2 (recall weighted)", f3),
               ("precision", "Precision", f3), ("recall", "Recall (sensitivity)", f3), ("specificity", "Specificity", f3),
               ("npv", "Negative predictive value", f3), ("fpr", "False positive rate", f3), ("fnr", "False negative rate", f3),
               ("accuracy", "Accuracy", f3), ("balanced_accuracy", "Balanced accuracy", f3), ("mcc", "Matthews correlation (MCC)", f3),
               ("kappa", "Cohen's kappa", f3), ("ks", "KS statistic", f3), ("brier", "Brier score (lower is better)", f3),
               ("brier_skill", "Brier skill vs base rate", f3), ("log_loss", "Log loss (lower is better)", f3),
               ("ece", "Expected calibration error", f3), ("prevalence", "Share boosted (base rate)", f3)]


def section_classifier(C, llm, paired):
    P = C["platforms"]
    wr = json.load(open("results/weighted_recall.json"))
    tiles = []
    for p, lab in PLATFORMS:
        t, ci = P[p]["test_metrics"], P[p]["test_ci95"]
        items = [("ROC AUC", "roc_auc"), ("PR AUC", "pr_auc"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall"), ("MCC", "mcc")]
        tiles.append(f'<div class="tiles-block"><h4>{lab} <span class="sub">{esc(P[p]["selected_model"].split("(")[0].replace("_", " "))}, threshold {P[p]["threshold_from_train"]:.2f} from train</span></h4><div class="tiles">'
                     + "".join(f'<div class="tile"><div class="tl">{n}</div><div class="tv">{t[k]:.2f}</div><div class="tc">95% CI {ci[k][0]:.2f}–{ci[k][1]:.2f}</div></div>' for n, k in items)
                     + "</div></div>")

    cms = "".join(f'<div><h4>{lab}: locked test, {P[p]["test"]["n"]} posts</h4>{confusion(P[p]["test_metrics"], lab)}'
                  f'<p class="proof">Published {P[p]["test"]["pub_from"]} to {P[p]["test"]["pub_to"]}. Threshold {P[p]["threshold_from_train"]:.2f}, chosen on train.</p></div>' for p, lab in PLATFORMS)

    def roc(p, lab):
        c, ct = P[p]["curves_test"]["roc"], P[p]["curves_train_cv"]["roc"]
        return xy_chart([{"cls": "s-mute", "pts": list(zip(ct["fpr"], ct["tpr"]))},
                         {"cls": "s-blue", "pts": list(zip(c["fpr"], c["tpr"])),
                          "tip": lambda x, y, i: f"{lab} test: false positive rate {pct(x, 1)}, true positive rate {pct(y, 1)}"}],
                        label=f"{lab} ROC curve", diag=True, xlabel="False positive rate", ylabel="True positive rate (recall)",
                        xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")

    def pr(p, lab):
        c, ct = P[p]["curves_test"]["pr"], P[p]["curves_train_cv"]["pr"]
        base = P[p]["test_metrics"]["prevalence"]
        return xy_chart([{"cls": "s-mute", "pts": list(zip(ct["recall"], ct["precision"]))},
                         {"cls": "s-blue", "pts": list(zip(c["recall"], c["precision"])),
                          "tip": lambda x, y, i: f"{lab} test: recall {pct(x, 1)}, precision {pct(y, 1)}"},
                         {"cls": "s-ref", "pts": [(0, base), (1, base)]}],
                        label=f"{lab} precision-recall curve", xlabel="Recall", ylabel="Precision",
                        xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")

    def cal(p, lab):
        c = P[p]["curves_test"]["calibration"]
        return xy_chart([{"cls": "s-blue", "dots": True, "pts": [(r["mean_pred"], r["share_boosted"]) for r in c],
                          "tip": lambda x, y, i: f"{lab}: mean score {pct(x, 1)}, actually boosted {pct(y, 1)} ({c[i]['n']} posts)"}],
                        label=f"{lab} calibration", diag=True, xlabel="Mean model score (10 equal-size groups)", ylabel="Share actually boosted",
                        xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")

    def sweep(p, lab):
        s = P[p]["threshold_sweep_test"]
        thr = P[p]["threshold_from_train"]
        mk = lambda k: [(r["threshold"], r[k]) for r in s]
        tipf = lambda k, nm: (lambda x, y, i: f"{lab}, threshold {x:.2f}: {nm} {pct(y, 1)}")
        return xy_chart([{"cls": "s-blue", "pts": mk("precision"), "tip": tipf("precision", "precision")},
                         {"cls": "s-orange", "pts": mk("recall"), "tip": tipf("recall", "recall")},
                         {"cls": "s-aqua", "pts": mk("f1"), "tip": tipf("f1", "F1")}],
                        label=f"{lab} threshold sweep", xdom=(0, 1), vline=(thr, f"train choice {thr:.2f}"),
                        xlabel="Decision threshold", ylabel="Test metric", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")

    def gains(p, lab):
        g = P[p]["curves_test"]["gains"]
        return xy_chart([{"cls": "s-blue", "pts": [(0, 0)] + [(r["top_share"], r["boosted_captured"]) for r in g],
                          "tip": lambda x, y, i: f"{lab}: review the top {pct(x)} of scores, find {pct(y, 1)} of boosted posts"}],
                        label=f"{lab} cumulative gains", diag=True, xlabel="Share of posts reviewed (highest score first)",
                        ylabel="Share of boosted posts found", xfmt=lambda v: f"{v:g}", yfmt=lambda v: f"{v:g}")

    def grid(fn, title, keys, note):
        return (f'<div class="viz"><div class="viz-title">{title}</div>{legend(keys) if keys else ""}<div class="pair">'
                + "".join(f'<figure><figcaption>{lab}</figcaption>{fn(p, lab)}</figure>' for p, lab in PLATFORMS)
                + f'</div><p class="proof">{note}</p></div>')

    roc_tbl = table_view(["Platform", "ROC AUC test", "ROC AUC train CV", "PR AUC test", "PR AUC train CV", "Base rate test"],
                         [[lab, f3(P[p]["test_metrics"]["roc_auc"]), f3(P[p]["train_cv_metrics"]["roc_auc"]), f3(P[p]["test_metrics"]["pr_auc"]),
                           f3(P[p]["train_cv_metrics"]["pr_auc"]), pct(P[p]["test_metrics"]["prevalence"], 1)] for p, lab in PLATFORMS])
    cal_tbl = table_view(["Platform", "Group", "Posts", "Mean score", "Share boosted"],
                         [[lab, i + 1, r["n"], pct(r["mean_pred"], 1), pct(r["share_boosted"], 1)] for p, lab in PLATFORMS
                          for i, r in enumerate(P[p]["curves_test"]["calibration"])])
    hist_tbl = table_view(["Platform", "Score range", "Boosted posts", "Organic posts"],
                          [[lab, f'{h["edges"][i]:.2f}–{h["edges"][i + 1]:.2f}', h["boosted"][i], h["organic"][i]]
                           for p, lab in PLATFORMS for h in [P[p]["curves_test"]["histogram"]] for i in range(len(h["boosted"]))])
    sweep_tbl = table_view(["Platform", "Threshold", "Precision", "Recall", "F1", "Specificity", "Flagged share"],
                           [[lab, f'{r["threshold"]:.2f}', pct(r["precision"], 1), pct(r["recall"], 1), pct(r["f1"], 1), pct(r["specificity"], 1),
                             pct(r["flagged_share"], 1)] for p, lab in PLATFORMS for r in P[p]["threshold_sweep_test"]])
    gains_tbl = table_view(["Platform", "Top share reviewed", "Boosted found"],
                           [[lab, pct(r["top_share"]), pct(r["boosted_captured"], 1)] for p, lab in PLATFORMS for r in P[p]["curves_test"]["gains"]])

    hist = (f'<div class="viz"><div class="viz-title">Score distribution by true label (locked test)</div>'
            f'{legend([("s-blue", "Boosted (label)"), ("s-aqua", "Organic (label)")])}<div class="pair">'
            + "".join(f'<figure><figcaption>{lab}</figcaption>{mirror_hist(P[p]["curves_test"]["histogram"], P[p]["threshold_from_train"], f"{lab} score histogram")}</figure>' for p, lab in PLATFORMS)
            + f'</div><p class="proof">Bars show the share of each label group in each score bin, so both groups use the same scale.</p>{hist_tbl}</div>')

    # full metric tables
    full = []
    for p, lab in PLATFORMS:
        a, b, c, ci = P[p]["train_in_sample_metrics"], P[p]["train_cv_metrics"], P[p]["test_metrics"], P[p]["test_ci95"]
        rows = "".join(f'<tr><td>{n}</td><td class="r">{fn(a[k])}</td><td class="r">{fn(b[k])}</td><td class="r"><b>{fn(c[k])}</b></td>'
                       f'<td class="r">{(fn(ci[k][0]) + "–" + fn(ci[k][1])) if k in ci else ""}</td></tr>' for k, n, fn in METRIC_ROWS)
        cm = lambda m: f'{m["tp"]} / {m["fp"]} / {m["fn"]} / {m["tn"]}'
        rows += f'<tr><td>TP / FP / FN / TN</td><td class="r">{cm(a)}</td><td class="r">{cm(b)}</td><td class="r"><b>{cm(c)}</b></td><td></td></tr>'
        rows += f'<tr><td>Posts (boosted)</td><td class="r">{a["n"]:,} ({a["n_pos"]})</td><td class="r">{b["n"]:,} ({b["n_pos"]})</td><td class="r"><b>{c["n"]:,} ({c["n_pos"]})</b></td><td></td></tr>'
        full.append(f'<div><h4>{lab}</h4><div class="table-wrap"><table><thead><tr><th>Metric (threshold {P[p]["threshold_from_train"]:.2f})</th>'
                    f'<th class="r">Train, in-sample</th><th class="r">Train, CV by creator</th><th class="r">Test (locked)</th><th class="r">Test 95% CI</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div></div>')

    ops = []
    for p, lab in PLATFORMS:
        for o in P[p]["operating_points"]:
            t = o["test"]
            ops.append(f'<tr><td>{lab}</td><td>{esc(o["rule"])}</td><td class="r">{o["threshold"]:.2f}</td><td class="r">{pct(t["precision"], 1)}</td>'
                       f'<td class="r">{pct(t["recall"], 1)}</td><td class="r">{pct(t["f1"], 1)}</td><td class="r">{pct(t["flagged_share"], 1)}</td>'
                       f'<td class="r">{t["tp"]} / {t["fp"]} / {t["fn"]} / {t["tn"]}</td></tr>')

    base = []
    for p, lab in PLATFORMS:
        t = P[p]["test_metrics"]
        base.append(f'<tr class="hl"><td>{lab}</td><td>Model (this repo)</td><td class="r">{f3(t["roc_auc"])}</td><td class="r">{pct(t["precision"], 1)}</td><td class="r">{pct(t["recall"], 1)}</td><td class="r">{f3(t["f1"])}</td><td class="r">{f3(t["mcc"])}</td></tr>')
        for name, r in P[p]["rules_test"].items():
            base.append(f'<tr><td>{lab}</td><td>{esc(name.replace("ER", "engagement"))}</td><td class="r">{f3(r["roc_auc"])}</td><td class="r">{pct(r["precision"], 1)}</td><td class="r">{pct(r["recall"], 1)}</td><td class="r">{f3(r["f1"])}</td><td class="r">{f3(r["mcc"])}</td></tr>')
        base.append(f'<tr><td>{lab}</td><td>Always say "organic"</td><td class="r">0.500</td><td class="r">n/a</td><td class="r">0.0%</td><td class="r">0.000</td><td class="r">0.000</td></tr>')

    llm_rows = "".join(
        f'<tr class="{"hl" if r["detector"].startswith("ML") else ""}"><td>{"Instagram" if r["platform"] == "Instagram" else "TikTok"}</td><td>{esc(r["detector"].replace("LLM ", ""))}</td>'
        f'<td class="r">{f3(float(r["roc_auc"]))}</td><td class="r">{pct(float(r["precision"]), 1)}</td><td class="r">{pct(float(r["recall"]), 1)}</td><td class="r">{f3(float(r["f1"]))}</td></tr>'
        for r in sorted(llm, key=lambda r: (r["platform"] != "Tiktok", not r["detector"].startswith("ML"), -float(r["roc_auc"]))))
    paired_txt = "; ".join(f'{r["llm"]} on {"TikTok" if r["platform"] == "Tiktok" else "Instagram"}: {float(r["auc_ml_minus_llm"]):+.3f} ({"significant" if r["significant"] == "True" else "not significant"})' for r in paired)

    slices = []
    for p, lab in PLATFORMS:
        for k, v in P[p]["slices_test"].items():
            slices.append(f'<tr><td>{lab}</td><td>{esc(k)}</td><td class="r">{v["n"]} ({v["n_pos"]})</td><td class="r">{f3(v["roc_auc"])}</td>'
                          f'<td class="r">{f3(v["pr_auc"])}</td><td class="r">{pct(v["precision"], 1)}</td><td class="r">{pct(v["recall"], 1)}</td><td class="r">{f3(v["f1"])}</td></tr>')
        oc = P[p]["overfit_check"]
        slices.append(f'<tr><td>{lab}</td><td>Client holdout (train CV, grouped by client)</td><td class="r">{P[p]["train"]["n"]:,} ({P[p]["train"]["n_pos"]})</td>'
                      f'<td class="r">{f3(oc["train_cv_by_client_roc_auc"])}</td><td class="r"></td><td class="r"></td><td class="r"></td><td class="r"></td></tr>')

    sel = []
    for p, lab in PLATFORMS:
        for r in P[p]["selection_table"]:
            chosen = r["model"] == P[p]["selected_model"]
            sel.append(f'<tr class="{"hl" if chosen else ""}"><td>{lab}</td><td>{esc(r["model"].replace("_", " "))}</td><td class="r">{f3(r["cv_roc_auc"])}</td><td class="r">{f3(r["cv_pr_auc"])}</td></tr>')

    imp = []
    names = {"late_share_d7": "Share of day-30 views gained after day 7", "front_load_d1": "Share of day-30 views already there on day 1",
             "log_views30": "Public views at day 30", "log_accel_gap": "View jump minus like jump (days 8–30)", "log_er30": "Engagement rate at day 30",
             "log_vtf30": "Views ÷ followers at day 30", "late_share_d14": "Share of day-30 views gained after day 14",
             "lpv_dilution": "Likes per view, early vs late", "log_accel": "Biggest daily view jump, days 8–30", "comments_per_like": "Comments per like",
             "log_followers": "Followers"}
    for p, lab in PLATFORMS:
        top = list(P[p]["permutation_importance_test_pr_auc"].items())[:5]
        imp.append(f'<div><h4>{lab}: top inputs</h4>' + hbars([(names.get(k, k), max(v, 0), f"{lab}: PR AUC drops by {v:.3f} when this input is shuffled", None) for k, v in top],
                                                             vmax=max(v for _, v in top), fmt=lambda v: f"{v:.3f}") + "</div>")

    tt, ig = P["Tiktok"]["test_metrics"], P["Instagram"]["test_metrics"]
    return f"""
<section id="classifier">
  <div class="eyebrow">Part 2 · Which posts are paid</div>
  <h2>Evidence first, the model only where no paid record exists</h2>
  <p>Tom said the goal is not to predict boosting. We agree: proof comes first. The daily table uses Andrew's tag, then the ad link, then the opt-in or SocAPI gap, then the manual paid date.
  The model only scores posts with none of these. It uses public views, likes, comments and followers from the first 30 days, and nothing from the paid side.</p>
  {''.join(tiles)}
  <p class="proof">Locked test set: posts published on or after {C['test_cutoff']}, scored once after all choices were made on train. 95% intervals from a bootstrap that resamples whole creators ({C['n_boot']:,} draws).</p>

  <h3>Confusion matrix</h3>
  <div class="pair">{cms}</div>
  <p>TikTok: {tt['fp']} false positives in {tt['fp'] + tt['tn']} organic posts; {tt['fn']} boosts missed. Instagram: {ig['fp']} false positives, but {ig['fn']} boosts missed, so recall is the weak point.
  Many Instagram misses are small boosts: the posts the model catches hold {pct(wr['Instagram']['paid_view_weighted_recall'], 1)} of the paid views on opt-in posts ({wr['Instagram']['n_optin']} posts).</p>

  <h3>ROC and precision-recall curves</h3>
  {grid(roc, "ROC curve", [("s-blue", "Locked test"), ("s-mute", "Train, cross-validated by creator")], "Dashed line = random guess. The test curve sits close to the train CV curve, so the model did not overfit much.")}
  {grid(pr, "Precision-recall curve", [("s-blue", "Locked test"), ("s-mute", "Train, cross-validated by creator"), ("s-ref", "Base rate (random guess)")], "Precision-recall is the fairer view here because most posts are organic.")}
  {roc_tbl}

  <h3>Calibration and score distribution</h3>
  {grid(cal, "Calibration: does a score of 0.8 mean 80% boosted?", None, "Dots on the dashed line = well calibrated. Expected calibration error: TikTok " + f3(tt['ece']) + ", Instagram " + f3(ig['ece']) + ".")}
  {cal_tbl}
  {hist}

  <h3>Threshold choice</h3>
  {grid(sweep, "Precision, recall and F1 at each threshold (locked test)", [("s-blue", "Precision"), ("s-orange", "Recall"), ("s-aqua", "F1")], "The threshold was chosen on train (max F1 on out-of-fold scores). This view shows how test results move if the daily table uses another threshold.")}
  {sweep_tbl}
  <div class="table-wrap"><table><thead><tr><th>Platform</th><th>Operating point (chosen on train)</th><th class="r">Threshold</th><th class="r">Test precision</th><th class="r">Test recall</th><th class="r">Test F1</th><th class="r">Flagged</th><th class="r">TP / FP / FN / TN</th></tr></thead>
  <tbody>{''.join(ops)}</tbody></table></div>
  <p>For the daily table, a "sure" flag on Instagram can use the precision ≥ 95% point: test precision {pct(P['Instagram']['operating_points'][2]['test']['precision'], 1)} at recall {pct(P['Instagram']['operating_points'][2]['test']['recall'], 1)}.</p>
  {grid(gains, "Cumulative gains: review the highest scores first", None, "Dashed line = random order.")}
  {gains_tbl}

  <h3>All metrics: train vs cross-validation vs test</h3>
  <div class="stack">{''.join(full)}</div>
  <p class="proof">Accuracy alone is misleading here: always saying "organic" scores {pct(P['Tiktok']['always_organic_baseline_accuracy'], 1)} accuracy on TikTok and {pct(P['Instagram']['always_organic_baseline_accuracy'], 1)} on Instagram. Use F1, MCC and PR AUC.</p>

  <h3>Against simple rules and LLMs</h3>
  <div class="table-wrap"><table><thead><tr><th>Platform</th><th>Detector (locked test)</th><th class="r">ROC AUC</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">F1</th><th class="r">MCC</th></tr></thead>
  <tbody>{''.join(base)}</tbody></table></div>
  <div class="table-wrap"><table><thead><tr><th>Platform</th><th>Detector (same 150 test posts per platform, same public numbers)</th><th class="r">ROC AUC</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">F1</th></tr></thead>
  <tbody>{llm_rows}</tbody></table></div>
  <p class="proof">LLMs ran in Snowflake Cortex at threshold 0.5. Paired bootstrap, ML AUC minus LLM AUC: {esc(paired_txt)}.</p>

  <h3>Slices, model choice and inputs</h3>
  <div class="table-wrap"><table><thead><tr><th>Platform</th><th>Slice (locked test unless noted)</th><th class="r">Posts (boosted)</th><th class="r">ROC AUC</th><th class="r">PR AUC</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">F1</th></tr></thead>
  <tbody>{''.join(slices)}</tbody></table></div>
  <div class="pair">{''.join(imp)}</div>
  <p class="proof">Input importance = drop in test PR AUC when the input is shuffled (10 repeats).</p>
  <details class="tv"><summary>Show all 13 candidate models (train, 5-fold CV grouped by creator)</summary><div class="table-wrap"><table><thead><tr><th>Platform</th><th>Candidate</th><th class="r">CV ROC AUC</th><th class="r">CV PR AUC</th></tr></thead><tbody>{''.join(sel)}</tbody></table></div></details>
</section>"""


def section_gold(C, V):
    g = C["andrew_gold"]
    ig, tt = g["Instagram"], g["Tiktok"]
    tiers = V["evidence_tiers_all_posts"]
    tier_lab = {"CONFIRMED_PAID_TAG": "Andrew's post tag", "CONFIRMED_AD_LINK": "Ad link (ad name / Spark item id)",
                "MEASURED_OPTIN_GAP": "Opt-in below 80% of public", "MEASURED_SOCAPI_GAP": "SocAPI shows paid Facebook plays",
                "LOGGED_PAID_DATE_ONLY": "Manual paid date only", "NO_PAID_EVIDENCE": "No paid evidence (model scores it)"}
    rows = "".join(f'<tr><td>{i + 1}</td><td>{tier_lab[k]}</td><td class="r">{tiers["Instagram"].get(k, 0):,}</td><td class="r">{(f"{tiers["Tiktok"][k]:,}" if k in tiers["Tiktok"] else "n/a")}</td></tr>'
                   for i, k in enumerate(tier_lab))
    scored = [p for p in ig["posts"] + tt["posts"] if p["in_detector_data"]]
    gold_rows = "".join(
        f'<tr><td>{"Instagram" if p in ig["posts"] else "TikTok"}</td><td>{p["post_type"].title()}, {p["published"]}</td><td class="r">{int(p["days_publish_to_first_spend"])}</td>'
        f'<td>{"yes" if p["boost_inside_30_day_feature_window"] else "no (after day 28)"}</td><td>{"organic" if p["our_label"] == "NEG" else "boosted"}</td>'
        f'<td class="r">{p["model_score"]:.2f}</td><td>{"boosted" if p["model_flag"] else "organic"}</td></tr>' for p in scored)
    reel = sorted(int(p["days_publish_to_first_spend"]) for p in ig["posts"] if p["post_type"] != "CAROUSEL")
    caro = sorted(int(p["days_publish_to_first_spend"]) for p in ig["posts"] if p["post_type"] == "CAROUSEL")
    return f"""
<section id="gold">
  <div class="eyebrow">Part 2 · Andrew's tag as the gold check</div>
  <h2>Andrew's tag confirms our links on Meta, and finds TikTok boosts we missed</h2>
  <p>Since 2026-09-15 the paid team puts the organic post ID in <span class="num">PAID_MEDIA_UNIFIED.ext_p3_organic_post_id</span>. It holds {g['posts_tagged']} Meta and TikTok posts so far (plus 7 LinkedIn).</p>
  <ul>
    <li><b>Meta:</b> our ad-name rule finds {ig['found_by_link_rule_all']} of {ig['tagged']} tagged posts ({ig['tracked_in_bira']} are tracked campaign posts). The rule is safe to use for the history before September.</li>
    <li><b>TikTok:</b> the Spark item-id link finds {tt['found_by_link_rule_all']} of {tt['tagged']}, but <b>{tt['found_by_link_rule_tracked']} of the {tt['tracked_in_bira']} tracked campaign posts</b>. Those 3 boosts were invisible before the tag. Put the tag first.</li>
    <li><b>Timing:</b> tagged Instagram Reels and videos were boosted {reel[0]}–{reel[-1]} days after publish. {len(caro)} carousels were boosted {caro[0]}–{caro[-1]} days after publish. A 30-day model cannot see a boost that late; the tag and the ad link can.</li>
  </ul>
  <div class="stack">
    <div>
      <h4>Paid evidence on all campaign posts in BIRA</h4>
      <div class="table-wrap"><table><thead><tr><th>#</th><th>Evidence (strongest first)</th><th class="r">Instagram</th><th class="r">TikTok</th></tr></thead><tbody>{rows}</tbody></table></div>
      <p class="proof">Each post counts once, in its strongest tier. All publish dates. sql/05, run 2026-10-07.</p>
    </div>
    <div>
      <h4>The model on tagged posts it could score</h4>
      <div class="table-wrap"><table><thead><tr><th>Platform</th><th>Post</th><th class="r">Days to boost</th><th>In model window</th><th>Our label</th><th class="r">Score</th><th>Model says</th></tr></thead><tbody>{gold_rows}</tbody></table></div>
      <p class="proof">All {len(scored)} sit in the locked test set. The TikTok post boosted on day 22 had an "organic" label because the Spark link missed it. The model flagged it (0.95), so one test "false positive" is a real boost. Too few tagged posts exist yet to test the model on the tag alone.</p>
    </div>
  </div>
</section>"""


CSS = """
:root {
  --bg: #f5f6f8; --surface: #ffffff; --ink: #141820; --ink-2: #47505d; --muted: #6a7280; --rule: #dce0e6;
  --grid: #e7e9ee; --axis: #c3c6cd; --accent: #4a3aa7; --accent-soft: #ecebf8; --warn-soft: #fff4e6; --warn-ink: #8a4b00;
  --c-blue: #2a78d6; --c-orange: #eb6834; --c-aqua: #1baf7a; --c-mute: #9aa1ab;
  --q4: #104281; --q3: #256abf; --q2: #5598e7; --q1: #86b6ef; --qn: #d5d8de;
  --q4-ink: #ffffff; --q3-ink: #ffffff; --q2-ink: #0b0b0b; --q1-ink: #0b0b0b;
  --font-display: "Schibsted Grotesk", "Helvetica Neue", Arial, sans-serif;
  --font-body: "Public Sans", "Segoe UI", Roboto, Arial, sans-serif;
  --font-data: "JetBrains Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121418; --surface: #1a1d22; --ink: #eef1f5; --ink-2: #b9c1cc; --muted: #8c95a2; --rule: #2d323a;
    --grid: #262a31; --axis: #3a404a; --accent: #9d93ee; --accent-soft: #24223a; --warn-soft: #2f2414; --warn-ink: #f2b766;
    --c-blue: #3987e5; --c-orange: #d95926; --c-aqua: #199e70; --c-mute: #6b7280;
    --q4: #cde2fb; --q3: #86b6ef; --q2: #3987e5; --q1: #1c5cab; --qn: #3a3f48;
    --q4-ink: #0b0b0b; --q3-ink: #0b0b0b; --q2-ink: #0b0b0b; --q1-ink: #ffffff; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #121418; --surface: #1a1d22; --ink: #eef1f5; --ink-2: #b9c1cc; --muted: #8c95a2; --rule: #2d323a;
  --grid: #262a31; --axis: #3a404a; --accent: #9d93ee; --accent-soft: #24223a; --warn-soft: #2f2414; --warn-ink: #f2b766;
  --c-blue: #3987e5; --c-orange: #d95926; --c-aqua: #199e70; --c-mute: #6b7280;
  --q4: #cde2fb; --q3: #86b6ef; --q2: #3987e5; --q1: #1c5cab; --qn: #3a3f48;
  --q4-ink: #0b0b0b; --q3-ink: #0b0b0b; --q2-ink: #0b0b0b; --q1-ink: #ffffff; color-scheme: dark;
}
* { box-sizing: border-box; }
body { background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 16px; line-height: 1.6; }
.page { max-width: 1080px; margin: 0 auto; padding-inline: 20px; padding-block: 40px 64px; display: grid; gap: 48px; }
header, section { display: grid; gap: 14px; min-width: 0; }
.eyebrow { font-family: var(--font-data); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); }
h1 { font-family: var(--font-display); font-size: clamp(30px, 5vw, 44px); line-height: 1.1; margin: 0; text-wrap: balance; letter-spacing: -.01em; }
h2 { font-family: var(--font-display); font-size: clamp(22px, 3vw, 28px); line-height: 1.2; margin: 0; text-wrap: balance; }
h3 { font-family: var(--font-display); font-size: 19px; margin: 18px 0 0; text-wrap: balance; }
h4 { font-family: var(--font-display); font-size: 16px; margin: 4px 0 8px; }
h4 .sub, .st-label .sub { display: block; font-family: var(--font-body); font-weight: 400; font-size: 13px; color: var(--muted); }
p, ul, ol { margin: 0; max-width: 72ch; }
ul, ol { padding-left: 22px; display: grid; gap: 8px; }
.lede { font-size: 18px; color: var(--ink-2); max-width: 70ch; }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.num { font-family: var(--font-data); font-variant-numeric: tabular-nums; font-size: .92em; }
.proof { font-size: 13px; color: var(--muted); max-width: 80ch; }
.answer { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 22px 24px; display: grid; gap: 14px; }
.tag { font-family: var(--font-data); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
.caveat { background: var(--warn-soft); border-radius: 8px; padding: 14px 18px; display: grid; gap: 6px; }
.caveat .tag { color: var(--warn-ink); }
.table-wrap { overflow-x: auto; border: 1px solid var(--rule); border-radius: 8px; background: var(--surface); min-width: 0; }
table { border-collapse: collapse; width: 100%; font-size: 14px; font-variant-numeric: tabular-nums; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-family: var(--font-data); font-size: 11px; letter-spacing: .05em; text-transform: uppercase; color: var(--muted); font-weight: 600; background: var(--bg); }
td.r, th.r { text-align: right; white-space: nowrap; }
tr:last-child td { border-bottom: 0; }
tr.hl td { background: var(--accent-soft); }
.two, .pair { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 28px; }
.two > *, .pair > * { min-width: 0; }
.stack { display: grid; gap: 20px; min-width: 0; } .stack > * { min-width: 0; }
figure { margin: 0; display: grid; gap: 4px; min-width: 0; }
figcaption { font-weight: 600; font-size: 14px; }
.viz { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 18px 20px; display: grid; gap: 12px; min-width: 0; }
.viz-title { font-weight: 600; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 13px; color: var(--ink-2); }
.key { display: inline-flex; align-items: center; gap: 7px; }
.sw { width: 12px; height: 12px; border-radius: 3px; display: inline-block; background: var(--c-blue); }
.sw.s-blue { background: var(--c-blue); } .sw.s-orange { background: var(--c-orange); } .sw.s-aqua { background: var(--c-aqua); }
.sw.s-mute { background: var(--c-mute); } .sw.s-ref { background: transparent; border-top: 2px dashed var(--axis); border-radius: 0; height: 2px; }
.sw.q4 { background: var(--q4); } .sw.q3 { background: var(--q3); } .sw.q2 { background: var(--q2); } .sw.q1 { background: var(--q1); } .sw.qn { background: var(--qn); }
svg.chart { width: 100%; height: auto; display: block; overflow: visible; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .ref, svg .ln.s-ref { stroke: var(--axis); stroke-width: 1.5; stroke-dasharray: 4 4; fill: none; }
svg .tick { fill: var(--muted); font-size: 11.5px; font-family: var(--font-body); }
svg .axlab { fill: var(--ink-2); font-size: 12px; font-family: var(--font-body); }
svg .ln { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }
svg .ln.s-blue { stroke: var(--c-blue); } svg .ln.s-orange { stroke: var(--c-orange); } svg .ln.s-aqua { stroke: var(--c-aqua); } svg .ln.s-mute { stroke: var(--c-mute); }
svg .dot { stroke: var(--surface); stroke-width: 2; } svg .dot.s-blue { fill: var(--c-blue); }
svg .bar.s-blue { fill: var(--c-blue); } svg .bar.s-aqua { fill: var(--c-aqua); }
svg .hit { fill: transparent; cursor: crosshair; }
svg .hit:hover { fill: var(--ink); fill-opacity: .12; }
/* horizontal bars */
.hb { display: grid; gap: 8px; }
.hb-row { display: grid; grid-template-columns: minmax(0, 2.2fr) minmax(0, 3fr) 4.5em; gap: 12px; align-items: center; font-size: 14px; }
.hb-label { color: var(--ink-2); min-width: 0; }
.hb-track { height: 16px; }
.hb-bar { height: 16px; border-radius: 0 4px 4px 0; background: var(--c-blue); }
.hb-bar.s-mute { background: var(--c-mute); }
.hb-val { text-align: right; }
/* dot-range */
.dr { display: grid; gap: 2px; }
.dr-row { display: grid; grid-template-columns: minmax(0, 11em) minmax(0, 1fr); gap: 12px; align-items: center; font-size: 14px; }
.dr-label { color: var(--ink-2); }
.dr-track { position: relative; height: 30px; }
.dr-axis .dr-track { height: 20px; }
.dr-grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--grid); }
.dr-ref { position: absolute; top: 0; bottom: 0; width: 0; border-left: 1.5px dashed var(--axis); }
.dr-tick { position: absolute; transform: translateX(-50%); font-size: 11.5px; color: var(--muted); white-space: nowrap; }
.dr-whisk { position: absolute; height: 2px; border-radius: 1px; }
.dr-whisk.top { top: 9px; background: var(--c-blue); } .dr-whisk.bot { top: 19px; background: var(--c-orange); }
.dr-dot { position: absolute; width: 10px; height: 10px; border-radius: 50%; transform: translate(-50%, -50%); box-shadow: 0 0 0 2px var(--surface); cursor: default; }
.dr-dot.top { top: 10px; } .dr-dot.bot { top: 20px; }
.dr-dot.s-blue { background: var(--c-blue); } .dr-dot.s-orange { background: var(--c-orange); }
/* stacked */
.st { display: grid; gap: 12px; }
.st-row { display: grid; grid-template-columns: minmax(0, 11em) minmax(0, 1fr); gap: 12px; align-items: center; }
.st-label { font-weight: 600; font-size: 14px; }
.st-bar { display: flex; gap: 2px; height: 24px; }
.seg { min-width: 3px; }
.seg:first-child { border-radius: 4px 0 0 4px; } .seg:last-child { border-radius: 0 4px 4px 0; }
.seg.q4 { background: var(--q4); } .seg.q3 { background: var(--q3); } .seg.q2 { background: var(--q2); } .seg.q1 { background: var(--q1); } .seg.qn { background: var(--qn); }
/* tiles */
.tiles-block { display: grid; gap: 4px; }
.tiles { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; }
.tile { background: var(--surface); border: 1px solid var(--rule); border-radius: 8px; padding: 10px 12px; display: grid; gap: 2px; }
.tl { font-size: 12px; color: var(--muted); }
.tv { font-family: var(--font-display); font-size: 26px; font-weight: 700; line-height: 1.1; }
.tc { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
/* confusion matrix */
table.cmx { background: var(--surface); }
.cmx td.cm { text-align: center; padding: 12px 8px; border: 2px solid var(--surface); }
.cmx td.cm b { display: block; font-family: var(--font-display); font-size: 24px; line-height: 1.1; }
.cmx td.cm span { display: block; font-size: 12px; }
.cm.q4 { background: var(--q4); color: var(--q4-ink); } .cm.q3 { background: var(--q3); color: var(--q3-ink); }
.cm.q2 { background: var(--q2); color: var(--q2-ink); } .cm.q1 { background: var(--q1); color: var(--q1-ink); }
details.tv summary { cursor: pointer; font-size: 13px; color: var(--accent); }
details.tv[open] summary { margin-bottom: 8px; }
#tip { position: fixed; z-index: 10; max-width: 300px; background: var(--ink); color: var(--bg); font-size: 12.5px; line-height: 1.4; padding: 7px 9px; border-radius: 6px; pointer-events: none; }
footer { border-top: 1px solid var(--rule); padding-top: 18px; display: grid; gap: 10px; font-size: 14px; color: var(--ink-2); }
@media (max-width: 760px) {
  .two, .pair { grid-template-columns: minmax(0, 1fr); }
  .tiles { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .hb-row { grid-template-columns: minmax(0, 1fr) 4.5em; }
  .hb-label { grid-column: 1 / -1; }
  .dr-row, .st-row { grid-template-columns: minmax(0, 1fr); gap: 2px; }
  .dr-axis .dr-label { display: none; }
  .answer { padding: 18px; }
  th, td { padding: 7px 9px; }
  .cmx td.cm { padding: 10px 4px; }
  .cmx td.cm b { font-size: 20px; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

JS = """
(function () {
  var tip = document.getElementById('tip');
  function place(x, y) {
    var w = tip.offsetWidth, h = tip.offsetHeight, vw = window.innerWidth;
    var left = Math.min(Math.max(8, x + 12), vw - w - 8), top = y - h - 12;
    if (top < 8) top = y + 16;
    tip.style.left = left + 'px'; tip.style.top = top + 'px';
  }
  function show(el, x, y) { tip.textContent = el.getAttribute('data-tip'); tip.hidden = false; place(x, y); }
  document.addEventListener('pointerover', function (e) {
    var el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (el) show(el, e.clientX, e.clientY); else tip.hidden = true;
  });
  document.addEventListener('pointermove', function (e) { if (!tip.hidden) place(e.clientX, e.clientY); });
  document.addEventListener('focusin', function (e) {
    var el = e.target.closest ? e.target.closest('[data-tip]') : null;
    if (!el) return; var r = el.getBoundingClientRect(); show(el, r.left + r.width / 2, r.top);
  });
  document.addEventListener('focusout', function () { tip.hidden = true; });
  window.addEventListener('scroll', function () { tip.hidden = true; }, { passive: true });
})();
"""


def build():
    R, T, C, V, llm, paired = load()
    P = C["platforms"]
    rec = R["recipes"]
    best = rec["public - paid IG Impressions"]
    curve = rec["pre-boost read x organic curve (no fitting on these posts)"]
    a = R["andrew_confirmed"]
    ig_mm = R["metric_match"]["instagram_side"]
    fb_mm = R["metric_match"]["facebook_side"]
    cl = R["campaign_level"]
    pf = R["production_function_by_confidence"]
    g = C["andrew_gold"]
    cov = V["totals"]
    tt, ig = P["Tiktok"]["test_metrics"], P["Instagram"]["test_metrics"]
    rule = "Tom's rule: views > followers AND engagement < 1%"
    bt = T["curve_backtest_unboosted_creator_split"]

    answer = f"""
<div class="answer" id="answer">
  <div class="tag">Answer</div>
  <ol>
    <li><b>Impressions is the paid metric that matches.</b> Instagram: public − opt-in = {times(ig_mm['Impressions']['median'])} paid Instagram-placement impressions (typical; middle half {times(ig_mm['Impressions']['q25'])}–{times(ig_mm['Impressions']['q75'])}; {ig_mm['Impressions']['n']} boosted posts after the ads ended).
      Facebook: SocAPI's extra plays = {times(fb_mm['Video plays (starts)']['median'])} paid Facebook-placement video plays ({R['socapi_postflight']['posts']} posts). The 2-second, 3-second, ThruPlay and percent-watched counts are {times(ig_mm['3-second video views']['median'])} to {times(ig_mm['100% watched']['median'])} too small.</li>
    <li><b>The subtraction still fails for one post.</b> Paid is typically {pct(R['paid_share_of_public']['median'])} of public views on a boosted post, so a small paid error becomes a large organic error.
      Best case: typical error {pct(best['median_abs_error'])}, {pct(best['within_25pct'])} of posts within ±25%, {pct(best['negative_organic'])} negative. On Andrew's {a['posts']} tagged posts while the ads ran, {pct(a['public - paid IG impressions']['negative_organic'])} of days give negative organic.</li>
    <li><b>Use the pre-boost read × organic growth instead.</b> Typical error {pct(curve['median_abs_error'], 1)}; {pct(curve['within_25pct'])} of posts within ±25%; never negative ({curve['posts']} boosted Instagram posts vs opt-in truth). For a 10-post campaign total, {pct(cl['pre-boost read x organic curve']['within_25pct'])} of totals land within ±25% (subtraction: {pct(cl['subtraction: public - k x IG impressions']['within_25pct'])}).</li>
    <li><b>Andrew's tag is the best proof of paid.</b> Our ad-name rule finds {g['Instagram']['found_by_link_rule_all']} of {g['Instagram']['tagged']} tagged Meta posts. The TikTok Spark link misses all {g['Tiktok']['tracked_in_bira']} tagged TikTok campaign posts. The daily table puts the tag first.</li>
    <li><b>Where no paid record exists, the model works.</b> Locked test of later posts: TikTok F1 {f2(tt['f1'])}, ROC AUC {f2(tt['roc_auc'])}; Instagram F1 {f2(ig['f1'])}, ROC AUC {f2(ig['roc_auc'])}, precision {f2(ig['precision'])}. Tom's hand rule: F1 {f2(P['Tiktok']['rules_test'][rule]['f1'])} and {f2(P['Instagram']['rules_test'][rule]['f1'])}.</li>
  </ol>
  <div class="tag">Ask Andrew and the paid team for</div>
  <ol>
    <li>The post ID or URL on every boosted ad, on every platform. The match on his tag is exact.</li>
    <li>At least 7 days, best 14, between publish and the first boost. A pre-boost read on day 14 or later gives {pct(pf['high']['median_abs_error'], 1)} typical error ({pf['high']['posts']} posts, {pct(pf['high']['within_25pct'])} within ±25%). A read before day 7 gives {pct(pf['low']['median_abs_error'])}.</li>
    <li>A dark post, or a creator opt-in, when a boost must start on day 0. Today {pct(cov['Instagram']['not_separable'] / cov['Instagram']['posts'])} of boosted Instagram posts and {pct(cov['Tiktok']['not_separable'] / cov['Tiktok']['posts'])} of boosted TikTok posts cannot be separated.</li>
  </ol>
  <div class="caveat">
    <div class="tag">The caveat that can change the decision</div>
    <p><b>TikTok organic is UNVERIFIED after a boost.</b> TikTok opt-in views include Spark Ad views, so we have no organic truth there. The method passes a back-test on unboosted TikTok posts (day-14 read: {pct(bt['Tiktok']['read on day 14 -> predict day 30']['median_abs_error'])} typical error), but not on boosted ones. Label TikTok organic numbers as estimates until one dark-post or holdout test exists.</p>
  </div>
</div>"""

    validity = f"""
<section id="validity">
  <div class="eyebrow">Method</div>
  <h2>How we kept the numbers honest</h2>
  <div class="two">
    <ul>
      <li><b>Truth for organic:</b> Instagram opt-in (private) views. Before a boost they equal public views (typical ratio {f3(R['preboost_public_equals_optin']['median'])}), so they are a fair organic truth.</li>
      <li><b>No fitting on the test posts:</b> the organic curve comes from unboosted posts; the methods with a fitted factor use leave-one-out (the post itself is never in its own fit).</li>
      <li><b>Paid totals by placement</b> come from the EDW Meta ad tables. Andrew's unified table is built from them (its sums match), so every tier uses the same definitions.</li>
      <li><b>One read per post</b> for post-level errors (latest read at least 2 days after the last ad day). Campaign errors use 2,000 random 10-post groups.</li>
    </ul>
    <ul>
      <li><b>No paid data in model inputs.</b> No paid date, boost flag, ad name, spend, opt-in or SocAPI field. <span class="num">detector/features.py</span> asserts this.</li>
      <li><b>Locked test by time:</b> train before {C['test_cutoff']}, test after; scored once. Model family (13 candidates) and threshold chosen on train only, with 5-fold CV grouped by creator.</li>
      <li><b>Re-run check:</b> the test scores were re-created from code for this report and match the first run to 2×10⁻¹⁶. Nothing was tuned after the test was seen.</li>
      <li><b>Labels are not perfect.</b> TikTok "organic" means no VN ad link. Andrew's tag already shows one wrong "organic" label, so test precision is a floor.</li>
    </ul>
  </div>
</section>"""

    daily = f"""
<section id="daily">
  <div class="eyebrow">Next</div>
  <h2>The daily paid classification table</h2>
  <p>The pipeline is written and its logic is tested offline. It is not running yet: it needs a Snowflake service account and a schema we may write to.</p>
  <div class="table-wrap"><table><thead><tr><th>Step</th><th>What it does</th><th>File</th></tr></thead><tbody>
    <tr><td>1</td><td>Paid evidence tier and organic inputs for every Instagram and TikTok campaign post</td><td class="num">sql/05_boost_flags.sql</td></tr>
    <tr><td>2</td><td>30-day public features; model score where no evidence exists</td><td class="num">sql/02…, detector/</td></tr>
    <tr><td>3</td><td>Organic estimate with method and confidence</td><td class="num">reconcile/estimate.py</td></tr>
    <tr><td>4</td><td>Stage and MERGE into <span class="num">PAID_CLASSIFICATION__POST</span> (one row per post)</td><td class="num">sql/09…, pipeline/run_daily.py</td></tr>
  </tbody></table></div>
  <p>Paid status values: PAID_CONFIRMED (tag or ad link), PAID_MEASURED (opt-in or SocAPI gap), PAID_LOGGED (manual date only), PAID_PREDICTED, ORGANIC_PREDICTED, NOT_SCORED (under 28 days or no public views).</p>
</section>"""

    footer = f"""
<footer>
  <div><b>Sources.</b> BIRA mart and observation time series (public = Nimble, opt-in = private API) are the source of truth for organic. Paid: <span class="num">DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED</span> (Andrew's tag) and the EDW Meta and TikTok ad tables (placement split). SocAPI: the daily media-info table. Data read on 2026-10-07; reconciliation panel {R['panel']['obs_from']} to {R['panel']['obs_to']}.</div>
  <div><b>Terms.</b> Typical = median. Middle half = 25th to 75th percentile. Error = (estimate − opt-in) ÷ opt-in. Paid share = 1 − opt-in ÷ public. Precision = share of flagged posts that are boosted. Recall = share of boosted posts flagged.</div>
  <div><b>Code and SQL:</b> <a href="{REPO}">boosted_or_organic_detector</a> (branch <span class="num">claude/funny-franklin-5w06tt</span>). This page is built by <span class="num">reports/build_report.py</span> from aggregate results only.</div>
</footer>"""

    body = f"""<title>Paid vs Organic Separation</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@600;700&family=Public+Sans:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;600&display=swap">
<style>/* Layout: one wide reading column; paired small multiples (TikTok | Instagram) that stack on phones; every chart has a table view. */{CSS}</style>
<div class="page">
<header>
  <div class="eyebrow">BI research · for Tom · data read 2026-10-07</div>
  <h1>Paid vs Organic Separation</h1>
  <p class="lede">Tom asked: if we take the views on the platform and subtract one paid metric from Andrew's paid table, do we get organic views?
  We tested every paid metric against opt-in views, which are organic only. One metric matches, but the subtraction still fails for single posts. A different method works.
  Part 2 shows how well we can tell which posts are paid, with every standard classification output.</p>
</header>
{answer}
{section_math(R, T, V)}
{section_gold(C, V)}
{section_classifier(C, llm, paired)}
{validity}
{daily}
{footer}
</div>
<div id="tip" role="tooltip" hidden></div>
<script>{JS}</script>
"""
    os.makedirs("reports", exist_ok=True)
    open(OUT, "w").write(body)
    print("wrote", OUT, f"{len(body) / 1024:.0f} KB")


if __name__ == "__main__":
    build()
