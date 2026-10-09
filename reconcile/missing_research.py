"""Why some boosted posts still have no organic number, and what else we tried (aggregates only).

The posts with no split have no organic-only views before the boost (boosted at publish or before our first read), or
no clear jump in the daily views. We usually know WHEN those boosts started; what is missing is the organic LEVEL.
This script tests five other ways to get it. Truth = Instagram opt-in (organic only); opt-in that stopped updating is
excluded. Split by a hash of the post key where anything is fitted (tune half), scored on the other half (test).

  1 speed-up      : share of read-to-read intervals (after day 3) where the daily gain rises >= 25% (series, sql/15)
  2 after the ads : paid share of the daily public gain before / during / after the ads (panel, sql/06); and how well the
                    growth in one window predicts total size on organic posts (series)
  3 likes         : likes per view, organic vs boosted (day 30, sql/11); organic = likes with paid likes removed
  4 creator       : the creator's usual organic views (median day-30 views of their unboosted posts)
  5 slow ramp     : growth vs the organic curve over a 5-day window, for posts with no clear jump

Inputs : data/recon_panel_ig_full.psv, data/view_series.psv, data/boost_start_eval_keys.csv, data/v2_raw.psv,
         data/v2_extra.psv, data/optin_staleness.psv, outputs/paid_classification_2026-10-07.parquet (eval truth),
         outputs/paid_classification_2026-10-08.parquet (missing posts)
Output : results/missing_research.json
Run    : python3 -m reconcile.missing_research
"""
import hashlib
import json

import numpy as np
import pandas as pd

from detector.features_v2 import STALE_COLS, load as load_v2
from reconcile.boost_start_eval import frame
from reconcile.estimate import growth, load_curves
from reconcile.panel import load as load_panel


def half(k):
    return "tune" if int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 else "test"


def errors(est, truth):
    t = np.asarray(truth, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = pd.Series((np.asarray(est, float) - t) / t).replace([np.inf, -np.inf], np.nan).dropna()
    return {"posts": int(len(e)), "median_abs_error": float(e.abs().median()) if len(e) else None,
            "median_error": float(e.median()) if len(e) else None, "within_25pct": float((e.abs() <= .25).mean()) if len(e) else None}


def after_the_ads():
    d = load_panel().sort_values(["psrk", "od"])
    d = d[d.vp.notna() & d.vr.notna()].copy()
    d["dd"] = d.groupby("psrk").od.diff().dt.days
    d["dvp"] = d.groupby("psrk").vp.diff() / d.dd
    d["dvr"] = d.groupby("psrk").vr.diff() / d.dd
    d = d[d.dd.between(1, 3)]
    last = d[d.dvr > 0].groupby("psrk").od.max().rename("last_change")
    d = d.merge(last, on="psrk")
    d = d[d.od <= d.last_change]                         # opt-in still updating
    d["paid_share"] = ((d.dvp - d.dvr) / d.dvp).where(d.dvp > 0)
    phases = {"before the ads": d.dsf < 0, "during the ads": (d.dsf >= 0) & (d.dsl <= 0),
              "2 days after": d.dsl == 2, "3 days after": d.dsl == 3, "4-30 days after": d.dsl.between(4, 30)}
    return {k: {"posts": int(d[m].psrk.nunique()), "reads": int(m.sum()), "paid_share_of_daily_gain_median": float(d.paid_share[m].median())}
            for k, m in phases.items()}


def series_tests(C):
    d, _ = frame()
    out = {"speed_up": {}, "window_growth_organic": {}, "slow_ramp": {}, "no_clear_jump_instagram_optin": {}}
    rows = []
    for r in d.itertuples():
        a, v = r.ages, np.maximum.accumulate(r.views)
        rate = np.diff(v) / np.maximum(np.diff(a), 1)
        for i in range(1, len(rate)):
            if a[i] < 3 or rate[i - 1] <= 0:
                continue
            if r.grp.endswith("organic"):
                ph = "organic posts"
            elif pd.notna(r.fs_age) and r.fs_age <= a[i] <= r.fs_age + 7:
                ph = "first 7 days of ads"
            else:
                continue
            rows.append((r.platform, ph, rate[i] / rate[i - 1] > 1.25))
    x = pd.DataFrame(rows, columns=["platform", "phase", "up"])
    for (p, ph), g in x.groupby(["platform", "phase"]):
        out["speed_up"].setdefault(p, {})[ph] = {"intervals": int(len(g)), "share_up_25pct": float(g.up.mean())}
    # growth in one window predicts total size? (organic posts: truth = public views now / curve now)
    o = d[d.grp.str.endswith("organic")]
    for (lo, hi) in ((7, 21), (30, 60)):
        for p in ("Instagram", "Tiktok"):
            est, tru = [], []
            for r in o[o.platform == p].itertuples():
                c = C[p]; a = np.minimum(r.ages, 130); v = np.maximum.accumulate(r.views)
                i, j = np.where(a >= lo)[0], np.where(a <= hi)[0]
                if not len(i) or not len(j) or j[-1] <= i[0] or a[j[-1]] - a[i[0]] < (hi - lo) * .6 or not r.post_age_days == r.post_age_days:
                    continue
                est.append((v[j[-1]] - v[i[0]]) / (c[a[j[-1]]] - c[a[i[0]]]))
                tru.append(r.views_public_latest / c[min(int(r.post_age_days), 130)])
            out["window_growth_organic"][f"{p} day {lo}-{hi}"] = errors(est, tru)
    # slow ramp (5-day window) for posts with no clear jump
    nj = d[d.detect_note == "no clear jump in the first 90 days"]
    found, est, tru, fa = 0, [], [], [0, 0]
    for r in nj.itertuples():
        a = np.minimum(np.asarray(r.ages, int), 130); v = np.maximum.accumulate(r.views); c = C[r.platform]; u = v / c[a]
        hit = None
        for i in range(len(a) - 1):
            if a[i] < 3 or v[i] <= 0:
                continue
            j = min(int(np.searchsorted(a, a[i] + 5)), len(a) - 1)
            if j > i and a[j] - a[i] >= 2 and u[j] / u[i] >= 1.25 and u[-1] / u[i] >= 1.5:
                hit = i
                break
        if r.grp.endswith("organic"):
            fa[0] += hit is not None; fa[1] += 1
            continue
        found += hit is not None
        if hit is not None and r.grp == "ig_boosted" and r.organic_views_method == "opt-in private views (organic only)":
            est.append(min(v[hit] * growth(c, [a[hit]], [r.post_age_days])[0], r.views_public_latest))
            tru.append(min(r.views_optin_latest, r.views_public_latest))
    ig = nj[(nj.grp == "ig_boosted") & (nj.organic_views_method == "opt-in private views (organic only)")]
    ps = 1 - np.minimum(ig.views_optin_latest, ig.views_public_latest) / ig.views_public_latest
    out["no_clear_jump_instagram_optin"] = {"posts": int(len(ig)), "paid_share_median": float(ps.median()), "share_under_20pct_paid": float((ps < .2).mean())}
    out["slow_ramp"] = {"boosted_no_jump_posts": int((~nj.grp.str.endswith("organic")).sum()), "found": int(found),
                        "instagram_organic_vs_optin": errors(est, tru), "false_alarm_organic": float(fa[0] / max(fa[1], 1))}
    return out


def day30_tests():
    d = load_v2("data/v2_raw.psv", extra="data/v2_extra.psv")
    st = pd.read_csv("data/optin_staleness.psv", sep="|", header=None, names=STALE_COLS, dtype={"psrk": str})
    d = d.merge(st, on="psrk", how="left")
    ig = d[(d.platform == "Instagram") & d.v30.gt(0) & d.l30.notna() & d.o30.notna()].copy()
    ig = ig[~((ig.fz30 >= 7) & (ig.pg30 >= 0.05))]
    ig["org30"] = np.minimum(ig.o30, 1) * ig.v30
    ig["paid30"] = ig.v30 - ig.org30
    ig["lpv"] = ig.l30 / ig.v30
    ig["half"] = ig.psrk.map(half)
    org = ig[ig.lab30 == "N"]
    boo = ig[(ig.paid30 / ig.v30 >= 0.2) & (ig.lab30 == "P")].copy()
    og = org.groupby("creator").agg(c_v30=("v30", "median"), c_lpv=("lpv", "median"))
    boo = boo.merge(og, left_on="creator", right_index=True, how="left")
    boo["a"] = boo.c_lpv.fillna(org.lpv.median())
    tb = boo[boo.half == "tune"]
    k = float(np.clip(((tb.l30 - tb.a * tb.org30) / (tb.a * tb.paid30)).replace([np.inf, -np.inf], np.nan).dropna().median(), 0, .99))
    te = boo[boo.half == "test"]
    likes_est = np.minimum(((te.l30 - k * te.a * te.v30) / (te.a * (1 - k))).clip(lower=0), te.v30)
    has = boo.c_v30.notna()
    out = {"likes": {"likes_per_view_organic_median": float(org.lpv.median()), "likes_per_view_boosted_median": float(boo.lpv.median()),
                     "boosted_paid_share_median": float((boo.paid30 / boo.v30).median()),
                     "paid_like_rate_vs_organic_fitted_on_tune": k, "organic_from_likes_test": errors(likes_est, te.org30)},
           "creator": {"boosted_tune": errors(boo.c_v30[has & (boo.half == "tune")], boo.org30[has & (boo.half == "tune")]),
                       "boosted_test": errors(boo.c_v30[has & (boo.half == "test")], boo.org30[has & (boo.half == "test")]),
                       "boosted_posts_with_creator_history": int(has.sum()), "boosted_posts": int(len(boo))}}
    # leave-one-out on unboosted posts of creators with 2+ unboosted posts (both platforms)
    t = pd.read_parquet("outputs/paid_classification_2026-10-08.parquet")
    t = t.rename(columns={"POST_SCRAPER_REFERENCE_KEY": "psrk", "POST_PLATFORM": "platform"})
    t["psrk"] = t.psrk.astype(str)
    m = t.merge(d[["psrk", "platform", "creator", "v30"]], on=["psrk", "platform"], how="left")
    pool = m[(m.IS_PAID == False) & m.v30.gt(0) & m.creator.notna()]  # noqa: E712
    for p in ("Instagram", "Tiktok"):
        est, tru = [], []
        for _, g in pool[pool.platform == p].groupby("creator"):
            if len(g) < 2:
                continue
            for i in g.index:
                est.append(g.drop(i).v30.median()); tru.append(g.v30[i])
        out["creator"][f"unboosted_leave_one_out_{p.lower()}"] = errors(est, tru)
    hist = pool.groupby(["platform", "creator"]).size().rename("n").reset_index()
    miss = m[(m.IS_PAID == True) & m.ORGANIC_VIEWS_METHOD.fillna("").str.startswith("not separable")]  # noqa: E712
    miss = miss.merge(hist, on=["platform", "creator"], how="left")
    out["creator"]["missing_posts_with_creator_history"] = {p: {"missing": int((miss.platform == p).sum()), "with_history": int(((miss.platform == p) & (miss.n >= 1)).sum())}
                                                            for p in ("Instagram", "Tiktok")}
    return out


def main():
    C = load_curves()
    out = {"note": "Instagram truth = opt-in still updating; fitted parts tuned on half the posts and scored on the other half",
           "after_the_ads_panel": after_the_ads(), **series_tests(C), **day30_tests()}
    json.dump(out, open("results/missing_research.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
