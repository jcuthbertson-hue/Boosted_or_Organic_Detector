"""Test of the view-jump boost start (reconcile/boost_start.py). Aggregates only.

Posts: boosted posts (ad record or model flag) plus two organic samples (500 Instagram posts whose opt-in shows organic
only; 300 TikTok posts with a model score < 0.05), daily public views from sql/15 (pulled 2026-10-08).
Split by a hash of the post key: settings were chosen on the 'tune' half; the 'test' half was scored once.
  start     : boosted posts with an ad-log start date. Extra views already in the picked read, against the organic
              path from the true last clean read: usable <= 5%, bad > 5%; found_without_clean_read = the true start is
              before every read, so any pick is too late.
  organic   : Instagram boosted posts with live opt-in. Estimate from the picked read vs opt-in (organic only).
  paid      : the same posts. Paid = public - organic estimate vs public - opt-in.
  tiktok    : TikTok posts with a known pre-boost read. Estimate from the picked read vs estimate from the known date.
  false alarm: share of the organic samples where a pre-boost read is picked (it would remove real organic views).

Inputs : data/view_series.psv (sql/15), data/boost_start_eval_keys.csv (psrk, post_platform, grp),
         outputs/paid_classification_2026-10-07.parquet (the table before this step: evidence, dates, opt-in)
Output : results/boost_start_eval.json
Run    : python3 -m reconcile.boost_start_eval
"""
import hashlib
import json

import numpy as np
import pandas as pd

from reconcile.boost_start import PARAMS, detect_all, load_series
from reconcile.estimate import growth, load_curves

TABLE = "outputs/paid_classification_2026-10-07.parquet"
OPTIN = "opt-in private views (organic only)"
KNOWN = "pre-boost public read x organic curve"


def half(k):
    return "tune" if int(hashlib.md5(k.encode()).hexdigest(), 16) % 2 == 0 else "test"


def errors(est, truth):
    e = pd.Series((np.asarray(est, float) - np.asarray(truth, float)) / np.asarray(truth, float)).replace([np.inf, -np.inf], np.nan).dropna()
    return {"posts": int(len(e)), "median_abs_error": float(e.abs().median()) if len(e) else None,
            "median_error": float(e.median()) if len(e) else None,
            "within_10pct": float((e.abs() <= .10).mean()) if len(e) else None,
            "within_25pct": float((e.abs() <= .25).mean()) if len(e) else None}


def frame():
    s = load_series("data/view_series.psv")
    k = pd.read_csv("data/boost_start_eval_keys.csv", dtype=str).rename(columns={"post_platform": "platform"})
    t = pd.read_parquet(TABLE)
    t.columns = [c.lower() for c in t.columns]
    t = t.rename(columns={"post_scraper_reference_key": "psrk", "post_platform": "platform"})
    t["psrk"] = t.psrk.astype(str)
    d = s.merge(k, on=["psrk", "platform"]).merge(t, on=["psrk", "platform"], how="left")
    C = load_curves()
    d = d.merge(detect_all(d, C), on=["psrk", "platform"])
    d["found"] = d.views_preboost_detected.notna()
    pub = pd.to_datetime(d.post_published_date)
    d["fs_age"] = (pd.to_datetime(d.first_spend_date) - pub).dt.days
    d["half"] = d.psrk.map(half)
    est = pd.Series(np.nan, index=d.index)
    for p, c in C.items():
        m = d.found & (d.platform == p) & d.post_age_days.notna()
        g = growth(c, d.preboost_age_detected[m], d.post_age_days[m])
        est[m] = np.minimum(d.views_preboost_detected[m] * g, d.views_public_latest[m].fillna(np.inf))
    d["org_est"] = est
    return d, C


def extra_views(r, C):
    """Extra views in the picked read vs the organic path from the true last clean read (None if no clean read)."""
    a, v, c = r.ages, np.maximum.accumulate(r.views), C[r.platform]
    pre = np.where(a < r.fs_age)[0]
    if not len(pre):
        return np.nan
    k, i = pre[-1], int(np.where(a == r.preboost_age_detected)[0][0])
    return 0.0 if i <= k else v[i] / (v[k] * c[min(a[i], 130)] / c[min(a[k], 130)]) - 1


def score(d, C):
    out = {}
    b = d[d.grp.str.endswith("boosted")]
    for p in ("Instagram", "Tiktok"):
        g = b[(b.platform == p) & b.boost_evidence.isin(["CONFIRMED_AD_LINK", "CONFIRMED_PAID_TAG"]) & b.fs_age.notna()]
        f = g[g.found]
        x = f.apply(lambda r: extra_views(r, C), axis=1) if len(f) else pd.Series(dtype=float)
        out[f"start_{p.lower()}"] = {"posts": int(len(g)), "with_clean_read": int(sum((a < s).any() for a, s in zip(g.ages, g.fs_age))),
                                     "found": int(len(f)), "usable": int((x <= .05).sum()), "bad": int((x > .05).sum()),
                                     "found_without_clean_read": int(x.isna().sum())}
    ig = b[(b.platform == "Instagram") & (b.organic_views_method == OPTIN)]
    truth = np.minimum(ig.views_optin_latest, ig.views_public_latest)
    f = ig.found
    out["organic_instagram"] = {**errors(ig.org_est[f], truth[f]), "share_of_optin_posts_found": float(f.mean())}
    band = pd.cut(ig.preboost_age_detected, [2, 6, 13, 999], labels=["low (day 3-6)", "medium (day 7-13)", "high (day 14+)"])
    out["organic_instagram_by_confidence"] = {str(k): errors(ig.org_est[f & (band == k)], truth[f & (band == k)])
                                              for k in band.cat.categories}
    pub = ig.views_public_latest
    k = f & (pub > truth)       # true paid must be positive to score a relative error
    out["paid_instagram"] = errors((pub - ig.org_est)[k], (pub - truth)[k])
    tt = b[(b.platform == "Tiktok") & b.found & (b.organic_views_method == KNOWN)]
    out["tiktok_vs_known_date"] = errors(tt.org_est, tt.organic_views_est)
    for g, lab in (("ig_organic", "instagram"), ("tt_organic", "tiktok")):
        o = d[d.grp == g]
        out[f"false_alarm_{lab}"] = {"posts": int(len(o)), "found": int(o.found.sum()), "rate": float(o.found.mean())}
    return out


def main():
    d, C = frame()
    out = {"params": PARAMS, "series_pulled": "2026-10-08", "table": TABLE,
           "posts": {str(k): int(v) for k, v in d.groupby("grp").size().items()},
           "tune": score(d[d.half == "tune"], C), "test": score(d[d.half == "test"], C)}
    json.dump(out, open("results/boost_start_eval.json", "w"), indent=1, default=float)
    print(json.dumps(out["test"], indent=1, default=float))


if __name__ == "__main__":
    main()
