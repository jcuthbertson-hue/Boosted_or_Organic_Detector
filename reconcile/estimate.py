"""Organic view estimate per post, as used by the daily pipeline (pipeline/run_daily.py).

Method (chosen and tested in reconcile/analyze.py and reconcile/analyze_tiktok.py):
  1. Instagram opt-in post: organic = opt-in (private) views. They are organic only. Measured, not estimated.
     If the opt-in count stopped changing >= 7 days ago while public grew >= 5% (stale opt-in, plan amendment 4),
     organic = opt-in at the freeze x organic growth from the freeze age to today (same curve as method 2).
  2. Boosted post with a public read before the first spend day:
       organic = pre-boost public read x organic growth from that age to today (median curve of unboosted posts),
       capped at today's public views.
     Instagram test vs opt-in truth: median error 11.5%, 77% of posts within 25% (117 boosted posts).
  3. Boosted post with no read before the first spend day: not separable (NULL).
  4. No paid evidence: organic = public views.
We do NOT use "public - paid metric". It fails per post (median error 70% with the best metric, impressions)
because paid is a median 94% of public views on boosted posts, so a small paid error is a large organic error.

Confidence comes from the back-test on unboosted posts (creator split): pre-boost read on day 14+ -> about 5-10%
median error; day 7-13 -> about 10-20%; before day 7 -> 15-30% or more.
"""
import numpy as np
import pandas as pd

CURVES = {"Instagram": "reconcile/organic_curve_ig.csv", "Tiktok": "reconcile/organic_curve_tiktok.csv"}
MAX_AGE = 130


def load_curves():
    out = {}
    for platform, path in CURVES.items():
        c = pd.read_csv(path).set_index("age_days")["median_share_of_day120_views"]
        out[platform] = c.reindex(range(0, MAX_AGE + 1)).interpolate().ffill().bfill().values
    return out


def growth(curve, age_from, age_to):
    """Organic growth factor between two ages (>= 1). NaN where either age is unknown."""
    f, t = np.asarray(age_from, float), np.asarray(age_to, float)
    bad = np.isnan(f) | np.isnan(t)
    a0 = np.clip(np.where(bad, 0, f), 0, MAX_AGE).astype(int)
    a1 = np.clip(np.where(bad, 0, t), 0, MAX_AGE).astype(int)
    return np.where(bad, np.nan, np.maximum(curve[a1] / curve[a0], 1.0))


def confidence(age_pre):
    return np.select([age_pre >= 14, age_pre >= 7], ["high", "medium"], "low")


def estimate(df, curves=None):
    """df columns: platform, boosted (bool), views_public_latest, views_private_latest, age_latest,
    views_before_first_spend, preboost_age_days. Returns a frame with organic_views_est / _method / _confidence."""
    curves = curves or load_curves()
    d = df.reset_index(drop=True)
    est = pd.Series(np.nan, index=d.index)
    method = pd.Series("", index=d.index, dtype=object)
    conf = pd.Series("", index=d.index, dtype=object)

    no_paid = ~d.boosted.astype(bool)
    est[no_paid] = d.views_public_latest[no_paid]
    method[no_paid] = "public views (no paid evidence)"
    conf[no_paid] = "n/a"

    stale = d["optin_stale"].fillna(False).astype(bool) if "optin_stale" in d else pd.Series(False, index=d.index)
    optin = d.boosted.astype(bool) & (d.platform == "Instagram") & (d.views_private_latest > 0)
    live = optin & ~stale
    # capped at public views: the two latest totals can come from different reads
    est[live] = np.minimum(d.views_private_latest[live], d.views_public_latest[live].fillna(np.inf))
    method[live] = "opt-in private views (organic only)"
    conf[live] = "measured"
    nan = pd.Series(np.nan, index=d.index)
    frozen = (optin & stale & (d.get("optin_views_frozen", nan) > 0) & d.get("optin_freeze_age", nan).notna()
              & d.age_latest.notna())
    if frozen.any():
        g = growth(curves["Instagram"], d.optin_freeze_age[frozen], d.age_latest[frozen])
        est[frozen] = np.minimum(d.optin_views_frozen[frozen] * g, d.views_public_latest[frozen].fillna(np.inf))
        method[frozen] = "opt-in until it stopped updating, then organic curve"
        conf[frozen] = confidence(d.optin_freeze_age[frozen].values)
    optin = live | frozen

    rest = d.boosted.astype(bool) & ~optin
    has_pre = rest & (d.views_before_first_spend > 0) & d.preboost_age_days.notna() & d.age_latest.notna()
    for platform, curve in curves.items():
        m = has_pre & (d.platform == platform)
        if m.any():
            g = growth(curve, d.preboost_age_days[m], d.age_latest[m])
            est[m] = np.minimum(d.views_before_first_spend[m] * g, d.views_public_latest[m].fillna(np.inf))
            method[m] = "pre-boost public read x organic curve"
            conf[m] = confidence(d.preboost_age_days[m].values)
    none = rest & ~has_pre
    method[none] = "not separable: boosted before the first public read"
    conf[none] = "none"
    return pd.DataFrame({"organic_views_est": est.round(), "organic_views_method": method,
                         "organic_views_confidence": conf})
