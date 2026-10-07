"""TikTok and the curve method back-test.

TikTok opt-in views include Spark Ad views, so there is no organic ground truth after a boost.
We therefore (1) find which paid metric explains the public lift, (2) compare the subtraction estimate
with the pre-boost x organic-curve estimate, and (3) back-test the curve method on UNBOOSTED posts
(pretend a boost started on day 3 / 7 / 14 and predict day 30), for both platforms.

Inputs: data/recon_tiktok_postflight.csv, data/detector_dataset_v2.psv, reconcile/organic_curve_*.csv
Output: results/reconciliation_tiktok.json
"""
import json

import numpy as np
import pandas as pd

from detector.features import load as load_detector
from reconcile.analyze import curve_factor, err_summary, ratio_summary


def main():
    out = {}
    t = pd.read_csv("data/recon_tiktok_postflight.csv", dtype={"psrk": str})
    for c in ["pub", "first_spend", "od_pre", "od_last"]:
        t[c] = pd.to_datetime(t[c])
    t = t[(t.plays_same_app > 0) & t.v_pub_pre.notna()]
    lift = t.v_pub_last - t.v_pub_pre
    out["tiktok_posts"] = int(len(t))
    out["tiktok_metric_match"] = {"Video plays (starts)": ratio_summary(lift / t.plays_same_app),
                                  "Impressions": ratio_summary(lift / t.impr_same_app),
                                  "2-second plays": ratio_summary(lift / t.plays_2s)}
    out["tiktok_optin_equals_public"] = ratio_summary(t.v_priv_last / t.v_pub_last)
    out["tiktok_paid_share_of_public"] = ratio_summary(t.plays_same_app / t.v_pub_last)
    curve_tt = pd.read_csv("reconcile/organic_curve_tiktok.csv").set_index("age_days")["median_share_of_day120_views"]
    age_pre = (t.od_pre - t.pub).dt.days
    age_last = (t.od_last - t.pub).dt.days
    est_curve = t.v_pub_pre * curve_factor(curve_tt, age_pre, age_last)
    est_sub = t.v_pub_last - t.plays_same_app
    out["tiktok_subtraction_vs_curve"] = {
        "subtraction_negative_share": float((est_sub < 0).mean()),
        "subtraction_below_preboost_read_share": float((est_sub < t.v_pub_pre).mean()),
        "subtraction_over_curve": ratio_summary(est_sub / est_curve),
    }

    # back-test of the curve method on unboosted posts (detector NEG labels; days 0-30 only)
    d = load_detector()
    neg = d[(d.label == "NEG") & (d.v30 > 0)]
    curves = {"Instagram": pd.read_csv("reconcile/organic_curve_ig.csv").set_index("age_days")["median_share_of_day120_views"],
              "Tiktok": curve_tt}
    bt = {}
    for plat, g in neg.groupby("platform"):
        c = curves[plat]
        res = {}
        for day, col in [(3, "v3"), (7, "v7"), (14, "v14")]:
            ok = g[col] > 0
            est = g.loc[ok, col] * curve_factor(c, np.full(ok.sum(), day), np.full(ok.sum(), 30))
            res[f"read on day {day} -> predict day 30"] = err_summary(est, g.loc[ok, "v30"])
        bt[plat] = res
    out["curve_backtest_unboosted"] = bt
    # The curves were built from these same unboosted posts, so the back-test above is in-sample.
    # Out-of-sample version: 2-fold split by creator; the day-k -> day-30 growth factor (median of v30 / vk)
    # is fitted on one half of the creators and tested on the other half.
    cf = {}
    for plat, g in neg.groupby("platform"):
        creators = g.handle.unique()
        half = set(np.random.default_rng(7).permutation(creators)[: len(creators) // 2])
        fold = g.handle.isin(half).values
        res = {}
        for day, col in [(3, "v3"), (7, "v7"), (14, "v14")]:
            ok = (g[col] > 0).values
            est = np.full(len(g), np.nan)
            for f in (True, False):
                fit, test = ok & (fold == f), ok & (fold != f)
                est[test] = g[col].values[test] * np.median(g.v30.values[fit] / g[col].values[fit])
            res[f"read on day {day} -> predict day 30"] = err_summary(est[ok], g.v30.values[ok])
        cf[plat] = res
    out["curve_backtest_unboosted_creator_split"] = cf
    json.dump(out, open("results/reconciliation_tiktok.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
