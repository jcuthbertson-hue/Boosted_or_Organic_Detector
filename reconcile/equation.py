"""Does organic + boosted = total seen on platform, for posts VN boosted itself? (aggregates only)

Uses the reconciliation panel (sql/06): one read per post at least 2 days after the last ad day, so paid delivery is final.
  Instagram side : creator organic (opt-in) + a paid Instagram metric  vs  Nimble public views (Instagram only)
  Full platform  : organic + paid Instagram impressions + paid Facebook plays (+ other placements)  vs  SocAPI total plays
                   (Instagram + Facebook), on the posts whose read has SocAPI data
Paid placement split: Snowflake query over all Meta ads linked to Instagram campaign posts (run 2026-10-08, see PLACEMENTS).

Inputs : data/recon_panel_ig_full.psv (+ pub dates), via reconcile.panel
Output : results/vn_boost_equation.json
Run    : python3 -m reconcile.equation
"""
import json

import numpy as np
import pandas as pd

from reconcile.panel import METRICS, load

# SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE x DIM_FACEBOOK_ADS__AD (ads linked to Instagram campaign posts by
# post key), summed by placement. Query ID 01c79964-0013-7819-0004-060222cdd9ae, run 2026-10-08.
PLACEMENTS = {"source": "Snowflake EDW Meta ad tables, ads linked to Instagram campaign posts, all dates; run 2026-10-08",
              "rows": [{"placement": "Facebook", "ads": 1155, "impressions": 649544453, "plays": 243757656},
                       {"placement": "Instagram", "ads": 1169, "impressions": 245842307, "plays": 187935416},
                       {"placement": "Audience Network", "ads": 15, "impressions": 72691, "plays": 65860},
                       {"placement": "Unknown", "ads": 742, "impressions": 21598, "plays": 17654},
                       {"placement": "Messenger", "ads": 4, "impressions": 3592, "plays": 3132}]}


BINS = np.arange(.5, 1.5001, .025)  # dot-plot bins for the per-post ratio; the end bins also hold everything beyond them


def ratio(lhs, rhs):
    return pd.Series(np.asarray(lhs, float) / np.asarray(rhs, float)).replace([np.inf, -np.inf], np.nan).dropna()


def closure(lhs, rhs):
    r = ratio(lhs, rhs)
    return {"posts": int(len(r)), "median": float(r.median()), "q25": float(r.quantile(.25)), "q75": float(r.quantile(.75)),
            "median_abs_gap": float((r - 1).abs().median()),
            "within_10pct": float(np.mean(np.abs(r - 1) <= .10)), "within_25pct": float(np.mean(np.abs(r - 1) <= .25))}


def dots(lhs, rhs):
    """post counts per 2.5-point bin of the ratio (a histogram, no per-post values)"""
    r = ratio(lhs, rhs)
    return {"counts": np.histogram(r.clip(BINS[0], BINS[-1] - 1e-9), BINS)[0].tolist(),
            "below_first": int((r < BINS[0]).sum()), "above_last": int((r >= BINS[-1]).sum())}


def main():
    d = load()
    pf = d[d.dsl >= 2].sort_values("od").groupby("psrk").tail(1)
    s = d[d.api_total.notna() & (d.dsl >= 2)].sort_values("od").groupby("psrk").tail(1)
    other = s.starts_ot.fillna(0)
    out = {"read_rule": "one read per post, at least 2 days after the last ad day",
           "instagram_side": {METRICS[m]: closure(pf.vr + pf[f"{m}_ig"], pf.vp) for m in ("impr", "starts", "v3s", "thru")},
           "full_platform": {
               "with paid Facebook video plays": closure(s.vr + s.impr_ig + s.api_fb + s.starts_fb + other, s.api_total),
               "with paid Facebook impressions": closure(s.vr + s.impr_ig + s.api_fb + s.impr_fb + other, s.api_total)},
           "socapi": {"posts": int(len(s)), "reads_from": str(s.od.min().date()), "reads_to": str(s.od.max().date()),
                      "instagram_plays_over_nimble_median": float((s.api_ig / s.vp).median()),
                      "total_over_nimble_median": float((s.api_total / s.vp).median()),
                      "facebook_share_of_total_median": float(((s.api_total - s.api_ig) / s.api_total).median()),
                      "facebook_only_field_filled": int((s.api_fb > 0).sum())},
           # all 20 posts added together: what the total is made of (big posts weigh more)
           "pooled_full_platform": {"organic": float(s.vr.sum()), "paid_instagram_impressions": float(s.impr_ig.sum()),
                                    "paid_facebook_plays": float(s.starts_fb.sum()), "paid_other_plays": float(other.sum()),
                                    "socapi_total": float(s.api_total.sum()), "nimble": float(s.vp.sum())},
           "placements": PLACEMENTS,
           "dots": {"bin_from": float(BINS[0]), "bin_width": .025,
                    "instagram_side": dots(pf.vr + pf.impr_ig, pf.vp),
                    "full_platform": dots(s.vr + s.impr_ig + s.api_fb + s.starts_fb + other, s.api_total)}}
    json.dump(out, open("results/vn_boost_equation.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("instagram_side", "full_platform", "socapi", "pooled_full_platform")}, indent=1))


if __name__ == "__main__":
    main()
