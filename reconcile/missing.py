"""Can we predict the parts we do not measure? (aggregates only)

Three tests and one count:
  paid_from_organic_estimate : Instagram paid views = public (Nimble) - organic estimate (pre-boost read x organic curve),
                               against true paid = public - opt-in organic. Same 117 posts as the pre-boost test in analyze.py.
  facebook_factor            : Facebook part of the total (SocAPI total - SocAPI Instagram) = k x paid Facebook video plays.
                               k is fitted on the other posts each time (leave one out), 20 posts with SocAPI data.
  coverage                   : boosted posts in the daily table, by how we know they are boosted and how we get organic
                               (creator data, known start date, start found from the view jump, or missing, with why).

Inputs : data/recon_panel_ig_full.psv (+ pub dates), reconcile/organic_curve_ig.csv, latest outputs/paid_classification_*.parquet
Output : results/predict_missing.json
Run    : python3 -m reconcile.missing
"""
import glob
import json

import numpy as np
import pandas as pd

from reconcile.analyze import curve_factor
from reconcile.panel import load


def errors(est, truth):
    e = ((np.asarray(est, float) - np.asarray(truth, float)) / np.asarray(truth, float))
    e = pd.Series(e).replace([np.inf, -np.inf], np.nan).dropna()
    return {"posts": int(len(e)), "median_abs_error": float(e.abs().median()),
            "within_10pct": float((e.abs() <= .10).mean()), "within_25pct": float((e.abs() <= .25).mean())}


def closure(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    return {"posts": int(len(r)), "median": float(r.median()),
            "within_10pct": float((np.abs(r - 1) <= .10).mean()), "within_25pct": float((np.abs(r - 1) <= .25).mean())}


def coverage(path):
    d = pd.read_parquet(path)
    d.columns = [c.lower() for c in d.columns]
    m_ = d.organic_views_method.fillna("")
    kind = np.select([m_.str.startswith("opt-in"), m_ == "pre-boost public read x organic curve",
                      m_ == "pre-boost read found from the view jump x organic curve", m_.str.startswith("not separable")],
                     ["creator_data", "estimated_known_start", "estimated_view_jump", "missing"], "")
    reason = {"not separable: boosted before the first public read": "boosted_before_first_read",
              "not separable: first jump before day 3": "jump_before_day_3",
              "not separable: no clear jump in the first 90 days": "no_clear_jump",
              "not separable: fewer than 2 public reads in the first 90 days": "too_few_reads",
              "not separable: no public reads in the first 90 days": "too_few_reads"}
    out = {"source": path.split("/")[-1].replace(".parquet", "") + " (daily table, aggregates only)"}
    for p, g in d.groupby("post_platform"):
        b = g.is_paid == True  # noqa: E712  (is_paid is None for posts the model cannot score)
        k = pd.Series(kind, index=d.index)[g.index][b]
        miss = m_[g.index][b][k == "missing"].map(reason)
        out[p] = {"posts": int(len(g)), "boosted_by_record": int((b & (g.paid_basis == "evidence")).sum()),
                  "boosted_by_model": int((b & (g.paid_basis == "model")).sum()), "not_scored": int((g.paid_status == "NOT_SCORED").sum()),
                  "organic_from": {x: int((k == x).sum()) for x in ("creator_data", "estimated_known_start", "estimated_view_jump", "missing")},
                  "missing_by_reason": {x: int((miss == x).sum()) for x in ("boosted_before_first_read", "jump_before_day_3", "no_clear_jump", "too_few_reads")},
                  "missing_by_basis": {"record": int(((k == "missing") & (g.paid_basis[b] == "evidence")).sum()),
                                       "model": int(((k == "missing") & (g.paid_basis[b] == "model")).sum())}}
        assert sum(out[p]["organic_from"].values()) == out[p]["boosted_by_record"] + out[p]["boosted_by_model"], p
        assert sum(out[p]["missing_by_reason"].values()) == out[p]["organic_from"]["missing"], p
    return out


def main():
    d = load()
    pub = pd.read_csv("data/recon_panel_pub_dates.psv", sep="|", header=None, names=["psrk", "pub"], dtype=str)
    pub["pub"] = pd.to_datetime(pub.pub)
    d = d.merge(pub, on="psrk", how="left")
    d["age"] = (d.od - d.pub).dt.days
    curve = pd.read_csv("reconcile/organic_curve_ig.csv").set_index("age_days")["median_share_of_day120_views"]

    # same posts and reads as the pre-boost test in reconcile/analyze.py
    pf = d[d.dsl >= 2].sort_values("od").groupby("psrk").tail(1)
    pre = (d[d.dsf < 0].sort_values("od").groupby("psrk").tail(1)[["psrk", "vp", "age"]]
           .rename(columns={"vp": "vp_pre", "age": "age_pre"}))
    j = pf.merge(pre, on="psrk")
    j["org_est"] = j.vp_pre * curve_factor(curve, j.age_pre, j.age)
    j = j[j.vp > j.vr]                     # true paid = public - opt-in must be positive to score a relative error
    paid = {"read_rule": "latest read at least 2 days after the last ad day; pre-boost read = latest read before the first spend day",
            "paid_share_median": float(((j.vp - j.vr) / j.vp).median()),
            "public_minus_organic_estimate": errors(j.vp - j.org_est, j.vp - j.vr),
            "paid_instagram_impressions": errors(j.impr_ig, j.vp - j.vr),
            "organic_estimate": errors(j.org_est, j.vr)}

    s = d[d.api_total.notna() & (d.dsl >= 2)].sort_values("od").groupby("psrk").tail(1).reset_index(drop=True)
    k = (s.api_total - s.api_ig) / s.starts_fb
    loo = np.array([k.drop(i).median() for i in s.index])
    other = s.starts_ot.fillna(0)
    fb = {"k_median_all_posts": float(k.median()), "reads_from": str(s.od.min().date()), "reads_to": str(s.od.max().date()),
          "total_from_nimble_plus_k_fb_plays": errors(s.vp + loo * s.starts_fb, s.api_total),
          "total_from_nimble_plus_fb_plays_no_factor": errors(s.vp + s.starts_fb, s.api_total),
          "equation_with_factor": closure((s.vr + s.impr_ig + loo * s.starts_fb + other) / s.api_total)}

    paths = sorted(p for p in glob.glob("outputs/paid_classification_*.parquet") if "TEST" not in p)
    out = {"paid_from_organic_estimate": paid, "facebook_factor": fb, "coverage": coverage(paths[-1]) if paths else None}
    json.dump(out, open("results/predict_missing.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
