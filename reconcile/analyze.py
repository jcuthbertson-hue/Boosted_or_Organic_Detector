"""Tom's question: can we take the platform total, subtract a paid metric from Andrew's paid table,
and get organic views? Checked against opt-in (private) Instagram views, which are organic only.

Inputs (exported by sql/06_reconciliation_panel.sql and companions):
  data/recon_panel_ig_full.psv      post x day panel (public, opt-in, SocAPI, paid by placement)
  data/recon_panel_pub_dates.psv    psrk|publish date
  reconcile/organic_curve_ig.csv    median organic view curve (share of day-120 views by age), unboosted IG posts
Output: results/reconciliation.json

Run from repo root:  python3 -m reconcile.analyze
"""
import json
import os

import numpy as np
import pandas as pd

from reconcile.estimate import estimate
from reconcile.panel import METRICS, load

RNG = np.random.default_rng(7)


def err_summary(est, truth):
    est, truth = np.asarray(est, float), np.asarray(truth, float)
    ok = np.isfinite(est) & np.isfinite(truth) & (truth > 0)
    e = (est[ok] - truth[ok]) / truth[ok]
    return {"posts": int(ok.sum()),
            "median_error": float(np.median(e)),                   # signed: + means organic overstated
            "median_abs_error": float(np.median(np.abs(e))),
            "within_10pct": float(np.mean(np.abs(e) <= .10)),
            "within_25pct": float(np.mean(np.abs(e) <= .25)),
            "within_50pct": float(np.mean(np.abs(e) <= .50)),
            "negative_organic": float(np.mean(est[ok] < 0))}


def ratio_summary(r):
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    return {"n": int(len(r)), "median": float(r.median()), "q25": float(r.quantile(.25)), "q75": float(r.quantile(.75))}


def curve_factor(curve, age_from, age_to):
    c = curve.reindex(range(0, 131)).interpolate().ffill()
    a0 = np.clip(np.asarray(age_from), 0, 130).astype(int)
    a1 = np.clip(np.asarray(age_to), 0, 130).astype(int)
    return c.values[a1] / c.values[a0]


def group_errors(est, truth, size=10, n=2000):
    est, truth = np.asarray(est, float), np.asarray(truth, float)
    out = []
    for _ in range(n):
        g = RNG.choice(len(est), size, replace=False)
        out.append(np.clip(est[g], 0, None).sum() / truth[g].sum() - 1)
    out = np.array(out)
    return {"group_size": size, "median_error": float(np.median(out)), "q25": float(np.quantile(out, .25)),
            "q75": float(np.quantile(out, .75)), "within_25pct": float(np.mean(np.abs(out) <= .25))}


def main():
    d = load()
    pub = pd.read_csv("data/recon_panel_pub_dates.psv", sep="|", header=None, names=["psrk", "pub"], dtype=str)
    pub["pub"] = pd.to_datetime(pub.pub)
    d = d.merge(pub, on="psrk", how="left")
    d["age"] = (d.od - d.pub).dt.days
    curve = pd.read_csv("reconcile/organic_curve_ig.csv").set_index("age_days")["median_share_of_day120_views"]

    out = {"panel": {"rows": int(len(d)), "posts": int(d.psrk.nunique()),
                     "posts_by_tier": d.groupby("tier").psrk.nunique().to_dict(),
                     "obs_from": str(d.od.min().date()), "obs_to": str(d.od.max().date())}}

    # one row per post: the latest day at least 2 days after the last ad day (paid delivery is final)
    pf = d[d.dsl >= 2].sort_values("od").groupby("psrk").tail(1).copy()
    out["postflight_posts"] = int(len(pf))
    out["paid_share_of_public"] = ratio_summary(1 - pf.vr / pf.vp)

    # 1) which paid metric matches what the platform counts
    match = {"instagram_side": {}, "facebook_side": {}}
    for m, name in METRICS.items():
        match["instagram_side"][name] = ratio_summary((pf.vp - pf.vr) / pf[f"{m}_ig"])
    s = d[d.api_total.notna() & (d.dsl >= 2)]
    for m, name in METRICS.items():
        match["facebook_side"][name] = ratio_summary((s.api_total - s.api_ig - s.api_fb) / s[f"{m}_fb"])
    out["metric_match"] = match
    out["socapi_postflight"] = {"rows": int(len(s)), "posts": int(s.psrk.nunique())}

    # 2) recipes, scored per post against opt-in organic
    rec = {}
    for m, name in METRICS.items():
        rec[f"public - paid IG {name}"] = err_summary(pf.vp - pf[f"{m}_ig"], pf.vr)
    k = ((pf.vp - pf.vr) / pf.impr_ig).replace([np.inf, -np.inf], np.nan)
    loo = np.array([pf.loc[i, "vp"] - k.drop(i).median() * pf.loc[i, "impr_ig"] for i in pf.index])
    rec["public - k x paid IG impressions (k fitted on other posts)"] = err_summary(loo, pf.vr)
    sp = s.sort_values("od").groupby("psrk").tail(1)
    rec["SocAPI total - all paid video plays (naive)"] = err_summary(sp.api_total - sp.starts_all, sp.vr)
    rec["SocAPI total - all paid impressions (naive)"] = err_summary(sp.api_total - sp.impr_all, sp.vr)
    rec["SocAPI total - FB cross-post - IG impressions - FB video plays (best mix)"] = err_summary(
        sp.api_total - sp.api_fb - sp.impr_ig - sp.starts_fb - sp.starts_ot.fillna(0), sp.vr)

    pre = (d[d.dsf < 0].sort_values("od").groupby("psrk").tail(1)[["psrk", "vp", "vr", "age"]]
           .rename(columns={"vp": "vp_pre", "vr": "vr_pre", "age": "age_pre"}))
    j = pf.merge(pre, on="psrk")
    out["preboost_posts"] = int(len(j))
    out["preboost_public_equals_optin"] = ratio_summary(j.vr_pre / j.vp_pre)
    rec["pre-boost public read (no adjustment)"] = err_summary(j.vp_pre, j.vr)
    g = j.vr / j.vp_pre
    flat = np.array([g.drop(i).median() for i in j.index])
    rec["pre-boost read x flat growth (fitted on other posts)"] = err_summary(j.vp_pre * flat, j.vr)
    cf = curve_factor(curve, j.age_pre, j.age)
    j["est_curve"] = j.vp_pre * cf
    rec["pre-boost read x organic curve (no fitting on these posts)"] = err_summary(j.est_curve, j.vr)
    # the production function (reconcile/estimate.py, used by the daily pipeline) on the same posts; opt-in is
    # hidden so the function must use the curve
    prod = estimate(pd.DataFrame({"platform": "Instagram", "boosted": True, "views_public_latest": j.vp,
                                  "views_private_latest": np.nan, "age_latest": j.age,
                                  "views_before_first_spend": j.vp_pre, "preboost_age_days": j.age_pre}))
    rec["production function: reconcile/estimate.py"] = err_summary(prod.organic_views_est, j.vr)
    out["production_function_by_confidence"] = {
        c: err_summary(prod.organic_views_est[prod.organic_views_confidence == c], j.vr[(prod.organic_views_confidence == c).values])
        for c in ["high", "medium", "low"]}
    out["recipes"] = rec

    # 3) campaign-level (sum over 10 posts)
    pf["est_loo"] = loo
    out["campaign_level"] = {
        "subtraction: public - k x IG impressions": group_errors(pf.est_loo, pf.vr),
        "pre-boost read x organic curve": group_errors(j.est_curve, j.vr),
    }

    # 4) Andrew-confirmed posts (Sep 2026+, mostly still running): in-flight days, paid through that day
    a = d[(d.tier == "andrew_confirmed") & (d.dsf >= 0)]
    out["andrew_confirmed"] = {
        "posts": int(a.psrk.nunique()), "post_days": int(len(a)),
        "public - paid IG impressions": err_summary(a.vp - a.impr_ig, a.vr),
        "paid_share_of_public": ratio_summary(1 - a.vr / a.vp),
        "socapi_days": int(a.api_total.notna().sum()),
    }

    # 5) timing: in-flight days, paid through the day vs through the day before (video plays)
    inf = d[(d.dsf >= 1) & (d.dsl < 0)]
    out["timing_in_flight"] = {
        "post_days": int(len(inf)),
        "paid through same day": err_summary(inf.vp - inf.starts_ig, inf.vr),
        "paid through day before": err_summary(inf.vp - inf.starts_ig_p, inf.vr),
    }

    # error vs paid share: why subtraction fails
    pf["abs_err"] = np.abs((pf.vp - pf.impr_ig - pf.vr) / pf.vr)
    pf["paid_share"] = 1 - pf.vr / pf.vp
    bins = pd.cut(pf.paid_share, [-1, .5, .8, .9, .95, 1.01], labels=["<50%", "50-80%", "80-90%", "90-95%", ">95%"])
    out["error_by_paid_share"] = (pf.groupby(bins, observed=True)
                                    .agg(posts=("abs_err", "size"), median_abs_error=("abs_err", "median"))
                                    .reset_index().astype({"paid_share": str}).to_dict("records"))

    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/reconciliation.json", "w"), indent=2, default=float)
    print(json.dumps({k: out[k] for k in ["panel", "postflight_posts", "paid_share_of_public", "preboost_posts"]}, indent=1, default=float))
    print(pd.DataFrame(out["recipes"]).T.round(3).to_string())
    print(json.dumps(out["campaign_level"], indent=1, default=float))
    print(json.dumps(out["andrew_confirmed"], indent=1, default=float))
    print(json.dumps(out["timing_in_flight"], indent=1, default=float))
    print(out["error_by_paid_share"])


if __name__ == "__main__":
    main()
