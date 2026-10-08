"""Why the Instagram model misses large boosts: compare missed boosts with caught boosts and organic posts on public
signals and on SocAPI (an independent paid-plays source that is never a model input).

Inputs : results/model_v2_test_predictions.csv (train_oof + test rows), results/model_v2_report.json, data/v2_raw.psv,
         data/socapi_ig.psv (psrk|api_total|api_ig|api_fb; latest SocAPI play counts per Instagram post, BIRA SocAPI fact)
Output : results/ig_miss_audit.json (aggregates only)
Run    : python3 -m detector.ig_miss_audit
"""
import json

import numpy as np
import pandas as pd

from detector.features_v2 import load


def main(H=30):
    r = json.load(open("results/model_v2_report.json"))
    thr = r["models"][f"Instagram_h{H}"]["threshold"]
    d = load().set_index("psrk")
    s = pd.read_csv("data/socapi_ig.psv", sep="|", header=None, names=["psrk", "api_total", "api_ig", "api_fb"], dtype={"psrk": str})
    s = s.set_index("psrk")
    # SocAPI paid share: plays not explained by Instagram organic or Facebook cross-posts, vs Instagram plays (sql/05 rule: > 25%)
    s["soc_gap"] = (s.api_total - s.api_ig - s.api_fb.fillna(0)) / s.api_ig.where(s.api_ig > 0)
    p = pd.read_csv("results/model_v2_test_predictions.csv", dtype={"psrk": str})
    out = {"horizon_days": H, "threshold": thr, "socapi_rule": "non-organic SocAPI plays > 25% of Instagram plays", "splits": {}}
    for split, name in (("train_oof", "train (out-of-fold)"), ("test", "locked test")):
        t = p[(p.platform == "Instagram") & (p.H == H) & (p.split == split)].copy()
        t["flag"] = t.p >= thr
        t = t.join(d[["v30", "l30", "followers", "o30"]], on="psrk").join(s.soc_gap, on="psrk")
        t["group"] = np.select([t.y == 0, t.material & t.flag, t.material & ~t.flag],
                               ["organic", "large boost, caught", "large boost, missed"], "smaller boost")
        rows = {}
        for g, x in t.groupby("group"):
            soc = x.soc_gap.dropna()
            rows[g] = {"posts": int(len(x)), "paid_share_optin_median": float((1 - x.o30).median()),
                       "likes_per_view_median": float((x.l30 / x.v30).median()),
                       "views_per_follower_median": float((x.v30 / x.followers).median()),
                       "with_socapi": int(len(soc)), "socapi_paid_over_25pct": int((soc > 0.25).sum())}
        out["splits"][name] = rows
    json.dump(out, open("results/ig_miss_audit.json", "w"), indent=1)
    for k, v in out["splits"].items():
        print(k)
        print(pd.DataFrame(v).T.round(3).to_string())


if __name__ == "__main__":
    main()
