"""Stale opt-in (plan amendment 4): how often the Instagram opt-in count stops updating, and what the fix changes.

Inputs : data/optin_staleness.psv (sql/13), data/optin_live.psv and data/optin_tier_changes.psv (amended sql/05 opt-in
         fields and tier changes), data/socapi_ig.psv, data/v2_raw.psv (+ extra), results/model_v2_test_predictions.csv,
         results/model_v2_report.json
Output : results/optin_staleness.json (aggregates only)
Run    : python3 -m detector.optin_staleness_summary
"""
import json

import numpy as np
import pandas as pd

from detector.features_v2 import STALE_COLS, STALE_DAYS, STALE_GROWTH, load
from detector.v2_relabel_check import socapi_paid


def main():
    st = pd.read_csv("data/optin_staleness.psv", sep="|", header=None, names=STALE_COLS, dtype={"psrk": str})
    live = pd.read_csv("data/optin_live.psv", sep="|", header=None, dtype={"psrk": str},
                       names=["psrk", "optin_ratio", "optin_ratio_latest", "optin_stale", "optin_freeze_age", "optin_views_frozen"])
    ch = pd.read_csv("data/optin_tier_changes.psv", sep="|", header=None, names=["psrk", "tier_new"], dtype=str)
    out = {"rule": f"opt-in unchanged >= {STALE_DAYS} days at the last read while public views grew >= {STALE_GROWTH:.0%}",
           "instagram_posts_2025_with_optin": int(len(st)),
           "stale_at_latest_read": int(((st.fza >= STALE_DAYS) & (st.pga >= STALE_GROWTH)).sum()),
           "all_dates_posts_with_optin": int(len(live)), "all_dates_stale_at_latest_read": int(live.optin_stale.astype(int).sum()),
           "evidence_tier_changes": {"from": "MEASURED_OPTIN_GAP", "posts": int(len(ch)), "to": ch.tier_new.value_counts().to_dict()}}
    d0, d1 = load(), load(optin_fix=True)
    ig = d0.platform == "Instagram"
    out["label_changes_instagram"] = {f"day_{H}": {"paid_before": int((d0.loc[ig, f"lab{H}"] == "P").sum()),
                                                   "paid_after": int((d1.loc[ig, f"lab{H}"] == "P").sum()),
                                                   "paid_to_no_label": int(((d0[f"lab{H}"] == "P") & (d1[f"lab{H}"] == "")).sum())}
                                      for H in (14, 30, 60)}
    # independent check: SocAPI paid plays on dropped vs kept paid labels (day 30, v2 predictions file)
    soc = socapi_paid()
    thr = json.load(open("results/model_v2_report.json"))["models"]["Instagram_h30"]["threshold"]
    p = pd.read_csv("results/model_v2_test_predictions.csv", dtype={"psrk": str})
    new = d1.set_index("psrk").lab30
    out["socapi_check_day30"] = {}
    for split in ("train_oof", "test"):
        t = p[(p.platform == "Instagram") & (p.H == 30) & (p.split == split)].copy()
        t["new"] = new.reindex(t.psrk).values
        t["soc"] = t.psrk.map(soc)
        t["flag"] = t.p >= thr
        res = {}
        for name, g in (("dropped", t[(t.y == 1) & (t.new == "")]), ("kept_paid", t[(t.y == 1) & (t.new == "P")]), ("organic", t[t.y == 0])):
            w = g.soc.dropna()
            res[name] = {"posts": int(len(g)), "with_socapi": int(len(w)), "socapi_paid": int(w.sum()),
                         "v2_model_flags_share": float(g.flag.mean()) if len(g) else np.nan}
        out["socapi_check_day30"]["train" if split == "train_oof" else "locked_test"] = res
    json.dump(out, open("results/optin_staleness.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:2500])


if __name__ == "__main__":
    main()
