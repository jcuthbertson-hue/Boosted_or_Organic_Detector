"""Month-by-month check inside the TRAINING period (no locked-test or fresh post is used).

For each test month 2026-03..2026-06: train on every earlier post (v2.2 configuration and threshold rule), score that
month. Answers two questions: how much does F1 swing from month to month, and does more training data help?
Output: results/month_folds.json (aggregates only). Run: python3 -m detector.month_folds"""
import json, numpy as np, pandas as pd
from sklearn.base import clone
from detector.features_v2 import load
from detector.train_v2 import dataset, candidates, oof, pick_threshold, at, TRAIN_END
d = load(optin_fix=True, horizon_followers="data/v2_followers.psv")
r = json.load(open("results/model_v2_2_report.json"))["models"]
rows = []
for pf, H in [("Instagram", 14), ("Instagram", 30), ("Tiktok", 30), ("Tiktok", 14)]:
    k = f"{pf}_h{H}"
    X, meta = dataset(d, H, pf, r[k]["feature_set"])
    base = candidates()[r[k]["model"]]
    for m in ["2026-03", "2026-04", "2026-05", "2026-06"]:
        start = pd.Timestamp(m + "-01"); end = start + pd.offsets.MonthBegin(1)
        trm = (meta.pub < start).values
        tem = ((meta.pub >= start) & (meta.pub < end)).values
        if meta.y[tem].nunique() < 2:
            continue
        # threshold from train out-of-fold scores (same rule as production), then one fit on all train rows
        p_oof = oof(base, X[trm], meta.y[trm], meta.creator[trm])
        thr = pick_threshold(meta.y[trm], p_oof)
        model = clone(base).fit(X[trm], meta.y[trm])
        p = model.predict_proba(X[tem])[:, 1]
        a = at(meta.y[tem].values, p, thr, meta.material[tem].values)
        rows.append({"platform": pf, "H": H, "test_month": m, "train_posts": int(trm.sum()), "train_paid": int(meta.y[trm].sum()),
                     "test_posts": int(tem.sum()), "test_paid": int(meta.y[tem].sum()),
                     **{x: round(a[x], 3) for x in ("precision", "recall", "f1", "material_recall")}})
        print(rows[-1], flush=True)
t = pd.DataFrame(rows)
summary = {f"{pf}_h{H}": {"months": int(len(g)), "f1_min": float(g.f1.min()), "f1_max": float(g.f1.max()),
                           "f1_median": float(g.f1.median()), "corr_train_size_f1": float(g.f1.corr(g.train_posts))}
           for (pf, H), g in t.groupby(["platform", "H"])}
json.dump({"folds": rows, "summary": summary}, open("results/month_folds.json", "w"), indent=1)
