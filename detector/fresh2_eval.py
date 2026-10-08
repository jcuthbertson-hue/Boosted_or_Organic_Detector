"""Plan amendment 6: the next clean test ("fresh-2") = posts published after 2026-09-24, never used before.

Run after a NEW pull of sql/11, sql/12, sql/13 and sql/14 into one folder (default data/fresh2/, same file names as
data/), when those posts have reached the model day (day 14 from about 2026-10-21, day 30 from about 2026-11-06).
Creator features need each creator's earlier posts, so pull all posts, not only the new ones.

Scores v2.2 (in use) and the v2.3 candidate (if its models exist) once, with the C1-C4 and C6 rules on the same posts,
and writes results/fresh2_eval.json (aggregates only). Decision rule (amendment 6): the set that passes more targets
is used; a tie keeps v2.2.

  python3 -m detector.fresh2_eval --data data/fresh2 --data-date 2026-10-22
"""
import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd

from detector.features_v2 import load
from detector.train_v2 import at, ece, evaluate, features

FRESH2_FROM = pd.Timestamp("2026-09-25")
SETS = {"v2.2": "models/boost_detector_v2_2", "v2.3": "models/boost_detector_v2_3"}
GOAL_F1 = {"Instagram": 0.80, "Tiktok": 0.93}


def score_set(d, prefix, H, pf):
    path = f"{prefix}_{pf.lower()}_h{H}.joblib"
    if not os.path.exists(path):
        return None
    b = joblib.load(path)
    X, ok = features(d, H, b["feature_set"])
    lab = d[f"lab{H}"]
    m = ok & (d.platform == pf) & lab.isin(["P", "N"]) & (d.pub >= FRESH2_FROM)
    if m.sum() == 0:
        return None
    g = d.loc[m, ["psrk", "platform", "pub", "creator", "spend_day", "tag"]].copy()
    g["y"] = (lab[m] == "P").astype(int)
    o = d.loc[m, f"ov{H}"] if f"ov{H}" in d else d.loc[m, f"o{H}"]
    g["material"] = (g.y == 1) if pf == "Tiktok" else (g.y == 1) & ((o <= 0.60) | g.spend_day.between(-3, H - 2))
    g["p"] = b["model"].predict_proba(X.loc[m, b["features"]])[:, 1]
    g["thr"] = b["threshold"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/fresh2")
    ap.add_argument("--data-date", required=True)
    a = ap.parse_args()
    f = lambda n: os.path.join(a.data, n)
    d = load(f("v2_raw.psv"), extra=f("v2_extra.psv"), optin_fix=True, stale=f("optin_staleness.psv"),
             horizon_followers=f("v2_followers.psv"))
    out = {"fresh2_from": str(FRESH2_FROM.date()), "data_date": a.data_date, "sets": {}}
    for name, prefix in SETS.items():
        res, preds = {}, {}
        for H in (14, 30, 60):
            for pf in ("Tiktok", "Instagram"):
                g = score_set(d, prefix, H, pf)
                if g is None or g.y.nunique() < 2:
                    continue
                preds[(pf, H)] = g
                m = evaluate(g.y.values, g.p.values, float(g.thr.iloc[0]), g)
                res[f"{pf}_h{H}"] = m
        if not res:
            continue
        T = {}
        for pf in GOAL_F1:
            k = f"{pf}_h30" if f"{pf}_h30" in res else f"{pf}_h14"          # day 30 when fresh-2 posts are old enough
            if k not in res:
                continue
            r = res[k]
            T[pf] = {"model": k, "C1": bool(r["precision"] >= 0.90 and r["material_recall"] >= 0.90),
                     "C2": bool(r["f1"] >= GOAL_F1[pf]), "C3": bool(r["ece"] <= 0.05)}
            if (pf, 30) in preds and (pf, 14) in preds:                    # C4 on the same posts
                j = preds[(pf, 30)].merge(preds[(pf, 14)], on="psrk", suffixes=("_30", "_14"))
                f1 = {H: at(j[f"y_{H}"], j[f"p_{H}"], float(j[f"thr_{H}"].iloc[0]))["f1"] for H in (30, 14)} if len(j) else {}
                T[pf]["C4"] = bool(f1 and f1[30] - f1[14] <= 0.05)
            tags = [g[(g.tag == 1) & g.spend_day.between(-3, H - 2)] for (p_, H), g in preds.items() if p_ == pf]
            tags = pd.concat(tags) if tags else pd.DataFrame()
            T[pf]["C6"] = bool(len(tags) == 0 or (tags.p >= tags.thr).all())
        out["sets"][name] = {"metrics": res, "targets": T,
                             "targets_passed": int(sum(v for t in T.values() for k_, v in t.items() if k_.startswith("C")))}
    if out["sets"]:
        best = max(out["sets"], key=lambda n: (out["sets"][n]["targets_passed"], n == "v2.2"))
        out["decision"] = {"use": best, "rule": "more targets passed; a tie keeps v2.2"}
    json.dump(out, open("results/fresh2_eval.json", "w"), indent=1, default=float)
    for n, s in out["sets"].items():
        print(n, "targets passed:", s["targets_passed"], {k: {c: v for c, v in t.items() if c != "model"} for k, t in s["targets"].items()})
        for k, m in s["metrics"].items():
            print("  ", k, "n", m["n"], "paid", m["n_pos"], "P %.3f R %.3f F1 %.3f caught %.3f" % (m["precision"], m["recall"], m["f1"], m["material_recall"]))
    print("decision:", out.get("decision"))


if __name__ == "__main__":
    main()
