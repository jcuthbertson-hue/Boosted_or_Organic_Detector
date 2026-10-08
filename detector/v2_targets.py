"""Score model v2 against the pre-registered targets (docs/IMPROVEMENT_PLAN.md). Run after `train_v2 --final`.

Writes results/model_v2_targets.json (aggregates only).
"""
import json

import numpy as np
import pandas as pd

from detector.features_v2 import build, load

DATA_DATE = pd.Timestamp("2026-10-07")


def main():
    r = json.load(open("results/model_v2_report.json"))
    m = r["models"]
    preds = pd.read_csv("results/model_v2_test_predictions.csv", dtype={"psrk": str})
    preds = preds[preds.split != "train_oof"]
    out = {"targets": {}}
    T = out["targets"]

    # C1: material recall >= 0.90 at precision >= 0.90 (locked test, day-30 model)
    T["C1"] = {pf: {"precision": m[f"{pf}_h30"]["test"]["precision"], "material_recall": m[f"{pf}_h30"]["test"]["material_recall"],
                    "ci95": {k: m[f"{pf}_h30"]["test"]["ci95"][k] for k in ("precision", "material_recall")}}
               for pf in ("Tiktok", "Instagram")}
    for v in T["C1"].values():
        v["pass"] = bool(v["precision"] >= 0.90 and v["material_recall"] >= 0.90)
    # C2: all-boost F1
    goal = {"Instagram": 0.80, "Tiktok": 0.93}
    T["C2"] = {pf: {"f1": m[f"{pf}_h30"]["test"]["f1"], "goal": g, "ci95": m[f"{pf}_h30"]["test"]["ci95"]["f1"],
                    "pass": bool(m[f"{pf}_h30"]["test"]["f1"] >= g)} for pf, g in goal.items()}
    # C3: calibration
    T["C3"] = {pf: {"ece": m[f"{pf}_h30"]["test"]["ece"], "pass": bool(m[f"{pf}_h30"]["test"]["ece"] <= 0.05)} for pf in goal}

    # C4: coverage among posts >= 14 days old with a public read by day 7, and day-14 vs day-30 F1 on the same test posts
    d = load()
    age = (DATA_DATE - d.pub).dt.days
    base = (age >= 14) & (d.a_min <= 7)
    _, ok30 = build(d, 30)
    _, ok14 = build(d, 14)
    scored = base & (ok30 | ok14)
    cov = {pf: float(scored[base & (d.platform == pf)].mean()) for pf in goal}
    cov["all"] = float(scored[base].mean())
    same = {}
    for pf in goal:
        a = preds[(preds.platform == pf) & (preds.split == "test") & (preds.H == 30)]
        b = preds[(preds.platform == pf) & (preds.split == "test") & (preds.H == 14)]
        j = a.merge(b, on="psrk", suffixes=("_30", "_14"))
        f1 = {}
        for H in (30, 14):
            y, p = j[f"y_{H}"].values, j[f"p_{H}"].values >= m[f"{pf}_h{H}"]["threshold"]
            tp, fp, fn = (p & (y == 1)).sum(), (p & (y == 0)).sum(), (~p & (y == 1)).sum()
            f1[H] = float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
        same[pf] = {"posts": int(len(j)), "f1_day30": f1[30], "f1_day14": f1[14], "gap": f1[30] - f1[14]}
    T["C4"] = {"coverage": cov, "posts_in_base": int(base.sum()), "untracked_note": "posts with no public read by day 7 are a data gap",
               "same_posts": same,
               "pass": bool(cov["all"] >= 0.85 and all(v["gap"] <= 0.05 for v in same.values()))}

    # C5: fresh holdout (day-14 model): C1 holds within its 95% interval
    T["C5"] = {}
    for pf in goal:
        f = m[f"{pf}_h14"].get("fresh")
        if not f:
            T["C5"][pf] = {"pass": False, "note": "no fresh labels"}
            continue
        T["C5"][pf] = {"n": f["n"], "n_pos": f["n_pos"], "precision": f["precision"], "material_recall": f["material_recall"],
                       "ci95": {k: f["ci95"][k] for k in ("precision", "material_recall")},
                       "pass": bool(f["ci95"]["precision"][1] >= 0.90 and f["ci95"]["material_recall"][1] >= 0.90)}

    # C6: every tagged post whose boost starts inside the window is flagged (test + fresh)
    g = []
    for (pf, H), grp in preds[preds.tag == 1].groupby(["platform", "H"]):
        inwin = grp[grp.spend_day.between(-3, H - 2)]
        thr = m[f"{pf}_h{H}"]["threshold"]
        g += [{"platform": pf, "H": int(H), "split": s, "score": round(float(p), 3), "flagged": bool(p >= thr)}
              for s, p in zip(inwin.split, inwin.p)]
    T["C6"] = {"posts": g, "pass": bool(g) and all(x["flagged"] for x in g)}

    out["all_pass"] = bool(all(v["pass"] if "pass" in v else all(x["pass"] for x in v.values()) for v in T.values()))
    json.dump(out, open("results/model_v2_targets.json", "w"), indent=1, default=float)
    for k, v in T.items():
        print(k, json.dumps(v, default=float)[:600])
    print("ALL PASS:", out["all_pass"])


if __name__ == "__main__":
    main()
