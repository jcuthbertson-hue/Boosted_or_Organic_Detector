"""Plan amendment 4, step 1: re-measure the UNCHANGED v2 models with corrected Instagram labels (stale opt-in).
No model, feature or threshold changes; only the labels of posts whose opt-in count froze are corrected.

Inputs : results/model_v2_test_predictions.csv, results/model_v2_report.json, data/v2_raw.psv (+ v2_extra, optin_staleness)
Output : results/model_v2_relabel.json (aggregates only)
Run    : python3 -m detector.v2_relabel_check
"""
import json

import numpy as np
import pandas as pd

from detector.features_v2 import load
from detector.train_v2 import evaluate


def socapi_paid(path="data/socapi_ig.psv"):
    s = pd.read_csv(path, sep="|", header=None, names=["psrk", "api_total", "api_ig", "api_fb"], dtype={"psrk": str}).set_index("psrk")
    return ((s.api_total - s.api_ig - s.api_fb.fillna(0)) / s.api_ig.where(s.api_ig > 0)) > 0.25


def main():
    rep = json.load(open("results/model_v2_report.json"))
    r = rep["models"]
    soc = socapi_paid()
    fixed_preds = []
    preds = pd.read_csv("results/model_v2_test_predictions.csv", dtype={"psrk": str})
    f = load(optin_fix=True).set_index(["psrk", "platform"])
    out = {"note": "v2 models unchanged; Instagram labels corrected for stale opt-in (plan amendment 4)", "models": {}}
    for (pf, H, split), g in preds[preds.split != "train_oof"].groupby(["platform", "H", "split"]):
        k = list(zip(g.psrk, g.platform))
        lab = np.array([f[f"lab{H}"].get(x, "") for x in k])
        ov = np.array([f[f"ov{H}"].get(x, np.nan) if f"ov{H}" in f else np.nan for x in k], dtype=float)
        keep = np.isin(lab, ["P", "N"])
        g = g[keep].copy()
        g["y"] = (lab[keep] == "P").astype(int)
        inw = g.spend_day.between(-3, H - 2)
        g["material"] = (g.y == 1) if pf == "Tiktok" else (g.y == 1) & ((ov[keep] <= 0.60) | inw)
        thr = r[f"{pf}_h{H}"]["threshold"]
        if g.y.nunique() < 2:
            continue
        m = evaluate(g.y.values, g.p.values, thr, g)
        m["dropped_posts"] = int((~keep).sum())
        out["models"][f"{pf}_h{H}_{split}"] = m
        rep["models"][f"{pf}_h{H}"][split] = m
        fixed_preds.append(g)
        if pf == "Instagram":
            # stress test: dropped posts that SocAPI shows as paid are added back as paid (and material)
            dropped = preds[(preds.platform == pf) & (preds.H == H) & (preds.split == split)][~keep].copy()
            back = dropped[dropped.psrk.map(soc).eq(True).values].assign(y=1, material=True)
            gs = pd.concat([g, back])
            ms = evaluate(gs.y.values, gs.p.values, thr, gs)
            out["models"][f"{pf}_h{H}_{split}"]["stress_socapi_paid_added_back"] = {
                "added": int(len(back)), **{k: ms[k] for k in ("precision", "recall", "f1", "material_recall")}, "ci95": ms["ci95"]}
        print(f"{pf} day {H} {split}: posts {m['n']} (paid {m['n_pos']}, dropped {m['dropped_posts']}) | precision {m['precision']:.3f} "
              f"recall {m['recall']:.3f} F1 {m['f1']:.3f} | large boosts caught {m['material_recall']:.3f} "
              f"(95% {m['ci95']['material_recall'][0]:.2f}-{m['ci95']['material_recall'][1]:.2f}) | AUC {m['roc_auc']:.3f}")
    json.dump(out, open("results/model_v2_relabel.json", "w"), indent=1, default=float)
    rep["labels"] = "corrected (plan amendment 4)"
    json.dump(rep, open("results/model_v2_report_fixed.json", "w"), indent=1, default=float)
    pd.concat(fixed_preds).to_csv("results/model_v2_test_predictions_fixed.csv", index=False)   # row-level, git-ignored


if __name__ == "__main__":
    main()
