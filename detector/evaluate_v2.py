"""Every standard classification output for model v2 (same layout as detector/evaluate.py, plus day-14 and fresh).

Inputs : results/model_v2_report.json, results/model_v2_test_predictions.csv (git-ignored), results/model_v2_selection.csv,
         data/v2_raw.psv (sql/11), models/boost_detector_v2_*.joblib, data/tagged_paid_posts.csv (sql/08),
         results/classification_metrics.json (v1, for the LLM comparison and the v1-vs-v2 table)
Output : results/classification_metrics_v2.json (aggregates only)
Run    : python3 -m detector.evaluate_v2
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from detector.evaluate import all_metrics, cluster_bootstrap, curves, operating_points, sweep, threshold_metrics
from detector.features_v2 import load
from detector.train_v2 import TEST_END, TRAIN_END, candidates, dataset

PLATFORMS = ["Tiktok", "Instagram"]
KEYS = ["roc_auc", "pr_auc", "brier", "accuracy", "balanced_accuracy", "precision", "recall", "specificity", "npv", "f1", "mcc", "kappa"]


def main():
    rep = json.load(open("results/model_v2_report.json"))
    preds = pd.read_csv("results/model_v2_test_predictions.csv", dtype={"psrk": str})
    sel = pd.read_csv("results/model_v2_selection.csv")
    v1 = json.load(open("results/classification_metrics.json"))
    d = load()
    out = {"test_cutoff": str(TRAIN_END.date()), "test_end": str(TEST_END.date()), "fresh": rep["fresh"], "n_boot": 1000,
           "bootstrap": "creator-cluster (resample creators, keep all their posts)", "platforms": {}, "day14": {}, "day60": {}, "v1_vs_v2": {}}
    for pf in PLATFORMS:
        for H in (30, 14, 60):
            if f"{pf}_h{H}" not in rep["models"]:
                continue
            r = rep["models"][f"{pf}_h{H}"]
            thr = r["threshold"]
            g = preds[(preds.platform == pf) & (preds.H == H)]
            tr, te, fr = g[g.split == "train_oof"], g[g.split == "test"], g[g.split == "fresh"]
            bundle = joblib.load(f"models/boost_detector_v2_{pf.lower()}_h{H}.joblib")
            X, meta = dataset(d, H, pf, r["feature_set"])
            trm = (meta.pub < TRAIN_END).values
            tem = ((meta.pub >= TRAIN_END) & (meta.pub <= TEST_END)).values
            p_in = bundle["model"].predict_proba(X[trm][bundle["features"]])[:, 1]
            m = {"selected_model": r["model"] + (" + isotonic" if r.get("calibrated") else ""), "feature_set": r["feature_set"],
                 "threshold_from_train": thr,
                 "train": {"n": int(len(tr)), "n_pos": int(tr.y.sum()), "pub_from": str(meta.pub[trm].min().date()), "pub_to": str(meta.pub[trm].max().date())},
                 "test": {"n": int(len(te)), "n_pos": int(te.y.sum()), "pub_from": str(meta.pub[tem].min().date()), "pub_to": str(meta.pub[tem].max().date())},
                 "creators_in_both": int(len(set(meta.creator[trm]) & set(meta.creator[tem])))}
            m["test_metrics"] = all_metrics(te.y.values, te.p.values, thr)
            m["test_ci95"] = cluster_bootstrap(te.y.values, te.p.values, thr, te.creator.values, KEYS)
            m["train_cv_metrics"] = all_metrics(tr.y.values, tr.p.values, thr)
            m["train_in_sample_metrics"] = all_metrics(meta.y[trm].values, p_in, thr)
            # client-holdout CV on train (GroupKFold by client hash), same model settings, no calibration
            base = candidates()[r["model"]]
            Xt, yt, ct = X[trm][bundle["features"]], meta.y[trm], meta.client[trm].fillna(-1)
            oof_c = np.full(len(yt), np.nan)
            for a, b in GroupKFold(n_splits=min(5, ct.nunique())).split(Xt, yt, ct):
                oof_c[b] = clone(base).fit(Xt.iloc[a], yt.iloc[a]).predict_proba(Xt.iloc[b])[:, 1]
            m["overfit_check"] = {"train_in_sample_roc_auc": m["train_in_sample_metrics"]["roc_auc"],
                                  "train_cv_by_creator_roc_auc": m["train_cv_metrics"]["roc_auc"],
                                  "train_cv_by_client_roc_auc": float(roc_auc_score(yt, oof_c)),
                                  "test_roc_auc": m["test_metrics"]["roc_auc"]}
            m["curves_test"] = curves(te.y.values, te.p.values)
            ct_ = curves(tr.y.values, tr.p.values)
            m["curves_train_cv"] = {"roc": ct_["roc"], "pr": ct_["pr"]}
            m["threshold_sweep_test"] = sweep(te.y.values, te.p.values)
            m["threshold_sweep_train_cv"] = sweep(tr.y.values, tr.p.values)
            m["operating_points"] = operating_points(tr.y.values, tr.p.values, te.y.values, te.p.values, thr)
            m["always_organic_baseline_accuracy"] = float(1 - te.y.mean())
            m["material"] = {"test_material_posts": int(te.material.sum()),
                             "test_material_recall": float((te.p[te.material] >= thr).mean()) if te.material.any() else float("nan")}
            # hand rules on the same test posts (day-30 public values)
            raw = d.set_index(["psrk", "platform"])
            k = list(zip(te.psrk, te.platform))
            v30 = np.array([raw.v30.get(x, np.nan) for x in k]); fol = np.array([raw.followers.get(x, np.nan) for x in k])
            eng = np.array([np.nansum([raw.l30.get(x, np.nan), raw.c30.get(x, np.nan)]) for x in k])
            vtf, er = v30 / fol, eng / v30
            rules = {"Hand rule: views > followers AND engagement < 1%": ((vtf > 1) & (er < 0.01)).astype(int), "views > followers": (vtf > 1).astype(int)}
            m["rules_test"] = {n: {**threshold_metrics(te.y.values, f, 0.5), "roc_auc": float(roc_auc_score(te.y.values, f))} for n, f in rules.items()}
            seen = set(meta.creator[trm])
            new = ~te.creator.isin(seen)
            sl = {"creators not seen in train": te[new], "creators seen in train": te[~new],
                  "material boosts only (+ all organic)": te[(te.y == 0) | te.material]}
            if pf == "Instagram":
                inw = te.spend_day.between(-3, H - 2)
                sl["positives from ad link or tag (+ all organic)"] = te[(te.y == 0) | inw]
                sl["positives from opt-in gap only (+ all organic)"] = te[(te.y == 0) | ~inw]
            m["slices_test"] = {n: {kk: v for kk, v in all_metrics(s.y.values, s.p.values, thr).items()
                                    if kk in ("n", "n_pos", "roc_auc", "pr_auc", "precision", "recall", "f1", "mcc")}
                                for n, s in sl.items() if s.y.nunique() == 2}
            if len(fr) and fr.y.nunique() == 2:
                m["fresh_metrics"] = all_metrics(fr.y.values, fr.p.values, thr)
                m["fresh_ci95"] = cluster_bootstrap(fr.y.values, fr.p.values, thr, fr.creator.values, KEYS)
                m["fresh_material_recall"] = float((fr.p[fr.material] >= thr).mean()) if fr.material.any() else float("nan")
            s_ = sel[(sel.H == H) & (sel.platform == pf)].sort_values("cv_pr_auc", ascending=False)
            m["selection_table"] = [{"model": f'{a} [{b}]', "cv_roc_auc": c, "cv_pr_auc": e} for a, b, c, e in
                                    zip(s_.model, s_.features, s_.cv_roc_auc, s_.cv_pr_auc)]
            base_model = bundle["model"]
            pi = permutation_importance(base_model, X[tem][bundle["features"]], meta.y[tem], scoring="average_precision", n_repeats=10, random_state=7)
            m["permutation_importance_test_pr_auc"] = dict(sorted(zip(bundle["features"], pi.importances_mean.round(4)), key=lambda kv: -kv[1]))
            out[{30: "platforms", 14: "day14", 60: "day60"}[H]][pf] = m
        # v1 vs v2 on the v1 locked test window (different label sets: v2 adds tags and relaxed read rules)
        a, b = v1["platforms"][pf]["test_metrics"], out["platforms"][pf]["test_metrics"]
        out["v1_vs_v2"][pf] = {k: {"v1": a[k], "v2": b[k]} for k in ("roc_auc", "pr_auc", "precision", "recall", "f1", "mcc", "ece", "n", "n_pos")}

    # gold check: posts in the paid team's post-ID tag, scored by v2 (any split)
    if os.path.exists("data/tagged_paid_posts.csv"):
        gold = pd.read_csv("data/tagged_paid_posts.csv", dtype={"psrk": str, "pid": str})
        gold = gold[gold.in_bira.astype(str).str.lower() == "true"]
        gold["platform"] = gold.pf.map({"meta": "Instagram", "tiktok": "Tiktok"})
        sc = preds.sort_values("H", ascending=False).drop_duplicates(["psrk", "platform"])
        j = gold.merge(sc, on=["psrk", "platform"], how="left")
        rows = []
        for _, x in j.iterrows():
            H = x.H if pd.notna(x.H) else None
            thr = rep["models"][f"{x.platform}_h{int(H)}"]["threshold"] if H else None
            rows.append({"platform": x.platform, "post_type": x.post_type, "published": str(x.pub_x)[:7] if "pub_x" in x else None,
                         "days_publish_to_first_spend": x.days_pub_to_spend,
                         "in_window": bool(pd.notna(x.days_pub_to_spend) and H and -3 <= x.days_pub_to_spend <= H - 2),
                         "scored": bool(H), "horizon": int(H) if H else None, "split": x.split if H else None,
                         "model_score": round(float(x.p), 3) if H else None, "flagged": bool(H and x.p >= thr)})
        # aggregates only (no per-post rows in the public repo)
        g = pd.DataFrame(rows)
        out["tag_gold_v2"] = {pf_: {"tagged_in_bira": int(len(x)), "scored": int(x.scored.sum()),
                                    "boost_inside_window": int((x.scored & x.in_window).sum()),
                                    "flagged_inside_window": int((x.scored & x.in_window & x.flagged).sum()),
                                    "flagged_outside_window": int((x.scored & ~x.in_window & x.flagged).sum()),
                                    "scored_outside_window": int((x.scored & ~x.in_window).sum())}
                              for pf_, x in g.groupby("platform")} if len(g) else {}
    out["llm_comparison"] = v1["llm_comparison"]
    json.dump(out, open("results/classification_metrics_v2.json", "w"), indent=1, default=float)
    for pf in PLATFORMS:
        t = out["platforms"][pf]["test_metrics"]
        print(pf, "day-30 test:", {k: round(t[k], 3) for k in ("precision", "recall", "f1", "roc_auc", "ece")},
              "| material recall", round(out["platforms"][pf]["material"]["test_material_recall"], 3))
        for blk, name in (("day14", "day-14"), ("day60", "day-60")):
            if pf in out[blk]:
                t = out[blk][pf]["test_metrics"]
                print(pf, name, "test:", {k: round(t[k], 3) for k in ("precision", "recall", "f1", "roc_auc", "ece")},
                      "| material recall", round(out[blk][pf]["material"]["test_material_recall"], 3))


if __name__ == "__main__":
    main()
