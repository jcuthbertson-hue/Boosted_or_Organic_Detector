"""Full classification report for the boost detector (all standard outputs, per platform).

Reads the scores that detector/train_eval.py wrote. It does NOT refit or retune anything, so the locked test set
stays scored once. The threshold is the one chosen on TRAIN out-of-fold scores.

Outputs per platform (TEST unless marked):
  confusion matrix (counts + rates); accuracy, balanced accuracy, precision, recall, specificity, NPV, FPR, FNR,
  F1, F2, MCC, Cohen's kappa; ROC curve + AUC; PR curve + average precision; KS; calibration curve, Brier,
  Brier skill, log loss, ECE; score histograms; cumulative gains; threshold sweep; operating points chosen on TRAIN;
  train in-sample vs train CV vs test; 95% intervals from a creator-cluster bootstrap; the two hand rules;
  slices; Andrew's confirmed paid posts (gold check); LLM comparison.

Inputs : results/test_predictions.csv, results/train_oof_predictions.csv, results/detector_report.json,
         data/detector_dataset_v2.psv (creator handle + rule inputs), data/andrew_gold.csv (sql/08),
         results/llm_vs_ml.csv, results/llm_vs_ml_paired_auc.csv
Output : results/classification_metrics.json  (aggregates only: no post ids, creators or client names)

Run from repo root:  python3 -m detector.evaluate
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, cohen_kappa_score, log_loss,
                             matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve)

from detector.features import build, load
from detector.train_eval import TEST_CUTOFF

N_BOOT = 1000
RNG = 7
PLATFORMS = ["Tiktok", "Instagram"]


def threshold_metrics(y, p, thr):
    y = np.asarray(y, int); flag = (np.asarray(p) >= thr).astype(int)
    tp = int(((flag == 1) & (y == 1)).sum()); fp = int(((flag == 1) & (y == 0)).sum())
    fn = int(((flag == 0) & (y == 1)).sum()); tn = int(((flag == 0) & (y == 0)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")
    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    f = lambda beta: div((1 + beta ** 2) * prec * rec, beta ** 2 * prec + rec) if tp else 0.0
    spec = div(tn, tn + fp)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": div(tp + tn, len(y)), "balanced_accuracy": (rec + spec) / 2,
            "precision": prec, "recall": rec, "specificity": spec, "npv": div(tn, tn + fn),
            "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp), "f1": f(1), "f2": f(2),
            "mcc": float(matthews_corrcoef(y, flag)) if len(set(flag)) > 1 else 0.0,
            "kappa": float(cohen_kappa_score(y, flag)), "flagged_share": div(tp + fp, len(y))}


def prob_metrics(y, p):
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    prev = y.mean(); brier = brier_score_loss(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "pr_auc_no_skill": float(prev), "ks": float(np.max(tpr - fpr)), "brier": float(brier),
            "brier_skill": float(1 - brier / (prev * (1 - prev))), "log_loss": float(log_loss(y, p)), "ece": float(ece)}


def all_metrics(y, p, thr):
    return {"n": int(len(y)), "n_pos": int(np.sum(y)), "prevalence": float(np.mean(y)),
            **prob_metrics(y, p), **threshold_metrics(y, p, thr)}


def cluster_bootstrap(y, p, thr, groups, keys, n=N_BOOT):
    """95% intervals, resampling whole creators (posts from one creator are not independent)."""
    rng = np.random.default_rng(RNG)
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in uniq}
    draws = {k: [] for k in keys}
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[i])) < 2:
            continue
        m = all_metrics(y[i], p[i], thr)
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def curves(y, p):
    y = np.asarray(y, int); p = np.asarray(p, float)
    fpr, tpr, _ = roc_curve(y, p, drop_intermediate=True)
    prec, rec, _ = precision_recall_curve(y, p)
    keep = np.unique(np.linspace(0, len(rec) - 1, min(len(rec), 150)).astype(int))
    order = np.argsort(-p, kind="stable"); ys = y[order]
    gains = [{"top_share": k / 100, "boosted_captured": float(ys[:int(round(len(y) * k / 100))].sum() / ys.sum())}
             for k in range(5, 101, 5)]
    q = pd.qcut(p, 10, duplicates="drop")
    cal = (pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True)
           .agg(mean_pred=("p", "mean"), share_boosted=("y", "mean"), n=("y", "size")).reset_index(drop=True))
    edges = np.linspace(0, 1, 21)
    hist = {lab: np.histogram(p[y == v], bins=edges)[0].tolist() for lab, v in [("boosted", 1), ("organic", 0)]}
    return {"roc": {"fpr": np.round(fpr, 4).tolist(), "tpr": np.round(tpr, 4).tolist()},
            "pr": {"recall": np.round(rec[keep], 4).tolist(), "precision": np.round(prec[keep], 4).tolist()},
            "gains": gains, "calibration": cal.round(4).to_dict("records"),
            "histogram": {"edges": edges.round(2).tolist(), **hist}}


def sweep(y, p):
    return [{"threshold": round(t, 2), **{k: v for k, v in threshold_metrics(y, p, t).items()
                                          if k in ("precision", "recall", "f1", "specificity", "flagged_share", "mcc")}}
            for t in np.arange(0.05, 0.96, 0.05)]


def operating_points(y_tr, p_tr, y_te, p_te, thr_f1):
    """Thresholds chosen on TRAIN out-of-fold scores only, then read on TEST."""
    grid = np.round(np.arange(0.02, 0.99, 0.01), 2)
    tr = [threshold_metrics(y_tr, p_tr, t) for t in grid]
    youden = grid[int(np.argmax([m["recall"] + m["specificity"] - 1 for m in tr]))]
    hp = [t for t, m in zip(grid, tr) if m["precision"] >= 0.95]
    hr = [t for t, m in zip(grid, tr) if m["recall"] >= 0.90]
    pts = {"max F1 on train (default)": thr_f1, "Youden J on train": float(youden)}
    if hp:
        pts["precision >= 95% on train"] = float(min(hp))
    if hr:
        pts["recall >= 90% on train"] = float(max(hr))
    out = []
    for name, t in pts.items():
        a, b = threshold_metrics(y_tr, p_tr, t), threshold_metrics(y_te, p_te, t)
        out.append({"rule": name, "threshold": float(t),
                    "train_cv": {k: a[k] for k in ("precision", "recall", "f1")},
                    "test": {k: b[k] for k in ("precision", "recall", "f1", "specificity", "flagged_share", "tp", "fp", "fn", "tn")}})
    return out


def andrew_gold(gold, te, tr):
    """Andrew's confirmed paid posts (PAID_MEDIA_UNIFIED.ext_p3_organic_post_id). Counts and scores, no ids."""
    out = {"source": "DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED.ext_p3_organic_post_id (filled from 2026-09-15)",
           "posts_tagged": int(len(gold))}
    for pf, name, rule in [("meta", "Instagram", "11-character ad-name token = shortcode"),
                           ("tiktok", "Tiktok", "TIKTOK_ITEM_ID = video id (Spark Ad)")]:
        g = gold[gold.pf == pf]
        tracked = g[g.in_bira]
        sc = pd.concat([te.assign(split="test"), tr.assign(split="train")])
        sc = sc[sc.platform == name][["psrk", "label", "p_boost", "split", "threshold"]]
        j = tracked.merge(sc, on="psrk", how="left")
        rows = []
        for _, r in j.sort_values("first_spend").iterrows():
            in_window = pd.notna(r.days_pub_to_spend) and r.days_pub_to_spend <= 28
            rows.append({"post_type": r.post_type, "published": str(r.pub)[:7], "days_publish_to_first_spend": r.days_pub_to_spend,
                         "boost_inside_30_day_feature_window": bool(in_window),
                         "found_by_our_link_rule": bool(r.found_by_our_link),
                         "in_detector_data": bool(pd.notna(r.p_boost)), "split": r.split if pd.notna(r.p_boost) else None,
                         "our_label": r.label if pd.notna(r.p_boost) else None,
                         "model_score": None if pd.isna(r.p_boost) else round(float(r.p_boost), 3),
                         "model_flag": None if pd.isna(r.p_boost) else bool(r.p_boost >= r.threshold)})
        out[name] = {"tagged": int(len(g)), "tracked_in_bira": int(len(tracked)),
                     "link_rule": rule,
                     "found_by_link_rule_all": int(g.found_by_our_link.sum()),
                     "found_by_link_rule_tracked": int(tracked.found_by_our_link.sum()),
                     "posts": rows}
    return out


def main():
    rep = json.load(open("results/detector_report.json"))
    te = pd.read_csv("results/test_predictions.csv", dtype={"psrk": str})
    tr = pd.read_csv("results/train_oof_predictions.csv", dtype={"psrk": str})
    d = build(load())[["psrk", "platform", "handle", "vtf30", "er30"]]
    te = te.merge(d, on=["psrk", "platform"], how="left")
    tr = tr.merge(d, on=["psrk", "platform"], how="left")
    assert te.handle.notna().all() and tr.handle.notna().all()
    for df in (te, tr):
        df["y"] = (df.label == "POS").astype(int)
        df["threshold"] = df.platform.map(lambda p: rep[p]["threshold"])

    keys = ["roc_auc", "pr_auc", "brier", "accuracy", "balanced_accuracy", "precision", "recall",
            "specificity", "npv", "f1", "mcc", "kappa"]
    out = {"test_cutoff": str(TEST_CUTOFF.date()), "n_boot": N_BOOT,
           "bootstrap": "creator-cluster (resample creators, keep all their posts)", "platforms": {}}
    for pf in PLATFORMS:
        a, b, r = tr[tr.platform == pf], te[te.platform == pf], rep[pf]
        thr = r["threshold"]
        m = {"selected_model": r["selected_model"], "threshold_from_train": thr,
             "train": r["train"], "test": r["test"], "creators_in_both": r["creators_in_both"],
             "selection_table": r["selection_table"],
             "permutation_importance_test_pr_auc": r["test_permutation_importance_pr_auc"]}
        m["test_metrics"] = all_metrics(b.y, b.p_boost, thr)
        m["test_ci95"] = cluster_bootstrap(b.y, b.p_boost, thr, b.handle, keys)
        m["train_cv_metrics"] = all_metrics(a.y, a.p_boost_oof, thr)
        m["train_in_sample_metrics"] = all_metrics(a.y, a.p_boost_in_sample, thr)
        m["overfit_check"] = r["overfit_check"]
        m["curves_test"] = curves(b.y, b.p_boost)
        m["curves_train_cv"] = {"roc": curves(a.y, a.p_boost_oof)["roc"], "pr": curves(a.y, a.p_boost_oof)["pr"]}
        m["threshold_sweep_test"] = sweep(b.y, b.p_boost)
        m["threshold_sweep_train_cv"] = sweep(a.y, a.p_boost_oof)
        m["operating_points"] = operating_points(a.y, a.p_boost_oof, b.y, b.p_boost, thr)
        m["always_organic_baseline_accuracy"] = float(1 - b.y.mean())
        rules = {"Tom's rule: views > followers AND engagement < 1%": ((b.vtf30 > 1) & (b.er30 < 0.01)).astype(int),
                 "views > followers": (b.vtf30 > 1).astype(int)}
        m["rules_test"] = {k: {**threshold_metrics(b.y, f, 0.5), "roc_auc": float(roc_auc_score(b.y, f))}
                           for k, f in rules.items()}
        seen = set(a.handle)
        sl = {"creators not seen in train": b[~b.handle.isin(seen)], "creators seen in train": b[b.handle.isin(seen)]}
        if pf == "Instagram":
            sl["positives from ad link only (+ all organic)"] = b[(b.y == 0) | (b.pos_source_ad == "ad_link")]
            sl["positives from opt-in gap only (+ all organic)"] = b[(b.y == 0) | (b.pos_source_ad != "ad_link")]
        m["slices_test"] = {k: {**{kk: v for kk, v in all_metrics(s.y, s.p_boost, thr).items()
                                   if kk in ("n", "n_pos", "roc_auc", "pr_auc", "precision", "recall", "f1", "mcc")}}
                            for k, s in sl.items() if s.y.nunique() == 2}
        out["platforms"][pf] = m

    if os.path.exists("data/andrew_gold.csv"):
        gold = pd.read_csv("data/andrew_gold.csv", dtype={"psrk": str, "pid": str})
        gold["in_bira"] = gold.in_bira.astype(str).str.lower() == "true"
        gold["found_by_our_link"] = gold.found_by_our_link.astype(str).str.lower() == "true"
        out["andrew_gold"] = andrew_gold(gold, te, tr)

    llm = pd.read_csv("results/llm_vs_ml.csv")
    out["llm_comparison"] = {"rows": llm.round(4).to_dict("records"),
                             "paired_auc": pd.read_csv("results/llm_vs_ml_paired_auc.csv").round(4).to_dict("records")}

    json.dump(out, open("results/classification_metrics.json", "w"), indent=1, default=float)
    for pf in PLATFORMS:
        m = out["platforms"][pf]; t = m["test_metrics"]; ci = m["test_ci95"]
        print(f"\n== {pf} ({m['selected_model']}, threshold {m['threshold_from_train']}) ==")
        print(f"confusion  TP {t['tp']}  FP {t['fp']}  FN {t['fn']}  TN {t['tn']}")
        for k in keys + ["ks", "log_loss", "ece", "brier_skill", "f2"]:
            c = ci.get(k)
            print(f"  {k:18s} {t[k]:.3f}" + (f"  [{c[0]:.3f}, {c[1]:.3f}]" if c else ""))
        print("  operating points:", [(o["rule"], o["threshold"], round(o["test"]["precision"], 3), round(o["test"]["recall"], 3)) for o in m["operating_points"]])
        print("  slices:", {k: (v["n_pos"], round(v["roc_auc"], 3)) for k, v in m["slices_test"].items()})
    if "andrew_gold" in out:
        for pf in PLATFORMS:
            g = out["andrew_gold"][pf]
            print(f"\nAndrew gold {pf}: tagged {g['tagged']}, tracked {g['tracked_in_bira']}, "
                  f"found by our link {g['found_by_link_rule_all']} (tracked {g['found_by_link_rule_tracked']})")
            for p in g["posts"]:
                if p["in_detector_data"]:
                    print("   ", p)


if __name__ == "__main__":
    main()
