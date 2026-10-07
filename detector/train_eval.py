"""Train and test the boost detector with a locked test set.

Protocol (one model per platform):
  1. Labeled posts only (POS / NEG). TEST = published on or after TEST_CUTOFF. TRAIN = before.
     The test set is not touched until step 5.
  2. Model selection on TRAIN only: StratifiedGroupKFold (5 folds), grouped by creator handle, so the
     same creator is never in a training fold and its validation fold. Score = PR-AUC.
  3. Decision threshold chosen on TRAIN out-of-fold scores (max F1). Not tuned on TEST.
  4. Refit the selected model on all TRAIN rows.
  5. Score TEST once. Report ROC-AUC, PR-AUC, precision, recall, F1, Brier score, with bootstrap
     95% intervals. Also report the fixed hand rules (views > followers AND ER < 1%; views > followers) on the same TEST rows.
  6. Overfitting check: TRAIN in-sample AUC vs TRAIN cross-validated AUC vs TEST AUC.
  7. Robustness: client-holdout (GroupKFold by organization) on TRAIN, and TEST slices
     (creators not seen in TRAIN; Instagram ad-link positives only).

Run from repo root:  python3 -m detector.train_eval
"""
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from detector.features import FEATURES, LABEL_ONLY, build, load

warnings.filterwarnings("ignore", category=FutureWarning)
TEST_CUTOFF = pd.Timestamp("2026-07-01")
RNG = 7
N_BOOT = 1000


def candidates():
    """Small, fixed search space. Every candidate imputes missing values with the TRAIN median."""
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    c = {}
    for C in [0.01, 0.1, 1.0, 10.0]:
        c[f"logistic_regression(C={C})"] = make_pipeline(imp(), StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    for C in [0.3, 1.0, 3.0]:
        c[f"svm_rbf(C={C})"] = make_pipeline(imp(), StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale", probability=True, random_state=RNG))
    for leaf in [5, 20]:
        c[f"random_forest(min_leaf={leaf})"] = make_pipeline(imp(), RandomForestClassifier(n_estimators=400, min_samples_leaf=leaf, random_state=RNG, n_jobs=-1))
    for lr in [0.03, 0.1]:
        for leaves in [7, 15]:
            c[f"gradient_boosting(lr={lr},leaves={leaves})"] = HistGradientBoostingClassifier(
                learning_rate=lr, max_leaf_nodes=leaves, max_iter=200, l2_regularization=1.0, random_state=RNG)
    return c


def rules(d):
    return {
        "rule: views>followers AND ER<1%": ((d.vtf30 > 1) & (d.er30 < 0.01)).astype(int).values,
        "rule: views>followers": (d.vtf30 > 1).astype(int).values,
    }


def oof_scores(model, X, y, groups, splitter):
    oof = np.full(len(y), np.nan)
    for tr, va in splitter.split(X, y, groups):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    return oof


def scores(y, p, thr):
    y = np.asarray(y); p = np.asarray(p); flag = (p >= thr).astype(int)
    out = {"n": int(len(y)), "n_pos": int(y.sum())}
    two = len(np.unique(y)) > 1
    out["roc_auc"] = roc_auc_score(y, p) if two else np.nan
    out["pr_auc"] = average_precision_score(y, p) if two else np.nan
    out["precision"] = precision_score(y, flag, zero_division=0)
    out["recall"] = recall_score(y, flag, zero_division=0)
    out["f1"] = f1_score(y, flag, zero_division=0)
    out["brier"] = brier_score_loss(y, np.clip(p, 0, 1)) if two else np.nan
    return out


def bootstrap_ci(y, p, thr, n=N_BOOT, seed=RNG):
    rng = np.random.default_rng(seed); y = np.asarray(y); p = np.asarray(p)
    keys = ["roc_auc", "pr_auc", "precision", "recall", "f1"]; draws = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        s = scores(y[i], p[i], thr)
        for k in keys:
            draws[k].append(s[k])
    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in draws.items()}


def best_f1_threshold(y, p):
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])


def run_platform(d, platform, report):
    data = d[(d.platform == platform) & d.label.isin(["POS", "NEG"]) & (d.v30 > 0) & (d.followers > 0)].copy()
    train = data[data.pub < TEST_CUTOFF].reset_index(drop=True)
    test = data[data.pub >= TEST_CUTOFF].reset_index(drop=True)
    Xtr, ytr, Xte, yte = train[FEATURES], train.y, test[FEATURES], test.y
    assert not set(Xtr.columns) & set(LABEL_ONLY)
    r = {"platform": platform,
         "train": {"n": len(train), "n_pos": int(ytr.sum()), "pub_from": str(train.pub.min().date()), "pub_to": str(train.pub.max().date())},
         "test": {"n": len(test), "n_pos": int(yte.sum()), "pub_from": str(test.pub.min().date()), "pub_to": str(test.pub.max().date())},
         "creators_in_both": int(len(set(train.handle) & set(test.handle)))}

    # 2) model selection on TRAIN only (creator-grouped CV)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
    sel = []
    oof_by = {}
    for name, model in candidates().items():
        oof = oof_scores(model, Xtr, ytr, train.handle, cv)
        oof_by[name] = oof
        sel.append({"model": name, "cv_roc_auc": roc_auc_score(ytr, oof), "cv_pr_auc": average_precision_score(ytr, oof)})
    sel = pd.DataFrame(sel).sort_values("cv_pr_auc", ascending=False)
    best_name = sel.iloc[0].model
    r["selection_table"] = sel.round(4).to_dict("records")
    r["selected_model"] = best_name

    # 3) threshold from TRAIN out-of-fold scores
    thr = best_f1_threshold(ytr, oof_by[best_name])
    r["threshold"] = thr

    # 4) refit on all TRAIN
    model = clone(candidates()[best_name]).fit(Xtr, ytr)
    p_train_in = model.predict_proba(Xtr)[:, 1]

    # robustness: client-holdout CV on TRAIN (organization groups)
    n_org = train.org.nunique()
    oof_org = oof_scores(candidates()[best_name], Xtr, ytr, train.org.fillna("NA"), GroupKFold(n_splits=min(5, n_org)))

    # 5) TEST, scored once
    p_test = model.predict_proba(Xte)[:, 1]
    r["overfit_check"] = {
        "train_in_sample_roc_auc": roc_auc_score(ytr, p_train_in),
        "train_cv_by_creator_roc_auc": roc_auc_score(ytr, oof_by[best_name]),
        "train_cv_by_client_roc_auc": roc_auc_score(ytr, oof_org),
        "test_roc_auc": roc_auc_score(yte, p_test),
    }
    r["test_model"] = {**scores(yte, p_test, thr), "ci95": bootstrap_ci(yte, p_test, thr)}
    r["test_rules"] = {}
    for name, flag in rules(test).items():
        r["test_rules"][name] = {**scores(yte, flag, 0.5), "ci95": bootstrap_ci(yte, flag, 0.5)}

    # slices
    seen = set(train.handle)
    m_new = ~test.handle.isin(seen)
    r["test_slice_new_creators"] = scores(yte[m_new], p_test[m_new.values], thr)
    if platform == "Instagram":
        m_ad = (test.y == 0) | (test.pos_source_ad == "ad_link")
        r["test_slice_ad_link_positives_only"] = scores(yte[m_ad], p_test[m_ad.values], thr)
    # calibration (deciles of predicted probability)
    bins = pd.qcut(p_test, q=10, duplicates="drop")
    r["test_calibration"] = (pd.DataFrame({"p": p_test, "y": yte}).groupby(bins, observed=True)
                             .agg(mean_pred=("p", "mean"), share_boosted=("y", "mean"), n=("y", "size"))
                             .reset_index(drop=True).round(3).to_dict("records"))
    # permutation importance on TEST (which inputs the model relies on)
    pi = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=10, random_state=RNG)
    r["test_permutation_importance_pr_auc"] = dict(sorted(zip(FEATURES, pi.importances_mean.round(4)), key=lambda kv: -kv[1]))

    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "threshold": thr, "platform": platform,
                 "train_pub_to": r["train"]["pub_to"]}, f"models/boost_detector_{platform.lower()}.joblib")
    test_out = test[["psrk", "platform", "org", "pub", "label", "pos_source_ad"]].copy()
    test_out["p_boost"] = p_test
    # TRAIN scores for the full metric report (detector/evaluate.py): out-of-fold (creator-grouped CV) and in-sample
    train_out = train[["psrk", "platform", "org", "pub", "label", "pos_source_ad"]].copy()
    train_out["p_boost_oof"] = oof_by[best_name]
    train_out["p_boost_in_sample"] = p_train_in
    report[platform] = r
    return test_out, train_out


def main():
    os.makedirs("results", exist_ok=True)
    d = build(load())
    report = {"test_cutoff": str(TEST_CUTOFF.date()), "features": FEATURES, "label_only_columns": LABEL_ONLY}
    outs = [run_platform(d, p, report) for p in ["Tiktok", "Instagram"]]
    pd.concat([o[0] for o in outs]).to_csv("results/test_predictions.csv", index=False)
    pd.concat([o[1] for o in outs]).to_csv("results/train_oof_predictions.csv", index=False)

    def fmt(x):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in x.items()}
    json.dump(report, open("results/detector_report.json", "w"), indent=2, default=float)
    for p in ["Tiktok", "Instagram"]:
        r = report[p]
        print(f"\n===== {p} =====  train {r['train']}  test {r['test']}  creators in both: {r['creators_in_both']}")
        print("selection (TRAIN, creator-grouped CV):")
        print(pd.DataFrame(r["selection_table"]).to_string(index=False))
        print("selected:", r["selected_model"], "| threshold (from TRAIN):", r["threshold"])
        print("overfit check:", fmt(r["overfit_check"]))
        t = r["test_model"]
        print("TEST model:", fmt({k: v for k, v in t.items() if k != "ci95"}))
        print("   95% CI:", {k: (round(a, 3), round(b, 3)) for k, (a, b) in t["ci95"].items()})
        for name, s in r["test_rules"].items():
            print(f"TEST {name}:", fmt({k: v for k, v in s.items() if k not in ('ci95', 'brier')}))
        print("TEST slice new creators:", fmt(r["test_slice_new_creators"]))
        if "test_slice_ad_link_positives_only" in r:
            print("TEST slice ad-link positives only:", fmt(r["test_slice_ad_link_positives_only"]))
        print("calibration:", r["test_calibration"])
        print("permutation importance (TEST, PR-AUC drop):", r["test_permutation_importance_pr_auc"])


if __name__ == "__main__":
    main()
