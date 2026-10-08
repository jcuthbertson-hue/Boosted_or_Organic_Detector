"""Model v2: horizon models (day 14, day 30) on richer public time-series features.

  python3 -m detector.train_v2 --develop     TRAIN only: creator-grouped CV for every candidate; prints the table.
  python3 -m detector.train_v2 --final       Fit the chosen candidates on TRAIN, score the locked TEST (published
                                             2026-07-01..09-09) and the FRESH holdout (2026-09-10..09-24, day-14 model)
                                             once; save models/ and results/model_v2_report.json.

Targets are pre-registered in docs/IMPROVEMENT_PLAN.md. No paid / opt-in field is a feature (detector/features_v2.py).
"""
import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

from detector.features_v2 import build, creator_history, creator_norms, jump_features, load

warnings.filterwarnings("ignore")
RNG = 7
TRAIN_END = pd.Timestamp("2026-07-01")
TEST_END = pd.Timestamp("2026-09-09")
FRESH = (pd.Timestamp("2026-09-10"), pd.Timestamp("2026-09-24"))
PLATFORMS = ["Tiktok", "Instagram"]
# output names; --v21 switches to the v2.1 run (plan amendment 4: corrected Instagram labels, Instagram only)
RUN = {"tag": "v2", "results": "results/model_v2", "models": "models/boost_detector_v2", "optin_fix": False, "followers": None}
HORIZONS = [60, 30, 14]          # day 60 added in plan amendment 3 (exploratory; C1-C6 use days 30 and 14)


def candidates():
    imp = lambda: SimpleImputer(strategy="median", add_indicator=True)
    c = {}
    for lr in (0.03, 0.06):
        for leaves in (15, 31):
            for msl in (20, 40):
                c[f"hgb(lr={lr},leaves={leaves},min_leaf={msl})"] = HistGradientBoostingClassifier(
                    learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=msl, max_iter=400, l2_regularization=1.0,
                    early_stopping=False, random_state=RNG)
    for leaf in (3, 10):
        c[f"random_forest(min_leaf={leaf})"] = make_pipeline(imp(), RandomForestClassifier(
            n_estimators=500, min_samples_leaf=leaf, max_features="sqrt", random_state=RNG, n_jobs=-1))
    for leaf in (3, 10):
        c[f"extra_trees(min_leaf={leaf})"] = make_pipeline(imp(), ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=leaf, max_features="sqrt", random_state=RNG, n_jobs=-1))
    return c


FEATURE_SETS = ["base", "base+creator"]                 # round 1 (all horizons)
ROUND2 = {60: ["base+creator", "base+creator+jump+cnorm"],  # plan amendment 3
          30: ["base+creator+jump+cnorm"], 14: ["base+creator+jump+cnorm"]}


def features(d, H, fset="base"):
    """Feature matrix + eligibility for a feature set name made of: base, creator, jump, cnorm (joined by +)."""
    X, ok = build(d, H)
    parts = [X]
    if "creator" in fset:
        parts.append(creator_history(d, H))
    if "jump" in fset:
        parts.append(jump_features(d, H))
    if "cnorm" in fset:
        parts.append(creator_norms(d, H))
    return pd.concat(parts, axis=1), ok


def dataset(d, H, platform, fset="base"):
    X, ok = features(d, H, fset)
    lab = d[f"lab{H}"]
    m = ok & (d.platform == platform) & lab.isin(["P", "N"])
    cols = ["psrk", "platform", "pub", "creator", "client", "o14", "o30", "o60", f"ov{H}", "spend_day", "tag", f"lab{H}"]
    meta = d.loc[m, [c for c in dict.fromkeys(cols) if c in d]].copy()
    meta["y"] = (lab[m] == "P").astype(int)
    o = meta[f"ov{H}"] if f"ov{H}" in meta else meta[f"o{H}"]     # corrected ratio when plan amendment 4 is on
    in_window = meta.spend_day.between(-3, H - 2)
    # material = what changes organic totals (pre-registered C1 definition)
    meta["material"] = np.where(platform == "Tiktok", meta.y == 1,
                                (meta.y == 1) & ((o <= 0.60) | in_window))
    X = X[m]
    return X.loc[:, X.notna().any()], meta          # e.g. Instagram has no public share counts


def oof(model, X, y, groups, n=5):
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=RNG)
    p = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y, groups):
        p[va] = clone(model).fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
    return p


def fit_calibrated(base, X, y, groups):
    """Isotonic calibration with creator-grouped 5-fold CV inside the given training rows."""
    splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG).split(X, y, groups))
    return CalibratedClassifierCV(clone(base), method="isotonic", cv=splits).fit(X, y)


def oof_calibrated(base, X, y, groups, n=5):
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=RNG + 1)
    p = np.full(len(y), np.nan)
    for tr, va in cv.split(X, y, groups):
        m = fit_calibrated(base, X.iloc[tr], y.iloc[tr], groups.iloc[tr])
        p[va] = m.predict_proba(X.iloc[va])[:, 1]
    return p


def at(y, p, thr, material=None):
    y = np.asarray(y); f = np.asarray(p) >= thr
    tp, fp, fn = int((f & (y == 1)).sum()), int((f & (y == 0)).sum()), int((~f & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out = {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
           "tp": tp, "fp": fp, "fn": fn, "tn": int((~f & (y == 0)).sum())}
    if material is not None:
        mat = np.asarray(material, bool)
        out["material_recall"] = float(f[mat].mean()) if mat.any() else float("nan")
        out["n_material"] = int(mat.sum())
    return out


def pick_threshold(y, p):
    """Pre-registered rule: max F1 among thresholds with OOF precision >= 0.90, else max F1."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    rows = [(t, at(y, p, t)) for t in grid]
    ok = [(t, m) for t, m in rows if m["precision"] >= 0.90]
    pool = ok if ok else rows
    return float(max(pool, key=lambda r: r[1]["f1"])[0])


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(bins) if (b == k).any()))


def develop(d, plan=None):
    """Train-only CV for every (horizon, platform, feature set, candidate). Rows are merged into
    results/model_v2_selection.csv (same key = replaced), so later rounds add to earlier ones."""
    plan = plan or {H: FEATURE_SETS for H in HORIZONS}
    # each finished fit is appended to a .jsonl next to the table, so a stopped run resumes where it left off
    part = f"{RUN['results']}_selection.partial.jsonl"
    done = {}
    if os.path.exists(part):
        for line in open(part):
            r = json.loads(line)
            done[(r["H"], r["platform"], r["features"], r["model"])] = r
    rows = []
    for H, fsets in plan.items():
        for pf in PLATFORMS:
          for fset in fsets:
            if all((H, pf, fset, name) in done for name in candidates()):
                rows += [done[(H, pf, fset, name)] for name in candidates()]
                continue
            X, meta = dataset(d, H, pf, fset)
            tr = (meta.pub < TRAIN_END).values
            Xt, yt, gt = X[tr], meta.y[tr], meta.creator[tr]
            for name, model in candidates().items():
                if (H, pf, fset, name) in done:
                    rows.append(done[(H, pf, fset, name)])
                    continue
                p = oof(model, Xt, yt, gt)
                thr = pick_threshold(yt, p)
                m = at(yt, p, thr, meta.material[tr])
                rows.append({"H": H, "platform": pf, "features": fset, "model": name, "n": len(yt), "pos": int(yt.sum()),
                             "cv_roc_auc": roc_auc_score(yt, p), "cv_pr_auc": average_precision_score(yt, p), "thr": thr,
                             **{k: m[k] for k in ("precision", "recall", "f1", "material_recall")}, "ece": ece(yt, p)})
                print(rows[-1], flush=True)
                with open(part, "a") as fh:
                    fh.write(json.dumps(rows[-1], default=float) + "\n")
    t = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    path = f"{RUN['results']}_selection.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        key = ["H", "platform", "features", "model"]
        old = old[~old.set_index(key).index.isin(t.set_index(key).index)]
        t = pd.concat([old, t], ignore_index=True)
    t.round(4).to_csv(path, index=False)
    best = t.loc[t.groupby(["H", "platform"]).cv_pr_auc.idxmax()]   # model AND feature set chosen on train CV
    print("\nBEST by CV PR-AUC:\n", best.round(3).to_string(index=False))
    return t


def cluster_ci(y, p, thr, material, groups, n=1000):
    """95% intervals resampling whole creators."""
    rng = np.random.default_rng(RNG)
    y, p, material, groups = map(np.asarray, (y, p, material, groups))
    uniq = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in uniq}
    keys = ["precision", "recall", "f1", "material_recall", "roc_auc", "pr_auc"]
    draws = {k: [] for k in keys}
    for _ in range(n):
        i = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[i])) < 2:
            continue
        m = at(y[i], p[i], thr, material[i])
        m["roc_auc"], m["pr_auc"] = roc_auc_score(y[i], p[i]), average_precision_score(y[i], p[i])
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def evaluate(y, p, thr, meta):
    m = at(y, p, thr, meta.material)
    m.update({"n": int(len(y)), "n_pos": int(np.sum(y)), "roc_auc": float(roc_auc_score(y, p)),
              "pr_auc": float(average_precision_score(y, p)), "ece": ece(y, p),
              "ci95": cluster_ci(y, p, thr, meta.material, meta.creator)})
    return m


def final(d):
    sel = pd.read_csv(f"{RUN['results']}_selection.csv")
    report = {"train_end": str(TRAIN_END.date()), "test": ["2026-07-01", str(TEST_END.date())],
              "fresh": [str(FRESH[0].date()), str(FRESH[1].date())], "models": {}}
    os.makedirs("models", exist_ok=True)
    preds = []
    for H in HORIZONS:
        for pf in PLATFORMS:
            s = sel[(sel.H == H) & (sel.platform == pf)].sort_values("cv_pr_auc", ascending=False).iloc[0]
            fset = s.get("features", "base")
            X, meta = dataset(d, H, pf, fset)
            tr = (meta.pub < TRAIN_END).values
            te = ((meta.pub >= TRAIN_END) & (meta.pub <= TEST_END)).values
            fr = ((meta.pub >= FRESH[0]) & (meta.pub <= FRESH[1])).values
            base = candidates()[s.model]
            Xtr, ytr, gtr = X[tr], meta.y[tr], meta.creator[tr]
            p_oof = oof(base, Xtr, ytr, gtr)
            calibrated = ece(ytr, p_oof) > 0.05                   # pre-registered rule (plan amendment 2)
            if calibrated:
                p_oof = oof_calibrated(base, Xtr, ytr, gtr)
                model = fit_calibrated(base, Xtr, ytr, gtr)
            else:
                model = clone(base).fit(Xtr, ytr)
            thr = pick_threshold(ytr, p_oof)
            oo = meta[tr][["psrk", "platform", "pub", "y", "material", "tag", "spend_day", "creator"]].copy()
            oo["p"], oo["H"], oo["split"] = p_oof, H, "train_oof"
            preds.append(oo)
            r = {"model": s.model, "feature_set": fset, "calibrated": bool(calibrated), "threshold": thr, "features": list(X.columns),
                 "train": {"n": int(tr.sum()), "n_pos": int(meta.y[tr].sum()), "cv_oof": at(meta.y[tr], p_oof, thr, meta.material[tr]),
                           "cv_roc_auc": float(roc_auc_score(meta.y[tr], p_oof)), "cv_pr_auc": float(average_precision_score(meta.y[tr], p_oof)),
                           "cv_ece": ece(meta.y[tr], p_oof), "in_sample_roc_auc": float(roc_auc_score(meta.y[tr], model.predict_proba(X[tr])[:, 1]))}}
            for name, msk in [("test", te), ("fresh", fr)]:
                if msk.sum() and meta.y[msk].nunique() == 2:
                    p = model.predict_proba(X[msk])[:, 1]
                    r[name] = evaluate(meta.y[msk].values, p, thr, meta[msk])
                    out = meta[msk][["psrk", "platform", "pub", "y", "material", "tag", "spend_day", "creator"]].copy()
                    out["p"], out["H"], out["split"] = p, H, name
                    preds.append(out)
            joblib.dump({"model": model, "features": list(X.columns), "feature_set": fset, "threshold": thr, "H": H, "platform": pf,
                         "train_end": str(TRAIN_END.date()), "version": RUN["tag"]}, f"{RUN['models']}_{pf.lower()}_h{H}.joblib")
            report["models"][f"{pf}_h{H}"] = r
            t = r.get("test", {})
            print(f"{pf} H{H} {s.model} thr {thr}: TEST P {t.get('precision', float('nan')):.3f} R {t.get('recall', float('nan')):.3f} "
                  f"F1 {t.get('f1', float('nan')):.3f} matR {t.get('material_recall', float('nan')):.3f} AUC {t.get('roc_auc', float('nan')):.3f} ECE {t.get('ece', float('nan')):.3f}", flush=True)
            if "fresh" in r:
                f_ = r["fresh"]
                print(f"   FRESH n {f_['n']} pos {f_['n_pos']}: P {f_['precision']:.3f} R {f_['recall']:.3f} F1 {f_['f1']:.3f} matR {f_['material_recall']:.3f}", flush=True)
    pd.concat(preds).to_csv(f"{RUN['results']}_test_predictions.csv", index=False)   # row-level, git-ignored
    report["labels"] = "corrected (plan amendment 4)" if RUN["optin_fix"] else "as pulled (sql/11)"
    json.dump(report, open(f"{RUN['results']}_report.json", "w"), indent=1, default=float)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--develop", action="store_true")
    ap.add_argument("--develop2", action="store_true", help="plan amendment 3: day-60 model and the jump / creator-norm features")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--v21", action="store_true", help="plan amendment 4: corrected Instagram labels, Instagram models only, v2.1 file names")
    ap.add_argument("--v23", action="store_true", help="plan amendment 6: re-run train-only model search with repaired inputs")
    ap.add_argument("--v22", action="store_true", help="plan amendment 5: followers at horizon; same configurations as v2 (TikTok) "
                                                       "and v2.1 (Instagram), corrected labels, v2.2 file names")
    a = ap.parse_args()
    if a.v21:
        RUN.update(tag="v2.1", results="results/model_v2_1", models="models/boost_detector_v2_1", optin_fix=True)
        PLATFORMS[:] = ["Instagram"]
    if a.v22:
        RUN.update(tag="v2.2", results="results/model_v2_2", models="models/boost_detector_v2_2", optin_fix=True,
                   followers="data/v2_followers.psv")
        sel = pd.concat([pd.read_csv("results/model_v2_selection.csv").query("platform == 'Tiktok'"),
                         pd.read_csv("results/model_v2_1_selection.csv").query("platform == 'Instagram'")])
        sel.to_csv("results/model_v2_2_selection.csv", index=False)     # the configurations v2.2 refits (no new search)
    if a.v23:
        RUN.update(tag="v2.3", results="results/model_v2_3", models="models/boost_detector_v2_3", optin_fix=True,
                   followers="data/v2_followers.psv")
    d = load(optin_fix=RUN["optin_fix"], horizon_followers=RUN["followers"])
    if a.develop:
        develop(d)
    if a.develop2:
        develop(d, ROUND2)
    if a.v23 and not a.final:
        develop(d, {H: ["base+creator", "base+creator+jump+cnorm"] for H in HORIZONS})
    if a.v21 and not (a.develop or a.develop2 or a.final):
        develop(d, {60: ["base+creator", "base+creator+jump+cnorm"], 30: ["base+creator", "base+creator+jump+cnorm"],
                    14: ["base+creator", "base+creator+jump+cnorm"]})
    if a.final:
        final(d)
