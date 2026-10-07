"""Score LLM answers against the locked-test labels, next to the ML model on the SAME posts.

Input files: results/llm/<model>.psv with lines  psrk|platform|label|<free-text answer>
(the answer must start with, or contain, a number from 0 to 1).
Works for any model, including a local Decider 4B run (see llm/README.md).

Run from repo root:  python3 -m llm.score_llm
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

NUM = re.compile(r"(?<![\d.])(0(?:\.\d+)?|1(?:\.0+)?|\.\d+)(?![\d.])")


def parse(ans):
    m = NUM.search(str(ans))
    return float(m.group(1)) if m else np.nan


def score(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    f = (p >= thr).astype(int)
    return {"n": len(y), "n_pos": int(y.sum()), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "precision": precision_score(y, f, zero_division=0), "recall": recall_score(y, f, zero_division=0),
            "f1": f1_score(y, f, zero_division=0)}


def main():
    ml = pd.read_csv("results/test_predictions.csv", dtype={"psrk": str})
    rep = json.load(open("results/detector_report.json"))
    rows = []
    for path in sorted(glob.glob("results/llm/*.psv")):
        name = os.path.basename(path)[:-4]
        llm = pd.read_csv(path, sep="|", header=None, names=["psrk", "platform", "label", "answer"], dtype=str,
                          keep_default_na=False, engine="python", quoting=3)
        llm["p"] = llm.answer.map(parse)
        llm["y"] = (llm.label == "POS").astype(int)
        j = llm.merge(ml[["psrk", "platform", "p_boost"]], on=["psrk", "platform"], how="left")
        for plat, g in j.groupby("platform"):
            ok = g.p.notna()
            rows.append({"detector": f"LLM {name}", "platform": plat, "parsed": f"{ok.sum()}/{len(g)}",
                         **score(g.y[ok], g.p[ok])})
            thr = rep[plat]["threshold"]
            rows.append({"detector": "ML model (same posts)", "platform": plat, "parsed": f"{g.p_boost.notna().sum()}/{len(g)}",
                         **score(g.y, g.p_boost, thr)})
    res = pd.DataFrame(rows).drop_duplicates(subset=["detector", "platform"])
    res.to_csv("results/llm_vs_ml.csv", index=False)
    print(res.round(3).to_string(index=False))


def paired_auc_test(n_boot=2000, seed=7):
    """Paired bootstrap: ROC-AUC(ML) - ROC-AUC(LLM) on the same posts. Returns 95% interval per model/platform."""
    ml = pd.read_csv("results/test_predictions.csv", dtype={"psrk": str})
    rng = np.random.default_rng(seed)
    out = []
    for path in sorted(glob.glob("results/llm/*.psv")):
        name = os.path.basename(path)[:-4]
        llm = pd.read_csv(path, sep="|", header=None, names=["psrk", "platform", "label", "answer"], dtype=str,
                          keep_default_na=False, engine="python", quoting=3)
        llm["p"] = llm.answer.map(parse); llm["y"] = (llm.label == "POS").astype(int)
        j = llm.merge(ml[["psrk", "platform", "p_boost"]], on=["psrk", "platform"]).dropna(subset=["p", "p_boost"])
        for plat, g in j.groupby("platform"):
            y, a, b = g.y.values, g.p_boost.values, g.p.values
            diffs = []
            for _ in range(n_boot):
                i = rng.integers(0, len(y), len(y))
                if len(np.unique(y[i])) < 2:
                    continue
                diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            out.append({"llm": name, "platform": plat, "auc_ml_minus_llm": roc_auc_score(y, a) - roc_auc_score(y, b),
                        "ci95_low": lo, "ci95_high": hi, "significant": not (lo <= 0 <= hi)})
    res = pd.DataFrame(out)
    res.to_csv("results/llm_vs_ml_paired_auc.csv", index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
    paired_auc_test()
