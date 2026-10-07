"""Score posts with the trained detectors (the PREDICTED tier).

Input : a pipe-separated export with the same columns as data/detector_dataset_v2.psv
        (run sql/02_detector_features_and_labels.sql without the label filter; posts need >= 28 days of data).
Output: results/predicted_boost_scores.csv  (psrk, platform, p_boost, predicted_boosted)

Use it only for posts with NO deterministic evidence in sql/05_boost_flags.sql (boost_evidence = NO_PAID_EVIDENCE).
Posts with a confirmed ad link or a measured opt-in / SocAPI gap do not need a model.

Run from repo root:  python3 -m detector.score --input data/detector_dataset_v2.psv
"""
import argparse

import joblib
import pandas as pd

from detector.features import FEATURES, build, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/detector_dataset_v2.psv")
    ap.add_argument("--output", default="results/predicted_boost_scores.csv")
    a = ap.parse_args()
    d = build(load(a.input))
    d = d[(d.v30 > 0) & (d.followers > 0)]
    out = []
    for platform in ["Tiktok", "Instagram"]:
        bundle = joblib.load(f"models/boost_detector_{platform.lower()}.joblib")
        g = d[d.platform == platform].copy()
        g["p_boost"] = bundle["model"].predict_proba(g[FEATURES])[:, 1]
        g["predicted_boosted"] = g.p_boost >= bundle["threshold"]
        out.append(g[["psrk", "platform", "pub", "p_boost", "predicted_boosted"]])
    res = pd.concat(out)
    res.to_csv(a.output, index=False)
    print(res.groupby("platform").predicted_boosted.agg(["size", "sum", "mean"]).round(3))


if __name__ == "__main__":
    main()
