"""Scorecard inputs for the production model set: TikTok v2 + Instagram v2.1, both on corrected labels
(TikTok labels do not use opt-in, so they are the same). Run after `train_v2 --v21 --final` and `v2_relabel_check`.

Writes results/model_v2_1_combined_report.json (aggregates) and results/model_v2_1_combined_predictions.csv (git-ignored),
then `python3 -m detector.v2_targets --v21` scores C1-C6 on them.
"""
import json

import pandas as pd


def main():
    tik = json.load(open("results/model_v2_report_fixed.json"))
    ig = json.load(open("results/model_v2_1_report.json"))
    rep = {**tik, "labels": "corrected (plan amendment 4)", "models": {}}
    rep["models"].update({k: v for k, v in tik["models"].items() if k.startswith("Tiktok")})
    rep["models"].update({k: v for k, v in ig["models"].items() if k.startswith("Instagram")})
    rep["model_versions"] = {"Tiktok": "v2", "Instagram": "v2.1"}
    json.dump(rep, open("results/model_v2_1_combined_report.json", "w"), indent=1, default=float)
    a = pd.read_csv("results/model_v2_test_predictions_fixed.csv", dtype={"psrk": str})       # test + fresh, corrected labels
    o = pd.read_csv("results/model_v2_test_predictions.csv", dtype={"psrk": str})             # train out-of-fold (TikTok labels unchanged)
    b = pd.read_csv("results/model_v2_1_test_predictions.csv", dtype={"psrk": str})
    tik = pd.concat([o[(o.platform == "Tiktok") & (o.split == "train_oof")], a[a.platform == "Tiktok"]])
    pd.concat([tik, b[b.platform == "Instagram"]]).to_csv("results/model_v2_1_combined_predictions.csv", index=False)
    print({k: (v.get("model"), v.get("threshold")) for k, v in rep["models"].items()})


if __name__ == "__main__":
    main()
