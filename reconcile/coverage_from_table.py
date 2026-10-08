"""Rebuild results/organic_coverage.json from the daily paid table (aggregates only).

The first version of this file came from a Snowflake run of sql/05 before the frozen opt-in fix (plan amendment 4),
so it counted posts whose opt-in count had stopped updating as measured organic. The daily table has the fix: those
posts use opt-in until it froze, then the organic curve. Scope: every post in the table with any paid evidence.

Input : outputs/paid_classification_<date>.parquet (pipeline/run_daily.py output; row level, git-ignored)
Output: results/organic_coverage.json
Run   : python3 -m reconcile.coverage_from_table outputs/paid_classification_2026-10-07.parquet
"""
import json
import sys

import pandas as pd

CATS = ["measured_optin", "optin_frozen", "est_high", "est_medium", "est_low", "not_separable"]


def category(method, conf):
    if method == "opt-in private views (organic only)":
        return "measured_optin"
    if method == "opt-in until it stopped updating, then organic curve":
        return "optin_frozen"
    if method == "pre-boost public read x organic curve" and conf in ("high", "medium", "low"):
        return f"est_{conf}"
    if str(method).startswith("not separable"):
        return "not_separable"
    raise ValueError(f"paid post with no organic category: {method!r} / {conf!r}")


def main(path):
    d = pd.read_parquet(path, columns=["POST_PLATFORM", "BOOST_EVIDENCE", "ORGANIC_VIEWS_METHOD", "ORGANIC_VIEWS_CONFIDENCE", "RUN_DATE"])
    run = str(d.RUN_DATE.max())
    paid = d[d.BOOST_EVIDENCE != "NO_PAID_EVIDENCE"].copy()
    paid["cat"] = [category(m, c) for m, c in zip(paid.ORGANIC_VIEWS_METHOD, paid.ORGANIC_VIEWS_CONFIDENCE)]
    by_ev = []
    for (pf, ev), g in paid.groupby(["POST_PLATFORM", "BOOST_EVIDENCE"]):
        n = g.cat.value_counts()
        by_ev.append({"platform": pf, "evidence": ev, "posts": int(len(g)), **{c: int(n.get(c, 0)) for c in CATS}})
    totals = {}
    for pf, g in paid.groupby("POST_PLATFORM"):
        n = g.cat.value_counts()
        totals[pf] = {"posts": int(len(g)), **{c: int(n.get(c, 0)) for c in CATS}}
    tiers = {"source": f"daily paid table, data read {run}"}
    for pf, g in d.groupby("POST_PLATFORM"):
        tiers[pf] = {k: int(v) for k, v in g.BOOST_EVIDENCE.value_counts().items()}
    out = {"source": f"pipeline/run_daily.py output ({path.rsplit('/', 1)[-1]}), data read {run}; sql/05 with the frozen opt-in fix",
           "scope": "all in-feed Instagram / TikTok campaign posts in the daily table with any paid evidence (all publish dates)",
           "rule": "measured_optin = Instagram opt-in views that kept updating; optin_frozen = opt-in until it stopped updating, then "
                   "the organic curve; est_* = pre-boost public read x organic curve, confidence by age of the pre-boost read "
                   "(high >= day 14, medium day 7-13, low < day 7); not_separable = no public read before the boost start",
           "by_evidence": by_ev, "totals": totals, "evidence_tiers_all_posts": tiers}
    json.dump(out, open("results/organic_coverage.json", "w"), indent=1)
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "outputs/paid_classification_2026-10-07.parquet")
