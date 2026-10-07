"""Load the reconciliation panel exported by sql/06_reconciliation_panel.sql."""
import numpy as np
import pandas as pd

COLS = ["psrk", "tier", "spend_total", "od", "dsf", "dsl", "vp", "vr", "api_total", "api_ig", "api_fb",
        "impr_ig", "impr_fb", "reach_ig", "reach_fb", "starts_ig", "starts_fb", "starts_ot", "starts_ig_p", "starts_fb_p",
        "v2s_ig", "v2s_fb", "v3s_ig", "v3s_fb", "thru_ig", "thru_fb", "p25_ig", "p25_fb", "p100_ig", "p100_fb"]
TIERS = {"a": "andrew_confirmed", "u": "unified_name", "e": "edw_taxonomy"}
# Paid metrics in the Meta ad tables / Andrew's unified table, with the names Paid uses
METRICS = {"impr": "Impressions", "reach": "Reach", "starts": "Video plays (starts)", "v2s": "2-second plays",
           "v3s": "3-second video views", "thru": "ThruPlays", "p25": "25% watched", "p100": "100% watched"}


def load(path="data/recon_panel_ig_full.psv"):
    d = pd.read_csv(path, sep="|", header=None, names=COLS, dtype={"psrk": str, "tier": str, "od": str})
    d["tier"] = d.tier.map(TIERS)
    d["od"] = pd.to_datetime(d.od)
    for c in COLS[2:]:
        if c not in ("od",):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["api_fb"] = d.api_fb.fillna(0)
    for m in METRICS:
        if f"{m}_fb" in d:
            d[f"{m}_all"] = d[f"{m}_ig"].fillna(0) + d[f"{m}_fb"].fillna(0) + (d["starts_ot"].fillna(0) if m == "starts" else 0)
    return d
