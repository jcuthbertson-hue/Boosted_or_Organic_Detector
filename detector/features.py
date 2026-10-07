"""Load the detector dataset and build model features.

Input: data/detector_dataset_v2.psv (pipe-separated, no header), exported from
sql/02_detector_features_and_labels.sql.

Leakage rules (see README "Validity"):
  * FEATURES are built only from PUBLIC organic metrics in the first 30 days after publish
    (views, likes, comments, shares, followers). Every post uses the same 30-day window.
  * NO feature reads: PAID_ACTIVITY_DATE, any boosted / pre-boost flag, ad names, ad spend, ad dates,
    opt-in (private) views, or SocAPI plays. Those columns are LABEL_ONLY.
"""
import numpy as np
import pandas as pd

COLS = ["psrk", "platform", "post_type", "org", "campaign_key", "handle", "pub", "n_obs", "age_last",
        "v1", "v3", "v7", "v14", "v30", "l7", "l30", "c30", "s30", "followers",
        "max_rate_d8_30", "max_like_rate_d8_30",
        "optin_ratio_d30", "paid_date", "first_spend", "spend", "pos_source_ad", "label"]

# Columns that may be used to build labels or to slice results, but NEVER as model inputs.
LABEL_ONLY = ["optin_ratio_d30", "paid_date", "first_spend", "spend", "pos_source_ad", "label"]

NUM = ["n_obs", "age_last", "v1", "v3", "v7", "v14", "v30", "l7", "l30", "c30", "s30", "followers",
       "max_rate_d8_30", "max_like_rate_d8_30", "optin_ratio_d30", "spend"]

FEATURES = [
    "log_followers",      # creator size
    "log_views30",        # public views at day 30
    "log_vtf30",          # views / followers at day 30 ("view rate")
    "log_er30",           # (likes + comments) / views at day 30
    "front_load_d1",      # share of day-30 views already there on day 1
    "late_share_d7",      # share of day-30 views gained after day 7
    "late_share_d14",     # share of day-30 views gained after day 14
    "log_accel",          # biggest daily view gain on days 8-30 vs the average daily gain in days 0-7
    "log_accel_gap",      # view acceleration minus like acceleration (views jump, likes do not)
    "lpv_dilution",       # likes per view in days 0-7 vs likes per view on views gained after day 7
    "comments_per_like",
]


def load(path="data/detector_dataset_v2.psv"):
    df = pd.read_csv(path, sep="|", header=None, names=COLS, dtype=str, keep_default_na=False)
    for c in NUM:
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    for c in ["pub", "paid_date", "first_spend"]:
        df[c] = pd.to_datetime(df[c].replace("", np.nan), errors="coerce")
    return df


def build(df):
    """Add model features. Uses only public 30-day columns."""
    out = df.copy()
    f = out.followers.where(out.followers > 0)
    v = out.v30.where(out.v30 > 0)
    out["log_followers"] = np.log10(f)
    out["log_views30"] = np.log10(v)
    out["vtf30"] = v / f
    out["log_vtf30"] = np.log10(out.vtf30)
    out["er30"] = (out.l30.fillna(0) + out.c30.fillna(0)) / v
    out["log_er30"] = np.log10(out.er30.clip(lower=1e-5))
    out["front_load_d1"] = (out.v1 / v).clip(0, 1)
    out["late_share_d7"] = ((v - out.v7) / v).clip(0, 1)
    out["late_share_d14"] = ((v - out.v14) / v).clip(0, 1)
    early_rate = out.v7 / 7
    out["log_accel"] = np.log10((out.max_rate_d8_30.clip(lower=0) + 1) / (early_rate + 1))
    early_like_rate = out.l7 / 7
    log_like_accel = np.log10((out.max_like_rate_d8_30.clip(lower=0) + 1) / (early_like_rate + 1))
    out["log_accel_gap"] = out.log_accel - log_like_accel
    early_lpv = out.l7 / out.v7.where(out.v7 > 0)
    late_dv = (v - out.v7).where((v - out.v7) > 0)
    late_lpv = (out.l30 - out.l7).clip(lower=0) / late_dv
    out["lpv_dilution"] = np.log10((early_lpv + 1e-4) / (late_lpv + 1e-4))
    out["comments_per_like"] = (out.c30 / out.l30.where(out.l30 > 0)).clip(upper=5)
    out["y"] = (out.label == "POS").astype(int)
    assert not set(FEATURES) & set(LABEL_ONLY)
    return out
