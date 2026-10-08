"""Model v2 features: views / likes at many ages (sql/11), cut at a horizon H (14 or 30 days).

Only PUBLIC organic values up to day H are used. LABEL_ONLY columns (opt-in ratio, spend day, tag, manual paid date,
labels) are never features; build() asserts this.
"""
import numpy as np
import pandas as pd

RAW_COLS = ["psrk", "platform", "pub", "creator", "client", "n30", "n60", "a_min", "a_max", "a_max14",
            "v1", "v2", "v3", "v5", "v7", "v10", "v14", "v21", "v30", "v45", "v60",
            "l1", "l3", "l7", "l14", "l30", "l60", "c7", "c30", "s30", "followers",
            "r7", "r14", "r30", "r60", "o14", "o30", "spend_day", "tag", "paid_date", "lab30", "lab14"]
LABEL_ONLY = ["o14", "o30", "spend_day", "tag", "paid_date", "lab30", "lab14"]
AGES = {30: [1, 2, 3, 5, 7, 10, 14, 21, 30], 14: [1, 2, 3, 5, 7, 10, 14]}
LIKE_AGES = {30: [1, 3, 7, 14, 30], 14: [1, 3, 7, 14]}
RATES = {30: ["r7", "r14", "r30"], 14: ["r7", "r14"]}


def load(path="data/v2_raw.psv"):
    d = pd.read_csv(path, sep="|", header=None, names=RAW_COLS, dtype=str, keep_default_na=False)
    d["platform"] = d.platform.map({"I": "Instagram", "T": "Tiktok"})
    for c in RAW_COLS:
        if c not in ("psrk", "platform", "pub", "lab30", "lab14"):
            d[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce")
    d["pub"] = pd.to_datetime(d.pub)
    assert not d.duplicated(["psrk", "platform"]).any()
    return d


def _safe_div(a, b):
    b = b.where(b > 0)
    return a / b


def build(d, H):
    """Return (X, eligible mask). X has only horizon-H public features."""
    ages, lages = AGES[H], LIKE_AGES[H]
    vH = d[f"v{H}"].where(d[f"v{H}"] > 0)
    f = d.followers.where(d.followers > 0)
    X = pd.DataFrame(index=d.index)
    X["log_followers"] = np.log10(f)
    X["log_views"] = np.log10(vH)
    X["log_vtf"] = np.log10(_safe_div(vH, f))
    for a in ages[:-1]:                                    # growth-curve shape: share of day-H views reached by day a
        X[f"share_v{a}"] = _safe_div(d[f"v{a}"], vH).clip(0, 1)
    for a, b in zip(ages[:-1], ages[1:]):                  # daily gain per interval, relative to the average day-H rate
        X[f"rate_{a}_{b}"] = _safe_div((d[f"v{b}"] - d[f"v{a}"]).clip(lower=0) / (b - a), vH / H)
    early = _safe_div(d.v7 - d.v1.fillna(0), pd.Series(6.0, index=d.index)) if H >= 7 else None
    for a, b in zip(ages[:-1], ages[1:]):
        if a >= 7:                                         # acceleration after day 7 vs days 1-7
            X[f"accel_{a}_{b}"] = np.log10(((d[f"v{b}"] - d[f"v{a}"]).clip(lower=0) / (b - a) + 1) / (early.clip(lower=0) + 1))
    for r in RATES[H]:                                     # biggest single-day jump in the window, vs day-H views
        X[f"jump_{r}"] = _safe_div(d[r].clip(lower=0), vH)
    X["jump_late_vs_early"] = np.log10((d[RATES[H][-1]].clip(lower=0) + 1) / (d.r7.clip(lower=0) + 1))
    for a in lages:                                        # likes per view at each age
        X[f"lpv{a}"] = _safe_div(d[f"l{a}"], d[f"v{a}"])
    la, lb = lages[1], lages[-1]                           # likes per extra view after day 3 vs before
    X["like_dilution"] = np.log10((_safe_div(d[f"l{la}"], d[f"v{la}"]) + 1e-4) /
                                  (_safe_div((d[f"l{lb}"] - d[f"l{la}"]).clip(lower=0), (d[f"v{lb}"] - d[f"v{la}"]).where(lambda s: s > 0)) + 1e-4))
    X["comments_per_like"] = _safe_div(d.c30 if H == 30 else d.c7, d[f"l{lb}"]).clip(upper=5)
    if H == 30:
        X["shares_per_view"] = _safe_div(d.s30, vH)
    X["er"] = np.log10(_safe_div(d[f"l{lb}"].fillna(0) + (d.c30 if H == 30 else d.c7).fillna(0), vH).clip(lower=1e-5))
    assert not set(X.columns) & set(LABEL_ONLY)
    X = X.replace([np.inf, -np.inf], np.nan)
    if H == 30:
        eligible = (d.a_max >= 28) & (d.n30 >= 2) & vH.notna() & f.notna()
    else:
        eligible = (d.a_max14 >= 12) & (d.a_min <= 10) & vH.notna() & f.notna()
    return X, eligible
