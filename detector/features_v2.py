"""Model v2 features: views / likes at many ages (sql/11), cut at a horizon H (14 or 30 days).

Only PUBLIC organic values up to day H are used. LABEL_ONLY columns (opt-in ratio, spend day, tag, manual paid date,
labels) are never features; build() asserts this.
"""
import os

import numpy as np
import pandas as pd

RAW_COLS = ["psrk", "platform", "pub", "creator", "client", "n30", "n60", "a_min", "a_max", "a_max14",
            "v1", "v2", "v3", "v5", "v7", "v10", "v14", "v21", "v30", "v45", "v60",
            "l1", "l3", "l7", "l14", "l30", "l60", "c7", "c30", "s30", "followers",
            "r7", "r14", "r30", "r60", "o14", "o30", "spend_day", "tag", "paid_date", "lab30", "lab14"]
LABEL_ONLY = ["o14", "o30", "o60", "spend_day", "tag", "paid_date", "lab30", "lab14", "lab60", "ov14", "ov30", "ov60",
              "fz14", "pg14", "fz30", "pg30", "fz60", "pg60", "fza", "pga", "stale14", "stale30", "stale60", "n30_optin"]
# sql/12: what happens during the biggest view jump (likes per new view), extra ages, day-60 opt-in (label only)
EXTRA_COLS = ["psrk", "platform", "o60", "v25", "v28", "l21", "l45",
              "ja14", "jl14", "ml14", "ja30", "jl30", "ml30", "ja60", "jl60", "ml60"]
AGES = {60: [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60], 30: [1, 2, 3, 5, 7, 10, 14, 21, 30], 14: [1, 2, 3, 5, 7, 10, 14]}
LIKE_AGES = {60: [1, 3, 7, 14, 30, 60], 30: [1, 3, 7, 14, 30], 14: [1, 3, 7, 14]}
RATES = {60: ["r7", "r14", "r30", "r60"], 30: ["r7", "r14", "r30"], 14: ["r7", "r14"]}


STALE_DAYS, STALE_GROWTH = 7, 0.05      # plan amendment 4 (chosen on train)
STALE_COLS = ["psrk", "fz14", "pg14", "fz30", "pg30", "fz60", "pg60", "fza", "pga", "n30_optin"]
HF_COLS = ["psrk", "platform", "f14", "f30", "f60", "a_max30"]     # sql/14 (plan amendment 5): followers known at day H


def load(path="data/v2_raw.psv", extra="data/v2_extra.psv", optin_fix=False, stale="data/optin_staleness.psv",
         horizon_followers=None):
    """optin_fix=True applies plan amendment 4: Instagram opt-in labels use the last read where opt-in still updated.
    horizon_followers = sql/14 file (plan amendment 5): followers and day-30 eligibility use only reads up to day H."""
    d = pd.read_csv(path, sep="|", header=None, names=RAW_COLS, dtype=str, keep_default_na=False)
    d["platform"] = d.platform.map({"I": "Instagram", "T": "Tiktok"})
    for c in RAW_COLS:
        if c not in ("psrk", "platform", "pub", "lab30", "lab14"):
            d[c] = pd.to_numeric(d[c].replace("", np.nan), errors="coerce")
    d["pub"] = pd.to_datetime(d.pub)
    assert not d.duplicated(["psrk", "platform"]).any()
    if extra and os.path.exists(extra):
        e = pd.read_csv(extra, sep="|", header=None, names=EXTRA_COLS, dtype=str, keep_default_na=False)
        e["platform"] = e.platform.map({"I": "Instagram", "T": "Tiktok"})
        for c in EXTRA_COLS[2:]:
            e[c] = pd.to_numeric(e[c].replace("", np.nan), errors="coerce")
        for c in ("jl14", "ml14", "jl30", "ml30", "jl60", "ml60"):
            e[c] = e[c] / 1e5                                  # stored as likes per new view x 100000
        n = len(d)
        d = d.merge(e, on=["psrk", "platform"], how="left", validate="one_to_one")
        assert len(d) == n
        d["lab60"] = label60(d)
    if horizon_followers:
        h = pd.read_csv(horizon_followers, sep="|", header=None, names=HF_COLS, dtype=str, keep_default_na=False)
        h["platform"] = h.platform.map({"I": "Instagram", "T": "Tiktok"})
        for c in HF_COLS[2:]:
            h[c] = pd.to_numeric(h[c].replace("", np.nan), errors="coerce")
        n = len(d)
        d = d.merge(h, on=["psrk", "platform"], how="left", validate="one_to_one")
        assert len(d) == n
    if optin_fix:
        d = fix_optin_labels(d, stale)
    return d


def followers_at(d, H):
    """Followers known at day H (sql/14) when loaded, else the sql/11 median over days 0-60 (v2 and v2.1 models)."""
    return d[f"f{H}"] if f"f{H}" in d else d.followers


def ig_labels(d, H, o):
    """Instagram label at horizon H from an opt-in ratio series o (same rules as sql/11 and label60)."""
    ev = d.spend_day.notna()
    p = (ev & d.spend_day.between(-3, H - 2)) | (o < 0.80)
    n = ~ev & d.paid_date.isna() & (d[f"o{H}"] >= 0.90)
    return pd.Series(np.where(p, "P", np.where(n, "N", "")), index=d.index)


def fix_optin_labels(d, path="data/optin_staleness.psv"):
    """Plan amendment 4. If the opt-in count has not changed for >= 7 days at the end of the window while public grew
    >= 5%, use the ratio at the freeze: o_valid = o_H x (1 + public growth since the freeze). A gap that appears only
    after opt-in froze gives no label. TikTok labels do not use opt-in and do not change."""
    st = pd.read_csv(path, sep="|", header=None, names=STALE_COLS, dtype={"psrk": str}).assign(platform="Instagram")
    d = d.merge(st, on=["psrk", "platform"], how="left", validate="one_to_one")
    ig = d.platform == "Instagram"
    for H in (14, 30, 60):
        if f"lab{H}" not in d:
            continue
        stale = (d[f"fz{H}"] >= STALE_DAYS) & (d[f"pg{H}"] >= STALE_GROWTH)
        d[f"stale{H}"] = stale & ig
        d[f"ov{H}"] = np.where(stale, d[f"o{H}"] * (1 + d[f"pg{H}"]), d[f"o{H}"])
        new = ig_labels(d, H, d[f"ov{H}"])
        d.loc[ig & stale, f"lab{H}"] = new[ig & stale]          # other rows keep the sql/11 label exactly
    return d


def label60(d):
    """Day-60 label, same rules as sql/11 lab30 with the window widened: P = paid evidence with first spend in
    [publish - 3, publish + 58], or Instagram opt-in / public < 0.80 on the latest read up to day 60; N = no paid evidence,
    no manual paid date, and (TikTok) the same client-coverage rule as lab30 / (Instagram) opt-in / public >= 0.90."""
    ev = d.spend_day.notna()
    p = (ev & d.spend_day.between(-3, 58)) | ((d.platform == "Instagram") & (d.o60 < 0.80))
    tik_n = (d.platform == "Tiktok") & (d.lab30 == "N")                  # lab30 N on TikTok = no evidence, no paid date, covered client
    ig_n = (d.platform == "Instagram") & ~ev & d.paid_date.isna() & (d.o60 >= 0.90)
    return np.where(p, "P", np.where(tik_n | ig_n, "N", ""))


def _safe_div(a, b):
    b = b.where(b > 0)
    return a / b


def build(d, H):
    """Return (X, eligible mask). X has only horizon-H public features."""
    ages, lages = AGES[H], LIKE_AGES[H]
    vH = d[f"v{H}"].where(d[f"v{H}"] > 0)
    fol = followers_at(d, H)
    f = fol.where(fol > 0)
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
    c = d.c30 if H >= 30 else d.c7                         # comments are read at days 7 and 30 only
    X["comments_per_like"] = _safe_div(c, d[f"l{lb}"]).clip(upper=5)
    if H >= 30:
        X["shares_per_view"] = _safe_div(d.s30, d.v30.where(d.v30 > 0))
    X["er"] = np.log10(_safe_div(d[f"l{lb}"].fillna(0) + c.fillna(0), vH).clip(lower=1e-5))
    assert not set(X.columns) & set(LABEL_ONLY)
    X = X.replace([np.inf, -np.inf], np.nan)
    if H == 60:
        eligible = (d.a_max >= 55) & (d.n60 >= 3) & vH.notna() & f.notna()
    elif H == 30:     # with sql/14: the last read up to day 30 (amendment 5), else up to day 60
        eligible = (d.get("a_max30", d.a_max) >= 28) & (d.n30 >= 2) & vH.notna() & f.notna()
    else:
        eligible = (d.a_max14 >= 12) & (d.a_min <= 10) & vH.notna() & f.notna()
    return X, eligible


def creator_history(d, H):
    """Creator-relative reach: this post's day-H views vs the median of the SAME creator's EARLIER posts (public views
    only, strictly earlier publish date, the post itself excluded). Known at scoring time; no label is used."""
    v = np.log10(d[f"v{H}"].where(d[f"v{H}"] > 0))
    t = pd.DataFrame({"creator": d.creator, "platform": d.platform, "pub": d.pub, "lv": v}).sort_values("pub")
    med, cnt = pd.Series(np.nan, index=t.index), pd.Series(0, index=t.index)
    for _, g in t.groupby(["platform", "creator"], sort=False):
        vals, dates = g.lv.values, g.pub.values
        for i, (ix, day) in enumerate(zip(g.index, dates)):
            prev = vals[:i][(dates[:i] < day) & ~np.isnan(vals[:i])]
            if len(prev):
                med[ix], cnt[ix] = np.median(prev), len(prev)
    out = pd.DataFrame(index=d.index)
    out["creator_rel_reach"] = (v - med.reindex(d.index))
    out["creator_prior_posts"] = np.log10(cnt.reindex(d.index) + 1)
    return out


def jump_features(d, H):
    """Likes per NEW view during the biggest daily view gain (and the lowest such value on any read interval that adds
    >= 5% of day-H views), relative to the post's own likes per view. Paid views bring few likes, organic spikes do not.
    Also the age of the biggest gain and (H >= 30) views at days 25 / 28. Public data only (sql/12)."""
    out = pd.DataFrame(index=d.index)
    if f"jl{H}" not in d:
        return out
    lb = {14: "l14", 30: "l30", 60: "l60"}[H]
    lpv = _safe_div(d[lb], d[f"v{H}"])
    out["jump_age"] = d[f"ja{H}"] / H
    out["jump_lpv_ratio"] = np.log10((d[f"jl{H}"] + 1e-4) / (lpv + 1e-4))
    out["min_new_lpv_ratio"] = np.log10((d[f"ml{H}"] + 1e-4) / (lpv + 1e-4))
    out["jump_lpv"] = np.log10(d[f"jl{H}"] + 1e-4)
    if H >= 30:
        vH = d[f"v{H}"].where(d[f"v{H}"] > 0)
        out["share_v25"] = _safe_div(d.v25, vH).clip(0, 1)
        out["share_v28"] = _safe_div(d.v28, vH).clip(0, 1)
        out["late_new_lpv_ratio"] = np.log10((_safe_div((d[lb] - d.l21).clip(lower=0), (d[f"v{H}"] - d.v21).where(lambda x: x > 0)) + 1e-4)
                                             / (_safe_div(d.l21, d.v21) + 1e-4))
    assert not set(out.columns) & set(LABEL_ONLY)
    return out.replace([np.inf, -np.inf], np.nan)


def _rel_to_prior(d, x):
    """x minus the median of x over the SAME creator's strictly earlier posts (same platform); NaN when none."""
    t = pd.DataFrame({"creator": d.creator, "platform": d.platform, "pub": d.pub, "x": x}).sort_values("pub")
    med = pd.Series(np.nan, index=t.index)
    for _, g in t.groupby(["platform", "creator"], sort=False):
        vals, dates = g.x.values, g.pub.values
        for i, (ix, day) in enumerate(zip(g.index, dates)):
            prev = vals[:i][(dates[:i] < day) & ~np.isnan(vals[:i])]
            if len(prev):
                med[ix] = np.median(prev)
    return x - med.reindex(d.index)


def creator_norms(d, H):
    """How this post differs from the same creator's earlier posts: likes per view, engagement, early-curve share and
    reach vs followers. A boost lowers likes per view and flattens the early share against the creator's own norm.
    Public data only; strictly earlier posts, so it is known at scoring time."""
    lb = {14: "l14", 30: "l30", 60: "l60"}[H]
    vH = d[f"v{H}"].where(d[f"v{H}"] > 0)
    out = pd.DataFrame(index=d.index)
    out["creator_rel_lpv"] = _rel_to_prior(d, np.log10(_safe_div(d[lb], vH) + 1e-4))
    out["creator_rel_share_v3"] = _rel_to_prior(d, _safe_div(d.v3, vH).clip(0, 1))
    fol = followers_at(d, H)
    out["creator_rel_vtf"] = _rel_to_prior(d, np.log10(_safe_div(vH, fol.where(fol > 0))))
    return out.replace([np.inf, -np.inf], np.nan)
