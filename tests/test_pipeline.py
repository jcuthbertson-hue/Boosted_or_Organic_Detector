"""Offline checks for the daily pipeline logic (no Snowflake). Run: python3 -m tests.test_pipeline"""
import datetime as dt

import numpy as np
import pandas as pd

from pipeline.run_daily import OUT_COLS, classify, to_table
from reconcile.estimate import estimate, load_curves


def flags():
    base = {"post_type": "REEL", "post_url": "", "pub": pd.Timestamp("2026-08-01"), "first_spend": pd.NaT,
            "paid_date": pd.NaT, "boost_start": pd.NaT, "spend": np.nan, "paid_impressions_ig": np.nan,
            "paid_impressions_fb": np.nan, "paid_plays_ig": np.nan, "paid_plays_fb": np.nan,
            "views_private_latest": np.nan, "views_before_first_spend": np.nan, "preboost_age_days": np.nan,
            "age_latest": 60, "views_public_latest": 100000.0}
    rows = [
        dict(psrk="a", platform="Instagram", boost_evidence="CONFIRMED_PAID_TAG", views_private_latest=8000.0),
        dict(psrk="b", platform="Instagram", boost_evidence="CONFIRMED_AD_LINK", views_before_first_spend=5000.0, preboost_age_days=14),
        dict(psrk="c", platform="Tiktok", boost_evidence="CONFIRMED_AD_LINK", views_before_first_spend=5000.0, preboost_age_days=2),
        dict(psrk="d", platform="Instagram", boost_evidence="MEASURED_SOCAPI_GAP"),
        dict(psrk="e", platform="Tiktok", boost_evidence="NO_PAID_EVIDENCE"),
        dict(psrk="f", platform="Tiktok", boost_evidence="NO_PAID_EVIDENCE"),
        dict(psrk="g", platform="Instagram", boost_evidence="NO_PAID_EVIDENCE"),
    ]
    return pd.DataFrame([{**base, **r} for r in rows])


def test_classify():
    scores = pd.DataFrame({"psrk": ["e", "f", "b"], "platform": ["Tiktok", "Tiktok", "Instagram"],
                           "model_score": [0.9, 0.1, 0.99], "model_threshold": [0.53, 0.53, 0.44]})
    d = classify(flags(), scores, dt.date(2026, 10, 7)).set_index("psrk")
    assert d.loc["a", "paid_status"] == "PAID_CONFIRMED" and d.loc["a", "organic_views_est"] == 8000
    assert d.loc["a", "organic_views_confidence"] == "measured"
    assert d.loc["b", "paid_status"] == "PAID_CONFIRMED" and d.loc["b", "organic_views_confidence"] == "high"
    assert np.isnan(d.loc["b", "model_score"]), "model score must be blank when evidence exists"
    assert 5000 <= d.loc["b", "organic_views_est"] <= 100000
    assert d.loc["c", "organic_views_confidence"] == "low"
    assert d.loc["d", "paid_status"] == "PAID_MEASURED" and np.isnan(d.loc["d", "organic_views_est"])
    assert d.loc["d", "organic_views_confidence"] == "none"
    assert d.loc["e", "paid_status"] == "PAID_PREDICTED" and np.isnan(d.loc["e", "organic_views_est"])
    assert d.loc["f", "paid_status"] == "ORGANIC_PREDICTED" and d.loc["f", "organic_views_est"] == 100000
    assert d.loc["g", "paid_status"] == "NOT_SCORED"
    assert (d.paid_views_on_platform_est.dropna() >= 0).all()
    t = to_table(d.reset_index())
    assert list(t.columns) == list(OUT_COLS) and len(t) == 7


def test_estimate_never_above_public_and_never_below_preboost():
    c = load_curves()
    d = pd.DataFrame({"platform": ["Instagram"] * 3 + ["Tiktok"] * 3, "boosted": True,
                      "views_public_latest": [1e4, 1e6, 3e3] * 2, "views_private_latest": np.nan,
                      "age_latest": [30, 90, 200] * 2, "views_before_first_spend": [2e3, 5e3, 2.9e3] * 2,
                      "preboost_age_days": [1, 7, 20] * 2})
    e = estimate(d, c)
    assert (e.organic_views_est <= d.views_public_latest).all()
    assert (e.organic_views_est >= d.views_before_first_spend.round()).all()
    assert list(e.organic_views_confidence) == ["low", "medium", "high"] * 2


if __name__ == "__main__":
    test_classify()
    test_estimate_never_above_public_and_never_below_preboost()
    print("pipeline tests passed")
