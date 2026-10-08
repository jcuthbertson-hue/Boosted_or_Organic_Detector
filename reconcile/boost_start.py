"""Boost start from the daily public view series, for boosted posts with no known start date (model flags, and evidence
posts with no spend or paid date). It finds the pre-boost read that the organic estimate needs (reconcile/estimate.py).

Method. For each interval between two reads, x = actual growth / organic-curve growth (median curve of unboosted posts,
reconcile/organic_curve_*.csv). Organic posts stay close to x = 1 after day 3 (99th percentile 1.03-1.27 by age band).
The boost is the first interval where
  (1) x >= 3.0 (read on day 0-1), 1.6 (day 2-3) or 1.15 (day 4+)   far beyond normal organic growth for that age,
  (2) excess views >= 0.2% of the latest views                         not rounding noise on a small post,
  (3) latest implied size / implied size at the read >= 1.5             views stay above the organic path (not a glitch).
The read before that interval is the pre-boost read; step back over a lead-in interval with x >= 1.15 (slow ramp).
If the first jump is before day 3, the post is not separable: early organic spikes look the same, and an estimate from a
day 0-2 read was 57% off on the tuning posts. Views are made non-decreasing first (a one-read dip would fake a jump).

Tested (reconcile/boost_start_eval.py, results/boost_start_eval.json): tuned on half of the posts, reported on the other half.

Input format (sql/15 output, one line per post): psrk|I or T|first read date|age0:views0;dage:dviews;...
"""
import numpy as np
import pandas as pd

PARAMS = dict(t_early=3.0, t_mid=1.6, t_late=1.15, m=0.002, s=1.5, t_lead=1.15, min_age=3)
NOTE_EARLY = "first jump before day 3"
NOTE_NONE = "no clear jump in the first 90 days"
NOTE_FEW = "fewer than 2 public reads in the first 90 days"


def _threshold(age, p):
    return p["t_early"] if age <= 1 else (p["t_mid"] if age <= 3 else p["t_late"])


def detect(ages, views, curve, p=PARAMS):
    """(index of the pre-boost read, note). Index is None when no usable pre-boost read is found."""
    a = np.minimum(np.asarray(ages, int), len(curve) - 1)
    v = np.maximum.accumulate(np.asarray(views, float))
    if len(a) < 2 or v[-1] <= 0:
        return None, NOTE_FEW
    c = curve[a]
    u = v / c                                   # implied end-of-curve organic size at each read
    for i in range(len(a) - 1):
        if v[i] <= 0:
            continue
        exc = v[i + 1] - v[i] * c[i + 1] / c[i]
        if u[i + 1] / u[i] >= _threshold(a[i], p) and exc >= p["m"] * v[-1] and u[-1] / u[i] >= p["s"]:
            j = i
            while j > 0 and v[j - 1] > 0 and u[j] / u[j - 1] >= p["t_lead"] and v[j] - v[j - 1] * c[j] / c[j - 1] > 0.01 * v[-1]:
                j -= 1
            return (j, "jump") if a[j] >= p["min_age"] else (None, NOTE_EARLY)
    return None, NOTE_NONE


def parse(line):
    psrk, p, od0, s = line.rstrip("\n").split("|")
    d = np.array([[float(x) for x in t.split(":")[:2]] for t in s.split(";")])
    return psrk, ("Instagram" if p == "I" else "Tiktok"), od0, np.cumsum(d[:, 0]).astype(int), np.cumsum(d[:, 1])


def load_series(path):
    rows = [parse(line) for line in open(path) if line.strip()]
    return pd.DataFrame(rows, columns=["psrk", "platform", "od0", "ages", "views"])


def detect_all(series, curves, p=PARAMS):
    """One row per post in series: preboost_age_detected, views_preboost_detected (NaN when none), detect_note."""
    out = []
    for r in series.itertuples():
        i, note = detect(r.ages, r.views, curves[r.platform], p)
        out.append((r.psrk, r.platform, r.ages[i] if i is not None else np.nan, r.views[i] if i is not None else np.nan, note))
    return pd.DataFrame(out, columns=["psrk", "platform", "preboost_age_detected", "views_preboost_detected", "detect_note"])
