"""Daily paid classification run. SCAFFOLD: the logic is tested offline (tests/test_pipeline.py); the Snowflake
read/write path has NOT been run yet (no service account or write schema yet).

Steps
  1. sql/05_boost_flags.sql                 evidence tier + organic-estimate inputs, every IG / TikTok campaign post
  2. sql/02_detector_features_and_labels.sql 30-day public features (posts with >= 28 days of data)
  3. model score for NO_PAID_EVIDENCE posts  (models/boost_detector_<platform>.joblib from detector/train_eval.py)
  4. organic estimate                       (reconcile/estimate.py)
  5. PAID_STATUS, then stage + MERGE into {{TARGET_SCHEMA}}.PAID_CLASSIFICATION__POST (sql/09)

Offline (default): read step 1 and 2 outputs from files, write outputs/paid_classification_<date>.csv (git-ignored).
  python3 -m pipeline.run_daily --flags data/flags.csv --features data/detector_dataset_v2.psv
Snowflake: pip install "snowflake-connector-python[pandas]"; set SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_ROLE,
SNOWFLAKE_WAREHOUSE and SNOWFLAKE_PASSWORD (or SNOWFLAKE_PRIVATE_KEY_PATH).
  python3 -m pipeline.run_daily --snowflake [--target-schema DB.SCHEMA]   (no --target-schema = read only, CSV out)
"""
import argparse
import datetime as dt
import os
import tempfile

import joblib
import numpy as np
import pandas as pd

from detector.features import COLS, FEATURES, build, load
from reconcile.estimate import estimate

STATUS = {"CONFIRMED_PAID_TAG": "PAID_CONFIRMED", "CONFIRMED_AD_LINK": "PAID_CONFIRMED",
          "MEASURED_OPTIN_GAP": "PAID_MEASURED", "MEASURED_SOCAPI_GAP": "PAID_MEASURED",
          "LOGGED_PAID_DATE_ONLY": "PAID_LOGGED"}
PAID = {"PAID_CONFIRMED", "PAID_MEASURED", "PAID_LOGGED", "PAID_PREDICTED"}
MODEL_VERSION = "boost_detector_v1 (train < 2026-07-01)"
# preference order per platform: v2.2 (followers known at the horizon, plan amendment 5), then v2.1 (Instagram labels
# corrected for stale opt-in, amendment 4), then v2
V2_MODELS = {"Tiktok": ["models/boost_detector_v2_2", "models/boost_detector_v2"],
             "Instagram": ["models/boost_detector_v2_2", "models/boost_detector_v2_1", "models/boost_detector_v2"]}
OPTIN_ORGANIC = 0.90     # Instagram: opt-in organic views >= 90% of public views = measured organic (same cut as the N label)


def score(features):
    """Model score per post. features = sql/02 output, already passed through detector.features.build."""
    out = []
    for platform in ["Tiktok", "Instagram"]:
        path = f"models/boost_detector_{platform.lower()}.joblib"
        g = features[(features.platform == platform) & (features.v30 > 0) & (features.followers > 0)].copy()
        if g.empty or not os.path.exists(path):
            continue
        bundle = joblib.load(path)
        g["model_score"] = bundle["model"].predict_proba(g[FEATURES])[:, 1]
        g["model_threshold"] = bundle["threshold"]
        out.append(g[["psrk", "platform", "model_score", "model_threshold"]])
    cols = ["psrk", "platform", "model_score", "model_threshold"]
    return pd.concat(out) if out else pd.DataFrame(columns=cols)


def score_v2(raw, raw_hf=None):
    """Model v2: the longest horizon a post qualifies for wins (day 60, then day 30, then day 14).
    raw = sql/11 (+ sql/12) output loaded with detector.features_v2.load; raw_hf = the same with sql/14 horizon followers,
    used for v2.2 models (they were trained on it). Each saved model names its feature set and version."""
    from detector.train_v2 import features
    out, cache = [], {}
    for H in (60, 30, 14):
        for platform in ["Tiktok", "Instagram"]:
            path = next((f"{pre}_{platform.lower()}_h{H}.joblib" for pre in V2_MODELS[platform]
                         if os.path.exists(f"{pre}_{platform.lower()}_h{H}.joblib")), None)
            if path is None:
                continue
            b = joblib.load(path)
            src = raw_hf if b.get("version") == "v2.2" else raw
            if src is None:
                raise ValueError(f"{path} needs sql/14 horizon followers (--features-v2-followers)")
            key = (H, b.get("feature_set", "base"), id(src))
            if key not in cache:
                cache[key] = features(src, H, key[1])
            X, ok = cache[key]
            m = ok & (src.platform == platform)
            if not m.any():
                continue
            g = src.loc[m, ["psrk", "platform"]].copy()
            g["model_score"] = b["model"].predict_proba(X.loc[m, b["features"]])[:, 1]
            g["model_threshold"] = b["threshold"]
            g["model_horizon"] = H
            g["model_version"] = f'boost_detector_{b.get("version", "v2")} (day {H}, train < {b.get("train_end", "2026-07-01")})'
            out.append(g)
    cols = ["psrk", "platform", "model_score", "model_threshold", "model_horizon", "model_version"]
    if not out:
        return pd.DataFrame(columns=cols)
    s = pd.concat(out).sort_values("model_horizon", ascending=False)
    return s.drop_duplicates(["psrk", "platform"])[cols]


PROVISIONAL_NOTE = "provisional: day-14 Instagram score, checked again at day 30 (the day-14 model misses more boosts)"


def add_model_note_v2(d, raw):
    """Why there is no v2 score; for a day-14 Instagram 'organic' call, that it is provisional until the day-30 score."""
    seen = raw.set_index(["psrk", "platform"])
    key = list(zip(d.psrk, d.platform))
    a_min = pd.Series([seen.a_min.get(k, np.nan) for k in key], index=d.index)
    pub = pd.to_datetime(d.pub, errors="coerce")
    d["model_note"] = np.select(
        [d.model_score.notna(), pub < pd.Timestamp("2025-01-01"), a_min.isna(), d.age_latest < 14, a_min > 10],
        [None, "published before 2025", "no public read in days 0-60", "under 14 days of data", "first public read after day 10"],
        "no public views or followers")
    # Instagram day-14 model: F1 0.80 vs 0.89 at day 30 on the same posts (C4 missed); a 'paid' call is still 88% right,
    # but an 'organic' call can be a boost that has not shown yet. The daily run re-scores the post at day 30.
    hz = d["model_horizon"] if "model_horizon" in d else pd.Series(np.nan, index=d.index)
    prov = (d.platform == "Instagram") & (hz == 14) & (d.paid_status == "ORGANIC_PREDICTED")
    d.loc[prov, "model_note"] = PROVISIONAL_NOTE
    return d


def classify(flags, scores, run_date):
    known = set(STATUS) | {"NO_PAID_EVIDENCE"}
    bad = ~flags.boost_evidence.isin(known)
    if bad.any():
        raise ValueError(f"unknown boost_evidence values: {sorted(flags.boost_evidence[bad].astype(str).unique())[:5]}")
    d = flags.merge(scores, on=["psrk", "platform"], how="left")
    status = d.boost_evidence.map(STATUS)
    no_ev = d.boost_evidence == "NO_PAID_EVIDENCE"
    status[no_ev & (d.model_score >= d.model_threshold)] = "PAID_PREDICTED"
    status[no_ev & (d.model_score < d.model_threshold)] = "ORGANIC_PREDICTED"
    # measured organic: opt-in shows (almost) all public views are organic, so no boost added views. Beats the model.
    # sql/05 optin_ratio = ratio at the last read where opt-in still updated; older flag files only carry the latest totals
    # sql/05 always has optin_ratio (NULL = no qualifying opt-in read); only flag files older than it fall back to totals
    optin = (pd.to_numeric(d["optin_ratio"], errors="coerce") if "optin_ratio" in d else
             d.views_private_latest / d.views_public_latest.where(d.views_public_latest > 0))
    stale = d["optin_stale"].fillna(False).astype(bool) if "optin_stale" in d else pd.Series(False, index=d.index)
    # stale opt-in only proves organic up to the freeze, so it is not measured organic today (plan amendment 4)
    measured_org = no_ev & (d.platform == "Instagram") & (optin >= OPTIN_ORGANIC) & ~stale
    status[measured_org] = "ORGANIC_MEASURED"
    d["paid_status"] = status.fillna("NOT_SCORED")
    # evidence always wins; the model score is kept on every scored post for comparison, but decides only without evidence
    d["paid_basis"] = np.select([~no_ev | measured_org, d.paid_status.isin(["PAID_PREDICTED", "ORGANIC_PREDICTED"])],
                                ["evidence", "model"], None)
    d["is_paid"] = d.paid_status.map(lambda s: None if s == "NOT_SCORED" else s in PAID)
    d["boosted"] = d.paid_status.isin(PAID)
    e = estimate(d)
    d = pd.concat([d.reset_index(drop=True), e], axis=1)
    d.loc[d.paid_status == "NOT_SCORED", "organic_views_method"] = "public views (no paid record; model could not score)"
    d["paid_views_on_platform_est"] = (d.views_public_latest - d.organic_views_est).where(d.boosted).clip(lower=0)
    if "model_version" in d:            # v2 scoring names the model per post
        d["model_version"] = d.model_version.where(d.model_score.notna(), None)
    else:
        d["model_version"] = np.where(d.model_score.notna(), MODEL_VERSION, None)
    d["run_date"] = run_date
    d["updated_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d


OUT_COLS = {  # table column -> frame column
    "POST_SCRAPER_REFERENCE_KEY": "psrk", "POST_PLATFORM": "platform", "POST_TYPE": "post_type", "POST_URL": "post_url",
    "POST_PUBLISHED_DATE": "pub", "PAID_STATUS": "paid_status", "BOOST_EVIDENCE": "boost_evidence",
    "IS_PAID": "is_paid", "PAID_BASIS": "paid_basis",
    "BOOST_START_DATE": "boost_start", "FIRST_SPEND_DATE": "first_spend", "AD_SPEND": "spend",
    "PAID_IMPRESSIONS_IG": "paid_impressions_ig", "PAID_IMPRESSIONS_FB": "paid_impressions_fb",
    "PAID_PLAYS_IG": "paid_plays_ig", "PAID_PLAYS_FB": "paid_plays_fb",
    "MODEL_SCORE": "model_score", "MODEL_THRESHOLD": "model_threshold", "MODEL_HORIZON_DAYS": "model_horizon",
    "MODEL_VERSION": "model_version", "MODEL_NOTE": "model_note",
    "VIEWS_PUBLIC_LATEST": "views_public_latest", "VIEWS_OPTIN_LATEST": "views_private_latest",
    "VIEWS_PUBLIC_PREBOOST": "views_before_first_spend", "PREBOOST_AGE_DAYS": "preboost_age_days",
    "POST_AGE_DAYS": "age_latest", "ORGANIC_VIEWS_EST": "organic_views_est", "ORGANIC_VIEWS_METHOD": "organic_views_method",
    "ORGANIC_VIEWS_CONFIDENCE": "organic_views_confidence", "PAID_VIEWS_ON_PLATFORM_EST": "paid_views_on_platform_est",
    "RUN_DATE": "run_date", "UPDATED_AT": "updated_at"}


def add_model_note(d, features):
    """Why the model has no score for a post (the model needs >= 28 days and >= 5 public reads in days 0-30, posts from 2025)."""
    have = set(zip(features.psrk, features.platform))
    pub = pd.to_datetime(d.pub, errors="coerce")
    infeat = pd.Series([(a, b) in have for a, b in zip(d.psrk, d.platform)], index=d.index)
    d["model_note"] = np.select(
        [d.model_score.notna(), pub < pd.Timestamp("2025-01-01"), d.age_latest < 28, ~infeat],
        [None, "published before 2025", "under 28 days of data", "fewer than 5 public reads in days 0-30"],
        "no public views or followers")
    return d


def to_table(d):
    """Table columns in order. A column the input does not carry (e.g. POST_URL from a compact export) is left empty."""
    return pd.DataFrame({k: (d[v] if v in d else pd.Series([None] * len(d))) for k, v in OUT_COLS.items()})


def write_parquet(table, path):
    t = table.copy()
    for c in ["POST_PUBLISHED_DATE", "BOOST_START_DATE", "FIRST_SPEND_DATE", "RUN_DATE"]:
        t[c] = pd.to_datetime(t[c], errors="coerce").dt.date
    t["IS_PAID"] = t["IS_PAID"].astype("boolean")
    for c in ["PAID_IMPRESSIONS_IG", "PAID_IMPRESSIONS_FB", "PAID_PLAYS_IG", "PAID_PLAYS_FB", "VIEWS_PUBLIC_LATEST", "VIEWS_OPTIN_LATEST",
              "VIEWS_PUBLIC_PREBOOST", "PREBOOST_AGE_DAYS", "POST_AGE_DAYS", "ORGANIC_VIEWS_EST", "PAID_VIEWS_ON_PLATFORM_EST"]:
        t[c] = pd.to_numeric(t[c], errors="coerce").round().astype("Int64")
    if "POST_URL" in t and t["POST_URL"].isna().all():
        t = t.drop(columns="POST_URL")
    t.to_parquet(path, index=False)   # UPPERCASE names = the Snowflake table columns (sql/09), so COPY INTO matches by name
    return t


def read_flags(path, optin_live=None, tier_changes=None):
    """Offline flags (sql/05 output). optin_live / tier_changes bring an older flags file up to plan amendment 4
    (stale opt-in): the opt-in fields of the new sql/05, and the evidence tier of the posts it changes."""
    f = pd.read_csv(path, dtype={"psrk": str})
    f.columns = [c.lower() for c in f.columns]
    for c in ["pub", "obs_latest", "first_spend", "paid_date", "boost_start"]:
        if c in f:
            f[c] = pd.to_datetime(f[c], errors="coerce")
    for path in (optin_live, tier_changes):
        if path and not os.path.exists(path):
            raise FileNotFoundError(path)
    if optin_live:
        o = pd.read_csv(optin_live, sep="|", header=None, dtype={"psrk": str},
                        names=["psrk", "optin_ratio", "optin_ratio_latest", "optin_stale", "optin_freeze_age", "optin_views_frozen"])
        o["optin_stale"] = o.optin_stale.astype(int).astype(bool)
        o["platform"] = "Instagram"
        f = f.merge(o, on=["psrk", "platform"], how="left", validate="one_to_one")
    if tier_changes:
        c = pd.read_csv(tier_changes, sep="|", header=None, names=["psrk", "tier_new"], dtype=str).assign(platform="Instagram")
        if c.psrk.duplicated().any():
            raise ValueError(f"duplicate posts in {tier_changes}")
        f = f.merge(c, on=["psrk", "platform"], how="left", validate="many_to_one")
        wrong = f.tier_new.notna() & (f.boost_evidence != "MEASURED_OPTIN_GAP")
        if wrong.any():     # the stale opt-in fix only moves posts out of MEASURED_OPTIN_GAP
            raise ValueError(f"{int(wrong.sum())} listed posts are not MEASURED_OPTIN_GAP in {tier_changes}")
        f["boost_evidence"] = f.tier_new.fillna(f.boost_evidence)
        f = f.drop(columns="tier_new")
    return f


# ---------- Snowflake (untested scaffold) ----------
def connect():
    import snowflake.connector
    kw = {k: os.environ[f"SNOWFLAKE_{k.upper()}"] for k in ["account", "user", "role", "warehouse"]}
    if os.environ.get("SNOWFLAKE_PRIVATE_KEY_PATH"):
        kw["private_key_file"] = os.environ["SNOWFLAKE_PRIVATE_KEY_PATH"]
    else:
        kw["password"] = os.environ["SNOWFLAKE_PASSWORD"]
    return snowflake.connector.connect(**kw)


def query(conn, path):
    sql = open(path).read()
    cur = conn.cursor()
    cur.execute(sql)
    df = cur.fetch_pandas_all()
    df.columns = [c.lower() for c in df.columns]
    return df


def write(conn, table, target_schema):
    from snowflake.connector.pandas_tools import write_pandas
    db, schema = target_schema.split(".")
    ddl = open("sql/09_paid_classification_table.sql").read().replace("{{TARGET_SCHEMA}}", target_schema)
    create, merge = ddl.split("MERGE INTO", 1)
    create = create[create.index("CREATE TABLE"):]
    cur = conn.cursor()
    cur.execute(create)
    cur.execute(f"CREATE TABLE IF NOT EXISTS {target_schema}.PAID_CLASSIFICATION__POST_STAGE LIKE {target_schema}.PAID_CLASSIFICATION__POST")
    write_pandas(conn, table, "PAID_CLASSIFICATION__POST_STAGE", database=db, schema=schema, overwrite=True,
                 auto_create_table=False, use_logical_type=True)
    cur.execute("MERGE INTO" + merge)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flags", default="data/flags.csv", help="sql/05 output (offline mode)")
    ap.add_argument("--optin-live", default=None, help="offline: opt-in fields of the amended sql/05 (psrk|ratio|latest|stale|freeze age|views)")
    ap.add_argument("--tier-changes", default=None, help="offline: posts whose evidence tier the amended sql/05 changes (psrk|tier)")
    ap.add_argument("--features", default="data/detector_dataset_v2.psv", help="offline, --model v1 only: sql/02 output")
    ap.add_argument("--model", choices=["v2", "v1"], default="v2", help="v2: day 60/30/14 models (TikTok v2, Instagram v2.1)")
    ap.add_argument("--features-v2", default="data/v2_raw.psv", help="offline: sql/11 output (model v2)")
    ap.add_argument("--features-v2-extra", default="data/v2_extra.psv", help="offline: sql/12 output (model v2)")
    ap.add_argument("--features-v2-followers", default="data/v2_followers.psv", help="offline: sql/14 output (model v2.2)")
    ap.add_argument("--snowflake", action="store_true")
    ap.add_argument("--target-schema", default=None, help="DB.SCHEMA to MERGE into; omit to write CSV only")
    ap.add_argument("--run-date", default=str(dt.date.today()))
    ap.add_argument("--parquet", default=None, help="also write the table to this Parquet file (needs pyarrow)")
    a = ap.parse_args()

    if a.snowflake:
        conn = connect()
        flags = query(conn, "sql/05_boost_flags.sql")
        for c in ["pub", "obs_latest", "first_spend", "paid_date", "boost_start"]:
            flags[c] = pd.to_datetime(flags[c], errors="coerce")
        if a.model == "v1":
            raw1 = query(conn, "sql/02_detector_features_and_labels.sql")
            with tempfile.NamedTemporaryFile("w", suffix=".psv", delete=False) as tmp:
                raw1[COLS].to_csv(tmp.name, sep="|", header=False, index=False)
            features = build(load(tmp.name))
        else:   # sql/11 and sql/12 return one pipe-separated string per post (column s), read by detector.features_v2.load
            paths = {}
            for name, sql in (("raw", "sql/11_v2_features_and_labels.sql"), ("extra", "sql/12_v2_jump_features.sql"),
                              ("followers", "sql/14_v2_horizon_followers.sql")):
                q = query(conn, sql).sort_values("rn")
                with tempfile.NamedTemporaryFile("w", suffix=".psv", delete=False) as tmp:
                    tmp.write("\n".join(q.s.astype(str)) + "\n")
                paths[name] = tmp.name
            a.features_v2, a.features_v2_extra, a.features_v2_followers = paths["raw"], paths["extra"], paths["followers"]
    else:
        conn = None
        flags = read_flags(a.flags, a.optin_live, a.tier_changes)
        if a.model == "v1":
            features = build(load(a.features))

    if a.model == "v2":
        from detector.features_v2 import load as load_v2
        raw = load_v2(a.features_v2, extra=a.features_v2_extra)
        raw_hf = load_v2(a.features_v2, extra=a.features_v2_extra, horizon_followers=a.features_v2_followers) \
            if a.features_v2_followers and os.path.exists(a.features_v2_followers) else None
        d = add_model_note_v2(classify(flags, score_v2(raw, raw_hf), pd.Timestamp(a.run_date).date()), raw_hf if raw_hf is not None else raw)
    else:
        d = add_model_note(classify(flags, score(features), pd.Timestamp(a.run_date).date()), features)
    table = to_table(d)
    os.makedirs("results", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)   # row-level output: git-ignored
    out = f"outputs/paid_classification_{a.run_date}.csv"
    table.to_csv(out, index=False)
    print(table.groupby(["POST_PLATFORM", "PAID_STATUS"]).size().to_string())
    print(table.groupby(["POST_PLATFORM", "ORGANIC_VIEWS_CONFIDENCE"]).size().to_string())
    print("wrote", out)
    if a.parquet:
        write_parquet(table, a.parquet)
        print("wrote", a.parquet)
    if conn is not None and a.target_schema:
        write(conn, table, a.target_schema)
        print("merged into", f"{a.target_schema}.PAID_CLASSIFICATION__POST")


if __name__ == "__main__":
    main()
