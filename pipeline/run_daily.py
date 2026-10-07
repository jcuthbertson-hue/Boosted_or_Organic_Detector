"""Daily paid classification run. SCAFFOLD: the logic is tested offline (tests/test_pipeline.py); the Snowflake
read/write path has NOT been run yet (no service account or write schema yet).

Steps
  1. sql/05_boost_flags.sql                 evidence tier + organic-estimate inputs, every IG / TikTok campaign post
  2. sql/02_detector_features_and_labels.sql 30-day public features (posts with >= 28 days of data)
  3. model score for NO_PAID_EVIDENCE posts  (models/boost_detector_<platform>.joblib from detector/train_eval.py)
  4. organic estimate                       (reconcile/estimate.py)
  5. PAID_STATUS, then stage + MERGE into {{TARGET_SCHEMA}}.PAID_CLASSIFICATION__POST (sql/09)

Offline (default): read step 1 and 2 outputs from files, write results/paid_classification_<date>.csv (git-ignored).
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


def classify(flags, scores, run_date):
    d = flags.merge(scores, on=["psrk", "platform"], how="left")
    status = d.boost_evidence.map(STATUS)
    no_ev = d.boost_evidence == "NO_PAID_EVIDENCE"
    status[no_ev & (d.model_score >= d.model_threshold)] = "PAID_PREDICTED"
    status[no_ev & (d.model_score < d.model_threshold)] = "ORGANIC_PREDICTED"
    d["paid_status"] = status.fillna("NOT_SCORED")
    d.loc[~no_ev, ["model_score", "model_threshold"]] = np.nan     # the model only speaks where there is no evidence
    d["boosted"] = d.paid_status.isin(PAID)
    e = estimate(d)
    d = pd.concat([d.reset_index(drop=True), e], axis=1)
    d["paid_views_on_platform_est"] = (d.views_public_latest - d.organic_views_est).where(d.boosted).clip(lower=0)
    d["model_version"] = np.where(d.model_score.notna(), MODEL_VERSION, None)
    d["run_date"] = run_date
    d["updated_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    return d


OUT_COLS = {  # table column -> frame column
    "POST_SCRAPER_REFERENCE_KEY": "psrk", "POST_PLATFORM": "platform", "POST_TYPE": "post_type", "POST_URL": "post_url",
    "POST_PUBLISHED_DATE": "pub", "PAID_STATUS": "paid_status", "BOOST_EVIDENCE": "boost_evidence",
    "BOOST_START_DATE": "boost_start", "FIRST_SPEND_DATE": "first_spend", "SPEND_USD": "spend",
    "PAID_IMPRESSIONS_IG": "paid_impressions_ig", "PAID_IMPRESSIONS_FB": "paid_impressions_fb",
    "PAID_PLAYS_IG": "paid_plays_ig", "PAID_PLAYS_FB": "paid_plays_fb",
    "MODEL_SCORE": "model_score", "MODEL_THRESHOLD": "model_threshold", "MODEL_VERSION": "model_version",
    "VIEWS_PUBLIC_LATEST": "views_public_latest", "VIEWS_OPTIN_LATEST": "views_private_latest",
    "VIEWS_PUBLIC_PREBOOST": "views_before_first_spend", "PREBOOST_AGE_DAYS": "preboost_age_days",
    "POST_AGE_DAYS": "age_latest", "ORGANIC_VIEWS_EST": "organic_views_est", "ORGANIC_VIEWS_METHOD": "organic_views_method",
    "ORGANIC_VIEWS_CONFIDENCE": "organic_views_confidence", "PAID_VIEWS_ON_PLATFORM_EST": "paid_views_on_platform_est",
    "RUN_DATE": "run_date", "UPDATED_AT": "updated_at"}


def to_table(d):
    return pd.DataFrame({k: d[v] for k, v in OUT_COLS.items()})


def read_flags(path):
    f = pd.read_csv(path, dtype={"psrk": str})
    f.columns = [c.lower() for c in f.columns]
    for c in ["pub", "obs_latest", "first_spend", "paid_date", "boost_start"]:
        if c in f:
            f[c] = pd.to_datetime(f[c], errors="coerce")
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
    ap.add_argument("--features", default="data/detector_dataset_v2.psv", help="sql/02 output, pipe-separated (offline mode)")
    ap.add_argument("--snowflake", action="store_true")
    ap.add_argument("--target-schema", default=None, help="DB.SCHEMA to MERGE into; omit to write CSV only")
    ap.add_argument("--run-date", default=str(dt.date.today()))
    a = ap.parse_args()

    if a.snowflake:
        conn = connect()
        flags = query(conn, "sql/05_boost_flags.sql")
        raw = query(conn, "sql/02_detector_features_and_labels.sql")
        with tempfile.NamedTemporaryFile("w", suffix=".psv", delete=False) as tmp:
            raw[COLS].to_csv(tmp.name, sep="|", header=False, index=False)
        features = build(load(tmp.name))
        for c in ["pub", "obs_latest", "first_spend", "paid_date", "boost_start"]:
            flags[c] = pd.to_datetime(flags[c], errors="coerce")
    else:
        conn = None
        flags = read_flags(a.flags)
        features = build(load(a.features))

    d = classify(flags, score(features), pd.Timestamp(a.run_date).date())
    table = to_table(d)
    os.makedirs("results", exist_ok=True)
    out = f"results/paid_classification_{a.run_date}.csv"
    table.to_csv(out, index=False)
    print(table.groupby(["POST_PLATFORM", "PAID_STATUS"]).size().to_string())
    print(table.groupby(["POST_PLATFORM", "ORGANIC_VIEWS_CONFIDENCE"]).size().to_string())
    print("wrote", out)
    if conn is not None and a.target_schema:
        write(conn, table, a.target_schema)
        print("merged into", f"{a.target_schema}.PAID_CLASSIFICATION__POST")


if __name__ == "__main__":
    main()
