-- Paid classification table (one row per post, refreshed daily by pipeline/run_daily.py).
-- NOT CREATED YET. Needs a schema we are allowed to write to; {{TARGET_SCHEMA}} is a placeholder
-- (for example a BI sandbox schema). Ask the data platform owner before the first run.

CREATE TABLE IF NOT EXISTS {{TARGET_SCHEMA}}.PAID_CLASSIFICATION__POST (
  POST_SCRAPER_REFERENCE_KEY  VARCHAR      NOT NULL,
  POST_PLATFORM               VARCHAR      NOT NULL,   -- Instagram | Tiktok
  POST_TYPE                   VARCHAR,
  POST_URL                    VARCHAR,
  POST_PUBLISHED_DATE         DATE,
  -- paid or not
  PAID_STATUS                 VARCHAR      NOT NULL,   -- PAID_CONFIRMED | PAID_MEASURED | PAID_LOGGED | PAID_PREDICTED | ORGANIC_PREDICTED | NOT_SCORED
  BOOST_EVIDENCE              VARCHAR      NOT NULL,   -- tier from sql/05 (CONFIRMED_PAID_TAG, CONFIRMED_AD_LINK, ...)
  IS_PAID                     BOOLEAN,                 -- NULL when not scored
  PAID_BASIS                  VARCHAR,                 -- evidence | model
  BOOST_START_DATE            DATE,                    -- first spend day or manual paid date, whichever is first
  FIRST_SPEND_DATE            DATE,
  AD_SPEND                    NUMBER(18, 2),            -- as reported in the ad tables (account currency)
  PAID_IMPRESSIONS_IG         NUMBER,                  -- Meta, Instagram placement (EDW ad tables)
  PAID_IMPRESSIONS_FB         NUMBER,                  -- Meta, Facebook placement
  PAID_PLAYS_IG               NUMBER,
  PAID_PLAYS_FB               NUMBER,
  MODEL_SCORE                 FLOAT,                   -- 0-1, every post with >= 28 days of data; decides only when there is no evidence
  MODEL_THRESHOLD             FLOAT,
  MODEL_VERSION               VARCHAR,
  MODEL_NOTE                  VARCHAR,                 -- why there is no model score (too young, too few reads, ...)
  -- views
  VIEWS_PUBLIC_LATEST         NUMBER,                  -- Nimble public (Instagram: organic + paid served on Instagram)
  VIEWS_OPTIN_LATEST          NUMBER,                  -- opt-in private (Instagram: organic only)
  VIEWS_PUBLIC_PREBOOST       NUMBER,                  -- last public read before the boost start
  PREBOOST_AGE_DAYS           NUMBER,
  POST_AGE_DAYS               NUMBER,
  ORGANIC_VIEWS_EST           NUMBER,
  ORGANIC_VIEWS_METHOD        VARCHAR,                 -- see reconcile/estimate.py
  ORGANIC_VIEWS_CONFIDENCE    VARCHAR,                 -- measured | high | medium | low | none | n/a
  PAID_VIEWS_ON_PLATFORM_EST  NUMBER,                  -- public - organic (Instagram: paid plays served on Instagram only)
  RUN_DATE                    DATE         NOT NULL,
  UPDATED_AT                  TIMESTAMP_NTZ NOT NULL,
  PRIMARY KEY (POST_SCRAPER_REFERENCE_KEY, POST_PLATFORM)
);

-- The pipeline loads today's rows into {{TARGET_SCHEMA}}.PAID_CLASSIFICATION__POST_STAGE (same columns), then:
MERGE INTO {{TARGET_SCHEMA}}.PAID_CLASSIFICATION__POST t
USING {{TARGET_SCHEMA}}.PAID_CLASSIFICATION__POST_STAGE s
  ON t.POST_SCRAPER_REFERENCE_KEY = s.POST_SCRAPER_REFERENCE_KEY AND t.POST_PLATFORM = s.POST_PLATFORM
WHEN MATCHED THEN UPDATE SET
  POST_TYPE = s.POST_TYPE, POST_URL = s.POST_URL, POST_PUBLISHED_DATE = s.POST_PUBLISHED_DATE,
  PAID_STATUS = s.PAID_STATUS, BOOST_EVIDENCE = s.BOOST_EVIDENCE, IS_PAID = s.IS_PAID, PAID_BASIS = s.PAID_BASIS, BOOST_START_DATE = s.BOOST_START_DATE,
  FIRST_SPEND_DATE = s.FIRST_SPEND_DATE, AD_SPEND = s.AD_SPEND,
  PAID_IMPRESSIONS_IG = s.PAID_IMPRESSIONS_IG, PAID_IMPRESSIONS_FB = s.PAID_IMPRESSIONS_FB,
  PAID_PLAYS_IG = s.PAID_PLAYS_IG, PAID_PLAYS_FB = s.PAID_PLAYS_FB,
  MODEL_SCORE = s.MODEL_SCORE, MODEL_THRESHOLD = s.MODEL_THRESHOLD, MODEL_VERSION = s.MODEL_VERSION, MODEL_NOTE = s.MODEL_NOTE,
  VIEWS_PUBLIC_LATEST = s.VIEWS_PUBLIC_LATEST, VIEWS_OPTIN_LATEST = s.VIEWS_OPTIN_LATEST,
  VIEWS_PUBLIC_PREBOOST = s.VIEWS_PUBLIC_PREBOOST, PREBOOST_AGE_DAYS = s.PREBOOST_AGE_DAYS, POST_AGE_DAYS = s.POST_AGE_DAYS,
  ORGANIC_VIEWS_EST = s.ORGANIC_VIEWS_EST, ORGANIC_VIEWS_METHOD = s.ORGANIC_VIEWS_METHOD,
  ORGANIC_VIEWS_CONFIDENCE = s.ORGANIC_VIEWS_CONFIDENCE, PAID_VIEWS_ON_PLATFORM_EST = s.PAID_VIEWS_ON_PLATFORM_EST,
  RUN_DATE = s.RUN_DATE, UPDATED_AT = s.UPDATED_AT
WHEN NOT MATCHED THEN INSERT VALUES (
  s.POST_SCRAPER_REFERENCE_KEY, s.POST_PLATFORM, s.POST_TYPE, s.POST_URL, s.POST_PUBLISHED_DATE,
  s.PAID_STATUS, s.BOOST_EVIDENCE, s.IS_PAID, s.PAID_BASIS, s.BOOST_START_DATE, s.FIRST_SPEND_DATE, s.AD_SPEND,
  s.PAID_IMPRESSIONS_IG, s.PAID_IMPRESSIONS_FB, s.PAID_PLAYS_IG, s.PAID_PLAYS_FB,
  s.MODEL_SCORE, s.MODEL_THRESHOLD, s.MODEL_VERSION, s.MODEL_NOTE,
  s.VIEWS_PUBLIC_LATEST, s.VIEWS_OPTIN_LATEST, s.VIEWS_PUBLIC_PREBOOST, s.PREBOOST_AGE_DAYS, s.POST_AGE_DAYS,
  s.ORGANIC_VIEWS_EST, s.ORGANIC_VIEWS_METHOD, s.ORGANIC_VIEWS_CONFIDENCE, s.PAID_VIEWS_ON_PLATFORM_EST,
  s.RUN_DATE, s.UPDATED_AT);
