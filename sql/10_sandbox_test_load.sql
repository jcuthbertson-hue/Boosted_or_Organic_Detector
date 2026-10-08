-- Load the paid classification file into a TEST area in Snowflake, test file first.
-- Run in a Snowsight worksheet, top to bottom, as role USERS_BUSINESS_INTELLIGENCE.
-- NOT EXECUTED YET: the analysis session's Snowflake connection may not create objects, so a person runs this.
--
-- Safety
--   * Everything lives in DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON. No production table is read for writing.
--   * Visible to the USERS_BUSINESS_INTELLIGENCE role (the BI team) and admins, not to other teams.
--     Snowflake does not allow tables in personal databases (USER$...), so "only me" is not possible.
--   * Step A loads a 500-row TEST file. Run step B (all posts) only if every check in step A matches.
--
-- Files (from pipeline/run_daily.py --parquet, column names = UPPERCASE table columns):
--   paid_classification_2026-10-07_TEST_500.parquet   500 posts (250 Instagram, 250 TikTok), random sample
--   paid_classification_2026-10-07.parquet            41,484 posts (all Instagram / TikTok campaign posts in BIRA)
--   Models: boost_detector_v2.2 on both platforms (day 60 / 30 / 14; results/model_v2_2_targets.json);
--   Instagram opt-in evidence corrected for stale opt-in (plan amendment 4); followers known at the horizon (amendment 5)

USE ROLE USERS_BUSINESS_INTELLIGENCE;
USE WAREHOUSE BI_WAREHOUSE;

-- 0. Test area ------------------------------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON
  COMMENT = 'Personal sandbox (jcuthbertson): paid vs organic research. Test tables only; safe to drop.';
USE SCHEMA DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON;
CREATE FILE FORMAT IF NOT EXISTS PARQUET_FMT TYPE = PARQUET;
CREATE STAGE IF NOT EXISTS LOAD_STAGE FILE_FORMAT = PARQUET_FMT COMMENT = 'Upload area for paid classification test files';

-- Upload the two .parquet files to LOAD_STAGE:
--   Snowsight: open the stage LOAD_STAGE in this schema and add the files, or
--   SnowSQL / Snowflake CLI:
--     PUT file:///<path>/paid_classification_2026-10-07_TEST_500.parquet @LOAD_STAGE AUTO_COMPRESS = FALSE;
--     PUT file:///<path>/paid_classification_2026-10-07.parquet          @LOAD_STAGE AUTO_COMPRESS = FALSE;
LIST @LOAD_STAGE;

-- A. TEST load: 500 posts -------------------------------------------------------------------------------------
CREATE OR REPLACE TABLE PAID_CLASSIFICATION__POST_TEST
  USING TEMPLATE (
    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
    FROM TABLE(INFER_SCHEMA(LOCATION => '@LOAD_STAGE/paid_classification_2026-10-07_TEST_500.parquet',
                            FILE_FORMAT => 'PARQUET_FMT')));
COPY INTO PAID_CLASSIFICATION__POST_TEST
  FROM @LOAD_STAGE/paid_classification_2026-10-07_TEST_500.parquet
  FILE_FORMAT = (FORMAT_NAME = 'PARQUET_FMT') MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- A1. Rows and paid split. Expected:
--   Instagram  250 rows | paid 42 | organic 144 | unknown 64
--   Tiktok     250 rows | paid 31 | organic 152 | unknown 67
SELECT POST_PLATFORM, COUNT(*) rows_, COUNT_IF(IS_PAID) paid, COUNT_IF(NOT IS_PAID) organic, COUNT_IF(IS_PAID IS NULL) unknown
FROM PAID_CLASSIFICATION__POST_TEST GROUP BY 1 ORDER BY 1;

-- A2. One row per post. Expected: 500 | 500
SELECT COUNT(*) rows_, COUNT(DISTINCT POST_SCRAPER_REFERENCE_KEY || '|' || POST_PLATFORM) posts FROM PAID_CLASSIFICATION__POST_TEST;

-- A3. Every post links back to BIRA. Expected: 500 | 500
SELECT COUNT(*) rows_, COUNT_IF(m.k IS NOT NULL) found_in_bira
FROM PAID_CLASSIFICATION__POST_TEST p
LEFT JOIN (SELECT DISTINCT POST_SCRAPER_REFERENCE_KEY k, POST_PLATFORM pf
           FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE) m
  ON m.k = p.POST_SCRAPER_REFERENCE_KEY AND m.pf = p.POST_PLATFORM;

-- B. FULL load: all 41,484 posts (only after A1-A3 match) --------------------------------------------------------
CREATE OR REPLACE TABLE PAID_CLASSIFICATION__POST
  USING TEMPLATE (
    SELECT ARRAY_AGG(OBJECT_CONSTRUCT(*))
    FROM TABLE(INFER_SCHEMA(LOCATION => '@LOAD_STAGE/paid_classification_2026-10-07.parquet',
                            FILE_FORMAT => 'PARQUET_FMT')));
COPY INTO PAID_CLASSIFICATION__POST
  FROM @LOAD_STAGE/paid_classification_2026-10-07.parquet
  FILE_FORMAT = (FORMAT_NAME = 'PARQUET_FMT') MATCH_BY_COLUMN_NAME = CASE_INSENSITIVE;

-- B1. Expected:
--   Instagram  23,552 rows | paid 3,112 | organic 13,962 | unknown 6,478
--   Tiktok     17,932 rows | paid 1,877 | organic 11,529 | unknown 4,526
SELECT POST_PLATFORM, COUNT(*) rows_, COUNT_IF(IS_PAID) paid, COUNT_IF(NOT IS_PAID) organic, COUNT_IF(IS_PAID IS NULL) unknown
FROM PAID_CLASSIFICATION__POST GROUP BY 1 ORDER BY 1;

-- B2. Expected: 41,484 | 41,484 | 41,484
SELECT COUNT(*) rows_, COUNT(DISTINCT POST_SCRAPER_REFERENCE_KEY || '|' || POST_PLATFORM) posts, COUNT_IF(m.k IS NOT NULL) found_in_bira
FROM PAID_CLASSIFICATION__POST p
LEFT JOIN (SELECT DISTINCT POST_SCRAPER_REFERENCE_KEY k, POST_PLATFORM pf
           FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE) m
  ON m.k = p.POST_SCRAPER_REFERENCE_KEY AND m.pf = p.POST_PLATFORM;

-- C. How to use it: attach the paid call to BIRA rows (one BIRA row per post x campaign) --------------------------
SELECT m.CAMPAIGN_ORGANIZATION, m.POST_URL, m.POST_PLATFORM, m.VIEWS_LATEST,
       p.IS_PAID, p.PAID_BASIS, p.PAID_STATUS, p.MODEL_SCORE, p.ORGANIC_VIEWS_EST, p.ORGANIC_VIEWS_CONFIDENCE
FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE m
LEFT JOIN DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON.PAID_CLASSIFICATION__POST p
  ON p.POST_SCRAPER_REFERENCE_KEY = m.POST_SCRAPER_REFERENCE_KEY AND p.POST_PLATFORM = m.POST_PLATFORM
LIMIT 100;

-- D. Clean up when done -----------------------------------------------------------------------------------------
-- DROP TABLE IF EXISTS DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON.PAID_CLASSIFICATION__POST_TEST;
-- REMOVE @DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON.LOAD_STAGE;
-- DROP SCHEMA IF EXISTS DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON;   -- removes everything above
