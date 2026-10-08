-- Model v2 input: one row per in-feed Instagram / TikTok campaign post (published 2025+, at least 7 days of data).
-- Output is one pipe-separated string per post (column order = detector/features_v2.py RAW_COLS), so it pages out of
-- the Snowflake MCP; run the SELECT without CONCAT_WS for a normal table.
--
-- FEATURES (public organic data only, never paid / opt-in / SocAPI):
--   cumulative views at ages 1,2,3,5,7,10,14,21,30,45,60 (last real read on or before that day), likes at 1,3,7,14,30,60,
--   comments at 7 and 30, shares at 30, median followers, number of reads, first / last read age,
--   biggest daily view gain in days 2-7, 8-14, 15-30, 31-60 and the age of the biggest gain.
--   The model for horizon H only uses values up to day H (enforced in Python).
-- LABELS (label-only columns; never features):
--   evidence = ad link (EDW Meta taxonomy / TikTok Spark item id) or the post-ID tag in PAID_MEDIA_UNIFIED.
--   lab30 / lab14: P = paid evidence with first spend in [publish - 3, publish + H - 2], or Instagram opt-in / public < 0.80
--                  on the latest read up to day H; N = no paid evidence at all, no manual paid date, and
--                  (TikTok) the client has >= 5 evidence-linked TikTok posts / (Instagram) opt-in / public >= 0.90 up to day H.
--   Group keys for cross-validation are hashes (no creator or client names leave Snowflake).
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform,
         ANY_VALUE(CAMPAIGN_ORGANIZATION) org, ANY_VALUE(CHANNEL_HANDLE) handle,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc,
         MIN(POST_PUBLISHED_AT)::DATE pub, MIN(POST_PAID_ACTIVITY_DATE) paid_date
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1, 2),
meta_tok AS (SELECT a.AD_KEY, t.value::STRING tok FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
             LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t),
meta_link AS (
  SELECT DISTINCT a.AD_KEY, m.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a JOIN mart m ON m.platform = 'Instagram' AND m.psrk = a.POST_SCRAPER_REFERENCE_KEY
  UNION SELECT DISTINCT t.AD_KEY, m.psrk FROM meta_tok t JOIN mart m ON m.platform = 'Instagram' AND m.sc = t.tok),
paid AS (
  SELECT 'Instagram' platform, l.psrk, MIN(IFF(f.AD_SPEND > 0, f.DATE, NULL)) first_spend, SUM(f.AD_SPEND) spend
  FROM meta_link l JOIN SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = l.AD_KEY GROUP BY 1, 2
  UNION ALL
  SELECT 'Tiktok', a.TIKTOK_ITEM_ID, MIN(IFF(f.SPEND > 0, f.DATE, NULL)), SUM(f.SPEND)
  FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD a JOIN SNOWFLAKE_EDW.EDW.FACT_TIKTOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = a.AD_KEY
  WHERE a.TIKTOK_ITEM_ID IS NOT NULL GROUP BY 1, 2),
tag AS (
  SELECT m.platform, m.psrk, MIN(u."date") first_spend, SUM(u."spend") spend
  FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED u JOIN mart m ON m.platform = 'Instagram' AND m.sc = u."ext_p3_organic_post_id"
  WHERE u."platform" = 'meta' AND u."spend" > 0 GROUP BY 1, 2
  UNION ALL
  SELECT m.platform, m.psrk, MIN(u."date"), SUM(u."spend")
  FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED u JOIN mart m ON m.platform = 'Tiktok' AND m.psrk = u."ext_p3_organic_post_id"
  WHERE u."platform" = 'tiktok' AND u."spend" > 0 GROUP BY 1, 2),
ev AS (
  SELECT platform, psrk, MIN(first_spend) first_spend, MAX(is_tag) has_tag FROM (
    SELECT platform, psrk, first_spend, 0 is_tag FROM paid WHERE spend > 0
    UNION ALL SELECT platform, psrk, first_spend, 1 FROM tag) GROUP BY 1, 2),
org_cov AS (SELECT m.platform, m.org, COUNT(*) n_linked FROM mart m JOIN ev e ON e.psrk = m.psrk AND e.platform = m.platform GROUP BY 1, 2),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform, t.OBSERVATION_DATE od,
         DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age,
         t.VIEWS_PUBLIC v, t.LIKES_PUBLIC l, t.COMMENTS_PUBLIC c, t.SHARES_PUBLIC s, t.VIEWS_PRIVATE vr, t.CHANNEL_FOLLOWERS_COMBINED f
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN mart m ON m.psrk = t.POST_SCRAPER_REFERENCE_KEY AND m.platform = t.POST_PLATFORM
  WHERE NOT t.IS_OBSERVATION_ESTIMATED_ONLY),
ts_d AS (
  SELECT ts.*, LAG(v) OVER (PARTITION BY psrk, platform ORDER BY od) v_prev, LAG(od) OVER (PARTITION BY psrk, platform ORDER BY od) od_prev
  FROM ts WHERE age BETWEEN 0 AND 60 AND v IS NOT NULL),
ts_r AS (SELECT ts_d.*, IFF(od_prev IS NOT NULL, (v - v_prev) / NULLIF(DATEDIFF('day', od_prev, od), 0), NULL) rate FROM ts_d),
feat AS (
  SELECT psrk, platform,
    COUNT_IF(age <= 30) n30, COUNT(*) n60, MIN(age) a_min, MAX(age) a_max,
    MAX(IFF(age <= 1, v, NULL)) v1, MAX(IFF(age <= 2, v, NULL)) v2, MAX(IFF(age <= 3, v, NULL)) v3, MAX(IFF(age <= 5, v, NULL)) v5,
    MAX(IFF(age <= 7, v, NULL)) v7, MAX(IFF(age <= 10, v, NULL)) v10, MAX(IFF(age <= 14, v, NULL)) v14, MAX(IFF(age <= 21, v, NULL)) v21,
    MAX(IFF(age <= 30, v, NULL)) v30, MAX(IFF(age <= 45, v, NULL)) v45, MAX(IFF(age <= 60, v, NULL)) v60,
    MAX(IFF(age <= 1, l, NULL)) l1, MAX(IFF(age <= 3, l, NULL)) l3, MAX(IFF(age <= 7, l, NULL)) l7, MAX(IFF(age <= 14, l, NULL)) l14,
    MAX(IFF(age <= 30, l, NULL)) l30, MAX(IFF(age <= 60, l, NULL)) l60,
    MAX(IFF(age <= 7, c, NULL)) c7, MAX(IFF(age <= 30, c, NULL)) c30, MAX(IFF(age <= 30, s, NULL)) s30,
    MEDIAN(f) followers,
    MAX(IFF(age BETWEEN 2 AND 7, rate, NULL)) r7, MAX(IFF(age BETWEEN 8 AND 14, rate, NULL)) r14,
    MAX(IFF(age BETWEEN 15 AND 30, rate, NULL)) r30, MAX(IFF(age BETWEEN 31 AND 60, rate, NULL)) r60,
    MAX(IFF(age <= 14, age, NULL)) a_max14,
    -- label-only: opt-in / public on the latest read up to day 14 / 30
    MAX_BY(vr / NULLIF(v, 0), IFF(vr IS NOT NULL AND age <= 14, od, NULL)) o14,
    MAX_BY(vr / NULLIF(v, 0), IFF(vr IS NOT NULL AND age <= 30, od, NULL)) o30
  FROM ts_r GROUP BY 1, 2)
SELECT ROW_NUMBER() OVER (ORDER BY m.platform, m.psrk) rn,
  CONCAT_WS('|', m.psrk, LEFT(m.platform, 1), m.pub::STRING,
    (ABS(HASH(m.handle)) % 1000000000)::STRING, (ABS(HASH(m.org)) % 1000000)::STRING,
    f.n30::STRING, f.n60::STRING, f.a_min::STRING, f.a_max::STRING, COALESCE(f.a_max14::STRING, ''),
    COALESCE(f.v1::STRING, ''), COALESCE(f.v2::STRING, ''), COALESCE(f.v3::STRING, ''), COALESCE(f.v5::STRING, ''), COALESCE(f.v7::STRING, ''),
    COALESCE(f.v10::STRING, ''), COALESCE(f.v14::STRING, ''), COALESCE(f.v21::STRING, ''), COALESCE(f.v30::STRING, ''),
    COALESCE(f.v45::STRING, ''), COALESCE(f.v60::STRING, ''),
    COALESCE(f.l1::STRING, ''), COALESCE(f.l3::STRING, ''), COALESCE(f.l7::STRING, ''), COALESCE(f.l14::STRING, ''),
    COALESCE(f.l30::STRING, ''), COALESCE(f.l60::STRING, ''),
    COALESCE(f.c7::STRING, ''), COALESCE(f.c30::STRING, ''), COALESCE(f.s30::STRING, ''), COALESCE(ROUND(f.followers)::STRING, ''),
    COALESCE(ROUND(f.r7)::STRING, ''), COALESCE(ROUND(f.r14)::STRING, ''), COALESCE(ROUND(f.r30)::STRING, ''), COALESCE(ROUND(f.r60)::STRING, ''),
    -- label-only columns
    COALESCE(ROUND(f.o14, 3)::STRING, ''), COALESCE(ROUND(f.o30, 3)::STRING, ''),
    COALESCE(DATEDIFF('day', m.pub, e.first_spend)::STRING, ''), IFF(e.has_tag = 1, '1', ''), IFF(m.paid_date IS NOT NULL, '1', ''),
    CASE WHEN e.psrk IS NOT NULL AND e.first_spend BETWEEN DATEADD('day', -3, m.pub) AND DATEADD('day', 28, m.pub) THEN 'P'
         WHEN m.platform = 'Instagram' AND f.o30 < 0.80 THEN 'P'
         WHEN m.platform = 'Tiktok' AND e.psrk IS NULL AND m.paid_date IS NULL AND COALESCE(oc.n_linked, 0) >= 5 THEN 'N'
         WHEN m.platform = 'Instagram' AND e.psrk IS NULL AND m.paid_date IS NULL AND f.o30 >= 0.90 THEN 'N' ELSE '' END,
    CASE WHEN e.psrk IS NOT NULL AND e.first_spend BETWEEN DATEADD('day', -3, m.pub) AND DATEADD('day', 12, m.pub) THEN 'P'
         WHEN m.platform = 'Instagram' AND f.o14 < 0.80 THEN 'P'
         WHEN m.platform = 'Tiktok' AND e.psrk IS NULL AND m.paid_date IS NULL AND COALESCE(oc.n_linked, 0) >= 5 THEN 'N'
         WHEN m.platform = 'Instagram' AND e.psrk IS NULL AND m.paid_date IS NULL AND f.o14 >= 0.90 THEN 'N' ELSE '' END) s
FROM mart m
JOIN feat f ON f.psrk = m.psrk AND f.platform = m.platform
LEFT JOIN ev e ON e.psrk = m.psrk AND e.platform = m.platform
LEFT JOIN org_cov oc ON oc.platform = m.platform AND oc.org = m.org
WHERE f.a_max >= 7
