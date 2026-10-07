-- Boosted-or-Organic Detector: training / test table. One row per in-feed IG or TikTok campaign post.
--
-- LEAKAGE RULES (strict)
--   * FEATURES use public organic data only, from the first 30 days after publish (same window for every post).
--     No feature reads PAID_ACTIVITY_DATE, IS_OBSERVATION_BOOSTED / PREBOOST flags, ad names ("Boosted"), spend,
--     or any ad table. No feature reads opt-in (private) or SocAPI values (they are used for labels).
--   * LABELS use paid data and opt-in data only.
--
-- LABEL RULES
--   TikTok POS : a TikTok ad with spend > 0 uses this exact video (TIKTOK_ITEM_ID = post id),
--                and the first spend day is between publish - 3 and publish + 28.
--   TikTok NEG : no ad link, no PAID_ACTIVITY_DATE, and the client has >= 5 ad-linked TikTok posts
--                (VN runs that client's TikTok paid, so a VN boost would show up). "Silver" negative.
--   IG POS     : (a Meta ad links to the post by taxonomy key or shortcode in the ad name, with first spend
--                between publish - 3 and publish + 28)
--                OR (opt-in organic views / public views < 0.80 on the same day, latest day <= day 30).
--   IG NEG     : no ad link, no PAID_ACTIVITY_DATE, and opt-in organic / public >= 0.90 on the same day
--                (latest day <= day 30). Organic is checked against the creator's own data.
--   Everything else -> UNK (not used to train or test).
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform,
         ANY_VALUE(POST_TYPE) post_type, ANY_VALUE(CAMPAIGN_ORGANIZATION) org, ANY_VALUE(CAMPAIGN_KEY) campaign_key,
         ANY_VALUE(CHANNEL_HANDLE) handle,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc,
         MIN(POST_PUBLISHED_AT)::DATE pub, MIN(POST_PAID_ACTIVITY_DATE) paid_date
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram','Tiktok') AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1,2),
-- ---------- paid links (labels only) ----------
meta_tok AS (SELECT a.AD_KEY, t.value::STRING tok FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
             LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t),
meta_link AS (
  SELECT DISTINCT a.AD_KEY, m.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a JOIN mart m ON m.platform='Instagram' AND m.psrk = a.POST_SCRAPER_REFERENCE_KEY
  UNION SELECT DISTINCT t.AD_KEY, m.psrk FROM meta_tok t JOIN mart m ON m.platform='Instagram' AND m.sc = t.tok),
paid AS (
  SELECT 'Instagram' platform, l.psrk, MIN(IFF(f.AD_SPEND > 0, f.DATE, NULL)) first_spend, SUM(f.AD_SPEND) spend
  FROM meta_link l JOIN SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = l.AD_KEY GROUP BY 1,2
  UNION ALL
  SELECT 'Tiktok', a.TIKTOK_ITEM_ID, MIN(IFF(f.SPEND > 0, f.DATE, NULL)), SUM(f.SPEND)
  FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD a JOIN SNOWFLAKE_EDW.EDW.FACT_TIKTOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = a.AD_KEY
  WHERE a.TIKTOK_ITEM_ID IS NOT NULL GROUP BY 1,2),
org_cov AS (
  SELECT m.platform, m.org, COUNT(*) n_linked FROM mart m JOIN paid p ON p.psrk = m.psrk AND p.platform = m.platform AND p.spend > 0 GROUP BY 1,2),
-- ---------- organic time series, first 30 days ----------
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform, t.OBSERVATION_DATE od,
         DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age,
         t.VIEWS_PUBLIC v, t.LIKES_PUBLIC l, t.COMMENTS_PUBLIC c, t.SHARES_PUBLIC s,
         t.VIEWS_PRIVATE vr, t.CHANNEL_FOLLOWERS_COMBINED f
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN mart m ON m.psrk = t.POST_SCRAPER_REFERENCE_KEY AND m.platform = t.POST_PLATFORM
  WHERE NOT t.IS_OBSERVATION_ESTIMATED_ONLY),
ts_d AS (
  SELECT ts.*, LAG(v) OVER (PARTITION BY psrk, platform ORDER BY od) v_prev, LAG(l) OVER (PARTITION BY psrk, platform ORDER BY od) l_prev,
         LAG(od) OVER (PARTITION BY psrk, platform ORDER BY od) od_prev
  FROM ts WHERE age BETWEEN 0 AND 30 AND v IS NOT NULL),
feat AS (
  SELECT psrk, platform,
    COUNT(*) n_obs, MAX(age) age_last,
    MAX(IFF(age <= 1, v, NULL)) v1, MAX(IFF(age <= 3, v, NULL)) v3, MAX(IFF(age <= 7, v, NULL)) v7,
    MAX(IFF(age <= 14, v, NULL)) v14, MAX(v) v30,
    MAX(IFF(age <= 7, l, NULL)) l7, MAX(l) l30, MAX(c) c30, MAX(s) s30,
    MEDIAN(f) followers,
    MAX(IFF(age > 7 AND od_prev IS NOT NULL, (v - v_prev) / NULLIF(DATEDIFF('day', od_prev, od), 0), NULL)) max_rate_d8_30,
    MAX(IFF(age > 7 AND od_prev IS NOT NULL, (l - l_prev) / NULLIF(DATEDIFF('day', od_prev, od), 0), NULL)) max_like_rate_d8_30,
    -- label input only: opt-in organic / public on the latest day (<= day 30) that has both
    MAX_BY(vr / NULLIF(v, 0), IFF(vr IS NOT NULL, od, NULL)) optin_ratio_d30
  FROM ts_d GROUP BY 1,2)
SELECT m.psrk, m.platform, m.post_type, m.org, m.campaign_key, m.handle, m.pub,
       f.n_obs, f.age_last, f.v1, f.v3, f.v7, f.v14, f.v30, f.l7, f.l30, f.c30, f.s30, f.followers,
       f.max_rate_d8_30, f.max_like_rate_d8_30,
       -- label inputs (never used as features)
       f.optin_ratio_d30, m.paid_date, p.first_spend, IFF(p.spend > 0, p.spend, NULL) spend,
       CASE WHEN p.spend > 0 AND p.first_spend BETWEEN DATEADD('day', -3, m.pub) AND DATEADD('day', 28, m.pub) THEN 'ad_link' END pos_source_ad,
       CASE
         WHEN m.platform = 'Tiktok' AND p.spend > 0 AND p.first_spend BETWEEN DATEADD('day', -3, m.pub) AND DATEADD('day', 28, m.pub) THEN 'POS'
         WHEN m.platform = 'Tiktok' AND COALESCE(p.spend, 0) = 0 AND m.paid_date IS NULL AND COALESCE(oc.n_linked, 0) >= 5 THEN 'NEG'
         WHEN m.platform = 'Instagram' AND ((p.spend > 0 AND p.first_spend BETWEEN DATEADD('day', -3, m.pub) AND DATEADD('day', 28, m.pub))
                                            OR f.optin_ratio_d30 < 0.80) THEN 'POS'
         WHEN m.platform = 'Instagram' AND COALESCE(p.spend, 0) = 0 AND m.paid_date IS NULL AND f.optin_ratio_d30 >= 0.90 THEN 'NEG'
         ELSE 'UNK' END label
FROM mart m
JOIN feat f ON f.psrk = m.psrk AND f.platform = m.platform
LEFT JOIN paid p ON p.psrk = m.psrk AND p.platform = m.platform
LEFT JOIN org_cov oc ON oc.platform = m.platform AND oc.org = m.org
WHERE f.n_obs >= 5 AND f.age_last >= 28
