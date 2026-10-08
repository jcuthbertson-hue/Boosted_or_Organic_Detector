-- Model v2.2 input (plan amendment 5): follower count known at each horizon, and the last read up to day 30.
-- Same posts as sql/11; join on (psrk, platform). One pipe-separated string per post: psrk|platform|f14|f30|f60|a_max30.
--   f_H     = median CHANNEL_FOLLOWERS_COMBINED over reads on days 0..H (sql/11 `followers` used days 0..60 for every H)
--   a_max30 = age of the last read on days 0..30 (day-30 eligibility; sql/11 a_max looks up to day 60)
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform, MIN(POST_PUBLISHED_AT)::DATE pub
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1, 2),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform,
         DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age, t.VIEWS_PUBLIC v, t.CHANNEL_FOLLOWERS_COMBINED f
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN mart m ON m.psrk = t.POST_SCRAPER_REFERENCE_KEY AND m.platform = t.POST_PLATFORM
  WHERE NOT t.IS_OBSERVATION_ESTIMATED_ONLY),
agg AS (   -- same read set as sql/11 (ages 0..60, public views present)
  SELECT psrk, platform, MAX(age) a_max,
         MEDIAN(IFF(age <= 14, f, NULL)) f14, MEDIAN(IFF(age <= 30, f, NULL)) f30, MEDIAN(f) f60,
         MAX(IFF(age <= 30, age, NULL)) a_max30
  FROM ts WHERE age BETWEEN 0 AND 60 AND v IS NOT NULL GROUP BY 1, 2)
SELECT ROW_NUMBER() OVER (ORDER BY m.platform, m.psrk) rn,
  CONCAT_WS('|', m.psrk, LEFT(m.platform, 1), COALESCE(ROUND(a.f14)::STRING, ''), COALESCE(ROUND(a.f30)::STRING, ''),
            COALESCE(ROUND(a.f60)::STRING, ''), COALESCE(a.a_max30::STRING, '')) s
FROM mart m JOIN agg a ON a.psrk = m.psrk AND a.platform = m.platform
WHERE a.a_max >= 7
