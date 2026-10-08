-- Daily public views per post, days 0-90, for the view-jump boost start (reconcile/boost_start.py). One pipe-separated
-- string per post (column s) so it pages out of the Snowflake MCP, same pattern as sql/11 and sql/12.
--
-- s = psrk | I or T | first read date | age0:views0;dage:dviews;...
--   only reads where public views changed (a carried-forward value is not a new read: the July 2026 scraper gap kept
--   views flat for about two weeks, then caught up in one step); every read to day 30, then the first read in each
--   3-day bucket (keeps the output small; the boost start after day 30 is then known to within 3 days).
-- The pipeline uses it only for boosted posts with no known start date (model flags, evidence with no spend/paid date).
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1, 2),
r AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform, t.OBSERVATION_DATE od,
         DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age, MAX(t.VIEWS_PUBLIC) v
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN mart m ON m.psrk = t.POST_SCRAPER_REFERENCE_KEY AND m.platform = t.POST_PLATFORM
  WHERE NOT t.IS_OBSERVATION_ESTIMATED_ONLY AND t.VIEWS_PUBLIC > 0
  GROUP BY 1, 2, 3, 4 HAVING age BETWEEN 0 AND 90),
changed AS (SELECT * FROM r QUALIFY v <> COALESCE(LAG(v) OVER (PARTITION BY psrk, platform ORDER BY od), -1)),
thin AS (SELECT * FROM changed QUALIFY age <= 30 OR ROW_NUMBER() OVER (PARTITION BY psrk, platform, FLOOR(age / 3) ORDER BY od) = 1),
d AS (
  SELECT psrk, platform, od, age, v,
         age - LAG(age) OVER (PARTITION BY psrk, platform ORDER BY od) da, v - LAG(v) OVER (PARTITION BY psrk, platform ORDER BY od) dv
  FROM thin),
p AS (
  SELECT psrk, platform, MIN(od) od0,
         LISTAGG(IFF(da IS NULL, age || ':' || v, da || ':' || dv), ';') WITHIN GROUP (ORDER BY od) s
  FROM d GROUP BY 1, 2)
SELECT ROW_NUMBER() OVER (ORDER BY platform, psrk) rn, psrk || '|' || LEFT(platform, 1) || '|' || od0 || '|' || s s
FROM p
