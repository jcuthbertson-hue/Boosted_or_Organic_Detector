-- Organic view curve: median share of day-120 public views by post age, unboosted posts only.
-- Output -> reconcile/organic_curve_ig.csv and reconcile/organic_curve_tiktok.csv (age_days, median_share_of_day120_views, posts).
-- Unboosted = label 'NEG' in the output of sql/02 (no ad link, no paid date, and Instagram opt-in = public views /
-- TikTok client runs its paid through our ad accounts). Save the sql/02 output as a table or use RESULT_SCAN, then
-- replace {{DETECTOR_DATASET}} below. Run once per platform ({{PLATFORM}} = 'Instagram' or 'Tiktok').
-- Rebuild every quarter; the curve is used by reconcile/estimate.py.
WITH neg AS (
  SELECT psrk FROM {{DETECTOR_DATASET}} WHERE label = 'NEG' AND platform = {{PLATFORM}}),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age, t.VIEWS_PUBLIC v
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN neg ON neg.psrk = t.POST_SCRAPER_REFERENCE_KEY
  WHERE t.POST_PLATFORM = {{PLATFORM}} AND t.VIEWS_PUBLIC > 0),
ref AS (SELECT psrk, MAX_BY(v, age) v120 FROM ts WHERE age BETWEEN 110 AND 130 GROUP BY 1),
r AS (SELECT ts.psrk, ts.age, LEAST(ts.v / ref.v120, 1.5) ratio FROM ts JOIN ref ON ref.psrk = ts.psrk WHERE ts.age BETWEEN 0 AND 130)
SELECT age age_days, ROUND(MEDIAN(ratio), 4) median_share_of_day120_views, COUNT(*) posts
FROM r GROUP BY 1 ORDER BY 1
