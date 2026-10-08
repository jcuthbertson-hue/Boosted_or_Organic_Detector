-- Opt-in staleness check, one row per Instagram campaign post (same posts as sql/11).
-- WHY: the Instagram "opt-in gap" label (opt-in / public < 0.80) assumes the opt-in count keeps updating. If the
-- opt-in (private) count freezes while public views keep growing, the ratio falls with no paid views at all.
-- Train-only audit (model v2 plan, amendment 4): 34% of missed opt-in-only "boosts" had a flat opt-in series
-- (<= 2 distinct values over >= 4 reads in days 0-30), vs 2.5% of caught boosts and 2.5% of organic posts.
--
-- For each window W (14, 30, 60 days, and all reads): reads with both values, then
--   fz_W = days the opt-in value has been unchanged at the end of the window (last read age - age it first reached
--          that value), pg_W = public growth over those days (public at last read / public when opt-in froze - 1).
-- A frozen opt-in tail with public growth means the ratio at the end of the window is not a paid measure.
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, MIN(POST_PUBLISHED_AT)::DATE pub
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM = 'Instagram' AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age,
         t.VIEWS_PUBLIC v, t.VIEWS_PRIVATE vr
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN mart m ON m.psrk = t.POST_SCRAPER_REFERENCE_KEY
  WHERE t.POST_PLATFORM = 'Instagram' AND NOT t.IS_OBSERVATION_ESTIMATED_ONLY
    AND t.VIEWS_PRIVATE IS NOT NULL AND t.VIEWS_PUBLIC > 0 AND t.OBSERVATION_DATE >= t.PUBLISHED_DATETIME::DATE),
w AS (SELECT column1 win FROM VALUES (14), (30), (60), (100000)),
tw AS (SELECT w.win, ts.* FROM ts JOIN w ON ts.age <= w.win),
last_ AS (SELECT win, psrk, MAX_BY(vr, age) vr_last, MAX(age) a_last, MAX_BY(v, age) v_last, COUNT(*) n FROM tw GROUP BY 1, 2),
frz AS (
  SELECT l.win, l.psrk, l.a_last, l.n, l.v_last, MIN(t.age) a_freeze, MIN_BY(t.v, t.age) v_freeze
  FROM last_ l JOIN tw t ON t.win = l.win AND t.psrk = l.psrk AND t.vr = l.vr_last
  GROUP BY 1, 2, 3, 4, 5),
s AS (
  SELECT psrk,
    MAX(IFF(win = 14, a_last - a_freeze, NULL)) fz14, MAX(IFF(win = 14, v_last / NULLIF(v_freeze, 0) - 1, NULL)) pg14,
    MAX(IFF(win = 30, a_last - a_freeze, NULL)) fz30, MAX(IFF(win = 30, v_last / NULLIF(v_freeze, 0) - 1, NULL)) pg30,
    MAX(IFF(win = 60, a_last - a_freeze, NULL)) fz60, MAX(IFF(win = 60, v_last / NULLIF(v_freeze, 0) - 1, NULL)) pg60,
    MAX(IFF(win = 100000, a_last - a_freeze, NULL)) fza, MAX(IFF(win = 100000, v_last / NULLIF(v_freeze, 0) - 1, NULL)) pga,
    MAX(IFF(win = 30, n, NULL)) n30
  FROM frz GROUP BY 1)
SELECT ROW_NUMBER() OVER (ORDER BY psrk) rn,
  CONCAT_WS('|', psrk, COALESCE(fz14::STRING, ''), COALESCE(ROUND(pg14, 3)::STRING, ''), COALESCE(fz30::STRING, ''),
            COALESCE(ROUND(pg30, 3)::STRING, ''), COALESCE(fz60::STRING, ''), COALESCE(ROUND(pg60, 3)::STRING, ''),
            COALESCE(fza::STRING, ''), COALESCE(ROUND(pga, 3)::STRING, ''), COALESCE(n30::STRING, '')) s
FROM s
