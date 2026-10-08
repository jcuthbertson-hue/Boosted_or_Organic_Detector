-- Model v2 extra input: what happens DURING the biggest view jump, plus day-60 values. Same posts as sql/11.
-- Join to sql/11 on (psrk, platform). One pipe-separated string per post (column order = detector/features_v2.py
-- EXTRA_COLS) so it pages out of the Snowflake MCP.
--
-- WHY: paid views come with very few likes. On the read interval with the biggest daily view gain, likes per new view
-- drop for a boost but stay normal for an organic spike. Train-CV error analysis (Instagram, day 30) showed boosts with
-- 40-80% paid views and boosts that start after day 14 are the ones the model misses.
--
-- FEATURES (public organic data only), for each window W in (14, 30, 60), reads with age 2..W:
--   ja_W = age of the read that ends the interval with the biggest daily view gain
--   jl_W = likes per new view on that interval (x 100000, integer)
--   ml_W = lowest likes per new view on any interval that adds >= 5% of the day-W views (x 100000, integer)
--   v25, v28, l21, l45 = cumulative views / likes by that age (last real read on or before that day)
-- LABEL-ONLY: o60 = opt-in / public on the latest read up to day 60 (never a feature).
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform, MIN(POST_PUBLISHED_AT)::DATE pub
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1, 2),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform, t.OBSERVATION_DATE od,
         DATEDIFF('day', t.PUBLISHED_DATETIME::DATE, t.OBSERVATION_DATE) age,
         t.VIEWS_PUBLIC v, t.LIKES_PUBLIC l, t.VIEWS_PRIVATE vr
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN mart m ON m.psrk = t.POST_SCRAPER_REFERENCE_KEY AND m.platform = t.POST_PLATFORM
  WHERE NOT t.IS_OBSERVATION_ESTIMATED_ONLY),
ts_d AS (
  SELECT ts.*,
         LAG(v) OVER (PARTITION BY psrk, platform ORDER BY od) v_prev, LAG(l) OVER (PARTITION BY psrk, platform ORDER BY od) l_prev,
         LAG(od) OVER (PARTITION BY psrk, platform ORDER BY od) od_prev,
         MAX(IFF(age <= 14, v, NULL)) OVER (PARTITION BY psrk, platform) v14w,
         MAX(IFF(age <= 30, v, NULL)) OVER (PARTITION BY psrk, platform) v30w,
         MAX(IFF(age <= 60, v, NULL)) OVER (PARTITION BY psrk, platform) v60w
  FROM ts WHERE age BETWEEN 0 AND 60 AND v IS NOT NULL),
iv AS (
  SELECT ts_d.*, v - v_prev dv,
         (v - v_prev) / NULLIF(DATEDIFF('day', od_prev, od), 0) rate,
         IFF(v - v_prev > 0 AND l IS NOT NULL AND l_prev IS NOT NULL, GREATEST(l - l_prev, 0) / (v - v_prev), NULL) ilpv
  FROM ts_d WHERE od_prev IS NOT NULL),
jf AS (
  SELECT psrk, platform,
    MAX_BY(age, IFF(age BETWEEN 2 AND 14, rate, NULL)) ja14, MAX_BY(ilpv, IFF(age BETWEEN 2 AND 14, rate, NULL)) jl14,
    MAX_BY(age, IFF(age BETWEEN 2 AND 30, rate, NULL)) ja30, MAX_BY(ilpv, IFF(age BETWEEN 2 AND 30, rate, NULL)) jl30,
    MAX_BY(age, IFF(age BETWEEN 2 AND 60, rate, NULL)) ja60, MAX_BY(ilpv, IFF(age BETWEEN 2 AND 60, rate, NULL)) jl60,
    MIN(IFF(age BETWEEN 2 AND 14 AND dv >= 0.05 * v14w, ilpv, NULL)) ml14,
    MIN(IFF(age BETWEEN 2 AND 30 AND dv >= 0.05 * v30w, ilpv, NULL)) ml30,
    MIN(IFF(age BETWEEN 2 AND 60 AND dv >= 0.05 * v60w, ilpv, NULL)) ml60
  FROM iv GROUP BY 1, 2),
lv AS (
  SELECT psrk, platform, MAX(age) a_max,
    MAX(IFF(age <= 25, v, NULL)) v25, MAX(IFF(age <= 28, v, NULL)) v28,
    MAX(IFF(age <= 21, l, NULL)) l21, MAX(IFF(age <= 45, l, NULL)) l45,
    MAX_BY(vr / NULLIF(v, 0), IFF(vr IS NOT NULL AND age <= 60, od, NULL)) o60          -- label-only
  FROM ts_d GROUP BY 1, 2)
SELECT ROW_NUMBER() OVER (ORDER BY m.platform, m.psrk) rn,
  CONCAT_WS('|', m.psrk, LEFT(m.platform, 1),
    COALESCE(ROUND(x.o60, 3)::STRING, ''),
    COALESCE(x.v25::STRING, ''), COALESCE(x.v28::STRING, ''), COALESCE(x.l21::STRING, ''), COALESCE(x.l45::STRING, ''),
    COALESCE(j.ja14::STRING, ''), COALESCE(ROUND(j.jl14 * 100000)::STRING, ''), COALESCE(ROUND(j.ml14 * 100000)::STRING, ''),
    COALESCE(j.ja30::STRING, ''), COALESCE(ROUND(j.jl30 * 100000)::STRING, ''), COALESCE(ROUND(j.ml30 * 100000)::STRING, ''),
    COALESCE(j.ja60::STRING, ''), COALESCE(ROUND(j.jl60 * 100000)::STRING, ''), COALESCE(ROUND(j.ml60 * 100000)::STRING, '')) s
FROM mart m
JOIN lv x ON x.psrk = m.psrk AND x.platform = m.platform
LEFT JOIN jf j ON j.psrk = m.psrk AND j.platform = m.platform
WHERE x.a_max >= 7
