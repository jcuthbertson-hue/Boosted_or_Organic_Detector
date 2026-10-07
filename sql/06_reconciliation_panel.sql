-- Subtraction test, post x observation day:  platform total - paid metric = organic ?
-- Ground truth for organic = opt-in (private) Instagram views on the same day (organic only, Instagram only).
--
-- Sample tiers (start with posts in the paid team's post-ID tag; go back in time only if the sample is too small):
--   post_id_tag      : post id in the paid team's post-ID field PAID_MEDIA_UNIFIED."ext_p3_organic_post_id" (from Sep 2026)
--   unified_name     : post id in an ad name inside the unified paid table
--   edw_taxonomy     : post id in the ad-name taxonomy of the full Meta ad tables (older ads, same slot)
-- Paid metrics come from SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE, the table the unified table is built
-- from (its sums match the unified table), so every tier uses identical metric definitions.
-- Organic side (same day): BIRA observation time series (Nimble public + opt-in private), plus SocAPI plays on
-- SocAPI pull days (platform total incl. Facebook).
-- Paid is cumulative by placement through the observation day (_d) and through the day before (_p).
WITH bira AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM = 'Instagram' AND POST_TYPE <> 'STORY'
  GROUP BY 1),
tagged AS (SELECT DISTINCT b.psrk FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED pm JOIN bira b ON b.sc = pm."ext_p3_organic_post_id"),
unified_name AS (
  SELECT DISTINCT b.psrk FROM (SELECT DISTINCT "ad_name" FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED WHERE "platform" = 'meta') pm,
         LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(pm."ad_name", '[A-Za-z0-9_-]{11}')) t JOIN bira b ON b.sc = t.value::STRING),
links AS (
  SELECT DISTINCT a.AD_KEY, b.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a JOIN bira b ON b.psrk = a.POST_SCRAPER_REFERENCE_KEY
  UNION
  SELECT DISTINCT a.AD_KEY, b.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
         LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t JOIN bira b ON b.sc = t.value::STRING),
paid_loaded AS (SELECT MAX(DATE) max_d FROM SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE),
paid_daily AS (
  SELECT l.psrk, f.DATE d,
         CASE WHEN f.PLATFORM = 'instagram' THEN 'ig' WHEN f.PLATFORM = 'facebook' THEN 'fb' ELSE 'ot' END g,
         SUM(f.AD_SPEND) spend, SUM(f.IMPRESSIONS) impr, SUM(f.REACH) reach, SUM(f.VIDEO_PLAY_ACTIONS) starts,
         SUM(f.PLAYS_2_SECONDS) v2s, SUM(f.VIDEO_VIEWS) vviews, SUM(f.VIDEO_THRUPLAY_WATCHED) thru,
         SUM(f.VIDEO_P_25_WATCHED) p25, SUM(f.VIDEO_P_100_WATCHED) p100
  FROM links l JOIN SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = l.AD_KEY GROUP BY 1, 2, 3),
post_paid AS (
  SELECT psrk, MIN(IFF(spend > 0, d, NULL)) first_spend, MAX(IFF(spend > 0, d, NULL)) last_spend, SUM(spend) spend_total
  FROM paid_daily GROUP BY 1 HAVING SUM(spend) > 0),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.OBSERVATION_DATE od, t.VIEWS_PUBLIC vp, t.VIEWS_PRIVATE vr
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN post_paid p ON p.psrk = t.POST_SCRAPER_REFERENCE_KEY
  WHERE t.POST_PLATFORM = 'Instagram' AND t.VIEWS_PUBLIC > 0 AND t.VIEWS_PRIVATE > 0
    AND t.OBSERVATION_DATE BETWEEN DATEADD('day', -3, p.first_spend) AND DATEADD('day', 45, p.last_spend)
    AND t.OBSERVATION_DATE <= (SELECT max_d FROM paid_loaded)),
soc AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, OBSERVATION_DATE od, API_PLAY_COUNT api_total, API_IG_PLAY_COUNT api_ig, API_FB_PLAY_COUNT api_fb
  FROM AI_ANALYSIS.AI_ANALYSIS_NEW.FACT_ORGANIC__POST_PERFORMANCE_MEDIA_INFO
  WHERE MEDIA_INFO_STATUS = 'SUCCESS' AND API_PLAY_COUNT IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (PARTITION BY POST_SCRAPER_REFERENCE_KEY, OBSERVATION_DATE ORDER BY MEDIA_INFO_FETCHED_AT DESC) = 1)
SELECT ts.psrk,
       CASE WHEN ts.psrk IN (SELECT psrk FROM tagged) THEN 'post_id_tag'
            WHEN ts.psrk IN (SELECT psrk FROM unified_name) THEN 'unified_name' ELSE 'edw_taxonomy' END tier,
       p.first_spend, p.last_spend, ROUND(p.spend_total) spend_total,
       ts.od, DATEDIFF('day', p.first_spend, ts.od) dsf, DATEDIFF('day', p.last_spend, ts.od) dsl,
       ts.vp, ts.vr, s.api_total, s.api_ig, s.api_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.impr, 0)) impr_ig,   SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.impr, 0)) impr_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.reach, 0)) reach_ig, SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.reach, 0)) reach_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.starts, 0)) starts_ig, SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.starts, 0)) starts_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ot', pd.starts, 0)) starts_ot,
       SUM(IFF(pd.d < ts.od AND pd.g = 'ig', pd.starts, 0)) starts_ig_p, SUM(IFF(pd.d < ts.od AND pd.g = 'fb', pd.starts, 0)) starts_fb_p,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.v2s, 0)) v2s_ig,       SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.v2s, 0)) v2s_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.vviews, 0)) v3s_ig,    SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.vviews, 0)) v3s_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.thru, 0)) thru_ig,     SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.thru, 0)) thru_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.p25, 0)) p25_ig,       SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.p25, 0)) p25_fb,
       SUM(IFF(pd.d <= ts.od AND pd.g = 'ig', pd.p100, 0)) p100_ig,     SUM(IFF(pd.d <= ts.od AND pd.g = 'fb', pd.p100, 0)) p100_fb
FROM ts
JOIN post_paid p ON p.psrk = ts.psrk
LEFT JOIN soc s ON s.psrk = ts.psrk AND s.od = ts.od
LEFT JOIN paid_daily pd ON pd.psrk = ts.psrk
GROUP BY ts.psrk, p.first_spend, p.last_spend, p.spend_total, ts.od, ts.vp, ts.vr, s.api_total, s.api_ig, s.api_fb
