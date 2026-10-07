-- View reconciliation for boosted posts (Tom's question: "if Instagram shows 5M views for a boosted post,
-- what do we see in Nimble public, opt-in private, SocAPI and the paid media tables?").
-- One row per ad-linked post with spend, where the ads finished before our latest observation,
-- so all paid delivery is inside the numbers we compare.
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc,
         ANY_VALUE(CAMPAIGN_ORGANIZATION) org, MIN(POST_PUBLISHED_AT)::DATE pub, BOOLOR_AGG(POST_IS_OPT_IN) opt_in
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram','Tiktok') AND POST_TYPE <> 'STORY' AND POST_PUBLISHED_AT >= '2025-01-01'
  GROUP BY 1,2),
meta_tok AS (SELECT a.AD_KEY, t.value::STRING tok FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
             LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t),
meta_link AS (
  SELECT DISTINCT a.AD_KEY, m.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a JOIN mart m ON m.platform='Instagram' AND m.psrk = a.POST_SCRAPER_REFERENCE_KEY
  UNION SELECT DISTINCT t.AD_KEY, m.psrk FROM meta_tok t JOIN mart m ON m.platform='Instagram' AND m.sc = t.tok),
paid_daily AS (
  SELECT 'Instagram' platform, l.psrk, f.DATE d, f.AD_SPEND spend, f.IMPRESSIONS impr, f.VIDEO_PLAY_ACTIONS plays,
         f.PLAYS_2_SECONDS plays_2s, f.VIDEO_THRUPLAY_WATCHED thruplay, f.PLATFORM placement
  FROM meta_link l JOIN SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = l.AD_KEY
  UNION ALL
  SELECT 'Tiktok', a.TIKTOK_ITEM_ID, f.DATE, f.SPEND, f.IMPRESSIONS, f.VIDEO_PLAY_ACTIONS, f.VIDEO_WATCHED_2_S, f.VIDEO_VIEWS_P_100, 'tiktok'
  FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD a JOIN SNOWFLAKE_EDW.EDW.FACT_TIKTOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = a.AD_KEY
  WHERE a.TIKTOK_ITEM_ID IS NOT NULL),
paid_span AS (
  SELECT platform, psrk, MIN(IFF(spend > 0, d, NULL)) first_spend, MAX(IFF(spend > 0, d, NULL)) last_spend, SUM(spend) spend
  FROM paid_daily GROUP BY 1,2 HAVING SUM(spend) > 0),
ts AS (
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform, t.OBSERVATION_DATE od, t.VIEWS_PUBLIC vp, t.VIEWS_PRIVATE vr,
         t.LIKES_PUBLIC lp, t.CHANNEL_FOLLOWERS_COMBINED fol
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN paid_span p ON p.psrk = t.POST_SCRAPER_REFERENCE_KEY AND p.platform = t.POST_PLATFORM
  WHERE t.VIEWS_PUBLIC IS NOT NULL),
obs AS (
  SELECT p.platform, p.psrk, p.first_spend, p.last_spend, p.spend,
         MAX(IFF(ts.od < p.first_spend, ts.od, NULL)) od_pre, MAX(ts.od) od_last
  FROM paid_span p JOIN ts ON ts.psrk = p.psrk AND ts.platform = p.platform GROUP BY 1,2,3,4,5),
soc AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, MAX(POST_SOCAPI_IG_OBSERVATION_DATE_LATEST) soc_od,
         MAX(API_PLAY_COUNT_LATEST) api_total, MAX(API_IG_PLAY_COUNT_LATEST) api_ig, MAX(API_FB_PLAY_COUNT_LATEST) api_fb
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.SOCAPI_IG_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_PERFORMANCE GROUP BY 1)
SELECT o.platform, o.psrk, m.org, m.opt_in, m.pub, o.first_spend, o.last_spend, o.od_pre, o.od_last, ROUND(o.spend) spend,
       pre.vp v_pub_pre, pre.vr v_priv_pre, lst.vp v_pub_last, lst.vr v_priv_last, lst.fol followers,
       SUM(IFF(pd.d BETWEEN o.first_spend AND o.od_last, pd.impr, 0)) impr,
       SUM(IFF(pd.d BETWEEN o.first_spend AND o.od_last, pd.plays, 0)) plays,
       SUM(IFF(pd.d BETWEEN o.first_spend AND o.od_last AND pd.placement IN ('instagram','tiktok'), pd.plays, 0)) plays_same_app,
       SUM(IFF(pd.d BETWEEN o.first_spend AND o.od_last AND pd.placement = 'facebook', pd.plays, 0)) plays_fb_placement,
       SUM(IFF(pd.d BETWEEN o.first_spend AND o.od_last AND pd.placement IN ('instagram','tiktok'), pd.impr, 0)) impr_same_app,
       SUM(IFF(pd.d BETWEEN o.first_spend AND o.od_last, pd.plays_2s, 0)) plays_2s,
       s.soc_od, s.api_total, s.api_ig, s.api_fb,
       SUM(IFF(pd.d BETWEEN o.first_spend AND s.soc_od AND pd.placement = 'facebook', pd.plays, 0)) plays_fb_placement_to_soc,
       SUM(IFF(pd.d BETWEEN o.first_spend AND s.soc_od AND pd.placement = 'instagram', pd.plays, 0)) plays_ig_placement_to_soc
FROM obs o
JOIN mart m ON m.psrk = o.psrk AND m.platform = o.platform
LEFT JOIN ts pre ON pre.psrk = o.psrk AND pre.platform = o.platform AND pre.od = o.od_pre
LEFT JOIN ts lst ON lst.psrk = o.psrk AND lst.platform = o.platform AND lst.od = o.od_last
LEFT JOIN paid_daily pd ON pd.psrk = o.psrk AND pd.platform = o.platform
LEFT JOIN soc s ON o.platform = 'Instagram' AND s.psrk = o.psrk
WHERE o.od_last > o.last_spend AND o.od_pre IS NOT NULL
GROUP BY 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,s.soc_od,s.api_total,s.api_ig,s.api_fb
