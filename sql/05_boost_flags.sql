-- Boost flags + organic view estimate, one row per in-feed Instagram / TikTok campaign post.
-- Deterministic evidence tiers (strongest first). The ML score ("PREDICTED") is added by detector/score.py.
--   1 CONFIRMED_AD_LINK      an ad with spend > 0 runs this exact post
--                            (TikTok: TIKTOK_ITEM_ID = post id; Meta: ad-name taxonomy key or shortcode)
--   2 MEASURED_OPTIN_GAP     Instagram: opt-in organic views < 80% of public views on the same day
--   3 MEASURED_SOCAPI_GAP    Instagram: SocAPI total plays - IG plays - FB cross-post plays > 25% of IG plays
--   4 LOGGED_PAID_DATE_ONLY  only the manual PAID_ACTIVITY_DATE says paid (may be a dark post)
--   0 NO_PAID_EVIDENCE       none of the above (score it with the ML model)
--
-- Organic views (Instagram), best source first:
--   opt-in private views (organic only)  >  public - 1.14 x paid Instagram-placement plays  >  pre-boost read
--   (SocAPI-only evidence: NULL, not separable)  >  public (no paid evidence)
-- Organic views (TikTok): opt-in includes Spark Ad views, so it does NOT help. We give a range:
--   low  = last public read before the first spend day (pre-boost)
--   high = public - paid plays (paid plays from the TikTok ad table)
-- 1.14 is the median of (public - opt-in) / paid IG-placement plays on 119 opt-in boosted posts (IQR 1.08-1.24).
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc,
         ANY_VALUE(POST_URL) post_url, MIN(POST_PUBLISHED_AT)::DATE pub, MIN(POST_PAID_ACTIVITY_DATE) paid_date,
         MAX(VIEWS_LATEST) views_public_latest, MAX(VIEWS_LATEST_PRIVATE) views_private_latest,
         MAX(VIEWS_PREBOOST) views_preboost, MAX(POST_OBSERVATION_DATE_LATEST) obs_latest
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') AND POST_TYPE <> 'STORY'
  GROUP BY 1, 2),
meta_tok AS (SELECT a.AD_KEY, t.value::STRING tok FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
             LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t),
meta_link AS (
  SELECT DISTINCT a.AD_KEY, m.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a JOIN mart m ON m.platform = 'Instagram' AND m.psrk = a.POST_SCRAPER_REFERENCE_KEY
  UNION SELECT DISTINCT t.AD_KEY, m.psrk FROM meta_tok t JOIN mart m ON m.platform = 'Instagram' AND m.sc = t.tok),
paid AS (
  SELECT 'Instagram' platform, l.psrk, MIN(IFF(f.AD_SPEND > 0, f.DATE, NULL)) first_spend, SUM(f.AD_SPEND) spend,
         SUM(IFF(f.PLATFORM = 'instagram', f.VIDEO_PLAY_ACTIONS, 0)) paid_plays_same_app,
         SUM(IFF(f.PLATFORM = 'facebook', f.VIDEO_PLAY_ACTIONS, 0)) paid_plays_facebook,
         SUM(f.IMPRESSIONS) paid_impressions
  FROM meta_link l JOIN SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = l.AD_KEY
  JOIN mart m ON m.psrk = l.psrk AND m.platform = 'Instagram'
  WHERE f.DATE <= COALESCE(m.obs_latest, CURRENT_DATE)
  GROUP BY 1, 2
  UNION ALL
  SELECT 'Tiktok', a.TIKTOK_ITEM_ID, MIN(IFF(f.SPEND > 0, f.DATE, NULL)), SUM(f.SPEND),
         SUM(f.VIDEO_PLAY_ACTIONS), 0, SUM(f.IMPRESSIONS)
  FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD a JOIN SNOWFLAKE_EDW.EDW.FACT_TIKTOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = a.AD_KEY
  JOIN mart m ON m.psrk = a.TIKTOK_ITEM_ID AND m.platform = 'Tiktok'
  WHERE f.DATE <= COALESCE(m.obs_latest, CURRENT_DATE)
  GROUP BY 1, 2),
optin AS (   -- latest day with both public and opt-in views
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform,
         MAX_BY(VIEWS_PRIVATE / NULLIF(VIEWS_PUBLIC, 0), OBSERVATION_DATE) optin_ratio
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES
  WHERE VIEWS_PRIVATE IS NOT NULL AND VIEWS_PUBLIC IS NOT NULL AND POST_PLATFORM = 'Instagram'
  GROUP BY 1, 2),
pre AS (     -- last public read before the first spend day
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform, MAX_BY(t.VIEWS_PUBLIC, t.OBSERVATION_DATE) views_before_first_spend
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN paid p ON p.psrk = t.POST_SCRAPER_REFERENCE_KEY AND p.platform = t.POST_PLATFORM
  WHERE t.VIEWS_PUBLIC IS NOT NULL AND t.OBSERVATION_DATE < p.first_spend
  GROUP BY 1, 2),
soc AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, MAX(API_PLAY_COUNT_LATEST) api_total, MAX(API_IG_PLAY_COUNT_LATEST) api_ig, MAX(API_FB_PLAY_COUNT_LATEST) api_fb
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.SOCAPI_IG_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_PERFORMANCE GROUP BY 1)
SELECT m.psrk, m.platform, m.post_url, m.pub, m.obs_latest,
  CASE
    WHEN p.spend > 0 THEN 'CONFIRMED_AD_LINK'
    WHEN m.platform = 'Instagram' AND o.optin_ratio < 0.80 THEN 'MEASURED_OPTIN_GAP'
    WHEN m.platform = 'Instagram' AND (s.api_total - s.api_ig - COALESCE(s.api_fb, 0)) > 0.25 * NULLIF(s.api_ig, 0) THEN 'MEASURED_SOCAPI_GAP'
    WHEN m.paid_date IS NOT NULL THEN 'LOGGED_PAID_DATE_ONLY'
    ELSE 'NO_PAID_EVIDENCE' END boost_evidence,
  p.first_spend, m.paid_date, ROUND(p.spend, 2) spend, p.paid_plays_same_app, p.paid_plays_facebook, p.paid_impressions,
  o.optin_ratio, s.api_total, s.api_ig, s.api_fb,
  m.views_public_latest, m.views_private_latest, pr.views_before_first_spend, m.views_preboost,
  CASE
    WHEN m.platform = 'Instagram' AND m.views_private_latest IS NOT NULL THEN m.views_private_latest
    WHEN m.platform = 'Instagram' AND p.spend > 0 THEN GREATEST(m.views_public_latest - 1.14 * p.paid_plays_same_app, COALESCE(pr.views_before_first_spend, 0))
    WHEN m.platform = 'Instagram' AND m.paid_date IS NOT NULL THEN m.views_preboost
    WHEN m.platform = 'Instagram' AND (s.api_total - s.api_ig - COALESCE(s.api_fb, 0)) > 0.25 * NULLIF(s.api_ig, 0) THEN NULL
    WHEN m.platform = 'Tiktok' AND p.spend > 0 THEN pr.views_before_first_spend
    WHEN m.platform = 'Tiktok' AND m.paid_date IS NOT NULL THEN m.views_preboost
    ELSE m.views_public_latest END organic_views_est,
  CASE WHEN m.platform = 'Tiktok' AND p.spend > 0 THEN GREATEST(m.views_public_latest - p.paid_plays_same_app, COALESCE(pr.views_before_first_spend, 0)) END organic_views_est_high,
  CASE
    WHEN m.platform = 'Instagram' AND m.views_private_latest IS NOT NULL THEN 'opt-in private (organic only)'
    WHEN m.platform = 'Instagram' AND p.spend > 0 THEN 'public minus 1.14 x paid IG-placement plays'
    WHEN m.platform = 'Tiktok' AND p.spend > 0 THEN 'pre-boost public read (low); public minus paid plays (high)'
    WHEN m.paid_date IS NOT NULL THEN 'pre-boost read (manual paid date)'
    WHEN m.platform = 'Instagram' AND (s.api_total - s.api_ig - COALESCE(s.api_fb, 0)) > 0.25 * NULLIF(s.api_ig, 0)
      THEN 'not separable: SocAPI shows paid plays, no opt-in / ad link / paid date'
    ELSE 'public (no paid evidence; check ML score)' END organic_views_source
FROM mart m
LEFT JOIN paid p ON p.psrk = m.psrk AND p.platform = m.platform
LEFT JOIN optin o ON o.psrk = m.psrk AND o.platform = m.platform
LEFT JOIN pre pr ON pr.psrk = m.psrk AND pr.platform = m.platform
LEFT JOIN soc s ON m.platform = 'Instagram' AND s.psrk = m.psrk
