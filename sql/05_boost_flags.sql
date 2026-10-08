-- Paid evidence + organic-estimate inputs, one row per in-feed Instagram / TikTok campaign post.
-- pipeline/run_daily.py runs this, adds the model score (sql/02 features) and the organic estimate
-- (reconcile/estimate.py), then MERGEs the result into the paid classification table (sql/09).
--
-- Evidence tiers (strongest first):
--   CONFIRMED_PAID_TAG     the paid team's post-ID tag names this post: PAID_MEDIA_UNIFIED."ext_p3_organic_post_id" (from Sep 2026).
--                          The paid team enters it per ad, so it is the most direct proof that a post is paid.
--   CONFIRMED_AD_LINK      an ad with spend > 0 runs this post (TikTok: TIKTOK_ITEM_ID = video id;
--                          Meta: taxonomy key or 11-character shortcode in the ad name). Check vs the post-ID tag:
--                          Meta 21 of 21 tagged posts found; TikTok 6 of 9 (0 of the 3 tracked campaign posts).
--   MEASURED_OPTIN_GAP     Instagram: opt-in organic views < 80% of public views on the same day. If the opt-in count
--                          stopped changing >= 7 days ago while public grew >= 5% (stale opt-in), the ratio from the
--                          last read where opt-in still changed is used (plan amendment 4); a gap that appears only
--                          after opt-in froze is not evidence.
--   MEASURED_SOCAPI_GAP    Instagram: SocAPI total - IG plays - FB cross-post plays > 25% of IG plays
--   LOGGED_PAID_DATE_ONLY  only the manual PAID_ACTIVITY_DATE says paid (may be a dark post)
--   NO_PAID_EVIDENCE       none of the above (the model scores it)
--
-- Organic-estimate inputs: latest public and opt-in views, and the last public read before the boost start
-- (first spend day from the post-ID tag or an ad link, or the manual paid date, whichever is first) with its age. The estimate itself needs the organic curve, so it is computed in Python.
WITH mart AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc,
         ANY_VALUE(POST_URL) post_url, ANY_VALUE(POST_TYPE) post_type, MIN(POST_PUBLISHED_AT)::DATE pub,
         MIN(POST_PAID_ACTIVITY_DATE) paid_date,
         MAX(VIEWS_LATEST) views_public_latest, MAX(VIEWS_LATEST_PRIVATE) views_private_latest,
         MAX(POST_OBSERVATION_DATE_LATEST) obs_latest
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') AND POST_TYPE <> 'STORY'
  GROUP BY 1, 2),
tag AS (     -- post-ID tag (unified paid table; metrics are all placements together)
  SELECT platform, psrk, MIN(d) first_spend, SUM(spend) spend, SUM(impr) impr, SUM(starts) starts FROM (
    SELECT m.platform, m.psrk, u."date" d, u."spend" spend, u."impressions" impr, u."video_starts" starts
    FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED u JOIN mart m ON m.platform = 'Instagram' AND m.sc = u."ext_p3_organic_post_id"
    WHERE u."platform" = 'meta' AND u."spend" > 0 AND u."date" <= COALESCE(m.obs_latest, CURRENT_DATE)
    UNION ALL
    SELECT m.platform, m.psrk, u."date", u."spend", u."impressions", u."video_starts"
    FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED u JOIN mart m ON m.platform = 'Tiktok' AND m.psrk = u."ext_p3_organic_post_id"
    WHERE u."platform" = 'tiktok' AND u."spend" > 0 AND u."date" <= COALESCE(m.obs_latest, CURRENT_DATE))
  GROUP BY 1, 2),
meta_tok AS (SELECT a.AD_KEY, t.value::STRING tok FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
             LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t),
meta_link AS (
  SELECT DISTINCT a.AD_KEY, m.psrk FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a JOIN mart m ON m.platform = 'Instagram' AND m.psrk = a.POST_SCRAPER_REFERENCE_KEY
  UNION SELECT DISTINCT t.AD_KEY, m.psrk FROM meta_tok t JOIN mart m ON m.platform = 'Instagram' AND m.sc = t.tok),
paid AS (    -- EDW ad tables (placement split on Meta)
  SELECT 'Instagram' platform, l.psrk, MIN(IFF(f.AD_SPEND > 0, f.DATE, NULL)) first_spend, SUM(f.AD_SPEND) spend,
         SUM(IFF(f.PLATFORM = 'instagram', f.IMPRESSIONS, 0)) impr_ig, SUM(IFF(f.PLATFORM = 'facebook', f.IMPRESSIONS, 0)) impr_fb,
         SUM(IFF(f.PLATFORM = 'instagram', f.VIDEO_PLAY_ACTIONS, 0)) plays_ig, SUM(IFF(f.PLATFORM = 'facebook', f.VIDEO_PLAY_ACTIONS, 0)) plays_fb
  FROM meta_link l JOIN SNOWFLAKE_EDW.EDW.FACT_FACEBOOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = l.AD_KEY
  JOIN mart m ON m.psrk = l.psrk AND m.platform = 'Instagram'
  WHERE f.DATE <= COALESCE(m.obs_latest, CURRENT_DATE)
  GROUP BY 1, 2
  UNION ALL
  SELECT 'Tiktok', a.TIKTOK_ITEM_ID, MIN(IFF(f.SPEND > 0, f.DATE, NULL)), SUM(f.SPEND),
         SUM(f.IMPRESSIONS), 0, SUM(f.VIDEO_PLAY_ACTIONS), 0
  FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD a JOIN SNOWFLAKE_EDW.EDW.FACT_TIKTOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = a.AD_KEY
  JOIN mart m ON m.psrk = a.TIKTOK_ITEM_ID AND m.platform = 'Tiktok'
  WHERE f.DATE <= COALESCE(m.obs_latest, CURRENT_DATE)
  GROUP BY 1, 2),
first_spend AS (   -- first ad spend day (post-ID tag or ad link)
  SELECT platform, psrk, MIN(first_spend) first_spend FROM (
    SELECT platform, psrk, first_spend FROM tag UNION ALL SELECT platform, psrk, first_spend FROM paid WHERE spend > 0)
  GROUP BY 1, 2),
boost_start AS (   -- earliest sign of paid: first spend day or the manual paid date, whichever is first
  SELECT platform, psrk, MIN(d) boost_start FROM (
    SELECT platform, psrk, first_spend d FROM first_spend UNION ALL SELECT platform, psrk, paid_date FROM mart WHERE paid_date IS NOT NULL)
  GROUP BY 1, 2),
optin_ts AS (   -- reads with both public and opt-in views
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform, OBSERVATION_DATE od,
         DATEDIFF('day', PUBLISHED_DATETIME::DATE, OBSERVATION_DATE) age, VIEWS_PUBLIC v, VIEWS_PRIVATE vr
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES
  WHERE VIEWS_PRIVATE IS NOT NULL AND VIEWS_PUBLIC > 0 AND POST_PLATFORM = 'Instagram' AND NOT IS_OBSERVATION_ESTIMATED_ONLY),
optin_last AS (SELECT psrk, platform, MAX_BY(vr, od) vr_last, MAX_BY(v, od) v_last, MAX(age) a_last FROM optin_ts GROUP BY 1, 2),
optin AS (   -- latest opt-in ratio, and the ratio when the opt-in count last changed (plan amendment 4: stale opt-in)
  SELECT l.psrk, l.platform, l.vr_last / NULLIF(l.v_last, 0) optin_ratio_latest,
         MIN(t.age) optin_freeze_age, l.a_last - MIN(t.age) optin_frozen_days,
         l.v_last / NULLIF(MIN_BY(t.v, t.age), 0) - 1 public_growth_since_freeze,
         l.vr_last / NULLIF(MIN_BY(t.v, t.age), 0) optin_ratio_live,
         (l.a_last - MIN(t.age) >= 7 AND l.v_last / NULLIF(MIN_BY(t.v, t.age), 0) - 1 >= 0.05) optin_stale,
         IFF(l.a_last - MIN(t.age) >= 7 AND l.v_last / NULLIF(MIN_BY(t.v, t.age), 0) - 1 >= 0.05,
             l.vr_last / NULLIF(MIN_BY(t.v, t.age), 0), l.vr_last / NULLIF(l.v_last, 0)) optin_ratio
  FROM optin_last l JOIN optin_ts t ON t.psrk = l.psrk AND t.platform = l.platform AND t.vr = l.vr_last
  GROUP BY l.psrk, l.platform, l.vr_last, l.v_last, l.a_last),
pre AS (     -- last real public read before the boost start
  SELECT t.POST_SCRAPER_REFERENCE_KEY psrk, t.POST_PLATFORM platform,
         MAX_BY(t.VIEWS_PUBLIC, t.OBSERVATION_DATE) views_before_first_spend, MAX(t.OBSERVATION_DATE) preboost_read_date
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_TIMESERIES t
  JOIN boost_start s ON s.psrk = t.POST_SCRAPER_REFERENCE_KEY AND s.platform = t.POST_PLATFORM
  WHERE t.VIEWS_PUBLIC > 0 AND NOT t.IS_OBSERVATION_ESTIMATED_ONLY AND t.OBSERVATION_DATE < s.boost_start
  GROUP BY 1, 2),
soc AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, MAX(API_PLAY_COUNT_LATEST) api_total, MAX(API_IG_PLAY_COUNT_LATEST) api_ig, MAX(API_FB_PLAY_COUNT_LATEST) api_fb
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.SOCAPI_IG_FACT_ORGANIC__CAMPAIGN_POST_OBSERVATION_PERFORMANCE GROUP BY 1)
SELECT m.psrk, m.platform, m.post_type, m.post_url, m.pub, m.obs_latest, DATEDIFF('day', m.pub, m.obs_latest) age_latest,
  CASE
    WHEN tg.spend > 0 THEN 'CONFIRMED_PAID_TAG'
    WHEN p.spend > 0 THEN 'CONFIRMED_AD_LINK'
    WHEN m.platform = 'Instagram' AND o.optin_ratio < 0.80 THEN 'MEASURED_OPTIN_GAP'
    WHEN m.platform = 'Instagram' AND (s.api_total - s.api_ig - COALESCE(s.api_fb, 0)) > 0.25 * NULLIF(s.api_ig, 0) THEN 'MEASURED_SOCAPI_GAP'
    WHEN m.paid_date IS NOT NULL THEN 'LOGGED_PAID_DATE_ONLY'
    ELSE 'NO_PAID_EVIDENCE' END boost_evidence,
  fs.first_spend, m.paid_date, bs.boost_start, ROUND(COALESCE(p.spend, tg.spend), 2) spend,
  p.impr_ig paid_impressions_ig, p.impr_fb paid_impressions_fb, p.plays_ig paid_plays_ig, p.plays_fb paid_plays_fb,
  tg.impr paid_impressions_tagged, tg.starts paid_plays_tagged,
  o.optin_ratio, o.optin_ratio_latest, o.optin_stale, o.optin_freeze_age, o.optin_frozen_days, o.public_growth_since_freeze,
  s.api_total, s.api_ig, s.api_fb,
  m.views_public_latest, m.views_private_latest,
  pr.views_before_first_spend, DATEDIFF('day', m.pub, pr.preboost_read_date) preboost_age_days
FROM mart m
LEFT JOIN tag tg ON tg.psrk = m.psrk AND tg.platform = m.platform
LEFT JOIN paid p ON p.psrk = m.psrk AND p.platform = m.platform
LEFT JOIN first_spend fs ON fs.psrk = m.psrk AND fs.platform = m.platform
LEFT JOIN boost_start bs ON bs.psrk = m.psrk AND bs.platform = m.platform
LEFT JOIN optin o ON o.psrk = m.psrk AND o.platform = m.platform
LEFT JOIN pre pr ON pr.psrk = m.psrk AND pr.platform = m.platform
LEFT JOIN soc s ON m.platform = 'Instagram' AND s.psrk = m.psrk
