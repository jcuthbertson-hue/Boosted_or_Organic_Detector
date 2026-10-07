-- Q1. Can the ad tables tell us WHICH organic post was boosted?
-- Run each block separately. Results are counts only.

-- 1a. TikTok: Spark Ads carry the organic video id (TIKTOK_ITEM_ID). Match it to tracked campaign posts.
WITH posts AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, MIN(POST_PAID_ACTIVITY_DATE) paid_date
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM = 'Tiktok' GROUP BY 1),
spend AS (
  SELECT a.TIKTOK_ITEM_ID psrk, SUM(f.SPEND) spend, MIN(IFF(f.SPEND > 0, f.DATE, NULL)) first_spend
  FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD a JOIN SNOWFLAKE_EDW.EDW.FACT_TIKTOK_ADS__AD_PERFORMANCE f ON f.AD_KEY = a.AD_KEY
  WHERE a.TIKTOK_ITEM_ID IS NOT NULL GROUP BY 1 HAVING SUM(f.SPEND) > 0)
SELECT COUNT(*) tiktok_posts,
       COUNT_IF(s.psrk IS NOT NULL) linked_to_ad_with_spend,
       COUNT_IF(s.psrk IS NOT NULL AND p.paid_date IS NULL) linked_but_no_logged_paid_date,
       COUNT_IF(p.paid_date IS NOT NULL AND s.psrk IS NULL) logged_paid_date_but_no_ad_link,
       COUNT_IF(s.psrk IS NOT NULL AND p.paid_date IS NOT NULL AND ABS(DATEDIFF('day', p.paid_date, s.first_spend)) <= 1) paid_date_within_1d_of_first_spend
FROM posts p LEFT JOIN spend s ON s.psrk = p.psrk;

-- 1b. Meta: the ad's own Instagram permalink is a NEW media object, never the organic post.
--     Decode each ad shortcode to its media id and look for it in the full post universe.
WITH ad_sc AS (
  SELECT DISTINCT REGEXP_SUBSTR(COALESCE(a.INSTAGRAM_PERMALINK_URL, c.INSTAGRAM_PERMALINK_URL), 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2) sc
  FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a LEFT JOIN SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__CREATIVE c ON a.CREATIVE_KEY = c.CREATIVE_KEY),
chars AS (
  SELECT sc, f.index pos, LENGTH(sc) len,
         POSITION(f.value::STRING, 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_') - 1 d
  FROM ad_sc, LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(sc, '.')) f WHERE LENGTH(sc) BETWEEN 10 AND 11),
dec AS (
  SELECT sc, SUM(IFF(pos < len - 6, d * POWER(64, len - 7 - pos), 0))::NUMBER(38,0) hi,
             SUM(IFF(pos >= len - 6, d * POWER(64, len - 1 - pos), 0))::NUMBER(38,0) lo
  FROM chars GROUP BY sc),
pk AS (SELECT sc, (hi * 68719476736 + lo)::STRING pk FROM dec)   -- 68719476736 = 64^6
SELECT (SELECT COUNT(*) FROM pk) ad_media_objects, COUNT(DISTINCT pk.sc) matching_any_tracked_post
FROM pk JOIN DM_BUSINESS_INTELLIGENCE.PUBLIC.DT_DIM_ORGANIC__POST_CANONICAL c ON c.POST_SCRAPER_REFERENCE_KEY = pk.pk;

-- 1c. Meta: the only link is the ad-name taxonomy ("handle ~ SHORTCODE"), parsed into
--     DIM_FACEBOOK_ADS__AD.POST_SCRAPER_REFERENCE_KEY, plus any 11-character shortcode in the ad name.
--     PAID_MEDIA_UNIFIED.ext_p3_organic_post_id is almost never filled.
SELECT "platform", "publisher", COUNT(*) rows_, COUNT_IF(NULLIF("ext_p3_organic_post_id", '') IS NOT NULL) rows_with_organic_post_id,
       COUNT_IF(NULLIF("meta_instagram_permalink_url", '') IS NOT NULL) rows_with_ig_permalink,
       COUNT_IF(NULLIF("tiktok_item_id", '') IS NOT NULL) rows_with_tiktok_item_id
FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED
WHERE "platform" IN ('meta', 'tiktok') GROUP BY 1, 2 ORDER BY 3 DESC;
