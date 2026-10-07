-- Posts in the paid team's post-ID tag (gold positives), for checking labels, links and the model.
-- PAID_MEDIA_UNIFIED."ext_p3_organic_post_id" holds the organic post id the ad runs (IG shortcode or TikTok video id).
-- The paid team fills it from Sep 2026, so the set is small and grows each week.
-- Per post: is it a tracked campaign post (BIRA), and do our own link rules also find it?
--   Meta   : 11-character token in an EDW ad name = shortcode (the rule in sql/02 and sql/05)
--   TikTok : DIM_TIKTOK_ADS__AD.TIKTOK_ITEM_ID = video id (Spark Ad link)
-- Export to data/tagged_paid_posts.csv (header row, comma-separated).
WITH u AS (
  SELECT "platform" pf, "ext_p3_organic_post_id" pid, MIN("date") first_spend, ROUND(SUM("spend")) spend
  FROM DM_PAID_MEDIA.PUBLIC.PAID_MEDIA_UNIFIED
  WHERE "ext_p3_organic_post_id" IS NOT NULL AND "ext_p3_organic_post_id" <> '' AND "platform" IN ('meta', 'tiktok')
  GROUP BY 1, 2),
m AS (
  SELECT POST_SCRAPER_REFERENCE_KEY psrk, POST_PLATFORM platform, ANY_VALUE(POST_TYPE) post_type,
         ANY_VALUE(REGEXP_SUBSTR(POST_URL, 'instagram\\.com/(p|reel|reels|tv)/([A-Za-z0-9_-]+)', 1, 1, 'e', 2)) sc,
         MIN(POST_PUBLISHED_AT)::DATE pub
  FROM DM_BUSINESS_INTELLIGENCE.BI_REPORTING_APP.BIRA_MART_ORGANIC__CAMPAIGN_POST_PERFORMANCE
  WHERE POST_PLATFORM IN ('Instagram', 'Tiktok') GROUP BY 1, 2),
meta_tok AS (SELECT DISTINCT t.value::STRING tok FROM SNOWFLAKE_EDW.EDW.DIM_FACEBOOK_ADS__AD a,
             LATERAL FLATTEN(input => REGEXP_SUBSTR_ALL(a.AD_NAME, '[A-Za-z0-9_-]{11}')) t),
tt_item AS (SELECT DISTINCT TIKTOK_ITEM_ID item FROM SNOWFLAKE_EDW.EDW.DIM_TIKTOK_ADS__AD WHERE TIKTOK_ITEM_ID IS NOT NULL)
SELECT u.pf, u.pid, m.psrk, m.post_type, m.pub, u.first_spend, DATEDIFF('day', m.pub, u.first_spend) days_pub_to_spend, u.spend,
       m.psrk IS NOT NULL in_bira,
       CASE WHEN u.pf = 'meta' THEN u.pid IN (SELECT tok FROM meta_tok) ELSE u.pid IN (SELECT item FROM tt_item) END found_by_our_link
FROM u
LEFT JOIN m ON (u.pf = 'meta' AND m.platform = 'Instagram' AND m.sc = u.pid) OR (u.pf = 'tiktok' AND m.platform = 'Tiktok' AND m.psrk = u.pid)
ORDER BY u.pf, u.first_spend
