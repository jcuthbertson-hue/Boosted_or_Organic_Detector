-- LLM baseline: Snowflake Cortex COMPLETE on a fixed sample of LOCKED-TEST posts (published >= 2026-07-01).
-- Same public 30-day numbers the ML model sees. No paid, opt-in or label fields go into the prompt.
-- {model} = openai-gpt-5 | claude-sonnet-4-5 | llama3.1-70b.  {result_id} = query id of the 02_ export.
WITH t AS (
  SELECT f.value v FROM TABLE(RESULT_SCAN('{result_id}')) r, LATERAL FLATTEN(input => PARSE_JSON(r.J)) f
  WHERE f.value:LABEL::STRING IN ('POS','NEG') AND f.value:PUB::DATE >= '2026-07-01'
    AND f.value:V30::FLOAT > 0 AND f.value:FOLLOWERS::FLOAT > 0
  QUALIFY ROW_NUMBER() OVER (PARTITION BY f.value:PLATFORM::STRING ORDER BY HASH(f.value:PSRK::STRING)) <= 150),
p AS (
  SELECT v:PSRK::STRING psrk, v:PLATFORM::STRING platform, v:LABEL::STRING label,
    'You are a social media analyst. Decide if paid advertising (boosting) was applied to this creator post during its first 30 days.\n'
    || 'Background: Organic views are front-loaded (most views in the first days) and come with normal engagement. Paid boosting adds many views from people who do not follow the creator, often days after posting, with few extra likes or comments, so views can exceed followers and engagement rate falls.\n'
    || 'Platform: ' || v:PLATFORM::STRING || '\n'
    || 'Creator followers: ' || COALESCE(ROUND(v:FOLLOWERS::FLOAT)::STRING, 'unknown') || '\n'
    || 'Public views on day 1: ' || COALESCE(v:V1::STRING, 'unknown') || '; day 3: ' || COALESCE(v:V3::STRING, 'unknown')
    || '; day 7: ' || COALESCE(v:V7::STRING, 'unknown') || '; day 14: ' || COALESCE(v:V14::STRING, 'unknown') || '; day 30: ' || COALESCE(v:V30::STRING, 'unknown') || '\n'
    || 'Likes on day 7: ' || COALESCE(v:L7::STRING, 'unknown') || '; day 30: ' || COALESCE(v:L30::STRING, 'unknown') || '\n'
    || 'Comments on day 30: ' || COALESCE(v:C30::STRING, 'unknown') || '\n'
    || 'Shares on day 30: ' || COALESCE(v:S30::STRING, 'unknown') || '\n'
    || 'Reply with only a number from 0 to 1: the probability that this post was boosted with paid ads.' prompt
  FROM t)
SELECT LISTAGG(psrk || '|' || platform || '|' || label || '|' || REPLACE(REPLACE(SNOWFLAKE.CORTEX.COMPLETE('{model}', prompt), '\n', ' '), '|', ' '), '\n') csv
FROM p
