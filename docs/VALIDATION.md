# Validation Report (data:validate-data checklist)

### Overall assessment: Share with noted caveats

The detector is methodologically sound for its stated use (flag likely boosts where no paid evidence exists, and
separate organic from paid views where evidence exists). The caveats below must travel with any number.

### Methodology review
- **Question:** "Was this post boosted, and how many of its views are organic?" Labels come from paid / opt-in data;
  features come from public data only. The model answers "does the public pattern look like a boost?".
- **Data:** BIRA campaign post mart and daily observation time series, SocAPI Instagram fact, Meta and TikTok ad dims
  and facts (Snowflake, pulled 2026-10-07). Posts published 2025-01-01 or later, in-feed Instagram and TikTok.
- **Population:** posts with >= 5 observations and >= 28 days of data; views and followers > 0. Stories, image posts
  with no public views, and YouTube are out of scope for the model.
- **Grain:** one row per (platform, post id). Checked: 0 duplicate post ids after every join (6,416 rows in, 6,416 out).

### Issues found
1. [Medium] Instagram organic labels need opt-in data, so the Instagram model is trained mostly on opt-in creators.
   Impact: performance on non-opt-in creators is not directly measured.
2. [Medium] Silver organic labels: a post with no VN ad link may still have been boosted by the client or creator.
   Impact: test precision is a floor. Spot checks show most "false positives" have boost signatures.
3. [Low] Instagram ad-link-only slice has 15 test positives; its metrics have wide intervals.
4. [Low] LLM baselines use a fixed 0.5 threshold (not tuned), and 150 posts per platform.

### Leakage and bias checks
- Paid date, boosted / pre-boost flags, ad names, spend, ad dates, opt-in and SocAPI values are never features (asserted in code).
- Fixed 30-day feature window for every post (no window-length leak).
- Time-based locked test set (no look-ahead); creator-grouped CV for selection; threshold from train only.
- Selection bias: positives required the first spend within 28 days of publish, matching the feature window.

### Calculation spot-checks
- Shortcode → media id decoder: verified against Python on 8 posts (exact match).
- Reconciliation example (one boosted Reel): SocAPI total 4.89M = opt-in organic 0.01M + paid IG plays 3.08M (×1.06 in public)
  + paid FB plays 1.51M (×1.07 in SocAPI). Components add up within 7%.
- Opt-in same-day ratio: private and public reads come from the same observation day (staleness 0 days for all 1,422 posts checked).

### Required caveats for stakeholders
- Precision and recall are measured against labels that miss some boosts; treat precision as a lower bound.
- Organic estimates for boosted Instagram posts without opt-in use a 1.14 factor (median, IQR 1.08-1.24), not a platform rule.
- TikTok organic views for boosted posts are a range (pre-boost read to public minus paid plays), not one number.
- YouTube cannot be reconciled: no ad-to-video link exists in the warehouse.
