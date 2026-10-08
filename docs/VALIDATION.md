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

### Phase 2 checks (subtraction question, 2026-10-07)
- **Re-run reproducibility:** `detector/train_eval.py` was re-run to save train out-of-fold scores. Test scores match the
  first run to 2.2e-16 and the report JSON is identical. No model, feature or threshold changed after the test was seen.
- **Grain:** reconciliation panel = one row per (post, observation day): 13,859 rows, 0 duplicate post-days, 159 posts.
  Post-level errors use one read per post (the latest read at least 2 days after the last ad day).
- **Truth:** Instagram opt-in views. Before the first ad day, opt-in / public = 1.001 (117 posts), so opt-in is organic.
- **No fitting on scored posts:** the organic curve is built from unboosted posts only; fitted factors use leave-one-out.
- **In-sample warning fixed:** the first curve back-test reused the posts that built the curve. A creator-split version
  (fit on half the creators, test on the other half) gives the same picture (Instagram day-7 read 8.3% vs 10.0% error).
- **Production function check:** `reconcile/estimate.py` on the 117 boosted posts reproduces the analysis
  (11.5% typical error, 77% within 25%). Confidence tiers: high 8.6% (65 posts), medium 12.5% (20), low 30.5% (32).
- **Gold check (post-ID tag, `sql/08`):** the Meta ad-name rule finds 21 of 21 tagged posts; the TikTok Spark link
  finds 0 of 3 tagged campaign posts. One TikTok test "false positive" is a tagged paid post (label error, model right).
- **Cluster bootstrap:** classification intervals resample whole creators (1,000 draws), so they are wider than row bootstrap.
- **SQL logic check:** `sql/05` was run read-only in Snowflake on 2026-10-07; tier counts and pre-boost coverage are in
  `results/organic_coverage.json`.

### Model v2 checks (2026-10-08)
- **Pre-registration:** targets C1-C6 written before any v2 experiment (`docs/IMPROVEMENT_PLAN.md`); three amendments,
  each logged before the held-out evaluation. All model, feature-set, calibration and threshold choices use train
  cross-validation grouped by creator (192 candidate runs, `results/model_v2_selection.csv`).
- **Held-out use:** locked test (posts 2026-07-01 to 09-09) second look; fresh posts (2026-09-10 to 09-24) one look.
  Results are reported as measured, including the three Instagram misses (C1, C2, C4).
- **Overfitting:** train in-sample AUC is 1.00 for every v2 model (boosted trees memorise train). Use the train-CV vs test
  gap instead: TikTok 0.989 -> 0.977; Instagram 0.927 -> 0.876. Client-holdout CV: 0.977 / 0.918.
- **Calibration:** isotonic calibration (creator-grouped CV on train) for the Instagram models, because their train
  out-of-fold calibration error was above 0.05 (plan rule). Test calibration error 0.043 (Instagram), 0.023 (TikTok).
- **Fresh posts are a small sample:** 10 TikTok and 19 Instagram paid posts, so C5 can only say the target is not ruled
  out (precision intervals 42-92% and 50-92%).
- **Label audit (Instagram):** missed large boosts look organic on public data and are mostly not confirmed by SocAPI
  (3 of 26 test, 23 of 56 train). Instagram opt-in "paid" labels need an audit before the model target can be met.
- **Measured organic:** Instagram posts with no paid evidence and opt-in >= 90% of public views are `ORGANIC_MEASURED`
  (same cut as the organic training label). The offline run uses the latest mart totals for this ratio; `sql/05` uses
  the latest read with both values.
- **Join check:** the v2 Parquet has 41,484 rows, 41,484 unique (post, platform) keys; same keys as the v1 file, which
  matched BIRA on every sampled key.

### Required caveats for stakeholders
- Precision and recall are measured against labels that miss some boosts; treat precision as a lower bound.
- Do not use "public - paid metric" per post. Use opt-in when it exists, else the pre-boost read x organic curve.
  The 1.14 factor from phase 1 is retired: impressions match best (1.07x), but subtraction still fails per post.
- TikTok organic after a boost is UNVERIFIED: opt-in includes Spark Ad views, so there is no organic truth. The curve
  method is back-tested only on unboosted TikTok posts.
- Boosts that start before the first public read (12% of boosted Instagram posts, 24% of TikTok) cannot be separated.
- Instagram model "organic" is not proof: on the locked test it misses 17% of large boosts. A model "paid" is 92% right.
- YouTube cannot be reconciled: no ad-to-video link exists in the warehouse.
