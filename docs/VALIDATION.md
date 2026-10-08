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
- Fixed feature window per model (days 0-14, 0-30 or 0-60) for every post; follower counts follow the same window (v2.2).
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
- **Pre-registration:** targets C1-C6 written before any v2 experiment (`docs/IMPROVEMENT_PLAN.md`). Amendments 1-3 were
  logged before the first held-out evaluation; amendment 4 (frozen opt-in label rule) and amendment 5 (leak repair)
  after it. The amendment 4 rule was set on train posts, but after the first test results had been seen. All model, feature-set, calibration and threshold choices use train cross-validation grouped by creator
  (v2: 192 runs, `results/model_v2_selection.csv`; Instagram v2.1: 72 runs, `results/model_v2_1_selection.csv`).
- **Held-out use:** locked test (posts 2026-07-01 to 09-09): v1 once, v2 once, the label fix (third look), the leak
  repair (fourth look). Fresh posts (2026-09-10 to 09-24): three times. Every result is reported, including the first, uncorrected one. Next clean test:
  posts published after 2026-09-24.
- **Production scorecard** (v2.2 on both platforms, corrected labels; `results/model_v2_2_targets.json`): C1, C3, C5,
  C6 met on both. C2: Instagram met (0.892); TikTok 0.927 vs 0.93 (missed by 0.003; interval 0.89-0.96). C4: TikTok met;
  Instagram missed (day-14 F1 0.80 vs day-30 0.89).
- **Leak repair (amendment 5):** an independent audit found `followers` was the median over days 0-60 for every model
  day, so day-14 and day-30 models saw follower counts from after their window. v2.2 uses the median up to the model day
  (`sql/14`) and day-30 eligibility up to day 30. Effect on the locked test (day 30): TikTok F1 0.931 -> 0.927;
  Instagram large boosts caught 89.6% -> 90.4%. Day-60 models are unchanged (their window was already 0-60).
- **Overfitting:** train in-sample AUC is about 1.00 for the tree models (they memorise train). Use the train-CV vs test
  gap instead (v2.2, day 30): TikTok 0.989 -> 0.979; Instagram 0.957 -> 0.966.
- **Calibration:** test calibration error 0.024 (TikTok), 0.028 (Instagram); no isotonic step needed (train
  out-of-fold error under 0.05 for every v2.2 model).
- **Fresh posts are a small sample:** 10 TikTok and 16 Instagram paid posts, so C5 can only say the target is not ruled
  out (precision intervals 35-92% and 56-94%).
- **Label audit (Instagram, frozen opt-in):** the opt-in count stopped updating on 2,177 of 6,030 Instagram posts with
  opt-in (`results/optin_staleness.json`). Labels whose gap appears only after the freeze were dropped (day 30: 207).
  Independent check with SocAPI: 11 of 68 dropped test posts show paid plays, vs 60 of 126 kept paid posts and 11 of
  501 organic posts. So the rule mostly removes wrong labels; about 1 in 6 dropped posts is a real boost (stress test:
  counting them as paid gives 89.7% for the v2 model).
- **Measured organic and evidence:** `sql/05` now uses the opt-in ratio from the last read where opt-in still changed.
  706 Instagram posts lose `MEASURED_OPTIN_GAP` (460 to no evidence, 211 to manual paid date, 35 to SocAPI gap). Stale
  opt-in is not `ORGANIC_MEASURED`; its organic estimate is opt-in at the freeze x organic curve. The freeze starts after
  the last read with a different opt-in value (23 of 6,303 posts had their final value earlier too; fixed 2026-10-08,
  6 stale flags and 2 gap decisions changed; the training labels came from the sql/13 pull before this fix, so at most
  23 posts could carry the old freeze). Checked in Snowflake: 0 duplicate (post, date) reads, 0 zero-public reads
  with opt-in, 0 reads before publish.
- **Independent tests (2026-10-08):** five separate test agents checked leakage, every number in the docs, pipeline edge
  cases, SQL vs Python consistency, and privacy. Bugs found and fixed: the follower-count leak, opt-in estimate above
  public views, crashes on unknown ages, silent handling of unknown evidence tiers and bad input files, the freeze start
  above, opt-in fallback without the stale check, and the Snowflake path still scoring with v1. Passed: no paid field in
  any model input, creator features use only earlier posts, thresholds and calibration fit on train only, no creator in
  both a training and a validation fold, no post IDs or names in tracked files.
- **Instagram day 14 (C4 miss), train-only analysis:** the day-14 model catches 83.7% of large boosts out of fold
  (381 posts): 88% of opt-in gaps but 67% of ad-linked boosts, including boosts that start on days 0-6. Of its 62 misses,
  25 are caught by the day-30 model on the same posts (the boost had only a few days to show). No feature tried closes
  the gap; treat a day-14 Instagram "organic" as provisional until the day-30 score exists.
- **Not evaluated by design:** posts whose paid evidence starts after the model window get no label (TikTok day 14:
  11% of posts with evidence). The daily table gives those posts their status from the evidence, never from the model.
- **Join check:** the Parquet has 41,484 rows and 41,484 unique (post, platform) keys, the same keys as the v1 file.

### Required caveats for stakeholders
- Precision and recall are measured against labels that miss some boosts; treat precision as a lower bound.
- Do not use "public - paid metric" per post. Use opt-in when it exists, else the pre-boost read x organic curve.
  The 1.14 factor from phase 1 is retired: impressions match best (1.07x), but subtraction still fails per post.
- TikTok organic after a boost is UNVERIFIED: opt-in includes Spark Ad views, so there is no organic truth. The curve
  method is back-tested only on unboosted TikTok posts.
- Boosts that start before the first public read (12% of boosted Instagram posts, 24% of TikTok) cannot be separated.
- Instagram model (v2.2, corrected labels): it misses about 10% of large boosts at day 30 and more at day 14 (F1 0.79);
  a model "paid" is 94% right. Treat a day-14 Instagram "organic" as provisional until the day-30 score exists.
- YouTube cannot be reconciled: no ad-to-video link exists in the warehouse.
