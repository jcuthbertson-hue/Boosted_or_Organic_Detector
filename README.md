# Boosted or Organic Detector

Separates paid from organic views on creator posts, and finds which posts are paid.
Built for the BI team's paid-vs-organic research (Instagram first, then TikTok; YouTube checked).

> This repository is public. It holds code, SQL and aggregate results only.
> Row-level data, client names, spend per post and the trained models are kept out by `.gitignore`.
> Recommended: make the repository private.

**Full report with every chart and metric:** `reports/boost_report.html` (built by `python3 -m reports.build_report`).

## Bottom line (data read 2026-10-07)

The question: if we take the views on the platform and subtract one paid metric from the unified paid table, do we get
organic views? We tested every paid metric against Instagram opt-in views, which are organic only.

1. **Impressions is the paid metric that matches.** Instagram: public - opt-in = 1.07 x paid Instagram-placement
   impressions (median; middle half 1.01-1.15; 146 boosted posts after the ads ended). Facebook: SocAPI's extra plays =
   1.08 x paid Facebook-placement video plays (20 posts). 2-second, 3-second, ThruPlay and %-watched counts are 5-110x too small.
2. **The subtraction still fails for one post.** Paid is a median 94% of public views on a boosted post, so a small paid
   error becomes a large organic error. Best case (public - paid IG impressions): median error 70%, 31% of posts within
   +/-25%, 11% negative. On the 10 posts in the paid team's post-ID tag, 47% of in-flight days give negative organic.
3. **Use the pre-boost read x organic growth instead.** Last public read before the first ad day, grown by the median
   curve of unboosted posts: median error 11.5%, 77% of posts within +/-25%, never negative (117 boosted Instagram
   posts vs opt-in). 10-post campaign totals: 95% within +/-25% (subtraction: 59%). A read on day 14+ gives 8.6% error.
4. **The paid team's post-ID tag is the best proof of paid** (`PAID_MEDIA_UNIFIED.ext_p3_organic_post_id`, Sep 2026+).
   Our Meta ad-name rule finds 21 of 21 tagged posts. The TikTok Spark item-id link finds 0 of 3 tagged campaign posts.
5. **Where no paid record exists, the model (v2) meets every target on TikTok, not on Instagram** (locked test of later
   posts, day-30 model). TikTok: F1 0.93, large boosts caught 92% at 95% precision. Instagram: 92% of flags are paid,
   but it catches 83% of large boosts (target 90%; 95% interval 75-89%). The 26 missed Instagram boosts look organic
   in public data (0.09 views per follower, same as organic posts; caught boosts 2.7); only opt-in calls them paid.
   So on Instagram a model flag is reliable, a model "organic" is not proof. Use opt-in or an ad link when they exist.

**Caveat that can change the decision:** TikTok organic after a boost is UNVERIFIED. TikTok opt-in includes Spark Ad
views, so there is no organic truth. The curve method passes a back-test on unboosted TikTok posts only.

### Coverage today (all boosted campaign posts in BIRA)

| | Instagram (2,946 posts) | TikTok (1,093 posts) |
|---|---|---|
| Organic measured (opt-in) | 71% | 0% |
| Estimate, high or medium confidence (pre-boost read on day 7+) | 12% | 45% |
| Estimate, low confidence (read before day 7) | 5% | 31% |
| Not separable (boosted before the first read) | 12% | 24% |

## Detector results: model v2 against targets set before the test

Targets: `docs/IMPROVEMENT_PLAN.md` (written before any v2 experiment; amendments logged before any held-out look).
Scorecard: `results/model_v2_targets.json`. Every output: `results/classification_metrics_v2.json`.

| Target | TikTok | Instagram |
|---|---|---|
| C1 Large boosts caught at >= 90% precision (locked test, day 30) | 92% at 95% precision: **met** | 83% at 92%: **missed** (95% interval 75-89%) |
| C2 F1 (TikTok >= 0.93, Instagram >= 0.80) | 0.931: **met** | 0.781: **missed** (v1 0.771) |
| C3 Calibration error <= 0.05 | 0.023: **met** | 0.043: **met** |
| C4 Day-14 F1 within 0.05 of day 30 (same posts); coverage >= 85% | 0.87 vs 0.88: **met** | 0.71 vs 0.81: **missed** |
| C5 Fresh posts (published Sep 10-24, day-14 model), C1 inside its 95% interval | 90% caught, 69% precise, 10 paid posts: met, small sample | 88% caught, 75% precise, 19 paid posts: met, small sample |
| C6 Tagged paid posts with the boost inside the window are flagged | 1 of 1: **met** | 8 of 8: **met** (11 of 11 model scores, days 14 and 30) |

Coverage (C4): 99.8% of posts tracked by day 7 and at least 14 days old get a score (target 85%).

| Locked test (posts 2026-07-01 to 09-09) | Posts (paid) | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| TikTok day 14 / 30 / 60 | 689 (68) / 758 (132) / 450 (104) | 0.96 / 0.95 / 0.99 | 0.79 / 0.92 / 0.88 | 0.87 / 0.93 / 0.93 | 0.97 / 0.98 / 0.97 |
| Instagram day 14 / 30 / 60 | 597 (108) / 701 (195) / 496 (168) | 0.90 / 0.92 / 0.92 | 0.57 / 0.68 / 0.73 | 0.70 / 0.78 / 0.81 | 0.84 / 0.88 / 0.90 |

v1 vs v2 on the same locked test (day 30): TikTok F1 0.926 -> 0.931, AUC 0.964 -> 0.977; Instagram F1 0.771 -> 0.781,
AUC 0.888 -> 0.876. The v2 gains are scoring earlier (day 14) and later (day 60, sees boosts that start after day 21),
calibration, and coverage, not day-30 accuracy. The day-60 model was added after the targets were set and has no target.

**Why Instagram misses** (`results/ig_miss_audit.json`): the missed large boosts look organic on public data. SocAPI
shows paid plays on 3 of 26 missed test boosts (caught: 58 of 122; organic: 11 of 501) and on 23 of 56 missed train
boosts. So some are real boosts that public data cannot see, and some opt-in "paid" labels are doubtful. No public
feature tried on train moved Instagram past about 87% (details: plan amendment 3).

## Validity (what keeps the numbers honest)

- **No paid information in model inputs.** Only public views, likes, comments, shares and followers up to the model
  day (14, 30 or 60), and the same creator's earlier posts. `detector/features.py` and `detector/features_v2.py` assert
  that no paid date, boost flag, ad name, spend, opt-in or SocAPI field is a feature.
- **Targets before tests.** v2 targets were written before any v2 experiment. Every v2 change (features, day-60 model,
  calibration) was chosen on train cross-validation and logged in the plan before the held-out evaluation. The locked
  test was used once for v1 and once more for v2 (second look); the fresh posts were used once.
- **Locked test by time** (train before 2026-07-01). Model family (13 candidates) and threshold chosen on train only,
  with 5-fold CV grouped by creator. Re-running the code reproduces the test scores exactly.
- **Organic truth** = Instagram opt-in views (equal to public before a boost: ratio 1.001, 117 posts). The organic curve
  is built from unboosted posts only; fitted factors use leave-one-out; the curve back-test uses a creator split.
- **Uncertainty:** creator-cluster bootstrap for classifier intervals; 2,000 random 10-post groups for campaign errors.
- Details and caveats: `docs/VALIDATION.md`.

## Daily paid classification table (scaffold)

`pipeline/run_daily.py` runs `sql/05` (evidence + organic inputs), marks Instagram posts with opt-in >= 90% of public
views as `ORGANIC_MEASURED`, scores the other posts with no evidence (longest model the post's data allows: day 60,
then 30, then 14), adds the organic
estimate (`reconcile/estimate.py`), and MERGEs into `PAID_CLASSIFICATION__POST` (`sql/09`). The logic is tested offline
(`python3 -m tests.test_pipeline`). The Snowflake read/write path is NOT tested yet: it needs a service account and a
schema we may write to.

## All-post export (Parquet)

`python3 -m pipeline.run_daily --flags data/flags.csv --features data/features_all.psv --features-v2 data/v2_raw.psv --parquet outputs/paid_classification_<date>.parquet`
(`data/v2_raw.psv` = `sql/11`, `data/v2_extra.psv` = `sql/12`, read together)
writes one row per Instagram / TikTok campaign post (git-ignored; row-level). Column names are the UPPERCASE
table columns. Join back to Snowflake on `POST_SCRAPER_REFERENCE_KEY` + `POST_PLATFORM` (BIRA mart).
To put it in Snowflake, run `sql/10_sandbox_test_load.sql` in a worksheet: it loads a 500-row test file into
`DM_BUSINESS_INTELLIGENCE.SANDBOX_JCUTHBERTSON` first, with expected check results, then the full file. Key columns:

| Column | Meaning |
|---|---|
| `IS_PAID` | True / False; empty when the post has no paid record and the model could not score it |
| `PAID_BASIS` | `evidence` (tag, ad link, opt-in or SocAPI gap, opt-in >= 90% organic, manual date) or `model` |
| `PAID_STATUS`, `BOOST_EVIDENCE` | status and the strongest proof tier |
| `MODEL_SCORE`, `MODEL_THRESHOLD`, `MODEL_HORIZON_DAYS`, `MODEL_NOTE` | calibrated 0-1 score from the day-60, 30 or 14 model, cut-off from train, which model, reason when there is no score |
| `ORGANIC_VIEWS_EST`, `ORGANIC_VIEWS_METHOD`, `ORGANIC_VIEWS_CONFIDENCE` | organic views (opt-in measured, or pre-boost read x organic curve) |

## How to run

```bash
pip install -r requirements.txt
# exports (Snowflake -> data/, git-ignored): sql/02 -> data/detector_dataset_v2.psv, sql/06 -> data/recon_panel_ig_full.psv,
# sql/08 -> data/tagged_paid_posts.csv
python3 -m detector.train_eval        # train, select, test once; models/ + results/detector_report.json
python3 -m detector.evaluate          # every classification output -> results/classification_metrics.json
# model v2 (sql/11 -> data/v2_raw.psv, sql/12 -> data/v2_extra.psv, SocAPI latest plays -> data/socapi_ig.psv)
python3 -m detector.train_v2 --develop --develop2   # train-only CV for every candidate -> results/model_v2_selection.csv
python3 -m detector.train_v2 --final  # fit on train, score locked test + fresh posts once -> models/, results/model_v2_report.json
python3 -m detector.v2_targets        # C1-C6 scorecard -> results/model_v2_targets.json
python3 -m detector.evaluate_v2       # every output -> results/classification_metrics_v2.json
python3 -m detector.ig_miss_audit     # why Instagram misses -> results/ig_miss_audit.json
python3 -m reconcile.analyze          # subtraction test on Instagram -> results/reconciliation.json
python3 -m reconcile.analyze_tiktok   # TikTok + curve back-test -> results/reconciliation_tiktok.json
python3 -m reports.build_report       # reports/boost_report.html
python3 -m tests.test_pipeline        # offline checks of the daily pipeline logic
python3 -m pipeline.run_daily --flags data/flags.csv --features-v2 data/v2_raw.psv   # offline daily run (flags = sql/05 output)
```

| File | Purpose |
|---|---|
| `sql/01_ad_to_post_link_coverage.sql` | How many posts the ad tables can link |
| `sql/02_detector_features_and_labels.sql` | Training / test table: 30-day public features + labels |
| `sql/03_view_reconciliation.sql` | Phase 1: public vs opt-in vs SocAPI vs paid plays |
| `sql/04_llm_cortex_eval.sql` | LLM baseline in Snowflake Cortex |
| `sql/05_boost_flags.sql` | Daily input: evidence tier (post-ID tag first) + organic-estimate inputs |
| `sql/06_reconciliation_panel.sql` | Post x day panel for the subtraction test (public, opt-in, SocAPI, paid by placement and metric) |
| `sql/07_organic_curve.sql` | Organic growth curve from unboosted posts |
| `sql/08_tagged_paid_posts.sql` | The paid team's tagged posts and whether our link rules find them |
| `sql/09_paid_classification_table.sql` | Table DDL + MERGE for the daily table (not executed) |
| `sql/10_sandbox_test_load.sql` | Load the Parquet export into a BI sandbox schema: test file first, with checks |
| `sql/11_v2_features_and_labels.sql` | Model v2 input: views / likes at many ages, day-14 and day-30 labels |
| `sql/12_v2_jump_features.sql` | Model v2 input: likes per new view in the biggest jump, day-60 values and label input |
| `docs/IMPROVEMENT_PLAN.md` | v2 targets and every change, logged before the held-out evaluation |
| `reconcile/` | Subtraction test, TikTok check, organic curve, production organic estimate |
| `detector/` | Features, train / test protocol, full evaluation, scoring |
| `pipeline/`, `tests/` | Daily run scaffold and its offline tests |
| `reports/` | HTML report builder and output |
| `llm/` | LLM prompt, scorer, local-model harness |
