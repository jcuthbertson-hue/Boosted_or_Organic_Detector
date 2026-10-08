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
5. **Where no paid record exists, the model (v2.2) sits on the target lines on both platforms** (locked test of later
   posts, day-30 model). TikTok: 91.6% of large boosts caught at 93.8% precision, F1 0.927 (target 0.93). Instagram:
   90.4% caught at 93.9% precision (target 90%; 95% interval 83-96%), F1 0.89. The weak spot is the day-14 Instagram
   model (F1 0.80 vs 0.89 at day 30). The first Instagram result (83%) was held down by bad labels: the opt-in count
   stopped updating on 2,177 of 6,030 Instagram posts while public views kept growing, so organic posts looked paid.
   The same fix corrects the daily table: 706 posts lose a false "measured paid" tag.

**Caveat that can change the decision:** TikTok organic after a boost is UNVERIFIED. TikTok opt-in includes Spark Ad
views, so there is no organic truth. The curve method passes a back-test on unboosted TikTok posts only.

### Coverage today (all boosted campaign posts in BIRA)

| | Instagram (2,946 posts) | TikTok (1,093 posts) |
|---|---|---|
| Organic measured (opt-in) | 71% | 0% |
| Estimate, high or medium confidence (pre-boost read on day 7+) | 12% | 45% |
| Estimate, low confidence (read before day 7) | 5% | 31% |
| Not separable (boosted before the first read) | 12% | 24% |

## Detector results: targets set before the test

Targets: `docs/IMPROVEMENT_PLAN.md` (written before any v2 experiment; every change logged there: amendments 4 and 5
came after the first held-out evaluation). Models in use: **v2.2** on both platforms = the configurations picked in
development (TikTok v2, Instagram v2.1), refit with labels corrected for frozen opt-in and follower counts known at
each horizon (an independent audit found the earlier follower count looked past the model day). Scorecard:
`results/model_v2_2_targets.json`; every output: `results/classification_metrics_v2_2.json`. Earlier runs stay in
`results/model_v2_targets.json` (v2) and `results/model_v2_1_targets.json` (v2.1).

| Target | TikTok (v2.2) | Instagram (v2.2) |
|---|---|---|
| C1 >= 90% of large boosts caught, at >= 90% precision (locked test, day 30) | 91.6% at 93.8%: **met** | 90.4% at 93.9%: **met** (95% interval 83-96%) |
| C2 F1 (TikTok >= 0.93, Instagram >= 0.80) | 0.927: **missed by 0.003** (95% interval 0.89-0.96) | 0.892: **met** |
| C3 Calibration error <= 0.05 | 0.024: **met** | 0.028: **met** |
| C4 Day-14 F1 within 0.05 of day 30 (same posts) | 0.86 vs 0.87: **met** | 0.80 vs 0.89: **missed** |
| C5 Fresh posts (published Sep 10-24, day-14 model), C1 inside its 95% interval | 90% caught, 64% precise, 10 paid posts: met, small sample | 88% caught, 78% precise, 16 paid posts: met, small sample |
| C6 Tagged paid posts with the boost inside the window are flagged | 1 of 1: **met** | 8 of 8: **met** (10 of 10 model scores, days 14 and 30) |

Coverage (C4): 99.8% of posts at least 14 days old with a public read by day 7 get a score (target 85%).
C1 and C2 sit within half a point of their lines on both platforms; that is inside the noise of one test set.

How the Instagram number moved (day 30, locked test, large boosts caught):

| Run | Change | Caught | Precision | Look at the locked test |
|---|---|---|---|---|
| v2 | labels as pulled | 82.6% | 92.3% | second (first for v2) |
| v2, same model | labels corrected for frozen opt-in | 91.3% (89.7% if the 11 SocAPI-paid dropped posts count as paid) | 90.8% | third |
| v2.1 | retrained on corrected labels | 89.6% | 93.9% | third |
| **v2.2 (in use)** | followers known at the horizon | **90.4%** | **93.9%** | fourth |

The model in use was fixed by rules set before each look (corrected labels, then the leak repair), not by which
result scored best on this test.

| Locked test (posts 2026-07-01 to 09-09), v2.2, labels corrected | Posts (paid) | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| TikTok day 14 / 30 / 60 | 689 (68) / 755 (131) / 450 (104) | 0.95 / 0.94 / 0.99 | 0.79 / 0.92 / 0.88 | 0.86 / 0.93 / 0.93 | 0.97 / 0.98 / 0.97 |
| Instagram day 14 / 30 / 60 | 571 (82) / 631 (126) / 439 (111) | 0.88 / 0.94 / 0.95 | 0.72 / 0.85 / 0.87 | 0.79 / 0.89 / 0.91 | 0.92 / 0.97 / 0.98 |

TikTok v1 vs v2 in the same date window (738 vs 758 posts): F1 0.926 -> 0.931, AUC 0.964 -> 0.977. The day-60 model was
added after the targets were set and has no target.

**Frozen opt-in** (`results/optin_staleness.json`, `sql/13`): on 2,177 of 6,030 Instagram posts with opt-in, the opt-in
count had not changed for 7+ days at the latest read while public views grew 5%+. The ratio then falls with no paid
views, so organic posts got "paid" labels. Corrected: the ratio is taken at the start of the final unchanged opt-in run;
a gap that appears only after the freeze gives no label (day 30: 207 Instagram labels). Check: SocAPI shows paid plays
on 11 of 68 dropped test posts vs 60 of 126 kept paid posts and 11 of 501 organic posts. Before the fix, no public
feature tried on train moved day-30 Instagram past about 87% (`results/ig_miss_audit.json`, plan amendment 3).

## Validity (what keeps the numbers honest)

- **No paid information in model inputs.** Only public views, likes, comments, shares and followers up to the model
  day (14, 30 or 60; followers fixed in v2.2), and the same creator's earlier posts. `detector/features.py` and `detector/features_v2.py` assert
  that no paid date, boost flag, ad name, spend, opt-in or SocAPI field is a feature.
- **Targets before tests.** v2 targets were written before any v2 experiment. Every v2 change (features, day-60 model,
  calibration, the frozen opt-in label rule) was chosen on train cross-validation and logged in the plan before it was
  applied to held-out posts. The locked
  test was used once for v1, once for v2, once after the label fix and once after the leak repair (fourth look); the
  fresh posts three times. The next clean test is posts published after 2026-09-24.
- **Locked test by time** (train before 2026-07-01). Model family (13 candidates) and threshold chosen on train only,
  with 5-fold CV grouped by creator. Re-running the code reproduces the test scores exactly.
- **Organic truth** = Instagram opt-in views (equal to public before a boost: ratio 1.001, 117 posts). The organic curve
  is built from unboosted posts only; fitted factors use leave-one-out; the curve back-test uses a creator split.
- **Uncertainty:** creator-cluster bootstrap for classifier intervals; 2,000 random 10-post groups for campaign errors.
- Details and caveats: `docs/VALIDATION.md`.

## Daily paid classification table (scaffold)

`pipeline/run_daily.py` runs `sql/05` (evidence + organic inputs; Instagram opt-in evidence uses the last read where
opt-in still changed), marks Instagram posts with live opt-in >= 90% of public views as `ORGANIC_MEASURED`, scores the other posts with no evidence (longest model the post's data allows: day 60,
then 30, then 14), adds the organic
estimate (`reconcile/estimate.py`), and MERGEs into `PAID_CLASSIFICATION__POST` (`sql/09`). The logic is tested offline
(`python3 -m tests.test_pipeline`). The Snowflake read/write path is NOT tested yet: it needs a service account and a
schema we may write to.

## All-post export (Parquet)

`python3 -m pipeline.run_daily --flags data/flags.csv --optin-live data/optin_live.psv --tier-changes data/optin_tier_changes.psv --parquet outputs/paid_classification_<date>.parquet`
(model inputs default to `data/v2_raw.psv` = `sql/11`, `data/v2_extra.psv` = `sql/12`, `data/v2_followers.psv` = `sql/14`;
`--snowflake` runs the same SQL directly)
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
| `ORGANIC_VIEWS_EST`, `ORGANIC_VIEWS_METHOD`, `ORGANIC_VIEWS_CONFIDENCE` | organic views: opt-in measured; opt-in at the freeze x organic curve when opt-in stopped updating; or pre-boost read x organic curve |

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
# frozen opt-in fix (plan amendment 4; sql/13 -> data/optin_staleness.psv)
python3 -m detector.v2_relabel_check  # same v2 models, corrected labels -> results/model_v2_relabel.json
python3 -m detector.train_v2 --v21 && python3 -m detector.train_v2 --v21 --final   # Instagram v2.1 on corrected labels
python3 -m detector.combine_v21 && python3 -m detector.v2_targets --v21 && python3 -m detector.evaluate_v2 --v21
python3 -m detector.optin_staleness_summary   # -> results/optin_staleness.json
# leak repair (plan amendment 5; sql/14 -> data/v2_followers.psv): same configurations, refit
python3 -m detector.train_v2 --v22 --final && python3 -m detector.v2_targets --v22 && python3 -m detector.evaluate_v2 --v22
python3 -m reconcile.analyze          # subtraction test on Instagram -> results/reconciliation.json
python3 -m reconcile.analyze_tiktok   # TikTok + curve back-test -> results/reconciliation_tiktok.json
python3 -m reports.build_report       # reports/boost_report.html
python3 -m tests.test_pipeline        # offline checks of the daily pipeline logic
python3 -m pipeline.run_daily --flags data/flags.csv --features-v2 data/v2_raw.psv   # offline daily run (flags = amended sql/05 output)
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
| `sql/13_optin_staleness.sql` | Where the Instagram opt-in count stopped updating (label and evidence fix) |
| `sql/14_v2_horizon_followers.sql` | Follower count known at day 14 / 30 / 60 (v2.2 leak repair) |
| `docs/IMPROVEMENT_PLAN.md` | v2 targets and every change, logged before the held-out evaluation |
| `reconcile/` | Subtraction test, TikTok check, organic curve, production organic estimate |
| `detector/` | Features, train / test protocol, full evaluation, scoring |
| `pipeline/`, `tests/` | Daily run scaffold and its offline tests |
| `reports/` | HTML report builder and output |
| `llm/` | LLM prompt, scorer, local-model harness |
