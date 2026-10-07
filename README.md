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
5. **Where no paid record exists, the model works** (locked test of later posts): TikTok F1 0.93, ROC AUC 0.96;
   Instagram F1 0.77, ROC AUC 0.89, precision 0.92. The hand rule "views > followers and ER < 1%": F1 0.84 / 0.54.

**Caveat that can change the decision:** TikTok organic after a boost is UNVERIFIED. TikTok opt-in includes Spark Ad
views, so there is no organic truth. The curve method passes a back-test on unboosted TikTok posts only.

### Coverage today (all boosted campaign posts in BIRA)

| | Instagram (2,946 posts) | TikTok (1,093 posts) |
|---|---|---|
| Organic measured (opt-in) | 71% | 0% |
| Estimate, high or medium confidence (pre-boost read on day 7+) | 12% | 45% |
| Estimate, low confidence (read before day 7) | 5% | 31% |
| Not separable (boosted before the first read) | 12% | 24% |

## Detector results (locked test set, scored once)

| | TikTok | Instagram |
|---|---|---|
| Train / test posts | 1,161 / 738 | 2,575 / 698 |
| Selected model (chosen on train only) | random forest | gradient boosting |
| Test ROC AUC (95% CI, creator bootstrap) | **0.964** (0.938-0.985) | **0.888** (0.839-0.927) |
| Test PR AUC | 0.943 | 0.847 |
| Test precision / recall / F1 (threshold from train) | 0.94 / 0.92 / 0.93 | 0.92 / 0.66 / 0.77 |
| Test MCC | 0.91 | 0.72 |
| Confusion matrix TP / FP / FN / TN | 119 / 8 / 11 / 600 | 128 / 11 / 65 / 494 |
| Overfit check: train in-sample -> train CV -> test AUC | 0.998 -> 0.987 -> 0.964 | 0.964 -> 0.925 -> 0.888 |

All outputs (ROC, PR, calibration, score histogram, threshold sweep, operating points, gains, 20 metrics with
intervals, rules, LLMs, slices) are in `results/classification_metrics.json` and the HTML report.

## Validity (what keeps the numbers honest)

- **No paid information in model inputs.** Only public views, likes, comments, shares and followers from days 0-30.
  `detector/features.py` asserts that no paid date, boost flag, ad name, spend, opt-in or SocAPI field is a feature.
- **Locked test by time** (train before 2026-07-01). Model family (13 candidates) and threshold chosen on train only,
  with 5-fold CV grouped by creator. Re-running the code reproduces the test scores exactly.
- **Organic truth** = Instagram opt-in views (equal to public before a boost: ratio 1.001, 117 posts). The organic curve
  is built from unboosted posts only; fitted factors use leave-one-out; the curve back-test uses a creator split.
- **Uncertainty:** creator-cluster bootstrap for classifier intervals; 2,000 random 10-post groups for campaign errors.
- Details and caveats: `docs/VALIDATION.md`.

## Daily paid classification table (scaffold)

`pipeline/run_daily.py` runs `sql/05` (evidence + organic inputs), scores posts with no evidence, adds the organic
estimate (`reconcile/estimate.py`), and MERGEs into `PAID_CLASSIFICATION__POST` (`sql/09`). The logic is tested offline
(`python3 -m tests.test_pipeline`). The Snowflake read/write path is NOT tested yet: it needs a service account and a
schema we may write to.

## How to run

```bash
pip install -r requirements.txt
# exports (Snowflake -> data/, git-ignored): sql/02 -> data/detector_dataset_v2.psv, sql/06 -> data/recon_panel_ig_full.psv,
# sql/08 -> data/andrew_gold.csv
python3 -m detector.train_eval        # train, select, test once; models/ + results/detector_report.json
python3 -m detector.evaluate          # every classification output -> results/classification_metrics.json
python3 -m reconcile.analyze          # Tom's math on Instagram -> results/reconciliation.json
python3 -m reconcile.analyze_tiktok   # TikTok + curve back-test -> results/reconciliation_tiktok.json
python3 -m reports.build_report       # reports/boost_report.html
python3 -m tests.test_pipeline        # offline checks of the daily pipeline logic
python3 -m pipeline.run_daily --flags data/flags.csv   # offline daily run (flags = sql/05 output)
```

| File | Purpose |
|---|---|
| `sql/01_ad_to_post_link_coverage.sql` | How many posts the ad tables can link |
| `sql/02_detector_features_and_labels.sql` | Training / test table: 30-day public features + labels |
| `sql/03_view_reconciliation.sql` | Phase 1: public vs opt-in vs SocAPI vs paid plays |
| `sql/04_llm_cortex_eval.sql` | LLM baseline in Snowflake Cortex |
| `sql/05_boost_flags.sql` | Daily input: evidence tier (post-ID tag first) + organic-estimate inputs |
| `sql/06_reconciliation_panel.sql` | Post x day panel for Tom's math (public, opt-in, SocAPI, paid by placement and metric) |
| `sql/07_organic_curve.sql` | Organic growth curve from unboosted posts |
| `sql/08_andrew_gold_posts.sql` | The paid team's tagged posts and whether our link rules find them |
| `sql/09_paid_classification_table.sql` | Table DDL + MERGE for the daily table (not executed) |
| `reconcile/` | Tom's math, TikTok check, organic curve, production organic estimate |
| `detector/` | Features, train / test protocol, full evaluation, scoring |
| `pipeline/`, `tests/` | Daily run scaffold and its offline tests |
| `reports/` | HTML report builder and output |
| `llm/` | LLM prompt, scorer, local-model harness |
