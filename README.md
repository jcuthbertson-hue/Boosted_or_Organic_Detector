# Boosted or Organic Detector

Finds which creator posts were boosted with paid ads, and separates their organic views from paid views.
Built for the BI team's paid-vs-organic research (Instagram first, then TikTok; YouTube checked).

> This repository is public. It holds code, SQL and aggregate test metrics only.
> Row-level data, client names, spend and the trained models are kept out by `.gitignore`.

## Bottom line

1. **Use evidence first, the model second.** Most boosted posts can be flagged with facts, not guesses:
   - **TikTok:** a Spark Ad carries the organic video id (`TIKTOK_ITEM_ID`). Join it to the post. This is exact.
   - **Instagram (opt-in creators):** opt-in views are organic only. If opt-in views are below 80% of public views on the same day, the post was boosted.
     Same-day opt-in ÷ public views: median 0.06 on ad-linked boosted posts (164 posts) vs 0.97 on posts with no ad link (1,258 posts).
   - **Instagram (SocAPI):** SocAPI total plays minus Instagram plays minus Facebook cross-post plays = paid plays served on Facebook.
   - **Meta ads in general:** the ad's own Instagram link is a **new media object**. 0 of 41,487 Meta ad media ids match any of the
     warehouse's tracked posts. The only ad-to-post link is the ad-name taxonomy (`handle ~ SHORTCODE`), and that is often missing.
2. **For posts with no evidence, a model trained on public data works well on TikTok and well enough on Instagram.**
   It was tested on a locked set of later posts (published 2026-07-01 to 2026-09-09) that the model never saw.
3. **Hand rules and LLMs are weaker.** The rule "views > followers and engagement < 1%" misses about a quarter of
   TikTok boosts and more than half of Instagram boosts. GPT-5, Claude Sonnet 4.5 and Llama 3.1 70B rank TikTok posts as well as the
   model, but flag too many organic posts; on Instagram they are significantly worse.

### How the view numbers fit together (boosted posts, ads finished before our latest read)

| Source | Instagram | TikTok |
|---|---|---|
| Public (Nimble) views | organic + paid plays served **on Instagram** | organic + paid plays |
| Opt-in (private) views | **organic only** (moves 1% of the public lift after a boost) | same as public (includes paid) |
| SocAPI total plays | organic + paid on Instagram + **paid on Facebook** + FB cross-post | n/a |
| Paid ad table plays | split by placement (instagram / facebook) | one number |
| Public lift after boost ÷ paid plays (same app) | median 1.15 (IQR 1.09-1.52), 159 posts | median 1.17 (IQR 1.08-1.47), 512 posts |
| (Public - opt-in) ÷ paid Instagram plays | median 1.14 (IQR 1.08-1.24), 119 opt-in posts | - |
| (SocAPI total - IG - FB cross-post) ÷ paid Facebook plays | median 1.08 (IQR 1.05-1.13), 85 posts | - |

So for a boosted Instagram Reel: **organic = opt-in views**, or if there is no opt-in, **organic ≈ public - 1.14 × paid Instagram-placement plays**.
For TikTok, opt-in does not help; use the last public read before the first spend day (low) and public - paid plays (high).
YouTube: the warehouse has no ad-to-video link (0 matches), so the math cannot be checked yet.

## Detector results (locked test set, scored once)

Labels: TikTok boosted = Spark Ad with spend on that exact video (first spend within 28 days of publish).
Instagram boosted = ad link with spend, or a same-day opt-in gap (< 0.80). Organic = no ad link, no paid date, and
(TikTok) the client's TikTok paid runs through our ad accounts / (Instagram) opt-in matches public (≥ 0.90).

| | TikTok | Instagram |
|---|---|---|
| Train / test posts | 1,161 / 738 | 2,575 / 698 |
| Boosted in test | 130 | 193 |
| Selected model (chosen on train only) | random forest | gradient boosting |
| Test ROC AUC (95% CI) | **0.964** (0.940-0.986) | **0.888** (0.855-0.917) |
| Test precision / recall (threshold from train) | 0.94 / 0.92 | 0.92 / 0.66 |
| Rule "views > followers and ER < 1%": precision / recall | 0.97 / 0.74 | 0.74 / 0.42 |
| Creators never seen in train, ROC AUC | 0.956 | 0.919 |
| Overfit check: train in-sample → train CV → test AUC | 0.998 → 0.987 → 0.964 | 0.964 → 0.925 → 0.888 |
| Weighted recall | missed TikTok boosts had median spend $3 (detected: $12,298) | detected posts hold 99.7% of paid views (opt-in measured) |

LLM baseline, same 300 locked-test posts (150 per platform), same public numbers in the prompt, threshold 0.5:

| Detector | Instagram AUC | TikTok AUC | Instagram precision / recall | TikTok precision / recall |
|---|---|---|---|---|
| ML model | **0.910** | 0.957 | **0.93 / 0.68** | **0.89 / 0.91** |
| GPT-5 (Snowflake Cortex) | 0.823 | 0.959 | 0.52 / 0.83 | 0.63 / 0.91 |
| Claude Sonnet 4.5 (Cortex) | 0.790 | 0.816 | 0.53 / 0.76 | 0.55 / 0.65 |
| Llama 3.1 70B (Cortex) | 0.785 | 0.961 | 0.30 / 0.98 | 0.31 / 1.00 |

Paired bootstrap (2,000 draws): the ML model is significantly better than every LLM on Instagram; on TikTok, GPT-5 and Llama
tie with it on ranking (AUC difference not significant) but are poorly calibrated. Full tables: `results/`.

## Validity (what we did to keep the model honest)

- **No paid information in features.** Features use only public views, likes, comments, shares and followers from the first 30 days.
  No feature reads the paid activity date, any boosted / pre-boost flag, ad names, ad spend, ad dates, opt-in views or SocAPI plays
  (`detector/features.py` asserts this). Those fields are used only to build labels.
- **Same window for every post.** All features use days 0-30, and every post needs at least 28 days of data, so a longer tracking
  window cannot leak the label.
- **Locked test set.** Split by publish date: train before 2026-07-01, test after. The test set was scored once, after all choices.
- **Choices made on train only.** Model family and settings (13 candidates) were picked with 5-fold cross-validation grouped by
  creator; the decision threshold was picked on train out-of-fold scores.
- **Overfitting check.** Train in-sample vs cross-validated vs test AUC is reported for each platform (table above).
- **Uncertainty.** 95% bootstrap intervals on all test metrics; paired bootstrap for model-vs-LLM comparisons.
- **Robustness slices.** Client-holdout cross-validation (TikTok 0.980, Instagram 0.913 AUC on train), creators unseen in train,
  and Instagram ad-link positives only (15 test positives: AUC 0.887; small sample).
- **Spot checks.** Most test "false positives" look like real boosts that the labels missed (for example, 100K views on a 1,860-follower
  account, 0.1% engagement, 99.6% of views after day 7). So reported precision is a floor, not a ceiling.

Known limits:
- Instagram labels lean on opt-in creators (882 of 1,045 boosted labels come from the opt-in gap). The model is trained mostly on
  opt-in creators and then applied to all creators.
- Organic labels are "silver": TikTok organic means no VN ad link, not proof that no one else boosted the post.
- Image and carousel posts with no public views cannot be scored.
- The 1.14 factor (public lift vs paid plays) is an empirical median, not a platform rule.

## How to run

```bash
pip install -r requirements.txt
# 1. Export the feature table from Snowflake (sql/02_detector_features_and_labels.sql) to data/detector_dataset_v2.psv
#    (pipe-separated, columns as in detector/features.py COLS)
python3 -m detector.train_eval        # train, select, test once; writes results/detector_report.json and models/
python3 -m detector.score             # PREDICTED tier for posts with no deterministic evidence
python3 -m llm.score_llm              # LLM vs ML on the same locked-test posts (needs results/llm/*.psv)
python3 -m llm.export_prompts         # harness for a local model (Decider 2B/4B etc.)
python3 -m llm.run_local --base-url http://localhost:1234/v1 --model decider-4b
```

| File | Purpose |
|---|---|
| `sql/01_ad_to_post_link_coverage.sql` | How many posts the ad tables can link (TikTok item id, Meta taxonomy, Meta permalink test) |
| `sql/02_detector_features_and_labels.sql` | Training / test table: 30-day public features + labels |
| `sql/03_view_reconciliation.sql` | Public vs opt-in vs SocAPI vs paid plays for boosted posts |
| `sql/04_llm_cortex_eval.sql` | LLM baseline in Snowflake Cortex |
| `sql/05_boost_flags.sql` | Production flags: evidence tier + organic view estimate per post |
| `detector/` | Features, training / test protocol, scoring |
| `llm/` | LLM prompt, scorer, local-model harness |
