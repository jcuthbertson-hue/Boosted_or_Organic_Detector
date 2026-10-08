# Model v2: what "predicts correctly" means (written before any v2 experiment)

Date: 2026-10-08. Baseline = model v1 (locked test, posts published 2026-07-01 to 2026-09-09).

## Baseline (v1, locked test)

| | TikTok | Instagram |
|---|---|---|
| F1 / precision / recall | 0.93 / 0.94 / 0.92 | 0.77 / 0.92 / 0.66 |
| Posts the model can score | about 65% of all posts (needs 28 days and 5 public reads in days 0-30) | |

Error analysis on TRAIN out-of-fold scores only (Instagram): boosts that add up to 40% of public views are caught
9-29% of the time; boosts with 40-95% paid are caught 79-93%; over 95% paid, 99%. Boosts that start on days 21-28 are
caught 41% of the time (the 30-day window closes before the lift shows).

## Targets (all must hold; measured once on held-out data)

| # | Target | Why |
|---|---|---|
| C1 | **Material boosts**: recall >= 0.90 at precision >= 0.90, each platform. Material = Instagram paid share >= 40% of public views (opt-in measured), any Instagram ad-linked or tagged boost, any TikTok ad-linked or tagged boost. | These are the boosts that change organic totals. |
| C2 | All boosts: F1 Instagram >= 0.80 (v1 0.77), TikTok >= 0.93 (keep). | No loss on the full label set. |
| C3 | Calibration error (ECE) <= 0.05 per platform. | A score of 0.8 should mean about 80% paid. |
| C4 | Coverage: >= 85% of posts at least 14 days old get a score, and the day-14 model's F1 is within 0.05 of the day-30 model on the same test posts. | v1 scores only about 65% of posts and only after day 28. |
| C5 | Fresh holdout (posts published 2026-09-10 to 2026-09-24, never used before): C1 holds within its 95% interval. | Proof on data no one has looked at. |
| C6 | Every tagged paid post whose boost starts inside the model window is flagged. | Gold check. |

## Rules that keep it honest

- No paid field is a model input (paid date, boost flags, ad names, spend, opt-in, SocAPI). Asserted in code.
- Time split unchanged: train = published before 2026-07-01. All feature, model and threshold choices use train
  cross-validation grouped by creator. The v1 locked test is used once more, for the final v2 model only
  (second look; stated in the report). The fresh holdout is used once.
- Thresholds are set on train out-of-fold scores.
- Every test result is reported, including targets that are missed.
