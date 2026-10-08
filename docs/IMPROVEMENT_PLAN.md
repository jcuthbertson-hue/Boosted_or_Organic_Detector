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

## Amendment 1 (2026-10-08, after the data pull, before any v2 model was trained)

The v2 data pull (`sql/11`) shows that about 11,000 of the 41,484 campaign posts have no public read in their first
60 days (tracking starts late or never). No model on public data can score them. C4 is therefore measured on posts
that are tracked early: **>= 85% of posts at least 14 days old that have a public read by day 7 get a score.**
Untracked posts are reported separately as a data gap, not a model result.

Threshold rule (fixed now): on train out-of-fold scores, take the threshold with the highest F1 among thresholds whose
out-of-fold precision is at least 0.90; if none reaches 0.90, take the max-F1 threshold.

## Amendment 2 (2026-10-08, during train-only development, before any held-out evaluation)

- Feature sets compared on train CV: `base` (public curve, jump, like and engagement features up to day H) and
  `base+creator` (adds the post's day-H views vs the median of the same creator's earlier posts; public data only).
  Model family, settings and feature set are all picked by highest train CV PR AUC.
- Calibration rule: if the picked model's train out-of-fold ECE is above 0.05, wrap it in isotonic calibration fitted
  with creator-grouped 5-fold CV on train, and pick the threshold on the calibrated out-of-fold scores.

## Amendment 3 (2026-10-08, after round-1 train-only development, before any held-out evaluation)

Round 1 (train CV, 96 fits): TikTok meets C1 on train CV at day 30 and day 14 (material recall 0.95 / 0.96 at precision
>= 0.95). Instagram does not: material recall 0.86 (day 30) and 0.80 (day 14) at precision 0.90.

Train-only error analysis (Instagram, day 30, out-of-fold):
- The 107 missed material boosts look organic on every public signal: reach vs followers 0.28 (organic 0.13, caught
  boosts 3.26), likes per view 0.020 (organic 0.020, caught boosts 0.003). Against the same creator's earlier posts,
  likes per view do not drop (+0.06 log10), but a 58% paid share (their median) should halve them.
- Independent check with SocAPI play counts: 12 of 34 missed opt-in-only boosts show > 25% non-organic plays (caught:
  192 of 334; organic: 21 of 1,034). So some misses are real boosts that public data cannot see by day 30, and some
  labels are doubtful.
- 17 of 23 ad-linked boosts with < 20% paid share by day 30 start on days 15-28: the window closes before the lift.
- False alarms are real organic posts: 2 of 38 with SocAPI data show paid plays.
- No change tried on train CV moved Instagram material recall past 0.87: likes per new view during the biggest jump,
  creator norms (likes per view, early share, reach), more sample weight on material boosts, or the lowest threshold
  with precision >= 0.90.

Changes (all decided on train CV only):
1. Feature set `base+creator+jump+cnorm` (sql/12: likes per new view during the biggest view jump, jump age, views at
   days 25 / 28; creator norms) joins the comparison. Pick rule unchanged: highest train CV PR AUC.
2. A day-60 model (labels: paid evidence with first spend in [publish - 3, publish + 58] or Instagram opt-in / public
   < 0.80 by day 60; organic rules as lab30, with opt-in by day 60). It is to catch boosts that start after day 21.
   It is exploratory: C1-C6 stay on the day-30 and day-14 models. Its test result is reported apart. In the daily table
   the longest horizon a post has wins (60, then 30, then 14).
3. The pre-registered targets and material definition do not change. On the train-CV evidence above, Instagram C1 is
   expected to fail; it is reported as measured.

## Amendment 4 (2026-10-08, after the v2 held-out evaluation; label rule chosen on train only)

Finding (train posts only, `sql/13_optin_staleness.sql`): the Instagram opt-in count often stops updating while public
views keep growing, so opt-in / public falls with no paid views. 34% of missed opt-in-only "boosts" in train had a flat
opt-in series in days 0-30 (caught boosts 2.5%, organic 2.5%). Rule tested on train opt-in-only positives (day 30):
opt-in frozen >= 7 days at the end of the window while public grew >= 5%: 168 posts, SocAPI shows paid on 29 of 100
(not frozen: 187 of 340); the v2 model catches 46% of them (not frozen: 90%). Results were the same for 3-10 days and
5-10% growth, so the rule is not tuned to one cut.

Label correction (Instagram only; TikTok labels do not use opt-in):
- Stale opt-in at horizon H = the opt-in value has not changed for >= 7 days at the last read up to day H, while public
  views grew >= 5% over those days.
- For stale posts, the opt-in ratio is taken at the last read where opt-in was still updating:
  o_valid = opt-in at the freeze / public at the freeze = o_H x (1 + public growth since the freeze).
- P = paid evidence inside the window, or o_valid < 0.80. A gap that appears only after opt-in froze gives no label.
  Material uses o_valid <= 0.60. Organic (N) rules do not change.

How it is used, in this order, and reported as such:
1. Re-measure: the v2 models do not change; locked test and fresh labels are corrected. This is a third look at the
   locked test and a second look at the fresh posts, with no model choice made on them.
2. Retrain (v2.1): Instagram candidates re-run on corrected train labels with train-only CV; the same pick rules.
   Its locked-test result is a third look and is labelled so. New posts (published after 2026-09-24) are the next clean test.
3. Production: `MEASURED_OPTIN_GAP` and the opt-in organic estimate must use the last live opt-in read (sql/05 change).

## Amendment 5 (2026-10-08, leak repair after an independent leakage audit)

Found by the audit (train-only checks): (1) `followers` is the median over reads on days 0-60 (sql/11), so the day-14 and
day-30 models see follower counts from after their window (features log_followers, log_vtf, creator_rel_vtf; 4 of the 6
production models). Dropping the three features entirely lowers train-CV PR AUC by only 0.002-0.008, so the leak is
small, but it is a leak. (2) Day-30 eligibility uses the last read up to day 60, not day 30.

Repair, decided before any re-evaluation:
- Followers at horizon H = median of reads on days 0..H (`sql/14_v2_horizon_followers.sql`); day-30 eligibility uses the
  last read up to day 30.
- Same model configurations as picked before (algorithm, settings, feature set: TikTok from v2 development, Instagram
  from v2.1 development). Each model is refit on train; calibration rule and threshold rule unchanged (train out-of-fold).
- Run name v2.2 (new files; v2 and v2.1 results stay as they are). Evaluated on the locked test (fourth look) and fresh
  posts (third look); both before and after are reported.
- Not changed: posts whose paid evidence starts after the model window have no label and are not evaluated. In the daily
  table those posts get their status from the evidence, never from the model, so the evaluation matches where the model
  is used.
