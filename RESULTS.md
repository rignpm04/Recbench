# Results — recbench v0.1 (baselines, feature baseline, LLM baseline)

Run date: Oct 3, 2026. Generator and baselines as committed in `pre-registration v0.1`. All numbers are from
`score.py`; the full long-format table is `results.csv` (regenerable). Seven rule / truth-discovery methods,
one supervised feature baseline (`feat_hgb`), one LLM baseline (`llm_deepseek`, DeepSeek `deepseek-chat`,
temperature 0, JSON mode, 200 `hard` cases).

LLM run log (from `llm_baseline.py`): **200 cases | tokens in 509,256 (25,472 cache hits) out 604,809  | cost $0.43 (from DeepSeek usage dashboard, Oct 3) | failures 0**

## Headline, hard split (200 cases, 1,530 decidable queries)

| method | q_acc | 95% CI | ECE | Brier | wrong-OW@0.9 | cov@0.9 | sel@80 | undec conf | macro-F1 | AUC(err) |
|---|---|---|---|---|---|---|---|---|---|---|
| feat_hgb | **94.2%** | 92.9–95.2 | **0.019** | **0.054** | **1.8%** | 75.4% | **97.6%** | 0.597 | **0.905** | **0.995** |
| llm_deepseek | 93.6% | 92.5–94.5 | 0.041 | 0.062 | 4.3% | 69.1% | 95.1% | 0.648 | 0.677 | 0.877 |
| newest_observed | 92.2% | 91.0–93.2 | 0.078 | 0.078 | 7.8% | 100% | 92.0% | 1.000 | 0.602 | 0.511 |
| newest_arrival | 91.0% | 89.6–92.1 | 0.090 | 0.090 | 9.0% | 100% | 91.1% | 1.000 | 0.595 | 0.516 |
| dawid_skene | 90.2% | 88.7–91.6 | 0.075 | 0.080 | 6.9% | 93.2% | 93.5% | 0.489 | 0.595 | 0.800 |
| truthfinder | 89.9% | 88.6–91.2 | 0.068 | 0.078 | 6.1% | 91.2% | 93.5% | **0.440** | 0.594 | 0.800 |
| time_decayed_vote | 89.7% | 88.2–91.1 | 0.044 | 0.080 | 2.9% | 65.8% | 95.6% | 0.468 | 0.599 | 0.836 |
| source_priority | 88.8% | 87.4–90.3 | 0.112 | 0.112 | 11.2% | 100% | 89.0% | 1.000 | 0.584 | 0.537 |
| majority_vote | 77.1% | 75.4–78.8 | 0.052 | 0.125 | 2.3% | 54.1% | 85.5% | 0.552 | 0.455 | 0.907 |

q_acc = current-value accuracy over decidable queries. wrong-OW@0.9 = of answers given with confidence ≥ 0.9,
the fraction that are wrong. sel@80 = accuracy on the 80% most confident answers. undec conf = mean confidence
on queries that are undecidable by construction (lower is better; rules report 1.0).

### Accuracy by conflict type, hard split

| conflict | n | feat_hgb | llm_deepseek | newest_obs | source_prio | time_decayed | truthfinder |
|---|---|---|---|---|---|---|---|
| unit error | 188 | 83.0% | 81.4% | 71.3% | **85.1%** | 78.7% | 72.9% |
| typo | 163 | 83.4% | 82.2% | 73.6% | **85.3%** | 78.5% | 74.2% |
| OCR digit | 58 | 81.0% | **84.5%** | 74.1% | 70.7% | 77.6% | 67.2% |
| stale recall | 79 | 81.0% | 82.8% | 71.8% | **84.0%** | 76.7% | 74.2% |
| stale re-import | 771 | **94.0%** | 93.0% | 91.2% | 87.2% | 88.8% | 89.2% |
| correction | 120 | 80.0% | **85.0%** | 75.0% | **85.0%** | 75.8% | 72.5% |
| contamination | 53 | 86.8% | 86.8% | 75.5% | **88.7%** | 81.1% | 79.2% |
| wrong field | 118 | **84.7%** | 83.9% | 78.0% | 69.5% | 75.4% | 72.9% |
| injection | 25 | 84.0% | 80.0% | 72.0% | **92.0%** | 84.0% | 72.0% |

By field type (hard): weight (drifting number) feat_hgb 82.0%, llm 82.0%, source_priority 85.0%, newest_obs 72.5%;
regime fields (diet/medication/clinic) feat_hgb 92.6%, newest_obs 91.2%, llm 91.2%; immutable fields ~97–100% for all.

### Per-assertion validity, hard split (precision / recall)

| method | valid | superseded | erroneous | AUC(superseded) |
|---|---|---|---|---|
| feat_hgb | 0.89 / 0.96 | 0.92 / 0.86 | **0.98 / 0.83** | **0.963** |
| llm_deepseek | 0.93 / 0.59 | 0.66 / 0.86 | 0.43 / 0.80 | 0.801 |
| newest_observed | 0.92 / 0.92 | 0.77 / 0.93 | 0.53 / 0.02 | 0.872 |
| truthfinder | 0.89 / 0.90 | 0.75 / 0.88 | 0.38 / 0.04 | 0.871 |

## Heldout split (500 cases, shifted generator settings; LLM not yet run)

| method | q_acc | 95% CI | ECE | wrong-OW@0.9 | sel@80 | macro-F1 | erroneous P/R |
|---|---|---|---|---|---|---|---|
| feat_hgb | **91.2%** | 90.3–92.1 | **0.024** | **3.4%** | **95.5%** | **0.858** | 0.77 / 0.74 |
| newest_observed | 90.7% | 89.7–91.6 | 0.093 | 9.3% | 90.4% | 0.624 | 0.47 / 0.03 |
| newest_arrival | 89.9% | 88.9–90.9 | 0.101 | 10.1% | 89.6% | 0.624 | 0.23 / 0.06 |
| dawid_skene | 89.2% | 88.2–90.2 | 0.090 | 8.4% | 91.6% | 0.628 | 0.41 / 0.07 |
| truthfinder | 89.1% | 88.1–90.2 | 0.085 | 8.3% | 91.6% | 0.627 | 0.40 / 0.07 |
| time_decayed_vote | 88.1% | 87.0–89.2 | 0.046 | 5.6% | 93.7% | 0.625 | 0.36 / 0.11 |
| source_priority | 82.6% | 81.3–83.9 | 0.174 | 17.4% | 82.5% | 0.574 | 0.13 / 0.16 |
| majority_vote | 77.4% | 76.2–78.7 | 0.068 | 5.2% | 85.3% | 0.465 | 0.10 / 0.22 |

Under shift the feature model keeps its lead on calibration and error detection but its margin on q_acc over
newest-observed shrinks to 0.5 points and erroneous recall drops from 0.83 to 0.74.

## Train split (2,000 cases; feat_hgb numbers are in-sample)

feat_hgb 95.2% (ECE 0.022, wrong-OW 0.6%, macro-F1 0.936); newest_observed 93.9%; newest_arrival 93.5%;
truthfinder 92.8%; dawid_skene 92.7%; time_decayed_vote 92.7%; source_priority 89.8%; majority_vote 84.5%.

## Prediction scorecard (see PREDICTIONS.md)

| # | prediction | result |
|---|---|---|
| 1 | newest_observed ≥ newest_arrival on every split, gap widest on hard | **held** (train +0.4, heldout +0.8, hard +1.2) |
| 2 | pure rules report confidence 1.0; undec conf = 1.0, wrong-OW = 1 − accuracy | **held** |
| 3 | all seven baselines: erroneous recall < 0.20 | **held** (max 0.19, majority_vote) |
| 4 | source_priority best rule on unit/typo/stale-recall, worst on regime fields, no edge on OCR | **held** |
| 5 (blind) | LLM within ±3 pts of newest_observed; ≥3 pts worse on weight on hard; ECE > 0.10; erroneous recall > 0.20 | **mostly failed**: +1.4 pts overall, equal on weight (82.0%), ECE 0.041. Recall > 0.20 held (0.80). |
| 6 | feature baseline beats rules on calibration and error detection; not every rule on weight | **held** as rewritten (source_priority 85.0% vs 82.0% on weight, hard) |

## What this means

- On the aggregate number everything competent lands at 92–94% on `hard`, because immutable fields dominate.
  The informative differences are in the conflict-type rows, calibration, per-entry labeling, and cost.
- The LLM is a strong current-value reader and a poor entry labeler: it flags 41% of valid entries as not valid
  and a third of what it calls erroneous isn't. Its wrong-overwrite rate at 0.9 confidence is 2.4× the feature
  model's, and it is the most confident method on undecidable cases.
- The LLM wins the semantic cases (OCR digits, corrections), which is the cascade scenario anticipated in the
  kill criteria.
- The bar for a learned reconciler is therefore: match `feat_hgb` on calibration and error precision with no
  per-field feature engineering, hold up under the shifted split, and beat the LLM on cost and on the
  undecidable cases.

## Next

1. LLM baseline on 200 `heldout` cases (the shift comparison the table above is missing).
2. Public real-data sets (Stock, Flight, Book) mapped into the case format.
3. Hub environment wrapper.
4. The reconciler. Predictions for it are items 7–8 in PREDICTIONS.md and remain blind.
