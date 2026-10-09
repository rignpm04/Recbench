# Results — recbench v0.1 (baselines, feature baseline, LLM baseline)

Run dates: Oct 3, 2026. Generator and baselines as committed in `pre-registration v0.1`. All numbers are from
`score.py`; the full long-format table is `results.csv` (regenerable). Seven rule / truth-discovery methods,
one supervised feature baseline (`feat_hgb`), one LLM baseline (`llm_deepseek`: DeepSeek `deepseek-chat`,
billed as `deepseek-flash`; temperature 0, JSON mode).

LLM runs: hard, 200 cases | tokens in 509,256 (25,472 cache hits) out 604,809 | cost $0.44 (DeepSeek usage
export, Oct 3) | failures 0. Heldout, first 200 of 500 cases | cost: **Heldout, first 200 of 500 cases | tokens in 433,827 (25,600 cache hits) out 504,395 | cost $0.36 (DeepSeek usage export, Oct 3) | failures 0.** | failures 0.

Scoring note: since v0.2 of `score.py`, every method in a split is scored on the cases all methods covered
(`COMMON_CASES_ONLY = True`). On `hard` that is all 200 cases. On `heldout` it is the first 200 of 500, which
are a slightly harder draw than the full split (feat_hgb 90.1% on them vs 91.2% on all 500; newest_observed
89.4% vs 90.7%). Full-split numbers for the non-LLM methods are kept below for reference.

## Hard split (200 cases, 1,530 decidable queries)

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
on queries that are undecidable by construction (lower is better; pure rules report 1.0).

### Accuracy by conflict type, hard

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
regime fields (diet/medication/clinic) feat_hgb 92.6%, newest_obs 91.2%, llm 91.2%; immutable fields 97–100% for all.

### Per-assertion validity, hard (precision / recall)

| method | valid | superseded | erroneous | AUC(superseded) |
|---|---|---|---|---|
| feat_hgb | 0.89 / 0.96 | 0.92 / 0.86 | **0.98 / 0.83** | **0.963** |
| llm_deepseek | 0.93 / 0.59 | 0.66 / 0.86 | 0.43 / 0.80 | 0.801 |
| newest_observed | 0.92 / 0.92 | 0.77 / 0.93 | 0.53 / 0.02 | 0.872 |
| truthfinder | 0.89 / 0.90 | 0.75 / 0.88 | 0.38 / 0.04 | 0.871 |

## Heldout split (shifted generator settings) — the 200 cases all methods covered

| method | q_acc | 95% CI | ECE | Brier | wrong-OW@0.9 | cov@0.9 | sel@80 | undec conf | macro-F1 | AUC(err) |
|---|---|---|---|---|---|---|---|---|---|---|
| llm_deepseek | **90.4%** | 89.0–91.9 | **0.021** | 0.084 | 6.7% | 69.8% | 92.6% | 0.631 | 0.687 | 0.881 |
| feat_hgb | 90.1% | 88.9–91.5 | 0.036 | **0.080** | **4.4%** | 72.3% | **94.9%** | 0.610 | **0.846** | **0.983** |
| newest_observed | 89.4% | 87.9–90.9 | 0.106 | 0.106 | 10.6% | 100% | 89.5% | 1.000 | 0.618 | 0.514 |
| newest_arrival | 88.5% | 87.1–90.0 | 0.115 | 0.115 | 11.5% | 100% | 88.7% | 1.000 | 0.619 | 0.525 |
| dawid_skene | 87.8% | 86.3–89.4 | 0.105 | 0.106 | 9.8% | 95.5% | 90.7% | 0.510 | 0.612 | 0.747 |
| truthfinder | 87.8% | 86.3–89.5 | 0.098 | 0.104 | 9.7% | 95.0% | 90.7% | **0.460** | 0.612 | 0.747 |
| time_decayed_vote | 87.3% | 85.8–89.0 | 0.055 | 0.100 | 6.8% | 73.1% | 92.7% | 0.456 | 0.610 | 0.805 |
| source_priority | 81.4% | 79.8–83.2 | 0.186 | 0.186 | 18.6% | 100% | 81.5% | 1.000 | 0.567 | 0.550 |
| majority_vote | 76.2% | 74.2–78.2 | 0.081 | 0.144 | 6.8% | 58.3% | 83.8% | 0.601 | 0.454 | 0.887 |

### Accuracy by conflict type, heldout (common 200)

| conflict | n | feat_hgb | llm_deepseek | newest_obs | time_decayed | truthfinder |
|---|---|---|---|---|---|---|
| unit error | 113 | 72.6% | **75.2%** | 69.9% | 69.0% | 69.9% |
| typo | 84 | 73.8% | **75.0%** | 66.7% | 69.0% | 70.2% |
| OCR digit | 26 | 65.4% | 69.2% | 69.2% | **73.1%** | **73.1%** |
| stale recall | 99 | **76.8%** | 73.7% | 70.7% | 74.7% | 72.7% |
| stale re-import | 478 | 89.3% | **89.7%** | 89.5% | 85.6% | 87.9% |
| correction | 38 | 73.7% | **76.3%** | 68.4% | 68.4% | 73.7% |
| contamination | 27 | **88.9%** | 81.5% | 77.8% | **88.9%** | 77.8% |
| wrong field | 119 | 80.7% | **81.5%** | 75.6% | 77.3% | 67.2% |
| injection | 12 | 58.3% | **66.7%** | 50.0% | 58.3% | 58.3% |

By field type (heldout, common 200): weight feat_hgb 73.9%, llm 74.9%, newest_obs 70.9%; regime fields
feat_hgb 87.1%, llm 87.4%, newest_obs 86.4%; vaccine dates 89.4% for every method; immutable fields 97–100%.

### Per-assertion validity, heldout (common 200; precision / recall)

| method | valid | superseded | erroneous | AUC(superseded) |
|---|---|---|---|---|
| feat_hgb | 0.89 / 0.95 | 0.92 / 0.84 | **0.77 / 0.71** | **0.955** |
| llm_deepseek | 0.93 / 0.66 | 0.71 / 0.89 | 0.38 / 0.73 | 0.821 |
| newest_observed | 0.91 / 0.94 | 0.85 / 0.90 | 0.50 / 0.03 | 0.890 |

Full-split heldout numbers (all 500 cases, non-LLM methods): feat_hgb 91.2% (ECE 0.024, wrong-OW 3.4%,
macro-F1 0.858); newest_observed 90.7%; newest_arrival 89.9%; dawid_skene 89.2%; truthfinder 89.1%;
time_decayed_vote 88.1%; source_priority 82.6%; majority_vote 77.4%.

### Shift: hard → heldout (same method, both on 200 cases)

| method | hard | heldout | change |
|---|---|---|---|
| feat_hgb | 94.2% | 90.1% | −4.1 |
| llm_deepseek | 93.6% | 90.4% | −3.2 |
| newest_observed | 92.2% | 89.4% | −2.8 |
| truthfinder | 89.9% | 87.8% | −2.1 |
| time_decayed_vote | 89.7% | 87.3% | −2.4 |

The shifted prior costs the learned and LLM methods more than it costs the rules. Erroneous recall for feat_hgb
drops from 0.83 to 0.71 under shift; the LLM's over-flagging (valid recall 0.59–0.66, erroneous precision
0.38–0.43) is the same on both splits.

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
| 5 (blind) | LLM within ±3 pts of newest_observed; ≥3 pts worse on weight on hard; ECE > 0.10; erroneous recall > 0.20 | **mostly failed**: +1.4 pts overall, equal on weight (82.0%), ECE 0.021–0.041. Recall > 0.20 held (0.73–0.80). |
| 6 | feature baseline beats rules on calibration and error detection; not every rule on weight | **held** as rewritten (source_priority 85.0% vs 82.0% on weight, hard) |

Correction to an earlier draft of this file: the LLM's 89.4% on vaccine dates was read as a weakness; on the
common 200 heldout cases every method scores 89.4% there. It was a subset effect, not an LLM effect.

## What this means

- On the aggregate number everything competent lands at 90–94%, because immutable fields dominate. The
  informative differences are in the conflict-type rows, calibration, per-entry labeling, shift, and cost.
- The LLM is a strong current-value reader, as good as the feature model on both splits and the best single
  method on the semantic conflicts (typos, corrections, wrong field, injections under shift). It is a poor entry
  labeler: it flags a third to 41% of valid entries as not valid, and more than half of what it calls erroneous
  isn't. Its wrong-overwrite rate at 0.9 confidence is 1.5–2.4× the feature model's, it is the most confident
  method on undecidable cases, and it costs seconds and about $0.002 per record rather than milliseconds.
- The feature model owns per-entry precision and the overwrite safety number, but it is schema-specific
  (features hand-written for these eight fields) and loses 4 points plus a third of its error recall under a
  shifted prior.
- Neither is robust to the shift; the rules are, because they encode nothing.
- The bar for a learned reconciler: the feature model's per-entry precision and wrong-overwrite rate, the LLM's
  accuracy on semantic conflicts, less degradation under shift than either, no per-field feature engineering,
  and millisecond cost.

## Next

1. Public real-data sets (Stock, Flight, Book) mapped into the case format.
2. Hub environment wrapper.
3. The reconciler. Predictions for it are items 7–8 in PREDICTIONS.md and remain blind.
