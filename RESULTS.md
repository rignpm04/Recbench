# Results — recbench v0.4 (generator v2, tuned baselines, temperature scaling, three LLM forms, reconciler v1)

**Corrected Oct 9, 2026.** Two errors in the first version of this file are fixed, and post-hoc analyses are
added; see "Corrections and post-hoc analyses" at the end. Corrected sentences are marked "(corrected)".

Run dates: Oct 8–9, 2026, on the generator and scripts committed as `v0.4: pre-registration (PREDICTIONS 9-14) +
generator v2, tuning, calibration, LLM variants` (committed before any run). All numbers are from `score.py`; the
long-format table is `results.csv` (regenerable). Every method is tuned on `val` only (`tune.py`,
`feature_baseline.py`; at most 40 configurations each; the reconciler is untuned) and every method's confidences
are temperature-scaled with one temperature per method fitted on `val` (`calibrate.py`). Temperature scaling is
monotone, so accuracies and labels are the raw ones; ECE, Brier, wrong-overwrite and coverage are post-scaling.

**v0.3 numbers are not comparable to these.** Generator v2 changed every split (benign notes, bad corrections,
a `val` split, a snapshot regime in train/val and a `snapshot` split; see README "Protocol"). The reconciler was
retrained from scratch on the v2 prior (same settings as v0.3: 0.62M parameters, 8 epochs × 20,000 fresh cases,
47 minutes on a MacBook Pro MPS). The two shortcuts the v1 generator leaked — "has a note ⇒ injection" and
"is corrected ⇒ erroneous" — are gone.

Methods: three rules (`newest_observed`, `newest_arrival`, `source_priority` with tuned order); four classical
truth-discovery methods (`majority_vote`, `time_decayed_vote`, `dawid_skene`, `truthfinder`, tuned); the feature
model `feat_hgb` (gradient-boosted trees on 36 hand-made features, grid-tuned on val, no isotonic calibration this
time so every method gets the same post-hoc treatment); DeepSeek `deepseek-chat` in three forms — `llm_deepseek`
(zero-shot, JSON mode, temperature 0), `llm_deepseek_fs` (two solved train cases in the prompt: train-00002,
train-00012), `llm_deepseek_cot` (reason in ≤ 250 words, five samples at temperature 0.7, majority vote, confidence
= agreement × stated confidence); `reconciler`; and `reconciler_permuted`, the same reconciler trained on shuffled
targets (the leak control). LLM runs cover the first 200 cases of val, hard and heldout (and zero-shot on 200
snapshot cases); `score.py` therefore scores heldout and snapshot on the 200 cases every method covered.

## Hard split (200 cases, 1,472 decidable queries, 55 undecidable)

| method | q_acc | 95% CI | ECE | Brier | wrong-OW@0.9 | cov@0.9 | sel@80 | undec conf | macro-F1 | erroneous P/R | AUC(err) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_deepseek_cot | **94.6%** | 93.4–95.7 | 0.060 | 0.062 | 3.5% | 69.3% | 96.0% | **0.377** | 0.767 | 0.82 / 0.78 | 0.971 |
| reconciler | 94.0% | 92.8–95.2 | **0.021** | **0.053** | 3.4% | 79.3% | **96.5%** | 0.555 | 0.898 | 0.92 / 0.86 | 0.994 |
| feat_hgb | 93.8% | 92.5–95.0 | 0.027 | 0.054 | **2.7%** | 80.7% | 97.3% | 0.638 | **0.913** | **0.95 / 0.90** | **0.996** |
| llm_deepseek_fs | 93.1% | 91.9–94.3 | 0.085 | 0.095 | 6.0% | 75.9% | 93.8% | 0.708 | 0.718 | 0.74 / 0.60 | 0.814 |
| llm_deepseek | 93.0% | 91.7–94.2 | 0.026 | 0.060 | 4.3% | 69.7% | 95.8% | 0.640 | 0.670 | 0.46 / 0.73 | 0.870 |
| truthfinder | 91.4% | 90.2–92.9 | 0.024 | 0.078 | 8.0% | 97.9% | 92.2% | 0.489 | 0.597 | 0.51 / 0.03 | 0.659 |
| newest_observed | 91.2% | 89.9–92.6 | 0.001 | 0.080 | 8.8% | 100% | 91.5% | 0.912 | 0.595 | 0.49 / 0.02 | 0.511 |
| dawid_skene | 91.2% | 90.0–92.7 | 0.027 | 0.077 | 8.0% | 97.9% | 92.2% | 0.490 | 0.595 | 0.48 / 0.03 | 0.659 |
| newest_arrival | 90.9% | 89.5–92.3 | 0.001 | 0.083 | 9.1% | 100% | 90.9% | 0.908 | 0.598 | 0.30 / 0.05 | 0.519 |
| time_decayed_vote | 90.8% | 89.5–92.3 | 0.047 | 0.071 | 2.8% | 65.6% | 96.4% | 0.444 | 0.600 | 0.49 / 0.06 | 0.801 |
| source_priority | 88.0% | 86.4–89.5 | 0.010 | 0.106 | – | 0% | 88.2% | 0.889 | 0.594 | 0.24 / 0.04 | 0.515 |
| majority_vote | 77.2% | 75.1–78.8 | 0.025 | 0.121 | 1.4% | 52.0% | 85.3% | 0.575 | 0.455 | 0.14 / 0.17 | 0.945 |
| reconciler_permuted (control) | 63.7% | 61.7–65.8 | 0.051 | 0.098 | 1.4% | 52.0% | 75.6% | 0.324 | 0.228 | 0.00 / 0.00 | 0.636 |

q_acc = current-value accuracy over decidable queries. wrong-OW@0.9 = of answers given with confidence ≥ 0.9, the
fraction that are wrong. sel@80 = accuracy on the 80% most confident answers. undec conf = mean confidence on
queries that are undecidable by construction (lower is better). After scaling, the three pure rules report one
constant confidence (their val accuracy, 0.889–0.912), which is why their ECE is ~0 and `source_priority` never
reaches the 0.9 threshold.

### Accuracy by conflict type, hard (queries on fields touched by that conflict)

| conflict | n | reconciler | feat_hgb | llm_cot | llm_fs | llm (zero) | newest_obs | truthfinder | time_decayed | source_prio |
|---|---|---|---|---|---|---|---|---|---|---|
| unit error | 185 | 80.0% | 80.0% | **85.9%** | 74.1% | 76.8% | 65.9% | 68.6% | 69.7% | 62.2% |
| typo | 171 | 82.5% | 82.5% | **86.0%** | 75.4% | 76.6% | 69.0% | 71.3% | 71.9% | 52.0% |
| OCR digit | 56 | **78.6%** | **78.6%** | 76.8% | 62.5% | 64.3% | 55.4% | 57.1% | 62.5% | 55.4% |
| stale recall | 152 | 79.6% | 80.3% | **86.2%** | 75.7% | 77.0% | 65.8% | 68.4% | 70.4% | 61.2% |
| stale re-import | 754 | 93.6% | 93.4% | **94.2%** | 92.0% | 91.6% | 89.3% | 89.4% | 89.7% | 85.5% |
| correction | 181 | 82.3% | 82.9% | **87.3%** | 75.7% | 78.5% | 69.6% | 71.8% | 72.4% | 65.7% |
| bad correction | 84 | **84.5%** | 83.3% | 83.3% | 75.0% | 78.6% | 65.5% | 71.4% | 73.8% | 64.3% |
| contamination | 50 | 86.0% | 82.0% | **90.0%** | 84.0% | 84.0% | 70.0% | 68.0% | 68.0% | 60.0% |
| wrong field | 114 | **92.1%** | 89.5% | 91.2% | **92.1%** | 91.2% | 84.2% | 86.0% | 87.7% | 88.6% |
| injection | 32 | 78.1% | 78.1% | 78.1% | 78.1% | 78.1% | 71.9% | 75.0% | **84.4%** | **84.4%** |
| benign note | 104 | 91.3% | 89.4% | **92.3%** | 88.5% | 89.4% | 86.5% | 85.6% | 82.7% | 85.6% |

By field type (hard): weight (drift_num) llm_cot 85.5%, feat 81.0%, reconciler 80.5%, llm 77.0%, newest_obs
67.0%; regime fields (diet / medication / clinic) llm_fs 92.8%, llm_cot 92.7%, reconciler 92.7%, feat 91.9%,
newest_obs 91.0%; vaccine dates 95.6% for every method; immutable fields 98–100% for all but `source_priority`.

### Per-assertion validity, hard (precision / recall)

| method | valid | superseded | erroneous | AUC(superseded) |
|---|---|---|---|---|
| feat_hgb | 0.90 / 0.94 | 0.91 / 0.88 | **0.95 / 0.90** | 0.964 |
| reconciler | 0.89 / 0.94 | 0.91 / 0.86 | 0.92 / 0.86 | **0.966** |
| llm_deepseek_cot | 0.93 / 0.61 | 0.65 / 0.94 | 0.82 / 0.78 | 0.891 |
| llm_deepseek_fs | 0.95 / 0.60 | 0.63 / 0.95 | 0.74 / 0.60 | 0.794 |
| llm_deepseek | 0.94 / 0.57 | 0.64 / 0.89 | 0.46 / 0.73 | 0.780 |
| newest_observed | 0.92 / 0.89 | 0.75 / 0.93 | 0.49 / 0.02 | 0.860 |
| truthfinder | 0.92 / 0.90 | 0.75 / 0.91 | 0.51 / 0.03 | 0.870 |
| time_decayed_vote | 0.86 / 0.93 | 0.77 / 0.83 | 0.49 / 0.06 | 0.712 |

## Heldout split (shifted generator settings) — the 200 cases all methods covered (1,472 decidable queries)

| method | q_acc | 95% CI | ECE | Brier | wrong-OW@0.9 | cov@0.9 | sel@80 | undec conf | macro-F1 | erroneous P/R | AUC(err) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llm_deepseek_cot | **92.3%** | 90.8–93.6 | 0.055 | 0.073 | 6.2% | 77.0% | 93.9% | **0.408** | 0.777 | 0.76 / 0.75 | 0.969 |
| reconciler | 92.1% | 90.7–93.4 | **0.032** | **0.070** | 5.5% | 81.7% | 94.7% | 0.576 | **0.858** | **0.82 / 0.78** | **0.990** |
| llm_deepseek | 92.1% | 90.7–93.3 | 0.042 | 0.072 | 6.7% | 73.3% | 93.3% | 0.662 | 0.704 | 0.49 / 0.70 | 0.880 |
| feat_hgb | 91.3% | 89.8–92.5 | 0.043 | 0.074 | **4.5%** | 80.4% | **95.5%** | 0.662 | 0.843 | 0.71 / 0.80 | 0.987 |
| newest_observed | 91.3% | 89.8–92.7 | 0.001 | 0.079 | 8.7% | 100% | 90.8% | 0.912 | 0.619 | 0.49 / 0.04 | 0.520 |
| llm_deepseek_fs | 91.1% | 89.7–92.4 | 0.099 | 0.110 | 8.5% | 79.1% | 91.5% | 0.769 | 0.747 | 0.82 / 0.59 | 0.815 |
| dawid_skene | 90.8% | 89.2–92.2 | 0.034 | 0.080 | 8.4% | 98.7% | 91.2% | 0.503 | 0.621 | 0.46 / 0.05 | 0.648 |
| truthfinder | 90.7% | 89.1–92.1 | 0.029 | 0.080 | 8.4% | 98.7% | 91.2% | 0.497 | 0.621 | 0.45 / 0.05 | 0.647 |
| newest_arrival | 90.4% | 89.0–91.8 | 0.004 | 0.087 | 9.6% | 100% | 89.7% | 0.908 | 0.616 | 0.23 / 0.06 | 0.525 |
| time_decayed_vote | 90.4% | 89.0–91.7 | 0.056 | 0.082 | 5.5% | 74.7% | 93.8% | 0.462 | 0.630 | 0.43 / 0.08 | 0.782 |
| source_priority | 88.5% | 86.6–90.0 | 0.005 | 0.102 | – | 0% | 88.3% | 0.889 | 0.631 | 0.26 / 0.10 | 0.545 |
| majority_vote | 79.1% | 76.9–81.1 | 0.030 | 0.126 | 3.7% | 55.7% | 86.4% | 0.538 | 0.480 | 0.09 / 0.19 | 0.919 |
| reconciler_permuted (control) | 65.4% | 63.1–67.1 | 0.067 | 0.109 | 3.7% | 55.7% | 77.2% | 0.330 | 0.233 | 0.00 / 0.00 | 0.580 |

Full-split heldout (all 500 cases, methods without an LLM run): reconciler 92.3% (corrected; ECE 0.023 after scaling,
wrong-OW 5.0%, macro-F1 0.865, erroneous P/R 0.82/0.78, AUC(err) 0.990); feat_hgb 92.0%; newest_observed 91.6%;
truthfinder 91.3%; dawid_skene 91.2%; newest_arrival and time_decayed_vote 90.9%; source_priority 87.8%;
majority_vote 78.1%.

### Accuracy by conflict type, heldout (common 200)

| conflict | n | reconciler | feat_hgb | llm_cot | llm_fs | llm (zero) | newest_obs | truthfinder | time_decayed |
|---|---|---|---|---|---|---|---|---|---|
| unit error | 117 | 79.5% | 76.1% | **83.8%** | 69.2% | 81.2% | 76.9% | 72.6% | 71.8% |
| typo | 92 | 72.8% | 69.6% | **76.1%** | 71.7% | 75.0% | 68.5% | 66.3% | 69.6% |
| OCR digit | 30 | **76.7%** | **76.7%** | 70.0% | 66.7% | 73.3% | 73.3% | 73.3% | 73.3% |
| stale recall | 85 | 76.5% | 74.1% | **84.7%** | 71.8% | 81.2% | 77.6% | 72.9% | 75.3% |
| stale re-import | 465 | **92.0%** | 90.8% | 91.6% | 90.5% | 91.6% | 90.3% | 89.2% | 89.2% |
| correction | 96 | 76.0% | 75.0% | **85.4%** | 75.0% | 84.4% | 79.2% | 75.0% | 76.0% |
| bad correction | 23 | 65.2% | 60.9% | **69.6%** | 56.5% | **69.6%** | 60.9% | 60.9% | 65.2% |
| contamination | 16 | **87.5%** | 81.2% | **87.5%** | 81.2% | **87.5%** | **87.5%** | 81.2% | 68.8% |
| wrong field | 109 | **78.0%** | 74.3% | 77.1% | **78.0%** | **78.0%** | 71.6% | 70.6% | 75.2% |
| injection | 4 | 75.0% | 75.0% | 75.0% | 75.0% | **100%** | 50.0% | 50.0% | 75.0% |
| benign note | 128 | 86.7% | 85.9% | 89.1% | **89.8%** | 89.1% | 87.5% | 85.9% | 82.8% |

By field type (heldout, common 200): weight llm_cot 83.8%, llm 81.3%, reconciler 80.3%, newest_obs 78.3%, feat
76.8%; regime fields reconciler 89.4%, llm_fs 89.2%, llm 89.0%, llm_cot 88.7%, feat 88.5%, newest_obs 87.9%;
vaccine dates 95.3% for every method; immutable fields 94–100%.

### Per-assertion validity, heldout (common 200; precision / recall)

| method | valid | superseded | erroneous | AUC(superseded) |
|---|---|---|---|---|
| reconciler | 0.88 / 0.95 | 0.92 / 0.82 | **0.82 / 0.78** | **0.955** |
| feat_hgb | 0.88 / 0.94 | 0.91 / 0.82 | 0.71 / 0.80 | 0.948 |
| llm_deepseek_cot | 0.91 / 0.68 | 0.70 / 0.92 | 0.76 / 0.75 | 0.900 |
| llm_deepseek_fs | 0.91 / 0.68 | 0.68 / 0.92 | 0.82 / 0.59 | 0.807 |
| llm_deepseek | 0.92 / 0.65 | 0.68 / 0.91 | 0.49 / 0.70 | 0.804 |
| newest_observed | 0.90 / 0.94 | 0.85 / 0.88 | 0.49 / 0.04 | 0.883 |

### Shift: hard → heldout (same method, 200 cases each)

| method | hard | heldout | change |
|---|---|---|---|
| reconciler | 94.0% | 92.1% | −1.9 |
| feat_hgb | 93.8% | 91.3% | −2.5 |
| llm_deepseek_cot | 94.6% | 92.3% | −2.3 |
| llm_deepseek | 93.0% | 92.1% | −0.9 |
| llm_deepseek_fs | 93.1% | 91.1% | −2.0 |
| newest_observed | 91.2% | 91.3% | +0.1 |
| truthfinder | 91.4% | 90.7% | −0.7 |
| dawid_skene | 91.2% | 90.8% | −0.4 |

The shifted prior costs the learned and LLM methods 1–2.5 points and the rules nothing; in v0.3 the learned
methods lost 3–4 points, so the v2 prior (and the removal of the two shortcuts) narrowed the gap. The reconciler's
erroneous recall holds under shift (0.86 → 0.78) and its precision stays at 0.82; the feature model's precision
falls from 0.95 to 0.71.

## Snapshot split (many anonymous sources, same day, copied errors) — 200 cases, 752 queries

| method | q_acc | 95% CI | ECE | Brier | wrong-OW@0.9 | cov@0.9 | sel@80 | macro-F1 | erroneous P/R | AUC(err) |
|---|---|---|---|---|---|---|---|---|---|---|
| feat_hgb | **89.9%** | 86.9–92.9 | **0.034** | 0.079 | 1.8% | 65.3% | 95.3% | 0.616 | 0.91 / 0.89 | 0.979 |
| majority_vote | 89.6% | 86.4–92.8 | 0.273 | 0.164 | 3.8% | 7.0% | 93.5% | 0.613 | 0.91 / 0.88 | 0.972 |
| time_decayed_vote | 89.5% | 86.4–92.5 | 0.273 | 0.165 | 3.8% | 7.0% | 93.5% | 0.615 | 0.92 / 0.88 | 0.972 |
| truthfinder | 89.5% | 86.4–92.5 | 0.413 | 0.275 | 3.8% | 7.0% | 91.2% | 0.615 | 0.92 / 0.88 | 0.850 |
| reconciler | 88.7% | 85.0–92.2 | 0.046 | **0.071** | **1.3%** | 60.6% | **96.5%** | 0.615 | 0.92 / 0.88 | **0.983** |
| dawid_skene | 88.4% | 84.8–91.7 | 0.035 | 0.086 | 7.5% | 79.9% | 92.5% | 0.608 | 0.90 / 0.87 | 0.934 |
| llm_deepseek (zero-shot) | 84.8% | 81.4–88.6 | 0.110 | 0.147 | 0.7% | 36.0% | 87.7% | 0.519 | 0.65 / 0.83 | 0.890 |
| newest_* / source_priority | 63.0% | 58.8–67.4 | 0.26–0.28 | 0.30–0.31 | 37.0% | 100% | 62.6% | 0.478 | 0.58 / 0.79 | 0.740 |
| reconciler_permuted (control) | 32.0% | 28.2–35.5 | 0.084 | 0.175 | 3.8% | 7.0% | 37.9% | 0.263 | 0.00 / 0.00 | 0.553 |

Independent errors (`source_error`, n = 570): feat 88.9%, majority 88.8%, time-decayed / truthfinder 88.6%,
reconciler 87.4%, dawid_skene 87.4%, llm 84.0%. Copied errors (`copied_error`, n = 597): feat 87.8%, majority
87.4%, time-decayed / truthfinder 87.1%, reconciler 86.1%, dawid_skene 85.6%, llm 81.2%. The recency rules are at
63% because every claim is on day 0 and they pick an arbitrary source. The vote-share confidences of majority /
time-decayed / TruthFinder are not comparable across regimes (many clusters ⇒ small shares), so one temperature
fitted on the mixed val split leaves them at ECE 0.27–0.41 here; the reconciler's and the feature model's learned
confidences carry over (ECE 0.046 / 0.034).

## Public real-data sets (zero-shot for every method; no time dimension; `superseded` never occurs)

| set | cases | reconciler | feat_hgb | majority | time-decayed | TruthFinder | Dawid–Skene | newest / source-priority |
|---|---|---|---|---|---|---|---|---|
| Stock (1% tolerance) | 400 | **93.2%** (92.5–94.0) | 93.1% | **93.2%** | 93.1% | 93.1% | **93.2%** | 74.5% |
| Flight (±10 min, exact gates) | 600 | 83.8% (82.7–84.7) | 84.5% | **85.4%** | 85.2% | 85.2% | 84.2% | 81.1% |
| Book, strict (author multiset = gold) | 100 | 76.0% (66–84) | 73.0% | **77.0%** | **77.0%** | **77.0%** | 74.0% | 43.0% |
| Book, subset (partial list counts) | 100 | 100% | 100% | 100% | 100% | 100% | 100% | 92.0% |

Calibration and error detection on the real sets (after the synthetic-val temperature; no real labels used
anywhere): Stock — reconciler ECE 0.019, wrong-OW 3.3%, erroneous P/R 0.88/0.85 (v0.3: labels collapsed to
"everything valid"); feat 0.027 / 2.9% / 0.91/0.79; majority ECE 0.163 (vote shares). Flight — reconciler ECE
0.088, wrong-OW 5.7%, erroneous P/R 0.52/0.46; every method's error detection is weak there (P/R 0.51–0.56 /
0.40–0.55). Book strict — reconciler ECE 0.088, wrong-OW 7.5%, erroneous P/R 0.80/0.81; dawid_skene is
overconfident (wrong-OW 22.3%). The v0.3 → v0.4 change for the reconciler: Stock 83.1 → 93.2, Book 71.0 → 76.0,
Flight 88.9 → 83.8. The snapshot regime in the prior carried over to Stock and Book and cost five points on
Flight, where the reconciler is now the weakest competent method.

## Train split (500 cases; feat_hgb and reconciler numbers are in-sample)

feat_hgb 94.7% (ECE 0.014, wrong-OW 1.9%, macro-F1 0.934); reconciler 94.1% (0.021, 3.7%, 0.901);
time_decayed_vote 93.1%; truthfinder 92.8%; dawid_skene 92.7%; newest_arrival 91.3%; newest_observed 91.2%;
source_priority 89.4%; majority_vote 85.3%; reconciler_permuted 69.2%. By regime (train): longitudinal queries —
feat 95.1%, reconciler 94.6%, newest_arrival 93.5%; snapshot queries (n = 299) — feat 91.0%, majority 90.6%,
time-decayed / truthfinder 90.0%, reconciler 89.6%, rules 68.6%.

## Tuning (tune.py, feature_baseline.py; chosen on val, 300 cases)

| method | default → tuned val q_acc | chosen | gain on hard (default → tuned) |
|---|---|---|---|
| source_priority | 86.7% → 88.9% | owner > email_forward > vet_pdf > note_text > extractor | 88.0% with the tuned order |
| time_decayed_vote | 93.0% → 93.9% | decay 60 days; weights vet 0.8, email 0.5, owner 1.0, extractor 0.6, note 0.3 | 89.3% → 90.8% |
| dawid_skene | 92.9% → 93.8% | window 15 days, 5 iterations, initial accuracy 0.6 | 90.0% → 91.2% |
| truthfinder | 92.7% → 93.7% | window 15 days, gamma 0.1, dampening 0.3 | 90.0% → 91.4% |
| feat_hgb | grid of 18 (val q_acc 0.950–0.954) | cluster task: 200 iters, lr 0.06, 15 leaves; assertion task: 200 iters, lr 0.03, 31 leaves | flat: tuning does not matter |

The tuner moved the owner above the vet in `source_priority` (recent owner entries beat stale vet records in this
data) and shrank the truth-discovery window to 15 days. Tuning for query accuracy lowers those two methods'
per-entry AUC(err) (0.81 → 0.66 on hard): 15-day windows make per-window clusters tiny. Their per-entry numbers
are reported as they fall out.

## Calibration (calibrate.py; one temperature per method, fitted on val by NLL)

| method | T (query) | T (entry labels) | hard ECE raw → scaled | heldout-500 ECE raw → scaled |
|---|---|---|---|---|
| reconciler | 1.72 | **1.02** | 0.034 → 0.021 | 0.054 → 0.023 |
| feat_hgb | 0.84 | 1.00 | 0.020 → 0.027 | 0.017 → 0.031 |
| llm_deepseek | 0.83 | 2.45 | 0.025 → 0.026 | 0.037 → 0.042 (common 200) |
| llm_deepseek_cot | 0.69 | 1.28 | 0.088 → 0.060 | 0.053 → 0.055 (common 200) |
| llm_deepseek_fs | 0.98 | 1.53 | 0.083 → 0.085 | 0.096 → 0.099 (common 200) |
| dawid_skene / truthfinder | 2.42 / 2.49 | 2.33 / 2.43 | 0.081 → 0.027 / 0.024 | 0.084 → 0.029 / 0.081 → 0.023 |
| time_decayed_vote / majority_vote | 1.68 / 1.64 | 3.40 / 4.05 | 0.038 → 0.047 / 0.053 → 0.025 | 0.051 → 0.048 / 0.067 → 0.029 |
| newest_observed / newest_arrival / source_priority | 2.96 / 3.02 / 3.31 | ~3 | 0.088 → 0.001 (constant confidence) | |

The reconciler's entry-label distribution needed no correction (T = 1.02): the PFN-style training produced
calibrated per-entry probabilities out of the box (so did the feature model's training: T = 1.00; see Corrections,
item 10). Its query confidence was overconfident (T = 1.72; the cause is Corrections, item 5). The feature
model was slightly underconfident raw (T = 0.84), and scaling it on val made it a little worse on hard and heldout.
DeepSeek's verbalized confidences were close to calibrated zero-shot; the CoT agreement score was *overconfident*
raw (0.088) and the few-shot confidences were the worst calibrated of any learned method.

## Leak control (reconciler_permuted)

Same model, same data stream, targets shuffled within each case; 2 epochs × 5,000 cases. Hard 63.7% (newest_observed
91.2%: 27.5 points below), entry macro-F1 0.228, erroneous recall 0.00; snapshot 32.0%, Stock 35.1%, Book 35.0%.
The 63–65% floor on the longitudinal splits is the fraction of queries with a single candidate value. AUC(err) is
0.636 on hard rather than the pre-registered < 0.60: shuffling labels *within* a case preserves the case's share of
errors, so the control can learn "messy case ⇒ more errors" without knowing which entries; a cross-case shuffle is
the stricter control and goes into v0.5. (This explanation is incomplete, and kill criterion 14 was triggered; the
encoder audit and a cross-case control are in Corrections, items 3–4.)

## LLM runs: cost, speed, and parse coverage

DeepSeek `deepseek-chat` (billed as `deepseek-flash`), 200 cases per run. Token counts are the API's usage figures
(final console lines of the batch; for the four re-runs, per-reply logs summed by `llm_costs.py`). Estimates use
$0.15 / $0.60 per million input / output tokens; actual billing ran ≈ 1.3× the estimates, and the re-runs were ~90%
prompt-cache hits (identical prompts), which DeepSeek bills at a fraction of the input rate. DeepSeek console total
for all v0.4 LLM work, Oct 8–9, 2026: **$19.64** (6,680 requests, 39.6M tokens), of which roughly $6 was the
discarded work listed in the scoring notes below; the ten reported runs estimate to $9.51 at list rates.

| run | calls | tokens in / out | est. cost | wall time | parse failures |
|---|---|---|---|---|---|
| zero-shot val | 200 | 354,799 / 383,791 | $0.28 | 19 min | 0 |
| zero-shot hard | 200 | 520,090 / 610,182 | $0.44 | 29 min | 0 |
| zero-shot heldout (re-run) | 200 | 444,767 / 511,485 | $0.37 | ~25 min | 0 |
| zero-shot snapshot | 200 | 355,635 / 376,331 | $0.28 | 19 min | 0 |
| few-shot val (re-run) | 200 | 1,086,399 / 372,791 | $0.39 | ~20 min | 0 |
| few-shot hard (re-run) | 200 | 1,251,690 / 587,145 | $0.54 | ~25 min | 0 |
| few-shot heldout (re-run) | 200 | 1,176,367 / 491,583 | $0.47 | ~25 min | 0 |
| CoT + SC val | 1,000 | 1,810,995 / 2,584,305 | $1.82 | 2.3 h | 0 |
| CoT + SC hard | 1,000 | 2,637,450 / 3,826,783 | $2.69 | 3.1 h | 1 sample |
| CoT + SC heldout | 1,000 | 2,260,835 / 3,138,117 | $2.22 | 2.7 h | 0 |

Per record at list rates: zero-shot ≈ $0.002 and ~7 s; few-shot ≈ $0.002–0.003 (the two examples add ~4,000
input tokens per call, almost all cache hits); CoT ≈ $0.009–0.013 and ~55 s. Parse coverage after the fix: every
query answered in all seven single-call runs (100%); one dropped sample in 3,000 across the CoT runs. The reconciler and the
feature model answer in milliseconds on a laptop CPU/GPU; the classical methods in microseconds.

Scoring notes (recorded as they happened):

1. **CoT truncation.** The first CoT runs used a 6,000-token output cap; 2.9% of val samples and 7.5% of hard
   samples (rising with record length; one case down to a single vote) were cut off mid-JSON and dropped. The
   runs were discarded, the cap raised to 8,000 and the reasoning limited to 250 words, and all three CoT splits
   re-run from scratch: 1 failed sample in 3,000. Cost of the discarded runs ≈ $3.5–4.
2. **Heldout backup.** `calibrate.py` was run while the zero-shot heldout run was still writing (62 of 200 cases);
   it backed up the partial file and, on the next run, rebuilt the heldout predictions from that backup, so an
   intermediate `score.py` output had heldout on 62 common cases. `calibrate.py` now merges newly appended cases
   into its backup; zero-shot heldout was re-run in full.
3. **Few-shot parser.** In the first few-shot runs 31 of 200 hard cases (and a similar share on val and heldout)
   came back with every entry labeled but the current values in a JSON shape the parser did not accept, and were
   scored as unanswered (few-shot hard 78.9%, immutable fields 84.6%). The parser now accepts the alternative shapes
   (bare values get confidence 0.5), every raw reply is logged, and the three few-shot runs were re-run: hard 93.1%,
   heldout 91.1%. Both fixes favored the baseline, not the reconciler.

## Prediction scorecard (PREDICTIONS.md items 9–14; items 1–8 were scored in v0.3)

| # | prediction | result |
|---|---|---|
| 9 (blind) | after scaling, every non-constant method ECE ≤ 0.05 on hard; reconciler heldout ECE < 0.03 and wrong-OW < 4% | **mostly held.** Hard ECE 0.021–0.047 for every method except the two LLM prompt variants (CoT 0.060, few-shot 0.085). Reconciler heldout ECE 0.023 on all 500 cases (✓) but 0.032 on the common 200 (✗ by 0.002); wrong-OW 5.0–5.5% (✗). |
| 10 | tuned classical gains ≤ 2 points on hard; tuned truth discovery within 1 point of newest_observed; erroneous recall < 0.20 | **held.** +1.2 (DS), +1.4 (TF), +1.5 (TDV); 91.2 / 91.4 / 90.8 vs 91.2; erroneous recall 0.03–0.06. |
| 11 (blind) | few-shot within ±2 of zero-shot; CoT ≤ +2 q_acc, lower raw ECE than zero-shot, ≥ 5× cost; neither LLM form reaches feat_hgb's wrong-OW | **mostly held.** Few-shot +0.1 / −1.0; CoT +1.6 / +0.2; cost 6×; wrong-OW 3.5 / 6.0 / 4.3% vs 2.7% (✓). Raw ECE: CoT 0.088 vs 0.025 zero-shot (✗) — agreement × stated confidence is *more* overconfident than a stated number. |
| 12 | feat_hgb's injection accuracy on hard ≥ 5 points below v0.3's 84.0%; reconciler drops less; both erroneous recall > 0.75 | **half held.** feat 78.1% (−5.9 ✓); reconciler 78.1% (−9.9 from 88.0%, ✗: it had leaned on the flag more); recall 0.90 / 0.86 (✓). |
| 13 (blind) | reconciler within 2 points of majority vote on snapshot with erroneous recall > 0.5; zero-shot Stock ≥ 88%; Flight ≥ 88% | **held except Flight.** 88.7 vs 89.6 (✓), recall 0.88 (✓); Stock 93.2% (✓, from 83.1); Flight 83.8% (✗, from 88.9). |
| 14 (blind) | permuted control ≥ 10 points below newest_observed on hard, macro-F1 < 0.45, AUC(err) < 0.60 | **held except AUC(err).** 27.5 points below, macro-F1 0.228, AUC(err) 0.636 (within-case shuffle keeps case-level error rates; see above; explanation incomplete, see Corrections, item 4). |

Kill criteria (corrected): two of the three v0.4 kill criteria were triggered. **Item 9** for the reconciler:
heldout wrong-overwrite after scaling 5.0% (500 cases) / 5.5% (common 200) ≥ 4%, and ECE 0.032 ≥ 0.03 on the
common 200. As pre-registered, calibration is not claimed as the reconciler's advantage and the feature model is
the calibrated reference. **Item 14**: AUC(err) 0.636 ≥ 0.60, so the encoder had to be audited before anything
else; the audit found no leak (Corrections, items 3–4). The item 13 criterion (Stock < 88%) was not triggered
(93.2%). The first version of this file said "Kill criteria: none triggered" and limited the calibration claim to
the entry labels; both statements are withdrawn.

## What this means

- On the aggregate number the top of the table is a three-way tie: DeepSeek with chain-of-thought and five votes
  (94.6% hard / 92.3% heldout), the reconciler (94.0 / 92.1) and the feature model (93.8 / 91.3), all with
  overlapping confidence intervals. The best tuned classical method is 2.5–3 points behind on hard and within a
  point of the recency rule on heldout.
- The three differ on everything else. The reconciler has no schema-specific features (the feature model has 36
  hand-written ones for these eight fields) and no per-record LLM cost (CoT is ~$0.012 and ~55 s per record;
  the reconciler is milliseconds). Per-entry labeling: feature model 0.913 macro-F1, reconciler 0.898, the LLM
  forms 0.67–0.78 (they still over-flag valid entries: valid recall 0.57–0.68). Calibration after scaling: the
  reconciler's ECE 0.021–0.032 against CoT's 0.055–0.060; its entry probabilities were calibrated before scaling.
  (Corrected reading: kill criterion 9 was triggered, so this is not claimed as the reconciler's advantage; see
  Kill criteria above and Corrections, items 5 and 10.)
- Where each wins on hard: CoT on the semantic conflicts (unit errors, typos, stale recall, corrections,
  contamination: 86–90%; these are mostly weight questions, where every clustering method is capped, see
  Corrections, item 6); the reconciler and feature model on OCR digits (78.6) and wrong-field (92.1 /
  89.5); `source_priority` and `time_decayed_vote` on injections (84.4%), where every learned method and the
  LLM sit at 78.1% now that a note no longer identifies an injection.
- Under the shifted prior the reconciler loses 1.9 points, the feature model 2.5, CoT 2.3, the rules nothing;
  the reconciler's erroneous precision holds (0.82) where the feature model's drops to 0.71.
- Zero-shot on the public sets the reconciler is at voting level on Stock and Book with working error labels,
  and below voting on Flight. The snapshot regime made the prior cover Stock-like data and cost Flight; the
  dependence on the prior is the central limitation and the real sets are where it shows.
- The bar for the next version is unchanged in kind: beat CoT on the semantic conflicts at millisecond cost, close
  the wrong-overwrite gap to the feature model (3.4–5.5% vs 2.7–4.5%), and recover Flight.

## Next (v0.5)

1. Scaling curve: reconciler at 0.6M / ~2.5M / ~10M parameters with 4× data, rented GPU; cross-case permutation
   control.
2. Fine-tuned small LLM (LoRA, Qwen-class) on the identical generated cases — the "is it the data or the model"
   comparison.
3. Claude (current Sonnet, via OpenRouter) in the same three prompt forms; the whole LLM table re-run at the
   8,000-token cap with the lenient parser.
4. Flight failure analysis before any prior change; MARC-record set and inter-cataloger agreement if the library
   data arrives.
5. Hub environment: `vf-eval` end-to-end, `prime env push`, bounty application (`recbench_env/`, generator v2).

## Corrections and post-hoc analyses (added Oct 9, 2026)

The first version of this file (commit `v0.4: results`) had the two errors corrected in items 1–2. Items 3–10 are
post-hoc: they were run after the results above were known, were not pre-registered, and are reported so the v0.5
changes can be traced to them. They use the committed `reconciler.pt` run on CPU, which reproduces the reported
predictions (hard 94.0%, macro-F1 0.898, raw ECE 0.034; heldout raw ECE 0.054), and a re-run of the feature model
that lands within 2 hard-split queries of the reported one (scikit-learn version). Numbers that were chosen after
looking at `hard` or `heldout` are marked as such and are not results.

**Corrections**

1. **Two of the three v0.4 kill criteria were triggered; this file first said none were.**
   *Item 9*: "If item 9 fails for the reconciler (ECE ≥ 0.03 or wrong-OW ≥ 4% on heldout after scaling),
   calibration by construction is not a selling point; the paper reports the feature model as the calibrated
   reference." The reconciler's heldout wrong-overwrite after scaling was 5.0% (all 500 cases) and 5.5% (common
   200), and its ECE on the common 200 was 0.032. Triggered: calibration is not claimed as the reconciler's
   advantage, the feature model is the calibrated reference, and the earlier narrowing of the claim to "entry
   labels" is withdrawn.
   *Item 14*: "If item 14 fails, stop and audit the encoder before anything else." AUC(err) was 0.636 against a
   pre-registered < 0.60. Triggered; the audit is item 3.
   *Item 13* (Stock < 88%): not triggered (93.2%).
2. **The reconciler's full-split heldout accuracy is 92.3% (3,399 of 3,682 decidable queries), not 92.1%.** The
   92.1% was the training script's internal metric, which counts a query as right only when the model picks the
   first cluster that matches the truth; `score.py` counts any matching value. All common-200 heldout figures
   (reconciler 92.1%) are `score.py`'s and are unchanged.

**Post-hoc analyses (not pre-registered)**

3. **Encoder audit (kill criterion 14).** Every key the encoder may not read was stripped from the val, hard,
   heldout and snapshot case files — labels, error types, the hidden truth, query answers and answer types, the
   decidable flag, generator knobs, snapshot source accuracies — and `reconciler.pt` was run on the stripped
   files. All 8,539 query answers and 84,698 entry predictions are bit-identical to those from the full files.
   The model's inputs do not contain the answer. v0.5 makes this test part of the protocol.
4. **The permutation controls.** The explanation in "Leak control" (the within-case control learns case-level
   error rates) is incomplete. A retrained within-case control (CPU, same settings; pooled AUC(err) 0.642 on hard
   vs 0.636 above) ranks erroneous entries above valid ones *inside* cases (mean within-case AUC 0.736 on hard,
   0.662 on heldout), which case-level rates alone cannot do. Its per-entry outputs carry tiny systematic offsets
   by field and source (about ±0.002; weight, note and corrected entries slightly higher), and a rank metric picks
   them up. A cross-case control (every training target drawn independently from the train label marginal,
   otherwise the same settings) scores AUC(err) 0.446 on hard and 0.419 on heldout, at the single-candidate floor
   on accuracy (hard 64.5%) and with macro-F1 0.228. A model with no label information gives near-constant outputs
   whose small offsets can push AUC to either side of 0.5, so AUC is not a usable null test on its own; the scrub
   test in item 3 is the leak test. Training on shuffled labels can show that scores come from learning, but it
   cannot detect an input leak: a leaked input stops predicting shuffled targets, so the control scores low either
   way.
5. **What drives the reconciler's wrong-overwrite rate.** On 3.7% of hard and 5.1% of heldout decidable queries
   no candidate value in the record matches the current truth (the truth changed and nobody reported it, or the
   right weight reading sits in a cluster whose mean is off; see item 6). The reconciler's training skips these
   queries and its query head must put all probability on the candidates, so it answers them confidently: 132 of
   its 151 wrong answers at ≥ 0.9 on heldout after scaling are on these queries (hard: 34 of 40). On queries that
   do have a right candidate its wrong-overwrite is 0.7% on heldout and 0.5% on hard (feature model 1.0% and
   0.7%), and a temperature fitted on them alone is 1.00 rather than 1.72. One temperature for all queries makes
   the answerable ones underconfident (heldout ECE on them 0.008 raw → 0.036 scaled) and leaves 132 of the 189
   unanswerable ones at ≥ 0.9. The failure behind kill criterion 9 is in the training objective, not in the
   scaling.
6. **Every clustering method's weight answer is capped.** `cluster()` grows numeric clusters by chaining (a
   weight cluster can span 11% while the tolerance is 4%) and answers with the cluster mean, which mixes old and
   new readings. On hard no clustering method — the classical baselines, the feature model, the reconciler — can
   exceed 87.5% on weight, while picking the right existing reading would reach 97.5% (heldout: 84.9% vs 92.7%).
   The conflict types where CoT leads on hard (unit error, typo, stale recall, correction) are 83–100% weight
   queries, so that comparison largely measures this cap. (Answering with the newest reading of the chosen cluster
   would put the reconciler at 94.9% on hard; that number was chosen after looking at hard and is not a result.)
7. **The LLM prompt did not state the grading rules.** It gave no tolerances and did not say that an older
   reading within tolerance of the current value counts as "valid". On hard, 19.8% of valid labels are older
   weight readings within 4% of the current weight, which the prompt's wording ("still the current value")
   invites the model to call "superseded"; the LLM rows' per-entry numbers (valid recall 0.57–0.68) may partly
   reflect this. On snapshot, true numeric values are jittered within tolerance while copied errors are exact
   copies: without the tolerance, the most common exact value is right on 40% of `num_rel` queries, against 88%
   with it.
8. **Paired comparisons.** Overlapping confidence intervals are not a test. Paired over cases (bootstrap and
   McNemar), reconciler minus feature model: +0.34 points on hard (95% CI −0.14 to +0.87; p = 0.30) and +0.30 on
   all 500 heldout cases (−0.08 to +0.73; p = 0.19). The +0.8 on the common 200 heldout cases (p = 0.017) does
   not hold on the full split. The LLM rows have not been paired-tested yet.
9. **Unequal training data.** The reconciler trained on 160,000 generated cases (8 epochs × 20,000 fresh), the
   feature model on the 2,000 cases of `cases_train.jsonl`. Retrained on 20,000 cases from the reconciler's own
   training stream (same hyperparameters), the feature model scores 93.95% on hard and 92.1% on heldout; paired
   against the reconciler on hard: +0.07 points, McNemar p = 1.0.
10. **Entry-label calibration** (`score.py` does not measure it; v0.4 temperatures applied): top-label ECE of the
    per-entry distributions, reconciler 0.008 (val) / 0.015 (hard) / 0.028 (heldout), feature model 0.010 /
    0.023 / 0.031. Both learned models are calibrated on entry labels; it does not set the reconciler apart.

Items 3–10 are addressed in the v0.5 protocol (PREDICTIONS.md, v0.5 section).
