# Pre-registered predictions — recbench (v0.1 items 1–8, v0.4 items 9–14)

Commit this file (and the generator at its current knobs) BEFORE running `llm_baseline.py` or training any model.
The commit date is the timestamp. Edit nothing after results exist; add a `RESULTS.md` next to it instead.

Honesty note: items 1–4 were observed on a 300-case development run (knobs slightly lower than the committed
ones) while the generator was being debugged, and item 6 was run on Oct 3, 2026 before this file was committed.
They are recorded as expectations, not as blind predictions. Items 5, 7 and 8 are blind.

## Rules and truth discovery (expectations, seen in development)

1. `newest_observed` ≥ `newest_arrival` on every split; the gap is widest on `hard` (stale re-imports).
2. The three pure rules report confidence 1.0 everywhere, so `undecC` = 1.0 and `wrongOW` = 1 − accuracy:
   confidently wrong by construction.
3. Every one of the seven baselines has per-assertion **erroneous recall under 0.20**. None of them detects
   errors; they relabel older non-matching entries as "superseded".
4. `source_priority` is the best rule on `unit` / `typo` / `stale_recall` conflicts (vet beats owner) and the
   worst on regime fields, with no advantage on `ocr_digit` (its favored source is the one making that error).

## LLM full-history baseline (blind)

5. DeepSeek reading the whole history: `q_acc` within ±3 points of `newest_observed` on `train`, and at least
   3 points worse on `drift_num` (weight) on `hard`, because of unit conversion and typo arithmetic.
   Verbalized confidence: `ECE` > 0.10. `answered_rate` < 100% from JSON failures. Erroneous recall > 0.20
   (it is the first method that will catch some errors).

## Feature baseline (observed Oct 3, 2026, before commit)

6. Gradient-boosted trees on hand-made features per candidate value and per assertion (`feature_baseline.py`):
   on `hard`, q_acc 94.2% vs 92.2% for newest-observed, ECE 0.019, wrong-overwrite@0.9 1.8%, erroneous recall
   0.83; on `heldout` (shifted prior), q_acc 91.2% vs 90.7%, erroneous recall 0.74. It does NOT beat every rule
   on `drift_num` on `hard` (source_priority 85.0% vs 82.0%): vet records dominate weight there.
   This is now the bar the reconciler has to clear, not the rules.

## The reconciler (blind)

7. On `heldout`: `q_acc` > 91.2% (the feature baseline), `ECE` < 0.03, erroneous recall > 0.80 (vs 0.74),
   `wrongOW` < 2%, `undecC` < 0.6 — and, unlike the feature baseline, with no per-field feature engineering, so
   the same model runs on a schema it was not written for.
8. First failure: field types absent from training (a new field class in a shift test) and histories longer than
   anything trained on. Second failure: injected instructions, if the training prior under-samples them.

## Kill criteria

- If item 6 holds and item 7 does not beat it by a meaningful margin on `heldout` and on the real sets,
  the finding is "feature engineering suffices"; publish that.
- If the LLM baseline wins on free-text-style conflicts (corrections, injections) by a wide margin,
  the finding is "semantics beat structure there"; design the cascade.

---

# v0.4 additions (generator v2, tuned baselines, temperature scaling, LLM prompt variants, leak controls)

Commit this section BEFORE running the v0.4 pipeline on the committed generator. Honesty note: the v0.4 scripts
were smoke-tested in a sandbox on the classical baselines and the feature model before this was written, so
items 10 and 12 are expectations informed by that; items 9, 11, 13 and 14 are blind (no reconciler, LLM or
calibration run existed when they were written). Generator v2 changes every split (benign notes, bad corrections,
a snapshot regime in train/val, a new `val` and `snapshot` split), so v0.3 numbers are not directly comparable.

9. **Temperature scaling (calibrate.py), blind.** After one temperature per method fitted on `val`: every method
   with a non-constant confidence has hard-split ECE ≤ 0.05; the reconciler reaches ECE < 0.03 and
   wrong-overwrite@0.9 < 4% on `heldout` (it missed both raw at v0.3: 0.075 / 8.0%). q_acc and macro-F1 do not
   change (scaling is monotone). The pure rules become constant-confidence methods at their val accuracy.
10. **Tuned classical baselines (tune.py), seen in development.** Equal-budget tuning on `val` gains ≤ 2 points of
    q_acc on `hard` for any classical method; the tuned truth-discovery methods land within 1 point of
    `newest_observed` on `hard`; their erroneous recall stays < 0.20 (the label heuristic, not the parameters,
    limits them).
11. **LLM prompt variants, blind.** Few-shot (2 worked examples) lands within ±2 points of zero-shot on `hard`.
    CoT + self-consistency (5 samples) gains ≤ 2 points of q_acc, has lower raw ECE than zero-shot (agreement is
    a better confidence than a verbalized number), and costs ≥ 5×. After scaling, neither LLM variant reaches the
    feature model's wrong-overwrite@0.9 on `hard`.
12. **Generator v2 flags removed, blind for the learned models.** With benign notes and bad corrections in the
    data, the feature model's accuracy on injection-tagged queries on `hard` falls ≥ 5 points below its v0.3
    value (84.0%); the reconciler's falls by less. Both keep erroneous recall > 0.75 on `hard`.
13. **Snapshot regime, blind.** The reconciler trained on the v2 prior (15% snapshot cases) lands within 2 points
    of majority vote on the `snapshot` split with erroneous recall > 0.5 there (v0.3 zero-shot on real snapshot
    sets: ~0), and its zero-shot Stock accuracy rises from 83.1% to ≥ 88%; Flight stays ≥ 88%.
14. **Permutation control, blind.** A reconciler trained with `PERMUTE_LABELS = True` (same settings, shuffled
    targets) scores ≥ 10 points below `newest_observed` on `hard`, entry macro-F1 < 0.45, AUC(erroneous) < 0.60.
    If it does not, the pipeline leaks and no v0.4 result stands.

## v0.4 kill criteria

- If item 9 fails for the reconciler (ECE ≥ 0.03 or wrong-OW ≥ 4% on heldout after scaling), calibration by
  construction is not a selling point; the paper reports the feature model as the calibrated reference.
- If item 13 fails on Stock (< 88%), the snapshot regime did not transfer; widen the prior again (more sources,
  longer records) before the scaling run, and say so.
- If item 14 fails, stop and audit the encoder before anything else.

---

---

# v0.5 additions (generator v3; candidate and calibration fixes; equal training data; stated grading rules; sealed test splits)

Commit this section, with the v0.5 code, BEFORE any v0.5 run. Edit nothing after results exist. (An earlier draft of a
v0.5 section, written on the evening of Oct 9 before the audit below, was never committed; this replaces it.)

## Honesty note: what was seen before this was written

- **v0.4 results, studied in detail.** The Oct 9 audit (RESULTS.md v0.4, "Corrections and post-hoc analyses") looked at
  v0.4's predictions on `hard` and `heldout` query by query, including re-scoring the reconciler's hard predictions
  with a newest-reading answer (94.9%). So `hard` and `heldout` are development splits now. **Every v0.5 claim rests on
  the sealed test splits** (`test_hard` 1,000 cases, `test_heldout` 1,000, `test_snapshot` 500), which `gen.py` writes
  only when `MAKE_TEST_SPLITS = True`, after every method is frozen (see "Protocol" below). LLM rows use their first
  200 cases.
- **Flight.** `flight_diag.py` read the Flight days ≤ 2011-12-15 (Oct 9) and generator v3's snapshot changes were
  designed from it. `flight_dev` is development data. `flight_test` was only ever scored inside v0.4's all-days
  aggregate; it is the Flight test here, but it is not independent of `flight_dev` (the same flights, sources and
  feeds recur every day). Stock and Book were scored in v0.3 and v0.4 and the v2 snapshot regime was introduced
  because they trailed; they are development data too. A transfer claim needs a public set no version has touched.
- **Sandbox runs of the v0.5 code, on `train` and `val` only:** generation (counts below), `tune.py`,
  `baselines.py`, a feature model trained on 1,280 stream cases, reconcilers trained on 640 to 9,600 cases (incl. the
  leak test and both permutation controls) and two at full length (next item), `calibrate.py`, `score.py`, the LLM
  script against a fake API, and `realdata_map.py` on synthetic Flight / Stock files. No v0.5 method was run on
  `hard`, `heldout`, `snapshot`, a public set or a sealed test split.
- **Two independent code reviews (Oct 9-10) and two first drafts dropped before any run.** Weight readings that no
  longer chain into one cluster: on `val` they split the votes of a drifting weight (majority vote's weight accuracy
  79.7% -> 76.2%) and gave 48% of weight queries two or more correct candidates, for +0.2 points of attainable
  accuracy. A "none of these" option trained toward the first matching candidate only: with two correct candidates it
  splits the probability between them. Every other review finding is fixed below.
- **The reconciler's answer head was chosen on `val`.** Four versions were compared after 9,600 training cases; two
  were then trained at full length in the sandbox (CPU; the seed-1 stream of 160,000 cases, every other setting as
  below) and scored on `val` only: one sigmoid probability per candidate, and the softmax with a "none of these"
  option trained on every right answer (item 2). A rule written down before those two runs picked the head: the
  higher `val` q_acc if ahead by >= 0.5 points; else the lower share of unanswerable `val` queries answered at >= 0.9
  after the `val` temperature; if those are within 5 points, the higher entry macro-F1. On `val` (4,215 decidable
  queries, 131 unanswerable): sigmoid 94.66%, T 1.03, 51% (67 / 131); softmax + none 94.40%, T 1.03, 46% (60 / 131),
  so the softmax + none head (q_acc within 0.5; 5.3 points fewer confident unanswerable answers). For reference, the
  v0.4 checkpoint (generator v2, 160,000 cases), given the v0.5 candidate values: 94.54%, T 1.82, 80%. So the
  reconciler's `val` temperature (the first clause of item 17) was seen before this commit: that clause is an
  expectation, not a blind prediction. No prediction's numbers were changed after these runs.

## What changed (all applied to every method it concerns)

1. **Candidate values (every method that groups readings).** A chosen cluster's stated value is the mean of its most
   recent readings (`cluster_answer`; v0.4: the mean of all its readings, old and new -- a weight cluster chains
   across a drift and could span 11% on a 4% tolerance, so its mean could be wrong while its newest reading was
   right). Cluster membership is unchanged. Same-day data (snapshot, public sets): unchanged. On `train` / `val` the
   best possible weight accuracy for a clustering method rises from 92.0% / 89.9% to 95.0% / 94.4% (any single
   reading: 97.0% / 95.7%).
2. **Reconciler: "none of these" and every right answer.** The query head's softmax over the candidate values gets one
   more option, a learned "none of these", and is trained to maximise the total probability of the right answers:
   every candidate whose stated value matches the truth counts (13% / 15% of weight queries on `train` / `val` have two
   or more), and "none" is the target when no candidate is right (the truth changed and nobody reported it). v0.4
   trained toward the first matching candidate only and skipped the queries with none. The stated value is the most
   probable candidate; its confidence is the total probability of the candidates whose value matches it. Chosen on
   `val` (honesty note). Architecture, features and training budget otherwise unchanged (0.62M parameters, 8 epochs x
   20,000 fresh cases).
3. **Equal training data.** The feature model trains on exactly the reconciler's 160,000 training cases (v0.4: the
   2,000 of `cases_train.jsonl`); its grid is evaluated on the first 10,000 of them, the chosen settings fitted on
   all. Its confidence is the chosen cluster's own probability (v0.4 divided it by the sum over the field's clusters
   when that sum exceeded 1, which splits it when two clusters are right).
4. **LLM prompt states the grading rules**: each field's tolerance; that an entry is valid when its value is within
   tolerance of the current value; and that "erroneous" comes first (a value that was wrong when stated is erroneous
   even if it matches the current value). Same prompt for DeepSeek and Claude, and in the Hub environment.
5. **Generator v3.** Snapshot regime: feeds (`snap_feeds`, `snap_feed_share`), per-field reliability
   (`snap_field_corr`), copy ceiling 1.0, exact correct values (`snap_exact_valid`); every range includes the v2
   behaviour. All splits: one rounding rule for displayed weights (v2 gave contamination and genuine corrections 2
   decimals), and OCR digit errors keep the reading's other decimals (v2 left a weight below 10 with one decimal,
   where every other process shows two); `undecidable_ids` lists every entry whose label is a coin flip (the
   same-day pair and earlier readings of its two values); `val` 600 cases (was 300); sealed test splits. `hard` /
   `heldout` differ from v0.4 only in the trailing decimals of 124 / 109 weight readings (and 1 / 1 labels).
6. **Public-set mapping**: placeholder gates (no digit) → null; seeded shuffle of each case's assertions (majority
   vote's tie-break was file order, and the first-listed Flight source is an airline); Flight split by day into
   `flight_dev` / `flight_test`; `*_nogold` variants of Flight and Stock without the gold-providing sources (Flight's
   gold is the three airline sites, which are also inputs; Stock's nasdaq.com source, if present); numeric queries
   carry their tolerance in words, as generated ones do.
7. **Protocol and scoring**: the leak test (strip every non-observable key and the case ids; predictions must be
   identical, unrounded); a cross-case permutation control; seeds 1-3 for both learned models; paired comparisons
   (bootstrap CI + exact McNemar) against the reconciler and between every two of the reconciler, the feature model
   and the LLM forms; answerability (ceiling, wrong-overwrite on answerable queries, confidence on unanswerable
   ones); entry-label calibration (ECE, Brier, log-loss); macro-F1 over the classes present; coin-flip entries left
   out of the per-entry metrics and of the entry temperature; full-split tables next to the LLM common-cases tables
   (a case with no entries and no queries counts as covered by every method); sel@k counts ties at the cut in
   proportion; a re-run method replaces its raw backup before calibration; an LLM case whose call got no reply is
   retried, not saved as an empty answer.

Expected (deterministic) generator counts: train 2,000 cases (309 snapshot), 102,864 assertions, 14,037 queries;
val 600 (63 snapshot), 29,875 / 4,289; heldout 500, 35,127 / 3,788; hard 200, 16,765 / 1,527; snapshot 300,
17,052 / 1,189.

## Predictions (blind for the sealed test splits; "test_hard" = all 1,000 cases unless an LLM is involved)

15. **Leak test (requirement).** With every non-observable key stripped (and case ids replaced), the seed-1
    reconciler's answers and entry probabilities are identical, unrounded, to its predictions from the full case files
    (both computed on the CPU) on every split, public and test splits included.
16. **Cross-case control** (`PERMUTE = "cross"`, 2 epochs x 5,000 cases), on `test_hard`: entry macro-F1 < 0.30;
    entry log-loss no lower than the best constant label distribution's minus 0.01; q_acc at least 10 points below
    `newest_observed`. AUC(err) is reported with no threshold (a label-free model gave 0.42-0.45 in the v0.4 audit).
    The within-case control is reported with no thresholds.
17. **"None of these."** The reconciler's query temperature fitted on `val` lies in [0.8, 1.3] (v0.4: 1.72). After
    scaling, on `test_heldout`: wrong-overwrite@0.9 <= 3.0% (v0.4 heldout: 5.0%), at most 40% of unanswerable queries
    answered at >= 0.9 (v0.4 heldout: 70%); on `test_hard` and `test_heldout`: wrong-overwrite on answerable queries
    <= 1.0%.
18. **Candidate fix.** The reconciler's weight accuracy on `test_hard` >= 85% (v0.4 hard: 80.5%), and the weight
    difference between DeepSeek-CoT and the reconciler is <= 2 points either way (v0.4 hard: CoT +5.0).
19. **Equal data.** Reconciler minus feature model (seed 1 each) on `test_hard` and on `test_heldout`: |difference|
    <= 0.5 points with McNemar p > 0.05 on both, and entry macro-F1 within 0.02 on both. (Informed by the audit: on
    hard, a feature model trained on 20,000 stream cases was +0.07 points from the reconciler.)
20. **Replicates.** Across seeds 1-3, the sd of q_acc is <= 0.3 points on `test_hard` and `test_heldout` for both
    learned models.
21. **DeepSeek with the grading rules** (first 200 cases): zero-shot valid recall >= 0.80 and entry macro-F1 >= 0.80
    on `test_hard` (v0.4 hard: 0.57 / 0.67); every DeepSeek form's entry macro-F1 stays at least 0.03 below the
    reconciler's; zero-shot on `test_snapshot` within 3 points of majority vote (v0.4 snapshot: 84.8 vs 89.6).
22. **Claude** (current Sonnet via OpenRouter; same ten runs; first 200 cases): zero-shot within +-2 points of
    DeepSeek zero-shot and CoT within +-2 of DeepSeek CoT on `test_hard`; per-token price >= 10x DeepSeek's; the best
    Claude form does not beat the reconciler by more than 1.0 point on `test_hard`; no LLM form reaches the
    reconciler's entry macro-F1 on `test_hard`.
23. **Top of the table** (`test_hard`, the 200 LLM cases): the reconciler is within 1.0 point of the best LLM form,
    and no pairwise difference among the reconciler, the feature model and the best LLM form has McNemar p < 0.05.
24. **Snapshot** (`test_snapshot`): the reconciler lands within 1.0 point of the best classical method, leads every
    classical method by >= 2 points on `feed_error` queries, and has erroneous recall > 0.80.
25. **Flight.** On `flight_test` the reconciler's gate accuracy is within 1.0 point of majority vote's and its overall
    q_acc within 1.0 point of majority's; on `flight_dev`, majority vote's gate accuracy is 1-3 points below its
    v0.4 value (the file-order tie-break is gone). Without the airline sources (`flight_test_nogold`) every method
    scores lower than on `flight_test`, and the order of the reconciler and majority vote does not change. If the
    reconciler is more than 2 points below majority on `flight_test` gates, the feed / per-field-reliability
    explanation is wrong or insufficient; report it as such.
26. **Stock / Book.** Under the shuffle every method's Stock q_acc moves < 1.0 point from v0.4 and Book strict < 3
    points; the reconciler is within 1.0 point of majority on Stock and 2.0 on Book strict; on `stock_nogold` every
    method is within 1.0 point of its Stock score.
27. **LoRA (if run).** A Qwen-class model (<= 8B) LoRA-fine-tuned on generated train cases (the number of cases is
    stated; if it is below the reconciler's 160,000 the row is reported as data-limited) beats every prompted LLM form
    on `test_hard` q_acc and macro-F1, lands within 1.5 points of the reconciler on `test_hard`, is >= 1 point below
    it on `test_heldout`, and costs >= 100x the reconciler per record.
28. **Scaling (if run).** Reconciler at ~0.6M / ~2.5M / ~10M parameters on 4x the data: the query log-loss on `val`
    falls with size and the 2.5M -> 10M step is <= 10% of it; q_acc gains <= 1.0 point from 0.6M to 10M on
    `test_hard`; the 10M model's `flight_test` q_acc is not more than 0.5 above the 2.5M model's.

## Protocol (v0.5)

- **Phase 1 (development).** After this commit: generate the development splits, map the public sets, run every
  non-LLM method (both learned models at seeds 1-3; both permutation controls; the leak test), calibrate, score, and
  DeepSeek zero-shot on `val` and `hard` only (parsing and prompt check). Nothing in phase 1 is a result.
- **Freeze.** Any change made after phase 1 is committed as `v0.5: frozen` with a list of what changed and why,
  before the test splits exist. Predictions above are not edited.
- **Phase 2 (test).** Set `MAKE_TEST_SPLITS = True` and run `gen.py` once; run every frozen method on the test splits
  (learned models from their saved checkpoints / pickles; the seed-1 reconciler with the leak test, item 15), the
  full LLM batch (`BATCH_RUNS_FINAL`) for DeepSeek and Claude, `calibrate.py`, `score.py`. The paper reports these
  tables.

## Kill criteria (v0.5)

- Item 15 fails: stop. No v0.5 result stands until the leak is found and every method is re-run.
- Item 16 fails: stop and audit the encoder before anything else.
- Item 17 fails on wrong-overwrite (`test_heldout` >= 4% after scaling, the v0.4 threshold): calibration is not
  claimed as the reconciler's advantage; the feature model is the calibrated reference.
- Item 19 holds: the paper reports that, on the schema it was written for and with the same data, engineered
  features match the transformer; the reconciler's case rests on what the feature model cannot do without new code
  (new fields and sources, the public sets) and on cost.
- Item 27: if LoRA beats the reconciler by >= 1.5 points on both `test_hard` and `test_heldout` with at least the
  reconciler's training data, the paper's claim moves to the generated data; the reconciler becomes the cheap
  baseline.

## Amendment (Oct 10, 2026, before any LLM run; committed with `llm_baseline.py` and `llm_costs.py`)

No prediction above is edited. The LLM protocol changes as follows, because of how the Claude API works today:

- **Claude through Anthropic's own API** (`PROVIDER = "anthropic"`, model `claude-sonnet-5-5`, Claude Sonnet 5.5),
  not OpenRouter. "Current Sonnet via OpenRouter" in item 22 reads "current Sonnet (Claude Sonnet 5.5) via the
  Anthropic API". Same prompts, parser, cases, forms and method names (`llm_claude*`).
- **Temperature:** Claude Sonnet 5.5 rejects any temperature other than its default 1.0, so all three Claude forms
  sample at 1.0 (zero-shot is not greedy; the five CoT samples are drawn at 1.0, not 0.7). DeepSeek keeps 0 and 0.7.
- **Thinking:** Claude Sonnet 5.5 thinks before answering by default; every Claude request sets
  `thinking: {"type": "between_tools"}`, which turns that off, so Claude, like DeepSeek's chat model, reasons only
  where the CoT prompt asks it to.
- **Message Batches API** for the Claude runs: the same requests at half price, asynchronously.
- **Output cap 16,000 tokens per reply for both providers** (was 8,000), so CoT replies are not cut off;
  `llm_costs.py` counts replies that hit the cap.
- **Phase-1 DeepSeek check** adds CoT on `val` (`BATCH_RUNS_DEV`) to measure cut-offs with the new prompt before the
  final runs.
- **Items 27 (LoRA) and 28 (scaling) are deferred to v0.6.** They are not run in v0.5. If they are run, it is in v0.6
  with their own pre-registration, each model frozen on development data before it touches a test split, and
  reported whatever it shows.
