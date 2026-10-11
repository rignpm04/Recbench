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

---

# Parliament amendment (Oct 10, 2026, before any method read a Parliament case)

Committed with `parliament_probe.py`, `parliament_map.py` and the script changes listed below. No prediction above is
edited. This adds a real longitudinal set to v0.5: members of the UK House of Commons, their party, seat and majority
as stated over time by English Wikipedia and Wikidata, graded against the UK Parliament Members API. It is the first
set in recbench with a time dimension and a "superseded" label that is not generated. Phase 1 reads its 50 development
cases; its 400 test cases are built during phase 1, sealed, and read only in phase 2, after the freeze.

## Honesty note: what was seen before this was written

- **The probe** (`parliament_probe.py`, run twice on the 50 development members, the second time after parser fixes):
  its report -- label counts by source and field (enwiki majority: 51 valid, 66 superseded, 34 erroneous), update lags
  (median 0.7-4.8 days after a real change), the copying table (60% of the Wikidata entries made inside the window
  came from bot or tool edits, none citing Parliament's data), a list of about 30 entries whose value never occurs in
  Parliament's records, with their labels (vandalism, edits to the wrong item, typos, annotations the parser then
  misread -- since fixed), and the GO verdict. These are development members.
- **The development build** (`parliament_map.py`, 50 cases), counts only: 460 entries (median 9 per case, max 25;
  enwiki 337, wikidata 50, wikidata_bot 73); labels valid 244, superseded 117, erroneous 47, can't tell 52; 50 queries
  per field; without the bot entries 387 entries (valid 182, superseded 108, erroneous 47, can't tell 50).
- **No method -- rule, learned model or LLM -- has read a Parliament case.** No test member's history has been
  downloaded.
- **Changes to the mapping after that build, before any method run**, none of which changes an entry or a label: (1)
  the fields are declared with recbench's temporal types (party and seat `regime_cat`, majority `drift_num`) instead
  of the generic same-day types the build used (`cat`, `num_abs`), for the reason in the next item; (2) entries are
  listed day by day, as generated longitudinal records are (same-day entries in a seeded random order, v0.5 change 6),
  instead of fully shuffled; (3) the `_nobot` cases get their own case ids; (4) everything the two Parliament scripts
  keep moved from `realdata/` to `parliament_data/`, because `realdata_map.py` reads every file under `realdata/` (the
  download cache and the sealed test files must not be among them).
- **A sandbox check on synthetic data (`val`, longitudinal records only; no Parliament data), Oct 10.** Prompted by
  this set: in the v0.5 prior, unknown fields and sources (the "other" embeddings) occur only in same-day snapshot
  records. The v0.5 reconciler checkpoint trained in the sandbox at full length (honesty note above) was run on val
  records cut to their diet / medication / clinic entries, with the field and source names replaced by names it has
  never seen. Superseded recall on those entries: 0.88 with the names kept, 0.15 renamed with the fields declared
  `regime_cat`, 0.00 with `cat`; q_acc 91.9% -> 85.1% / 74.7% (newest_observed: 91.5%). Weight alone, renamed and
  declared `drift_num`: superseded recall 0.68 -> 0.19. The sandbox feature model (trained on only 1,280 stream cases)
  kept its numbers when renamed (`regime_cat`: q_acc 92.2%, macro-F1 0.898 vs 0.911 named; `cat`: 90.5%, 0.742). This
  is why the types were changed and why `reconciler_anon` (below) exists. A reconciler trained with `ANON = True`
  (same seed-1 stream, CPU) on the same check: superseded recall 0.82 renamed with `regime_cat` (0.10 with `cat`: the
  declared type still matters), q_acc 91.6%; weight alone renamed 0.57 (0.56 with its name); the four fields together
  renamed: superseded recall 0.74 and entry macro-F1 0.829 (v0.5 checkpoint: 0.21, 0.613); on the val records as
  generated, q_acc 95.1% and macro-F1 0.892 on the longitudinal ones (v0.5 checkpoint: 95.1%, 0.894), 86.5% on the
  snapshot ones (85.3%). Its full run's printout: val q_acc 0.946, entry macro-F1 0.879 (v0.5 checkpoint: 0.944,
  0.880). No setting was tuned on these numbers (`P_ANON = 0.25` was fixed before the run).
- **An independent review** (a fresh model instance, Oct 10) of the scripts and of a draft of this section. It led to
  (3), (4) and the same-day rule in (2), to the definitions under "How the predictions are read", to item 35's
  wording, and to the consequences added under the kill criteria. No prediction's threshold was changed after it; item
  31's kill criterion was set at a clear miss (3 points, 0.60) after the final sandbox run, whose superseded recall on
  renamed records (0.74) sits close to the prediction's 0.70.

## The set (`parliament_map.py`; downloads, parsers and answer key in `parliament_probe.py`)

- **Frame:** every House of Commons member active between 2015-05-07 and 2026-10-10 (1,271 members). One seeded
  shuffle (seed 20261010); a member is taken in that order when their Wikidata item is found (the single item carrying
  their Parliament id, property P10428, else the item in mySociety's people.json at a pinned commit), Parliament's
  records give a Commons seat in the frame, and they sat more than 3 days in it. **Development (`parliament_dev`)** =
  the first 50 taken (the probe's sample); **test (`parliament_test`)** = the next 400.
- **One case per member.** Fields `parliament.party`, `parliament.constituency` (`regime_cat`) and
  `parliament.majority` (`drift_num`, compared exactly). Entries: every change of a field's value in the member's
  English Wikipedia infobox (lead section) and Wikidata item, from the first revision up to the reference date, each
  dated by its edit (observed day = arrived day) and listed day by day (same-day entries in a seeded random order);
  "not an MP" values, unreadable revisions and revisions showing two current values are gaps. Sources: `enwiki`,
  `wikidata` (edits by people), `wikidata_bot` (bots and tools such as QuickStatements). A member with no entry or no
  answerable field gives no case (the build prints how many), so a file can hold fewer than 400 cases.
- **Answer key and labels:** Parliament's records (party, seat, majority of the election that began the seat term) on
  the reference date -- 2026-10-10 for sitting members, else the member's last Commons day -- and on each entry's day;
  recbench's rule (wrong on its day -> erroneous; right then and at the reference date -> valid; right then, changed
  since -> superseded). An entry is "can't tell" (in `undecidable_ids`, left out of every per-entry metric, as the
  generator's coin flips are) when Parliament has no single clear value on its day or on the reference date, or when
  moving it by one day would change its label. A field gets a query when it has an entry and a clear answer.
- **`_nobot` variants** (`parliament_dev_nobot`, `parliament_test_nobot`; case ids `parliament_nobot-<member>`): the
  same members without `wikidata_bot` entries (bulk edits carry no references, so they may copy Parliament's own
  data). A field whose only entries came from bots has no query there.

## Script changes in this commit

- `KNOWN_SPLITS` (both copies of `recbench_common.py`), `REAL_SPLITS` (`train_reconciler.py`), `EVAL_FILES`
  (`feature_baseline.py`): the four Parliament files.
- `llm_baseline.py`: `BATCH_RUNS_DEV` adds DeepSeek zero-shot on `parliament_dev`; `BATCH_RUNS_FINAL` adds the three
  forms on `parliament_test` and on `parliament_test_nobot` (first 200 cases each, `MAX_CASES`). Same prompt (it
  already states each field's tolerance and the label rules), parser and models. Estimated cost: about $15 for Claude
  through the Batch API and $2 for DeepSeek.
- `train_reconciler.py`: **`ANON = True` trains `reconciler_anon`**, identical to the reconciler except that a share
  `P_ANON = 0.25` of its training records is encoded without field and source identities (every field and source gets
  the "other" embedding; local ids, types and features unchanged). Same 160,000 training cases, seed 1, fixed
  settings, last epoch. With `ANON = False` the training stream and the model are exactly v0.5's (checked bit for bit
  in the sandbox, twice). The reconciler of items 15-26 is unchanged; `reconciler_anon` is an added row.
- `score.py`: `reconciler_anon` joins the pairwise comparisons; the full-split table ("all N cases") also prints the
  per-label precision / recall.

## Protocol

- **Phase 1:** `parliament_map.py` writes `parliament_dev` (+ `_nobot`) next to the scripts first, then downloads the
  test members and seals `parliament_test` (+ `_nobot`) in `parliament_data/cases/`, printing their size and SHA-256
  fingerprints but no labels. It refuses to run while any file of the test split sits next to the scripts. Every
  non-LLM method runs on `parliament_dev` as on the other development splits; `reconciler_anon` is trained (seed 1,
  with the leak test); DeepSeek zero-shot reads `parliament_dev`. Nothing in phase 1 is a result. A mapping bug found
  on `parliament_dev` may be fixed before the freeze: the fix, the dev counts before and after, and the rebuilt test
  files' new fingerprints go into `v0.5: frozen`.
- **Freeze:** `v0.5: frozen` lists the fingerprints of the sealed test files.
- **Phase 2:** `parliament_map.py` with `INSTALL_TEST = True` copies the sealed files next to the scripts (no rebuild,
  no download) and prints their fingerprints, which must equal the frozen ones. Every frozen method runs on them
  (learned models from their checkpoints; the seed-1 reconciler and `reconciler_anon` with the leak test); the LLM
  forms read the first 200 cases of each file (`BATCH_RUNS_FINAL`). Temperatures are the `val` ones; no Parliament
  data is used for tuning or calibration.
- **Caveats stated with the results:** the case text names parties and constituencies, so an LLM may answer from what
  it learned in pre-training (no control is run); Wikipedia's and Wikidata's histories are public and may be in any
  model's pre-training data; supersession concentrates in members who sat through several elections.

## How the predictions are read

From `score.py` after `calibrate.py`, on `parliament_test` (every case in the file) unless stated. Methods that cover
every case are read from the full-split table ("all N cases") and its per-label block; any comparison with an LLM form
uses the common-cases table (the LLM's 200 cases). "Classical methods" = the seven of `baselines.py` (newest_observed,
newest_arrival, source_priority, majority_vote, time_decayed_vote, dawid_skene, truthfinder); "learned models" =
`reconciler_anon` and `feat_hgb` (seed 1 each); "the reconciler" = v0.5's seed-1 `reconciler`. Macro-F1 is score.py's
macro-F1 over the classes present; a "point" is a percentage point of q_acc. A prediction with several clauses is
scored clause by clause.

## Predictions (blind)

29. **Leak test (requirement).** The reconciler and `reconciler_anon` pass item 15's leak test on `parliament_test`
    and `parliament_test_nobot`.
30. **The gap.** The reconciler's superseded recall is below 0.50, and its q_acc is at least 3 points below
    `newest_observed`'s: it reads these records as snapshots.
31. **The fix.** `reconciler_anon` is within 1.5 points of `newest_observed` on q_acc (either way) and at least 3
    points above the reconciler; its superseded recall is >= 0.70; its entry macro-F1 exceeds the reconciler's by >=
    0.10. On `test_hard` and `test_heldout` it is within 0.5 points of the reconciler's q_acc and within 0.02 of its
    entry macro-F1 (hiding names costs nothing on the records it was built for).
32. **Feature model.** `feat_hgb` is within 1.5 points of `newest_observed` on q_acc with superseded recall >= 0.70,
    and `reconciler_anon` minus `feat_hgb` is within 1.5 points on q_acc and within 0.05 on entry macro-F1 (item 19's
    parity, on real records).
33. **Rules.** `newest_observed` reaches q_acc >= 90%; `majority_vote` is at least 2 points below it (old values pile
    up entries across sources and reverts; on Flight, voting is the strong rule); every classical method's erroneous
    recall is below 0.20 (item 3, on real records).
34. **Real errors.** At least one learned model has erroneous recall >= 0.25. On the 200 LLM cases: every LLM form's
    q_acc is within 3 points of `newest_observed`'s; at least one Claude form has erroneous recall >= 0.40 and above
    both learned models'; at least one Claude form has an entry macro-F1 at least as high as both learned models'
    (vandalism, edits to the wrong member and typos are easier to see with language and world knowledge than with
    structure).
35. **Bots do not carry the results.** For each classical method, each learned model, the reconciler and each LLM
    form, q_acc on `parliament_test_nobot` is within 2 points of its `parliament_test` value (same tables and case
    sets as above; the `_nobot` file lacks only the queries whose every entry came from a bot).

## Kill criteria and consequences (Parliament)

- Item 29 fails: stop. No reconciler number on Parliament stands until the leak is found.
- Item 31 fails clearly (`reconciler_anon` more than 3 points below `newest_observed` on q_acc, or superseded recall
  below 0.60): hiding names in training is not enough for real dated records. The paper says the reconciler does not
  transfer to them yet, and v0.6's generator must produce dated records with unfamiliar fields and sources. A milder
  miss is reported as such. Its last clause fails (`reconciler_anon` more than 0.5 points below the reconciler on
  `test_hard` or `test_heldout`, or 0.02 on macro-F1): hiding names has a cost in distribution; both rows are reported
  with that trade-off.
- Item 30 fails (the reconciler within 3 points of `newest_observed` with superseded recall >= 0.50): the synthetic
  diagnosis did not carry over to real records. Both rows are reported and the paper says so.
- Item 32 holds: the feature model reads new fields and sources without new code, which corrects the reason given
  under "Kill criteria (v0.5)" for item 19 ("what the feature model cannot do without new code (new fields and sources
  ...)"). The reconciler's case on real records then rests on calibration, entry labels and cost, not on transfer, and
  the paper says so.
- Item 34's first clause fails (no learned model reaches erroneous recall 0.25): the generator's error processes do
  not describe real errors (vandalism and its revert, edits to the wrong record). The entry-level error claims are
  synthetic-only, and v0.6 adds such processes.
- The LLM form with the highest entry macro-F1 beats the learned model with the highest entry macro-F1 by >= 0.10 (on
  the 200 cases): "semantics beat structure" on real records (v0.1 kill criterion). The paper proposes the cascade (a
  learned model for answers and calibration, an LLM pass on the entries it flags).
