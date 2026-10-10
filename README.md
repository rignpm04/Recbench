# recbench v0.5 — longitudinal record reconciliation benchmark

Cite all versions: [10.5281/zenodo.23205119](https://doi.org/10.5281/zenodo.23205119)

One entity. A hidden truth. The stream of assertions the app would have seen, from sources that err in different
ways. Every assertion is labeled **valid / superseded / erroneous** against the hidden truth at "now", and every
field has a current-value query. Methods are scored on accuracy, calibration, wrong-overwrite rate, selective
accuracy, and per-assertion validity.

Everything is synthetic. No app data, no keys needed except for the LLM baselines.
Python 3.9, standard library only (feature model: scikit-learn; reconciler: torch in a 3.10+ venv). Every script
runs from PyCharm's Run button with this folder as the working directory (the reconciler from the venv's Terminal).

## Run order (v0.5)

v0.5 runs in two phases. The claims rest on **sealed test splits** that are generated once, after every method is
frozen; `hard` and `heldout` are development splits (they were studied in detail during the v0.4 audit).

**Phase 1 — development (nothing here is a result)**

0. Archive every output of the previous version (`cases_*`, `preds_*`, `raw_preds_*`, `replies_*`, `results.csv`,
   `calibration.json`, `tuned_params.json`, `*.pt`, `*.pkl`) into an archive folder: old predictions carry the same
   case ids as the new cases.
1. `gen.py` → `cases_train` (2000; ~15% snapshot regime), `cases_val` (600, same settings as train, separate seed),
   `cases_heldout` (500, shifted settings), `cases_hard` (200, conflict knobs up), `cases_snapshot` (300). ~6 s.
2. `realdata_map.py` → the public sets (Stock, Flight dev / test, Book, and the `*_nogold` variants).
3. `tune.py` → equal-budget hyperparameter search for the classical baselines on `val` only → `tuned_params.json`.
4. `baselines.py` → `preds_<method>_<split>.jsonl` for 7 rule / truth-discovery methods on every split.
5. `feature_baseline.py` → gradient-boosted trees on hand-made features, trained on the reconciler's own 160,000
   training cases; its grid on `val`. Seeds 1, 2, 3 (`SEED`; 2 and 3 reuse seed 1's settings). Saves `<name>.pkl`.
6. `train_reconciler.py` (venv) → the reconciler, seeds 1, 2, 3 (`SEED`), ~50 min each; seed 1 with
   `LEAK_TEST = True`. Then the two controls: `PERMUTE = "within"` and `PERMUTE = "cross"` (~10 min each).
7. `llm_baseline.py` with `BATCH_RUNS = BATCH_RUNS_DEV` (DeepSeek zero-shot on `val` and `hard`, CoT on `val`; ~$3–5):
   parsing, prompt and cut-off check. Claude: once, `PROVIDER = "anthropic"` with `SMOKE = True` (3 train cases per
   form into `smoke_*` files, never scored; checks the key, the model id and the batch path for under $1).
8. `calibrate.py` → one temperature per method fitted on `val`; rewrites every `preds_*` (raw copies `raw_preds_*`).
9. `score.py` → the tables and `results.csv`. `flight_diag.py` (optional) → the Flight diagnostic on `flight_dev`.

**Freeze.** Any change after phase 1 is committed as `v0.5: frozen` with what changed and why, before step 10.

**Phase 2 — the sealed test splits (the results)**

10. `gen.py` with `MAKE_TEST_SPLITS = True` → `cases_test_hard` (1000), `cases_test_heldout` (1000),
    `cases_test_snapshot` (500). Other splits are rewritten identically.
11. `baselines.py`; `feature_baseline.py` with `EVAL_ONLY = True` (seeds 1-3); `train_reconciler.py` with
    `EVAL_ONLY = True` (seeds 1-3, both controls; seed 1 with `LEAK_TEST = True`, so the leak test covers the test
    splits).
12. `llm_baseline.py` with `BATCH_RUNS = BATCH_RUNS_FINAL`, once with `PROVIDER = "deepseek"` and once with
    `PROVIDER = "anthropic"` (first 200 cases of each test split, plus `val` for calibration). Claude runs through
    the Message Batches API: the script submits the batches and waits (usually under an hour; press Run again later
    if you stop it).
13. `calibrate.py`, `score.py`, `llm_costs.py`.

`score.py` scores every method in a split on the cases all methods covered (`COMMON_CASES_ONLY = True`) and, when an
LLM covers only some cases, adds a full-split table for every method that covers them all (a case with no entries and
no queries counts as covered by every method). The `val` split is never a results table (`SKIP_SPLITS`).

## Protocol (v0.5)

- **Tuning.** Every method's hyperparameters are chosen on `val` (train distribution, separate seed) with at most
  40 configurations: `tune.py` for the classical baselines, `feature_baseline.py` for the trees. The reconciler uses
  fixed settings except its answer head, chosen on `val` between two versions before the pre-registration
  (PREDICTIONS.md, v0.5 honesty note); `heldout` is printed per epoch for monitoring only and the last epoch is the
  reported model. Nothing is ever chosen on `hard`, `heldout`, `snapshot`, a test split or a public set.
- **Equal training data.** The feature model and the reconciler learn from the same 160,000 generated cases (the
  reconciler's stream: seed × 10⁶ + epoch × 10⁴ + step, 32 cases per step). v0.4 gave the feature model 2,000.
- **Candidate values.** Methods that group readings into clusters state `cluster_answer()` for the cluster they
  pick: the mean of its most recent readings (v0.4: the mean of all its readings, old and new, although a weight
  cluster chains across a drift). Cluster membership is unchanged. Same-day data is unaffected.
- **Confidence of the learned models (v0.5).** The reconciler's softmax over the candidate values has a "none of these"
  option and is trained on every right answer: every candidate whose stated value matches the truth counts, and
  "none" is the target when no candidate is right (the truth changed and nobody reported it). Its confidence is the
  total probability of the candidates whose value matches the stated one; the feature model's is the chosen
  cluster's own probability.
- **Calibration.** One post-hoc temperature per method for the query confidence and one for the label
  distribution, fitted on `val` by NLL, applied everywhere (real sets included — no real labels are used).
  Temperature scaling is monotone, so accuracy and labels are untouched.
- **LLM.** Three forms of the same model — zero-shot JSON, few-shot (2 worked train cases with gold answers),
  chain-of-thought + self-consistency (5 samples; value by majority, confidence = agreement × stated confidence) —
  for DeepSeek and for Claude (Sonnet 5.5, through Anthropic's API and its Batch API). The prompt states each field's
  tolerance and the label rules (including that "erroneous" comes first: a value wrong when stated is erroneous even
  if it matches today's). Output cap 16,000 tokens per reply. Claude Sonnet 5.5 accepts no temperature (it samples
  at 1.0 in every form; DeepSeek uses 0 and 0.7) and its up-front thinking is turned off, so it reasons only where
  the CoT prompt asks (PREDICTIONS.md, v0.5 amendment).
- **Leak test.** `train_reconciler.py` with `LEAK_TEST = True` strips every key the encoder may not read from the case
  files (labels, error types, the truth, query answers / types / decidable flags, knobs, snapshot parameters,
  `undecidable_ids`; case ids replaced by opaque ones) and checks that every prediction is identical, unrounded. The
  permutation controls (`PERMUTE = "within"` / `"cross"`) show that the scores come from learning; they cannot detect
  an input leak, which is why the leak test exists.
- **Replicates.** Seeds 1-3 for the reconciler and the feature model; `score.py` prints mean ± sd.
- **Statistics.** `score.py` compares every method with the reconciler on the same queries: q_acc difference,
  paired bootstrap 95% CI over cases, exact McNemar p; then every two of the reconciler, the feature model and the
  LLM forms. Overlapping marginal CIs are not used as a test.
- **Generator v2 flags removed.** `note_text` entries are mostly benign notes (correct at observation), so "has a note"
  does not identify an injection; owner corrections also restate right values or make them wrong (`bad_correction`).
- **Generator v3 (v0.5): the snapshot regime widened after a diagnostic, not fitted to it.** `flight_diag.py` showed
  what Flight has that v2 never produced: sources that are copies of a feed (identical rows), wrong values that are
  almost always shared, and reliability that differs by field. v3 adds `snap_feeds` / `snap_feed_share` (member
  sources show a feed's rows verbatim; `feed_error`), `snap_field_corr` (per-field accuracy), raises `snap_copy`'s
  ceiling to 1.0, and adds `snap_exact_valid` (some correct numeric readings state the exact value; with only jittered
  correct values, "the most common exact value" was a copied error far more often than in real data). Every range
  includes the v2 behaviour. Also: one rounding rule for displayed weights (v2 gave contamination and genuine
  corrections 2 decimals, and OCR'd weights below 10 one); `undecidable_ids` lists every entry whose label is a coin
  flip (an undecidable same-day pair and earlier readings of its two values), for the scorer.
- **Public-set mapping (v0.5), applied to every method.** Gate values with no digit ("Terminal") are placeholders and
  become null; assertions are shuffled within each case with a fixed seed, so file order carries no information;
  Flight is split by day into `flight_dev` (≤ 2011-12-15, read by the diagnostic) and `flight_test`; `*_nogold`
  variants of Flight and Stock drop the gold-providing sources (Flight's gold is the three airline sites, which are
  also inputs).

## Splits

- **train**: the generated cases for anything that wants a fixed file (the learned models train on the stream).
- **val** (600): train settings, separate seed. Tuning and calibration only.
- **heldout** (500): *different* generator settings — species mix, locale (lb vs kg), drift rates, source cadence,
  lags, conflict rates all shifted. Development split. Longitudinal only.
- **hard** (200): train settings with every conflict knob turned up. Development split. Longitudinal only.
- **snapshot** (300): the snapshot regime only: 5–40 anonymous sources (`src_00`…), 2–6 generic fields, every claim
  on day 0, per-case source reliability, copy rate, feeds and per-field reliability. Labels are valid / erroneous.
- **test_hard** (1000), **test_heldout** (1000), **test_snapshot** (500): the sealed test splits (same settings as
  hard / heldout / snapshot, new seeds), generated once in phase 2.

The knobs are the `KNOBS` dict at the top of `gen.py`. Change them there; everything downstream follows.

## What is in a case

```
case_id, split, regime ("longitudinal" | "snapshot"), now_day, undecidable_ids
longitudinal:  truth: species, sex, birth_day, weight_knots [[day, kg]...], diet/medication/vet_clinic regimes
               [[start_day, value]...], rabies_events [day...]
snapshot:      truth_values {field: value}, field_types {field: [type, tolerance]}, source_accuracy (per source,
               emitter mean), field_accuracy {source: {field: acc}}, snapshot_params (mean_acc, copy_rate,
               field_corr, exact_valid, n_feeds, feed_share, feed_of {source: feed})
assertions:    id, field, value, unit, observed_day, arrived_day, source, extractor_conf, error_type, text,
               corrects (id), duplicate_of (id), label
queries:       field, answer, answer_type, decidable, tolerance
conflict_types: which conflict processes fired in this case
```

Methods may read only the observable assertion keys (`id, field, value, unit, observed_day, arrived_day, source,
extractor_conf, text, corrects, duplicate_of`), `now_day`, the queried field names and `field_types`.

Pet fields: `species`, `sex` (immutable categorical), `birth_day` (immutable, ±60 days), `weight_kg` (drifting
number, ±4%), `diet`, `medication`, `vet_clinic` (regime-switching categorical), `rabies_day` (dated event, ±7 days).

Sources and their error models: `owner` (typed; unit mix-ups, typos, stale recall, birth year off by one),
`vet_pdf` (accurate, arrives days late, OCR digit errors), `extractor` (chat-derived; has a confidence; sometimes
files the right value under the wrong field), `email_forward` (old vet records re-sent later; duplicates),
`note_text` (benign notes and, rarely, injected instructions). Plus: cross-pet contamination from a household's
other pet, owner corrections that reference an earlier entry (right, redundant, or wrong), and deliberately
undecidable same-day contradictions. Snapshot sources: `source_error`, `copied_error`, `feed_error`.

Two timestamps matter: `observed_day` is when the fact was true; `arrived_day` is when the app got it.
Re-imports and backfill deliver old facts late. "Newest arrival wins" is the current app failure.

Labels: an assertion is **erroneous** if its value did not match the truth on its observed day (this comes first),
**superseded** if it matched then but does not match the current value, **valid** if it matched then and matches the
current value. "Matches" uses the field's tolerance, so an older weight within 4% of the current weight is valid.
Weight is compared in kg after unit conversion, so a correctly labeled lb value is not an error, and a value with the
wrong unit label is.

## Prediction file format (how to add a method)

One JSON line per case:

```
{"case_id": "...", "method": "...",
 "queries":    {"<field>": {"value": <normalized value>, "confidence": <0-1>}},
 "assertions": {"<id>": {"label": "valid|superseded|erroneous", "p_valid": .., "p_superseded": .., "p_erroneous": ..}}}
```

Normalized value: kg for `weight_kg`, relative day numbers for `birth_day` / `rabies_day`, lowercase strings for
categorical fields. Save as `preds_<method>_<split>.jsonl` and `score.py` finds it; run the method on `val` too
so `calibrate.py` can scale it.

## Metrics (score.py)

- `q_acc` — current-value accuracy over decidable queries, with a bootstrap 95% CI over cases.
- `ECE`, `Brier` — calibration of the query confidence (15 bins).
- `wrongOW` — of answers given with confidence ≥ 0.9, the fraction that are wrong (the safety number).
  `cov@.9` — the fraction of answers that reach 0.9.
- `sel@80` — accuracy on the 80% most confident answers (what you gain by asking the owner about the rest).
- `undecC` — mean confidence on queries that are undecidable by construction. Lower is better.
- Answerability: a decidable query is answerable when some reading in the record matches the truth (the ceiling for
  any method that states a value from the record). Per method: accuracy and wrongOW on answerable queries; mean
  confidence and share ≥ 0.9 on unanswerable ones.
- Per-assertion validity: precision/recall per label, macro-F1 over the labels present (`macroF1(3)` = v0.4's
  three-label average), ROC AUC for "erroneous" and "superseded", entry ECE / Brier / log-loss and p_erroneous ECE.
  Entries whose label is a coin flip by construction (`undecidable_ids`) are left out.
- Accuracy by conflict type (queries on fields touched by that conflict), by field type, and by regime.
- Paired comparisons with the reconciler and among the reconciler, the feature model and the LLM forms; mean ± sd
  over seeds.

## Sanity rule

If `newest_observed` scores above ~95% overall, the generator is too easy: raise the conflict knobs until the
simple rules fail on the realistic cases (unit errors, stale re-imports, contamination) and hold there.
The by-conflict-type table is the informative one; the overall number is dominated by the easy immutable fields.

## Real-data sets (v0.2 mapping, v0.5 fixes)

`realdata_map.py` maps Stock (55 sources, July 2011), Flight (38 sources, Dec 2011) and Book (author lists) from
lunadong.com/fusionDataSets.htm into the case format. Download the zips into a `realdata/` folder next to the script,
run once with `INSPECT = True` to check the file roles, then with `INSPECT = False`. Stock and Flight are sampled to
400 and 600 cases (`MAX_CASES`); the 600 flight-days are split by day (~270 dev / ~330 test). The nasdaq truth has no
sign on Change % / Change $, so those two are compared by magnitude. Book author lists are compared as surname
multisets, written twice: strict (`cases_book.jsonl`) and subset (`cases_book_subset.jsonl`). These sets have no time
dimension (observed = arrived = day 0), so they test source-conflict resolution, not supersession. Every method runs
on them zero-shot; nothing is fitted on them. Gold standards: Flight = the three airline websites (also inputs; see
the `_nogold` files), Stock = nasdaq.com's values (`STOCK_TRUTH_PREFER`; `realdata_map.py` prints whether a nasdaq
source is among the inputs and drops it in `cases_stock_nogold.jsonl`), Book = the book covers.

## Hub environment

`recbench_env/` wraps the generator as a Prime Intellect / verifiers environment (single-turn, JSON answer,
rewards = value accuracy + calibration (Brier) + label F1). It is self-contained (vendored copies of `gen.py`
and `recbench_common.py`) and needs Python 3.10+ for `verifiers`; see `recbench_env/README.md`.

## Known limits

- Rule and truth-discovery baselines emit per-assertion labels through a shared heuristic (`assign_labels` in
  baselines.py), so their erroneous recall is near zero. They are tuned for query accuracy.
- `feature_baseline.py` is schema-specific: its features are hand-written for the pet sources and field types.
- Free-text values are not modeled; extraction from text is out of scope by design.
- The snapshot regime is a caricature of the public sets; it widens the prior, it does not reproduce them.
- No public dataset tests the time dimension (supersession); the three public sets are snapshots and have all been
  used during development.
- Flight gate values come in several spellings of the same gate ("c16" / "16"); the gold standard's spelling decides.
