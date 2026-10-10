# recbench v0.4 — longitudinal record reconciliation benchmark


Cite all versions: [10.5281/zenodo.23205119](https://doi.org/10.5281/zenodo.23205119)

One entity. A hidden truth. The stream of assertions the app would have seen, from sources that err in different
ways. Every assertion is labeled **valid / superseded / erroneous** against the hidden truth at "now", and every
field has a current-value query. Methods are scored on accuracy, calibration, wrong-overwrite rate, selective
accuracy, and per-assertion validity.

Everything is synthetic. No app data, no keys needed except for the LLM baselines.
Python 3.9, standard library only (feature model: scikit-learn; reconciler: torch in a 3.10+ venv). Every script
runs from PyCharm's Run button with this folder as the working directory.

## Run order (v0.4)

1. `gen.py` → `cases_train` (2000; 15% snapshot regime), `cases_val` (300, same settings as train, separate seed),
   `cases_heldout` (500, shifted settings), `cases_hard` (200, conflict knobs up), `cases_snapshot` (300). ~6 s.
   After any generator change, delete the old `preds_*.jsonl`, `raw_preds_*.jsonl`, `tuned_params.json`,
   `calibration.json` and `results.csv` first: old predictions carry the same case ids as the new cases.
2. `tune.py` → equal-budget hyperparameter search for the classical baselines on `val` only → `tuned_params.json`. ~1 min.
3. `baselines.py` → `preds_<method>_<split>.jsonl` for 7 rule / truth-discovery methods on every split, with the
   tuned parameters. ~40 s.
4. `feature_baseline.py` → gradient-boosted trees on hand-made features; its own grid on `val` (same budget);
   predictions on every split, zero-shot on `snapshot` and the real sets. ~5–8 min.
5. `train_reconciler.py` (3.11 venv, torch) → the reconciler; predictions on every split. ~1–1.5 h on a MacBook.
   Then once more with `PERMUTE_LABELS = True`, `EPOCHS = 2`, `N_TRAIN_CASES = 5000` → the permutation control
   (`preds_reconciler_permuted_*`). ~10 min.
6. `llm_baseline.py` → paste the key; run `MODE = "zero_shot"` on `val`, `hard`, `heldout`; then `"few_shot"` and
   `"cot_sc"` on the same three splits (`MAX_CASES = 200`). Resumable. Roughly $10 in total on DeepSeek.
7. `calibrate.py` → fits one temperature per method on `val` and rewrites every `preds_*` file (raw copies kept
   as `raw_preds_*`). Methods without a `val` run are listed as uncalibrated.
8. `score.py` → the tables, `results.csv`.
9. `realdata_map.py` (optional, before 3) → maps the public truth-discovery sets (Stock, Flight, Book from
   lunadong.com/fusionDataSets.htm) into the case format; see the v0.2 notes below.

`score.py` scores every method in a split on the cases all methods covered (`COMMON_CASES_ONLY = True`), so an
LLM run on a subset is compared on the same cases as everything else; the header line says when that applies.
The `val` split is never a results table (`SKIP_SPLITS`).

## Protocol (v0.4)

- **Tuning.** Every method's hyperparameters are chosen on `val` (train distribution, separate seed) with at most
  40 configurations: `tune.py` for the classical baselines (decay constants, windows, iterations, source
  priorities / weights), `feature_baseline.py` for the trees. The reconciler uses the fixed v0 settings and no
  selection at all; `heldout` is printed per epoch for monitoring only and the last epoch is the reported model.
  Nothing is ever chosen on `heldout`, `hard`, `snapshot` or a real set. The paper reports tuned numbers only;
  `tune.py` prints default-vs-tuned on `val` for the record.
- **Calibration.** One post-hoc temperature per method for the query confidence and one for the label
  distribution, fitted on `val` by NLL, applied everywhere (real sets included — no real labels are used).
  Temperature scaling is monotone, so accuracy and labels are untouched; ECE, Brier, wrong-overwrite and coverage
  are what it changes. The feature model's isotonic calibration from v0.1 is off by default so every method gets
  the same treatment.
- **LLM.** Three fair forms of the same model: zero-shot JSON, few-shot (2 worked train cases with gold answers),
  chain-of-thought + self-consistency (5 samples; value by majority, confidence = agreement × stated confidence).
- **Leak guards.** The reconciler's encoder reads only `OBSERVABLE_KEYS` (field, value, unit, the two days,
  source, extractor confidence, note presence, corrects / duplicate_of presence) from a scrubbed copy of each
  assertion; labels, error types and the truth are read separately into the targets. `PERMUTE_LABELS = True`
  trains the same model on shuffled targets; it must score at chance, or no result stands. The feature model's
  `field_rows` reads the same keys.
- **Generator v2 flags removed.** `note_text` entries are now mostly benign notes (correct at observation), so
  "has a note" no longer identifies an injection; owner corrections now also restate right values or make them
  wrong (`bad_correction`), so "corrects" / "is corrected" no longer identify the labels.

## Splits

- **train**: the training distribution for any learned model. 15% of cases are in the snapshot regime.
- **val**: train settings, separate seed. Tuning and calibration only.
- **heldout**: *different* generator settings — species mix, locale (lb vs kg), drift rates, source cadence, lags,
  conflict rates all shifted. Tests whether a method survives a change in the prior. Longitudinal only.
- **hard**: train settings with every conflict knob turned up. Longitudinal only.
- **snapshot**: the snapshot regime only: 5–40 anonymous sources (`src_00`…), 2–6 generic fields (`num_rel_*`,
  `num_abs_*`, `cat_*` with fixed tolerances, registered through `field_types` like the real-data cases), every
  claim on day 0, per-case source reliability and copy rate, so some cases have a popular wrong value. Labels are
  valid / erroneous only. This is the shape of the public truth-discovery sets.

The knobs are the `KNOBS` dict at the top of `gen.py`. Change them there; everything downstream follows.

## What is in a case

```
case_id, split, regime ("longitudinal" | "snapshot"), now_day
longitudinal:  truth: species, sex, birth_day, weight_knots [[day, kg]...], diet/medication/vet_clinic regimes
               [[start_day, value]...], rabies_events [day...]
snapshot:      truth_values {field: value}, field_types {field: [type, tolerance]}, source_accuracy, snapshot_params
assertions:    id, field, value, unit, observed_day, arrived_day, source, extractor_conf, error_type, text,
               corrects (id), duplicate_of (id), label
queries:       field, answer, answer_type, decidable, tolerance
conflict_types: which conflict processes fired in this case
```

Pet fields: `species`, `sex` (immutable categorical), `birth_day` (immutable, ±60 days), `weight_kg` (drifting
number, ±4%), `diet`, `medication`, `vet_clinic` (regime-switching categorical), `rabies_day` (dated event, ±7 days).

Sources and their error models: `owner` (typed; unit mix-ups, typos, stale recall, birth year off by one),
`vet_pdf` (accurate, arrives days late, OCR digit errors), `extractor` (chat-derived; has a confidence; sometimes
files the right value under the wrong field), `email_forward` (old vet records re-sent later; duplicates),
`note_text` (benign notes and, rarely, injected instructions). Plus: cross-pet contamination from a household's
other pet, owner corrections that reference an earlier entry (right, redundant, or wrong), and deliberately
undecidable same-day contradictions. Snapshot sources: `source_error` (independent wrong value) and
`copied_error` (a wrong value shared across sources).

Two timestamps matter: `observed_day` is when the fact was true; `arrived_day` is when the app got it.
Re-imports and backfill deliver old facts late. "Newest arrival wins" is the current app failure.

Labels: an assertion is **erroneous** if its value was never true for this entity at its observed day,
**superseded** if it was true then but the field has changed since, **valid** if it is still the current value.
Weight is compared in kg after unit conversion, so a correctly labeled lb value is not an error, and a value with
the wrong unit label is.

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
- accuracy by conflict type (queries on fields touched by that conflict), by field type, and by regime.
- per-assertion validity: precision/recall per label, macro-F1, ROC AUC for "erroneous" and "superseded".

## Sanity rule

If `newest_observed` scores above ~95% overall, the generator is too easy: raise the conflict knobs until the
simple rules fail on the realistic cases (unit errors, stale re-imports, contamination) and hold there.
The by-conflict-type table is the informative one; the overall number is dominated by the easy immutable fields.

## Real-data sets (v0.2)

`realdata_map.py` maps Stock (55 sources, July 2011), Flight (38 sources, Dec 2011) and Book (227 vendors, author
lists) into the case format. Download the zips into a `realdata/` folder next to the script, run once with
`INSPECT = True` to check the file roles it guessed (the roles and value formats of the Oct 2026 downloads are
already handled), then with `INSPECT = False` to write `cases_stock.jsonl`, `cases_flight.jsonl`,
`cases_book.jsonl`. Stock and Flight are sampled to 400 and 600 cases (`MAX_CASES`); the nasdaq truth has no sign
on Change % / Change $, so those two are compared by magnitude. Book author lists are compared as surname
multisets; it is written twice, strict (`cases_book.jsonl`) and subset (`cases_book_subset.jsonl`), because
first-author-only listings dominate that set. These sets have no time dimension (observed = arrived = day 0), so
they test source-conflict resolution, not supersession; `superseded` is never a true label there. Every method
runs on them zero-shot; nothing is ever fitted on them.

## Hub environment

`recbench_env/` wraps the generator as a Prime Intellect / verifiers environment (single-turn, JSON answer,
rewards = value accuracy + calibration (Brier) + label F1). It is self-contained (vendored copies of `gen.py`
and `recbench_common.py`) and needs Python 3.10+ for `verifiers`; see `recbench_env/README.md` for `vf-eval`
and `prime env push`.

## Known limits

- Rule and truth-discovery baselines emit per-assertion labels through a shared heuristic
  (`assign_labels` in baselines.py): anything that doesn't match the chosen value is "superseded" if it was
  observed before the chosen value's newest support, else "erroneous", with the entry's own cluster confidence as
  its probability. That is the reason their erroneous recall is near zero; it is the gap a learned model is meant
  to fill. They are tuned for query accuracy; their per-entry numbers are reported as they fall out.
- `feature_baseline.py` is schema-specific: its features are hand-written for the pet sources and field types
  (generic fields and unknown sources fall into "other" slots). That is the point of comparison for a model that
  reads raw assertions.
- Free-text values are not modeled; the generator produces structured assertions only. Extraction from text is
  out of scope by design.
- The snapshot regime is a caricature of the public sets: fixed tolerances, one copy pool per field, no
  source-level copying structure. It widens the prior; it does not reproduce Stock or Book.
