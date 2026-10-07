# recbench v0.3 — longitudinal record reconciliation benchmark
DOI: 10.5281/zenodo.23205119
One pet. A hidden true timeline. The stream of assertions the app would have seen, from sources that err in
different ways. Every assertion is labeled **valid / superseded / erroneous** against the hidden truth at "now",
and every field has a current-value query. Methods are scored on accuracy, calibration, wrong-overwrite rate,
selective accuracy, and per-assertion validity.

Everything is synthetic. No app data, no keys needed except for the LLM baseline.
Python 3.9, standard library only. Every script runs from PyCharm's Run button with this folder as the working
directory (PyCharm's default for a script in this folder).

## Run order

1. `gen.py` → writes `cases_train.jsonl`, `cases_heldout.jsonl`, `cases_hard.jsonl` (about 2,700 cases, ~10 s).
2. `baselines.py` → writes `preds_<method>_<split>.jsonl` for 7 rule / truth-discovery methods (~30 s).
3. `score.py` → prints the tables, writes `results.csv`.
4. `feature_baseline.py` → supervised baseline on hand-made features (gradient-boosted trees; needs
   `pip install scikit-learn`, which you have if joblib/scaler models run). Trains on `cases_train.jsonl`,
   writes `preds_feat_hgb_<split>.jsonl` for all three splits, prints AUCs. About 1 minute. Set `MODEL = "tabpfn"`
   after `pip install tabpfn` to try TabPFN on the same features.
5. `llm_baseline.py` → paste your DeepSeek (or OpenRouter) key at the top, pick `SPLIT`, run. Resumable.
   Then run `score.py` again; it picks up any `preds_*_<split>.jsonl` file automatically.
6. `realdata_map.py` (optional) → maps the public truth-discovery sets (Stock, Flight, Book from
   lunadong.com/fusionDataSets.htm) into the same case format. Download the zips into a `realdata/` folder next
   to the script, run once with `INSPECT = True` to check the file roles it guessed (the roles and value formats
   of the Oct 2026 downloads are already handled), then with `INSPECT = False` to write `cases_stock.jsonl`,
   `cases_flight.jsonl`, `cases_book.jsonl`. Stock and Flight are sampled to 400 and 600 cases (`MAX_CASES`);
   the nasdaq truth has no sign on Change % / Change $, so those two are compared by magnitude. Book author lists
   are compared as surname multisets (initials and first names ignored); it is written twice, strict
   (`cases_book.jsonl`: the list must equal the gold) and subset (`cases_book_subset.jsonl`: a partial list that
   is a subset of the gold counts as evidence for it), because first-author-only listings dominate that set. `baselines.py` and `score.py` pick up
   any `cases_<name>.jsonl` automatically. These sets have no time dimension (observed = arrived = day 0), so
   they test source-conflict resolution, not supersession; `superseded` is never a true label there.

`score.py` scores every method in a split on the cases all methods covered (`COMMON_CASES_ONLY = True`), so an
LLM run on a subset is compared on the same cases as everything else; the header line says when that applies.

Default sizes: train 2000, heldout 500, hard 200. `MAX_CASES_PER_SPLIT` in `baselines.py` and `MAX_CASES` in
`llm_baseline.py` let you run a quick subset.

## Splits

- **train**: the training distribution for any learned model (same knobs the model will be trained on).
- **heldout**: *different* generator settings — species mix, locale (lb vs kg), drift rates, source cadence, lags,
  conflict rates all shifted. Tests whether a method survives a change in the prior.
- **hard**: train settings with every conflict knob turned up.

The knobs are the `KNOBS` dict at the top of `gen.py`. Change them there; everything downstream follows.

## What is in a case

```
case_id, split, now_day
truth:      species, sex, birth_day, weight_knots [[day, kg]...], diet/medication/vet_clinic regimes
            [[start_day, value]...], rabies_events [day...]
assertions: id, field, value, unit, observed_day, arrived_day, source, extractor_conf, error_type, text,
            corrects (id), duplicate_of (id), label
queries:    field, answer, answer_type, decidable, tolerance
conflict_types: which conflict processes fired in this case
```

Fields: `species`, `sex` (immutable categorical), `birth_day` (immutable, ±60 days), `weight_kg` (drifting number,
±4%), `diet`, `medication`, `vet_clinic` (regime-switching categorical), `rabies_day` (dated event, ±7 days).

Sources and their error models: `owner` (typed; unit mix-ups, typos, stale recall, birth year off by one),
`vet_pdf` (accurate, arrives days late, OCR digit errors), `extractor` (chat-derived; has a confidence; sometimes
files the right value under the wrong field), `email_forward` (old vet records re-sent later; duplicates),
`note_text` (injected instructions). Plus: cross-pet contamination from a household's other pet, owner
corrections that reference an earlier entry, and deliberately undecidable same-day contradictions.

Two timestamps matter: `observed_day` is when the fact was true; `arrived_day` is when the app got it.
Re-imports and backfill deliver old facts late. "Newest arrival wins" is the current app failure.

Labels: an assertion is **erroneous** if its value was never true for this pet at its observed day,
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
categorical fields. Save as `preds_<method>_<split>.jsonl` and `score.py` finds it.

## Metrics (score.py)

- `q_acc` — current-value accuracy over decidable queries, with a bootstrap 95% CI over cases.
- `ECE`, `Brier` — calibration of the query confidence (15 bins).
- `wrongOW` — of answers given with confidence ≥ 0.9, the fraction that are wrong (the safety number).
  `cov@.9` — the fraction of answers that reach 0.9.
- `sel@80` — accuracy on the 80% most confident answers (what you gain by asking the owner about the rest).
- `undecC` — mean confidence on queries that are undecidable by construction. Lower is better.
- accuracy by conflict type (queries on fields touched by that conflict) and by field type.
- per-assertion validity: precision/recall per label, macro-F1, ROC AUC for "erroneous" and "superseded".

## Sanity rule

If `newest_observed` scores above ~95% overall, the generator is too easy: raise the conflict knobs until the
simple rules fail on the realistic cases (unit errors, stale re-imports, contamination) and hold there.
The by-conflict-type table is the informative one; the overall number is dominated by the easy immutable fields.

## Hub environment

`recbench_env/` wraps the generator as a Prime Intellect / verifiers environment (single-turn, JSON answer,
rewards = value accuracy + calibration (Brier) + label F1). It is self-contained (vendored copies of `gen.py`
and `recbench_common.py`) and needs Python 3.10+ for `verifiers`; see `recbench_env/README.md` for `vf-eval`
and `prime env push`.

## Known limits of v0

- Rule and truth-discovery baselines emit per-assertion labels through a shared heuristic
  (`assign_labels` in baselines.py): anything that doesn't match the chosen value is "superseded" if it was
  observed before the chosen value's newest support, else "erroneous". That is the reason their erroneous-recall
  is near zero; it is the gap a learned model is meant to fill.
- `feature_baseline.py` is schema-specific: its features are hand-written for these eight fields. That is the
  point of comparison for a model that reads raw assertions.
- Free-text values are not modeled; the generator produces structured assertions only. Extraction from text is
  out of scope by design.
