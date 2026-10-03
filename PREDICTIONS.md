# Pre-registered predictions — recbench v0

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
