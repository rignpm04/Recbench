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
