# recbench — what the reconciler is compared against

Every method runs on the same frozen generator and splits (train / val / heldout / hard / snapshot) and zero-shot
on the public sets (Stock / Flight / Book). Every method is tuned on `val` only (equal budget) and gets the same
post-hoc temperature scaling fitted on `val`. The paper reports one table from one benchmark version; every row
below is re-run whenever the generator changes.

| # | Opponent | Form | Trained on our data? | Question it answers | Status |
|---|---|---|---|---|---|
| 1 | Newest observed | rule | no | What "newest fact wins" gets you | v0.4 done |
| 2 | Newest arrival | rule | no | What the app does today (last write wins) | v0.4 done |
| 3 | Source priority | rule, tuned source order | no | Does a trusted-source ranking fix it? | v0.4 done |
| 4 | Majority vote | classical truth discovery | no | Does counting agreement fix it? | v0.4 done |
| 5 | Time-decayed vote | classical, tuned decay + source weights | no | Does recency-weighted agreement fix it? | v0.4 done |
| 6 | Dawid–Skene (1979) | classical, tuned window / iterations / init | no | Does inferring per-source accuracy fix it? | v0.4 done |
| 7 | TruthFinder (2008) | classical, tuned window / gamma / damping | no | Does iterative source-trust ↔ claim-confidence fix it? | v0.4 done |
| 8 | Feature model | gradient-boosted trees on hand-made features; hyperparameters tuned on val | yes (same generated cases) | Is a transformer needed, or do engineered features suffice? | v0.4 done |
| 9 | DeepSeek, zero-shot | prompted LLM, JSON output, temperature 0 | no | Can an untrained LLM do it from instructions alone? | v0.4 done |
| 10 | DeepSeek, few-shot | prompted LLM + 2 solved training cases in the prompt | no | Does showing it worked examples close the gap? | v0.4 done |
| 11 | DeepSeek, CoT + self-consistency | reason step by step (≤ 250 words), 5 samples at T=0.7, majority vote, confidence = agreement × stated confidence | no | Does reasoning + voting close the gap, and at what cost? | v0.4 done |
| 12 | Claude (current Sonnet), zero-shot | same prompt and output format as #9 | no | Does a frontier model change the answer? | v0.5 planned |
| 13 | Claude, few-shot | same as #10 | no | Same question, with examples | v0.5 planned |
| 14 | Claude, CoT + self-consistency | same as #11 | no | Same question, strongest prompted form (run only if #12 lands within ~1 point of the reconciler; ~$100+) | v0.5 conditional |
| 15 | Fine-tuned small LLM (LoRA, Qwen-class) | an open LLM trained on the identical generated cases | yes | Is the gain from the training data or from the small typed model? | v0.5 planned (rented GPU, ~1 day) |
| 16 | Reconciler at 3 sizes (0.6M / ~2.5M / ~10M params, data ×4) | our model, scaling curve | yes | Does it improve with scale, or is the task saturated? | v0.5 planned (rented GPU) |
| 17 | Reconciler, permuted labels | our model trained on shuffled targets | yes (shuffled) | Control: do the scores come from learning, not a pipeline leak? | v0.4 done (passes) |
| 18 | Jev | probability-native decision model, API | no | Does a model built to output probabilities beat text output? | optional |

## Notes

- **Cost column for the paper:** rules and classical methods run in milliseconds; the feature model and the
  reconciler in milliseconds per record; DeepSeek zero-shot ≈ $0.002 per record and seconds; CoT ≈ 5× that;
  Claude ≈ 10–20× DeepSeek per token (check OpenRouter rates before running).
- **How to run #12–14:** `llm_baseline.py` with `BASE_URL` = OpenRouter, `MODEL` = the current Claude Sonnet id,
  `MODEL_TAG = "claude"`, `JSON_MODE = False`; the same `BATCH_RUNS` list; include `val` so `calibrate.py` can scale it.
- **CoT truncation:** the first v0.4 CoT runs at a 6,000-token output cap dropped 3–7.5% of samples; the reported runs
  use an 8,000-token cap and a 250-word reasoning limit (1 dropped sample in 3,000). v0.5 keeps those settings.
- **What the reconciler has to beat:** the feature model on per-entry precision and wrong-overwrite rate, the best
  LLM form on the semantic conflicts (corrections, injections), less degradation under the shifted prior than
  either, no per-field feature engineering, millisecond cost, and zero-shot transfer to the real sets.
