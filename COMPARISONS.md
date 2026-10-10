# recbench — what the reconciler is compared against (v0.5)

Every method runs on the same frozen generator and splits and zero-shot on the public sets (Stock, Flight dev / test,
Book, and the `_nogold` variants). v0.5 = generator v3; every row is (re-)run on it, and the paper's tables come from
the sealed test splits (`test_hard`, `test_heldout`, `test_snapshot`) generated once after every method is frozen.
Every method is tuned on `val` only (equal budget), gets the same post-hoc temperature scaling fitted on `val`, and is
compared with the reconciler on the same queries (paired bootstrap CI + McNemar); the reconciler, the feature model and
the LLM forms are also compared pairwise. The paper reports one table from one benchmark version.

| # | Opponent | Form | Trained on our data? | Question it answers | v0.5 |
|---|---|---|---|---|---|
| 1 | Newest observed | rule | no | What "newest fact wins" gets you | re-run |
| 2 | Newest arrival | rule | no | What the app does today (last write wins) | re-run |
| 3 | Source priority | rule, tuned source order | no | Does a trusted-source ranking fix it? | re-run |
| 4 | Majority vote | classical truth discovery | no | Does counting agreement fix it? | re-run (states the newest readings of its cluster) |
| 5 | Time-decayed vote | classical, tuned decay + source weights | no | Does recency-weighted agreement fix it? | re-run (same) |
| 6 | Dawid–Skene (1979) | classical, tuned window / iterations / init | no | Does inferring per-source accuracy fix it? | re-run (same) |
| 7 | TruthFinder (2008) | classical, tuned window / gamma / damping | no | Does iterative source-trust ↔ claim-confidence fix it? | re-run (same) |
| 8 | Feature model | gradient-boosted trees on hand-made features; tuned on val | yes — **the same 160,000 cases as the reconciler** (v0.4: 2,000) | Is a transformer needed, or do engineered features suffice? | seeds 1-3 |
| 9 | DeepSeek, zero-shot | prompted LLM, JSON output, temperature 0 | no | Can an untrained LLM do it from instructions alone? | prompt now states the grading rules |
| 10 | DeepSeek, few-shot | + 2 solved training cases in the prompt | no | Does showing worked examples close the gap? | same |
| 11 | DeepSeek, CoT + self-consistency | reason (≤ 250 words), 5 samples at T=0.7, majority vote, confidence = agreement × stated confidence | no | Does reasoning + voting close the gap, and at what cost? | same |
| 12 | Claude (current Sonnet), zero-shot | same prompt and output format as #9 (OpenRouter) | no | Does a frontier model change the answer? | phase 2 |
| 13 | Claude, few-shot | same as #10 | no | Same question, with examples | phase 2 |
| 14 | Claude, CoT + self-consistency | same as #11 | no | Same question, strongest prompted form | phase 2 |
| 15 | Fine-tuned small LLM (LoRA, Qwen-class) | an open LLM trained on generated cases | yes | Is the gain from the training data or from the small typed model? | optional; the number of training cases is stated, and below 160,000 the row is "data-limited" |
| 16 | Reconciler at 3 sizes (0.6M / ~2.5M / ~10M, data ×4) | our model, scaling curve | yes | Does it improve with scale, or is the task saturated? | optional; judged on the test splits and val log-loss |
| 17 | Reconciler, permuted labels | within-case shuffle and cross-case (targets drawn from the label marginal) | yes (shuffled) | Do the scores come from learning? | both re-run; the leak itself is tested by the scrub test |
| 18 | Jev | probability-native decision model, API | no | Does a model built to output probabilities beat text output? | optional |

## Notes

- **Cost column for the paper:** rules and classical methods run in microseconds per record; the feature model and
  the reconciler in milliseconds; DeepSeek zero-shot ≈ $0.002 per record and seconds; CoT ≈ 5× that. Claude: check
  OpenRouter's rate for the Sonnet id you use and record it in RESULTS.md. DeepSeek's ten v0.4 runs used ~12M input
  and ~13M output tokens; the Claude batch will be of that order.
- **How to run #12–14:** `llm_baseline.py` with `PROVIDER = "claude"` (paste the OpenRouter key, the exact Claude
  Sonnet model id and its prices into `PROVIDERS["claude"]`), `BATCH_RUNS = BATCH_RUNS_FINAL` after the test splits
  exist. `val` is included so `calibrate.py` can scale it.
- **CoT truncation:** the reported runs use an 8,000-token output cap and a 250-word reasoning limit (v0.4: 1 dropped
  sample in 3,000).
- **What the reconciler has to beat:** with equal data, the feature model on accuracy, per-entry labels and
  wrong-overwrite on the schema the features were written for; the best LLM form on the semantic conflicts; less
  degradation under the shifted prior; and, because it needs no per-field code, zero-shot transfer to the public sets.
