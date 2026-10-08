# tune.py -- equal-budget hyperparameter search for the classical baselines (recbench v0.4)
# Run from PyCharm after gen.py. Reads cases_val.jsonl ONLY (the tuning split: train settings, separate seed),
# writes tuned_params.json, which baselines.py applies to every split. Python 3.9, stdlib only.
#
# Protocol: every tunable method gets at most BUDGET configurations, chosen on val by query accuracy (ties: lower
# Brier). Nothing from heldout / hard / snapshot / the real sets is ever read here. Methods with no
# hyperparameters (newest_*, majority_vote) are listed as such. The reconciler and feature model are tuned under
# the same budget inside their own scripts (feature_baseline.py TUNE, train_reconciler.py uses fixed v0 settings).

import itertools
import json
import random
import time
from typing import Any, Dict, List, Tuple

import baselines as B
from recbench_common import load_jsonl, match, norm_value

# ============================================================ settings
VAL_FILE = "cases_val.jsonl"
OUT_FILE = "tuned_params.json"
BUDGET = 40
SEED = 3

SOURCE_WEIGHT_PRESETS = {
    "default":     {"vet_pdf": 1.0, "email_forward": 0.9, "owner": 0.7, "extractor": 0.5, "note_text": 0.3},
    "flat":        {"vet_pdf": 1.0, "email_forward": 1.0, "owner": 1.0, "extractor": 1.0, "note_text": 1.0},
    "vet_heavy":   {"vet_pdf": 1.0, "email_forward": 0.5, "owner": 0.4, "extractor": 0.3, "note_text": 0.1},
    "owner_heavy": {"vet_pdf": 0.8, "email_forward": 0.5, "owner": 1.0, "extractor": 0.6, "note_text": 0.3},
    "no_email":    {"vet_pdf": 1.0, "email_forward": 0.2, "owner": 0.7, "extractor": 0.5, "note_text": 0.3},
}


def grids() -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(SEED)
    g: Dict[str, List[Dict[str, Any]]] = {}
    # source_priority: the default order + random permutations of the five pet sources
    srcs = ["vet_pdf", "email_forward", "owner", "extractor", "note_text"]
    perms = [dict(B.PARAMS["source_priority"])]
    seen = {tuple(sorted(perms[0].items()))}
    while len(perms) < BUDGET:
        order = srcs[:]
        rng.shuffle(order)
        d = {s: 5 - i for i, s in enumerate(order)}
        key = tuple(sorted(d.items()))
        if key not in seen:
            seen.add(key)
            perms.append(d)
    g["source_priority"] = [{"source_priority": p} for p in perms]
    # time-decayed vote: decay constant x source-weight preset
    g["time_decayed_vote"] = [{"decay_tau_days": tau, "source_weight": SOURCE_WEIGHT_PRESETS[name]}
                              for tau in (15, 30, 60, 120, 240, 480, 1e6) for name in SOURCE_WEIGHT_PRESETS]
    # Dawid-Skene: window x iterations x initial accuracy
    g["dawid_skene"] = [{"window_days": w, "ds_iters": it, "ds_init_acc": a}
                        for w, it, a in itertools.product((15, 30, 60, 120, 365, 100000), (5, 15, 40), (0.6, 0.8))]
    # TruthFinder: window x gamma x dampening
    g["truthfinder"] = [{"window_days": w, "tf_gamma": gm, "tf_dampen": dm}
                        for w, gm, dm in itertools.product((15, 30, 60, 120, 365, 100000), (0.1, 0.3, 1.0), (0.3, 0.5))]
    for m in g:
        g[m] = g[m][:BUDGET]
    return g


# ============================================================ evaluation on val
def evaluate(method: str, cases: List[Dict[str, Any]]) -> Tuple[float, float]:
    """(query accuracy over decidable queries, Brier) for the current B.PARAMS."""
    ok, n, brier = 0, 0, 0.0
    for c in cases:
        pred = B.predict_case(method, c)
        for q in c["queries"]:
            if not q["decidable"]:
                continue
            p = pred["queries"].get(q["field"])
            n += 1
            if p is None or p.get("value") is None:
                brier += 1.0
                continue
            hit = 1 if match(q["field"], norm_value(q["field"], p["value"]), norm_value(q["field"], q["answer"])) else 0
            ok += hit
            brier += (float(p["confidence"]) - hit) ** 2
    return (ok / n if n else 0.0), (brier / n if n else 1.0)


def main() -> None:
    t0 = time.time()
    cases = load_jsonl(VAL_FILE)
    print("tuning on %s (%d cases), budget %d configs per method\n" % (VAL_FILE, len(cases), BUDGET))
    defaults = json.loads(json.dumps(B.PARAMS))
    result: Dict[str, Any] = {}
    print("%-18s %8s %8s %8s   %s" % ("method", "default", "tuned", "configs", "chosen"))
    for method in B.METHODS:
        B.set_params(defaults)
        base_acc, base_brier = evaluate(method, cases)
        grid = grids().get(method)
        if not grid:
            print("%-18s %7.1f%% %8s %8s   (no hyperparameters)" % (method, 100 * base_acc, "-", "-"))
            continue
        best = None
        for cfg in grid:
            B.set_params(defaults)
            B.set_params(cfg)
            acc, br = evaluate(method, cases)
            key = (acc, -br)
            if best is None or key > best[0]:
                best = (key, cfg, acc, br)
        _, cfg, acc, br = best
        result[method] = {"params": cfg, "val_q_acc": round(acc, 4), "val_brier": round(br, 4),
                          "default_val_q_acc": round(base_acc, 4), "default_val_brier": round(base_brier, 4),
                          "n_configs": len(grid)}
        shown = {k: (v if not isinstance(v, dict) else "{" + ", ".join("%s:%s" % kv for kv in v.items()) + "}")
                 for k, v in cfg.items()}
        print("%-18s %7.1f%% %7.1f%% %8d   %s" % (method, 100 * base_acc, 100 * acc, len(grid), shown))
    B.set_params(defaults)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print("\nwrote %s in %.0fs. Now run baselines.py (it applies these), then score.py." % (OUT_FILE, time.time() - t0))


if __name__ == "__main__":
    main()
