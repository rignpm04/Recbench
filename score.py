# score.py -- scores every preds_<method>_<split>.jsonl against the cases (recbench v0)
# Run from PyCharm after baselines.py (and llm_baseline.py). Prints tables, writes results.csv.
# Python 3.9, stdlib only.

import csv
import glob
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

from recbench_common import FIELD_TYPE, LABELS, load_jsonl, match, norm_value

# ============================================================ settings
SPLITS = {"train": "cases_train.jsonl", "heldout": "cases_heldout.jsonl", "hard": "cases_hard.jsonl"}
RESULTS_CSV = "results.csv"
COMMON_CASES_ONLY = True   # score every method in a split on the cases ALL methods covered (apples to apples)
ECE_BINS = 15
OVERWRITE_THRESHOLD = 0.90
COVERAGES = [0.9, 0.8, 0.7]
BOOTSTRAP = 300
SEED = 7

CONFLICT_TYPES = ["unit", "typo", "ocr_digit", "stale_recall", "stale_reimport", "duplicate", "correction",
                  "contamination", "wrong_field", "injection"]
# (undecidable queries are excluded from accuracy; they are scored by confidence: undecC / undecidable_auto_rate)


# ============================================================ metric helpers
def ece(confs: List[float], correct: List[int], bins: int = ECE_BINS) -> float:
    if not confs:
        return float("nan")
    tot = len(confs)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confs[i] for i in idx) / len(idx)
        e += len(idx) / tot * abs(acc - conf)
    return e


def brier(confs: List[float], correct: List[int]) -> float:
    if not confs:
        return float("nan")
    return sum((c - y) ** 2 for c, y in zip(confs, correct)) / len(confs)


def auc(scores: List[float], labels: List[int]) -> float:
    """ROC AUC by the rank-sum formula with average ranks for ties."""
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    rank_pos = sum(r for r, y in zip(ranks, labels) if y == 1)
    return (rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def selective_accuracy(confs: List[float], correct: List[int], coverage: float) -> float:
    if not confs:
        return float("nan")
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    k = max(1, int(round(coverage * len(confs))))
    return sum(correct[i] for i in order[:k]) / k


def prf(pred: List[str], true: List[str], label: str) -> Tuple[float, float, float]:
    tp = sum(1 for p, t in zip(pred, true) if p == label and t == label)
    fp = sum(1 for p, t in zip(pred, true) if p == label and t != label)
    fn = sum(1 for p, t in zip(pred, true) if p != label and t == label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def fmt(x: Any, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   -  "
    if pct:
        return "%5.1f%%" % (100 * x)
    return "%6.3f" % x


# ============================================================ scoring one (split, method)
def score(cases: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    pmap = {p["case_id"]: p for p in preds}
    rows = []            # one row per decidable query
    undec_conf = []      # confidences on undecidable queries
    answered = 0
    total_q = 0
    a_true, a_pred, p_err, p_sup, y_err, y_sup = [], [], [], [], [], []
    per_case_acc: List[Tuple[int, int]] = []    # (correct, n) for bootstrap

    for c in cases:
        p = pmap.get(c["case_id"])
        pq = p["queries"] if p else {}
        pa = p["assertions"] if p else {}
        n_ok, n = 0, 0
        fields_with = {}
        for a in c["assertions"]:
            for key in ([a["error_type"]] if a["error_type"] else []) + \
                       (["correction"] if a.get("corrects") is not None else []) + \
                       (["duplicate"] if a.get("duplicate_of") is not None else []):
                fields_with.setdefault(key, set()).add(a["field"])
        for q in c["queries"]:
            field = q["field"]
            total_q += 1
            ans = norm_value(field, q["answer"])
            pred = pq.get(field)
            conf, ok = 0.0, 0
            if pred is not None and pred.get("value") is not None:
                try:
                    pv = norm_value(field, pred["value"])
                    conf = min(1.0, max(0.0, float(pred.get("confidence", 0.0))))
                    ok = 1 if match(field, pv, ans) else 0
                    answered += 1
                except (ValueError, TypeError):
                    conf, ok = 0.0, 0          # unparseable value counts as unanswered and wrong
            if not q["decidable"]:
                undec_conf.append(conf)
                continue
            tags = [t for t in CONFLICT_TYPES if field in fields_with.get(t, set())]
            rows.append({"field": field, "ftype": FIELD_TYPE[field], "conf": conf, "ok": ok, "tags": tags})
            n_ok += ok
            n += 1
        per_case_acc.append((n_ok, n))
        for a in c["assertions"]:
            pl = pa.get(str(a["id"]))
            a_true.append(a["label"])
            if pl:
                a_pred.append(pl.get("label", "valid"))
                p_err.append(float(pl.get("p_erroneous", 0.0)))
                p_sup.append(float(pl.get("p_superseded", 0.0)))
            else:
                a_pred.append("valid")
                p_err.append(0.0)
                p_sup.append(0.0)
            y_err.append(1 if a["label"] == "erroneous" else 0)
            y_sup.append(1 if a["label"] == "superseded" else 0)

    confs = [r["conf"] for r in rows]
    oks = [r["ok"] for r in rows]
    res: Dict[str, Any] = {}
    res["n_queries"] = len(rows)
    res["answered_rate"] = answered / total_q if total_q else float("nan")
    res["q_acc"] = sum(oks) / len(oks) if oks else float("nan")
    res["ece"] = ece(confs, oks)
    res["brier"] = brier(confs, oks)
    hi = [i for i, c in enumerate(confs) if c >= OVERWRITE_THRESHOLD]
    res["coverage@0.9"] = len(hi) / len(confs) if confs else float("nan")
    res["wrong_overwrite@0.9"] = (1 - sum(oks[i] for i in hi) / len(hi)) if hi else float("nan")
    for cov in COVERAGES:
        res["sel_acc@%d" % int(cov * 100)] = selective_accuracy(confs, oks, cov)
    res["undecidable_mean_conf"] = sum(undec_conf) / len(undec_conf) if undec_conf else float("nan")
    res["undecidable_auto_rate"] = (sum(1 for c in undec_conf if c >= OVERWRITE_THRESHOLD) / len(undec_conf)) if undec_conf else float("nan")
    # by field type
    for ft in sorted(set(FIELD_TYPE.values())):
        sub = [r for r in rows if r["ftype"] == ft]
        res["acc:ftype:" + ft] = sum(r["ok"] for r in sub) / len(sub) if sub else float("nan")
    # by conflict type
    for t in CONFLICT_TYPES:
        sub = [r for r in rows if t in r["tags"]]
        res["acc:conflict:" + t] = sum(r["ok"] for r in sub) / len(sub) if sub else float("nan")
        res["n:conflict:" + t] = len(sub)
    # assertion-level
    f1s = []
    for lab in LABELS:
        pr, rc, f1 = prf(a_pred, a_true, lab)
        res["asrt_%s_precision" % lab] = pr
        res["asrt_%s_recall" % lab] = rc
        f1s.append(f1)
    res["asrt_macro_f1"] = sum(f1s) / len(f1s)
    res["asrt_auc_erroneous"] = auc(p_err, y_err)
    res["asrt_auc_superseded"] = auc(p_sup, y_sup)
    # bootstrap CI on q_acc
    rng = random.Random(SEED)
    accs = []
    for _ in range(BOOTSTRAP):
        sample = [per_case_acc[rng.randrange(len(per_case_acc))] for _ in range(len(per_case_acc))]
        ok_s = sum(s[0] for s in sample)
        n_s = sum(s[1] for s in sample)
        accs.append(ok_s / n_s if n_s else 0.0)
    accs.sort()
    res["q_acc_ci_lo"] = accs[int(0.025 * len(accs))]
    res["q_acc_ci_hi"] = accs[min(len(accs) - 1, int(0.975 * len(accs)))]
    return res


# ============================================================ driver
def discover_methods(split: str) -> List[str]:
    suffix = "_%s.jsonl" % split
    out = []
    for path in sorted(glob.glob("preds_*%s" % suffix)):
        name = os.path.basename(path)[len("preds_"):-len(suffix)]
        out.append(name)
    return out


def main() -> None:
    all_rows = []
    for split, cpath in SPLITS.items():
        if not os.path.exists(cpath):
            continue
        cases = load_jsonl(cpath)
        methods = discover_methods(split)
        if not methods:
            continue
        results = {}
        loaded = {m: load_jsonl("preds_%s_%s.jsonl" % (m, split)) for m in methods}
        common = None
        if COMMON_CASES_ONLY:
            for m in methods:
                ids = set(p["case_id"] for p in loaded[m])
                common = ids if common is None else (common & ids)
        for m in methods:
            preds = loaded[m]
            ids = set(p["case_id"] for p in preds)
            if common is not None:
                ids = ids & common
                preds = [p for p in preds if p["case_id"] in ids]
            sub = [c for c in cases if c["case_id"] in ids]   # LLM runs may cover a subset
            results[m] = score(sub, preds)
            results[m]["n_cases"] = len(sub)
            for k, v in results[m].items():
                all_rows.append((split, m, k, v))

        print("\n=== %s split ===" % split)
        if common is not None and len(common) < len(cases):
            print("(scored on the %d cases every method covered; set COMMON_CASES_ONLY = False for full-split numbers)"
                  % len(common))
        print("%-18s %6s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s"
              % ("method", "cases", "q_acc", "ci_lo", "ci_hi", "ECE", "Brier", "wrongOW", "cov@.9",
                 "sel@80", "undecC", "macroF1", "AUCerr"))
        for m in methods:
            r = results[m]
            print("%-18s %6d %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s"
                  % (m, r["n_cases"], fmt(r["q_acc"], True), fmt(r["q_acc_ci_lo"], True), fmt(r["q_acc_ci_hi"], True),
                     fmt(r["ece"]), fmt(r["brier"]), fmt(r["wrong_overwrite@0.9"], True), fmt(r["coverage@0.9"], True),
                     fmt(r["sel_acc@80"], True), fmt(r["undecidable_mean_conf"]), fmt(r["asrt_macro_f1"]),
                     fmt(r["asrt_auc_erroneous"])))
        print("\n-- accuracy by conflict type (queries on fields affected by that conflict) --")
        header = "%-16s" % "conflict" + "".join("%12s" % m[:11] for m in methods)
        print(header)
        for t in CONFLICT_TYPES:
            n = results[methods[0]]["n:conflict:" + t]
            line = "%-16s" % ("%s (n=%d)" % (t, n))[:16]
            for m in methods:
                line += "%12s" % fmt(results[m]["acc:conflict:" + t], True)
            print(line)
        print("\n-- accuracy by field type --")
        for ft in sorted(set(FIELD_TYPE.values())):
            line = "%-16s" % ft
            for m in methods:
                line += "%12s" % fmt(results[m]["acc:ftype:" + ft], True)
            print(line)
        print("\n-- per-assertion validity (precision / recall) --")
        for m in methods:
            r = results[m]
            print("%-18s valid %5.2f/%5.2f  superseded %5.2f/%5.2f  erroneous %5.2f/%5.2f  AUC(sup) %s"
                  % (m, r["asrt_valid_precision"], r["asrt_valid_recall"], r["asrt_superseded_precision"],
                     r["asrt_superseded_recall"], r["asrt_erroneous_precision"], r["asrt_erroneous_recall"],
                     fmt(r["asrt_auc_superseded"])))

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "metric", "value"])
        for split, m, k, v in all_rows:
            w.writerow([split, m, k, "" if (isinstance(v, float) and math.isnan(v)) else v])
    print("\nwrote %s (%d rows)" % (RESULTS_CSV, len(all_rows)))


if __name__ == "__main__":
    main()
