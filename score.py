# score.py -- scores every preds_<method>_<split>.jsonl against the cases (recbench v0.5)
# Run from PyCharm after the methods (and after calibrate.py for the temperature-scaled tables).
# Prints tables, writes results.csv. Python 3.9, stdlib only.
#
# v0.5 additions (pre-registered; every method is scored the same way):
#   - Answerability. A decidable query is "answerable" when at least one reading in the record matches the current
#     truth (method-independent). The split header prints the share (the ceiling for any method that states a value
#     from the record). Per method: wrong-overwrite@0.9 on answerable queries only, and on unanswerable queries the
#     mean confidence and the share answered at >= 0.9. In v0.4, 87% of the reconciler's confident mistakes on
#     heldout were unanswerable queries.
#   - Entry-label calibration: multiclass Brier, top-label ECE, the ECE of p_erroneous and the log-loss (v0.4
#     measured none), next to the log-loss of the best constant label distribution (the permutation controls'
#     null: a model that learned nothing about entries cannot beat it).
#   - Macro-F1 over the label classes that occur in the scored entries (v0.4 averaged all three, so a split without
#     superseded entries -- snapshot, public sets -- capped it at 0.67; that number is kept as macro-F1(3)).
#   - Entries whose label is a coin flip by construction (case["undecidable_ids"]: an undecidable same-day pair and
#     the earlier readings of its two values) are left out of every per-entry metric.
#   - Paired comparisons against REFERENCE_METHOD: q_acc difference with a paired bootstrap CI over cases and an
#     exact McNemar p-value on the same queries (v0.4 compared overlapping marginal CIs). Then every pair among the
#     reconciler, the feature model and the LLM forms (seed replicates and controls excluded).
#   - A case with no assertions and no queries (generator v3's feeds can leave one) counts as covered by every
#     method: it carries nothing to score, and a method that writes no record for it is not "partial".
#   - When an LLM run covers only some cases, the common-cases table is followed by a full-split table for every
#     method that covers all cases (v0.4 needed a manual re-run with COMMON_CASES_ONLY = False).
#   - Replicates: methods named <base>_seed<N> are summarised with <base> as mean +- sd.
#   - Prediction files are matched to splits by the longest split name (test_hard before hard), among every split
#     name the scripts write (recbench_common.KNOWN_SPLITS) and the cases files present; a prediction file that
#     shares no case id with its split is reported and left out instead of emptying the common-cases table.
#   - sel@k counts answers tied at the cut in proportion (v0.4: file order decided ties).

import csv
import glob
import math
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from recbench_common import FIELD_TYPE, KNOWN_SPLITS, LABELS, ftype_of, load_jsonl, match, norm_value, regime_of

# ============================================================ settings
SPLITS = {"train": "cases_train.jsonl", "heldout": "cases_heldout.jsonl", "hard": "cases_hard.jsonl",
          "snapshot": "cases_snapshot.jsonl", "val": "cases_val.jsonl"}
SKIP_SPLITS = ["val"]      # val is the tuning / calibration split: in-sample for tuned methods, so not a result table
RESULTS_CSV = "results.csv"
COMMON_CASES_ONLY = True   # score every method in a split on the cases ALL methods covered (apples to apples)
ECE_BINS = 15
OVERWRITE_THRESHOLD = 0.90
COVERAGES = [0.9, 0.8, 0.7]
BOOTSTRAP = 300
SEED = 7
REFERENCE_METHOD = "reconciler"    # paired comparisons are against this method (skipped if it has no predictions)
PAIRED_BOOTSTRAP = 2000
PAIRWISE_METHODS = ["reconciler", "feat_hgb", "reconciler_anon"]   # + every llm_* method: all pairs among them
                                                                     # (prediction 23; reconciler_anon: Parliament amendment)
PAIRWISE_PREFIXES = ["llm_"]

CONFLICT_TYPES = ["unit", "typo", "ocr_digit", "stale_recall", "stale_reimport", "duplicate", "correction",
                  "bad_correction", "contamination", "wrong_field", "injection", "benign_note",
                  "source_error", "copied_error", "feed_error"]
# (undecidable queries are excluded from accuracy; they are scored by confidence: undecC / undecidable_auto_rate)
# v2 tags: bad_correction = a "correction" that made a right entry wrong; benign_note = ordinary note_text entry;
# source_error / copied_error = snapshot-regime errors (independent / shared across sources).
# v3 tag: feed_error = a wrong value carried verbatim by a feed (a block of sources showing identical rows).


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
    """Accuracy on the `coverage` most confident answers. Answers tied with the last one taken count in proportion
    (the expected accuracy under a random tie-break; v0.4 took ties in file order, so for a method with one constant
    confidence sel@80 was the accuracy of the first 80% of cases)."""
    if not confs:
        return float("nan")
    order = sorted(range(len(confs)), key=lambda i: -confs[i])
    k = max(1, int(round(coverage * len(confs))))
    thr = confs[order[k - 1]]
    above = [i for i in order if confs[i] > thr]
    tied = [i for i in order if confs[i] == thr]
    return (sum(correct[i] for i in above) + (k - len(above)) * sum(correct[i] for i in tied) / len(tied)) / k


def prf(pred: List[str], true: List[str], label: str) -> Tuple[float, float, float]:
    tp = sum(1 for p, t in zip(pred, true) if p == label and t == label)
    fp = sum(1 for p, t in zip(pred, true) if p == label and t != label)
    fn = sum(1 for p, t in zip(pred, true) if p != label and t == label)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value from the two discordant counts. Integer arithmetic: a float 2.0 ** n
    overflows from n = 1024 discordant queries on (Python divides two ints exactly at any size)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, (2 * sum(math.comb(n, i) for i in range(k + 1))) / (2 ** n))


def abbr(m: str, w: int = 13) -> str:
    """Column label for a method name (long names keep their start and end, so variants stay distinguishable)."""
    if len(m) <= w:
        return m
    head = w // 2 - 1
    return m[:head] + "~" + m[-(w - head - 1):]


def fmt(x: Any, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "   -  "
    if pct:
        return "%5.1f%%" % (100 * x)
    return "%6.3f" % x


_ANSWERABLE: Dict[Tuple[str, str], bool] = {}


def answerable(case: Dict[str, Any], field: str, ans: Any) -> bool:
    """Method-independent: does any reading of this field in the record match the current truth? (cached)"""
    key = (case["case_id"], field)
    if key not in _ANSWERABLE:
        _ANSWERABLE[key] = _answerable(case, field, ans)
    return _ANSWERABLE[key]


def _answerable(case: Dict[str, Any], field: str, ans: Any) -> bool:
    for a in case["assertions"]:
        if a["field"] != field:
            continue
        try:
            if match(field, norm_value(field, a["value"], a.get("unit")), ans):
                return True
        except (ValueError, TypeError):
            continue
    return False


# ============================================================ scoring one (split, method)
def score(cases: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Dict[str, Any]:
    pmap = {p["case_id"]: p for p in preds}
    rows = []            # one row per decidable query
    undec_conf = []      # confidences on undecidable queries
    answered = 0
    total_q = 0
    a_true, a_pred, p_err, p_sup, y_err, y_sup, dists = [], [], [], [], [], [], []
    per_case_acc: List[Tuple[int, int]] = []    # (correct, n) for bootstrap
    qrows: List[Tuple[str, str, int]] = []      # (case_id, field, ok) for paired tests

    for c in cases:
        p = pmap.get(c["case_id"])
        pq = p["queries"] if p else {}
        pa = p["assertions"] if p else {}
        n_ok, n = 0, 0
        fields_with = {}
        regime = regime_of(c)
        for a in c["assertions"]:
            for key in ([a["error_type"]] if a["error_type"] else []) + \
                       (["correction"] if a.get("corrects") is not None else []) + \
                       (["duplicate"] if a.get("duplicate_of") is not None else []) + \
                       (["benign_note"] if (a["source"] == "note_text" and not a["error_type"]) else []):
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
            rows.append({"field": field, "ftype": ftype_of(field), "conf": conf, "ok": ok, "tags": tags,
                         "regime": regime, "answerable": answerable(c, field, ans)})
            qrows.append((c["case_id"], field, ok))
            n_ok += ok
            n += 1
        per_case_acc.append((n_ok, n))
        skip = set(c.get("undecidable_ids") or [])
        for a in c["assertions"]:
            if a["id"] in skip:
                continue
            pl = pa.get(str(a["id"]))
            a_true.append(a["label"])
            if pl:
                a_pred.append(pl.get("label", "valid"))
                pe, ps = float(pl.get("p_erroneous", 0.0)), float(pl.get("p_superseded", 0.0))
                pv_ = float(pl.get("p_valid", max(0.0, 1.0 - pe - ps)))
            else:
                a_pred.append("valid")
                pv_, ps, pe = 1.0, 0.0, 0.0
            p_err.append(pe)
            p_sup.append(ps)
            tot = pv_ + ps + pe
            dists.append([pv_ / tot, ps / tot, pe / tot] if tot > 0 else [1.0 / 3] * 3)
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
    # answerability (v0.5)
    ans_rows = [r for r in rows if r["answerable"]]
    una_rows = [r for r in rows if not r["answerable"]]
    res["answerable_share"] = len(ans_rows) / len(rows) if rows else float("nan")
    hi_a = [r for r in ans_rows if r["conf"] >= OVERWRITE_THRESHOLD]
    res["wrong_overwrite@0.9:answerable"] = (1 - sum(r["ok"] for r in hi_a) / len(hi_a)) if hi_a else float("nan")
    res["q_acc:answerable"] = sum(r["ok"] for r in ans_rows) / len(ans_rows) if ans_rows else float("nan")
    res["unanswerable_mean_conf"] = sum(r["conf"] for r in una_rows) / len(una_rows) if una_rows else float("nan")
    res["unanswerable_auto_rate"] = (sum(1 for r in una_rows if r["conf"] >= OVERWRITE_THRESHOLD) / len(una_rows)) if una_rows else float("nan")
    # by field type
    for ft in sorted(set(r["ftype"] for r in rows)):
        sub = [r for r in rows if r["ftype"] == ft]
        res["acc:ftype:" + ft] = sum(r["ok"] for r in sub) / len(sub) if sub else float("nan")
    # by regime (longitudinal / snapshot / real)
    for rg in sorted(set(r["regime"] for r in rows)):
        sub = [r for r in rows if r["regime"] == rg]
        res["acc:regime:" + rg] = sum(r["ok"] for r in sub) / len(sub) if sub else float("nan")
        res["n:regime:" + rg] = len(sub)
    # by conflict type
    for t in CONFLICT_TYPES:
        sub = [r for r in rows if t in r["tags"]]
        res["acc:conflict:" + t] = sum(r["ok"] for r in sub) / len(sub) if sub else float("nan")
        res["n:conflict:" + t] = len(sub)
    # assertion-level
    f1s, f1_present = [], []
    present = set(a_true)
    for lab in LABELS:
        pr, rc, f1 = prf(a_pred, a_true, lab)
        res["asrt_%s_precision" % lab] = pr
        res["asrt_%s_recall" % lab] = rc
        f1s.append(f1)
        if lab in present:
            f1_present.append(f1)
    res["asrt_macro_f1"] = sum(f1_present) / len(f1_present) if f1_present else float("nan")
    res["asrt_macro_f1_3"] = sum(f1s) / len(f1s)
    res["asrt_auc_erroneous"] = auc(p_err, y_err)
    res["asrt_auc_superseded"] = auc(p_sup, y_sup)
    # entry-label calibration (v0.5)
    if dists:
        ys = [LABELS.index(t) for t in a_true]
        res["entry_brier"] = sum(sum((d[k] - (1 if k == y else 0)) ** 2 for k in range(3)) for d, y in zip(dists, ys)) / len(dists)
        tops = [max(range(3), key=lambda k: d[k]) for d in dists]
        res["entry_ece"] = ece([d[t] for d, t in zip(dists, tops)], [1 if t == y else 0 for t, y in zip(tops, ys)])
        res["perr_ece"] = ece([d[2] for d in dists], y_err)
        res["entry_nll"] = -sum(math.log(min(1.0, max(1e-4, d[y]))) for d, y in zip(dists, ys)) / len(dists)
        freq = [ys.count(k) / len(ys) for k in range(3)]
        res["entry_nll_best_constant"] = -sum(math.log(max(1e-4, freq[y])) for y in ys) / len(ys)
    else:
        res["entry_brier"] = res["entry_ece"] = res["perr_ece"] = float("nan")
        res["entry_nll"] = res["entry_nll_best_constant"] = float("nan")
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
    res["_qrows"] = qrows
    return res


def paired(ra: Dict[str, Any], rb: Dict[str, Any]) -> Dict[str, float]:
    """A minus B on the queries both scored: q_acc difference, paired bootstrap 95% CI over cases, McNemar p."""
    ka = {(cid, f): ok for cid, f, ok in ra["_qrows"]}
    kb = {(cid, f): ok for cid, f, ok in rb["_qrows"]}
    keys = [k for k in ka if k in kb]
    if not keys:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan"), "p": float("nan"), "a_only": 0, "b_only": 0}
    by_case: Dict[str, List[int]] = {}
    a_only = b_only = 0
    for k in keys:
        x, y = ka[k], kb[k]
        s = by_case.setdefault(k[0], [0, 0, 0])
        s[0] += x; s[1] += y; s[2] += 1
        a_only += 1 if (x and not y) else 0
        b_only += 1 if (y and not x) else 0
    cases = list(by_case.values())
    n = sum(s[2] for s in cases)
    diff = (sum(s[0] for s in cases) - sum(s[1] for s in cases)) / n
    rng = random.Random(SEED + 1)
    ds = []
    for _ in range(PAIRED_BOOTSTRAP):
        smp = [cases[rng.randrange(len(cases))] for _ in cases]
        m = sum(s[2] for s in smp)
        ds.append((sum(s[0] for s in smp) - sum(s[1] for s in smp)) / m if m else 0.0)
    ds.sort()
    return {"diff": diff, "lo": ds[int(0.025 * len(ds))], "hi": ds[min(len(ds) - 1, int(0.975 * len(ds)))],
            "p": mcnemar_p(a_only, b_only), "a_only": a_only, "b_only": b_only}


# ============================================================ driver
def all_splits() -> Dict[str, str]:
    splits = dict(SPLITS)
    for path in sorted(glob.glob("cases_*.jsonl")):
        name = os.path.basename(path)[len("cases_"):-len(".jsonl")]
        splits.setdefault(name, path)
    return {k: v for k, v in splits.items() if os.path.exists(v) and k not in SKIP_SPLITS}


def pred_index() -> Dict[str, List[str]]:
    """{split: [method, ...]} -- each preds file is matched to the LONGEST split name it ends with, so
    preds_x_test_hard.jsonl belongs to test_hard, never to hard (method "x_test")."""
    names = set(SPLITS) | set(KNOWN_SPLITS) | set(os.path.basename(p)[len("cases_"):-len(".jsonl")]
                                                for p in glob.glob("cases_*.jsonl"))
    known = sorted(names, key=lambda s: -len(s))
    out: Dict[str, List[str]] = {}
    for path in sorted(glob.glob("preds_*.jsonl")):
        stem = os.path.basename(path)[len("preds_"):-len(".jsonl")]
        for s in known:
            if stem.endswith("_" + s) and len(stem) > len(s) + 1:
                out.setdefault(s, []).append(stem[:-len(s) - 1])
                break
    return out


def print_table(split: str, results: Dict[str, Dict[str, Any]], methods: List[str]) -> None:
    print("%-26s %6s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s"
          % ("method", "cases", "q_acc", "ci_lo", "ci_hi", "ECE", "Brier", "wrongOW", "cov@.9",
             "sel@80", "undecC", "macroF1", "AUCerr"))
    for m in methods:
        r = results[m]
        print("%-26s %6d %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s %7s"
              % (m[:26], r["n_cases"], fmt(r["q_acc"], True), fmt(r["q_acc_ci_lo"], True), fmt(r["q_acc_ci_hi"], True),
                 fmt(r["ece"]), fmt(r["brier"]), fmt(r["wrong_overwrite@0.9"], True), fmt(r["coverage@0.9"], True),
                 fmt(r["sel_acc@80"], True), fmt(r["undecidable_mean_conf"]), fmt(r["asrt_macro_f1"]),
                 fmt(r["asrt_auc_erroneous"])))
    print("\n-- answerability and entry calibration (answerable = some reading in the record matches the truth) --")
    print("%-26s %10s %12s %12s %11s %10s %10s %9s %9s %9s" % ("method", "acc(ans)", "wrongOW(ans)", "conf(unans)",
                                                              ">=.9(unans)", "entryECE", "entryBrier", "pErrECE",
                                                              "entryNLL", "macroF1(3)"))
    for m in methods:
        r = results[m]
        print("%-26s %10s %12s %12s %11s %10s %10s %9s %9s %9s" % (
            m[:26], fmt(r["q_acc:answerable"], True), fmt(r["wrong_overwrite@0.9:answerable"], True),
            fmt(r["unanswerable_mean_conf"]), fmt(r["unanswerable_auto_rate"], True), fmt(r["entry_ece"]),
            fmt(r["entry_brier"]), fmt(r["perr_ece"]), fmt(r["entry_nll"]), fmt(r["asrt_macro_f1_3"])))
    any_r = results[methods[0]]
    print("(entryNLL of the best constant label distribution for these entries: %s -- a model that learned nothing "
          "about entries cannot beat it)" % fmt(any_r["entry_nll_best_constant"]))


def print_validity(results: Dict[str, Dict[str, Any]], methods: List[str]) -> None:
    print("\n-- per-assertion validity (precision / recall) --")
    for m in methods:
        r = results[m]
        print("%-26s valid %5.2f/%5.2f  superseded %5.2f/%5.2f  erroneous %5.2f/%5.2f  AUC(sup) %s"
              % (m[:26], r["asrt_valid_precision"], r["asrt_valid_recall"], r["asrt_superseded_precision"],
                 r["asrt_superseded_recall"], r["asrt_erroneous_precision"], r["asrt_erroneous_recall"],
                 fmt(r["asrt_auc_superseded"])))


def print_paired(results: Dict[str, Dict[str, Any]], methods: List[str], rows_out: list, scope: str) -> None:
    if REFERENCE_METHOD not in results:
        return
    ref = results[REFERENCE_METHOD]
    print("\n-- paired vs %s (same queries; diff = method minus %s; bootstrap 95%% CI over cases; exact McNemar) --"
          % (REFERENCE_METHOD, REFERENCE_METHOD))
    print("%-26s %9s %18s %9s %10s" % ("method", "diff", "95% CI", "McNemar p", "wins/losses"))
    for m in methods:
        if m == REFERENCE_METHOD:
            continue
        pr = paired(results[m], ref)
        print("%-26s %+8.2f%% [%+6.2f, %+6.2f] %9.3f %5d/%-5d" % (
            m[:26], 100 * pr["diff"], 100 * pr["lo"], 100 * pr["hi"], pr["p"], pr["a_only"], pr["b_only"]))
        for k in ("diff", "lo", "hi", "p"):
            rows_out.append((scope, m, "paired_vs_%s:%s" % (REFERENCE_METHOD, k), pr[k]))


def pairwise_set(methods: List[str]) -> List[str]:
    out = []
    for m in methods:
        if re.search(r"_seed\d+$", m) or "permuted" in m:
            continue
        if m in PAIRWISE_METHODS or any(m.startswith(px) for px in PAIRWISE_PREFIXES):
            out.append(m)
    return out


def print_pairwise(results: Dict[str, Dict[str, Any]], methods: List[str], rows_out: list, scope: str) -> None:
    """Every pair among the reconciler, the feature model and the LLM forms, on the same queries (diff = first
    minus second; methods ordered by q_acc)."""
    ms = pairwise_set(methods)
    if len(ms) < 2 or (len(ms) == 2 and REFERENCE_METHOD in ms):   # that one pair is in the paired-vs-reference block
        return
    ms = sorted(ms, key=lambda m: -results[m]["q_acc"] if not math.isnan(results[m]["q_acc"]) else 0.0)
    llms = [m for m in ms if any(m.startswith(px) for px in PAIRWISE_PREFIXES)]
    print("\n-- pairwise (same queries; diff = first minus second; bootstrap 95%% CI over cases; exact McNemar)%s --"
          % ((" best LLM form here by q_acc: %s" % llms[0]) if llms else ""))
    print("%-41s %9s %18s %9s %10s" % ("pair", "diff", "95% CI", "McNemar p", "wins/losses"))
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            a, b = ms[i], ms[j]
            pr = paired(results[a], results[b])
            print("%-41s %+8.2f%% [%+6.2f, %+6.2f] %9.3f %5d/%-5d" % (
                ("%s - %s" % (abbr(a, 20), abbr(b, 18)))[:41], 100 * pr["diff"], 100 * pr["lo"], 100 * pr["hi"],
                pr["p"], pr["a_only"], pr["b_only"]))
            for k in ("diff", "lo", "hi", "p"):
                rows_out.append((scope, "%s|%s" % (a, b), "pairwise:%s" % k, pr[k]))


def print_seed_summary(results: Dict[str, Dict[str, Any]]) -> None:
    groups: Dict[str, List[str]] = {}
    for m in results:
        mm = re.match(r"^(.*)_seed(\d+)$", m)
        if mm and mm.group(1) in results:
            groups.setdefault(mm.group(1), [mm.group(1)]).append(m)
    if not groups:
        return
    print("\n-- replicates (mean +- sd over seeds) --")
    for base, ms in sorted(groups.items()):
        line = "%-26s n=%d" % (base[:26], len(ms))
        for key, pct in (("q_acc", True), ("ece", False), ("wrong_overwrite@0.9", True), ("asrt_macro_f1", False),
                         ("asrt_auc_erroneous", False)):
            vals = [results[m][key] for m in ms if not math.isnan(results[m][key])]
            if not vals:
                continue
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
            line += " | %s %s +- %s" % (key, ("%.2f%%" % (100 * mu)) if pct else ("%.3f" % mu),
                                        ("%.2f" % (100 * sd)) if pct else ("%.3f" % sd))
        print(line)


def score_block(split: str, cases: List[Dict[str, Any]], loaded: Dict[str, List[Dict[str, Any]]],
                methods: List[str], ids: Optional[set], scope: str, all_rows: list,
                cover: Dict[str, set]) -> Dict[str, Dict[str, Any]]:
    results = {}
    for m in methods:
        mids = cover[m] if ids is None else (cover[m] & ids)
        preds = [p for p in loaded[m] if p["case_id"] in mids]
        sub = [c for c in cases if c["case_id"] in mids]
        results[m] = score(sub, preds)
        results[m]["n_cases"] = len(sub)
        for k, v in results[m].items():
            if not k.startswith("_"):
                all_rows.append((scope, m, k, v))
    return results


def main() -> None:
    all_rows = []
    index = pred_index()
    for split, cpath in all_splits().items():
        methods = sorted(index.get(split, []))
        if not methods:
            continue
        cases = load_jsonl(cpath)
        case_ids = set(c["case_id"] for c in cases)
        loaded = {m: load_jsonl("preds_%s_%s.jsonl" % (m, split)) for m in methods}
        for m in list(methods):
            stray = [p["case_id"] for p in loaded[m] if p["case_id"] not in case_ids]
            if stray and len(stray) == len(loaded[m]):
                print("\n*** WARNING: preds_%s_%s.jsonl shares no case id with %s (e.g. %s) -- a prediction file from "
                      "another version or split? It is left out of the %s tables. ***" % (m, split, cpath, stray[0], split))
                methods.remove(m)
            elif stray:
                print("WARNING: preds_%s_%s.jsonl has %d case ids not in %s (e.g. %s) -- an old prediction file?"
                      % (m, split, len(stray), cpath, stray[0]))
        if not methods:
            continue
        empty_ids = set(c["case_id"] for c in cases if not c["queries"] and not c["assertions"])
        cover = {m: set(p["case_id"] for p in loaded[m]) | empty_ids for m in methods}
        common = None
        if COMMON_CASES_ONLY:
            for m in methods:
                common = set(cover[m]) if common is None else (common & cover[m])
            common &= case_ids
        n_cal = sum(1 for m in methods if loaded[m] and all("calibrated" in p for p in loaded[m]))
        cal_note = ""
        if n_cal == len(methods):
            cal_note = " (all methods temperature-scaled on val)"
        elif n_cal:
            cal_note = " (%d of %d methods temperature-scaled on val; raw: %s)" % (
                n_cal, len(methods), ", ".join(m for m in methods if not (loaded[m] and all("calibrated" in p for p in loaded[m]))))
        results = score_block(split, cases, loaded, methods, common, split, all_rows, cover)
        any_r = results[methods[0]]
        print("\n=== %s split%s ===" % (split, cal_note))
        if common is not None and len(common) < len(cases):
            print("(scored on the %d cases every method covered; the full-split table for methods covering every case follows)"
                  % len(common))
        print("answerable decidable queries: %s of %d (the ceiling for any method that states a value from the record)"
              % (fmt(any_r["answerable_share"], True), any_r["n_queries"]))
        print_table(split, results, methods)
        has_conflicts = any(results[methods[0]]["n:conflict:" + t] > 0 for t in CONFLICT_TYPES)
        if has_conflicts:
            print("\n-- accuracy by conflict type (queries on fields affected by that conflict) --")
            header = "%-16s" % "conflict" + "".join("%14s" % abbr(m) for m in methods)
            print(header)
            for t in CONFLICT_TYPES:
                n = results[methods[0]]["n:conflict:" + t]
                if n == 0:
                    continue
                line = "%-16s" % ("%s (n=%d)" % (t, n))[:16]
                for m in methods:
                    line += "%14s" % fmt(results[m]["acc:conflict:" + t], True)
                print(line)
        regimes = sorted(set(k[len("acc:regime:"):] for m in methods for k in results[m] if k.startswith("acc:regime:")))
        if len(regimes) > 1:
            print("\n-- accuracy by regime --")
            print("%-16s" % "regime" + "".join("%14s" % abbr(m) for m in methods))
            for rg in regimes:
                n = results[methods[0]].get("n:regime:" + rg, 0)
                line = "%-16s" % ("%s (n=%d)" % (rg, n))[:16]
                for m in methods:
                    line += "%14s" % fmt(results[m].get("acc:regime:" + rg, float("nan")), True)
                print(line)
        print("\n-- accuracy by field type --")
        print("%-16s" % "field type" + "".join("%14s" % abbr(m) for m in methods))
        ftypes = sorted(set(k[len("acc:ftype:"):] for m in methods for k in results[m] if k.startswith("acc:ftype:")))
        for ft in ftypes:
            line = "%-16s" % ft
            for m in methods:
                line += "%14s" % fmt(results[m].get("acc:ftype:" + ft, float("nan")), True)
            print(line)
        print_validity(results, methods)
        print_paired(results, methods, all_rows, split)
        print_pairwise(results, methods, all_rows, split)
        print_seed_summary(results)
        # full split for the methods that cover every case (when an LLM run covers a subset)
        if common is not None and len(common) < len(cases):
            full = [m for m in methods if case_ids <= cover[m]]
            if len(full) >= 1:
                scope = split + ":all"
                fres = score_block(split, cases, loaded, full, case_ids, scope, all_rows, cover)
                print("\n=== %s split, all %d cases (methods covering every case) ===" % (split, len(cases)))
                print_table(split, fres, full)
                print_validity(fres, full)          # Parliament amendment: per-label numbers on the full split too
                print_paired(fres, full, all_rows, scope)
                print_pairwise(fres, full, all_rows, scope)
                print_seed_summary(fres)

    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "method", "metric", "value"])
        for split, m, k, v in all_rows:
            w.writerow([split, m, k, "" if (isinstance(v, float) and math.isnan(v)) else v])
    print("\nwrote %s (%d rows)" % (RESULTS_CSV, len(all_rows)))


if __name__ == "__main__":
    main()
