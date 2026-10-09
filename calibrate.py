# calibrate.py -- one post-hoc temperature scaling for EVERY method (recbench v0.4)
# Run from PyCharm after every method has written its predictions (baselines.py, feature_baseline.py,
# train_reconciler.py, llm_baseline.py -- the LLM needs a run on the val split too). Python 3.9, stdlib only.
#
# For each method with predictions on `val`: fit one temperature for the query confidence and one for the
# per-assertion label distribution by minimizing negative log-likelihood on val, then apply the two numbers to
# that method's predictions on every split (including the real sets -- no real labels are used anywhere).
# The raw files are kept as raw_preds_<method>_<split>.jsonl; preds_<method>_<split>.jsonl becomes the scaled
# version that score.py reads. Re-running is safe: it always starts from the raw files.
#
# Temperature scaling never changes which value is chosen or which label is predicted, only how confident the
# method is -- so q_acc and macro-F1 are untouched; ECE, Brier, wrong-overwrite and coverage move.

import glob
import json
import math
import os
import shutil
from typing import Any, Dict, List, Tuple

from recbench_common import LABELS, load_jsonl, match, norm_value, write_jsonl
from score import ece as ece_score

# ============================================================ settings
VAL_SPLIT = "val"
VAL_CASES = "cases_val.jsonl"
OUT_JSON = "calibration.json"
CLIP = 1e-3               # probabilities are clipped to [CLIP, 1-CLIP] before taking logits (rules emit exact 0/1)
T_RANGE = (-4.0, 4.0)     # search range for log(T)
OVERWRITE_THRESHOLD = 0.90


# ============================================================ helpers
def clip(p: float) -> float:
    return min(1.0 - CLIP, max(CLIP, float(p)))


def logit(p: float) -> float:
    p = clip(p)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def scale_binary(p: float, T: float) -> float:
    return sigmoid(logit(p) / T)


def scale_dist(ps: List[float], T: float) -> List[float]:
    logs = [math.log(clip(p)) / T for p in ps]
    mx = max(logs)
    ex = [math.exp(v - mx) for v in logs]
    z = sum(ex)
    return [e / z for e in ex]


def golden(f, lo: float, hi: float, iters: int = 60) -> float:
    g = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def fit_T_binary(ps: List[float], ys: List[int]) -> float:
    if not ps:
        return 1.0
    def nll(logT):
        T = math.exp(logT)
        return -sum(math.log(clip(scale_binary(p, T)) if y else clip(1 - scale_binary(p, T))) for p, y in zip(ps, ys)) / len(ps)
    return math.exp(golden(nll, T_RANGE[0], T_RANGE[1]))


def fit_T_dist(dists: List[List[float]], ys: List[int]) -> float:
    if not dists:
        return 1.0
    def nll(logT):
        T = math.exp(logT)
        return -sum(math.log(clip(scale_dist(d, T)[y])) for d, y in zip(dists, ys)) / len(dists)
    return math.exp(golden(nll, T_RANGE[0], T_RANGE[1]))


# ============================================================ data access
def raw_path(method: str, split: str) -> str:
    return "raw_preds_%s_%s.jsonl" % (method, split)


def pred_path(method: str, split: str) -> str:
    return "preds_%s_%s.jsonl" % (method, split)


def ensure_raw(method: str, split: str) -> str:
    """Back up the untouched predictions once; always return the raw file path.
    If the preds file has grown since the backup (a resumable run appended cases after calibrate.py ran), the new,
    unscaled records are merged into the backup so nothing is lost."""
    rp, pp = raw_path(method, split), pred_path(method, split)
    if not os.path.exists(rp):
        shutil.copyfile(pp, rp)
        return rp
    if os.path.exists(pp):
        raw_ids = set(p["case_id"] for p in load_jsonl(rp))
        extra = [p for p in load_jsonl(pp) if p["case_id"] not in raw_ids and "calibrated" not in p]
        if extra:
            with open(rp, "a", encoding="utf-8") as f:
                for p in extra:
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            print("  %s/%s: merged %d new cases into the raw backup" % (method, split, len(extra)))
    return rp


def known_splits() -> List[str]:
    """Every split with a cases file, longest name first (so book_subset is matched before book)."""
    names = [os.path.basename(p)[len("cases_"):-len(".jsonl")] for p in glob.glob("cases_*.jsonl")]
    return sorted(set(names), key=lambda s: -len(s))


def split_method(filename: str):
    """'preds_<method>_<split>.jsonl' / 'raw_preds_...' -> (method, split) using the known split names."""
    name = os.path.basename(filename)
    name = name[len("raw_preds_"):] if name.startswith("raw_") else name[len("preds_"):]
    name = name[:-len(".jsonl")]
    for s in known_splits():
        if name.endswith("_" + s):
            return name[:-len(s) - 1], s
    return None, None


def methods_with_val() -> List[str]:
    out = set()
    for path in glob.glob("preds_*_%s.jsonl" % VAL_SPLIT) + glob.glob("raw_preds_*_%s.jsonl" % VAL_SPLIT):
        m, s = split_method(path)
        if m and s == VAL_SPLIT:
            out.add(m)
    return sorted(out)


def splits_of(method: str) -> List[str]:
    out = []
    for s in known_splits():
        if os.path.exists(pred_path(method, s)) or os.path.exists(raw_path(method, s)):
            out.append(s)
    return sorted(out)


def query_pairs(cases: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Tuple[List[float], List[int]]:
    pmap = {p["case_id"]: p for p in preds}
    ps, ys = [], []
    for c in cases:
        p = pmap.get(c["case_id"])
        if not p:
            continue
        for q in c["queries"]:
            if not q["decidable"]:
                continue
            item = p["queries"].get(q["field"])
            if not item or item.get("value") is None:
                continue
            try:
                ok = 1 if match(q["field"], norm_value(q["field"], item["value"]), norm_value(q["field"], q["answer"])) else 0
            except (ValueError, TypeError):
                continue
            ps.append(float(item.get("confidence", 0.0)))
            ys.append(ok)
    return ps, ys


def entry_pairs(cases: List[Dict[str, Any]], preds: List[Dict[str, Any]]) -> Tuple[List[List[float]], List[int]]:
    pmap = {p["case_id"]: p for p in preds}
    ds, ys = [], []
    for c in cases:
        p = pmap.get(c["case_id"])
        if not p:
            continue
        for a in c["assertions"]:
            item = p["assertions"].get(str(a["id"]))
            if not item:
                continue
            ds.append([float(item.get("p_valid", 0.0)), float(item.get("p_superseded", 0.0)), float(item.get("p_erroneous", 0.0))])
            ys.append(LABELS.index(a["label"]))
    return ds, ys


def apply(preds: List[Dict[str, Any]], Tq: float, Te: float) -> List[Dict[str, Any]]:
    out = []
    for p in preds:
        rec = {"case_id": p["case_id"], "method": p["method"], "queries": {}, "assertions": {},
               "calibrated": {"T_query": round(Tq, 4), "T_entry": round(Te, 4)}}
        for f, item in p["queries"].items():
            new = dict(item)
            try:
                new["confidence"] = round(scale_binary(float(item.get("confidence", 0.0)), Tq), 4)
            except (ValueError, TypeError):
                pass
            rec["queries"][f] = new
        for aid, item in p["assertions"].items():
            d = scale_dist([float(item.get("p_valid", 0.0)), float(item.get("p_superseded", 0.0)),
                            float(item.get("p_erroneous", 0.0))], Te)
            rec["assertions"][aid] = {"label": item.get("label", LABELS[d.index(max(d))]),
                                      "p_valid": round(d[0], 4), "p_superseded": round(d[1], 4), "p_erroneous": round(d[2], 4)}
        out.append(rec)
    return out


def wrong_ow(ps: List[float], ys: List[int]) -> float:
    hi = [y for p, y in zip(ps, ys) if p >= OVERWRITE_THRESHOLD]
    return (1 - sum(hi) / len(hi)) if hi else float("nan")


def fmt(x: float) -> str:
    return "  -  " if (x != x) else "%.3f" % x


# ============================================================ driver
def main() -> None:
    val_cases = load_jsonl(VAL_CASES)
    methods = methods_with_val()
    if not methods:
        print("no predictions on the %s split found; run the methods on cases_val.jsonl first" % VAL_SPLIT)
        return
    all_methods = sorted(set(m for m, _ in (split_method(p) for p in glob.glob("preds_*.jsonl")) if m))
    skipped = [m for m in all_methods if m not in methods]
    report: Dict[str, Any] = {}
    print("%-20s %7s %7s | %s" % ("method", "T_query", "T_entry", "ECE / wrong-OW@0.9 before -> after, per split"))
    for m in methods:
        raw_val = load_jsonl(ensure_raw(m, VAL_SPLIT))
        ps, ys = query_pairs(val_cases, raw_val)
        ds, ls = entry_pairs(val_cases, raw_val)
        Tq, Te = fit_T_binary(ps, ys), fit_T_dist(ds, ls)
        report[m] = {"T_query": round(Tq, 4), "T_entry": round(Te, 4), "val_n_queries": len(ps), "splits": {}}
        line = "%-20s %7.3f %7.3f |" % (m, Tq, Te)
        for split in splits_of(m):
            cases_path = "cases_%s.jsonl" % split
            raw = load_jsonl(ensure_raw(m, split))
            scaled = apply(raw, Tq, Te)
            write_jsonl(pred_path(m, split), scaled)
            if os.path.exists(cases_path):
                cases = load_jsonl(cases_path)
                p0, y0 = query_pairs(cases, raw)
                p1, y1 = query_pairs(cases, scaled)
                e0, e1 = ece_score(p0, y0), ece_score(p1, y1)
                w0, w1 = wrong_ow(p0, y0), wrong_ow(p1, y1)
                report[m]["splits"][split] = {"ece_before": e0, "ece_after": e1, "wrong_ow_before": w0, "wrong_ow_after": w1}
                line += " %s %s/%s->%s/%s" % (split[:5], fmt(e0), fmt(w0), fmt(e1), fmt(w1))
        print(line)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    if skipped:
        print("\nNOT calibrated (no %s predictions): %s" % (VAL_SPLIT, ", ".join(skipped)))
    print("\nwrote %s. Now run score.py (it reads the scaled preds_* files; raw copies are raw_preds_*)." % OUT_JSON)


if __name__ == "__main__":
    main()
