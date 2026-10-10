# flight_diag.py -- why does the reconciler lose to majority on Flight? (recbench v0.5; written for v0.4's Flight result)
# v0.5 fix: the per-source table now prints for cases_flight_dev.jsonl (it looked only for the v0.4 "flight" key).
# Read-only; changes nothing in the benchmark. Run from PyCharm's Run button after score.py (same folder).
# Python 3.9, stdlib + recbench_common.py (+ gen.py, only to print the generator's snapshot knob ranges).
#
# Reads:  cases_flight.jsonl                      (from realdata_map.py)
#         cases_snapshot.jsonl                    (the generated snapshot regime, for the side-by-side)
#         preds_<MODEL>_<split>.jsonl, preds_<REF>_<split>.jsonl   (disagreement listing; skipped if missing)
# Prints: (1) the same structure table for Flight and for the generated snapshot cases, next to the generator's knobs,
#         so every number that falls outside the generated range is visible;
#         (2) per-source accuracy on numeric fields (Flight: times) vs categorical fields (Flight: gates), pooled;
#         (3) where MODEL and REF disagree: who was right, whether the model overrides correct majorities, and whether
#             the sources it sided with were the ones that were accurate on the OTHER fields of the same case.
# Only Flight days <= DEV_LAST_DAY are read while DEV_ONLY = True, so the later days stay untouched (flight_test).

import math
import os
import statistics
from typing import Any, Dict, List, Optional, Tuple

from recbench_common import cluster, ftype_of, load_jsonl, match, norm_value, register_field_types

MODEL = "reconciler"           # the method under study
REF = "majority_vote"          # the method it is compared with
DEV_ONLY = True                # True: Flight cases from days after DEV_LAST_DAY are ignored (kept for the final test)
DEV_LAST_DAY = "2011-12-15"    # Flight is one file per day, Dec 2011; case_id = flight-<YYYY-MM-DD>-<flight>
N_LIST = 20                    # how many categorical (gate) disagreements to print in full
# (split name, cases file, label). v0.5 writes Flight as flight_dev / flight_test; the v0.4 all-days file is the fallback.
DATASETS = [("flight_dev", "cases_flight_dev.jsonl", "Flight (dev days)"),
            ("flight", "cases_flight.jsonl", "Flight"),
            ("snapshot", "cases_snapshot.jsonl", "generated snapshot")]

try:
    from gen import KNOBS
    _K = KNOBS["train"]
    KNOB_TEXT = {
        "sources": "snap_sources %s" % (_K["snap_sources"],), "fields": "snap_fields %s" % (_K["snap_fields"],),
        "cover": "snap_cover %s" % (_K["snap_cover"],), "mean_acc": "snap_mean_acc %s" % (_K["snap_mean_acc"],),
        "acc_sd": "snap_acc_sd %s" % (_K["snap_acc_sd"],), "copy": "snap_copy %s" % (_K["snap_copy"],),
        "pool": "snap_pool %s" % (_K["snap_pool"],),
        "corr": "snap_field_corr %s (1.0 = one accuracy per source)" % (_K.get("snap_field_corr", (1.0, 1.0)),)}
except Exception:  # gen.py absent or different version
    KNOB_TEXT = {"sources": "snap_sources (5, 40)", "fields": "snap_fields (2, 6)", "cover": "snap_cover (0.3, 1.0)",
                 "mean_acc": "snap_mean_acc (0.35, 0.95)", "acc_sd": "snap_acc_sd 0.20", "copy": "snap_copy (0.2, 0.9)",
                 "pool": "snap_pool (1, 2)",
                 "corr": "latent 1.0 (one accuracy per source); compare the two measured columns, not to 1.0"}


# ---------------------------------------------------------------- helpers
def group_of(field: str) -> str:
    return "num" if ftype_of(field) in ("num_rel", "num_abs") else "cat"


def pct(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = mean(xs), mean(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def fmt(x: Any, nd: int = 2) -> str:
    if isinstance(x, float):
        return "nan" if math.isnan(x) else ("%%.%df" % nd) % x
    return str(x)


def day_of(case_id: str) -> Optional[str]:
    parts = case_id.split("-")
    if len(parts) >= 4 and parts[1].isdigit() and len(parts[1]) == 4:
        return "-".join(parts[1:4])
    return None


# ---------------------------------------------------------------- per-case analysis
def analyze_case(c: Dict[str, Any]) -> Dict[str, Any]:
    """Structure of one case: clusters per query, per-source accuracy by field group, coverage, copy structure."""
    register_field_types(c)
    truth = {q["field"]: norm_value(q["field"], q["answer"]) for q in c["queries"]}
    by_field: Dict[str, List[Tuple[str, Any]]] = {}
    src_ok: Dict[str, Dict[str, List[int]]] = {}        # source -> group -> [ok, ...]
    src_field_ok: Dict[str, Dict[str, int]] = {}        # source -> field -> ok   (for "accuracy elsewhere")
    src_fields: Dict[str, set] = {}
    for a in c["assertions"]:
        f = a["field"]
        if f not in truth:
            continue
        try:
            v = norm_value(f, a["value"])
        except (ValueError, TypeError):
            continue
        ok = 1 if match(f, v, truth[f]) else 0
        by_field.setdefault(f, []).append((a["source"], v))
        src_ok.setdefault(a["source"], {}).setdefault(group_of(f), []).append(ok)
        src_field_ok.setdefault(a["source"], {})[f] = ok
        src_fields.setdefault(a["source"], set()).add(f)
    queries = []
    for f, items in by_field.items():
        cl = cluster(f, items)
        sizes = sorted((len(x["members"]) for x in cl), reverse=True)
        top = sizes[0]
        truth_cl = next((x for x in cl if match(f, x["value"], truth[f])), None)
        truth_n = len(truth_cl["members"]) if truth_cl else 0
        n_top = sum(1 for s in sizes if s == top)
        wrong_cl = [x for x in cl if x is not truth_cl]
        n_wrong = sum(len(x["members"]) for x in wrong_cl)
        n_wrong_shared = sum(len(x["members"]) for x in wrong_cl if len(x["members"]) >= 2)
        queries.append({
            "field": f, "group": group_of(f), "n": len(items), "n_clusters": len(cl), "top_share": top / len(items),
            "truth_share": truth_n / len(items), "majority_strict": int(truth_n == top and n_top == 1),
            "majority_tie": int(truth_n == top and n_top > 1), "n_wrong": n_wrong, "n_wrong_shared": n_wrong_shared,
            "n_wrong_clusters": len(wrong_cl), "clusters": cl, "truth": truth[f]})
    n_fields = len(truth)
    src_acc = {s: mean([ok for g in d.values() for ok in g]) for s, d in src_ok.items()}
    src_n = {s: sum(len(g) for g in d.values()) for s, d in src_ok.items()}
    cover = [len(src_fields[s]) / n_fields for s in src_fields] if n_fields else []
    pairs = [(mean(d["num"]), mean(d["cat"])) for d in src_ok.values() if "num" in d and "cat" in d]
    return {"case_id": c["case_id"], "n_sources": len(src_fields), "n_fields": n_fields,
            "n_assertions": sum(len(v) for v in by_field.values()), "queries": queries, "cover": cover,
            "src_acc": src_acc, "src_n": src_n, "src_ok": src_ok, "src_field_ok": src_field_ok, "pairs": pairs}


def structure_table(name: str, infos: List[Dict[str, Any]]) -> Dict[str, str]:
    """One column of the side-by-side table."""
    col: Dict[str, str] = {}
    col["cases"] = "%d" % len(infos)
    col["sources per case (mean, 5-95%)"] = "%s (%s-%s)" % (fmt(mean([i["n_sources"] for i in infos]), 1),
                                                           fmt(pct([float(i["n_sources"]) for i in infos], 0.05), 0),
                                                           fmt(pct([float(i["n_sources"]) for i in infos], 0.95), 0))
    col["fields per case (mean, min-max)"] = "%s (%d-%d)" % (fmt(mean([i["n_fields"] for i in infos]), 1),
                                                           min(i["n_fields"] for i in infos), max(i["n_fields"] for i in infos))
    col["assertions per case (mean)"] = fmt(mean([i["n_assertions"] for i in infos]), 1)
    col["coverage per source (mean, 5-95%)"] = "%s (%s-%s)" % (fmt(mean([x for i in infos for x in i["cover"]])),
                                                              fmt(pct([x for i in infos for x in i["cover"]], 0.05)),
                                                              fmt(pct([x for i in infos for x in i["cover"]], 0.95)))
    qs = [q for i in infos for q in i["queries"]]
    n_all = sum(q["n"] for q in qs)
    col["erroneous share of assertions"] = fmt(sum(q["n_wrong"] for q in qs) / max(1, n_all))
    case_mean_acc = [mean(list(i["src_acc"].values())) for i in infos if i["src_acc"]]
    col["per-case mean source accuracy (5/50/95%)"] = "%s / %s / %s" % (fmt(pct(case_mean_acc, 0.05)), fmt(pct(case_mean_acc, 0.5)),
                                                                        fmt(pct(case_mean_acc, 0.95)))
    sds = []
    for i in infos:
        accs = [i["src_acc"][s] for s in i["src_acc"] if i["src_n"][s] >= 2]
        if len(accs) >= 2:
            sds.append(statistics.pstdev(accs))
    col["within-case sd of source accuracy (median)"] = fmt(pct(sds, 0.5))
    pairs = [p for i in infos for p in i["pairs"]]
    col["within-case corr(source acc on num, on cat)"] = "%s (n=%d source-cases)" % (
        fmt(pearson([p[0] for p in pairs], [p[1] for p in pairs])), len(pairs))
    n_wrong = sum(q["n_wrong"] for q in qs)
    col["share of wrong assertions whose value >=2 sources share"] = fmt(sum(q["n_wrong_shared"] for q in qs) / max(1, n_wrong))
    wc = [float(q["n_wrong_clusters"]) for q in qs]
    col["distinct wrong values per query (mean, p90)"] = "%s (%s)" % (fmt(mean(wc)), fmt(pct(wc, 0.9), 0))
    col["top value share (mean)"] = fmt(mean([q["top_share"] for q in qs]))
    col["truth share of assertions (mean)"] = fmt(mean([q["truth_share"] for q in qs]))
    for g, label in (("num", "numeric fields (Flight: times)"), ("cat", "categorical fields (Flight: gates)")):
        gq = [q for q in qs if q["group"] == g]
        if not gq:
            col["majority strictly correct, %s" % label] = "-"
            continue
        col["queries, %s" % label] = "%d" % len(gq)
        col["majority strictly correct, %s" % label] = "%s (tied %s)" % (fmt(mean([q["majority_strict"] for q in gq])),
                                                                         fmt(mean([q["majority_tie"] for q in gq])))
        col["distinct values per query, %s" % label] = fmt(mean([q["n_clusters"] for q in gq]), 1)
        col["erroneous share, %s" % label] = fmt(sum(q["n_wrong"] for q in gq) / max(1, sum(q["n"] for q in gq)))
    return col


ROW_KNOB = {"sources per case (mean, 5-95%)": "sources", "fields per case (mean, min-max)": "fields",
            "coverage per source (mean, 5-95%)": "cover", "per-case mean source accuracy (5/50/95%)": "mean_acc",
            "within-case sd of source accuracy (median)": "acc_sd", "within-case corr(source acc on num, on cat)": "corr",
            "share of wrong assertions whose value >=2 sources share": "copy",
            "distinct wrong values per query (mean, p90)": "pool"}


# ---------------------------------------------------------------- per-source table (Flight: named sources recur)
def source_table(infos: List[Dict[str, Any]]) -> None:
    pooled: Dict[str, Dict[str, List[int]]] = {}
    for i in infos:
        for s, d in i["src_ok"].items():
            for g, oks in d.items():
                pooled.setdefault(s, {}).setdefault(g, []).extend(oks)
    rows = []
    for s, d in pooled.items():
        n_num, n_cat = len(d.get("num", [])), len(d.get("cat", []))
        rows.append((s, n_num, mean(d.get("num", [])), n_cat, mean(d.get("cat", []))))
    rows.sort(key=lambda r: -(r[1] + r[3]))
    print("\n  per-source accuracy pooled over cases (numeric = times, categorical = gates):")
    print("  %-28s %7s %8s %7s %8s" % ("source", "n_num", "acc_num", "n_cat", "acc_cat"))
    for s, n1, a1, n2, a2 in rows:
        print("  %-28s %7d %8s %7d %8s" % (s[:28], n1, fmt(a1), n2, fmt(a2)))
    both = [(a1, a2) for _s, n1, a1, n2, a2 in rows if n1 >= 20 and n2 >= 20]
    print("  pooled corr(acc_num, acc_cat) across sources with >=20 of each: %s (n=%d)" % (
        fmt(pearson([b[0] for b in both], [b[1] for b in both])), len(both)))


# ---------------------------------------------------------------- disagreements
def pred_map(path: str) -> Optional[Dict[str, Dict[str, Any]]]:
    if not os.path.exists(path):
        return None
    return {p["case_id"]: p for p in load_jsonl(path)}


def elsewhere_acc(info: Dict[str, Any], sources: List[str], field: str) -> float:
    vals = [ok for s in sources for f, ok in info["src_field_ok"].get(s, {}).items() if f != field]
    return mean(vals)


def disagreements(name: str, infos: List[Dict[str, Any]], pm: Dict[str, Dict[str, Any]], pr: Dict[str, Dict[str, Any]]) -> None:
    stats: Dict[str, Dict[str, int]] = {"num": {}, "cat": {}}
    bucket: Dict[Tuple[str, int], List[Tuple[int, int]]] = {}     # (group, majority_strict) -> [(model ok, ref ok)]
    listing = []
    side_model, side_ref, smaller = [], [], []
    for info in infos:
        m, r = pm.get(info["case_id"]), pr.get(info["case_id"])
        if not m or not r:
            continue
        for q in info["queries"]:
            f = q["field"]
            qm, qr = m["queries"].get(f), r["queries"].get(f)
            if not qm or not qr or qm.get("value") is None or qr.get("value") is None:
                continue
            try:
                vm, vr = norm_value(f, qm["value"]), norm_value(f, qr["value"])
            except (ValueError, TypeError):
                continue
            ok_m, ok_r = int(match(f, vm, q["truth"])), int(match(f, vr, q["truth"]))
            st = stats[q["group"]]
            st["queries"] = st.get("queries", 0) + 1
            st["model_ok"] = st.get("model_ok", 0) + ok_m
            st["ref_ok"] = st.get("ref_ok", 0) + ok_r
            bucket.setdefault((q["group"], q["majority_strict"]), []).append((ok_m, ok_r))
            if match(f, vm, vr):
                continue
            st["disagree"] = st.get("disagree", 0) + 1
            key = "model_right" if ok_m and not ok_r else ("ref_right" if ok_r and not ok_m else ("both_right" if ok_m else "neither"))
            st[key] = st.get(key, 0) + 1
            cl_m = next((x for x in q["clusters"] if match(f, x["value"], vm)), None)
            cl_r = next((x for x in q["clusters"] if match(f, x["value"], vr)), None)
            src_m = cl_m["members"] if cl_m else []
            src_r = cl_r["members"] if cl_r else []
            e_m, e_r = elsewhere_acc(info, src_m, f), elsewhere_acc(info, src_r, f)
            if q["group"] == "cat":
                if not math.isnan(e_m) and not math.isnan(e_r):
                    side_model.append(e_m)
                    side_ref.append(e_r)
                smaller.append(int(len(src_m) < len(src_r)))
                listing.append((info["case_id"], f, q["truth"], qm["value"], float(qm.get("confidence", 0.0)), ok_m,
                                qr["value"], ok_r, len(src_m), len(src_r), e_m, e_r,
                                sorted(((len(x["members"]), str(x["value"])) for x in q["clusters"]), reverse=True)[:4]))
    print("\n  %s vs %s on %s:" % (MODEL, REF, name))
    print("  %-28s %8s %9s %8s %8s %11s %9s %10s %8s" % ("fields", "queries", "model_acc", "ref_acc", "disagree",
                                                       "model_right", "ref_right", "both_right", "neither"))
    for g, label in (("num", "numeric (times)"), ("cat", "categorical (gates)")):
        st = stats[g]
        if not st.get("queries"):
            continue
        print("  %-28s %8d %9s %8s %8d %11d %9d %10d %8d" % (
            label, st["queries"], fmt(st["model_ok"] / st["queries"], 3), fmt(st["ref_ok"] / st["queries"], 3),
            st.get("disagree", 0), st.get("model_right", 0), st.get("ref_right", 0), st.get("both_right", 0), st.get("neither", 0)))
    print("  accuracy by whether the plain majority is strictly correct on that query:")
    for (g, ms), pairs in sorted(bucket.items()):
        print("    %-20s majority %-10s n=%5d  model %s  ref %s" % (
            "numeric" if g == "num" else "categorical", "correct" if ms else "wrong/tied", len(pairs),
            fmt(mean([p[0] for p in pairs]), 3), fmt(mean([p[1] for p in pairs]), 3)))
    if listing:
        print("  categorical disagreements: model picked the smaller cluster in %s of %d; mean accuracy-on-other-fields of the"
              " sources behind the model's value %s vs behind %s's value %s" % (
                  fmt(mean([float(x) for x in smaller])), len(smaller), fmt(mean(side_model)), REF, fmt(mean(side_ref))))
        print("  first %d categorical disagreements (M = model, R = %s; 'else' = those sources' accuracy on the case's other fields):"
              % (min(N_LIST, len(listing)), REF))
        for row in listing[:N_LIST]:
            cid, f, truth, vm, cm, okm, vr, okr, nm, nr, em, er, top = row
            print("    %-26s %-16s truth=%-8s M=%-8s(%.2f %s n=%d else=%s)  R=%-8s(%s n=%d else=%s)  votes=%s" % (
                cid[:26], f[-16:], str(truth)[:8], str(vm)[:8], cm, "ok" if okm else "X", nm, fmt(em),
                str(vr)[:8], "ok" if okr else "X", nr, fmt(er), " ".join("%s:%d" % (v[:6], n) for n, v in top)))


# ---------------------------------------------------------------- main
def main() -> None:
    columns: Dict[str, Dict[str, str]] = {}
    infos_by: Dict[str, List[Dict[str, Any]]] = {}
    for key, path, label in DATASETS:
        if not os.path.exists(path):
            print("%s: %s not found, skipped" % (label, path))
            continue
        cases = load_jsonl(path)
        if key == "flight" and os.path.exists("cases_flight_dev.jsonl"):
            continue                      # the v0.5 dev file is already in; skip the v0.4 all-days file
        if key == "flight" and DEV_ONLY:
            days = sorted(set(d for d in (day_of(c["case_id"]) for c in cases) if d))
            kept = [c for c in cases if (day_of(c["case_id"]) or "") <= DEV_LAST_DAY]
            print("Flight: %d cases over %d days (%s .. %s); DEV_ONLY keeps %d cases from days <= %s, leaves %d untouched"
                  % (len(cases), len(days), days[0] if days else "?", days[-1] if days else "?", len(kept), DEV_LAST_DAY,
                     len(cases) - len(kept)))
            cases = kept
            label = "Flight (dev days)"
        infos = [analyze_case(c) for c in cases]
        infos = [i for i in infos if i["queries"]]
        infos_by[key] = infos
        columns[label] = structure_table(label, infos)

    if columns:
        names = list(columns)
        rows = []
        for col in columns.values():
            for k in col:
                if k not in rows:
                    rows.append(k)
        w = max(len(r) for r in rows) + 2
        print("\n=== structure, side by side (generator knob ranges are the train split's) ===")
        print("%-*s" % (w, "metric") + "".join("%-30s" % n for n in names) + "generator knob")
        for r in rows:
            knob = KNOB_TEXT.get(ROW_KNOB.get(r, ""), "")
            print("%-*s" % (w, r) + "".join("%-30s" % columns[n].get(r, "-") for n in names) + knob)

    flight_key = "flight_dev" if "flight_dev" in infos_by else ("flight" if "flight" in infos_by else None)
    if flight_key:
        source_table(infos_by[flight_key])

    print("\n=== disagreements ===")
    for key, _path, label in DATASETS:
        if key not in infos_by:
            continue
        pm, pr = pred_map("preds_%s_%s.jsonl" % (MODEL, key)), pred_map("preds_%s_%s.jsonl" % (REF, key))
        if pm is None or pr is None:
            print("  %s: preds for %s and/or %s not found, skipped" % (label, MODEL, REF))
            continue
        disagreements(label, infos_by[key], pm, pr)


if __name__ == "__main__":
    main()
