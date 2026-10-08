# feature_baseline.py -- supervised baseline on hand-made features (recbench v0.4)
# "Same synthetic data, no fancy architecture": gradient-boosted trees (default) or TabPFN on per-candidate and
# per-assertion feature rows. Trained on cases_train.jsonl; hyperparameters chosen on cases_val.jsonl under the same
# budget as the classical baselines (TUNE); evaluated on every other split, zero-shot on snapshot / real sets.
# Run from PyCharm after gen.py. Needs: pip install scikit-learn   (TabPFN optional: pip install tabpfn)
# Writes preds_feat_<MODEL>_<split>.jsonl; calibrate.py then temperature-scales them like every other method.
# Python 3.9.

import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np

from recbench_common import (FIELD_TYPE, LABELS, SOURCES, cluster, current_truth, ftype_of, is_numeric, load_jsonl,
                             match, norm_value, write_jsonl)

# ============================================================ settings
MODEL = "hgb"                  # "hgb" = sklearn HistGradientBoosting (no extra install) | "tabpfn" = TabPFN v2 (pip install tabpfn)
TRAIN_FILE = "cases_train.jsonl"
VAL_FILE = "cases_val.jsonl"
EVAL_FILES = {"train": "cases_train.jsonl", "val": "cases_val.jsonl", "heldout": "cases_heldout.jsonl",
              "hard": "cases_hard.jsonl", "snapshot": "cases_snapshot.jsonl",
              "stock": "cases_stock.jsonl", "flight": "cases_flight.jsonl", "book": "cases_book.jsonl",
              "book_subset": "cases_book_subset.jsonl"}       # missing files are skipped
MAX_TRAIN_CASES = None         # None = all 2000; TabPFN caps rows itself (see TABPFN_MAX_ROWS)
TABPFN_MAX_ROWS = 10000        # TabPFN v2 context limit; rows are subsampled to this
CALIBRATION = None             # None (v0.4 default: calibrate.py temperature-scales every method the same way) | "isotonic" | "sigmoid"
TUNE = True                    # hgb only: pick max_iter / learning_rate / max_leaf_nodes on the val split (BUDGET configs)
BUDGET = 40
SEED = 11
TUNED_FILE = "tuned_params.json"   # the chosen settings are recorded here next to the classical baselines' (tune.py)

DEFAULT_HGB = {"max_iter": 400, "learning_rate": 0.06, "max_leaf_nodes": 31, "l2_regularization": 0.5}
HGB_GRID = [{"max_iter": it, "learning_rate": lr, "max_leaf_nodes": nl, "l2_regularization": 0.5}
            for it in (200, 600) for lr in (0.03, 0.06, 0.12) for nl in (15, 31, 63)][:BUDGET]

SOURCE_PRIORITY = {"vet_pdf": 5, "email_forward": 4, "owner": 3, "extractor": 2, "note_text": 1}
FTYPES = sorted(set(FIELD_TYPE.values())) + ["num_rel", "num_abs", "cat"]   # one-hot slots incl. real-data types


# ============================================================ feature extraction
def nv(a: Dict[str, Any]) -> Any:
    return norm_value(a["field"], a["value"], a.get("unit"))


def rel_dist(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-6)


def field_rows(case: Dict[str, Any], field: str, A: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[List[float]], List[List[float]]]:
    """Returns (clusters, cluster_feature_rows, assertion_feature_rows) for one field of one case.
    Only observable keys are read: field, value, unit, observed_day, arrived_day, source, extractor_conf, text
    (presence), corrects, duplicate_of (presence). Never label / error_type / truth."""
    now = case["now_day"]
    vals = {a["id"]: nv(a) for a in A}
    cl = cluster(field, [(a["id"], vals[a["id"]]) for a in A])
    by_id = {a["id"]: a for a in A}
    cluster_of = {}
    for ci, c in enumerate(cl):
        for m in c["members"]:
            cluster_of[m] = ci
    n = len(A)
    newest_obs = max(A, key=lambda x: (x["observed_day"], x["arrived_day"]))
    newest_arr = max(A, key=lambda x: (x["arrived_day"], x["observed_day"]))
    numeric = is_numeric(field)
    all_vals = [vals[a["id"]] for a in A] if numeric else []
    median = float(np.median(all_vals)) if numeric else 0.0
    vet = [a for a in A if a["source"] == "vet_pdf"]
    newest_vet_val = vals[max(vet, key=lambda x: x["observed_day"])["id"]] if (vet and numeric) else None
    corrected_ids = set(a["corrects"] for a in A if a.get("corrects") is not None)
    n_sources = float(len(set(a["source"] for a in A)))
    ftype_vec = [1.0 if ftype_of(field) == t else 0.0 for t in FTYPES]

    # per-cluster rows
    crow = []
    cl_newest_obs = [max(by_id[m]["observed_day"] for m in c["members"]) for c in cl]
    rank_obs = {ci: r for r, ci in enumerate(sorted(range(len(cl)), key=lambda i: -cl_newest_obs[i]))}
    for ci, c in enumerate(cl):
        mem = [by_id[m] for m in c["members"]]
        size = len(mem)
        srcs = [m["source"] for m in mem]
        obs_days = [m["observed_day"] for m in mem]
        arr_days = [m["arrived_day"] for m in mem]
        newest_c = max(obs_days)
        later_disagree = sum(1 for a in A if a["observed_day"] > newest_c and cluster_of[a["id"]] != ci)
        later_agree = 0
        confs = [m.get("extractor_conf") for m in mem if m.get("extractor_conf") is not None]
        row = ftype_vec + [
            float(n), float(size), size / n, float(len(cl)),
            float(now - newest_c), float(now - min(obs_days)), float(now - max(arr_days)),
            1.0 if newest_obs["id"] in c["members"] else 0.0,
            1.0 if newest_arr["id"] in c["members"] else 0.0,
            float(rank_obs[ci]),
            float(max(SOURCE_PRIORITY.get(s, 0) for s in srcs)),
            float(len(set(srcs))),
        ] + [float(srcs.count(s)) for s in SOURCES] + [
            float(np.mean(confs)) if confs else 1.0,
            float(later_disagree), float(later_agree),
            1.0 if any(m.get("corrects") is not None for m in mem) else 0.0,
            1.0 if any(m["id"] in corrected_ids for m in mem) else 0.0,
            1.0 if any(m.get("duplicate_of") is not None for m in mem) else 0.0,
            1.0 if any(m["source"] == "note_text" for m in mem) else 0.0,
            rel_dist(c["value"], median) if numeric else 0.0,
            rel_dist(c["value"], newest_vet_val) if (numeric and newest_vet_val is not None) else -1.0,
            float(max(obs_days) - min(obs_days)),
            # v0.4 additions
            float(np.mean([m["arrived_day"] - m["observed_day"] for m in mem])),
            float(sum(1 for m in mem if m.get("text")) / size),
            float(sum(1 for m in mem if m.get("unit") == "lb") / size),
            len(set(srcs)) / n_sources,
            n_sources,
        ]
        crow.append(row)

    # per-assertion rows
    arow = []
    for a in A:
        ci = cluster_of[a["id"]]
        c = cl[ci]
        size = len(c["members"])
        later = [b for b in A if b["observed_day"] > a["observed_day"]]
        later_dis = sum(1 for b in later if cluster_of[b["id"]] != ci)
        later_agr = sum(1 for b in later if cluster_of[b["id"]] == ci)
        earlier_agr = sum(1 for b in A if b["observed_day"] < a["observed_day"] and cluster_of[b["id"]] == ci)
        srcs_agree = len(set(by_id[m]["source"] for m in c["members"]))
        row = ftype_vec + [1.0 if a["source"] == s else 0.0 for s in SOURCES] + [
            float(a.get("extractor_conf") if a.get("extractor_conf") is not None else 1.0),
            float(now - a["observed_day"]), float(a["arrived_day"] - a["observed_day"]),
            float(n), float(size), size / n, float(len(cl)),
            1.0 if a["id"] == newest_obs["id"] else 0.0,
            1.0 if a["id"] == newest_arr["id"] else 0.0,
            float(later_dis), float(later_agr), float(earlier_agr), float(srcs_agree),
            1.0 if a.get("corrects") is not None else 0.0,
            1.0 if a["id"] in corrected_ids else 0.0,
            1.0 if a.get("duplicate_of") is not None else 0.0,
            float(rank_obs[ci]),
            rel_dist(vals[a["id"]], median) if numeric else 0.0,
            rel_dist(vals[a["id"]], newest_vet_val) if (numeric and newest_vet_val is not None) else -1.0,
            # v0.4 additions
            1.0 if a.get("text") else 0.0,
            1.0 if a.get("unit") == "lb" else 0.0,
            srcs_agree / n_sources,
            n_sources,
        ]
        arow.append(row)
    return cl, crow, arow


def build_dataset(cases: List[Dict[str, Any]], with_labels: bool = True):
    """Flatten cases into (X_cluster, y_cluster, X_assertion, y_assertion, index) arrays."""
    Xc, yc, Xa, ya, index = [], [], [], [], []
    for case in cases:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for a in case["assertions"]:
            groups.setdefault(a["field"], []).append(a)
        for q in case["queries"]:
            field = q["field"]
            A = groups.get(field)
            if not A:
                continue
            cl, crow, arow = field_rows(case, field, A)
            truth = current_truth(case, field) if with_labels else None
            c_start = len(Xc)
            for ci, c in enumerate(cl):
                Xc.append(crow[ci])
                yc.append(1 if (with_labels and match(field, c["value"], truth)) else 0)
            a_start = len(Xa)
            for a, row in zip(A, arow):
                Xa.append(row)
                ya.append(LABELS.index(a["label"]) if with_labels else 0)
            index.append({"case_id": case["case_id"], "field": field, "clusters": cl,
                          "c_slice": (c_start, c_start + len(cl)), "a_slice": (a_start, a_start + len(A)),
                          "a_ids": [a["id"] for a in A], "decidable": q["decidable"]})
    return np.array(Xc, dtype=float), np.array(yc), np.array(Xa, dtype=float), np.array(ya), index


# ============================================================ models
def make_model(params: Dict[str, Any]):
    if MODEL == "tabpfn":
        from tabpfn import TabPFNClassifier   # pip install tabpfn ; downloads weights on first use
        return TabPFNClassifier(random_state=SEED)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=SEED, **params)


def fit(X: np.ndarray, y: np.ndarray, params: Dict[str, Any], calibration=None):
    if MODEL == "tabpfn" and len(X) > TABPFN_MAX_ROWS:
        rng = np.random.RandomState(SEED)
        idx = rng.choice(len(X), TABPFN_MAX_ROWS, replace=False)
        X, y = X[idx], y[idx]
    base = make_model(params)
    if calibration:
        from sklearn.calibration import CalibratedClassifierCV
        clf = CalibratedClassifierCV(base, method=calibration, cv=3)
    else:
        clf = base
    clf.fit(X, y)
    return clf


def auc_binary(p: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


# ============================================================ selection on val
def query_accuracy(clf_c, Xc: np.ndarray, index: List[Dict[str, Any]], cases_by_id: Dict[str, Dict[str, Any]]) -> float:
    pc = clf_c.predict_proba(Xc)[:, 1]
    ok, n = 0, 0
    for item in index:
        if not item["decidable"]:
            continue
        s0, s1 = item["c_slice"]
        best = int(np.argmax(pc[s0:s1]))
        truth = current_truth(cases_by_id[item["case_id"]], item["field"])
        ok += 1 if match(item["field"], item["clusters"][best]["value"], truth) else 0
        n += 1
    return ok / n if n else 0.0


def macro_f1(clf_a, Xa: np.ndarray, ya: np.ndarray) -> float:
    pred = clf_a.predict(Xa)
    f1s = []
    for k in range(3):
        tp = int(((pred == k) & (ya == k)).sum())
        fp = int(((pred == k) & (ya != k)).sum())
        fn = int(((pred != k) & (ya == k)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def select(Xc, yc, Xa, ya, val_cases) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Equal-budget grid on val: cluster task by query accuracy, assertion task by macro-F1."""
    Xc_v, yc_v, Xa_v, ya_v, idx_v = build_dataset(val_cases)
    by_id = {c["case_id"]: c for c in val_cases}
    best_c, best_a = None, None
    t0 = time.time()
    for i, params in enumerate(HGB_GRID):
        qa = query_accuracy(fit(Xc, yc, params), Xc_v, idx_v, by_id)
        f1 = macro_f1(fit(Xa, ya, params), Xa_v, ya_v)
        if best_c is None or qa > best_c[0]:
            best_c = (qa, params)
        if best_a is None or f1 > best_a[0]:
            best_a = (f1, params)
        print("  config %2d/%d %s -> val q_acc %.3f, macro-F1 %.3f (%.0fs)" % (i + 1, len(HGB_GRID), params, qa, f1, time.time() - t0))
    print("chosen: cluster task %s (val q_acc %.3f) | assertion task %s (val macro-F1 %.3f)"
          % (best_c[1], best_c[0], best_a[1], best_a[0]))
    rec = {}
    if os.path.exists(TUNED_FILE):
        with open(TUNED_FILE, "r", encoding="utf-8") as f:
            rec = json.load(f)
    rec["feat_" + MODEL] = {"cluster_params": best_c[1], "assertion_params": best_a[1], "val_q_acc": round(best_c[0], 4),
                           "val_macro_f1": round(best_a[0], 4), "n_configs": len(HGB_GRID)}
    with open(TUNED_FILE, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=1)
    return best_c[1], best_a[1]


# ============================================================ driver
def predict_split(clf_c, clf_a, cases: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    Xc, yc, Xa, ya, index = build_dataset(cases, with_labels=True)
    pc = clf_c.predict_proba(Xc)[:, 1]
    pa = clf_a.predict_proba(Xa)
    label_order = list(clf_a.classes_)
    out: Dict[str, Dict[str, Any]] = {}
    for item in index:
        rec = out.setdefault(item["case_id"], {"case_id": item["case_id"], "method": "feat_" + MODEL,
                                               "queries": {}, "assertions": {}})
        s0, s1 = item["c_slice"]
        probs = pc[s0:s1]
        best = int(np.argmax(probs))
        total = float(probs.sum())
        conf = float(probs[best] / total) if total > 1.0 else float(probs[best])
        rec["queries"][item["field"]] = {"value": item["clusters"][best]["value"], "confidence": round(conf, 4)}
        a0, a1 = item["a_slice"]
        for aid, pr in zip(item["a_ids"], pa[a0:a1]):
            p = {LABELS[k]: 0.0 for k in range(3)}
            for j, k in enumerate(label_order):
                p[LABELS[int(k)]] = float(pr[j])
            lab = max(p, key=lambda l: p[l])
            rec["assertions"][str(aid)] = {"label": lab, "p_valid": round(p["valid"], 4),
                                           "p_superseded": round(p["superseded"], 4),
                                           "p_erroneous": round(p["erroneous"], 4)}
    # metrics for this split
    print("  %-11s cluster-task AUC %s | assertion-task macro-AUC(ovr) %s" % (
        split, "%.3f" % auc_binary(pc, yc) if len(set(yc.tolist())) > 1 else "  -  ", ovr_auc(pa, ya, label_order)))
    return [out[c["case_id"]] for c in cases if c["case_id"] in out]


def ovr_auc(pa: np.ndarray, ya: np.ndarray, label_order) -> str:
    vals = []
    for j, k in enumerate(label_order):
        yk = (ya == int(k)).astype(int)
        vals.append(auc_binary(pa[:, j], yk))
    return "%.3f (valid %.3f / superseded %.3f / erroneous %.3f)" % (
        float(np.nanmean(vals)), vals[label_order.index(0)] if 0 in label_order else float("nan"),
        vals[label_order.index(1)] if 1 in label_order else float("nan"),
        vals[label_order.index(2)] if 2 in label_order else float("nan"))


def main() -> None:
    t0 = time.time()
    train_cases = load_jsonl(TRAIN_FILE)
    if MAX_TRAIN_CASES:
        train_cases = train_cases[:MAX_TRAIN_CASES]
    Xc, yc, Xa, ya, _ = build_dataset(train_cases)
    print("train rows: cluster-task %d (pos %.1f%%) | assertion-task %d (valid/superseded/erroneous = %s) | %d features" % (
        len(Xc), 100 * yc.mean(), len(Xa), np.bincount(ya, minlength=3).tolist(), Xa.shape[1]))
    params_c, params_a = DEFAULT_HGB, DEFAULT_HGB
    if TUNE and MODEL == "hgb" and os.path.exists(VAL_FILE):
        print("tuning on %s (%d configs)..." % (VAL_FILE, len(HGB_GRID)))
        params_c, params_a = select(Xc, yc, Xa, ya, load_jsonl(VAL_FILE))
    clf_c = fit(Xc, yc, params_c, CALIBRATION)
    clf_a = fit(Xa, ya, params_a, CALIBRATION)
    print("fit done in %.0fs (model=%s, calibration=%s)" % (time.time() - t0, MODEL, CALIBRATION))
    for split, path in EVAL_FILES.items():
        if not os.path.exists(path):
            continue
        cases = load_jsonl(path)
        preds = predict_split(clf_c, clf_a, cases, split)
        write_jsonl("preds_feat_%s_%s.jsonl" % (MODEL, split), preds)
    print("done in %.0fs. train-split numbers are in-sample; judge on heldout / hard / snapshot. Now run calibrate.py, then score.py"
          % (time.time() - t0))


if __name__ == "__main__":
    main()
