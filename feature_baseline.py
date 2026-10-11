# feature_baseline.py -- supervised baseline on hand-made features (recbench v0.5)
# "Same synthetic data, no fancy architecture": gradient-boosted trees (default) or TabPFN on per-candidate and
# per-assertion feature rows. Hyperparameters chosen on cases_val.jsonl under the same budget as the classical
# baselines (TUNE); evaluated on every other split, zero-shot on snapshot / real sets.
# Run from PyCharm after gen.py. Needs: pip install scikit-learn   (TabPFN optional: pip install tabpfn)
# Writes preds_<METHOD_NAME>_<split>.jsonl; calibrate.py then temperature-scales them like every other method.
# Python 3.9.
#
# v0.5:
#   - EQUAL TRAINING DATA. TRAIN_SOURCE = "stream" trains on exactly the cases the reconciler trains on (its seeds:
#     SEED * 10**6 + epoch * 10**4 + step, 32 cases per step, 8 epochs x 625 steps = 160,000 cases). v0.4 trained this
#     model on the 2,000 cases of cases_train.jsonl while the reconciler saw 160,000. "file" = the v0.4 setting.
#     Features are built in chunks (float32) so memory stays ~3 GB. The grid is evaluated on the first N_TUNE_CASES
#     cases of the stream (time), the chosen settings are then fitted on all of it.
#   - SEED: the data-stream seed, as in train_reconciler.py. 1 = the reported model ("feat_hgb"); 2, 3 = replicates
#     ("feat_hgb_seed2", ...), which reuse the hyperparameters chosen for seed 1.
#   - The stated value of a chosen cluster is cluster_answer() (mean of its most recent readings), and the training
#     label of a cluster is whether that value matches the truth (v0.4: the mean of all its readings).
#   - Confidence = the chosen cluster's own probability of being correct (v0.4 divided it by the sum over the
#     field's clusters when that sum exceeded 1, which splits it whenever two clusters are both correct).
#   - A record is written for every case, including one with no assertions (generator v3's feeds can leave one).
#   - Seeds 2 and 3 stop if seed 1's settings are not in tuned_params.json (they never tune).

import json
import os
import pickle
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np

import gen
from recbench_common import (FIELD_TYPE, LABELS, SOURCES, cluster, cluster_answer, current_truth, ftype_of, is_numeric,
                             load_jsonl, match, norm_value, write_jsonl)

# ============================================================ settings
MODEL = "hgb"                  # "hgb" = sklearn HistGradientBoosting (no extra install) | "tabpfn" = TabPFN v2 (pip install tabpfn)
TRAIN_SOURCE = "stream"        # "stream" = the reconciler's own 160,000 training cases (v0.5) | "file" = cases_train.jsonl (v0.4)
SEED = 1                       # data-stream seed (as in train_reconciler.py): 1 = reported model, 2 / 3 = replicates
STREAM_EPOCHS = 8              # must equal train_reconciler.py EPOCHS
STREAM_STEPS = 625             # must equal train_reconciler.py N_TRAIN_CASES // BATCH_CASES
STREAM_BATCH = 32              # must equal train_reconciler.py BATCH_CASES
N_TUNE_CASES = 10000           # the grid is evaluated on the first N_TUNE_CASES stream cases; the chosen settings use all
CHUNK_CASES = 4000             # cases per feature-building chunk
TRAIN_FILE = "cases_train.jsonl"
VAL_FILE = "cases_val.jsonl"
EVAL_FILES = {"train": "cases_train.jsonl", "val": "cases_val.jsonl", "heldout": "cases_heldout.jsonl",
              "hard": "cases_hard.jsonl", "snapshot": "cases_snapshot.jsonl",
              "test_hard": "cases_test_hard.jsonl", "test_heldout": "cases_test_heldout.jsonl",
              "test_snapshot": "cases_test_snapshot.jsonl",
              "stock": "cases_stock.jsonl", "stock_nogold": "cases_stock_nogold.jsonl",
              "flight_dev": "cases_flight_dev.jsonl", "flight_test": "cases_flight_test.jsonl",
              "flight_dev_nogold": "cases_flight_dev_nogold.jsonl", "flight_test_nogold": "cases_flight_test_nogold.jsonl",
              "book": "cases_book.jsonl", "book_subset": "cases_book_subset.jsonl",
              "parliament_dev": "cases_parliament_dev.jsonl", "parliament_dev_nobot": "cases_parliament_dev_nobot.jsonl",
              "parliament_test": "cases_parliament_test.jsonl",
              "parliament_test_nobot": "cases_parliament_test_nobot.jsonl"}       # missing files are skipped
MAX_TRAIN_CASES = None         # TRAIN_SOURCE = "file" only: None = all 2000
TABPFN_MAX_ROWS = 10000        # TabPFN v2 context limit; rows are subsampled to this
CALIBRATION = None             # None (v0.4 default: calibrate.py temperature-scales every method the same way) | "isotonic" | "sigmoid"
TUNE = True                    # hgb only: pick max_iter / learning_rate / max_leaf_nodes on the val split (BUDGET configs)
BUDGET = 40
HGB_RANDOM_STATE = 10 + SEED   # 11 for seed 1, as in v0.4
METHOD_NAME = ("feat_" + MODEL) if SEED == 1 else ("feat_%s_seed%d" % (MODEL, SEED))
TUNED_FILE = "tuned_params.json"   # the chosen settings are recorded here next to the classical baselines' (tune.py)
EVAL_ONLY = False              # True: load MODEL_FILE (written by the training run) and only predict (e.g. on the test splits)
MODEL_FILE = METHOD_NAME + ".pkl"  # the two fitted models (gitignored; same seed and data reproduce them exactly)

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
    """Returns (clusters, cluster_feature_rows, assertion_feature_rows) for one field of one case. Each cluster dict
    also carries "answer" = cluster_answer(...), the value stated if the cluster is picked (v0.5).
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
        c["answer"] = cluster_answer(field, [by_id[m] for m in c["members"]])
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


def build_dataset(cases: List[Dict[str, Any]], with_labels: bool = True, keep_index: bool = True):
    """Flatten cases into (X_cluster, y_cluster, X_assertion, y_assertion, index) arrays (float32 features)."""
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
                yc.append(1 if (with_labels and truth is not None and match(field, c["answer"], truth)) else 0)
            a_start = len(Xa)
            for a, row in zip(A, arow):
                Xa.append(row)
                ya.append(LABELS.index(a["label"]) if with_labels else 0)
            if keep_index:
                index.append({"case_id": case["case_id"], "field": field, "clusters": cl,
                              "c_slice": (c_start, c_start + len(cl)), "a_slice": (a_start, a_start + len(A)),
                              "a_ids": [a["id"] for a in A], "decidable": q["decidable"]})
    return (np.array(Xc, dtype=np.float32).reshape(len(Xc), -1), np.array(yc, dtype=np.int64),
            np.array(Xa, dtype=np.float32).reshape(len(Xa), -1), np.array(ya, dtype=np.int64), index)


def stream_cases(seed: int, n_cases: int):
    """Yield the reconciler's training stream (train_reconciler.gen_cases with its seeds), in order."""
    done = 0
    for epoch in range(STREAM_EPOCHS):
        for s in range(STREAM_STEPS):
            rng = random.Random(seed * 10 ** 6 + epoch * 10 ** 4 + s)
            for i in range(STREAM_BATCH):
                if done >= n_cases:
                    return
                yield gen.gen_case(rng, "train", i)
                done += 1


def build_from_stream(seed: int, n_cases: int, mark: int):
    """Feature arrays for the first n_cases of the stream, built in chunks (memory). Also returns the number of
    cluster / assertion rows contributed by the first `mark` cases (the tuning subset)."""
    parts = {"Xc": [], "yc": [], "Xa": [], "ya": []}
    chunk: List[Dict[str, Any]] = []
    t0 = time.time()
    n = 0
    rows_at_mark = (0, 0)
    def flush():
        Xc, yc, Xa, ya, _ = build_dataset(chunk, keep_index=False)
        parts["Xc"].append(Xc); parts["yc"].append(yc); parts["Xa"].append(Xa); parts["ya"].append(ya)
        chunk.clear()
    for case in stream_cases(seed, n_cases):
        chunk.append(case)
        n += 1
        if len(chunk) >= CHUNK_CASES or n == mark:
            flush()
            if n == mark:
                rows_at_mark = (sum(len(x) for x in parts["Xc"]), sum(len(x) for x in parts["Xa"]))
            if n % CHUNK_CASES == 0 or n == mark:
                print("  stream: %d / %d cases featurized (%.0fs)" % (n, n_cases, time.time() - t0))
    if chunk:
        flush()
    if n < mark:
        rows_at_mark = (sum(len(x) for x in parts["Xc"]), sum(len(x) for x in parts["Xa"]))
    return (np.concatenate(parts["Xc"]), np.concatenate(parts["yc"]), np.concatenate(parts["Xa"]),
            np.concatenate(parts["ya"]), rows_at_mark)


# ============================================================ models
def make_model(params: Dict[str, Any]):
    if MODEL == "tabpfn":
        from tabpfn import TabPFNClassifier   # pip install tabpfn ; downloads weights on first use
        return TabPFNClassifier(random_state=HGB_RANDOM_STATE)
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(random_state=HGB_RANDOM_STATE, **params)


def fit(X: np.ndarray, y: np.ndarray, params: Dict[str, Any], calibration=None):
    if MODEL == "tabpfn" and len(X) > TABPFN_MAX_ROWS:
        rng = np.random.RandomState(HGB_RANDOM_STATE)
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
        ok += 1 if (truth is not None and match(item["field"], item["clusters"][best]["answer"], truth)) else 0
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
                           "val_macro_f1": round(best_a[0], 4), "n_configs": len(HGB_GRID),
                           "train_source": TRAIN_SOURCE, "tuned_on_cases": N_TUNE_CASES if TRAIN_SOURCE == "stream" else None}
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
        rec = out.setdefault(item["case_id"], {"case_id": item["case_id"], "method": METHOD_NAME,
                                               "queries": {}, "assertions": {}})
        s0, s1 = item["c_slice"]
        probs = pc[s0:s1]
        best = int(np.argmax(probs))
        conf = float(probs[best])                 # P(this cluster's stated value is correct)
        rec["queries"][item["field"]] = {"value": item["clusters"][best]["answer"], "confidence": round(conf, 4)}
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
    return [out.get(c["case_id"]) or {"case_id": c["case_id"], "method": METHOD_NAME, "queries": {}, "assertions": {}}
            for c in cases]


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
    if EVAL_ONLY:
        with open(MODEL_FILE, "rb") as f:
            clf_c, clf_a = pickle.load(f)
        print("%s | loaded %s" % (METHOD_NAME, MODEL_FILE))
        predict_all(clf_c, clf_a, t0)
        return
    n_stream = STREAM_EPOCHS * STREAM_STEPS * STREAM_BATCH
    print("%s | training data: %s | seed %d" % (METHOD_NAME, ("the reconciler's stream, %d cases" % n_stream)
                                                  if TRAIN_SOURCE == "stream" else TRAIN_FILE, SEED))
    rows_at_mark = (0, 0)
    if TRAIN_SOURCE == "stream":
        Xc, yc, Xa, ya, rows_at_mark = build_from_stream(SEED, n_stream, N_TUNE_CASES)
    else:
        train_cases = load_jsonl(TRAIN_FILE)
        if MAX_TRAIN_CASES:
            train_cases = train_cases[:MAX_TRAIN_CASES]
        Xc, yc, Xa, ya, _ = build_dataset(train_cases, keep_index=False)
    print("train rows: cluster-task %d (pos %.1f%%) | assertion-task %d (valid/superseded/erroneous = %s) | %d / %d features" % (
        len(Xc), 100 * yc.mean(), len(Xa), np.bincount(ya, minlength=3).tolist(), Xc.shape[1], Xa.shape[1]))
    params_c, params_a = DEFAULT_HGB, DEFAULT_HGB
    tuned = {}
    if os.path.exists(TUNED_FILE):
        with open(TUNED_FILE, "r", encoding="utf-8") as f:
            tuned = json.load(f).get("feat_" + MODEL, {})
    if SEED != 1:
        if TUNE and MODEL == "hgb" and not tuned:
            raise SystemExit("seed %d reuses the settings chosen for seed 1, but %s has no %r entry: run SEED = 1 first"
                             % (SEED, TUNED_FILE, "feat_" + MODEL))
        if tuned:
            params_c, params_a = tuned["cluster_params"], tuned["assertion_params"]
            print("replicate seed %d: using the settings chosen for seed 1 (%s)" % (SEED, TUNED_FILE))
    elif TUNE and MODEL == "hgb" and os.path.exists(VAL_FILE):
        if TRAIN_SOURCE == "stream" and len(Xc) > 0:
            # the grid runs on the first N_TUNE_CASES stream cases (the stream's first cases are in the arrays first)
            n_c, n_a = rows_at_mark
            print("tuning on %s (%d configs), grid fitted on the first %d stream cases..." % (VAL_FILE, len(HGB_GRID), N_TUNE_CASES))
            params_c, params_a = select(Xc[:n_c], yc[:n_c], Xa[:n_a], ya[:n_a], load_jsonl(VAL_FILE))
        else:
            print("tuning on %s (%d configs)..." % (VAL_FILE, len(HGB_GRID)))
            params_c, params_a = select(Xc, yc, Xa, ya, load_jsonl(VAL_FILE))
    clf_c = fit(Xc, yc, params_c, CALIBRATION)
    clf_a = fit(Xa, ya, params_a, CALIBRATION)
    print("fit done in %.0fs (model=%s, calibration=%s; iterations used: cluster %s, assertion %s)" % (
        time.time() - t0, MODEL, CALIBRATION, getattr(clf_c, "n_iter_", "?"), getattr(clf_a, "n_iter_", "?")))
    with open(MODEL_FILE, "wb") as f:
        pickle.dump((clf_c, clf_a), f)
    predict_all(clf_c, clf_a, t0)


def predict_all(clf_c, clf_a, t0: float) -> None:
    for split, path in EVAL_FILES.items():
        if not os.path.exists(path):
            continue
        cases = load_jsonl(path)
        preds = predict_split(clf_c, clf_a, cases, split)
        write_jsonl("preds_%s_%s.jsonl" % (METHOD_NAME, split), preds)
    print("done in %.0fs. %s Now run calibrate.py, then score.py" % (
        time.time() - t0, "(The train split is not in the training stream.)" if TRAIN_SOURCE == "stream"
        else "Train-split numbers are in-sample."))



if __name__ == "__main__":
    main()
