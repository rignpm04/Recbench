# baselines.py -- rule and truth-discovery baselines for recbench v0
# Run from PyCharm after gen.py. Reads cases_*.jsonl, writes preds_<method>_<split>.jsonl.
# Python 3.9, stdlib only.

import math
import time
from typing import Any, Dict, List, Optional, Tuple

from recbench_common import FIELDS, cluster, load_jsonl, match, norm_value, write_jsonl

# ============================================================ settings
SPLITS = {"train": "cases_train.jsonl", "heldout": "cases_heldout.jsonl", "hard": "cases_hard.jsonl"}
METHODS = ["newest_observed", "newest_arrival", "source_priority", "majority_vote",
           "time_decayed_vote", "dawid_skene", "truthfinder"]
MAX_CASES_PER_SPLIT = None     # e.g. 200 for a quick run; None = all

SOURCE_PRIORITY = {"vet_pdf": 5, "email_forward": 4, "owner": 3, "extractor": 2, "note_text": 1}
SOURCE_WEIGHT = {"vet_pdf": 1.0, "email_forward": 0.9, "owner": 0.7, "extractor": 0.5, "note_text": 0.3}
DECAY_TAU_DAYS = 120.0
WINDOW_DAYS = 60              # items for Dawid-Skene / TruthFinder = (field, 60-day window before now)
DS_ITERS = 15
TF_ITERS = 20
TF_GAMMA = 0.3
TF_DAMPEN = 0.5


# ============================================================ shared
def nv(a: Dict[str, Any]) -> Any:
    return norm_value(a["field"], a["value"], a.get("unit"))


def assign_labels(field: str, A: List[Dict[str, Any]], chosen: Any, chosen_conf: float,
                  claim_conf: Optional[Dict[int, float]] = None) -> Dict[int, Dict[str, Any]]:
    """Turn a chosen current value into per-assertion labels + probabilities."""
    claim_conf = claim_conf or {}
    support_days = [a["observed_day"] for a in A if match(field, nv(a), chosen)]
    newest_support = max(support_days) if support_days else -10 ** 9
    out = {}
    for a in A:
        q = claim_conf.get(a["id"], chosen_conf)
        q = min(1.0, max(0.0, q))
        if match(field, nv(a), chosen):
            out[a["id"]] = {"label": "valid", "p_valid": q, "p_superseded": 0.0, "p_erroneous": 1.0 - q}
        elif a["observed_day"] < newest_support:
            out[a["id"]] = {"label": "superseded", "p_valid": 0.0, "p_superseded": q, "p_erroneous": 1.0 - q}
        else:
            out[a["id"]] = {"label": "erroneous", "p_valid": 0.0, "p_superseded": 1.0 - chosen_conf, "p_erroneous": chosen_conf}
    return out


def by_field(case: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    g: Dict[str, List[Dict[str, Any]]] = {}
    for a in case["assertions"]:
        g.setdefault(a["field"], []).append(a)
    return g


# ============================================================ rule methods
def m_newest_observed(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    a = max(A, key=lambda x: (x["observed_day"], x["arrived_day"], x["id"]))
    return nv(a), 1.0, {}


def m_newest_arrival(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    a = max(A, key=lambda x: (x["arrived_day"], x["observed_day"], x["id"]))
    return nv(a), 1.0, {}


def m_source_priority(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    a = max(A, key=lambda x: (SOURCE_PRIORITY.get(x["source"], 0), x["observed_day"], x["arrived_day"], x["id"]))
    return nv(a), 1.0, {}


def m_majority_vote(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    cl = cluster(field, [(a["id"], nv(a)) for a in A])
    newest = max(A, key=lambda x: (x["observed_day"], x["arrived_day"]))
    def key(c):
        return (len(c["members"]), 1 if newest["id"] in c["members"] else 0)
    best = max(cl, key=key)
    conf = len(best["members"]) / float(len(A))
    claim = {}
    for c in cl:
        for mid in c["members"]:
            claim[mid] = len(c["members"]) / float(len(A))
    return best["value"], conf, claim


def m_time_decayed_vote(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    w = {}
    for a in A:
        age = max(0, now - a["observed_day"])
        w[a["id"]] = math.exp(-age / DECAY_TAU_DAYS) * SOURCE_WEIGHT.get(a["source"], 0.5) * (a.get("extractor_conf") or 1.0)
    cl = cluster(field, [(a["id"], nv(a)) for a in A])
    total = sum(w.values()) or 1e-9
    best = max(cl, key=lambda c: sum(w[m] for m in c["members"]))
    conf = sum(w[m] for m in best["members"]) / total
    claim = {}
    for c in cl:
        share = sum(w[m] for m in c["members"]) / total
        for m in c["members"]:
            claim[m] = share
    return best["value"], conf, claim


# ============================================================ truth discovery
def windows(field: str, A: List[Dict[str, Any]], now: int) -> Dict[int, List[Dict[str, Any]]]:
    items: Dict[int, List[Dict[str, Any]]] = {}
    for a in A:
        k = (now - a["observed_day"]) // WINDOW_DAYS
        items.setdefault(k, []).append(a)
    return items


def m_dawid_skene(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    """One-coin Dawid-Skene: one accuracy per source, items = (field, time window), classes = value clusters."""
    items = windows(field, A, now)
    structured = {}   # k -> (clusters, claims: list of (source, cluster_idx, assertion_id))
    for k, group in items.items():
        cl = cluster(field, [(a["id"], nv(a)) for a in group])
        idx = {}
        for ci, c in enumerate(cl):
            for m in c["members"]:
                idx[m] = ci
        claims = [(a["source"], idx[a["id"]], a["id"]) for a in group]
        structured[k] = (cl, claims)
    sources = sorted(set(a["source"] for a in A))
    acc = {s: 0.8 for s in sources}
    post: Dict[int, List[float]] = {}
    for _ in range(DS_ITERS):
        # E-step
        for k, (cl, claims) in structured.items():
            K = len(cl)
            logp = [0.0] * K
            for s, ci, _id in claims:
                p = min(0.99, max(0.01, acc[s]))
                for j in range(K):
                    logp[j] += math.log(p) if j == ci else math.log((1 - p) / max(1, K - 1))
            mx = max(logp)
            ex = [math.exp(v - mx) for v in logp]
            z = sum(ex)
            post[k] = [e / z for e in ex]
        # M-step
        num = {s: 0.0 for s in sources}
        den = {s: 0.0 for s in sources}
        for k, (cl, claims) in structured.items():
            for s, ci, _id in claims:
                num[s] += post[k][ci]
                den[s] += 1.0
        for s in sources:
            acc[s] = (num[s] + 1.0) / (den[s] + 2.0)     # light smoothing
    latest = min(structured.keys())
    cl, claims = structured[latest]
    p = post[latest]
    best = max(range(len(cl)), key=lambda j: p[j])
    claim_conf = {}
    for k, (clk, claimsk) in structured.items():
        for s, ci, _id in claimsk:
            claim_conf[_id] = post[k][ci]
    return cl[best]["value"], p[best], claim_conf


def m_truthfinder(field: str, A: List[Dict[str, Any]], now: int) -> Tuple[Any, float, Dict[int, float]]:
    """TruthFinder (Yin, Han & Yu 2008), simplified: iterate source trust <-> claim confidence."""
    items = windows(field, A, now)
    structured = {}
    for k, group in items.items():
        cl = cluster(field, [(a["id"], nv(a)) for a in group])
        srcs_of = []
        for c in cl:
            srcs_of.append([a["source"] for a in group if a["id"] in c["members"]])
        structured[k] = (cl, srcs_of)
    sources = sorted(set(a["source"] for a in A))
    trust = {s: 0.9 for s in sources}
    conf: Dict[int, List[float]] = {}
    for _ in range(TF_ITERS):
        for k, (cl, srcs_of) in structured.items():
            sig = []
            for ci, c in enumerate(cl):
                s_sum = sum(-math.log(max(1e-6, 1.0 - min(0.999, trust[s]))) for s in srcs_of[ci])
                sig.append(s_sum)
            # implication: numerically close clusters support each other a little
            sig2 = []
            for ci, c in enumerate(cl):
                extra = 0.0
                if field in ("weight_kg", "birth_day", "rabies_day"):
                    for cj, d in enumerate(cl):
                        if cj != ci and match(field, c["value"], d["value"] * 1.0):
                            extra += 0.5 * sig[cj]
                sig2.append(sig[ci] + extra)
            conf[k] = [1.0 / (1.0 + math.exp(-TF_GAMMA * s)) for s in sig2]
        new_trust = {}
        for s in sources:
            vals = []
            for k, (cl, srcs_of) in structured.items():
                for ci, srcs in enumerate(srcs_of):
                    vals.extend([conf[k][ci]] * srcs.count(s))
            new_trust[s] = sum(vals) / len(vals) if vals else trust[s]
        for s in sources:
            trust[s] = TF_DAMPEN * trust[s] + (1 - TF_DAMPEN) * new_trust[s]
    latest = min(structured.keys())
    cl, srcs_of = structured[latest]
    cf = conf[latest]
    total = sum(cf) or 1e-9
    best = max(range(len(cl)), key=lambda j: cf[j])
    claim_conf = {}
    for k, (clk, srcs_ofk) in structured.items():
        tot = sum(conf[k]) or 1e-9
        for ci, c in enumerate(clk):
            for m in c["members"]:
                claim_conf[m] = conf[k][ci] / tot
    return cl[best]["value"], cf[best] / total, claim_conf


METHOD_FN = {
    "newest_observed": m_newest_observed, "newest_arrival": m_newest_arrival, "source_priority": m_source_priority,
    "majority_vote": m_majority_vote, "time_decayed_vote": m_time_decayed_vote,
    "dawid_skene": m_dawid_skene, "truthfinder": m_truthfinder,
}


# ============================================================ driver
def predict_case(method: str, case: Dict[str, Any]) -> Dict[str, Any]:
    fn = METHOD_FN[method]
    now = case["now_day"]
    groups = by_field(case)
    out = {"case_id": case["case_id"], "method": method, "queries": {}, "assertions": {}}
    for q in case["queries"]:
        field = q["field"]
        A = groups.get(field, [])
        if not A:
            continue
        chosen, conf, claim_conf = fn(field, A, now)
        out["queries"][field] = {"value": chosen, "confidence": round(float(conf), 4)}
        labels = assign_labels(field, A, chosen, float(conf), claim_conf)
        for aid, lab in labels.items():
            out["assertions"][str(aid)] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in lab.items()}
    return out


def main() -> None:
    t0 = time.time()
    for split, path in SPLITS.items():
        cases = load_jsonl(path)
        if MAX_CASES_PER_SPLIT:
            cases = cases[:MAX_CASES_PER_SPLIT]
        for method in METHODS:
            preds = [predict_case(method, c) for c in cases]
            write_jsonl("preds_%s_%s.jsonl" % (method, split), preds)
            print("%-8s %-18s %d cases" % (split, method, len(preds)))
    print("done in %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
