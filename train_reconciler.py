# train_reconciler.py -- the PFN-style reconciler (recbench v0.5)
#
# A small transformer that reads a record as a SET of assertions (value, unit, two dates, source, confidence) and
# outputs (a) valid / superseded / erroneous probabilities for every assertion and (b) for every queried field, a
# probability over the candidate values present in the record and (v0.5) "none of these".
# Trained ONLY on freshly generated cases from gen.py (the prior); never sees a real label. Writes preds_<METHOD_NAME>_<split>.jsonl
# in the standard format, so score.py compares it against every baseline. Also scores the real sets (stock / flight /
# book) zero-shot if present.
#
# Needs: torch, numpy (python 3.10+ venv). From PyCharm's Terminal with the venv active: python train_reconciler.py.
# Prints metrics every epoch. Checkpoint: reconciler.pt (set EVAL_ONLY = True to just predict with it).
#
# v0.5 changes (pre-registered before any v0.5 run):
#   - "None of these" and several right answers. v0.4 put a softmax over the candidates, trained it toward the FIRST
#     matching candidate only, and skipped every query where no candidate is right (the truth changed and nobody
#     reported it) -- so it had to answer those confidently: 132 of its 151 wrong answers at >= 0.9 confidence on
#     heldout. The softmax now has one more option, a learned "none of these", and is trained to maximise the total
#     probability of the right answers: every candidate whose stated value matches the truth counts (so two right
#     candidates share the credit), and "none" is the target when no candidate is right. The stated value is the most
#     probable candidate; its confidence is the probability of all candidates whose value matches the stated value
#     (the scorer's test), so two right candidates do not split it. Chosen on val over one sigmoid per candidate
#     (both trained at full length on the same 160,000 cases; see PREDICTIONS.md, v0.5 honesty note).
#   - The stated value of a chosen cluster is recbench_common.cluster_answer() (mean of its most recent readings).
#   - q_acc in the printouts counts a query right when the stated value matches the truth (as score.py does). v0.4
#     counted it right only when the model picked the first matching cluster, which made the printed heldout q_acc
#     0.921 where score.py gives 92.3%.
#   - PERMUTE = "within" (v0.4 control: entry labels and candidate targets shuffled inside each case) or "cross"
#     (every entry label drawn independently from the train label marginal, every candidate target from the train
#     share of correct candidates; "none of these" is the target when no candidate drew a 1). Training on shuffled targets shows the scores come from learning; it cannot detect
#     an input leak. LEAK_TEST does that.
#   - LEAK_TEST = True: strips every key the encoder may not read from the case files (labels, error types, the truth,
#     query answers / types / decidable flags, knobs, snapshot parameters, undecidable ids; case ids replaced by
#     opaque ones), re-predicts, and checks that every answer and entry probability (unrounded) is identical. Both
#     predictions are made on the CPU (a copy of the model), so device arithmetic cannot cause a false alarm.
#   - SEED names the run: seed 1 = "reconciler" (the reported model); seeds 2, 3 = "reconciler_seed2" / "_seed3"
#     (replicates: different initialisation AND a different stream of training cases).
#   - Prediction reads up to EVAL_MAX_ENTRIES entries per record (training keeps MAX_ENTRIES); a longer record is
#     reported, never silently cut.
#
# Leak guards: the encoder only ever sees a scrubbed copy of each assertion holding the OBSERVABLE_KEYS below (never
# label / error_type, never the case's truth); the targets are read separately. Heldout is only monitored per epoch,
# never used for selection: the reported model is always the last epoch with the fixed settings below.

import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gen
from recbench_common import (FIELDS, FIELD_TYPE, LABELS, SOURCES, cluster, cluster_answer, current_truth, ftype_of,
                             is_numeric, load_jsonl, match, norm_value, write_jsonl)
from score import auc as rank_auc, ece as ece_score, prf

# ============================================================ settings
DEVICE = "auto"                 # "auto" | "mps" | "cuda" | "cpu"
N_TRAIN_CASES = 20000           # fresh cases generated per epoch (the prior is infinite)
EPOCHS = 8
BATCH_CASES = 32
LR = 3e-4
WEIGHT_DECAY = 0.01
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
FF = 256
DROPOUT = 0.1
MAX_ENTRIES = 640               # longest record used in TRAINING (longer generated records are cut there)
EVAL_MAX_ENTRIES = 2048         # longest record read at PREDICTION time (stock cases have ~560 entries)
SEED = 1                        # 1 = the reported model; 2, 3 = replicates (other init + other training cases)
VAL_CASES = 300                 # heldout cases used for the per-epoch metrics (monitoring only)
EVAL_ONLY = False               # True: load CHECKPOINT and only write predictions
PERMUTE = None                  # None | "within" | "cross"  (control runs; see the header)
LEAK_TEST = False               # True: after training / loading, run the scrub test (see the header)
EVAL_SPLITS = {"train": ("cases_train.jsonl", 500), "val": ("cases_val.jsonl", None),
               "heldout": ("cases_heldout.jsonl", None), "hard": ("cases_hard.jsonl", None),
               "snapshot": ("cases_snapshot.jsonl", None),
               "test_hard": ("cases_test_hard.jsonl", None), "test_heldout": ("cases_test_heldout.jsonl", None),
               "test_snapshot": ("cases_test_snapshot.jsonl", None)}       # missing files are skipped
REAL_SPLITS = ["stock", "stock_nogold", "flight_dev", "flight_test", "flight_dev_nogold", "flight_test_nogold",
               "book", "book_subset"]   # scored zero-shot when cases_<name>.jsonl exists

CHECKPOINT = "reconciler.pt" if SEED == 1 else "reconciler_seed%d.pt" % SEED
METHOD_NAME = "reconciler" if SEED == 1 else "reconciler_seed%d" % SEED
if PERMUTE:                     # the control runs are short: they only have to show chance-level numbers
    assert PERMUTE in ("within", "cross"), PERMUTE
    CHECKPOINT = "reconciler_permuted.pt" if PERMUTE == "within" else "reconciler_permuted_cross.pt"
    METHOD_NAME = "reconciler_permuted" if PERMUTE == "within" else "reconciler_permuted_cross"
    EPOCHS, N_TRAIN_CASES = 2, 5000

# the only assertion keys the encoder may read (label, error_type and the case truth are never on this list)
OBSERVABLE_KEYS = ("id", "field", "value", "unit", "observed_day", "arrived_day", "source", "extractor_conf",
                   "text", "corrects", "duplicate_of")

FTYPES = ["immutable_cat", "immutable_num", "drift_num", "regime_cat", "event_day", "num_rel", "num_abs", "cat"]
N_LOCAL_FIELDS = 32
N_LOCAL_SOURCES = 96
N_LOCAL_CLUSTERS = 64
N_NUM_FEATS = 16

LABEL_MARGINAL: List[float] = []   # filled for PERMUTE == "cross" from generated train cases (see main)
POS_RATE: List[float] = []         # PERMUTE == "cross": share of candidates that are correct in generated train cases


# ============================================================ encoding
def local_slots(rng: random.Random, n_slots: int, n_items: int) -> List[int]:
    """Random local ids; if there are more items than slots, ids repeat (v0.4 did this for clusters only)."""
    slots = rng.sample(range(n_slots), min(n_items, n_slots))
    return [slots[i % len(slots)] for i in range(n_items)]


def encode_case(case: Dict[str, Any], rng: random.Random, with_labels: bool = True,
                permute: Optional[str] = None, max_entries: int = MAX_ENTRIES) -> Optional[Dict[str, Any]]:
    """Turn one case into index/feature arrays plus the query structure.
    The encoder works on a scrubbed copy of the assertions (OBSERVABLE_KEYS only); labels and the true current
    value are read from the original case only when with_labels is set, and only into the target arrays."""
    raw = case["assertions"][:max_entries]
    if not raw:
        return None
    A = [{k: a.get(k) for k in OBSERVABLE_KEYS} for a in raw]
    gold = {a["id"]: a.get("label") for a in raw} if with_labels else {}
    if permute == "within" and with_labels:
        shuffled = [gold[a["id"]] for a in raw]
        rng.shuffle(shuffled)
        gold = {a["id"]: lab for a, lab in zip(raw, shuffled)}
    elif permute == "cross" and with_labels:
        gold = {a["id"]: rng.choices(LABELS, weights=LABEL_MARGINAL)[0] for a in raw}
    now = case["now_day"]
    n = len(A)
    fields = sorted(set(a["field"] for a in A))
    loc_field = {f: v for f, v in zip(fields, local_slots(rng, N_LOCAL_FIELDS, len(fields)))}
    sources = sorted(set(a["source"] for a in A))
    loc_src = {s: v for s, v in zip(sources, local_slots(rng, N_LOCAL_SOURCES, len(sources)))}
    corrected_ids = set(a["corrects"] for a in A if a.get("corrects") is not None)

    nv = {a["id"]: norm_value(a["field"], a["value"], a.get("unit")) for a in A}
    id_pos = {a["id"]: i for i, a in enumerate(A)}
    by_id = {a["id"]: a for a in A}
    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for a in A:
        by_field.setdefault(a["field"], []).append(a)

    cluster_of: Dict[int, int] = {}          # assertion id -> local cluster slot
    clusters_by_field: Dict[str, List[Dict[str, Any]]] = {}
    for f, group in by_field.items():
        cl = cluster(f, [(a["id"], nv[a["id"]]) for a in group])
        clusters_by_field[f] = cl
        slots = local_slots(rng, N_LOCAL_CLUSTERS, len(cl))
        for ci, c in enumerate(cl):
            c["answer"] = cluster_answer(f, [by_id[m] for m in c["members"]])
            for m in c["members"]:
                cluster_of[m] = slots[ci]

    ftype_idx = np.zeros(n, dtype=np.int64)
    field_idx = np.zeros(n, dtype=np.int64)
    lfield_idx = np.zeros(n, dtype=np.int64)
    src_idx = np.zeros(n, dtype=np.int64)
    lsrc_idx = np.zeros(n, dtype=np.int64)
    clus_idx = np.zeros(n, dtype=np.int64)
    feats = np.zeros((n, N_NUM_FEATS), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)

    for f, group in by_field.items():
        numeric = is_numeric(f)
        vals = [nv[a["id"]] for a in group]
        med = float(np.median(vals)) if numeric else 0.0
        order = sorted(group, key=lambda x: (-x["observed_day"], -x["arrived_day"]))
        rank = {a["id"]: r for r, a in enumerate(order)}
        newest_arr = max(group, key=lambda x: (x["arrived_day"], x["observed_day"]))["id"]
        cl = clusters_by_field[f]
        size_of = {}
        for c in cl:
            for m in c["members"]:
                size_of[m] = len(c["members"])
        for a in group:
            i = id_pos[a["id"]]
            ftype_idx[i] = FTYPES.index(ftype_of(f)) if ftype_of(f) in FTYPES else FTYPES.index("cat")
            field_idx[i] = FIELDS.index(f) if f in FIELDS else len(FIELDS)
            lfield_idx[i] = loc_field[f]
            src_idx[i] = SOURCES.index(a["source"]) if a["source"] in SOURCES else len(SOURCES)
            lsrc_idx[i] = loc_src[a["source"]]
            clus_idx[i] = cluster_of[a["id"]]
            v = nv[a["id"]]
            conf = a.get("extractor_conf")
            feats[i] = [
                float(np.clip((v - med) / max(abs(med), 1e-6), -5, 5)) if numeric else 0.0,
                float(np.clip(math.log(max(v, 1e-6) / max(med, 1e-6)), -5, 5)) if (numeric and v > 0 and med > 0) else 0.0,
                min(6.0, (now - a["observed_day"]) / 365.0),
                min(6.0, (a["arrived_day"] - a["observed_day"]) / 365.0),
                min(6.0, (now - a["arrived_day"]) / 365.0),
                rank[a["id"]] / max(1, len(group)),
                1.0 if rank[a["id"]] == 0 else 0.0,
                1.0 if a["id"] == newest_arr else 0.0,
                float(conf) if conf is not None else 1.0,
                1.0 if conf is None else 0.0,
                1.0 if a.get("unit") == "lb" else 0.0,
                1.0 if a.get("text") else 0.0,
                1.0 if a.get("corrects") is not None else 0.0,
                1.0 if a["id"] in corrected_ids else 0.0,
                size_of[a["id"]] / max(1, len(group)),
                min(1.0, len(group) / 80.0),
            ]
            if with_labels:
                labels[i] = LABELS.index(gold[a["id"]])

    # queries: for each queried field, the candidate clusters and (if labels) a 0/1 target per candidate: 1 when the
    # candidate's stated value matches the true current value (several can; when none does, "none of these" is right)
    queries = []
    for q in case["queries"]:
        f = q["field"]
        if f not in clusters_by_field:
            continue
        cl = clusters_by_field[f]
        members = [[m for m in c["members"]] for c in cl]
        targets, truth = [], None
        if with_labels:
            truth = current_truth(case, f)
            targets = [1.0 if (truth is not None and match(f, c["answer"], truth)) else 0.0 for c in cl]
        queries.append({"field": f, "member_ids": members, "values": [c["answer"] for c in cl],
                        "targets": targets, "truth": truth, "decidable": q["decidable"]})
    if with_labels and permute == "within":
        pooled = [t for q in queries for t in q["targets"]]
        rng.shuffle(pooled)
        k = 0
        for q in queries:
            q["targets"] = pooled[k:k + len(q["targets"])]
            k += len(q["targets"])
    elif with_labels and permute == "cross":
        for q in queries:
            q["targets"] = [1.0 if rng.random() < POS_RATE[0] else 0.0 for _ in q["targets"]]
    for q in queries:
        q["member_pos"] = [[id_pos[m] for m in ms if m in id_pos] for ms in q["member_ids"]]
    return {"case_id": case["case_id"], "n": n, "ids": [a["id"] for a in A],
            "ftype": ftype_idx, "field": field_idx, "lfield": lfield_idx, "src": src_idx, "lsrc": lsrc_idx,
            "clus": clus_idx, "feats": feats, "labels": labels, "queries": queries}


def collate(encoded: List[Dict[str, Any]], device) -> Dict[str, Any]:
    B = len(encoded)
    T = max(e["n"] for e in encoded)
    def pad_int(key):
        out = np.zeros((B, T), dtype=np.int64)
        for b, e in enumerate(encoded):
            out[b, :e["n"]] = e[key]
        return torch.from_numpy(out).to(device)
    feats = np.zeros((B, T, N_NUM_FEATS), dtype=np.float32)
    mask = np.ones((B, T), dtype=bool)          # True = padding
    for b, e in enumerate(encoded):
        feats[b, :e["n"]] = e["feats"]
        mask[b, :e["n"]] = False
    return {"ftype": pad_int("ftype"), "field": pad_int("field"), "lfield": pad_int("lfield"),
            "src": pad_int("src"), "lsrc": pad_int("lsrc"), "clus": pad_int("clus"),
            "feats": torch.from_numpy(feats).to(device), "labels": pad_int("labels"),
            "pad": torch.from_numpy(mask).to(device), "encoded": encoded}


# ============================================================ model
class Reconciler(nn.Module):
    def __init__(self):
        super().__init__()
        e = 24
        self.emb_ftype = nn.Embedding(len(FTYPES), e)
        self.emb_field = nn.Embedding(len(FIELDS) + 1, e)
        self.emb_lfield = nn.Embedding(N_LOCAL_FIELDS, e)
        self.emb_src = nn.Embedding(len(SOURCES) + 1, e)
        self.emb_lsrc = nn.Embedding(N_LOCAL_SOURCES, e)
        self.emb_clus = nn.Embedding(N_LOCAL_CLUSTERS, e)
        self.inp = nn.Sequential(nn.Linear(6 * e + N_NUM_FEATS, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, D_MODEL))
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, FF, DROPOUT, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(D_MODEL)
        self.entry_head = nn.Linear(D_MODEL, 3)
        self.query_head = nn.Sequential(nn.Linear(3 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1))
        self.null_rep = nn.Parameter(torch.randn(D_MODEL) * 0.02)      # v0.5: the "none of these" option

    def forward(self, batch):
        x = torch.cat([self.emb_ftype(batch["ftype"]), self.emb_field(batch["field"]), self.emb_lfield(batch["lfield"]),
                       self.emb_src(batch["src"]), self.emb_lsrc(batch["lsrc"]), self.emb_clus(batch["clus"]),
                       batch["feats"]], dim=-1)
        h = self.inp(x)
        h = self.encoder(h, src_key_padding_mask=batch["pad"])
        h = self.norm(h)
        return h

    def entry_logits(self, h):
        return self.entry_head(h)

    def query_logits(self, h_b, q):
        """h_b: (T, D) for one case; q: query dict -> C + 1 logits: one per candidate cluster, then "none of these"
        (read through one softmax)."""
        field_pos = [p for ms in q["member_pos"] for p in ms]
        field_rep = h_b[field_pos].mean(0)
        reps = torch.stack([h_b[ms].mean(0) for ms in q["member_pos"]] + [self.null_rep])   # (C + 1, D)
        fr = field_rep.unsqueeze(0).expand_as(reps)
        return self.query_head(torch.cat([reps, fr, reps * fr], dim=-1)).squeeze(-1)


# ============================================================ train / eval
def pick_device() -> torch.device:
    if DEVICE != "auto":
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def gen_cases(n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    return [gen.gen_case(rng, "train", i) for i in range(n)]


def step_loss(model, batch):
    h = model(batch)
    el = model.entry_logits(h)
    valid = ~batch["pad"]
    loss_e = F.cross_entropy(el[valid], batch["labels"][valid])
    logits, targets = [], []
    for b, e in enumerate(batch["encoded"]):
        for q in e["queries"]:
            if len(q["member_pos"]) < 1 or len(q["targets"]) != len(q["member_pos"]):
                continue
            logits.append(model.query_logits(h[b], q))
            targets.extend(q["targets"])
    # -log of the total probability of the right answers (every matching candidate), or of "none" when no candidate
    # matches; averaged over queries
    losses, k = [], 0
    for lg in logits:
        tg = torch.tensor(targets[k:k + len(lg) - 1], device=lg.device)
        k += len(lg) - 1
        lse = torch.logsumexp(lg, 0)
        if tg.sum() > 0.5:
            losses.append(lse - torch.logsumexp(lg[:-1][tg > 0.5], 0))
        else:
            losses.append(lse - lg[-1])
    loss_q = torch.stack(losses).mean() if losses else torch.zeros((), device=h.device)
    return loss_e + loss_q, loss_e.item(), loss_q.item()


@torch.no_grad()
def predict(model, cases: List[Dict[str, Any]], device, rng: random.Random, batch_size: int = 16,
            with_labels: bool = True, digits: Optional[int] = 4):
    """Returns preds in the standard format plus the flat arrays for quick metrics (empty if not with_labels).
    digits = None keeps full floats (the leak test compares unrounded outputs). Cases without assertions get an
    empty record, so every case is covered."""
    rnd = (lambda x: round(x, digits)) if digits is not None else (lambda x: x)
    model.eval()
    preds, q_conf, q_ok, q_dec, e_true, e_pred, e_perr = [], [], [], [], [], [], []
    too_long = [c["case_id"] for c in cases if len(c["assertions"]) > EVAL_MAX_ENTRIES]
    if too_long:
        print("  WARNING: %d records longer than EVAL_MAX_ENTRIES=%d; their extra entries get no prediction: %s"
              % (len(too_long), EVAL_MAX_ENTRIES, ", ".join(too_long[:5])))
    i = 0
    while i < len(cases):
        chunk = cases[i:i + batch_size]
        i += batch_size
        enc = [encode_case(c, rng, with_labels=with_labels, max_entries=EVAL_MAX_ENTRIES) for c in chunk]
        for c, e in zip(chunk, enc):
            if e is None:                                # a record with no assertions: nothing to state
                preds.append({"case_id": c["case_id"], "method": METHOD_NAME, "queries": {}, "assertions": {}})
        pairs = [(c, e) for c, e in zip(chunk, enc) if e is not None]
        if not pairs:
            continue
        batch = collate([e for _, e in pairs], device)
        h = model(batch)
        probs = F.softmax(model.entry_logits(h), dim=-1).cpu().numpy()
        for b, (case, e) in enumerate(pairs):
            rec = {"case_id": case["case_id"], "method": METHOD_NAME, "queries": {}, "assertions": {}}
            for pos, aid in enumerate(e["ids"]):
                p = probs[b, pos]
                lab = LABELS[int(np.argmax(p))]
                rec["assertions"][str(aid)] = {"label": lab, "p_valid": rnd(float(p[0])),
                                               "p_superseded": rnd(float(p[1])), "p_erroneous": rnd(float(p[2]))}
                if with_labels:
                    e_true.append(e["labels"][pos]); e_pred.append(int(np.argmax(p))); e_perr.append(float(p[2]))
            for q in e["queries"]:
                if not q["member_pos"]:
                    continue
                lg = model.query_logits(h[b], q)
                pq = F.softmax(lg, dim=0).cpu().numpy()[:-1]      # candidates (the last option is "none of these")
                best = int(np.argmax(pq))
                # P(the stated value is right) = the probability of every candidate whose value matches it
                conf = float(sum(pq[j] for j in range(len(pq)) if match(q["field"], q["values"][j], q["values"][best])))
                rec["queries"][q["field"]] = {"value": q["values"][best], "confidence": rnd(conf)}
                if with_labels:
                    ok = 1 if (q["truth"] is not None and match(q["field"], q["values"][best], q["truth"])) else 0
                    q_conf.append(conf); q_ok.append(ok); q_dec.append(q["decidable"])
            preds.append(rec)
    model.train()
    return preds, (q_conf, q_ok, q_dec, e_true, e_pred, e_perr)


def quick_metrics(stats) -> str:
    q_conf, q_ok, q_dec, e_true, e_pred, e_perr = stats
    dec = [i for i, d in enumerate(q_dec) if d]
    acc = sum(q_ok[i] for i in dec) / max(1, len(dec))
    ec = ece_score([q_conf[i] for i in dec], [q_ok[i] for i in dec])
    hi = [i for i in dec if q_conf[i] >= 0.9]
    wrong_ow = (1 - sum(q_ok[i] for i in hi) / len(hi)) if hi else float("nan")
    und = [q_conf[i] for i, d in enumerate(q_dec) if not d]
    labs_t = [LABELS[t] for t in e_true]; labs_p = [LABELS[p] for p in e_pred]
    present = [l for l in LABELS if l in set(labs_t)]
    f1s = [prf(labs_p, labs_t, l)[2] for l in present]
    pe, re, _ = prf(labs_p, labs_t, "erroneous")
    a = rank_auc(e_perr, [1 if t == 2 else 0 for t in e_true])
    return ("q_acc %.3f | ECE %.3f | wrongOW@.9 %.3f (cov %.2f) | undecC %.2f | entry macroF1 %.3f | err P/R %.2f/%.2f | AUCerr %.3f"
            % (acc, ec, wrong_ow, len(hi) / max(1, len(dec)), (sum(und) / len(und)) if und else float("nan"),
               sum(f1s) / max(1, len(f1s)), pe, re, a))


# ============================================================ leak test
ALLOWED_CASE_KEYS = ("case_id", "now_day", "field_types")      # field_types = the schema (type + tolerance): observable


def scrub(case: Dict[str, Any], i: int) -> Dict[str, Any]:
    """A copy of the case holding only what a method may see: the observable assertion keys, "now", the queried
    field names and the schema. Query answers, types and decidable flags are removed (decidable is set True) and the
    case id (which names the split) is replaced by an opaque one."""
    out = {k: case[k] for k in ALLOWED_CASE_KEYS if k in case}
    out["case_id"] = "case-%06d" % i
    out["assertions"] = [{k: a.get(k) for k in OBSERVABLE_KEYS} for a in case["assertions"]]
    out["queries"] = [{"field": q["field"], "decidable": True} for q in case["queries"]]
    return out


def leak_test(model, device) -> bool:
    """Identical (unrounded) predictions from scrubbed inputs, on every split present. Both sets of predictions are
    made on the CPU with a copy of the model, so a GPU's (MPS) run-to-run arithmetic cannot cause a false alarm."""
    import copy
    device = torch.device("cpu")
    model = copy.deepcopy(model).to(device)
    ok_all = True
    names = list(EVAL_SPLITS.items()) + [(s, ("cases_%s.jsonl" % s, None)) for s in REAL_SPLITS]
    for split, (path, cap) in names:
        if not os.path.exists(path):
            continue
        cases = load_jsonl(path)
        if cap:
            cases = cases[:cap]
        bs = 4 if split in REAL_SPLITS else 16
        full, _ = predict(model, cases, device, random.Random(SEED + 5), batch_size=bs, with_labels=True, digits=None)
        scr, _ = predict(model, [scrub(c, i) for i, c in enumerate(cases)], device, random.Random(SEED + 5),
                         batch_size=bs, with_labels=False, digits=None)
        n_q = n_e = d_q = d_e = 0
        for a, b in zip(full, scr):
            for f, it in a["queries"].items():
                n_q += 1
                if it != b["queries"].get(f):
                    d_q += 1
            for k, it in a["assertions"].items():
                n_e += 1
                if it != b["assertions"].get(k):
                    d_e += 1
        same = d_q == 0 and d_e == 0 and len(full) == len(scr)
        ok_all = ok_all and same
        print("  leak test %-18s %s: %d/%d answers and %d/%d entry predictions differ" % (
            split, "PASS" if same else "FAIL", d_q, n_q, d_e, n_e))
    print("LEAK TEST %s" % ("PASSED: predictions use only the observable keys" if ok_all else "FAILED -- stop and audit"))
    return ok_all


# ============================================================ main
def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = pick_device()
    model = Reconciler().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("%s | device %s | params %.2fM | d_model %d layers %d heads %d | seed %d" % (
        METHOD_NAME, device, n_params / 1e6, D_MODEL, N_LAYERS, N_HEADS, SEED))
    print("encoder input keys: %s" % ", ".join(OBSERVABLE_KEYS))
    if PERMUTE:
        print("*** PERMUTE = %r control run: training targets are shuffled; results should be at chance ***" % PERMUTE)
    if PERMUTE == "cross":
        # the two marginals the cross-case control draws from, measured on 1,000 generated training cases (as encoded
        # for training: first MAX_ENTRIES entries)
        sample = gen_cases(1000, seed=987654)
        counts = {l: 0 for l in LABELS}
        for c in sample:
            for a in c["assertions"][:MAX_ENTRIES]:
                counts[a["label"]] += 1
        tot = float(sum(counts.values()))
        LABEL_MARGINAL[:] = [counts[l] / tot for l in LABELS]
        pos = n_cand = 0
        m_rng = random.Random(987654)
        for c in sample:
            e = encode_case(c, m_rng, permute=None)
            if e is None:
                continue
            for q in e["queries"]:
                pos += sum(q["targets"]); n_cand += len(q["targets"])
        POS_RATE[:] = [pos / max(1, n_cand)]
        print("cross-case targets: entry labels from the train marginal %s; each candidate correct with p = %.4f"
              % ([round(x, 4) for x in LABEL_MARGINAL], POS_RATE[0]))

    val_cases = load_jsonl(EVAL_SPLITS["heldout"][0])[:VAL_CASES] if os.path.exists(EVAL_SPLITS["heldout"][0]) else []

    if not EVAL_ONLY:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        steps_per_epoch = max(1, N_TRAIN_CASES // BATCH_CASES)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch, pct_start=0.1)
        for epoch in range(EPOCHS):
            t0 = time.time()
            enc_rng = random.Random(SEED * 1000 + epoch)
            tot, tot_e, tot_q, nb = 0.0, 0.0, 0.0, 0
            for s in range(steps_per_epoch):
                cases = gen_cases(BATCH_CASES, seed=SEED * 10 ** 6 + epoch * 10 ** 4 + s)
                enc = [e for e in (encode_case(c, enc_rng, permute=PERMUTE) for c in cases) if e is not None]
                batch = collate(enc, device)
                loss, le, lq = step_loss(model, batch)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                tot += loss.item(); tot_e += le; tot_q += lq; nb += 1
                if (s + 1) % 100 == 0:
                    print("  epoch %d step %d/%d loss %.3f (entry %.3f, query %.3f) %.0fs"
                          % (epoch + 1, s + 1, steps_per_epoch, tot / nb, tot_e / nb, tot_q / nb, time.time() - t0))
            msg = "epoch %d done in %.0fs | train loss %.3f" % (epoch + 1, time.time() - t0, tot / max(1, nb))
            if val_cases:
                _, stats = predict(model, val_cases, device, random.Random(SEED + 99))
                msg += " | heldout(%d, monitoring only): %s" % (len(val_cases), quick_metrics(stats))
            print(msg)
            torch.save(model.state_dict(), CHECKPOINT)
    else:
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
        print("loaded", CHECKPOINT)

    for split, (path, cap) in list(EVAL_SPLITS.items()) + [(s, ("cases_%s.jsonl" % s, None)) for s in REAL_SPLITS]:
        if not os.path.exists(path):
            continue
        cases = load_jsonl(path)
        if cap:
            cases = cases[:cap]
        bs = 4 if split in REAL_SPLITS else 16
        preds, stats = predict(model, cases, device, random.Random(SEED + 5), batch_size=bs)
        write_jsonl("preds_%s_%s.jsonl" % (METHOD_NAME, split), preds)
        print("%-18s %4d cases | %s" % (split, len(preds), quick_metrics(stats)))
    if LEAK_TEST:
        leak_test(model, device)
    print("done. now run calibrate.py (temperature scaling on val), then score.py")


if __name__ == "__main__":
    main()
