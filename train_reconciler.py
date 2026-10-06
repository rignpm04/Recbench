# train_reconciler.py -- the PFN-style reconciler (recbench v0.3)
#
# A small transformer that reads a record as a SET of assertions (value, unit, two dates, source, confidence) and
# outputs (a) valid / superseded / erroneous probabilities for every assertion and (b) for every queried field, a
# probability over the candidate values present in the record. Trained ONLY on freshly generated cases from gen.py
# (the prior); never sees a real label. Writes preds_reconciler_<split>.jsonl in the standard format, so score.py
# compares it against every baseline. Also scores the real sets (stock / flight / book) zero-shot if present.
#
# Needs: torch, numpy (python 3.10+ venv). From PyCharm: set this script's interpreter to recbench_env/.venv and
# press Run. Prints metrics every epoch. Checkpoint: reconciler.pt (set EVAL_ONLY = True to just predict with it).

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
from recbench_common import (FIELDS, FIELD_TYPE, LABELS, SOURCES, cluster, current_truth, ftype_of, is_numeric,
                             load_jsonl, match, norm_value, write_jsonl)
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
MAX_ENTRIES = 640               # longest record the model accepts (stock cases have ~560 entries)
SEED = 1
VAL_CASES = 300                 # heldout cases used for the per-epoch metrics
EVAL_ONLY = False               # True: load CHECKPOINT and only write predictions
CHECKPOINT = "reconciler.pt"
METHOD_NAME = "reconciler"
EVAL_SPLITS = {"train": ("cases_train.jsonl", 500), "heldout": ("cases_heldout.jsonl", None),
               "hard": ("cases_hard.jsonl", None)}
REAL_SPLITS = ["stock", "flight", "book", "book_subset"]     # scored zero-shot when cases_<name>.jsonl exists

FTYPES = ["immutable_cat", "immutable_num", "drift_num", "regime_cat", "event_day", "num_rel", "num_abs", "cat"]
N_LOCAL_FIELDS = 32
N_LOCAL_SOURCES = 96
N_LOCAL_CLUSTERS = 64
N_NUM_FEATS = 16


# ============================================================ encoding
def encode_case(case: Dict[str, Any], rng: random.Random, with_labels: bool = True) -> Optional[Dict[str, Any]]:
    """Turn one case into index/feature arrays plus the query structure. Uses only observable fields."""
    A = case["assertions"][:MAX_ENTRIES]
    if not A:
        return None
    now = case["now_day"]
    n = len(A)
    fields = sorted(set(a["field"] for a in A))
    loc_field = {f: v for f, v in zip(fields, rng.sample(range(N_LOCAL_FIELDS), len(fields)))}
    sources = sorted(set(a["source"] for a in A))
    loc_src = {s: v for s, v in zip(sources, rng.sample(range(N_LOCAL_SOURCES), len(sources)))}
    corrected_ids = set(a["corrects"] for a in A if a.get("corrects") is not None)

    nv = {a["id"]: norm_value(a["field"], a["value"], a.get("unit")) for a in A}
    id_pos = {a["id"]: i for i, a in enumerate(A)}
    by_field: Dict[str, List[Dict[str, Any]]] = {}
    for a in A:
        by_field.setdefault(a["field"], []).append(a)

    cluster_of: Dict[int, int] = {}          # assertion id -> local cluster slot
    clusters_by_field: Dict[str, List[Dict[str, Any]]] = {}
    cl_slot: Dict[Tuple[str, int], int] = {}  # (field, cluster idx) -> local slot
    for f, group in by_field.items():
        cl = cluster(f, [(a["id"], nv[a["id"]]) for a in group])
        clusters_by_field[f] = cl
        slots = rng.sample(range(N_LOCAL_CLUSTERS), min(len(cl), N_LOCAL_CLUSTERS))
        for ci, c in enumerate(cl):
            slot = slots[ci % len(slots)]
            cl_slot[(f, ci)] = slot
            for m in c["members"]:
                cluster_of[m] = slot

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
                labels[i] = LABELS.index(a["label"])

    # queries: for each queried field, the candidate clusters and (if labels) the index of the true one
    queries = []
    for q in case["queries"]:
        f = q["field"]
        if f not in clusters_by_field:
            continue
        cl = clusters_by_field[f]
        members = [[m for m in c["members"]] for c in cl]
        target = -1
        if with_labels:
            truth = current_truth(case, f)
            for ci, c in enumerate(cl):
                if match(f, c["value"], truth):
                    target = ci
                    break
        queries.append({"field": f, "member_ids": members, "values": [c["value"] for c in cl],
                        "target": target, "decidable": q["decidable"]})
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
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)
        self.norm = nn.LayerNorm(D_MODEL)
        self.entry_head = nn.Linear(D_MODEL, 3)
        self.query_head = nn.Sequential(nn.Linear(3 * D_MODEL, D_MODEL), nn.GELU(), nn.Linear(D_MODEL, 1))

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
        """h_b: (T, D) for one case; q: query dict -> logits over its candidate clusters."""
        field_pos = [p for ms in q["member_pos"] for p in ms]
        field_rep = h_b[field_pos].mean(0)
        reps = torch.stack([h_b[ms].mean(0) for ms in q["member_pos"]])          # (C, D)
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
    q_losses = []
    for b, e in enumerate(batch["encoded"]):
        for q in e["queries"]:
            if q["target"] < 0 or len(q["member_pos"]) < 1:
                continue
            lg = model.query_logits(h[b], q)
            q_losses.append(F.cross_entropy(lg.unsqueeze(0), torch.tensor([q["target"]], device=lg.device)))
    loss_q = torch.stack(q_losses).mean() if q_losses else torch.zeros((), device=h.device)
    return loss_e + loss_q, loss_e.item(), loss_q.item()


@torch.no_grad()
def predict(model, cases: List[Dict[str, Any]], device, rng: random.Random, batch_size: int = 16):
    """Returns preds in the standard format plus the flat arrays for quick metrics."""
    model.eval()
    preds, q_conf, q_ok, q_dec, e_true, e_pred, e_perr = [], [], [], [], [], [], []
    i = 0
    while i < len(cases):
        chunk = cases[i:i + batch_size]
        i += batch_size
        enc = [encode_case(c, rng, with_labels=True) for c in chunk]
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
                rec["assertions"][str(aid)] = {"label": lab, "p_valid": round(float(p[0]), 4),
                                               "p_superseded": round(float(p[1]), 4), "p_erroneous": round(float(p[2]), 4)}
                e_true.append(e["labels"][pos]); e_pred.append(int(np.argmax(p))); e_perr.append(float(p[2]))
            for q in e["queries"]:
                if not q["member_pos"]:
                    continue
                lg = model.query_logits(h[b], q)
                pq = F.softmax(lg, dim=-1).cpu().numpy()
                best = int(np.argmax(pq))
                conf = float(pq[best])
                rec["queries"][q["field"]] = {"value": q["values"][best], "confidence": round(conf, 4)}
                ok = 1 if (q["target"] >= 0 and best == q["target"]) else 0
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
    f1s = [prf(labs_p, labs_t, l)[2] for l in LABELS]
    pe, re, _ = prf(labs_p, labs_t, "erroneous")
    a = rank_auc(e_perr, [1 if t == 2 else 0 for t in e_true])
    return ("q_acc %.3f | ECE %.3f | wrongOW@.9 %.3f (cov %.2f) | undecC %.2f | entry macroF1 %.3f | err P/R %.2f/%.2f | AUCerr %.3f"
            % (acc, ec, wrong_ow, len(hi) / max(1, len(dec)), (sum(und) / len(und)) if und else float("nan"),
               sum(f1s) / 3, pe, re, a))


def main() -> None:
    torch.manual_seed(SEED); np.random.seed(SEED)
    device = pick_device()
    model = Reconciler().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("device %s | params %.2fM | d_model %d layers %d heads %d" % (device, n_params / 1e6, D_MODEL, N_LAYERS, N_HEADS))

    val_cases = load_jsonl(EVAL_SPLITS["heldout"][0])[:VAL_CASES] if os.path.exists(EVAL_SPLITS["heldout"][0]) else []
    eval_rng = random.Random(SEED + 777)

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
                enc = [e for e in (encode_case(c, enc_rng) for c in cases) if e is not None]
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
                msg += " | heldout(%d): %s" % (len(val_cases), quick_metrics(stats))
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
        print("%-12s %4d cases | %s" % (split, len(preds), quick_metrics(stats)))
    print("done. now run score.py")


if __name__ == "__main__":
    main()
