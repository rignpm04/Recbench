# llm_costs.py -- exact token / cost / coverage accounting for the LLM runs (recbench v0.5)
# v0.5: prices per provider tag (llm_deepseek* / llm_claude*), the same numbers as in llm_baseline.py; a reply made
# through Anthropic's Batch API carries price_factor 0.5 and is costed at half price. "cut off" counts replies that hit
# the output cap (finish_reason "length").
# Run from PyCharm after llm_baseline.py. Reads replies_llm_*.jsonl (one line per API reply, written by
# llm_baseline.py since v0.4: tokens from the API's usage field) and the matching preds files. Python 3.9, stdlib only.
# Prints one row per (method, split): replies, cases, tokens in / out / cache-hit, estimated cost at the prices below,
# finish reasons, and how many cases came back with no current values (the parser-coverage number for RESULTS.md).

import glob
import json
import os
from typing import Any, Dict, List

from recbench_common import KNOWN_SPLITS

# USD per 1M (input, output) tokens by provider tag at the normal rate (DeepSeek bills cache hits and off-peak hours
# lower; the provider's console is the true cost)
PRICES = {"deepseek": (0.15, 0.60), "claude": (2.00, 10.00)}


def prices_for(method: str):
    tag = method[len("llm_"):].split("_")[0] if method.startswith("llm_") else ""
    return PRICES.get(tag, (0.0, 0.0))


def load(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def known_splits() -> List[str]:
    names = [os.path.basename(p)[len("cases_"):-len(".jsonl")] for p in glob.glob("cases_*.jsonl")] + KNOWN_SPLITS
    return sorted(set(names), key=lambda s: -len(s))


def split_method(name: str):
    for s in known_splits():
        if name.endswith("_" + s):
            return name[:-len(s) - 1], s
    return name, "?"


def main() -> None:
    rows = []
    for path in sorted(glob.glob("replies_llm_*.jsonl")):
        stem = os.path.basename(path)[len("replies_"):-len(".jsonl")]
        method, split = split_method(stem)
        replies = load(path)
        tin = sum(int(r.get("prompt_tokens", 0)) for r in replies)
        tout = sum(int(r.get("completion_tokens", 0)) for r in replies)
        tcache = sum(int(r.get("cache_hit_tokens", 0) or 0) for r in replies)
        finish: Dict[str, int] = {}
        for r in replies:
            k = str(r.get("finish_reason"))
            finish[k] = finish.get(k, 0) + 1
        cases = set(r["case_id"] for r in replies)
        p_in, p_out = prices_for(method)
        cost = sum(float(r.get("price_factor", 1.0) or 1.0) * (int(r.get("prompt_tokens", 0)) / 1e6 * p_in
                                                               + int(r.get("completion_tokens", 0)) / 1e6 * p_out)
                   for r in replies)
        finish["cut off"] = sum(1 for r in replies if r.get("finish_reason") == "length")
        # coverage from the (raw) preds file
        pred_path = "raw_preds_%s_%s.jsonl" % (method, split)
        if not os.path.exists(pred_path):
            pred_path = "preds_%s_%s.jsonl" % (method, split)
        n_pred = n_empty = n_q = n_ans = 0
        if os.path.exists(pred_path) and os.path.exists("cases_%s.jsonl" % split):
            preds = {p["case_id"]: p for p in load(pred_path)}
            qcount = {c["case_id"]: len(c["queries"]) for c in load("cases_%s.jsonl" % split)}
            for cid, p in preds.items():
                if not qcount.get(cid, 0):
                    continue                      # a case with nothing to ask (an empty record) is not a failure
                n_pred += 1
                n_q += qcount[cid]
                n_ans += len(p["queries"])
                if not p["queries"]:
                    n_empty += 1
        rows.append((method, split, len(replies), len(cases), tin, tout, tcache, cost, finish, n_pred, n_empty, n_q, n_ans))

    if not rows:
        print("no replies_llm_*.jsonl files found (they are written by llm_baseline.py v0.4+)")
        return
    print("%-20s %-9s %7s %6s %11s %11s %9s %8s  %s" % ("method", "split", "replies", "cases", "tokens_in", "tokens_out", "cache_hit", "est_$", "finish reasons | cases w/o answers | queries answered"))
    tot_in = tot_out = 0
    tot_cost = 0.0
    for m, s, nr, nc, tin, tout, tc, cost, finish, n_pred, n_empty, n_q, n_ans in rows:
        tot_in += tin; tot_out += tout; tot_cost += cost
        cov = ("%d/%d cases no answers | %d/%d queries (%.1f%%)" % (n_empty, n_pred, n_ans, n_q, 100.0 * n_ans / n_q)) if n_q else "-"
        print("%-20s %-9s %7d %6d %11d %11d %9d %8.3f  %s | %s" % (m, s, nr, nc, tin, tout, tc, cost, finish, cov))
    print("\ntotal: tokens in %d out %d | est. $%.2f at the PRICES per provider, batch replies at half price (cache hits are "
          "billed lower; the provider's console is the true cost)" % (tot_in, tot_out, tot_cost))


if __name__ == "__main__":
    main()
