# parliament_map.py -- the UK Parliament real-data splits for recbench (Oct 2026). No recbench method runs here.
#
# Run from PyCharm's Run button (Python 3.9, standard library only, needs internet the first time). Needs
# parliament_probe.py in the same folder: the downloads, the parsers and the answer key come from it, and so does its
# download cache (parliament_data/probe/cache/), so nothing is fetched twice. Stop and restart at any time: every
# download is cached, so it continues where it stopped; once everything is downloaded a run takes a few minutes.
# Everything it keeps goes under parliament_data/ (add it to .gitignore). The Oct 10 builds kept it in realdata/,
# which realdata_map.py reads in full; it is moved on the first run.
#
# Run it only after the Parliament predictions (PREDICTIONS.md, Parliament amendment) are committed. Two runs:
#   1. Phase 1: BUILD_TEST = True, INSTALL_TEST = False. Writes the development files next to the scripts first
#      (cases_parliament_dev.jsonl, cases_parliament_dev_nobot.jsonl: the dress rehearsal reads them; about 10
#      minutes, all cached), then downloads the 400 test members (about 6 hours; phase 1 can run meanwhile) and seals
#      the test files in parliament_data/cases/, where no recbench script looks. Their labels are not printed; their
#      SHA-256 fingerprints are (they go into the "v0.5: frozen" commit message).
#   2. Phase 2 (after "v0.5: frozen"): INSTALL_TEST = True. Copies the sealed test files next to the scripts -- no
#      rebuild, no download -- checks the fingerprints and prints the test summary.
#   It stops if any file for the test split sits next to the scripts while INSTALL_TEST = False.
#
# Members (same list, seed and order as parliament_probe.py)
#   frame   every House of Commons member active between FRAME_START and RUN_DATE (UK Parliament Members API)
#   dev     the 50 members the probe drew -> parliament_dev: development data (checks the mapping; the dress rehearsal
#           before the freeze); never part of a result
#   test    the next N_TEST members in the same seeded order -> parliament_test: built after the predictions were
#           committed, sealed until phase 2; the LLMs read its first LLM_CASES cases
#
# One case per member, in recbench's case format (no "truth" key: the "real" regime, like the Stock / Flight / Book
# sets, but longitudinal)
#   fields   parliament.party and parliament.constituency (regime_cat: a category that holds for a period, then
#            changes -- recbench's type for a pet's diet or clinic), parliament.majority (drift_num: a number that
#            changes over time, the type of a pet's weight; compared exactly)
#   entries  every change of a field's value in the member's Wikidata item and English Wikipedia article (the lead
#            section's infobox), from the first revision up to the reference date. An entry is dated by its edit:
#            observed day = arrived day = the edit's day, counted from the case's first entry (day 0). Entries are
#            listed day by day (arrival order at the record's resolution of one day); entries of the same day are in
#            a seeded random order, so the order inside a day carries no information (v0.5 change 6). Values that
#            say "not an MP" (every seat listed as ended -- how both sites mark dissolution), unreadable or broken
#            revisions, and revisions showing two different current values are gaps, not entries.
#   sources  enwiki; wikidata (edits by people); wikidata_bot (edits by bots or tools such as QuickStatements)
#   now      the reference date: RUN_DATE if the member is still in the Commons, else the member's last day there
#            (the earliest end date among Parliament's member endpoints)
#   answers  Parliament's records at the reference date: party, seat, and the majority of the election that began the
#            seat term. A field gets a query only when Parliament gives one clear value then and an entry exists.
#   labels   recbench's rule against Parliament's dated records: wrong on the entry's day -> erroneous; right then and
#            at the reference date -> valid; right then but changed since -> superseded. An entry is left out of
#            scoring (case["undecidable_ids"], as the generator's coin flips are) when Parliament has no clear value
#            on its day or at the reference date, or when moving it by one day would change its label. Such entries
#            stay in the case as inputs.
#
# Writes cases_parliament_dev.jsonl (+ cases_parliament_test.jsonl) and the *_nobot versions of each (the same members
# without wikidata_bot entries: bulk edits carry no references, so they may copy Parliament's own data; case ids
# "parliament_nobot-<member>"), and parliament_data/cases/parliament_members.json (who is in which split; no labels).

import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import parliament_probe as P   # noqa: E402  (shared downloads, parsers, answer key and cache)

# ============================================================ settings
BUILD_TEST = True                  # True: also build and seal the test split (400 members, ~6 hours the first time)
INSTALL_TEST = False               # phase 2 only (after "v0.5: frozen"): True copies the sealed test files next to the
                                   # scripts (no rebuild) and prints their summary; BUILD_TEST is then ignored
RUN_DATE = "2026-10-10"            # the probe's frozen run date: the frame and every current member's reference date
N_DEV = 50                         # the probe's sample
N_TEST = 400
LLM_CASES = 200                    # the LLM runs read the first LLM_CASES test cases (llm_baseline.py MAX_CASES)
WRITE_NOBOT = True
SHUFFLE_SEED = 20261010            # order of the entries inside a day (seeded, so file order carries no information)
FIELD_SPEC = {"party": ("parliament.party", ["regime_cat", None]),
              "constituency": ("parliament.constituency", ["regime_cat", None]),
              "majority": ("parliament.majority", ["drift_num", 0])}
MAJORITY_TOLERANCE = "0 votes (exact)"
MEMBERS_FILE = "parliament_members.json"
CASES_DIR = os.path.join(HERE, "parliament_data", "cases")   # sealed test files, member list
DAY = P.DAY


# ============================================================ members
def draw_members(frame: List[int], cw: Dict[str, str], run_t: int, frame_t0: int, n_wanted: int,
                 skipped: Counter) -> List[dict]:
    """The probe's draw (same seed, order and rules), continued to n_wanted accepted members."""
    order = list(frame)
    random.Random(P.SEED).shuffle(order)
    out: List[dict] = []
    for mid in order:
        if len(out) >= n_wanted:
            break
        hits = P.wd_by_parliament_id(mid)
        mys = cw.get(str(mid))
        if len(hits) == 1:
            qid, route = hits[0], "Wikidata " + P.WIKIDATA_PARL_ID
        elif mys:
            qid, route = mys, "mySociety"
        else:
            skipped["no Wikidata item found" if not hits else "several Wikidata items carry the id"] += 1
            continue
        h, bio, mem = P.member_records(mid)
        if not h:
            skipped["no record from /Members/History"] += 1
            continue
        g = P.Gold(h, bio, mem)
        win = g.window(frame_t0, run_t)
        if win is None:
            skipped["no Commons seat in the frame (History)"] += 1
            continue
        start, ref, ends, disagree = win
        if ref <= start + (P.SNAPSHOT_DAYS + 1) * DAY:        # the probe's rule, kept so the dev set is the same
            skipped["in the Commons under %d days in the frame" % (P.SNAPSHOT_DAYS + 1)] += 1
            continue
        out.append({"mid": mid, "qid": qid, "route": route, "gold": g, "start": start, "ref": ref,
                    "name": mem.get("nameDisplayAs") or h.get("nameDisplayAs") or "", "current": ref == run_t,
                    "end_disagree": disagree})
    return out


# ============================================================ full histories
def wikidata_full(qid: str, ref: int) -> List[Tuple[int, dict, Any, bool]]:
    meta = P.mw_revisions(P.WIKIDATA_API, {"prop": "revisions", "titles": qid, "rvprop": "ids|timestamp|user|comment|tags",
                                           "rvlimit": "max", "rvdir": "newer", "rvend": P.iso(ref)}, "Wikidata " + qid)
    keep = [r for r in meta if P.wb_action(r.get("comment")) not in P._SKIP_ACTIONS]
    ids = [r["revid"] for r in keep if isinstance(r.get("revid"), int)]
    stmts: Dict[int, Any] = {}
    for i in range(0, len(ids), 50):
        for rid, s in P.mw_revisions(P.WIKIDATA_API, {"prop": "revisions", "revids": "|".join(str(x) for x in ids[i:i + 50]),
                                                      "rvprop": "ids|content", "rvslots": "main"},
                                     transform=lambda r: (r.get("revid"), P.wd_extract(P.slot_text(r)))):
            stmts[rid] = s
    STATS["wikidata_revisions"] += len(meta)
    STATS["wikidata_revisions_read"] += len(keep)
    return [(P.ptime(r.get("timestamp")) or 0, r, stmts.get(r.get("revid")), False) for r in keep]


def enwiki_full(title: str, ref: int) -> List[Tuple[int, dict, Any, bool]]:
    def parse(r: dict) -> Tuple[dict, Optional[dict]]:
        text = P.slot_text(r)
        if text is None:
            return P.rev_meta(r), None
        params = P.infobox_params(text)
        return P.rev_meta(r), (None if params is None else P.wp_values(params))
    revs = P.mw_revisions(P.ENWIKI_API, {"prop": "revisions", "titles": title, "redirects": "1", "rvslots": "main",
                                         "rvsection": "0", "rvprop": "ids|timestamp|user|comment|tags|content",
                                         "rvlimit": "50", "rvdir": "newer", "rvend": P.iso(ref)},
                          "Wikipedia " + title, transform=parse)
    STATS["enwiki_revisions"] += len(revs)
    return [(P.ptime(meta.get("timestamp")) or 0, meta, vals, False) for meta, vals in revs]


# ============================================================ cases
STATS: Counter = Counter()


def case_value(field: str, v: Any) -> Any:
    if field == "majority":
        return int(v)
    v = str(v)
    return v[len("other:"):] if v.startswith("other:") else v


def placeholder_label(g: P.Gold, field: str, v: Any, t: int, ref: int) -> str:
    """Every entry needs a label in the file; a left-out (undecidable) entry gets its best guess here, which no score
    uses (score.py and calibrate.py skip case["undecidable_ids"])."""
    bad = (P.UNDEF, P.UNCLEAR)
    gt, gr = g.at(field, t), g.at(field, ref)
    if gt not in bad and gr not in bad:
        return "erroneous" if v != gt else ("valid" if v == gr else "superseded")
    if gr not in bad:
        return "valid" if v == gr else "erroneous"
    return "valid"


def build_case(m: dict, entries: List[dict], split: str) -> Optional[dict]:
    if not entries:
        return None
    g, ref = m["gold"], m["ref"]
    day0 = min(e["t"] for e in entries) // DAY

    def day(t: int) -> int:
        return int(t // DAY - day0)
    queries, truth, ftypes = [], {}, {}
    for f in P.FIELDS:
        name, spec = FIELD_SPEC[f]
        if not any(e["field"] == f for e in entries):
            continue
        ftypes[name] = spec
        gv = g.at(f, ref)
        if gv in (P.UNDEF, P.UNCLEAR):
            continue
        truth[name] = case_value(f, gv)
        q = {"field": name, "answer": truth[name], "answer_type": spec[0], "decidable": True}
        if f == "majority":
            q["tolerance"] = MAJORITY_TOLERANCE
        queries.append(q)
    if not queries:
        return None
    assertions, undecidable, keys = [], [], []
    for e in entries:
        lab = e["label"]
        a = {"id": None, "field": FIELD_SPEC[e["field"]][0], "value": case_value(e["field"], e["value"]), "unit": None,
             "observed_day": day(e["t"]), "arrived_day": day(e["t"]), "source": e["source"], "extractor_conf": None,
             "error_type": None, "text": None, "corrects": None, "duplicate_of": None,
             "label": lab if lab != P.CANT else placeholder_label(g, e["field"], e["value"], e["t"], ref),
             "_undecidable": lab == P.CANT}
        assertions.append(a)
        keys.append(hashlib.sha256(("%d|%d|%d|%s|%s|%s" % (SHUFFLE_SEED, m["mid"], e["t"], e["source"], e["field"],
                                                           e["value"])).encode("utf-8")).hexdigest())
    # day by day; inside a day a seeded order that depends only on the entry, so the _nobot file lists its entries
    # in the same relative order as the full file
    assertions = [assertions[i] for i in sorted(range(len(assertions)),
                                                key=lambda i: (assertions[i]["observed_day"], keys[i]))]
    case_id = ("parliament_nobot-%d" if split.endswith("_nobot") else "parliament-%d") % m["mid"]
    for i, a in enumerate(assertions):
        a["id"] = i
        if a.pop("_undecidable"):
            undecidable.append(i)
    return {"case_id": case_id, "split": split, "now_day": day(ref), "field_types": ftypes, "truth_values": truth,
            "household_other_pet": False, "knobs": {}, "conflict_types": [], "assertions": assertions,
            "queries": queries, "undecidable_ids": undecidable}


def fetch_labels(wd: List[Tuple[int, dict, Any, bool]]) -> None:
    """English labels of the positions, districts, groups and parties this item's statements point to."""
    need = set()
    for _, _, stmts, _ in wd:
        for s in stmts or []:
            if s["pid"] == "P39" and s["districts"]:
                need.add(s["value"])
                need.update(s["districts"])
                need.update(s["groups"])
            if s["pid"] == "P102":
                need.add(s["value"])
    need = sorted(x for x in need if isinstance(x, str) and x not in P.LABELS_CACHE)
    for k, v in P.wd_entities(need, "labels", {"languages": "en"}).items():
        P.LABELS_CACHE[k] = (((v or {}).get("labels") or {}).get("en") or {}).get("value", "")


def member_entries(m: dict) -> List[dict]:
    ref, g = m["ref"], m["gold"]
    wd = P.to_entries(m["mid"], "wikidata", m["wd"], ["party", "constituency"],
                      lambda stmts, t: P.wd_values(stmts, t, P.LABELS_CACHE), gaps=("none",))
    for e in wd:
        if e["automated"]:
            e["source"] = "wikidata_bot"
    wp = P.to_entries(m["mid"], "enwiki", m["wp"], P.FIELDS, lambda vals, t: vals, gaps=("none",))
    out = [e for e in wd + wp if e["t"] <= ref]
    for e in out:
        e["label"] = P.label_entry(g, e["field"], e["value"], e["t"], ref)
    return out


def summarize(name: str, cases: List[dict], members: Optional[List[dict]], dropped: Counter,
              sealed: bool = False) -> dict:
    """Counts for one written file. A sealed (phase-1) test file prints its size only: its labels stay unseen until
    phase 2."""
    labs: Counter = Counter()
    src: Counter = Counter()
    per_case = []
    for c in cases:
        und = set(c["undecidable_ids"])
        per_case.append(len(c["assertions"]))
        for a in c["assertions"]:
            labs[P.CANT if a["id"] in und else a["label"]] += 1
            src[a["source"]] += 1
    print("  %-26s %d cases (left out: %s)" % (name, len(cases), dict(dropped) or "none"))
    if not cases:
        return {}
    if sealed:
        print("    entries %d -- sealed: labels and sources are printed in phase 2 (INSTALL_TEST = True)"
              % sum(per_case))
        return {"cases": len(cases), "entries": sum(per_case), "dropped": dict(dropped), "sealed": True}
    q = Counter(qq["field"] for c in cases for qq in c["queries"])
    print("    entries %d (median %s per case, max %d) | by source %s" % (
        sum(per_case), statistics.median(per_case), max(per_case), dict(src)))
    print("    labels: valid %d, superseded %d, erroneous %d, left out (can't tell) %d" % (
        labs["valid"], labs["superseded"], labs["erroneous"], labs[P.CANT]))
    by_era = dict(Counter(P.era(P.day_of(m["start"])) for m in members)) if members else None
    print("    queries: %s%s" % (dict(q), (" | members by first year: %s" % by_era) if by_era else ""))
    return {"cases": len(cases), "entries": sum(per_case), "labels": dict(labs), "sources": dict(src),
            "queries": dict(q), "dropped": dict(dropped)}


# ============================================================ main
def build_members(group: List[dict], label: str, offset: int, total: int) -> None:
    """Downloads each member's histories and turns them into entries (in place). The histories are dropped once the
    entries exist: whole histories for hundreds of members would fill the memory."""
    ents = P.wd_entities([m["qid"] for m in group], "sitelinks", {"sitefilter": "enwiki"})
    for i, m in enumerate(group, 1):
        e = ents.get(m["qid"]) or {}
        m["qid_resolved"] = None if "missing" in e or not e else e.get("id", m["qid"])
        m["title"] = ((e.get("sitelinks") or {}).get("enwiki") or {}).get("title")
        m["gold"].fill_majority(0, m["ref"])           # every seat term up to the reference date
        print("  [%d/%d] %s %d %s" % (offset + i, total, label, m["mid"], m["name"]), flush=True)
        m["wd"] = wikidata_full(m["qid_resolved"], m["ref"]) if m["qid_resolved"] else []
        m["wp"] = enwiki_full(m["title"], m["ref"]) if m["title"] else []
        fetch_labels(m["wd"])
        m["entries"] = member_entries(m)
        del m["wd"], m["wp"]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_cases(path: str, cases: List[dict]) -> None:
    """Writes through a temporary file, so a script reading the file meanwhile never sees half of it."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    os.replace(tmp, path)


def write_split(split: str, group: List[dict], out_dir: str, sealed: bool) -> Dict[str, Any]:
    """Writes cases_<split>.jsonl (and the _nobot version) to out_dir; returns the summary for map_summary.json."""
    out: Dict[str, Any] = {}
    variants = [(split, lambda e: True)]
    if WRITE_NOBOT:
        variants.append((split + "_nobot", lambda e: e["source"] != "wikidata_bot"))
    for name, keep in variants:
        cases, dropped = [], Counter()
        for m in group:
            c = build_case(m, [e for e in m["entries"] if keep(e)], name)
            if c is None:
                dropped["no entry or no answerable field"] += 1
            else:
                cases.append(c)
        path = os.path.join(out_dir, "cases_%s.jsonl" % name)
        write_cases(path, cases)
        out[name] = summarize(name, cases, group, dropped, sealed=sealed)
        if sealed:
            out[name]["sha256"] = sha256_of(path)
            print("    SHA-256 %s" % out[name]["sha256"])
    return out


def test_files_here() -> List[str]:
    """Every file next to the scripts that belongs to the test split (cases, predictions, replies, batch state)."""
    return sorted(f for f in os.listdir(HERE) if "parliament_test" in f and os.path.isfile(os.path.join(HERE, f)))


def install_test() -> None:
    """Phase 2: copy the sealed test files next to the scripts (no rebuild), check them, print their summary."""
    print("recbench parliament_map -- phase 2: installing the sealed test files (no rebuild, no download)")
    summary: Dict[str, Any] = {}
    for name in ("parliament_test", "parliament_test_nobot") if WRITE_NOBOT else ("parliament_test",):
        src = os.path.join(CASES_DIR, "cases_%s.jsonl" % name)
        if not os.path.exists(src):
            raise SystemExit("No sealed file %s. Run phase 1 first (BUILD_TEST = True, INSTALL_TEST = False)."
                             % os.path.relpath(src, HERE))
        dst = os.path.join(HERE, "cases_%s.jsonl" % name)
        shutil.copyfile(src, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
        h_src, h_dst = sha256_of(src), sha256_of(dst)
        if h_src != h_dst:
            raise SystemExit("The copy of %s differs from the sealed file -- stop and tell the assistant." % name)
        with open(dst, encoding="utf-8") as f:
            cases = [json.loads(line) for line in f if line.strip()]
        summary[name] = summarize(name, cases, None, Counter())
        summary[name]["sha256"] = h_dst
        print("    SHA-256 %s (compare with the one listed in the \"v0.5: frozen\" commit)" % h_dst)
        if name == "parliament_test":
            print("    the LLM runs read the first %d of these cases" % min(LLM_CASES, len(cases)))
    with open(os.path.join(P.OUT, "map_summary_test.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("\ninstalled next to the scripts: %s" % ", ".join(test_files_here()))


def main() -> None:
    t_start = time.time()
    P.move_old_folders()                              # Oct 10 builds kept their files in realdata/
    os.makedirs(P.CACHE, exist_ok=True)
    os.makedirs(CASES_DIR, exist_ok=True)
    if INSTALL_TEST:
        install_test()
        return
    early = test_files_here()
    if early:
        raise SystemExit("Files of the sealed test split are next to the scripts while INSTALL_TEST = False: %s.\n"
                         "Before phase 2 no script may read the test split. If a preds_ / replies_ / batch_ file is "
                         "among them, a method has read it: stop and tell the assistant. If you are in phase 2, set "
                         "INSTALL_TEST = True." % ", ".join(early))
    run_t, frame_t0 = P.ptime(RUN_DATE), P.ptime(P.FRAME_START)
    n_all = N_DEV + (N_TEST if BUILD_TEST else 0)
    print("recbench parliament_map -- builds the Parliament splits, no method runs")
    print("frame %s .. %s; seed %d; dev %d; test %d (%s)" % (
        P.FRAME_START, RUN_DATE, P.SEED, N_DEV, N_TEST,
        ("built and sealed in " + os.path.relpath(CASES_DIR, HERE)) if BUILD_TEST else "not built this run"))
    frame, total = P.build_frame(RUN_DATE)
    print("frame: %d members (API says %s)" % (len(frame), total))
    cw, _first = P.load_crosswalk()
    summary: Dict[str, Any] = {"run_date": RUN_DATE, "seed": P.SEED}
    listing: Dict[str, Any] = {}

    # development split first: its files are next to the scripts before any test member is looked up
    skipped: Counter = Counter()
    dev = draw_members(frame, cw, run_t, frame_t0, N_DEV, skipped)
    summ_path = os.path.join(P.OUT, "probe_summary.json")
    if not os.path.exists(summ_path):
        raise SystemExit("%s not found: the development members cannot be checked against the probe's sample. Stop "
                         "and tell the assistant." % os.path.relpath(summ_path, HERE))
    with open(summ_path) as f:
        probe_sample = json.load(f).get("sample") or []
    if [m["mid"] for m in dev] != probe_sample:
        raise SystemExit("The development members differ from the probe's sample -- stop and tell the assistant.")
    print("dev members = the probe's %d (checked against %s); skipped on the way: %s" % (
        len(dev), os.path.relpath(summ_path, HERE), dict(skipped) or "none"))
    build_members(dev, "dev ", 0, n_all)
    print("\nDEVELOPMENT SPLIT (files next to the scripts)")
    summary.update(write_split("parliament_dev", dev, HERE, sealed=False))
    listing["parliament_dev"] = [member_row(m) for m in dev]
    print("  wrote cases_parliament_dev.jsonl%s (%.0f min so far)" % (
        " + cases_parliament_dev_nobot.jsonl" if WRITE_NOBOT else "", (time.time() - t_start) / 60.0), flush=True)

    if BUILD_TEST:
        print("\nTEST MEMBERS (the next %d in the same order)" % N_TEST, flush=True)
        skipped_t: Counter = Counter()
        members = draw_members(frame, cw, run_t, frame_t0, N_DEV + N_TEST, skipped_t)
        if [m["mid"] for m in members[:N_DEV]] != [m["mid"] for m in dev]:
            raise SystemExit("The continued draw does not start with the development members -- stop and tell the "
                             "assistant.")
        test = members[N_DEV:]
        print("test members drawn: %d; skipped on the way (dev part included): %s"
              % (len(test), dict(skipped_t) or "none"))
        build_members(test, "test", N_DEV, n_all)
        print("\nTEST SPLIT (sealed in %s)" % os.path.relpath(CASES_DIR, HERE))
        summary.update(write_split("parliament_test", test, CASES_DIR, sealed=True))
        listing["parliament_test"] = [member_row(m) for m in test]
        print("    the LLM runs will read the first %d of these cases" % min(LLM_CASES, len(test)))
        print("  put both SHA-256 lines in the \"v0.5: frozen\" commit message (phase 2 checks them)")
    with open(os.path.join(CASES_DIR, MEMBERS_FILE), "w", encoding="utf-8") as f:
        json.dump(listing, f, indent=1)
    with open(os.path.join(P.OUT, "map_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print("\ndone in %.0f min (downloads this run: %s); member list in %s" % (
        (time.time() - t_start) / 60.0, {k[10:]: v for k, v in P.STATS.items() if k.startswith("downloads_")} or 0,
        os.path.relpath(os.path.join(CASES_DIR, MEMBERS_FILE), HERE)))


def member_row(m: dict) -> dict:
    return {"member": m["mid"], "name": m["name"], "wikidata": m["qid_resolved"], "enwiki": m["title"],
            "reference_date": P.day_of(m["ref"]), "still_in_commons": m["current"]}


if __name__ == "__main__":
    main()
