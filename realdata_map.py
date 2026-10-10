# realdata_map.py -- map the public truth-discovery datasets into recbench cases (v0.5)
#
# v0.5 changes (pre-registered in PREDICTIONS.md before any method was re-run; they apply to every method):
#   - gate values with no digit ("Terminal") are placeholders, not gates -> null (they produced phantom ties)
#   - assertions are shuffled within each case with a fixed seed, as the generator does, so file order carries no
#     information (majority's tie-break was "first-listed source", which in Flight is the airline itself)
#   - Flight is written as two splits by day: cases_flight_dev.jsonl (days <= FLIGHT_DEV_LAST_DAY, the days
#     flight_diag.py read) and cases_flight_test.jsonl (later days, never inspected)
#   - "nogold" variants (audit, Oct 9): in Flight the gold standard is the data of the three airline websites, which
#     are also 3 of the 38 input sources; in Stock the gold comes from sites that are also inputs (Li et al., VLDB
#     2012). cases_flight_dev_nogold / cases_flight_test_nogold / cases_stock_nogold.jsonl are the same cases with
#     the gold-providing sources removed (GOLD_SOURCES below), so every method is also scored without an oracle vote.
#     The script prints which sources it removed; check them against the source list it prints.
#   - prints the largest record per dataset (the reconciler reads up to EVAL_MAX_ENTRIES entries per record)
#   - each numeric query carries its tolerance in words ("1% relative", "10 minutes"), as generated queries do, so an
#     LLM prompt states the right grading rule (v0.4 queries had none, so a prompt would have said "exact text")
#
# Datasets (Xin Luna Dong et al., https://www.lunadong.com/fusionDataSets.htm), download into DATA_DIR:
#   Stock : clean_stock.zip  + nasdaq_truth.zip (or pop_truth.zip)      55 sources, 1000 symbols, July 2011, 1 file/day
#   Flight: clean_flight.zip + flight_truth.zip                          38 sources, 1200+ flights, Dec 2011, 1 file/day
#   Book  : book.zip + book_golden.txt (and/or book_silver.txt)          894 sources, 1263 books, 2007
#
# Schemas as published (tab-separated):
#   stock  data : Source | Symbol | Change % | Last trading price | Open price | Change $ | Volume | Today's high |
#                 Today's low | Previous close | 52wk High | 52wk Low | Shares Outstanding | P/E | Market cap | Yield |
#                 Dividend | EPS
#   flight data : Source | Flight# | Scheduled departure | Actual departure | Departure gate | Scheduled arrival |
#                 Actual arrival | Arrival gate
#   book   data : Source | ISBN | Title | Author list              book gold: ISBN | author1; author2; ...
#
# FIRST RUN WITH INSPECT = True. It lists every file it found (zips are read in place, no unzipping needed), prints
# the first lines and column counts, and the role it guessed for each file. Check the guesses, fix anything in
# OVERRIDES, then set INSPECT = False and run again to write cases_stock.jsonl / cases_flight.jsonl / cases_book.jsonl.
# Every assertion is labeled valid/erroneous against the gold standard; there is no time dimension in these sets
# (observed_day = arrived_day = 0), so they test source-conflict resolution, not supersession.
# Python 3.9, stdlib only.

import glob
import io
import os
import random
import re
import zipfile
from typing import Any, Dict, List, Optional, Tuple

from recbench_common import match, norm_value, register_field_types, write_jsonl

# ============================================================ settings
DATA_DIR = "realdata"            # folder next to this script holding the downloaded zips / txt files
INSPECT = False                  # True the first time: list files + samples + guessed roles, write nothing.
OUT_FILES = {"stock": "cases_stock.jsonl", "flight": "cases_flight.jsonl", "book": "cases_book.jsonl"}
FLIGHT_DEV_LAST_DAY = "2011-12-15"   # Flight days <= this -> cases_flight_dev.jsonl; later -> cases_flight_test.jsonl
FLIGHT_FILES = {"dev": "cases_flight_dev.jsonl", "test": "cases_flight_test.jsonl"}
SHUFFLE_ASSERTIONS = True            # seeded per-case shuffle of assertion order (v0.5); False = file order (v0.4)
# gold-providing sources, matched on the source name lowercased with everything but letters and digits removed:
#   "exact" = the whole name, "prefix" = the start of the name. Flight: the three airline sites (named aa / ua / CO in
#   the Oct 2026 download). Stock: nasdaq.com supplies the nasdaq truth file used here (STOCK_TRUTH_PREFER).
GOLD_SOURCES = {"flight": ("exact", ["aa", "ua", "co"]), "stock": ("prefix", ["nasdaq"])}
WRITE_NOGOLD = True

STOCK_TRUTH_PREFER = "nasdaq"    # "nasdaq" (nasdaq.com values) or "pop" (majority of 5 providers)
STOCK_TOL_REL = 0.01             # two stock values are the same fact if within 1% (relative)
FLIGHT_TIME_TOL_MIN = 10         # two flight times are the same fact if within 10 minutes
BOOK_TRUTH_PREFER = "golden"     # "golden" (100 books, book covers) or "silver" (all books)
# Book is written twice: cases_book.jsonl (strict: the author list must equal the gold) and
# cases_book_subset.jsonl (a partial list that is a subset of the gold counts as evidence for the gold list).
BOOK_SUBSET_FILE = "cases_book_subset.jsonl"
MAX_CASES = {"stock": 400, "flight": 600, "book": None}     # cases per dataset (seeded random sample); None = all
SAMPLE_SEED = 7
# nasdaq truth reports Change % and Change $ without a sign ("1.59%" on a down day), so both are compared by magnitude
UNSIGNED_STOCK_ATTRS = {"change_pct", "change_usd"}

# role overrides: {"<file name as printed by INSPECT>": "stock_data" | "stock_truth" | "flight_data" |
#                  "flight_truth" | "book_data" | "book_truth" | "ignore"}
OVERRIDES: Dict[str, str] = {}

STOCK_ATTRS = ["change_pct", "last_price", "open_price", "change_usd", "volume", "high", "low", "prev_close",
               "wk52_high", "wk52_low", "shares_outstanding", "pe", "market_cap", "yield", "dividend", "eps"]
FLIGHT_ATTRS = ["sched_dep", "actual_dep", "dep_gate", "sched_arr", "actual_arr", "arr_gate"]
FLIGHT_TIME_ATTRS = {"sched_dep", "actual_dep", "sched_arr", "actual_arr"}

NULLS = {"", "-", "--", "n/a", "na", "null", "none", "nan", "unavailable", "not available", "?"}


# ============================================================ file discovery
def iter_files(data_dir: str):
    """Yield (display_name, container, text) for every text file under data_dir, reading zips in place."""
    for path in sorted(glob.glob(os.path.join(data_dir, "**", "*"), recursive=True)):
        if os.path.isdir(path):
            continue
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for member in sorted(z.namelist()):
                    if member.endswith("/") or os.path.basename(member).startswith("."):
                        continue
                    if "__MACOSX" in member:
                        continue
                    raw = z.read(member)
                    yield ("%s::%s" % (os.path.basename(path), member), os.path.basename(path), decode(raw))
        else:
            with open(path, "rb") as f:
                raw = f.read()
            yield (os.path.basename(path), os.path.basename(os.path.dirname(path)), decode(raw))


def decode(raw: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def split_line(line: str) -> List[str]:
    return [c.strip() for c in line.rstrip("\r\n").split("\t")]


def first_rows(text: str, n: int = 3) -> List[List[str]]:
    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(split_line(line))
            if len(rows) >= n:
                break
    return rows


def guess_role(name: str, container: str, rows: List[List[str]]) -> str:
    key = (name + " " + container).lower()
    ncol = len(rows[0]) if rows else 0
    truthy = any(w in key for w in ("truth", "golden", "silver", "gold"))
    if "book" in key:
        return "book_truth" if (truthy or ncol <= 2) else "book_data"
    if "flight" in key:
        return "flight_truth" if truthy else "flight_data"
    if "stock" in key or "nasdaq" in key or "pop_truth" in key:
        return "stock_truth" if truthy else "stock_data"
    # fall back on column counts
    if ncol in (17, 18):
        return "stock_data" if ncol == 18 else "stock_truth"
    if ncol in (7, 8):
        return "flight_data" if ncol == 8 else "flight_truth"
    if ncol in (2, 4):
        return "book_truth" if ncol == 2 else "book_data"
    return "ignore"


def date_of(name: str) -> Optional[str]:
    m = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", name)
    return "%s-%s-%s" % (m.group(1), m.group(2), m.group(3)) if m else None


# ============================================================ value parsing
def parse_number(s: str) -> Optional[float]:
    s = s.strip().replace(",", "").replace("$", "").replace("%", "")
    if s.lower() in NULLS:
        return None
    s = s.replace("+", "")
    mult = 1.0
    m = re.match(r"^(-?\d+(\.\d+)?)\s*([kKmMbBtT])$", s)
    if m:
        s = m.group(1)
        mult = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}[m.group(3).lower()]
    try:
        return float(s) * mult
    except ValueError:
        return None


def parse_time(s: str) -> Optional[float]:
    """'4:35 PM', '4:35p.m.', '16:35', '4:35pm (Dec 2)', 'Not available' -> minutes of day or None."""
    t = s.strip().lower()
    if t in NULLS or "not" in t and "avail" in t:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})\s*([ap])?\.?\s*m?\.?", t)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    ampm = m.group(3)
    if ampm == "p" and h < 12:
        h += 12
    if ampm == "a" and h == 12:
        h = 0
    if h > 23 or mi > 59:
        return None
    return float(h * 60 + mi)


def norm_gate(s: str) -> Optional[str]:
    t = re.sub(r"\s+", "", s.strip().upper())
    t = t.replace("GATE", "")
    if t.lower() in NULLS or not t:
        return None
    if not re.search(r"\d", t):          # v0.5: "Terminal", "TBD", ... are placeholders, not gates
        return None
    return t


def norm_isbn(s: str) -> str:
    """Normalize to ISBN-10 so the ISBN-13 entries in the gold file match the ISBN-10 data."""
    t = re.sub(r"[^0-9Xx]", "", s).upper()
    if len(t) == 13 and t.startswith("978"):
        core = t[3:12]
        total = sum((10 - i) * int(d) for i, d in enumerate(core))
        check = (11 - total % 11) % 11
        return core + ("X" if check == 10 else str(check))
    return t


AUTHOR_NOISE = {"unknown", "press", "publishing", "publications", "publisher", "inc", "ltd", "llc", "co",
                "editor", "editors", "ed", "eds", "staff", "various", "author", "authors", "et", "al", "jr", "sr",
                "ii", "iii", "iv", "phd", "md", "dr", "mr", "mrs", "ms", "prof", "certification", "corporation"}


def _surname(words: List[str]) -> Optional[str]:
    """Surname of one author given its words in 'First [M.] Last' order (noise words removed)."""
    toks = [re.sub(r"[^a-z]", "", w) for w in words]
    toks = [t for t in toks if len(t) >= 2 and t not in AUTHOR_NOISE]
    return toks[-1] if toks else None


def split_authors(s: str) -> List[Optional[str]]:
    """Author-list string -> list of surnames. Handles 'Last, First; Last, First', 'First Last; First Last',
    'First Last, First Last' (comma as separator) and 'Last, First, Last, First' (comma pairs)."""
    surnames: List[Optional[str]] = []
    for chunk in re.split(r"[;/&]|\band\b", s.lower()):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "," not in chunk:
            surnames.append(_surname(chunk.split()))
            continue
        pieces = [p.strip() for p in chunk.split(",") if p.strip()]
        if not pieces:
            continue
        nwords = [len(p.split()) for p in pieces]
        if all(n >= 2 for n in nwords):                  # 'first last, first last' -> commas separate authors
            for p in pieces:
                surnames.append(_surname(p.split()))
        elif nwords[0] == 1 and len(pieces) <= 2:        # 'last, first [middle]'
            surnames.append(_surname(pieces[:1]))
        elif all(n == 1 for n in nwords) and len(pieces) % 2 == 0:   # 'last, first, last, first'
            for i in range(0, len(pieces), 2):
                surnames.append(_surname(pieces[i:i + 1]))
        else:                                            # 'last, first, jr' and other mixes: first piece is the surname
            surnames.append(_surname(pieces[:1]))
            for p in pieces[1:]:
                if len(p.split()) >= 2:                  # a later 'first last' piece is another author
                    surnames.append(_surname(p.split()))
    return surnames


def norm_authors(s: str) -> Optional[str]:
    """Author list -> sorted multiset of surnames joined by ' | ' (order-free; initials and first names ignored).
    'Knuth, Donald E.' and 'Donald E. Knuth' -> 'knuth'; a partial list does not equal the full list."""
    if s.strip().lower() in NULLS:
        return None
    keys = [k for k in split_authors(s) if k]
    if not keys:
        return None
    return " | ".join(sorted(keys))


def authors_subset(claim: str, gold: str) -> bool:
    """True if the claimed surname multiset is a non-empty sub-multiset of the gold one."""
    from collections import Counter
    c, g = Counter(claim.split(" | ")), Counter(gold.split(" | "))
    return bool(c) and all(g[k] >= n for k, n in c.items())


# ============================================================ dataset readers
def read_stock(text: str, is_truth: bool) -> Dict[str, Dict[str, Tuple[str, Any]]]:
    """Returns {symbol: {attr: (source, value)}} for data (one entry per source via list) -- see collect()."""
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for line in text.splitlines():
        cols = split_line(line)
        if not any(cols):
            continue
        if is_truth:                       # symbol | 16 values (no source column)
            source, symbol, vals = "truth", cols[0], cols[1:17]
        else:                              # source | symbol | 16 values
            if len(cols) < 3:
                continue
            source, symbol, vals = cols[0], cols[1], cols[2:18]
        vals = list(vals) + [""] * (16 - len(vals))
        symbol = symbol.strip().upper()
        if not symbol or symbol.lower() == "symbol":
            continue
        attrs = {}
        for name, raw in zip(STOCK_ATTRS, vals):
            v = parse_number(raw)
            if v is not None:
                if name in UNSIGNED_STOCK_ATTRS:
                    v = abs(v)
                attrs["stock." + name] = v
        out.setdefault(symbol, []).append((source, attrs))
    return out


def read_flight(text: str, is_truth: bool) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for line in text.splitlines():
        cols = split_line(line)
        if not any(cols):
            continue
        if is_truth:                       # flight | 6 values
            source, flight, vals = "truth", cols[0], cols[1:7]
        else:                              # source | flight | 6 values
            if len(cols) < 3:
                continue
            source, flight, vals = cols[0], cols[1], cols[2:8]
        vals = list(vals) + [""] * (6 - len(vals))
        flight = re.sub(r"\s+", "", flight.strip().upper())
        if not flight or "flight" in flight.lower():
            continue
        attrs = {}
        for name, raw in zip(FLIGHT_ATTRS, vals):
            v = parse_time(raw) if name in FLIGHT_TIME_ATTRS else norm_gate(raw)
            if v is not None:
                attrs["flight." + name] = v
        out.setdefault(flight, []).append((source, attrs))
    return out


def read_book(text: str, is_truth: bool) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for line in text.splitlines():
        cols = split_line(line)
        if not any(cols):
            continue
        if is_truth and len(cols) >= 2 and len(cols) < 4:
            source, isbn, authors = "truth", cols[0], cols[-1]
        elif len(cols) >= 4:
            source, isbn, authors = cols[0], cols[1], cols[3]
        else:
            continue
        isbn = norm_isbn(isbn)
        if not isbn:
            continue
        a = norm_authors(authors)
        if a is None:
            continue
        out.setdefault(isbn, []).append((source, {"book.authors": a}))
    return out


FIELD_TYPES = {}
for _a in STOCK_ATTRS:
    FIELD_TYPES["stock." + _a] = ["num_rel", STOCK_TOL_REL]
for _a in FLIGHT_ATTRS:
    FIELD_TYPES["flight." + _a] = ["num_abs", FLIGHT_TIME_TOL_MIN] if _a in FLIGHT_TIME_ATTRS else ["cat", None]
FIELD_TYPES["book.authors"] = ["cat", None]


def tolerance_text(field: str) -> Optional[str]:
    """The grading tolerance in words, stored on each query as the generator does (v0.5; the LLM prompt and the Hub
    environment show it). None = exact match."""
    t, tol = FIELD_TYPES[field]
    if t == "num_rel":
        return "%g%% relative" % (tol * 100)
    if t == "num_abs":
        return "%g minutes (times are minutes of the day)" % tol if field.startswith("flight.") else "%g absolute" % tol
    return None


# ============================================================ case assembly
def make_cases(dataset: str, day: Optional[str], data: Dict[str, List[Tuple[str, Dict[str, Any]]]],
               truth: Dict[str, List[Tuple[str, Dict[str, Any]]]], subset_mode: bool = False) -> List[Dict[str, Any]]:
    cases = []
    for key, truth_rows in truth.items():
        claims = data.get(key)
        if not claims:
            continue
        truth_vals: Dict[str, Any] = {}
        for _src, attrs in truth_rows:
            truth_vals.update(attrs)
        if not truth_vals:
            continue
        split_name = dataset + ("_subset" if subset_mode else "")
        case: Dict[str, Any] = {
            "case_id": "%s-%s-%s" % (split_name, day or "all", key),
            "split": split_name, "now_day": 0,
            "field_types": {f: FIELD_TYPES[f] for f in truth_vals},
            "truth_values": truth_vals,
            "household_other_pet": False, "knobs": {}, "conflict_types": [],
        }
        register_field_types(case)
        assertions = []
        for source, attrs in claims:
            for field, value in attrs.items():
                if field not in truth_vals:
                    continue
                v = norm_value(field, value)
                t = norm_value(field, truth_vals[field])
                if subset_mode and field == "book.authors" and v != t and authors_subset(v, t):
                    value, v = truth_vals[field], t          # a partial list counts as evidence for the full list
                assertions.append({
                    "id": len(assertions), "field": field, "value": value, "unit": None,
                    "observed_day": 0, "arrived_day": 0, "source": source, "extractor_conf": None,
                    "error_type": None, "text": None, "corrects": None, "duplicate_of": None,
                    "label": "valid" if match(field, v, t) else "erroneous",
                })
        if not assertions:
            continue
        if SHUFFLE_ASSERTIONS:            # v0.5: file order is not information; the generator shuffles too
            random.Random("%d|%s" % (SAMPLE_SEED, case["case_id"])).shuffle(assertions)
            for i, a in enumerate(assertions):
                a["id"] = i
        case["assertions"] = assertions
        fields_with = set(a["field"] for a in assertions)
        case["queries"] = [dict({"field": f, "answer": truth_vals[f], "answer_type": FIELD_TYPES[f][0], "decidable": True},
                                **({"tolerance": tolerance_text(f)} if tolerance_text(f) else {}))
                           for f in truth_vals if f in fields_with]
        cases.append(case)
    return cases


# ============================================================ gold-source-removed variants (v0.5)
def _norm_source(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def is_gold_source(dataset: str, source: str) -> bool:
    mode, names = GOLD_SOURCES.get(dataset, ("exact", []))
    n = _norm_source(source)
    return (n in names) if mode == "exact" else any(n.startswith(x) for x in names)


def without_gold(cases: List[Dict[str, Any]], dataset: str, new_split: str) -> List[Dict[str, Any]]:
    """The same cases minus the gold-providing sources' assertions (ids renumbered in the shuffled order); a query
    whose field has no assertion left is dropped, and so is a case with none left."""
    out = []
    for c in cases:
        keep = [dict(a) for a in c["assertions"] if not is_gold_source(dataset, a["source"])]
        if not keep:
            continue
        for i, a in enumerate(keep):
            a["id"] = i
        nc = dict(c)
        nc["case_id"] = "%s-%s" % (new_split, c["case_id"].split("-", 1)[1])
        nc["split"] = new_split
        nc["assertions"] = keep
        fields = set(a["field"] for a in keep)
        nc["queries"] = [q for q in c["queries"] if q["field"] in fields]
        out.append(nc)
    return out


def report_gold(dataset: str, cases: List[Dict[str, Any]]) -> None:
    srcs = sorted(set(a["source"] for c in cases for a in c["assertions"]))
    gold = [x for x in srcs if is_gold_source(dataset, x)]
    print("  %s: %d sources: %s" % (dataset, len(srcs), ", ".join(srcs)))
    if gold:
        print("  %s: removed as gold-providing in the *_nogold files: %s" % (dataset, ", ".join(gold)))
    else:
        print("  %s: WARNING -- no source matched GOLD_SOURCES[%r]; the *_nogold files equal the originals. Fix "
              "GOLD_SOURCES from the list above." % (dataset, dataset))


# ============================================================ driver
def main() -> None:
    if not os.path.isdir(DATA_DIR):
        print("DATA_DIR '%s' not found. Create it next to this script and put the downloaded files in it." % DATA_DIR)
        return
    files = list(iter_files(DATA_DIR))
    roles = {}
    for name, container, text in files:
        rows = first_rows(text, 3)
        role = OVERRIDES.get(name) or guess_role(name, container, rows)
        roles[name] = role
        if INSPECT:
            print("\n%s  [%s]  lines=%d  cols=%s  date=%s  -> %s"
                  % (name, container, text.count("\n"), [len(r) for r in rows], date_of(name), role))
            for r in rows[:2]:
                print("   ", "\t".join(r)[:160])
    if INSPECT:
        print("\n%d files. Check the roles above; fix any with OVERRIDES, then set INSPECT = False." % len(files))
        return

    readers = {"stock": read_stock, "flight": read_flight, "book": read_book}
    for dataset in ("stock", "flight", "book"):
        data_by_day: Dict[Optional[str], Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {}
        truth_by_day: Dict[Optional[str], Dict[str, List[Tuple[str, Dict[str, Any]]]]] = {}
        for name, container, text in files:
            role = roles.get(name, "ignore")
            if not role.startswith(dataset):
                continue
            is_truth = role.endswith("truth")
            if dataset == "stock" and is_truth:
                k = (name + container).lower()
                if STOCK_TRUTH_PREFER == "nasdaq" and "pop" in k:
                    continue
                if STOCK_TRUTH_PREFER == "pop" and "pop" not in k:
                    continue
            if dataset == "book" and is_truth:
                k = name.lower()
                if BOOK_TRUTH_PREFER == "golden" and "silver" in k:
                    continue
                if BOOK_TRUTH_PREFER == "silver" and "golden" in k:
                    continue
            day = date_of(name) if dataset != "book" else None
            parsed = readers[dataset](text, is_truth)
            target = truth_by_day if is_truth else data_by_day
            bucket = target.setdefault(day, {})
            for key, rows in parsed.items():
                bucket.setdefault(key, []).extend(rows)
        cases: List[Dict[str, Any]] = []
        subset_cases: List[Dict[str, Any]] = []
        for day, truth in truth_by_day.items():
            data = data_by_day.get(day) or (data_by_day.get(None) if day is not None else None) or {}
            cases.extend(make_cases(dataset, day, data, truth))
            if dataset == "book":
                subset_cases.extend(make_cases(dataset, day, data, truth, subset_mode=True))
        n_total = len(cases)
        cap = MAX_CASES.get(dataset)
        if cap and len(cases) > cap:
            cases = random.Random(SAMPLE_SEED).sample(cases, cap)
            cases.sort(key=lambda c: c["case_id"])
        if not cases:
            print("%-6s no cases (no data/truth files matched; check roles with INSPECT = True)" % dataset)
            continue
        if dataset == "flight":           # v0.5: two splits by day; the test days were never inspected
            parts = {"dev": [], "test": []}
            for c in cases:
                day = c["case_id"].split("-", 1)[1][:10]
                parts["dev" if day <= FLIGHT_DEV_LAST_DAY else "test"].append(c)
            for part, sub in parts.items():
                for c in sub:
                    c["split"] = "flight_" + part
                    c["case_id"] = "flight_%s-%s" % (part, c["case_id"].split("-", 1)[1])
                write_jsonl(FLIGHT_FILES[part], sub)
                if WRITE_NOGOLD:
                    ng = without_gold(sub, "flight", "flight_%s_nogold" % part)
                    write_jsonl("cases_flight_%s_nogold.jsonl" % part, ng)
                    print("flight_%-4s nogold: %d cases, %d assertions -> cases_flight_%s_nogold.jsonl" % (
                        part, len(ng), sum(len(c["assertions"]) for c in ng), part))
                n_a = sum(len(c["assertions"]) for c in sub)
                n_e = sum(1 for c in sub for a in c["assertions"] if a["label"] == "erroneous")
                print("flight_%-4s %5d cases | %6d assertions | erroneous %.1f%% | days %s -> %s" % (
                    part, len(sub), n_a, 100.0 * n_e / max(1, n_a),
                    ("<= " if part == "dev" else "> ") + FLIGHT_DEV_LAST_DAY, FLIGHT_FILES[part]))
            report_gold("flight", cases)
            print("  flight: largest record %d assertions" % max(len(c["assertions"]) for c in cases))
            if os.path.exists(OUT_FILES["flight"]):
                print("note: %s is the v0.4 all-days file; delete it so it is not scored as a third Flight split" % OUT_FILES["flight"])
            continue
        write_jsonl(OUT_FILES[dataset], cases)
        if dataset == "stock" and WRITE_NOGOLD:
            report_gold("stock", cases)
            ng = without_gold(cases, "stock", "stock_nogold")
            write_jsonl("cases_stock_nogold.jsonl", ng)
            print("stock  nogold: %d cases, %d assertions -> cases_stock_nogold.jsonl" % (
                len(ng), sum(len(c["assertions"]) for c in ng)))
        if subset_cases:
            write_jsonl(BOOK_SUBSET_FILE, subset_cases)
            n_sub_err = sum(1 for c in subset_cases for a in c["assertions"] if a["label"] == "erroneous")
            n_sub = sum(len(c["assertions"]) for c in subset_cases)
            print("%-6s %5d cases, subset mode | erroneous %.1f%% -> %s" % ("book", len(subset_cases),
                  100.0 * n_sub_err / max(1, n_sub), BOOK_SUBSET_FILE))
        n_assert = sum(len(c["assertions"]) for c in cases)
        print("  %s: largest record %d assertions" % (dataset, max(len(c["assertions"]) for c in cases)))
        n_err = sum(1 for c in cases for a in c["assertions"] if a["label"] == "erroneous")
        n_src = len(set(a["source"] for c in cases for a in c["assertions"]))
        print("%-6s %5d cases (of %d) | %6d assertions (%.1f/case) | %d sources | erroneous %.1f%% | days %d -> %s"
              % (dataset, len(cases), n_total, n_assert, n_assert / len(cases), n_src,
                 100.0 * n_err / max(1, n_assert), len(truth_by_day), OUT_FILES[dataset]))
    print("now run baselines.py, then score.py (both pick up cases_stock/flight_dev/flight_test/book.jsonl automatically)")


if __name__ == "__main__":
    main()
