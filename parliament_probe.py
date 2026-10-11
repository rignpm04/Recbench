# parliament_probe.py -- can UK Parliament + Wikidata + Wikipedia make a real "superseded" test set? (recbench)
# Development tool, Oct 10, 2026. No recbench method runs on anything this script downloads.
#
# Run from PyCharm's Run button (Python 3.9, standard library only, needs internet). The first run takes roughly
# 30-90 minutes and prints progress. If it stops for any reason, press Run again: every download is cached, so it
# continues where it stopped; a re-run after a finished run is offline and prints the same numbers.
#
# What it does
#   1. Lists every House of Commons member active between FRAME_START and the run date (UK Parliament Members API).
#   2. Draws SAMPLE_SIZE of them at random (SEED). Each member's Wikidata item is found by its parliament.uk member ID
#      on Wikidata (property P10428), else through mySociety's id table; a member with neither is skipped and the next
#      one is drawn. It prints how the sample splits by first year in the Commons next to the whole frame, and warns
#      if they differ. These members are development data: they never go into a test set.
#   3. Answer key = Parliament's own dated records: party and seat histories (/Members/History, cross-checked with
#      /Members/{id}/Biography and /Members/{id}) and the election result that started each seat term (majority).
#   4. Sources = the member's Wikidata item history and English Wikipedia article history (lead section with the
#      infobox). Each source gives one entry per change of a field's value -- party, constituency, majority
#      (Wikipedia only) -- plus its value SNAPSHOT_DAYS after the member's window opens (the window opens at
#      FRAME_START or at the member's first Commons day after it, whichever is later).
#   5. Labels each entry with recbench's rule: wrong on the day it was stated -> erroneous; right then and still right
#      on the reference date -> valid; right then but changed since -> superseded. "Can't tell" when the answer key
#      has no value, or Parliament's endpoints disagree, at either date, or when moving the entry by
#      CANT_TELL_MARGIN_DAYS would change its label.
#   6. Prints counts and the go/no-go verdict. The thresholds below are fixed before the first run.
#
# Reference date per member: the run date if still in the Commons, else the member's last day there (the earliest
# end date among Parliament's three member endpoints).
#
# Writes parliament_data/probe/ (add parliament_data/ to .gitignore): cache/ (every download), run_date.txt,
# probe_entries.jsonl (every entry with its label), probe_summary.json. Delete that folder to start over. (Until
# Oct 10 this was realdata/parliament_probe/; it is moved automatically -- realdata/ is the public sets' folder, and
# realdata_map.py reads every file in it.)

import calendar
import gzip
import hashlib
import html
import http.client
import json
import os
import random
import re
import statistics
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

# ============================================================ settings
SEED = 20261010
SAMPLE_SIZE = 50
FRAME_START = "2015-05-07"           # 2015 general election
RUN_DATE = None                      # None = today (UTC) on the first run, then frozen in run_date.txt
SNAPSHOT_DAYS = 2                    # sources' first entry = their value this many days after the window opens
CANT_TELL_MARGIN_DAYS = 1            # Parliament's dates are days; source edits are to the second
OUT_DIR = os.path.join("parliament_data", "probe")
USER_AGENT = "recbench-parliament-probe/0.1 (https://github.com/rignpm04/Recbench)"   # Wikimedia requires a contact
PARLIAMENT_API = "https://members-api.parliament.uk/api"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
ENWIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_PARL_ID = "P10428"         # Wikidata property "parliament.uk member ID": finds a member's Wikidata item
# mySociety's member file, pinned to one commit: second way to find the Wikidata item (it lacks most members who
# entered in 2019 or later), and the frame's composition by first year for the representativeness check
PEOPLE_JSON_URL = ("https://raw.githubusercontent.com/mysociety/parlparse/"
                   "5b50bb67fcdffdf81e7c3d2501114ce83ebd53be/members/people.json")
PAUSE_S = {"parliament": 0.2, "wikimedia": 0.25, "github": 0.0}   # pause after each download
# Read requests go one at a time with a pause and no "maxlag": on Wikidata, maxlag also counts the query service's lag,
# which can stay high for hours and is meant to slow down bots that edit, not readers.
TIMEOUT_S = 90
MAX_RETRIES = 8

# Go / no-go thresholds (fixed before the first run)
GO_TWO_SOURCES_SHARE = 0.80      # party and constituency: share of members with entries from both sources
GO_SUPERSEDED_PER_MEMBER = 1.0   # superseded entries per sampled member
GO_MIN_ERRONEOUS = 20            # erroneous entries in total
GO_MAX_MACHINE_COPIED = 0.25     # share of all entries put in by a bot or tool AND citing Parliament's own data

# ============================================================ constants
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, OUT_DIR)
CACHE = os.path.join(OUT, "cache")
# where earlier versions kept their output, inside realdata/ (the public-set folder realdata_map.py reads in full)
OLD_FOLDERS = {os.path.join(HERE, "realdata", "parliament_probe"): OUT,
               os.path.join(HERE, "realdata", "parliament_cases"): os.path.join(HERE, "parliament_data", "cases")}


def move_old_folders() -> None:
    """Moves realdata/parliament_probe and realdata/parliament_cases (Oct 10 builds) to parliament_data/, once, so the
    download cache and the frozen run date are kept and realdata_map.py never reads them."""
    for old, new in OLD_FOLDERS.items():
        if not os.path.isdir(old):
            continue
        if os.path.exists(new):
            raise SystemExit("Both %s and %s exist. Move what is inside the first into the second (Finder), delete the "
                             "first, and press Run again." % (os.path.relpath(old, HERE), os.path.relpath(new, HERE)))
        os.makedirs(os.path.dirname(new), exist_ok=True)
        os.rename(old, new)
        print("moved %s -> %s (realdata/ is for the public sets only)" % (os.path.relpath(old, HERE),
                                                                        os.path.relpath(new, HERE)))
DAY = 86400
UNDEF = "<no value>"        # answer key has no value then (not in the Commons, or no election result found)
UNCLEAR = "<unclear>"       # Parliament's endpoints disagree
AMBIG = "<ambiguous>"       # a source shows two different current values at once (not counted as an entry)
CANT = "can't tell"
LABELS = ["valid", "superseded", "erroneous", CANT]
FIELDS = ["party", "constituency", "majority"]
SOURCES = ["wikidata", "enwiki"]
STATS: Counter = Counter()


# ============================================================ time helpers
def safe_timegm(y: int, mo: int, d: int, hh: int = 0, mi: int = 0, ss: int = 0) -> Optional[int]:
    try:
        mo = max(1, min(12, mo))
        last = calendar.monthrange(y, mo)[1]
        return calendar.timegm((y, mo, max(1, min(last, d)), hh, mi, ss, 0, 0, 0))
    except (ValueError, OverflowError, TypeError):
        return None


def ptime(s: Any) -> Optional[int]:
    """'2015-05-07T00:00:00', '2015-05-07T03:04:05Z' or '2015-05-07' -> seconds since 1970 (UTC)."""
    if not isinstance(s, str):
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2}))?", s.strip())
    if not m:
        return None
    return safe_timegm(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                       int(m.group(4) or 0), int(m.group(5) or 0), int(m.group(6) or 0))


def iso(t: int) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def day_of(t: Optional[int]) -> str:
    return "open" if t is None else time.strftime("%Y-%m-%d", time.gmtime(t))


# ============================================================ downloads (cached)
def _cache_file(url: str) -> str:
    return os.path.join(CACHE, hashlib.sha1(url.encode("utf-8")).hexdigest() + ".json.gz")


def _wait(headers: Any, attempt: int) -> float:
    try:
        v = headers.get("Retry-After") if headers is not None else None
        if v is not None:
            return min(300.0, max(1.0, float(v)))
    except (ValueError, TypeError):
        pass
    return min(120.0, 2.0 * (2 ** attempt))


def fetch(url: str, kind: str) -> Tuple[int, Optional[str]]:
    """GET -> (200, text) or (404, None). Retries network errors, 429 and 5xx; stops the run on anything else."""
    last: Any = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip",
                                                   "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                body = r.read()
                if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    body = gzip.decompress(body)
            STATS["downloads_" + kind] += 1
            return 200, body.decode("utf-8")
        except urllib.error.HTTPError as e:
            STATS["downloads_" + kind] += 1
            if e.code == 404 or (e.code == 400 and kind == "parliament"):
                return e.code, None
            if e.code in (429, 500, 502, 503, 504):
                w = _wait(e.headers, attempt)
                print("      HTTP %d from %s, waiting %.0f s" % (e.code, kind, w), flush=True)
                time.sleep(w)
                last = e
                continue
            raise SystemExit("\nHTTP %d for %s\nStopped. Nothing is lost: press Run again later." % (e.code, url))
        except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                raise SystemExit("\nHTTPS certificate check failed (%s).\nThis Python cannot check certificates. If it is"
                                 " a python.org install, run 'Install Certificates.command' in its Applications"
                                 " folder, then press Run again." % e)
            w = min(120.0, 2.0 * (2 ** attempt))
            print("      network problem (%s), retrying in %.0f s" % (e, w), flush=True)
            time.sleep(w)
            last = e
    raise SystemExit("\nGave up after %d tries: %s (%s)\nPress Run again later; finished downloads are kept."
                     % (MAX_RETRIES, url, last))


def get_json(base: str, params: Optional[Dict[str, Any]] = None, kind: str = "wikimedia") -> Any:
    url = base
    if params:
        url += "?" + urllib.parse.urlencode(sorted((k, str(v)) for k, v in params.items()))
    path = _cache_file(url)
    if os.path.exists(path):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    waits = 0
    while True:
        status, text = fetch(url, kind)
        try:
            data = None if text is None else json.loads(text)
        except ValueError:
            raise SystemExit("\nNot JSON from %s:\n%s\nPress Run again later." % (url, (text or "")[:300]))
        err = data.get("error") if isinstance(data, dict) else None
        if kind == "wikimedia" and err is not None:
            code = str(err.get("code") if isinstance(err, dict) else err)
            waits += 1
            STATS["wikimedia_error_retries"] += 1
            if code in ("maxlag", "ratelimited") and waits <= 30:
                time.sleep(5.0 if code == "maxlag" else 30.0)
                continue
            info = str(err.get("info", "") if isinstance(err, dict) else "").lower()
            transient = code.startswith("internal_api_error") or code in ("readonly", "timeout") or \
                "search" in code or "too busy" in info or "try again" in info
            if transient and waits <= 4:
                print("      Wikimedia API error %s, retrying in 20 s" % code, flush=True)
                time.sleep(20.0)
                continue
            # anything else would leave a hole in the data: stop instead of printing numbers built on it
            raise SystemExit("\nWikimedia API error %s for %s\n%s\nStopped. Finished downloads are kept: press Run"
                             " again later." % (code, url, str(err)[:300]))
        break
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)
    if PAUSE_S.get(kind):
        time.sleep(PAUSE_S[kind])
    return data


def mw_revisions(api: str, params: Dict[str, Any], label: str = "", follow: bool = True,
                 transform: Any = None) -> List[Any]:
    """Revisions of a prop=revisions query. follow=True follows 'continue' to the end; follow=False makes one request
    (used for 'the last revision before a date', where 'continue' would walk back through the whole history).
    transform(revision) is applied to each revision as it arrives, so page texts are not all held in memory."""
    base = dict(params, action="query", format="json", formatversion="2")
    cont: Dict[str, str] = {}
    out: List[Any] = []
    n = 0
    while True:
        p = dict(base)
        p.update(cont)
        data = get_json(api, p, "wikimedia")
        n += 1
        if not isinstance(data, dict):
            raise SystemExit("\nUnexpected reply from %s: %s\nPress Run again later." % (api, str(data)[:200]))
        for pg in (data.get("query") or {}).get("pages") or []:
            for r in pg.get("revisions") or []:
                out.append(transform(r) if transform else r)
        if label and n % 20 == 0:
            print("      %s: %d revisions so far" % (label, len(out)), flush=True)
        nxt = {k: str(v) for k, v in data["continue"].items()} if follow and "continue" in data else None
        if not nxt or nxt == cont:      # done (or the API repeated itself: stop rather than loop)
            break
        cont = nxt
    return out


def parl(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return get_json(PARLIAMENT_API + path, params, "parliament")


# ============================================================ text normalization
def ascii_fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


PARTY_RULES = [
    (r"\bspeaker\b", "speaker"),
    (r"social democratic (and )?labour|\bsdlp\b", "sdlp"),
    (r"independent group|change uk", "changeuk"),
    (r"uk independence|\bukip\b", "ukip"),
    (r"independent (labour|conservative|socialist|alliance|unionist)", "independent"),
    (r"scottish national|\bsnp\b", "snp"),
    (r"democratic unionist|\bdup\b", "dup"),
    (r"ulster unionist|\buup\b", "uup"),
    (r"traditional unionist|\btuv\b", "tuv"),
    (r"sinn fein", "sinnfein"),
    (r"\balliance\b", "alliance"),
    (r"plaid", "plaid"),
    (r"\bgreen", "green"),
    (r"reform uk|brexit party|^reform$", "reform"),
    (r"liberal democrat|lib dem", "libdem"),
    (r"\balba\b", "alba"),
    (r"workers party", "workers"),
    (r"\brespect\b", "respect"),
    (r"co ?operative|\blabour\b", "labour"),
    (r"conservative|\btory\b|\btories\b", "conservative"),
    (r"\bindependents?\b|non affiliated|no party|^none\b", "independent"),
    (r"\byour party\b", "yourparty"),
]


def norm_party(name: Optional[str], context: str = "") -> Optional[str]:
    """One key per party, the same for Parliament, Wikidata and Wikipedia. A suspended whip counts as Independent,
    as Parliament records it. Unknown names come back as 'other:<name>'."""
    c = ascii_fold(html.unescape(context or "")).lower()
    if "whip" in c and re.search(r"suspend|withdraw|remov|without|lost", c) and "restor" not in c:
        return "independent"
    t = ascii_fold(html.unescape(name or "")).lower()
    t = re.sub(r"\((uk|united kingdom|politician|political party)\)", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    if not t:
        return None
    for pat, key in PARTY_RULES:
        if re.search(pat, t):
            return key
    return "other:" + t


def norm_cons(s: Optional[str]) -> Optional[str]:
    """Constituency name -> key: no accents, case, punctuation, '(UK Parliament constituency)' or '&'."""
    if not s:
        return None
    if "[" in s or "{" in s or "<" in s:
        s = strip_markup(s)
    t = ascii_fold(html.unescape(s)).lower()
    t = re.sub(r"\([^)]*\)", " ", t).replace("&", " and ")
    t = re.sub(r"\buk parliament constituency\b|\bparliamentary constituency\b|\bconstituency\b", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t).strip()
    t = re.sub(r"\bkingston upon hull\b", "hull", t)      # Parliament: "Kingston upon Hull East"; pages: "Hull East"
    return t or None


# ============================================================ wikitext (Wikipedia infobox)
_REF_PAIR = re.compile(r"<ref\b(?:[^>/]|/(?!>))*>.*?</ref\s*>", re.S | re.I)
_REF_SELF = re.compile(r"<ref\b[^>]*/>", re.I)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_INNER_TPL = re.compile(r"\{\{([^{}]*)\}\}")
_KEEP_TPL = re.compile(r"^(nowrap|nobr|small|big|nobold|noitalic|longitem|nowrap begin|lang\b.*)$")
_LIST_TPL = re.compile(r"^(plainlist|plain list|flatlist|hlist|ubl|unbulleted list|bulleted list|"
                       r"br separated entries|collapsible list|ublist)$")


def clean_refs(s: str) -> str:
    return _REF_SELF.sub("", _REF_PAIR.sub("", _COMMENT.sub("", s)))


def split_top(s: str) -> List[str]:
    """Split on '|' outside {{...}} and [[...]]."""
    parts, buf, db, dl, i, n = [], [], 0, 0, 0, len(s)
    while i < n:
        two = s[i:i + 2]
        if two == "{{":
            db += 1
            buf.append(two)
            i += 2
            continue
        if two == "}}" and db > 0:
            db -= 1
            buf.append(two)
            i += 2
            continue
        if two == "[[":
            dl += 1
            buf.append(two)
            i += 2
            continue
        if two == "]]" and dl > 0:
            dl -= 1
            buf.append(two)
            i += 2
            continue
        if s[i] == "|" and db == 0 and dl == 0:
            parts.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(s[i])
        i += 1
    parts.append("".join(buf))
    return parts


def unwrap_templates(s: str) -> str:
    """Innermost templates first: keep the text of wrappers, lists and party templates, drop every other template."""
    def rep(m: "re.Match") -> str:
        name, _, rest = m.group(1).partition("|")
        n = re.sub(r"[\s_]+", " ", name.strip().lower())
        if _KEEP_TPL.match(n):
            return rest.split("|")[-1] if n.startswith("lang") else rest
        if _LIST_TPL.match(n):
            items = [x for x in split_top(rest) if not re.match(r"^\s*[a-z_]+\s*=", x)]
            return "\n".join(items)
        if n.startswith("party") or n == "abbr":
            return split_top(rest)[0]
        return ""
    for _ in range(12):
        s2 = _INNER_TPL.sub(rep, s)
        if s2 == s:
            break
        s = s2
    return s


def strip_markup(s: str) -> str:
    s = unwrap_templates(clean_refs(s))
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"\[\[(?:[^\[\]|]*\|)?([^\[\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[(?:https?:)?//[^\s\]]+\s*([^\]]*)\]", r"\1", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s).replace("\xa0", " ")
    s = re.sub(r"'{2,}", "", s)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


_INFOBOX = re.compile(r"\{\{\s*infobox[\s_]", re.I)


def infobox_params(text: str) -> Optional[Dict[str, str]]:
    text = clean_refs(text)
    m = _INFOBOX.search(text)
    if not m:
        return None
    i = j = m.start()
    depth, n = 0, len(text)
    while j < n - 1:
        two = text[j:j + 2]
        if two == "{{":
            depth += 1
            j += 2
            continue
        if two == "}}":
            depth -= 1
            j += 2
            if depth == 0:
                break
            continue
        j += 1
    body = text[i + 2:j - 2] if depth == 0 else text[i + 2:]
    params: Dict[str, str] = {}
    for p in split_top(body)[1:]:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        params[re.sub(r"[\s_]+", "_", k.strip().lower())] = v.strip()
    if depth != 0:
        params["__unclosed__"] = "1"      # the infobox never closes: its values cannot be trusted
    return params


def balanced(v: str) -> bool:
    """False when a value holds an unclosed [[ or {{ -- the edit broke the markup and the value swallowed the
    parameters after it (seen in the probe: 'Lies | primeminister1 = ...')."""
    v = clean_refs(v)
    return v.count("[[") == v.count("]]") and v.count("{{") == v.count("}}")


MONTHS = {m: i for i, m in enumerate(["january", "february", "march", "april", "may", "june", "july", "august",
                                      "september", "october", "november", "december"], 1)}


def wp_date(s: str) -> Optional[int]:
    if not s:
        return None
    m = re.search(r"\{\{\s*(?:start date(?: and age)?|dts|date)\s*\|\s*(?:df\s*=\s*\w+\s*\|\s*)?(\d{4})\s*\|\s*(\d{1,2})"
                  r"\s*(?:\|\s*(\d{1,2}))?", s, re.I)
    if m:
        return safe_timegm(int(m.group(1)), int(m.group(2)), int(m.group(3) or 1))
    t = strip_markup(s)
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", t)
    if m and m.group(2).lower() in MONTHS:
        return safe_timegm(int(m.group(3)), MONTHS[m.group(2).lower()], int(m.group(1)))
    m = re.search(r"([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})", t)
    if m and m.group(1).lower() in MONTHS:
        return safe_timegm(int(m.group(3)), MONTHS[m.group(1).lower()], int(m.group(2)))
    m = re.search(r"([A-Za-z]+)\s+(\d{4})", t)
    if m and m.group(1).lower() in MONTHS:
        return safe_timegm(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
    m = re.search(r"\b(\d{4})\b", t)
    return safe_timegm(int(m.group(1)), 1, 1) if m else None


def parse_majority(v: str) -> Optional[int]:
    s = re.sub(r"\([^)]*\)", " ", strip_markup(v))
    m = re.search(r"\d{1,3}(?:,\d{3})+|\d+", s)
    return int(m.group(0).replace(",", "")) if m else None


_MP_OFFICE = re.compile(r"member of parliament\b|^\s*mp\b", re.I)
_ANNOTATION = re.compile(r"^(?:[\s()\[\],.;:/\-\u2013\u2014]|\d|present|since|until|from|to|current|incumbent|and|"
                         r"onwards|c\.)*$", re.I)     # a line holding only dates / "present" / brackets


def wp_values(params: Dict[str, str]) -> Dict[str, Any]:
    """Party, current constituency and its majority from one infobox. 'none' = every MP seat listed has ended."""
    out: Dict[str, Any] = {"party": None, "constituency": None, "majority": None, "_raw": {}}
    if params.get("__unclosed__"):
        STATS["enwiki_revisions_broken_markup"] += 1
        return out
    raw = params.get("party")
    if raw is not None and not balanced(raw):
        STATS["enwiki_values_broken_markup"] += 1
        raw = None
    if raw is not None:
        pre = unwrap_templates(clean_refs(raw))
        cands: List[List[str]] = []          # [context text, party name]
        for seg in re.split(r"<br\s*/?>|\n|\*|\u2022|;", pre, flags=re.I):
            txt = strip_markup(seg)
            if not txt:
                continue
            links = re.findall(r"\[\[([^\[\]|]+)(?:\|([^\[\]]*))?\]\]", seg)
            if links:
                cands.append([txt, links[0][1] or links[0][0]])
            elif _ANNOTATION.match(txt) and cands:
                cands[-1][0] += " " + txt      # "(2019-present)" on its own line dates the party above it
            elif not _ANNOTATION.match(txt):
                cands.append([txt, re.sub(r"\([^)]*\d{4}[^)]*\)", " ", txt).strip()])
        if cands:
            pick = cands[0]
            if len(cands) > 1:
                STATS["enwiki_party_several_lines"] += 1
                cur = [c for c in cands if re.search(r"present|incumbent|since|current|\d{4}\s*[-\u2013\u2014]\s*\)?\s*$",
                                                     c[0], re.I)]
                if cur:
                    pick = cur[0]
            key = norm_party(pick[1], pick[0])
            if key:
                out["party"] = key
                out["_raw"]["party"] = pick[0][:80]
    entries = []
    broken = False
    for k, v in params.items():
        m = re.match(r"^(office|constituency_mp)(\d*)$", k)
        if not m:
            continue
        lines_txt = strip_markup(v)
        txt = lines_txt.replace("\n", " ").strip()
        if not txt:
            continue
        kind, idx = m.groups()
        if kind == "office" and not _MP_OFFICE.search(txt):
            continue
        if not balanced(v):
            broken = True
            continue
        if kind == "office":
            # "Member of Parliament / for Rayleigh and Wickford / Rayleigh (2001-2010)": the seat is the first line
            # after "for"; further lines name earlier seats
            mm = re.search(r"\bfor\b\s*(.*)$", lines_txt, re.I | re.S)
            rest = [x.strip() for x in (mm.group(1) if mm else "").split("\n") if x.strip()]
            if not rest:
                continue
            cons = rest[0]
        else:
            rest = [x.strip() for x in lines_txt.split("\n") if x.strip()]
            cons = rest[0]
        te_raw = clean_refs(params.get("term_end" + idx) or params.get("termend" + idx) or "").strip()
        te = strip_markup(te_raw) or te_raw          # {{end date|...}} strips to nothing but is still an end date
        ended = bool(te_raw) and not re.match(r"^(present|incumbent|current|-|\u2013|\u2014)?$", te.strip(), re.I)
        ts = wp_date(params.get("term_start" + idx) or params.get("termstart" + idx) or "")
        entries.append((cons, ended, ts, params.get("majority" + idx)))
    if broken:
        STATS["enwiki_values_broken_markup"] += 1
        entries = []                      # a seat value swallowed other parameters: no seat or majority this revision
    if entries:
        cur = [e for e in entries if not e[1]]
        if not cur:
            out["constituency"] = "none"
            out["_raw"]["constituency"] = "(every MP seat listed has an end date)"
        else:
            keys = {norm_cons(e[0]) for e in cur} - {None}
            pick_e = None
            if len(keys) == 1:
                pick_e = max(cur, key=lambda e: e[2] or 0)
            elif keys:
                dated = sorted((e[2], i) for i, e in enumerate(cur) if e[2] is not None)
                if len(dated) == len(cur) and (len(dated) == 1 or dated[-1][0] > dated[-2][0]):
                    pick_e = cur[dated[-1][1]]
                else:
                    out["constituency"] = AMBIG
            if pick_e is not None:
                out["constituency"] = norm_cons(pick_e[0])
                out["_raw"]["constituency"] = pick_e[0][:80]
                if pick_e[3]:
                    maj = parse_majority(pick_e[3])
                    if maj is not None:
                        out["majority"] = maj
                        out["_raw"]["majority"] = strip_markup(pick_e[3])[:40]
    return out


# ============================================================ Wikidata items
def wd_time(s: Any) -> Optional[int]:
    if not isinstance(s, str):
        return None
    m = re.match(r"^\+0*(\d{1,4})-(\d{2})-(\d{2})T", s)
    if not m:
        return None
    return safe_timegm(int(m.group(1)), int(m.group(2)) or 1, int(m.group(3)) or 1)


def snak_value(sn: Any) -> Any:
    if not isinstance(sn, dict) or sn.get("snaktype") != "value":
        return None
    dv = sn.get("datavalue") or {}
    v, typ = dv.get("value"), dv.get("type")
    if typ == "wikibase-entityid" and isinstance(v, dict):
        if v.get("id"):
            return v["id"]
        if v.get("numeric-id") is not None:
            return ("P%d" if v.get("entity-type") == "property" else "Q%d") % int(v["numeric-id"])
        return None
    if typ == "time" and isinstance(v, dict):
        return wd_time(v.get("time"))
    if typ == "string" and isinstance(v, str):
        return v
    return None


def legacy_claims(lst: list) -> Dict[str, list]:
    """Pre-2014 serialization ('m', 'q', 'refs'). The API re-serializes old revisions in the current format, so this
    is only a fallback."""
    def snak(s: Any) -> Optional[dict]:
        if not isinstance(s, list) or len(s) < 2 or not isinstance(s[1], int):
            return None
        if s[0] == "value" and len(s) == 4:
            return {"snaktype": "value", "property": "P%d" % s[1], "datavalue": {"type": s[2], "value": s[3]}}
        return {"snaktype": s[0], "property": "P%d" % s[1]}
    out: Dict[str, list] = {}
    for c in lst:
        if not isinstance(c, dict):
            continue
        ms = snak(c.get("m"))
        if not ms:
            continue
        qual: Dict[str, list] = {}
        for s in c.get("q") or []:
            sn = snak(s)
            if sn:
                qual.setdefault(sn["property"], []).append(sn)
        refs = []
        for rl in c.get("refs") or []:
            d: Dict[str, list] = {}
            for s in rl or []:
                sn = snak(s)
                if sn:
                    d.setdefault(sn["property"], []).append(sn)
            refs.append({"snaks": d})
        rank = {0: "deprecated", 1: "normal", 2: "preferred"}.get(c.get("rank", 1), "normal")
        out.setdefault(ms["property"], []).append({"mainsnak": ms, "qualifiers": qual, "references": refs,
                                                   "rank": rank})
    return out


def statement_info(st: Any) -> Optional[dict]:
    if not isinstance(st, dict):
        return None
    q = st.get("qualifiers") or {}
    if not isinstance(q, dict):
        q = {}

    def qv(pid: str) -> list:
        return [snak_value(s) for s in (q.get(pid) or [])]
    end_some = any(isinstance(s, dict) and s.get("snaktype") == "somevalue" for s in (q.get("P582") or []))
    starts = [v for v in qv("P580") if isinstance(v, int)]
    ends = [v for v in qv("P582") if isinstance(v, int)]
    ref_pids, ref_items, ref_urls, wiki = set(), set(), [], False
    for r in st.get("references") or []:
        snaks = (r or {}).get("snaks") or {}
        if not isinstance(snaks, dict):
            continue
        for pid, lst in snaks.items():
            ref_pids.add(pid)
            if pid == "P143":
                wiki = True
            for sn in lst or []:
                v = snak_value(sn)
                if pid == "P248" and isinstance(v, str):
                    ref_items.add(v)
                if pid == "P854" and isinstance(v, str):
                    ref_urls.append(v)
    return {"value": snak_value(st.get("mainsnak")), "rank": st.get("rank", "normal"),
            "start": min(starts) if starts else None, "end": min(ends) if ends else None, "end_some": end_some,
            "districts": [v for v in qv("P768") if isinstance(v, str)],
            "groups": [v for v in qv("P4100") if isinstance(v, str)],
            "ref_pids": sorted(ref_pids), "ref_items": sorted(ref_items), "ref_urls": ref_urls[:5],
            "wiki_import": wiki, "has_refs": bool(ref_pids)}


def wd_extract(content: Optional[str]) -> Optional[List[dict]]:
    """Position-held (P39) and party (P102) statements of one item revision; None if unreadable."""
    if content is None:
        return None
    try:
        d = json.loads(content)
    except ValueError:
        return None
    if not isinstance(d, dict):
        return None
    if "redirect" in d:
        return []
    claims = d.get("claims", d.get("statements"))
    if isinstance(claims, list) and claims:
        claims = legacy_claims(claims)
        STATS["wikidata_old_format_revisions"] += 1
    if not isinstance(claims, dict):
        claims = {}
    out = []
    for pid in ("P39", "P102"):
        for st in claims.get(pid) or []:
            x = statement_info(st)
            if x:
                x["pid"] = pid
                out.append(x)
    return out


_UKPARL = re.compile(r"parliament of the united kingdom", re.I)
_PARL_REF = re.compile(r"parliament\.uk|\buk parliament\b|parliament of the united kingdom|house of commons|"
                       r"members.? names|\bmnis\b|hansard", re.I)
_NOT_PARL_REF = re.compile(r"history of parliament", re.I)
_REF_PLAIN_PIDS = {"P248", "P854", "P143", "P813", "P1476", "P407", "P577", "P1683", "P2860"}


def wd_current(s: dict, t: int) -> bool:
    if s["rank"] == "deprecated" or s["end_some"]:
        return False
    if s["end"] is not None and s["end"] <= t:
        return False
    if s["start"] is not None and s["start"] > t + DAY:
        return False
    return True


def pick_one(cands: List[Tuple[str, dict]]) -> str:
    keys = {k for k, _ in cands}
    if len(keys) == 1:
        return next(iter(keys))
    pref = [(k, s) for k, s in cands if s["rank"] == "preferred"]
    if pref and len({k for k, _ in pref}) == 1:
        return pref[0][0]
    dated = sorted((s["start"], k) for k, s in cands if s["start"] is not None)
    if len(dated) == len(cands) and dated[-1][0] > dated[-2][0]:
        return dated[-1][1]
    return AMBIG


def is_parl_ref(s: dict, labels: Dict[str, str]) -> bool:
    texts = list(s["ref_urls"]) + [labels.get(q, "") for q in s["ref_items"]] + \
        [labels.get(p, "") for p in s["ref_pids"] if p not in _REF_PLAIN_PIDS]
    return any(x and _PARL_REF.search(x) and not _NOT_PARL_REF.search(x) for x in texts)


def wd_values(stmts: List[dict], t: int, labels: Dict[str, str]) -> Dict[str, Any]:
    """Current party and constituency of one item revision at time t."""
    out: Dict[str, Any] = {"party": None, "constituency": None, "_used": {"party": [], "constituency": []},
                           "_party_path": None, "_raw": {}}
    uk = [s for s in stmts if s["pid"] == "P39" and s["districts"] and s["value"]
          and _UKPARL.search(labels.get(s["value"], ""))]
    cur = [s for s in uk if wd_current(s, t)]
    cands = [(norm_cons(labels.get(d, d)), s, labels.get(d, d)) for s in cur for d in s["districts"]]
    cands = [c for c in cands if c[0]]
    if cands:
        out["constituency"] = pick_one([(k, s) for k, s, _ in cands])
        out["_used"]["constituency"] = [s for _, s, _ in cands]
        out["_raw"]["constituency"] = cands[0][2][:80]
    elif uk:
        out["constituency"] = "none"
        out["_raw"]["constituency"] = "(every Commons membership statement has ended)"
    groups = [(norm_party(labels.get(g, g)), s, labels.get(g, g)) for s in cur for g in s["groups"]]
    groups = [g for g in groups if g[0]]
    if groups:
        out["party"] = pick_one([(k, s) for k, s, _ in groups])
        out["_used"]["party"] = [s for _, s, _ in groups]
        out["_party_path"] = "parliamentary group (P4100)"
        out["_raw"]["party"] = groups[0][2][:80]
    else:
        pp = [(norm_party(labels.get(s["value"], s["value"])), s, labels.get(s["value"], s["value"]))
              for s in stmts if s["pid"] == "P102" and s["value"] and wd_current(s, t)]
        pp = [p for p in pp if p[0]]
        if pp:
            out["party"] = pick_one([(k, s) for k, s, _ in pp])
            out["_used"]["party"] = [s for _, s, _ in pp]
            out["_party_path"] = "member of political party (P102)"
            out["_raw"]["party"] = pp[0][2][:80]
    return out


_SKIP_ACTIONS = {
    "wbsetlabel-add", "wbsetlabel-set", "wbsetlabel-remove", "wbsetdescription-add", "wbsetdescription-set",
    "wbsetdescription-remove", "wbsetaliases-add", "wbsetaliases-set", "wbsetaliases-remove", "wbsetaliases-update",
    "wbsetaliases-add-remove", "wbsetlabeldescriptionaliases", "wbsetsitelink-add", "wbsetsitelink-add-both",
    "wbsetsitelink-set", "wbsetsitelink-set-badges", "wbsetsitelink-set-both", "wbsetsitelink-remove",
    "clientsitelink-update", "clientsitelink-remove", "wblinktitles-connect", "wbeditentity-update-languages",
    "wbeditentity-update-languages-short"}


def wb_action(comment: Optional[str]) -> str:
    m = re.match(r"^/\* ([A-Za-z0-9-]+)", comment or "")
    return m.group(1) if m else ""


_AUTO_COMMENT = re.compile(r"quickstatements|#temporary_batch|openrefine|harvesttemplates|petscan|mix.?n.?match|"
                           r"widar|wikidata.?todo|toolforge|toollabs|pywikibot|#\w*batch", re.I)


def automated(rev: dict) -> bool:
    u = rev.get("user") or ""
    if re.search(r"bot\b|^bot", u, re.I):          # ClueBot NG, AnomieBOT, Pi bot, KrBot, BotMultichill
        return True
    if _AUTO_COMMENT.search(rev.get("comment") or ""):
        return True
    return any(re.search(r"oauth|openrefine|quickstatements|bot", tg or "", re.I) for tg in rev.get("tags") or [])


# ============================================================ answer key (Parliament)
def commons(h: Any) -> bool:
    return h in (1, "1", "Commons", "commons")


class Gold:
    """Parliament's dated records for one member. Intervals are [start, end); end None = still open."""

    def __init__(self, h: dict, bio: dict, mem: dict):
        self.commons = []   # (start, end, constituency key, name, constituency id)
        for x in h.get("houseMembershipHistory") or []:
            s = ptime(x.get("membershipStartDate"))
            if commons(x.get("house")) and s is not None:
                self.commons.append((s, ptime(x.get("membershipEndDate")), norm_cons(x.get("membershipFrom")),
                                     x.get("membershipFrom"), x.get("membershipFromId")))
        self.commons.sort(key=lambda c: c[0])
        self.party = []     # (start, end, key, name)
        for x in h.get("partyHistory") or []:
            p = x.get("party") or {}
            s = ptime(x.get("startDate"))
            if s is not None and p.get("name"):
                self.party.append((s, ptime(x.get("endDate")), norm_party(p["name"]), p["name"]))
        self.party.sort(key=lambda p: p[0])
        self.bio_party = [(ptime(x.get("startDate")), ptime(x.get("endDate")), norm_party(x.get("name")), x.get("name"))
                          for x in bio.get("partyAffiliations") or []
                          if commons(x.get("house")) and ptime(x.get("startDate")) is not None]
        self.bio_cons = [(ptime(x.get("startDate")), ptime(x.get("endDate")), norm_cons(x.get("name")), x.get("name"))
                         for x in bio.get("representations") or []
                         if commons(x.get("house")) and ptime(x.get("startDate")) is not None]
        self.bio_house = [(ptime(x.get("startDate")), ptime(x.get("endDate")))
                          for x in bio.get("houseMemberships") or []
                          if commons(x.get("house")) and ptime(x.get("startDate")) is not None]
        self.latest = mem.get("latestHouseMembership") or {}
        self.majority: Dict[int, Any] = {}
        self.majority_source: Dict[int, str] = {}

    @staticmethod
    def _cover(ivs: list, t: int) -> list:
        return [iv for iv in ivs if iv[0] <= t and (iv[1] is None or t < iv[1])]

    def at(self, field: str, t: int) -> Any:
        cov = self._cover(self.commons, t)
        if not cov:
            return UNDEF
        checks = None
        if field == "constituency":
            vals = {c[2] for c in cov}
            checks = self.bio_cons
        elif field == "party":
            vals = {p[2] for p in self._cover(self.party, t)}
            checks = self.bio_party
            if not vals:
                return UNDEF
        else:
            vals = {self.majority.get(c[0], UNDEF) for c in cov}
        if len(vals) != 1:
            return UNCLEAR
        v = next(iter(vals))
        if v == UNDEF or v is None:
            return UNDEF
        if checks:
            other = {x[2] for x in self._cover(checks, t)}
            if other and other != {v}:
                return UNCLEAR
        return v

    def all_values(self, field: str) -> set:
        if field == "party":
            return {p[2] for p in self.party}
        if field == "constituency":
            return {c[2] for c in self.commons}
        return {v for v in self.majority.values() if isinstance(v, int)}

    def window(self, frame_t0: int, run_t: int) -> Optional[Tuple[int, int, Dict[str, Optional[int]], bool]]:
        ov = [c for c in self.commons if c[0] <= run_t and (c[1] is None or c[1] > frame_t0)]
        if not ov:
            return None
        start = max(frame_t0, min(c[0] for c in ov))
        last = max(ov, key=lambda c: c[0])
        ends: Dict[str, Optional[int]] = {"History": last[1]}
        if self.bio_house:
            ends["Biography"] = max(self.bio_house, key=lambda x: x[0])[1]
        if commons(self.latest.get("house")):
            ends["member record"] = ptime(self.latest.get("membershipEndDate"))
        known = [e for e in ends.values() if e is not None]
        ref = run_t if not known else min(run_t, min(known) - DAY)
        disagree = len({day_of(e) for e in ends.values()}) > 1
        return start, ref, ends, disagree

    def fill_majority(self, start: int, ref: int) -> None:
        for (s, e, _k, _n, cid) in self.commons:
            if s > ref or (e is not None and e <= start - 3 * DAY):
                continue
            if not isinstance(cid, int):
                self.majority[s] = UNDEF
                continue
            res = parl("/Location/Constituency/%d/ElectionResults" % cid)
            hits = [r for r in ((res or {}).get("value") or []) if isinstance(r, dict) and not r.get("isNotional")
                    and ptime(r.get("electionDate")) is not None and day_of(ptime(r.get("electionDate"))) == day_of(s)]
            maj = {r.get("majority") for r in hits if isinstance(r.get("majority"), int)}
            self.majority[s] = maj.pop() if len(maj) == 1 else UNDEF
            self.majority_source[s] = (hits[0].get("electionTitle") or "") if hits else ""

    def changes(self, field: str, start: int, ref: int) -> List[Tuple[int, Any, Any]]:
        """Real changes in the answer key inside (start, ref]: (time, old value, new value)."""
        if field == "party":
            seq = [(p[0], p[2]) for p in self.party if p[0] <= ref and (p[1] is None or p[1] > start - 400 * DAY)]
        elif field == "constituency":
            seq = [(c[0], c[2]) for c in self.commons if c[0] <= ref and (c[1] is None or c[1] > start - 400 * DAY)]
        else:
            seq = [(c[0], self.majority.get(c[0], UNDEF)) for c in self.commons
                   if c[0] <= ref and (c[1] is None or c[1] > start - 400 * DAY)]
        out = []
        for a, b in zip(seq, seq[1:]):
            if start < b[0] <= ref and a[1] != b[1] and UNDEF not in (a[1], b[1]) and None not in (a[1], b[1]):
                out.append((b[0], a[1], b[1]))
        return out


# ============================================================ labels
def label_entry(g: Gold, field: str, value: Any, t: int, ref: int) -> str:
    g_ref = g.at(field, ref)
    if g_ref in (UNDEF, UNCLEAR):
        return CANT
    labs = set()
    for dt in (-CANT_TELL_MARGIN_DAYS * DAY, 0, CANT_TELL_MARGIN_DAYS * DAY):
        g_then = g.at(field, t + dt)
        if g_then in (UNDEF, UNCLEAR):
            return CANT
        labs.add("erroneous" if value != g_then else ("valid" if value == g_ref else "superseded"))
    return labs.pop() if len(labs) == 1 else CANT


# ============================================================ steps
def load_run_date() -> str:
    path = os.path.join(OUT, "run_date.txt")
    if RUN_DATE:
        return RUN_DATE
    if os.path.exists(path):
        with open(path) as f:
            return f.read().strip()
    d = time.strftime("%Y-%m-%d", time.gmtime())
    with open(path, "w") as f:
        f.write(d + "\n")
    return d


def build_frame(run_day: str) -> Tuple[List[int], Any]:
    ids, skip, total = [], 0, None
    while True:
        d = parl("/Members/Search", {"MembershipInDateRange.WasMemberOfHouse": 1,
                                     "MembershipInDateRange.WasMemberOnOrAfter": FRAME_START,
                                     "MembershipInDateRange.WasMemberOnOrBefore": run_day,
                                     "skip": skip, "take": 20})
        items = (d or {}).get("items") or []
        if isinstance((d or {}).get("totalResults"), int):
            total = d["totalResults"]
        for it in items:
            v = (it or {}).get("value") or {}
            if isinstance(v.get("id"), int):
                ids.append(v["id"])
        skip += 20
        if not items or (total is not None and skip >= total):
            break
    return sorted(set(ids)), total


def load_crosswalk() -> Tuple[Dict[str, str], Dict[str, str]]:
    """mySociety's file -> (Parliament id -> Wikidata id, Parliament id -> first Commons start on or after the frame
    start, 'YYYY-MM-DD')."""
    d = get_json(PEOPLE_JSON_URL, None, "github") or {}
    cw: Dict[str, str] = {}
    parl_of: Dict[str, str] = {}
    for per in d.get("persons") or []:
        ids: Dict[str, str] = {}
        for i in per.get("identifiers") or []:
            ids.setdefault(i.get("scheme"), i.get("identifier"))
        if ids.get("datadotparl_id"):
            parl_of[per.get("id")] = str(ids["datadotparl_id"])
            if ids.get("wikidata"):
                cw[str(ids["datadotparl_id"])] = str(ids["wikidata"])
    commons_posts = {x.get("id") for x in d.get("posts") or [] if x.get("organization_id") == "house-of-commons"}
    first: Dict[str, str] = {}
    for m in d.get("memberships") or []:
        pid = parl_of.get(m.get("person_id"))
        if not pid or m.get("post_id") not in commons_posts:
            continue
        s, e = m.get("start_date") or "", m.get("end_date") or "9999-12-31"
        if e >= FRAME_START:
            first[pid] = min(first.get(pid, "9999-12-31"), max(s, FRAME_START))
    return cw, first


def wd_by_parliament_id(mid: int) -> List[str]:
    """Wikidata items carrying this parliament.uk member ID (search index)."""
    d = get_json(WIKIDATA_API, {"action": "query", "list": "search", "srnamespace": "0", "srlimit": "5",
                                "srsearch": "haswbstatement:%s=%d" % (WIKIDATA_PARL_ID, mid), "format": "json",
                                "formatversion": "2"}, "wikimedia")
    return [h.get("title") for h in ((d or {}).get("query") or {}).get("search") or []
            if re.match(r"^Q\d+$", str(h.get("title")))]


def era(day: str) -> str:
    y = int(day[:4])
    return "2015" if y <= 2015 else ("2016-2018" if y <= 2018 else "2019 or later")


def member_records(mid: int) -> Tuple[Optional[dict], dict, dict]:
    hist = parl("/Members/History", {"ids": mid})
    h = None
    for el in hist if isinstance(hist, list) else []:
        v = (el or {}).get("value") or {}
        if v.get("id") == mid:
            h = v
    bio = (parl("/Members/%d/Biography" % mid) or {}).get("value") or {}
    mem = (parl("/Members/%d" % mid) or {}).get("value") or {}
    return h, bio, mem


def wd_entities(ids: List[str], props: str, extra: Dict[str, str]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    ids = sorted(set(ids))
    for i in range(0, len(ids), 50):
        p = dict({"action": "wbgetentities", "ids": "|".join(ids[i:i + 50]), "props": props, "format": "json"},
                 **extra)
        d = get_json(WIKIDATA_API, p, "wikimedia")
        for k, v in ((d or {}).get("entities") or {}).items():
            out[k] = v
    return out


def slot_text(r: dict) -> Optional[str]:
    slot = (r.get("slots") or {}).get("main") or {}
    if slot.get("texthidden") or slot.get("textmissing") or slot.get("nosuchsection"):
        return None
    c = slot.get("content")
    return c if isinstance(c, str) else None


def rev_meta(r: dict) -> dict:
    return {k: r.get(k) for k in ("revid", "timestamp", "user", "comment", "tags")}


def wikidata_history(qid: str, snap: int, ref: int) -> Tuple[List[Tuple[int, dict, Any, bool]], dict]:
    """[(time, revision meta, statements)] for the item at `snap` and every claim-changing revision in (snap, ref]."""
    info = {"listed": 0, "read": 0, "skipped": 0, "unreadable": 0}
    base = mw_revisions(WIKIDATA_API, {"prop": "revisions", "titles": qid, "rvprop": "ids|timestamp|user|comment|tags",
                                       "rvlimit": "1", "rvdir": "older", "rvstart": iso(snap)}, follow=False)
    meta = mw_revisions(WIKIDATA_API, {"prop": "revisions", "titles": qid, "rvprop": "ids|timestamp|user|comment|tags",
                                       "rvlimit": "max", "rvdir": "newer", "rvstart": iso(snap + 1), "rvend": iso(ref)},
                        "Wikidata " + qid)
    info["listed"] = len(meta) + len(base[:1])
    keep = [r for r in meta if wb_action(r.get("comment")) not in _SKIP_ACTIONS]
    info["skipped"] = len(meta) - len(keep)
    todo = [(snap, r, True) for r in base[:1]] + [(ptime(r.get("timestamp")) or snap, r, False) for r in keep]
    ids = [r["revid"] for _, r, _ in todo if isinstance(r.get("revid"), int)]
    stmts: Dict[int, Any] = {}
    for i in range(0, len(ids), 50):
        for rid, s in mw_revisions(WIKIDATA_API, {"prop": "revisions", "revids": "|".join(str(x) for x in ids[i:i + 50]),
                                                  "rvprop": "ids|content", "rvslots": "main"},
                                   transform=lambda r: (r.get("revid"), wd_extract(slot_text(r)))):
            stmts[rid] = s
    out = []
    for t, r, is_base in todo:
        s = stmts.get(r.get("revid"))
        info["read"] += 1
        if s is None:
            info["unreadable"] += 1
        out.append((t, r, s, is_base))
    return out, info


ENWIKI_NO_SEAT: set = set()     # (title, office text) seen in infoboxes where no Commons seat was found


def enwiki_history(title: str, snap: int, ref: int) -> Tuple[List[Tuple[int, dict, Any, bool]], dict]:
    info = {"listed": 0, "no_infobox": 0, "unreadable": 0, "no_seat": 0, "no_party": 0}

    def parse(r: dict) -> Tuple[dict, Optional[dict]]:
        text = slot_text(r)
        if text is None:
            return rev_meta(r), None
        params = infobox_params(text)
        if params is None:
            return rev_meta(r), {"party": None, "constituency": None, "majority": None, "_raw": {}, "_noinfobox": True}
        vals = wp_values(params)
        if vals["constituency"] is None:
            for k, v in params.items():
                if re.match(r"^(office|constituency)", k):
                    ENWIKI_NO_SEAT.add((title, strip_markup(v).replace("\n", " ")[:60]))
        return rev_meta(r), vals

    common = {"prop": "revisions", "titles": title, "redirects": "1", "rvslots": "main", "rvsection": "0",
              "rvprop": "ids|timestamp|user|comment|tags|content"}
    base = mw_revisions(ENWIKI_API, dict(common, rvlimit="1", rvdir="older", rvstart=iso(snap)), follow=False,
                        transform=parse)
    revs = mw_revisions(ENWIKI_API, dict(common, rvlimit="50", rvdir="newer", rvstart=iso(snap + 1), rvend=iso(ref)),
                        "Wikipedia " + title, transform=parse)
    out = []
    for t, (meta, vals), is_base in [(snap, x, True) for x in base[:1]] + \
            [(ptime(x[0].get("timestamp")) or snap, x, False) for x in revs]:
        info["listed"] += 1
        if vals is None:
            info["unreadable"] += 1
        elif vals.get("_noinfobox"):
            info["no_infobox"] += 1
        else:
            info["no_seat"] += vals["constituency"] is None
            info["no_party"] += vals["party"] is None
        out.append((t, meta, vals, is_base))
    return out, info


def to_entries(mid: int, source: str, timeline: List[Tuple[int, dict, Any, bool]], fields: List[str],
               values_at: Any, gaps: Tuple[Any, ...] = ()) -> List[dict]:
    """One entry per change of a field's value (the first value counts as a change). Values in `gaps` (and None,
    AMBIG) are gaps, not values: after a gap the next value counts only if it differs from the last value."""
    prev: Dict[str, Any] = {f: None for f in fields}
    out = []
    for t, rev, payload, is_snapshot in timeline:
        if payload is None:
            continue
        # is_snapshot: the page as it stood at the snapshot date. Whoever edited it last did not necessarily set
        # these values, so "made by a bot/tool" is unknown (None) for those entries.
        vals = values_at(payload, t)
        for f in fields:
            v = vals.get(f)
            if v == AMBIG:
                STATS["ambiguous_%s_%s" % (source, f)] += 1
            if v is None or v == AMBIG or v in gaps:   # a gap, not a value: the next value counts only if it differs
                continue
            if v != prev[f]:
                e = {"member": mid, "source": source, "field": f, "value": v, "raw": vals["_raw"].get(f),
                     "time": iso(t), "t": t, "revid": rev.get("revid"), "user": rev.get("user"),
                     "snapshot": is_snapshot, "automated": None if is_snapshot else automated(rev)}
                if source == "wikidata":
                    used = vals["_used"].get(f) or []
                    e["cites_parliament"] = any(is_parl_ref(s, LABELS_CACHE) for s in used)
                    e["cites_wikipedia"] = any(s["wiki_import"] for s in used)
                    e["cites_anything"] = any(s.get("has_refs") for s in used)
                    if f == "party":
                        e["path"] = vals.get("_party_path")
                out.append(e)
            prev[f] = v
    return out


LABELS_CACHE: Dict[str, str] = {}


def pct(a: float, b: float) -> str:
    return "%.0f%%" % (100.0 * a / b) if b else "n/a"


# ============================================================ main
def main() -> None:
    move_old_folders()
    os.makedirs(CACHE, exist_ok=True)
    t_start = time.time()
    run_day = load_run_date()
    run_t = ptime(run_day)
    frame_t0 = ptime(FRAME_START)
    print("recbench parliament_probe -- development check, no method runs")
    print("run date %s (frozen in %s); seed %d; sample %d" % (run_day, os.path.join(OUT_DIR, "run_date.txt"), SEED,
                                                               SAMPLE_SIZE))

    print("\n[1/5] Members active in the Commons %s .. %s" % (FRAME_START, run_day), flush=True)
    frame, total = build_frame(run_day)
    print("      %d members (API says %s)" % (len(frame), total))
    if not frame:
        raise SystemExit("No members found -- check the Members API.")

    print("\n[2/5] mySociety's member file (pinned commit)", flush=True)
    cw, first = load_crosswalk()
    print("      Wikidata ids for %d of the %d members (the rest are looked up on Wikidata by %s)" %
          (sum(1 for m in frame if str(m) in cw), len(frame), WIKIDATA_PARL_ID))

    print("\n[3/5] Drawing the sample and downloading Parliament's records", flush=True)
    order = list(frame)
    random.Random(SEED).shuffle(order)
    skipped: Counter = Counter()
    sample: List[dict] = []
    for mid in order:
        if len(sample) >= SAMPLE_SIZE:
            break
        hits = wd_by_parliament_id(mid)
        mys = cw.get(str(mid))
        if len(hits) == 1:
            qid, route = hits[0], "Wikidata " + WIKIDATA_PARL_ID
            if mys and mys != qid:
                STATS["wikidata_id_routes_disagree"] += 1
        elif mys:
            qid, route = mys, "mySociety"
        else:
            skipped["no Wikidata item found" if not hits else "several Wikidata items carry the id"] += 1
            continue
        h, bio, mem = member_records(mid)
        if not h:
            skipped["no record from /Members/History"] += 1
            continue
        g = Gold(h, bio, mem)
        win = g.window(frame_t0, run_t)
        if win is None:
            skipped["no Commons seat in the frame (History)"] += 1
            continue
        start, ref, ends, disagree = win
        snap = start + SNAPSHOT_DAYS * DAY
        if ref <= snap + DAY:
            skipped["in the Commons under %d days in the frame" % (SNAPSHOT_DAYS + 1)] += 1
            continue
        g.fill_majority(start, ref)
        sample.append({"mid": mid, "qid": qid, "route": route,
                       "name": mem.get("nameDisplayAs") or h.get("nameDisplayAs") or "",
                       "gold": g, "start": start, "snap": snap, "ref": ref, "ends": ends, "end_disagree": disagree,
                       "current": ref == run_t})
    print("      %d members drawn; skipped: %s" % (len(sample), dict(skipped) or "none"))
    if not sample:
        raise SystemExit("No member could be drawn -- check the downloads above (Members API, Wikidata, mySociety).")
    routes = Counter(m["route"] for m in sample)
    print("      Wikidata item found by: %s%s" % (dict(routes), "; the two ways disagree for %d" %
                                                  STATS["wikidata_id_routes_disagree"]
                                                  if STATS.get("wikidata_id_routes_disagree") else ""))
    s_era = Counter(era(day_of(m["start"])) for m in sample)
    f_era = Counter(era(first[str(x)]) for x in frame if str(x) in first)
    keys = ["2015", "2016-2018", "2019 or later"]
    print("      first year in the Commons since %s -- sample: %s | frame: %s" % (
        FRAME_START, ", ".join("%s %d" % (k, s_era[k]) for k in keys),
        ", ".join("%s %d" % (k, f_era[k]) for k in keys) if f_era else "unknown"))
    rep_gap = None
    if f_era:
        rep_gap = abs(s_era["2019 or later"] / len(sample) - f_era["2019 or later"] / sum(f_era.values()))
        if rep_gap > 0.15:
            print("      WARNING: members who entered in 2019 or later are %.0f points off their share of the frame;"
                  " the verdict below does not describe the frame." % (100 * rep_gap))

    print("\n[4/5] Wikidata and Wikipedia histories", flush=True)
    ents = wd_entities([m["qid"] for m in sample], "sitelinks", {"sitefilter": "enwiki"})
    for m in sample:
        e = ents.get(m["qid"]) or {}
        m["qid_resolved"] = None if "missing" in e or not e else e.get("id", m["qid"])
        m["title"] = ((e.get("sitelinks") or {}).get("enwiki") or {}).get("title")
    for i, m in enumerate(sample, 1):
        print("  [%d/%d] %d %s: %s .. %s%s" % (i, len(sample), m["mid"], m["name"], day_of(m["snap"]), day_of(m["ref"]),
                                               "" if m["current"] else " (left)"), flush=True)
        m["wd"], m["wd_info"] = ([], {}) if not m["qid_resolved"] else wikidata_history(m["qid_resolved"], m["snap"],
                                                                                         m["ref"])
        m["wp"], m["wp_info"] = ([], {}) if not m["title"] else enwiki_history(m["title"], m["snap"], m["ref"])
        print("      Wikidata %s: %d revisions in the window, %d read; Wikipedia: %s" % (
            m["qid_resolved"] or "-", m["wd_info"].get("listed", 0), m["wd_info"].get("read", 0),
            ("%d revisions" % m["wp_info"].get("listed", 0)) if m["title"] else "no English article"), flush=True)

    # labels of everything the Wikidata statements point to (P39 positions, districts, groups, parties, references)
    need = set()
    for m in sample:
        for _, _, stmts, _ in m["wd"]:
            for s in stmts or []:
                if s["pid"] == "P39" and s["districts"]:
                    need.add(s["value"])
                    need.update(s["districts"])
                    need.update(s["groups"])
                if s["pid"] == "P102":
                    need.add(s["value"])
                if (s["pid"] == "P39" and s["districts"]) or s["pid"] == "P102":
                    need.update(s["ref_items"])
                    need.update(p for p in s["ref_pids"] if p not in _REF_PLAIN_PIDS)
    need.discard(None)
    lab = wd_entities(sorted(x for x in need if isinstance(x, str)), "labels", {"languages": "en"})
    for k, v in lab.items():
        LABELS_CACHE[k] = (((v or {}).get("labels") or {}).get("en") or {}).get("value", "")

    print("\n[5/5] Entries and labels", flush=True)
    entries: List[dict] = []
    wd_unmatched_p39: Counter = Counter()
    for m in sample:
        g = m["gold"]
        wd_entries = to_entries(m["mid"], "wikidata", m["wd"], ["party", "constituency"],
                                lambda stmts, t: wd_values(stmts, t, LABELS_CACHE))
        wp_entries = to_entries(m["mid"], "enwiki", m["wp"], FIELDS, lambda vals, t: vals)
        for e in wd_entries + wp_entries:
            e["label"] = label_entry(g, e["field"], e["value"], e["t"], m["ref"])
            e["in_answer_key"] = e["value"] in g.all_values(e["field"])
        entries.extend(wd_entries + wp_entries)
        for _, _, stmts, _ in m["wd"][-1:]:
            for s in stmts or []:
                if s["pid"] == "P39" and s["districts"] and not _UKPARL.search(LABELS_CACHE.get(s["value"], "")):
                    wd_unmatched_p39[LABELS_CACHE.get(s["value"]) or s["value"]] += 1

    # ------------------------------------------------------------ report
    n = len(sample)
    print("\n" + "=" * 100)
    print("ANSWER KEY (Parliament)")
    cur = sum(1 for m in sample if m["current"])
    print("  members: %d (still in the Commons %d, left %d)" % (n, cur, n - cur))
    chg = {f: [] for f in FIELDS}
    for m in sample:
        for f in FIELDS:
            chg[f].extend((m, c) for c in m["gold"].changes(f, m["snap"], m["ref"]))
    for f in FIELDS:
        print("  real changes inside the windows, %-12s %3d (members with one: %d)" %
              (f + ":", len(chg[f]), len({m["mid"] for m, _ in chg[f]})))
    no_maj = sum(1 for m in sample for s, v in m["gold"].majority.items() if v == UNDEF)
    print("  seat terms with no election result found: %d" % no_maj)
    dis = [m for m in sample if m["end_disagree"]]
    print("  members whose endpoints give different end dates: %d" % len(dis))
    for m in dis:
        print("    %d %s: %s" % (m["mid"], m["name"], ", ".join("%s %s" % (k, day_of(v)) for k, v in m["ends"].items())))
    unclear = Counter()
    for m in sample:
        for f in ("party", "constituency"):
            if m["gold"].at(f, m["ref"]) == UNCLEAR:
                unclear[f] += 1
    print("  members whose endpoints disagree on the value at the reference date: %s" % (dict(unclear) or "none"))

    print("\nSOURCES")
    print("  members with a Wikidata item: %d; with an English Wikipedia article: %d" %
          (sum(1 for m in sample if m["qid_resolved"]), sum(1 for m in sample if m["title"])))
    wdl = sum(m["wd_info"].get("listed", 0) for m in sample)
    wdr = sum(m["wd_info"].get("read", 0) for m in sample)
    wds = sum(m["wd_info"].get("skipped", 0) for m in sample)
    wdu = sum(m["wd_info"].get("unreadable", 0) for m in sample)
    wpl = sum(m["wp_info"].get("listed", 0) for m in sample)
    wpn = sum(m["wp_info"].get("no_infobox", 0) for m in sample)
    wpu = sum(m["wp_info"].get("unreadable", 0) for m in sample)
    wps = sum(m["wp_info"].get("no_seat", 0) for m in sample)
    wpp = sum(m["wp_info"].get("no_party", 0) for m in sample)
    print("  Wikidata revisions: %d (read %d, skipped as label/description/sitelink-only %d, unreadable %d)" %
          (wdl, wdr, wds, wdu))
    print("  Wikipedia revisions: %d (no infobox in the lead %d, unreadable %d); of those with an infobox, no Commons"
          " seat found %d, no party found %d" % (wpl, wpn, wpu, wps, wpp))
    per_member = Counter(e["member"] for e in entries)
    counts = [per_member.get(m["mid"], 0) for m in sample]
    print("  entries per member: median %s, min %d, max %d" % (statistics.median(counts) if counts else 0,
                                                              min(counts or [0]), max(counts or [0])))
    print("  %-10s %-13s %8s %16s" % ("source", "field", "entries", "members with one"))
    for s in SOURCES:
        for f in FIELDS:
            es = [e for e in entries if e["source"] == s and e["field"] == f]
            if s == "wikidata" and f == "majority":
                continue
            print("  %-10s %-13s %8d %16d" % (s, f, len(es), len({e["member"] for e in es})))
    paths = Counter(e.get("path") for e in entries if e["source"] == "wikidata" and e["field"] == "party")
    print("  Wikidata party entries by property: %s" % dict(paths))
    amb = {k: v for k, v in STATS.items() if k.startswith("ambiguous_")}
    print("  revisions showing two different current values (no entry made): %s" % (amb or "none"))

    print("\nLABELS")
    print("  %-10s %-13s %8s %11s %10s %11s %9s" % ("source", "field", "valid", "superseded", "erroneous", "can't tell",
                                                     "of them 'none'"))
    tot = Counter()
    for s in SOURCES:
        for f in FIELDS:
            if s == "wikidata" and f == "majority":
                continue
            es = [e for e in entries if e["source"] == s and e["field"] == f]
            c = Counter(e["label"] for e in es)
            tot.update(c)
            print("  %-10s %-13s %8d %11d %10d %11d %9d" % (s, f, c["valid"], c["superseded"], c["erroneous"], c[CANT],
                                                             sum(1 for e in es if e["value"] == "none")))
    print("  %-24s %8d %11d %10d %11d" % ("all", tot["valid"], tot["superseded"], tot["erroneous"], tot[CANT]))

    print("\nUPDATE LAG after a real change (first entry with the new value)")
    for s in SOURCES:
        for f in FIELDS:
            if s == "wikidata" and f == "majority":
                continue
            lags, never = [], 0
            for m, (tc, _old, new) in chg[f]:
                mine = [e for e in entries if e["member"] == m["mid"] and e["source"] == s and e["field"] == f]
                if not mine:          # this source says nothing about the field for this member
                    continue
                hits = [e["t"] for e in mine if e["value"] == new and e["t"] >= tc - CANT_TELL_MARGIN_DAYS * DAY]
                if hits:
                    lags.append((min(hits) - tc) / DAY)
                else:
                    never += 1
            if lags or never:
                print("  %-10s %-13s median %6.1f days over %d changes; not updated by the reference date: %d" %
                      (s, f, statistics.median(lags) if lags else float("nan"), len(lags), never))

    print("\nCOPYING")
    wd_all = [e for e in entries if e["source"] == "wikidata"]
    wp_all = [e for e in entries if e["source"] == "enwiki"]
    copied = [e for e in wd_all if e["automated"] and e.get("cites_parliament")]
    cites = [e for e in wd_all if e.get("cites_parliament")]
    cites_snap = [e for e in cites if e["automated"] is None]
    for name, es in (("Wikidata", wd_all), ("Wikipedia", wp_all)):
        known = [e for e in es if e["automated"] is not None]
        print("  %s entries from bot/tool edits: %d of %d made inside the window (%s); %d snapshot entries (who set"
              " them unknown)" % (name, sum(1 for e in known if e["automated"]), len(known),
                                  pct(sum(1 for e in known if e["automated"]), len(known)), len(es) - len(known)))
    print("  Wikidata entries citing Parliament's data: %d of %d (%s), %d of them snapshot entries; citing Wikipedia"
          " (imported from): %d" % (len(cites), len(wd_all), pct(len(cites), len(wd_all)), len(cites_snap),
                                    sum(1 for e in wd_all if e.get("cites_wikipedia"))))
    print("  machine-copied from Parliament (bot/tool edit AND cites Parliament): %d of all %d entries (%s)" %
          (len(copied), len(entries), pct(len(copied), len(entries))))
    if entries and len(copied) / len(entries) < GO_MAX_MACHINE_COPIED <= (len(copied) + len(cites_snap)) / len(entries):
        print("  NOTE: counting every snapshot entry that cites Parliament as copied would cross the threshold -- the"
              " copy check is inconclusive; who added those statements needs a look.")
    prov = Counter()
    for e in wd_all:
        who = "snapshot (setter unknown)" if e["automated"] is None else ("bot/tool" if e["automated"] else "person")
        what = "cites Parliament" if e.get("cites_parliament") else ("cites something else" if e.get("cites_anything")
                                                                     else "no reference")
        prov[(who, what)] += 1
    print("  Wikidata entries by who made them x what they cite:")
    for who in ("bot/tool", "person", "snapshot (setter unknown)"):
        print("    %-26s %s" % (who + ":", ", ".join("%s %d" % (w, prov[(who, w)]) for w in
                                                    ("cites Parliament", "cites something else", "no reference"))))
    bots = Counter(e.get("user") or "?" for e in wd_all if e["automated"])
    if bots:
        print("  bot/tool editors behind Wikidata entries: %s" % ", ".join("%s %d" % kv for kv in bots.most_common(6)))

    print("\nPARSING CHECKS")
    never = [e for e in entries if not e["in_answer_key"]]
    print("  entries whose value never appears in the member's answer key: %d ('not an MP' %d, majority %d, other %d)"
          % (len(never), sum(1 for e in never if e["value"] == "none"), sum(1 for e in never if e["field"] == "majority"),
             sum(1 for e in never if e["value"] != "none" and e["field"] != "majority")))
    gold_of = {m["mid"]: m["gold"] for m in sample}
    shown = [e for e in never if e["value"] != "none"]
    shown.sort(key=lambda e: (e["field"] == "majority", e["field"], e["member"], e["time"]))
    print("  the party/constituency ones, then majority (member, source, date: value -> Parliament then), up to 30:")
    for e in shown[:30]:
        print("    %5d %-8s %s %-12s %-34s -> %s [%s]" % (
            e["member"], e["source"], e["time"][:10], e["field"], str(e.get("raw") or e["value"])[:34],
            str(gold_of[e["member"]].at(e["field"], e["t"]))[:30], e["label"]))
    if wd_unmatched_p39:
        print("  Wikidata positions with an electoral district not counted as Commons seats (latest revision):")
        for k, c in wd_unmatched_p39.most_common(8):
            print("    %4d  %s" % (c, k))
    no_uk = [m["mid"] for m in sample if m["wd"] and not any(
        s["pid"] == "P39" and s["districts"] and _UKPARL.search(LABELS_CACHE.get(s["value"], ""))
        for _, _, stmts, _ in m["wd"][-1:] for s in stmts or [])]
    print("  members whose Wikidata item (latest revision) has no Commons seat statement: %d %s" %
          (len(no_uk), no_uk[:12] if no_uk else ""))
    if ENWIKI_NO_SEAT:
        print("  office lines in Wikipedia infoboxes where no Commons seat was found (one count per article), top 10:")
        for k, c in Counter(v for _, v in ENWIKI_NO_SEAT).most_common(10):
            print("    %4d  %s" % (c, k))
    if STATS.get("enwiki_party_several_lines"):
        print("  Wikipedia party fields with several lines: %d revisions" % STATS["enwiki_party_several_lines"])

    print("\nSENSITIVITY (not part of the verdict)")
    keep = [e for e in entries if e["value"] != "none"]
    print("  without 'not an MP' entries: superseded per member %.2f, erroneous %d, can't tell %d" % (
        sum(1 for e in keep if e["label"] == "superseded") / n if n else 0.0,
        sum(1 for e in keep if e["label"] == "erroneous"), sum(1 for e in keep if e["label"] == CANT)))
    by_era = {k: [m for m in sample if era(day_of(m["start"])) == k] for k in keys}
    if f_era and all(by_era[k] for k in keys if f_era[k]):
        fw = {k: f_era[k] / sum(f_era.values()) for k in keys}
        def wavg(fn):                       # frame-weighted mean over members of a per-member quantity
            return sum(fw[k] * sum(fn(m) for m in by_era[k]) / len(by_era[k]) for k in keys if by_era[k])
        mine = defaultdict(list)
        for e in entries:
            mine[e["member"]].append(e)
        w_two = {f: wavg(lambda m, f=f: float(all(any(e["source"] == s2 and e["field"] == f for e in mine[m["mid"]])
                                                    for s2 in SOURCES))) for f in ("party", "constituency")}
        w_sup = wavg(lambda m: sum(1 for e in mine[m["mid"]] if e["label"] == "superseded"))
        w_err = n * wavg(lambda m: sum(1 for e in mine[m["mid"]] if e["label"] == "erroneous"))
        w_cop = wavg(lambda m: sum(1 for e in mine[m["mid"]] if e["source"] == "wikidata" and e["automated"]
                                   and e.get("cites_parliament"))) / max(1e-9, wavg(lambda m: len(mine[m["mid"]])))
        print("  reweighted to the frame's mix of first years (%s):" % ", ".join("%s %.0f%%" % (k, 100 * fw[k])
                                                                                for k in keys))
        print("    1. both sources: party %.0f%%, constituency %.0f%%  2. superseded per member %.2f  3. erroneous"
              " %.0f  4. machine-copied %.0f%%" % (100 * w_two["party"], 100 * w_two["constituency"], w_sup, w_err,
                                                  100 * w_cop))
        w_ok = (min(w_two.values()) >= GO_TWO_SOURCES_SHARE and w_sup >= GO_SUPERSEDED_PER_MEMBER and
                w_err >= GO_MIN_ERRONEOUS and w_cop < GO_MAX_MACHINE_COPIED)
        print("    reweighted verdict: %s" % ("GO" if w_ok else "NO-GO"))
        for k in keys:
            ms = by_era[k]
            if not ms:
                print("    first year %-13s  0 members" % k)
                continue
            print("    first year %-13s %2d members: entries/member %.1f, superseded/member %.2f, erroneous/member %.2f"
                  % (k, len(ms), sum(len(mine[m["mid"]]) for m in ms) / len(ms),
                     sum(1 for m in ms for e in mine[m["mid"]] if e["label"] == "superseded") / len(ms),
                     sum(1 for m in ms for e in mine[m["mid"]] if e["label"] == "erroneous") / len(ms)))
    else:
        print("  reweighting not possible (a first-year group of the frame has no sampled member)")

    print("\nGO / NO-GO (thresholds fixed before the first run)")
    two = {}
    for f in ("party", "constituency"):
        both = sum(1 for m in sample if all(any(e["member"] == m["mid"] and e["source"] == s and e["field"] == f
                                                for e in entries) for s in SOURCES))
        two[f] = both / n if n else 0.0
    sup_pm = tot["superseded"] / n if n else 0.0
    cop = len(copied) / len(entries) if entries else 0.0
    checks = [
        ("1. both sources have entries, share of members: party %.0f%%, constituency %.0f%% (need >= %.0f%% each)"
         % (100 * two["party"], 100 * two["constituency"], 100 * GO_TWO_SOURCES_SHARE),
         min(two.values()) >= GO_TWO_SOURCES_SHARE),
        ("2. superseded entries per member: %.2f (need >= %.1f)" % (sup_pm, GO_SUPERSEDED_PER_MEMBER),
         sup_pm >= GO_SUPERSEDED_PER_MEMBER),
        ("3. erroneous entries: %d (need >= %d)" % (tot["erroneous"], GO_MIN_ERRONEOUS),
         tot["erroneous"] >= GO_MIN_ERRONEOUS),
        ("4. machine-copied from Parliament: %.0f%% of entries (need < %.0f%%)" % (100 * cop, 100 * GO_MAX_MACHINE_COPIED),
         cop < GO_MAX_MACHINE_COPIED),
    ]
    for text, ok in checks:
        print("  %s  %s" % ("PASS" if ok else "FAIL", text))
    verdict = "GO" if all(ok for _, ok in checks) else "NO-GO"
    if rep_gap is not None and rep_gap > 0.15:
        verdict += " (sample not representative of the frame -- see the warning in step 3)"
    print("  VERDICT: %s" % verdict)

    with open(os.path.join(OUT, "probe_entries.jsonl"), "w") as f:
        for e in entries:
            f.write(json.dumps({k: v for k, v in e.items() if k != "t"}) + "\n")
    summary = {"run_date": run_day, "seed": SEED, "sample": [m["mid"] for m in sample], "skipped": dict(skipped),
               "labels": {k: tot[k] for k in LABELS}, "two_sources": two, "superseded_per_member": sup_pm,
               "machine_copied_share": cop, "verdict": verdict, "checks": [[t, ok] for t, ok in checks],
               "routes": dict(routes), "sample_by_first_year": dict(s_era), "frame_by_first_year": dict(f_era),
               "downloads": {k: v for k, v in STATS.items() if k.startswith("downloads_")}}
    with open(os.path.join(OUT, "probe_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("\nwrote %s and %s (%.0f min; downloads this run: %s)" % (
        os.path.join(OUT_DIR, "probe_entries.jsonl"), os.path.join(OUT_DIR, "probe_summary.json"),
        (time.time() - t_start) / 60.0, {k[10:]: v for k, v in STATS.items() if k.startswith("downloads_")} or 0))


if __name__ == "__main__":
    main()
