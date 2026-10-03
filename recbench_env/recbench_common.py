# recbench_common.py -- shared helpers for the record-reconciliation benchmark (v0)
# Python 3.9, stdlib only. Imported by gen.py, baselines.py, score.py, llm_baseline.py.

import json
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------- fields
FIELDS = ["species", "sex", "birth_day", "weight_kg", "diet", "medication", "vet_clinic", "rabies_day"]

FIELD_TYPE = {
    "species": "immutable_cat",
    "sex": "immutable_cat",
    "birth_day": "immutable_num",
    "weight_kg": "drift_num",
    "diet": "regime_cat",
    "medication": "regime_cat",
    "vet_clinic": "regime_cat",
    "rabies_day": "event_day",
}

# tolerances used both for labeling truth and for scoring predictions
TOL_WEIGHT_REL = 0.04      # 4% relative on weight (scale noise + rounding live inside this)
TOL_BIRTH_DAYS = 60        # birthdays are often approximate
TOL_EVENT_DAYS = 7         # vaccine dates

LB_PER_KG = 2.20462
LABELS = ["valid", "superseded", "erroneous"]

SOURCES = ["owner", "vet_pdf", "extractor", "email_forward", "note_text"]

# ---------------------------------------------------------------- generic (real-data) fields
# Real-data cases (realdata_map.py) carry their own field definitions:
#   case["field_types"] = {field: ["num_rel", 0.01] | ["num_abs", 10] | ["cat", null]}
# load_jsonl registers them here so match()/cluster()/is_numeric() work without per-field code.
EXTRA_FIELD_TYPES: Dict[str, str] = {}
EXTRA_TOL: Dict[str, float] = {}


def register_field_types(case: Dict[str, Any]) -> None:
    for field, spec in (case.get("field_types") or {}).items():
        EXTRA_FIELD_TYPES[field] = spec[0]
        EXTRA_TOL[field] = float(spec[1]) if len(spec) > 1 and spec[1] is not None else 0.0


def ftype_of(field: str) -> str:
    """Field type name: one of the pet types, or num_rel / num_abs / cat for registered real-data fields."""
    if field in FIELD_TYPE:
        return FIELD_TYPE[field]
    return EXTRA_FIELD_TYPES.get(field, "cat")


def is_numeric(field: str) -> bool:
    return ftype_of(field) in ("immutable_num", "drift_num", "event_day", "num_rel", "num_abs")


def is_categorical(field: str) -> bool:
    return not is_numeric(field)


def to_kg(value: float, unit: Optional[str]) -> float:
    if unit == "lb":
        return float(value) / LB_PER_KG
    return float(value)


def norm_value(field: str, value: Any, unit: Optional[str] = None) -> Any:
    """Normalize an asserted value into the comparison space of its field."""
    if field == "weight_kg":
        return to_kg(value, unit or "kg")
    if field in ("birth_day", "rabies_day"):
        return float(value)
    if field not in FIELD_TYPE and is_numeric(field):
        return float(value)
    return str(value).strip().lower()


def match(field: str, a: Any, b: Any) -> bool:
    """True if two normalized values are the same fact."""
    if a is None or b is None:
        return False
    if field == "weight_kg":
        return abs(a - b) <= TOL_WEIGHT_REL * max(abs(b), 1e-9)
    if field == "birth_day":
        return abs(a - b) <= TOL_BIRTH_DAYS
    if field == "rabies_day":
        return abs(a - b) <= TOL_EVENT_DAYS
    t = ftype_of(field)
    if t == "num_rel":
        return abs(a - b) <= EXTRA_TOL.get(field, 0.0) * max(abs(b), 1e-9)
    if t == "num_abs":
        return abs(a - b) <= EXTRA_TOL.get(field, 0.0)
    return a == b


def cluster(field: str, items: List[Tuple[Any, Any]]) -> List[Dict[str, Any]]:
    """Group (key, normalized_value) pairs into clusters of matching values.
    Returns [{"value": representative, "members": [key, ...]}, ...]."""
    clusters: List[Dict[str, Any]] = []
    if is_categorical(field):
        index: Dict[str, int] = {}
        for key, v in items:
            if v not in index:
                index[v] = len(clusters)
                clusters.append({"value": v, "members": []})
            clusters[index[v]]["members"].append(key)
        return clusters
    # numeric: greedy, sorted by value, representative = running mean
    for key, v in sorted(items, key=lambda kv: kv[1]):
        placed = False
        for c in clusters:
            if match(field, v, c["value"]):
                c["members"].append(key)
                n = len(c["members"])
                c["value"] = c["value"] + (v - c["value"]) / n
                placed = True
                break
        if not placed:
            clusters.append({"value": v, "members": [key]})
    return clusters


# ---------------------------------------------------------------- IO
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict) and "field_types" in row:
                    register_field_types(row)
                rows.append(row)
    return rows


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- truth lookup
def truth_at(case: Dict[str, Any], field: str, day: int) -> Any:
    """The true value of `field` on `day` (normalized space)."""
    t = case["truth"]
    if field == "species":
        return t["species"]
    if field == "sex":
        return t["sex"]
    if field == "birth_day":
        return float(t["birth_day"])
    if field == "weight_kg":
        knots = t["weight_knots"]  # [[day, kg], ...] ascending
        if day <= knots[0][0]:
            return float(knots[0][1])
        if day >= knots[-1][0]:
            return float(knots[-1][1])
        lo, hi = 0, len(knots) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if knots[mid][0] <= day:
                lo = mid
            else:
                hi = mid
        d0, w0 = knots[lo]
        d1, w1 = knots[hi]
        if d1 == d0:
            return float(w0)
        return float(w0 + (w1 - w0) * (day - d0) / (d1 - d0))
    if field in ("diet", "medication", "vet_clinic"):
        regimes = t[field]  # [[start_day, value], ...] ascending
        cur = regimes[0][1]
        for start, val in regimes:
            if start <= day:
                cur = val
            else:
                break
        return str(cur).strip().lower()
    if field == "rabies_day":
        past = [e for e in t["rabies_events"] if e <= day]
        return float(max(past)) if past else None
    raise KeyError(field)


def current_truth(case: Dict[str, Any], field: str) -> Any:
    """Current true value: generator cases via truth_at; real-data cases via their static truth_values."""
    if "truth_values" in case:
        v = case["truth_values"].get(field)
        return None if v is None else norm_value(field, v)
    return truth_at(case, field, case["now_day"])


def serialize_assertion(a: Dict[str, Any]) -> str:
    """One-line human/LLM readable form (dates as relative days)."""
    unit = (" " + a["unit"]) if a.get("unit") else ""
    extra = ""
    if a.get("corrects") is not None:
        extra += " corrects=#%d" % a["corrects"]
    if a.get("text"):
        extra += ' note="%s"' % a["text"]
    conf = a.get("extractor_conf")
    conf_s = (" conf=%.2f" % conf) if conf is not None else ""
    return "#%d %s = %s%s | observed day %d | arrived day %d | source %s%s%s" % (
        a["id"], a["field"], a["value"], unit, a["observed_day"], a["arrived_day"], a["source"], conf_s, extra)
