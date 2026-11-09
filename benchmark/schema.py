"""The ck/1 result: one JSON line `CK_RESULT <json>` of at most 6000 characters, printed last. Pure Python."""

import json
import math

SCHEMA = "ck/1"
LIMIT = 6000
REJECTION_NOTE_LIMIT = 96
REQUIRED = ("schema", "check", "env", "calib", "gates", "shapes", "nearby", "verdict")


def new_result(check, env=None, calib=None):
    return dict(schema=SCHEMA, check=check, env=env or {}, calib=calib or {}, gates=[], shapes=[], nearby=[],
                verdict=dict(kept={}, near_ok=None, skipped=[], flags=[]), kill=None)


def _finite_numbers(value, path="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("nonfinite number at %s" % path)
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numbers(item, "%s.%s" % (path, key))
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _finite_numbers(item, "%s[%d]" % (path, i))


def _required_number(value, path, positive=False):
    if type(value) not in (int, float) or (positive and value <= 0):
        raise ValueError("%s must be a %snumber" % (path, "positive " if positive else ""))


def validate(d):
    """Raise ValueError unless d is a well-formed ck/1 result; return d."""
    if not isinstance(d, dict) or d.get("schema") != SCHEMA:
        raise ValueError("not a %s result" % SCHEMA)
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError("missing keys %s" % missing)
    _finite_numbers(d)
    for key in ("check", "env", "calib", "verdict"):
        if not isinstance(d[key], dict):
            raise ValueError("%s must be an object" % key)
    for key in ("gates", "shapes", "nearby"):
        if not isinstance(d[key], list):
            raise ValueError("%s must be a list" % key)
    for key in ("name", "carrier_sha", "spec_sha"):
        if key not in d["check"]:
            raise ValueError("check.%s missing" % key)
    for value, path in ((d["check"].get("shapes", []), "check.shapes"),
                        (d["verdict"].get("flags", []), "verdict.flags")):
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError("%s must be a list of strings" % path)
    rule = d["check"].get("predeclared")
    if rule is not None:
        if not isinstance(rule, dict) or any(not isinstance(rule.get(k), str) for k in ("shape", "arm", "metric", "op")):
            raise ValueError("check.predeclared needs shape, arm, metric and op")
        if rule["metric"] not in ("ratio", "saved_us") or rule["op"] not in ("<", ">"):
            raise ValueError("check.predeclared has an invalid metric or op")
        _required_number(rule.get("value"), "check.predeclared.value")
    gates = set()
    for g in d["gates"]:
        if not isinstance(g, dict) or not isinstance(g.get("name"), str) or type(g.get("pass")) is not bool:
            raise ValueError("gate needs a name and boolean pass")
        if g["name"] in gates:
            raise ValueError("duplicate gate %s" % g["name"])
        gates.add(g["name"])
    cells = set()
    for s in d["shapes"]:
        if not isinstance(s, dict) or not isinstance(s.get("key"), str) or not isinstance(s.get("arms"), list):
            raise ValueError("shape cell without key or arms")
        if s["key"] in cells:
            raise ValueError("duplicate shape cell %s" % s["key"])
        cells.add(s["key"])
        aa = s.get("aa")
        if not isinstance(aa, dict) or type(aa.get("valid")) is not bool:
            raise ValueError("%s needs aa.valid boolean" % s["key"])
        if aa["valid"]:
            if not s["arms"]:
                raise ValueError("incomplete shape cell %s: no arms" % s["key"])
            _required_number(s.get("base_us"), "%s.base_us" % s["key"], positive=True)
            for key in ("median", "band"):
                _required_number(aa.get(key), "%s.aa.%s" % (s["key"], key))
        arms = set()
        for a in s["arms"]:
            if not isinstance(a, dict) or not isinstance(a.get("name"), str) or not isinstance(a.get("status"), str):
                raise ValueError("arm without name or status in %s" % s["key"])
            if a["name"] in arms:
                raise ValueError("duplicate arm %s in %s" % (a["name"], s["key"]))
            arms.add(a["name"])
            if aa["valid"] and a["status"] in ("valid", "probe"):
                _required_number(a.get("ratio"), "%s.%s.ratio" % (s["key"], a["name"]), positive=True)
                _required_number(a.get("saved_us"), "%s.%s.saved_us" % (s["key"], a["name"]))
    return d


def rounded(x, digits=3):
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, dict):
        return {k: rounded(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rounded(v, digits) for v in x]
    return x


def dumps(d):
    return "CK_RESULT " + json.dumps(rounded(d), separators=(",", ":"), sort_keys=False, default=str)


def _drop_pairs(d):
    for s in d["shapes"]:
        for a in s["arms"]:
            a.pop("pairs", None)


def _drop_notes(d):
    for s in d["shapes"]:
        for a in s["arms"]:
            if a["status"] != "valid" and a.get("note"):
                a["note"] = a["note"][:REJECTION_NOTE_LIMIT]
            else:
                a.pop("note", None)
            for k in ("kernels", "routes"):
                a.pop(k, None)
    for n in d["nearby"]:
        n.pop("note", None)


def _drop_nearby(d):
    d["nearby"] = [n for n in d["nearby"] if n.get("status") != "pass"]


def _drop_details(d):
    # The structured predicate remains in check.predeclared, and evaluated is the kill decision.
    # Gate names and booleans are sufficient for ingest even when their explanations are long.
    if isinstance(d.get("kill"), dict) and d.get("check", {}).get("predeclared"):
        d["kill"].pop("rule", None)
    for gate in d["gates"]:
        gate.pop("detail", None)
    for s in d["shapes"]:
        s.pop("note", None)


def trim(d, limit=LIMIT):
    """-> (line, flags): fit by dropping pairs, successful notes, passing nearby rows, then duplicate rule text
    and gate details. Keep bounded rejection notes, arm name/status/ratios, A/A, gates, kill rule and verdict."""
    flags = []
    line = dumps(d)
    for step in (_drop_pairs, _drop_notes, _drop_nearby, _drop_details):
        if len(line) <= limit:
            break
        step(d)
        flags.append("TRIM_" + step.__name__[6:].upper())
        d["verdict"]["flags"] = sorted(set(d["verdict"]["flags"]) | set(flags[-1:]))
        line = dumps(d)
    if len(line) > limit:
        raise ValueError("CK_RESULT still %d characters after trimming" % len(line))
    return line, flags


def table_rows(d):
    """One short row per shape: key, base us, A/A, best arm ratio and saved us."""
    rows = []
    for s in d["shapes"]:
        arms = [a for a in s["arms"] if a["name"] != "twin" and "ratio" in a and a["status"] == "valid"]
        best = max(arms, key=lambda a: a["ratio"], default=None)
        aa = s.get("aa", {})
        rows.append("%s base %.1f aa %.3f band %.3f%s best %s" % (
            s["key"], s.get("base_us") if s.get("base_us") is not None else float("nan"),
            aa.get("median", float("nan")), aa.get("band", float("nan")),
            "" if aa.get("valid", True) else " AA-INVALID",
            "-" if best is None else "%s %.3f %+.1fus" % (best["name"], best["ratio"], best.get("saved_us", 0.0))))
    return rows
