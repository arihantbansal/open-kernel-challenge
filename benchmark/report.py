"""Validate a check result and print its predeclared verdict and a draft record.

The draft is printed only; this command never appends to a result file. Pass
--carrier and --spec to check local source and spec bindings.
"""

import argparse
import datetime
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import schema  # noqa: E402
from stats import eval_kill  # noqa: E402

TAIL = 16000
COLUMNS = ("id", "date", "area", "kind", "job", "board", "sha", "artifact", "baseline", "raw", "numbers", "verdict",
           "reopen", "supersedes", "notes")
AREA = {"muse_decode_layer": "layer", "muse_moe_w4a16": "moe", "muse_prefill_layer": "prefill",
        "muse_attention_decode": "attn", "muse_bf16_proj": "bf16"}
VERDICT = {"KILL": "REJECT", "PASS": "QUALIFIED", "INVALID_AA": "INVALID", "INVALID": "INVALID", "TRUNCATED": "INVALID"}


def extract(log_tail):
    """Return the last parsed, validated CK_RESULT, or None; reject a malformed latest record."""
    for line in reversed(log_tail.splitlines()):
        if line.startswith("CK_RESULT "):
            try:
                return schema.validate(json.loads(line[len("CK_RESULT "):]))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError("malformed latest CK_RESULT: %s" % exc) from exc
    return None


def _positive_finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value > 0)


def projection(shapes, key="pick"):
    """Whole-layer score relative to the entry from per-shape ratios (46 sliding : 16 full weights, six loads).
    Requires exactly one valid measurement of every scored cell. An unchanged cell counts 1.0.
    key "pick": the arm the check kept; "best": the best valid arm whatever its kind."""
    import score
    from recipes.layer import LAYER_CASES

    if len(shapes) != len(LAYER_CASES) or {s.get("key") for s in shapes} != set(LAYER_CASES):
        return None
    if any(not s.get("aa", {}).get("valid") or not _positive_finite(s.get("base_us")) for s in shapes):
        return None
    rows = []
    for s in shapes:
        valid = [a for a in s["arms"] if a["status"] == "valid" and a["name"] != "twin" and "ratio" in a]
        if any(not _positive_finite(a["ratio"]) for a in valid):
            return None
        picked = next((a for a in valid if a["name"] == s.get("pick")), None)
        if s.get("pick") is not None and picked is None:
            return None
        if key == "pick":
            ratio = picked["ratio"] if picked is not None else 1.0
        else:
            ratio = max([a["ratio"] for a in valid] + [1.0])
        rows.append(dict(load=s["key"][:-1], weight=46 if s["key"][-1] == "s" else 16,
                         baseline_ns=s["base_us"] * 1e3, median_ns=s["base_us"] * 1e3 / ratio))
    return score.score(rows)["speedup"]


def evaluate(res, log_len=0):
    """-> (outcome, notes). The predeclared rule is the one recorded in the carrier's own check header."""
    if res is None:
        return "TRUNCATED", ["no CK_RESULT in log_tail (%d chars%s)" % (log_len, ", at the cap" if log_len == TAIL else "")]
    try:
        schema.validate(res)
    except ValueError as exc:
        return "INVALID", [str(exc)]
    notes = list(res["verdict"].get("flags", []))
    failed_gates = [g for g in res["gates"] if not g.get("pass")]
    notes += ["gate %s FAILED" % g["name"] for g in failed_gates]
    if failed_gates:
        return "INVALID", notes
    missing_gates = {"kernels", "clone_state"} - {g["name"] for g in res["gates"]}
    if missing_gates:
        return "INVALID", notes + ["required gates missing: %s" % ", ".join(sorted(missing_gates))]
    if "NEARBY_NEEDS_ENTRY_OFF" in res["verdict"].get("flags", []):
        return "INVALID", notes + ["requested nearby validation did not run: entry_off missing"]
    if res["verdict"].get("near_ok") is False:
        return "INVALID", notes + ["nearby gate FAILED"]
    incomplete = [n.get("label") for n in res["nearby"] if n.get("status") != "pass"]
    if incomplete:
        return "INVALID", notes + ["nearby rows not passed: %s" % incomplete[:6]]
    if any(f.startswith("RAISED") or f == "BUDGET_EXHAUSTED" for f in res["verdict"].get("flags", [])):
        return "INVALID", notes
    expected = set(res["check"].get("shapes", ()))
    if expected - {s["key"] for s in res["shapes"]}:
        return "INVALID", notes + ["requested shape cells missing"]
    if any(not s.get("aa", {}).get("valid") for s in res["shapes"]):
        return "INVALID_AA", notes + ["matrix A/A invalid"]
    if "compound_decision" in res["check"]:
        return "INVALID", notes + ["unsupported compound decision %r; scalar rule not evaluated; "
                                   "requires supervised source-bound evaluation" % res["check"]["compound_decision"]]
    rule = (res["check"].get("predeclared") or {})
    if not rule.get("shape"):
        return "INVALID", notes + ["no predeclared rule in the carrier"]
    return eval_kill(rule, res["shapes"]), notes


def official_problems(data):
    """Prepare-time measurements cannot qualify a failed official check."""
    r = data.get("result") or {}
    problems = []
    if r.get("passed") is not True:
        problems.append("official correctness did not pass")
    if r.get("returncode") != 0:
        problems.append("official returncode is not zero")
    if r.get("error") or r.get("status") not in (None, "ok") or data.get("status") in ("failed", "rejected"):
        problems.append("official result has an error")
    return problems


def collect(data):
    """Accept an official result record or a runner wrapper around one."""
    status = data.get("raw_status") or data
    log = (status.get("result") or {}).get("log_tail") or ""
    problems = official_problems(status)
    try:
        res = extract(log)
        outcome, notes = evaluate(res, len(log))
    except ValueError as exc:
        res, outcome, notes = None, "INVALID", [str(exc)]
    if res:
        required = res["verdict"].get("raw_sha256", [])
        retained = [hashlib.sha256(s.encode()).hexdigest() for s in log.splitlines() if s.startswith("CK_RAW ")]
        if required and required != retained:
            problems.append("required raw timing records missing, changed or truncated")
        if "RAW_TRUNCATED" in res["verdict"].get("flags", []):
            problems.append("raw timing records exceeded the emission budget")
    if problems:
        outcome = "INVALID"
    return dict(result=res, outcome=outcome, notes=notes + problems)


def draft_row(path, data, res, outcome, notes, today=None):
    data = data.get("raw_status") or data
    r = data.get("result", data)
    chk = (res or {}).get("check", {})
    service_hash = data.get("content_hash") or ""
    name = chk.get("name", Path(path).stem)
    shapes = (res or {}).get("shapes", [])
    cells = []
    for s in shapes:
        best = max((a for a in s["arms"] if a["name"] != "twin" and a["status"] == "valid"
                    and _positive_finite(a.get("ratio"))),
                   key=lambda a: a["ratio"], default=None)
        baseline = "%.1fus" % s["base_us"] if _positive_finite(s.get("base_us")) else "unavailable"
        cells.append("%s %s aa %.3f%s" % (s["key"], baseline, s.get("aa", {}).get("median", 0),
                                              "" if best is None else " %s %.3f" % (best["name"], best["ratio"])))
    layer = chk.get("problem", "muse_decode_layer") == "muse_decode_layer"
    projectable = shapes and layer and outcome in ("PASS", "KILL")
    proj = projection(shapes, "best") if projectable else None
    pick = projection(shapes, "pick") if projectable else None
    calib = (res or {}).get("calib", {})
    result = "; ".join(cells[:4]) + ("; ..." if len(cells) > 4 else "")
    result += " | ovh %s us | flags %s" % (calib.get("ovh_us"), ",".join(notes) or "none")
    if pick is not None:
        result += " | projection vs entry (not a score): picked %.4f, best-valid %.4f" % (pick, proj)
    pre = chk.get("predeclared") or {}
    rule = pre.get("text") or pre.get("shape")
    board = data.get("board") or r.get("board") or chk.get("problem", "")
    cols = dict(id="r????", date=today or datetime.date.today().isoformat(), area=AREA.get(chk.get("problem", board), "layer"),
                kind="check", job=data.get("id", ""), board=AREA.get(board, board), sha=chk.get("git", ""),
                artifact="%s service_content_hash:%s" % (chk.get("name", "?"), service_hash[:12]),
                baseline="samerun:entry", raw=str(path),
                numbers=result, verdict=VERDICT[outcome],
                reopen="TODO: what would reopen it" if outcome == "KILL" else "", supersedes="",
                notes="predeclared: %s" % rule)
    return "\t".join(str(cols[c]).replace("\t", " ").replace("\n", " ") for c in COLUMNS)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_json", type=Path)
    ap.add_argument("--carrier", type=Path,
                    help="check carrier SHA-256 against the retained source binding, falling back to content_hash")
    ap.add_argument("--spec", type=Path, help="the spec; its hash must equal the carrier's recorded spec_sha")
    args = ap.parse_args(argv)
    data = json.loads(args.result_json.read_text())
    collected = collect(data)
    res, outcome, notes = (collected[k] for k in ("result", "outcome", "notes"))
    problems = []
    if args.carrier:
        binding_sha = (data.get("source") or {}).get("sha256")
        expected_sha = binding_sha or data.get("content_hash")
        if hashlib.sha256(args.carrier.read_bytes()).hexdigest() != expected_sha:
            problems.append("carrier sha256 differs from " + ("retained source binding" if binding_sha else "the result's content_hash"))
    if args.spec and res:
        import assemble
        # assemble_text emits exactly the first 16 hex digits in ck/1.
        if assemble.spec_sha(assemble.load_spec(args.spec))[:16] != res["check"]["spec_sha"]:
            problems.append("spec hash differs from the carrier's spec_sha")
    if problems:
        outcome = "INVALID"
    print("outcome", outcome, "|", "; ".join(notes + problems) or "clean")
    print("\t".join(COLUMNS))
    print(draft_row(args.result_json, data, res, outcome, notes + problems))
    return 1 if problems or outcome in ("INVALID", "INVALID_AA", "TRUNCATED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
