import json

import pytest

import assemble
import report
import schema
from test_budget_schema import result


def shapes():
    out = []
    for key in ("d1", "d16", "d32", "v1", "v16", "v32"):
        for sfx in "sf":
            out.append(dict(key=key + sfx, base_us=100.0, aa=dict(median=1.0, band=0.01, valid=True), pick="x",
                            arms=[dict(name="twin", status="valid", ratio=1.0, saved_us=0.0),
                                  dict(name="x", status="valid", ratio=1.1, saved_us=9.0)]))
    return out


def full(rule):
    d = result(0)
    d["check"].update(name="persist", git="abc1234", predeclared=rule)
    d["shapes"] = shapes()
    d["gates"] = [dict(name=n, **{"pass": True}) for n in ("kernels", "clone_state")]
    return d


def wrap(d, tail_extra=""):
    line, _ = schema.trim(d)
    return dict(id="0123456789abcdef", board="muse_decode_layer", content_hash="f" * 64,
                result=dict(log_tail="noise\n" + line + "\n" + tail_extra,
                            passed=True, returncode=0, error=None, status="ok"))


def test_extract_and_kill():
    rule = dict(shape="d1s", arm="x", metric="saved_us", op="<", value=10.0, text="saved_us<10 at d1s")
    data = wrap(full(rule))
    res = report.extract(data["result"]["log_tail"])
    outcome, notes = report.evaluate(res, 100)
    assert outcome == "KILL"
    row = report.draft_row("result.json", data, res, outcome, notes, today="2026-10-26").split("\t")
    assert len(row) == len(report.COLUMNS) and row[report.COLUMNS.index("verdict")] == "REJECT"
    assert row[report.COLUMNS.index("job")] == "0123456789abcdef"
    assert "projection" in row[report.COLUMNS.index("numbers")] and "1.1" in row[report.COLUMNS.index("numbers")]


def test_budget_flag_invalidates_even_a_complete_matrix():
    d = full(dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0))
    d["verdict"]["flags"] = ["BUDGET_EXHAUSTED"]
    assert report.evaluate(d)[0] == "INVALID"


@pytest.mark.parametrize("damage", [None, "missing", "changed", "partial"])
def test_required_raw_record_must_survive_the_returned_log(damage):
    import hashlib

    line = 'CK_RAW {"key":"d1s","times_us":{"base":[101.5,102.25]}}'
    d = full(dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0))
    d["verdict"]["raw_sha256"] = [hashlib.sha256(line.encode()).hexdigest()]
    data = wrap(d)
    retained = {None: line, "missing": "", "changed": line.replace("101.5", "101.6"), "partial": line[15:]}
    data["result"]["log_tail"] = retained[damage] + "\n" + data["result"]["log_tail"]
    got = report.collect(data)
    assert got["outcome"] == ("PASS" if damage is None else "INVALID")
    if damage:
        assert "raw" in " ".join(got["notes"]).lower()


@pytest.mark.parametrize("failure", ["missing_entry_off", "skipped_nearby"])
def test_unperformed_nearby_validation_cannot_qualify(failure):
    d = full(dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0))
    if failure == "missing_entry_off":
        d["verdict"].update(near_ok=None, flags=["NEARBY_NEEDS_ENTRY_OFF"])
    else:
        d["verdict"]["near_ok"] = True  # No failed rows, but one requested case never ran.
        d["nearby"] = [dict(label="d2s", status="skipped")]
    collected = report.collect(wrap(d))
    assert collected["outcome"] == "INVALID"
    assert "nearby" in " ".join(collected["notes"]).lower()


def test_pass_and_invalid_aa():
    rule = dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0)
    assert report.evaluate(report.extract(wrap(full(rule))["result"]["log_tail"]))[0] == "PASS"
    d = full(rule)
    d["shapes"][0]["aa"]["valid"] = False
    assert report.evaluate(report.extract(wrap(d)["result"]["log_tail"]))[0] == "INVALID_AA"


def test_truncated_and_gate_failure():
    outcome, notes = report.evaluate(report.extract("x" * 16000), 16000)
    assert outcome == "TRUNCATED" and "at the cap" in notes[0]
    d = full(dict(shape="d1s", arm="x", metric="ratio", op="<", value=1.0))
    d["gates"] = [dict(name="clone_state", **{"pass": False})]
    outcome, notes = report.evaluate(d)
    assert outcome == "INVALID"
    assert "gate clone_state FAILED" in notes


@pytest.mark.parametrize("failure,expected", [("nearby", "INVALID"), ("raised", "INVALID"),
                                              ("missing", "INVALID"), ("aa", "INVALID_AA")])
def test_non_kill_shape_failures_cannot_qualify(failure, expected):
    d = full(dict(shape="d1s", arm="x", metric="ratio", op="<", value=1.0))
    d["check"]["shapes"] = [s["key"] for s in d["shapes"]]
    if failure == "nearby":
        d["verdict"]["near_ok"] = False
    elif failure == "raised":
        d["verdict"]["flags"] = ["RAISED:v32f"]
    elif failure == "missing":
        d["shapes"].pop()
    else:
        d["shapes"][-1]["aa"]["valid"] = False
    assert report.evaluate(d)[0] == expected


@pytest.mark.parametrize("failure", [dict(passed=False), dict(returncode=1), dict(error="row bound exceeded")])
def test_official_failure_cannot_qualify(tmp_path, capsys, failure):
    data = wrap(full(dict(shape="d1s", arm="x", metric="ratio", op="<", value=1.0)))
    data["result"].update(failure)
    path = tmp_path / "failed.json"
    path.write_text(json.dumps(data))
    assert report.main([str(path)]) == 1
    text = capsys.readouterr().out
    assert "outcome INVALID" in text and "QUALIFIED" not in text


def test_carrier_mismatch_forces_invalid_cli_row(tmp_path, capsys):
    rule = dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0)
    data = wrap(full(rule))
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(data))
    carrier_path = tmp_path / "carrier.py"
    carrier_path.write_text("different carrier bytes")

    exit_code = report.main([str(result_path), "--carrier", str(carrier_path)])

    output = capsys.readouterr().out
    row = output.splitlines()[-1].split("\t")
    assert exit_code == 1
    assert row[report.COLUMNS.index("verdict")] == "INVALID"
    assert "carrier sha256 differs" in output


def test_spec_mismatch_forces_invalid_cli_row(tmp_path, monkeypatch, capsys):
    rule = dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0)
    data = wrap(full(rule))
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(data))
    spec_path = tmp_path / "spec.py"
    spec_path.write_text("SPEC = {}\n")
    monkeypatch.setattr("assemble.load_spec", lambda _path: {})
    monkeypatch.setattr("assemble.spec_sha", lambda _spec: "different")

    exit_code = report.main([str(result_path), "--spec", str(spec_path)])

    output = capsys.readouterr().out
    row = output.splitlines()[-1].split("\t")
    assert exit_code == 1
    assert row[report.COLUMNS.index("verdict")] == "INVALID"
    assert "spec hash differs" in output


def test_report_accepts_spec_fingerprint_emitted_by_assembly(tmp_path, capsys):
    from test_assemble import ENTRY, SPEC

    _, prov = assemble.assemble_text(ENTRY, assemble.sha(ENTRY), SPEC)
    result = full(SPEC["predeclared"])
    result["check"].update(prov)
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(wrap(result)))
    spec_path = tmp_path / "spec.py"
    spec_path.write_text("SPEC = " + repr(SPEC))

    assert report.main([str(result_path), "--spec", str(spec_path)]) == 0
    assert "spec hash differs" not in capsys.readouterr().out


def test_malformed_latest_ck_result_is_invalid_not_fallback(tmp_path, capsys):
    rule = dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0)
    data = wrap(full(rule))
    good_line = data["result"]["log_tail"].splitlines()[1]
    data["result"]["log_tail"] += "CK_RESULT {malformed}\n"
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(data))

    exit_code = report.main([str(result_path)])

    output = capsys.readouterr().out
    row = output.splitlines()[-1].split("\t")
    assert good_line.startswith("CK_RESULT ")
    assert exit_code == 1
    assert row[report.COLUMNS.index("verdict")] == "INVALID"
    assert "malformed latest CK_RESULT" in output


def test_projection_requires_six_loads():
    assert abs(report.projection(shapes(), "pick") - 1.1) < 1e-9
    assert report.projection(shapes()[:4], "pick") is None


@pytest.mark.parametrize("status", ["absent", "fallback", "dropped", "invalid_after_timing", "missing_ratio"])
def test_projection_rejects_missing_or_invalid_selected_arm(status):
    cells = shapes()
    if status == "absent":
        cells[-1]["arms"].pop()
    elif status == "missing_ratio":
        cells[-1]["arms"][-1].pop("ratio")
    else:
        cells[-1]["arms"][-1]["status"] = status
    assert report.projection(cells, "pick") is None
    assert report.projection(cells, "best") is None


def test_projection_preserves_valid_no_pick_as_unchanged():
    cells = shapes()
    for cell in cells:
        cell["pick"] = None
    assert report.projection(cells, "pick") == pytest.approx(1.0)
    assert report.projection(cells, "best") == pytest.approx(1.1)


@pytest.mark.parametrize("outcome", ["INVALID", "INVALID_AA", "TRUNCATED"])
def test_invalid_evidence_never_prints_whole_layer_projection(outcome):
    d = full(dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0))
    row = report.draft_row("invalid.json", wrap(d), d, outcome, ["evidence invalid"])
    numbers = row.split("\t")[report.COLUMNS.index("numbers")]
    assert "d1s 100.0us" in numbers
    assert "projection" not in numbers


@pytest.mark.parametrize("variant", ["sliding_only", "one_full_missing", "duplicate", "invalid_aa"])
def test_projection_requires_one_valid_measurement_for_every_scored_cell(variant):
    cells = shapes()
    if variant == "sliding_only":
        cells = [cell for cell in cells if cell["key"].endswith("s")]
    elif variant == "one_full_missing":
        cells.pop()
    elif variant == "duplicate":
        cells.append(cells[0])
    else:
        cells[-1]["aa"]["valid"] = False
    assert report.projection(cells, "pick") is None
    assert report.projection(cells, "best") is None


def test_incomplete_projection_preserves_component_evidence_without_whole_board_number():
    data = full(dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0))
    data["shapes"] = [cell for cell in data["shapes"] if cell["key"].endswith("s")]
    row = report.draft_row("partial.json", wrap(data), data, "INVALID", ["missing full cells"])
    numbers = row.split("\t")[report.COLUMNS.index("numbers")]
    assert "d1s 100.0us" in numbers
    assert "projection" not in numbers


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf"), "1.1", True])
def test_projection_rejects_invalid_measurement_ratios(ratio):
    cells = shapes()
    cells[-1]["arms"][-1]["ratio"] = ratio
    assert report.projection(cells, "pick") is None
    assert report.projection(cells, "best") is None


def test_failed_measurement_still_prints_invalid_evidence_row(tmp_path, capsys):
    data = full(dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0))
    data["shapes"][0].update(base_us=None, aa=dict(valid=False), pick=None)
    data["shapes"][0]["arms"] = [dict(name="x", status="raised", note="capture failed")]
    path = tmp_path / "failed_measurement.json"
    path.write_text(json.dumps(wrap(data)))

    assert report.main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "outcome INVALID_AA" in output
    assert "d1s unavailable" in output and "projection" not in output


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), float("-inf")])
def test_report_rejects_serialized_nonfinite_ratio(tmp_path, capsys, ratio):
    d = full(dict(shape="d1s", arm="x", metric="ratio", op="<", value=1.0))
    d["gates"] = [dict(name=n, **{"pass": True}) for n in ("kernels", "clone_state")]
    d["shapes"][0]["arms"][-1]["ratio"] = ratio
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(wrap(d)))

    assert report.main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "outcome INVALID" in output and "nonfinite" in output


@pytest.mark.parametrize("defect,error", [
    ("missing_gate", "required gates missing"),
    ("duplicate_gate", "duplicate gate"),
    ("duplicate_cell", "duplicate shape cell"),
    ("duplicate_arm", "duplicate arm"),
    ("missing_ratio", "ratio"),
    ("missing_saved_us", "saved_us"),
    ("missing_base", "base_us"),
    ("missing_aa_median", "aa.median"),
])
def test_report_rejects_incomplete_or_duplicate_evidence(tmp_path, capsys, defect, error):
    d = full(dict(shape="d1s", arm="x", metric="ratio", op="<", value=1.0))
    d["check"]["shapes"] = [s["key"] for s in d["shapes"]]
    d["gates"] = [dict(name=n, **{"pass": True}) for n in ("kernels", "clone_state")]
    cell = d["shapes"][-1]
    if defect == "missing_gate":
        d["gates"].pop()
    elif defect == "duplicate_gate":
        d["gates"].append(d["gates"][0])
    elif defect == "duplicate_cell":
        d["shapes"].append(cell)
    elif defect == "duplicate_arm":
        cell["arms"].append(dict(cell["arms"][-1]))
    elif defect == "missing_ratio":
        cell["arms"][-1].pop("ratio")
    elif defect == "missing_saved_us":
        cell["arms"][-1].pop("saved_us")
    elif defect == "missing_base":
        cell.pop("base_us")
    else:
        cell["aa"].pop("median")
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(wrap(d)))

    assert report.main([str(path)]) == 1
    output = capsys.readouterr().out
    assert "outcome INVALID" in output and error in output


@pytest.mark.parametrize("field,value", [
    ("check.shapes", [{}]),
    ("check.predeclared", {"shape": "d1s"}),
    ("verdict.flags", [None]),
    ("shapes", {}),
    ("gates", [None]),
    ("empty_arms", []),
])
def test_report_malformed_structure_has_clear_invalid_result(tmp_path, capsys, field, value):
    d = full(dict(shape="d1s", arm="x", metric="ratio", op="<", value=1.0))
    if field == "empty_arms":
        d["shapes"][-1]["arms"] = value
    elif "." in field:
        parent, child = field.split(".")
        d[parent][child] = value
    else:
        d[field] = value
    data = dict(id="test", board="muse_decode_layer", result=dict(
        passed=True, returncode=0, log_tail="CK_RESULT " + json.dumps(d)))
    path = tmp_path / "structure.json"
    path.write_text(json.dumps(data))

    assert report.main([str(path)]) == 1
    assert "outcome INVALID" in capsys.readouterr().out
