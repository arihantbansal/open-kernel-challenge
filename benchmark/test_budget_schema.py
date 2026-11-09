import json
from copy import deepcopy

import pytest

import schema
from budget import Budget, Emitter, RESULT_CAP, priority_order


def result(n_shapes=12, n_arms=6):
    d = schema.new_result(dict(name="t", carrier_sha="x", spec_sha="y"))
    for i in range(n_shapes):
        d["shapes"].append(dict(key="s%d" % i, base_us=100.123456, aa=dict(median=1.0004, band=0.01, valid=True),
                                pick=None, arms=[dict(name="a%d" % j, status="valid", ratio=1.0 + j / 100, saved_us=j * 0.37,
                                                      note="rel 1e-4 " * 8, pairs=[1.01] * 24, kernels=dict(seen=True))
                                                 for j in range(n_arms)]))
    d["nearby"] = [dict(label="s0+b2", status="pass", note="n" * 40) for _ in range(30)]
    d["kill"] = dict(rule="r", evaluated="PASS")
    return d


def test_validate():
    schema.validate(result())
    with pytest.raises(ValueError):
        schema.validate(dict(schema="ck/1"))
    with pytest.raises(ValueError):
        schema.validate(dict(result(), schema="other"))


def test_trim_fits_and_keeps_kill_and_gates():
    d = result()
    assert len(schema.dumps(d)) > RESULT_CAP
    line, flags = schema.trim(d)
    assert len(line) <= RESULT_CAP and flags
    out = json.loads(line[len("CK_RESULT "):])
    assert out["kill"]["evaluated"] == "PASS" and len(out["shapes"]) == 12
    assert out["shapes"][0]["base_us"] == 100.123  # rounded to 3 digits


def test_small_result_is_untouched():
    d = result(2, 2)
    line, flags = schema.trim(d)
    assert flags == [] and "pairs" in line


@pytest.mark.parametrize("status", ("dropped", "invalid_after_timing", "raised", "fallback"))
def test_trim_preserves_a_bounded_rejection_reason(status):
    d = result(n_shapes=12, n_arms=2)
    reason = "routed counts differ (201 vs 200 routes) (input 1)"
    d["shapes"][0]["arms"][1].update(status=status, note=reason + "; detail" * 2000)
    line, flags = schema.trim(d)
    out = schema.validate(json.loads(line.removeprefix("CK_RESULT ")))
    rejected = out["shapes"][0]["arms"][1]
    assert rejected.get("note", "").startswith(reason), rejected
    assert len(rejected["note"]) <= 96
    assert len(line) <= RESULT_CAP and "TRIM_NOTES" in flags
    assert all("note" not in a for s in out["shapes"] for a in s["arms"] if a["status"] == "valid")


def test_full_layer_stack_fits_without_losing_decision_fields():
    keys = [load + suffix for load in ("d1", "d16", "d32", "v1", "v16", "v32") for suffix in "sf"]
    rule = dict(shape="v16f", arm="reg", metric="saved_us", op="<", value=8.0,
                text="Kill if reg56-only saves <8us v16f or <15us v32f; require stack aggregate >=0.6% over original "
                     "V5+fa128, all gates and strict routed counts, all twelve cells valid, no meaningful regression "
                     "beyond max(2%,2x A/A band); nearby v16f/v32f batch +/-1, qlen +/-1, page64 and ctx+1 must pass")
    d = schema.new_result(dict(name="fa3_stack_no_c0_v1", carrier_sha=None, spec_sha="s" * 16,
                              entry_sha="e" * 16, kit_sha="k" * 16, git="ea36359de+dirty",
                              problem="muse_decode_layer", shapes=keys, predeclared=rule),
                          dict(gpu="NVIDIA H200", sms=132, torch="2.14.1+cu130", prepare_s=220.1),
                          dict(ovh_us=3.84, ovh_ungated_us=3.84, flush_mib=256, flush_kind="read", flags=[]))
    v5 = {"d1s", "d16s", "d16f", "d32s", "d32f", "v1s", "v1f"}
    for key in keys:
        names = ["twin", "stack"] + (["pre"] if key in v5 else ["reg"] if key in ("v16f", "v32f") else [])
        d["shapes"].append(dict(key=key, tokens=128, nsets=3, pick="stack", pass_="stack", base_us=230.464,
                                aa=dict(median=1.003, band=.014, valid=True),
                                arms=[dict(name=n, kind="control" if n == "twin" else "real", status="valid",
                                           ratio=1.021, saved_us=4.741, note="rel 1e-4 ids eq", pairs=[230.1] * 24,
                                           kernels=dict(seen=True, n=19), routes=dict(sum=318, hot=42, distinct=13))
                                      for n in names]))
    # Failed arms are evidence too; trimming must not erase their names or statuses.
    d["shapes"][1]["arms"].append(dict(name="dropped", status="fallback", note="expected kernel absent"))
    d["gates"] = [dict(name=n, **{"pass": n != "matrix"}, detail="captured execution and strict routed counts; " * 5)
                  for n in ("kernels", "clone_state", "fa3_extension", "v5_extensions", "matrix", "attention_nearby")]
    d["kill"] = dict(rule=rule["text"], evaluated="KILL")
    d["nearby"] = [dict(label="v16f+b15", status="pass", ratio=1.01, band=.012)] * 12
    d["verdict"].update(kept={k: "stack" for k in keys}, near_ok=True, flags=["ORIGINAL_FLAG"])
    before = deepcopy(d)
    line, flags = schema.trim(d)
    assert len(line) <= RESULT_CAP and "TRIM_DETAILS" in flags
    out = schema.validate(json.loads(line[len("CK_RESULT "):]))
    assert out["check"]["predeclared"] == before["check"]["predeclared"]
    assert out["kill"]["evaluated"] == "KILL" and "ORIGINAL_FLAG" in out["verdict"]["flags"]
    assert [(g["name"], g["pass"]) for g in out["gates"]] == [(g["name"], g["pass"]) for g in before["gates"]]
    for old, new in zip(before["shapes"], out["shapes"]):
        assert new["aa"] == old["aa"] and new["pick"] == old["pick"]
        assert [{k: a[k] for k in ("name", "status", "ratio", "saved_us") if k in a} for a in new["arms"]] == [
            {k: a[k] for k in ("name", "status", "ratio", "saved_us") if k in a} for a in old["arms"]]
    output = []
    emit = Emitter(output.append)
    for _ in range(100):
        emit.line("diagnostic " * 30)
    emit.final(schema.table_rows(out), line)
    assert line in "\n".join(output)[-16000:]


def test_emitter_caps_and_final_is_last():
    out = []
    e = Emitter(out.append, human_cap=100, raw_cap=60)
    kept = [e.line("x" * 30) for _ in range(6)]
    assert kept == [True] * 2 + [False] * 4 and e.dropped["human"] == 4
    assert e.raw("d1s", dict(times_us=dict(base=[1.0])))
    assert not e.raw("d1s", dict(times_us=dict(base=[1.0] * 30)))
    info = e.final(["row %d" % i for i in range(200)], "CK_RESULT {}")
    assert out[-1] == "CK_RESULT {}" and out[-2].startswith("CK_TABLE") and info["table_rows"] < 200
    with pytest.raises(ValueError):
        e.final([], "CK_RESULT " + "x" * RESULT_CAP)


def test_raw_samples_survive_profiler_output_between_cells_and_after_prepare():
    out = []
    e = Emitter(out.append, raw_cap=6000)
    samples = dict(times_us=dict(base=[113.504, 114.688], twin=[114.432, 113.216]))
    assert e.raw("v1s", samples)
    # Native profiler output bypasses Emitter's budgets. The judge retains the
    # last 16,000 characters, including output after prepare has returned.
    out.append("profiler_start\nprofiler_stop\n" * 1000)
    assert e.raw("v1f", samples)
    out.append("profiler_start\nprofiler_stop\n" * 1000)
    e.final([], "CK_RESULT {}")
    out.append("profiler_start\nprofiler_stop\n" * 100)
    tail = "\n".join(out)[-16000:]
    raw = [json.loads(s.removeprefix("CK_RAW ")) for s in tail.splitlines() if s.startswith("CK_RAW ")]
    assert raw == [dict(key=key, **samples) for key in ("v1s", "v1f")]


def test_budget_and_priority():
    now = [0.0]
    b = Budget(0.0, stop_s=10, near_stop_s=20, clock=lambda: now[0])
    assert b.ok("a")
    now[0] = 15
    assert not b.ok("b") and b.ok("n", near=True)
    now[0] = 25
    assert not b.ok("c", near=True) and len(b.skipped) == 2
    assert priority_order(["a", "b", "c"], ["c", "z"]) == ["c", "a", "b"]


def test_gpu_rounds_stop_between_replays_without_returning_partial_samples(monkeypatch):
    import sys
    import types
    import gpu
    from budget import BudgetExpired

    now, replayed = [0.0], []
    budget = Budget(0, stop_s=1, clock=lambda: now[0])
    cuda = types.SimpleNamespace(Event=lambda **kw: None, synchronize=lambda: None)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))

    def event_time(graph, *args):
        replayed.append(graph)
        now[0] += 2
        return 10

    monkeypatch.setattr(gpu, "_event_time", event_time)
    caps = {n: types.SimpleNamespace(graph=n) for n in ("base", "twin", "candidate")}
    with pytest.raises(BudgetExpired, match="timing"):
        gpu.time_rounds(caps, lambda r: None, lambda n: None, None, 0, rounds=4, budget=budget.require)
    assert len(replayed) == 1 and len(budget.skipped) == 1
