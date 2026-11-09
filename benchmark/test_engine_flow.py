"""ck_run end to end on the CPU: torch and every GPU primitive faked, so the control flow, the engine rules and the
final log lines are exercised without a GPU."""
import json
import statistics
import sys
import time
import types

import pytest

import assemble
import engine
import report
from budget import Emitter
from recipes.attn import CallbackRecipe

NAMES = {"entry": ["k_entry"], "good": ["k_good"], "same": ["k_entry"], "price": ["k_price"]}


class Recipe(CallbackRecipe):
    def live(self, src):
        return dict(src)

    def refresh(self, live, src):
        pass


@pytest.fixture(params=["module", "carrier"])
def fake(monkeypatch, request):
    if request.param == "carrier":
        from test_assemble import ENTRY, SPEC

        carrier = types.ModuleType("test_generated_carrier")
        monkeypatch.setitem(sys.modules, carrier.__name__, carrier)
        text, _ = assemble.assemble_text(ENTRY, assemble.sha(ENTRY), SPEC)
        exec(compile(text, "carrier", "exec"), carrier.__dict__)
        monkeypatch.setattr(sys.modules[__name__], "engine", carrier)
    t = types.ModuleType("torch")
    t.__version__ = "fake"
    t.device = lambda s: s
    t.cuda = types.SimpleNamespace(get_device_name=lambda i: "FAKE", synchronize=lambda: None, empty_cache=lambda: None,
                                   get_device_properties=lambda i: types.SimpleNamespace(multi_processor_count=132))
    monkeypatch.setitem(sys.modules, "torch", t)
    state = {"variant": "entry"}
    seen = {"nsets": []}

    class Cap:
        def __init__(self, fn):
            fn()
            self.out = ()
            self.variant = state["variant"]

        def replay(self):
            pass

    monkeypatch.setattr(engine, "Capture", Cap)
    monkeypatch.setattr(engine, "kernels_of", lambda run: NAMES[state["variant"]])
    monkeypatch.setattr(engine, "poison", lambda x: None)
    monkeypatch.setattr(engine, "calibrate", lambda f: dict(flush_mib=256, flush_kind="read", ovh_us=4.0, ovh_ungated_us=4.2, flags=[]))
    monkeypatch.setattr(engine, "Flush", lambda mib, dev: types.SimpleNamespace(mib=mib))
    monkeypatch.setattr(engine, "clocks", lambda: {"sm_mhz": 1980})
    monkeypatch.setattr(engine, "clone_state", lambda s: dict(s))
    monkeypatch.setattr(engine, "replay_valid", lambda *a: (True, "rel 0.0e+00 ids eq", {"routes": {"sum": 1}}))
    speed = {"entry": 100.0, "good": 90.0, "same": 100.0, "price": 70.0}

    def rounds(caps, refresh, poison_fn, flush, ovh, rounds=24):
        return {n: [speed[c.variant] + ovh] * rounds for n, c in caps.items()}
    monkeypatch.setattr(engine, "time_rounds", rounds)

    def hook(variant):
        def apply(st, key):
            old = st["variant"]
            st["variant"] = variant
            return lambda: st.update(variant=old)
        return apply

    recipe = Recipe({"s1": dict(batch=1), "s2": dict(batch=2)},
                    lambda case, seed, n: seen["nsets"].append(n) or [{"x": 0}] * n, lambda d, st: {},
                    lambda *a: (True, "ok", {}), key=lambda k: (1, 1, True))
    ns = {"h_good": hook("good"), "h_same": hook("same"), "h_price": hook("price"), "h_recipe": lambda fn: recipe}
    return state, ns, seen


def spec(**kw):
    base = dict(name="t", shapes=["s1", "s2"], recipe="callback", recipe_hook="h_recipe", rounds=4, bar=1.01,
                arms=[dict(name="good", apply="h_good"), dict(name="same", apply="h_same"),
                      dict(name="price", apply="h_price", kind="pricing")],
                predeclared=dict(shape="s1", arm="good", metric="saved_us", op="<", value=20.0, text="saved<20 at s1"))
    return dict(base, **kw)


def run(fake, sp, monkeypatch, lines):
    state, ns, seen = fake
    monkeypatch.setattr(engine, "Emitter", lambda *a, **k: Emitter(lines.append))
    engine.ck_run(state, time.time(), lambda d, s: (), sp, dict(name="t", spec_sha="s", predeclared=sp["predeclared"], git="x"), ns)
    return report.extract("\n".join(lines))


def test_matrix_rules_and_final_line(fake, monkeypatch):
    lines = []
    res = run(fake, spec(), monkeypatch, lines)
    assert lines[-1].startswith("CK_RESULT ") and lines[-2].startswith("CK_TABLE")
    cell = res["shapes"][0]
    arms = {a["name"]: a for a in cell["arms"]}
    assert arms["same"]["status"] == "fallback"                       # launches equal the entry's: never timed
    assert arms["good"]["status"] == "valid" and arms["good"]["saved_us"] == pytest.approx(100 * (1 - 0.9), rel=0.1)
    assert "ratio" in arms["price"] and cell["pick"] == "good"        # pricing arm timed; pick is the real arm
    assert cell["nsets"] == 1 and cell["note"].startswith("pricing")  # one input set forced by the pricing arm
    assert res["kill"]["evaluated"] == "KILL" and res["verdict"]["kept"] == {}
    assert [c["base_us"] for c in res["shapes"]] == [100.0, 100.0]
    assert [c["pick"] for c in res["shapes"]] == ["good", "good"]
    assert res["verdict"]["measurement_only"] is True and fake[0]["variant"] == "entry"
    gates = {g["name"]: g for g in res["gates"]}
    assert gates["kernels"]["pass"] is False and gates["clone_state"]["pass"] is True
    assert res["calib"]["ovh_us"] == 4.0 and res["env"]["gpu"] == "FAKE"
    assert not any(line.startswith("CK_RAW ") for line in lines)


@pytest.mark.parametrize("rounds", [4, 24])
def test_raw_result_reconstructs_paired_ratios_and_round_context(fake, monkeypatch, rounds):
    from stats import pair_ratios

    times = dict(base=[104.0123456789, 105.0234567891, 106.0345678912, 107.0456789123],
                 twin=[104.0234567891, 105.0345678912, 106.0456789123, 107.0567891234],
                 good=[94.0345678912, 95.0456789123, 96.0567891234, 97.0678912345])
    times = {name: samples * (rounds // 4) for name, samples in times.items()}
    overhead = 4.0123456789
    monkeypatch.setattr(engine, "time_rounds", lambda *a, **k: {n: list(v) for n, v in times.items()})
    monkeypatch.setattr(engine, "calibrate", lambda f: dict(
        flush_mib=256, ovh_us=overhead, ovh_ungated_us=overhead, flags=[]))
    lines = []
    res = run(fake, spec(raw=True, rounds=rounds, shapes=["s1"], arms=[dict(name="good", apply="h_good")]),
              monkeypatch, lines)
    raw_lines = [line for line in lines if line.startswith("CK_RAW ")]
    assert len(raw_lines) == 1  # One complete cell, never unrelated per-arm fragments.
    raw = json.loads(raw_lines[0].removeprefix("CK_RAW "))
    assert raw["times_us"] == times and raw["overhead_us"] == overhead
    assert raw["key"] == "s1" and raw["pass"] == "main" and raw["seed"] == 0x58524159
    assert raw["nsets"] == 3 and raw["input_indices"] == [r % 3 for r in range(rounds)]
    assert raw["names"] == ["base", "twin", "good"]
    assert len(raw["order_indices"]) == rounds
    assert raw["order_indices"][:4] == [[0, 1, 2], [2, 1, 0], [1, 2, 0], [0, 2, 1]]
    cell, = res["shapes"]
    for name in ("twin", "good"):
        reconstructed = pair_ratios(raw["times_us"]["base"], raw["times_us"][name], raw["overhead_us"])
        assert reconstructed == pair_ratios(times["base"], times[name], overhead)
        arm = next(a for a in cell["arms"] if a["name"] == name)
        assert arm["ratio"] == round(statistics.median(reconstructed), 3)
    assert "raw" not in cell and all("pairs" not in a for a in cell["arms"])
    assert lines[-1].startswith("CK_RESULT ") and len("\n".join(lines)) < 16000


def test_raw_budget_drops_whole_cell_and_discloses_omission(fake, monkeypatch):
    lines = []
    res = run(fake, spec(raw=True, rounds=1000, shapes=["s1"],
                        arms=[dict(name="good", apply="h_good")]), monkeypatch, lines)
    assert not any(line.startswith("CK_RAW ") for line in lines)
    assert "RAW_TRUNCATED" in res["verdict"]["flags"]
    assert res["shapes"][0]["pick"] == "good"
    assert lines[-1].startswith("CK_RESULT ") and len("\n".join(lines)) < 16000


def test_global_hook_winners_do_not_overwrite_other_shapes(fake, monkeypatch):
    sp = spec(arms=[dict(name="good", apply="h_good", shapes=["s1"]),
                    dict(name="price", apply="h_price", shapes=["s2"])])
    res = run(fake, sp, monkeypatch, [])
    assert [c["pick"] for c in res["shapes"]] == ["good", "price"]
    assert [c["base_us"] for c in res["shapes"]] == [100.0, 100.0]
    assert res["verdict"]["kept"] == {} and fake[0]["variant"] == "entry"


def test_budget_skips_are_listed_and_result_still_prints(fake, monkeypatch):
    lines = []
    res = run(fake, spec(stop_s=-1.0), monkeypatch, lines)
    assert res["shapes"] == [] and len(res["verdict"]["skipped"]) >= 2
    assert res["kill"]["evaluated"] == "INVALID_AA"


@pytest.mark.parametrize("stage", ["capture good", "timing twin round 0"])
def test_bounded_sweep_never_ranks_an_interrupted_cell(fake, monkeypatch, stage):
    original = engine.Budget

    class Deadline(original):
        def require(self, label, near=False):
            if label == stage:
                self.skipped.append(label)
                raise engine.BudgetExpired(label)
            super().require(label, near)

    monkeypatch.setattr(engine, "Budget", Deadline)
    attempted = []

    def rounds(caps, refresh, poison_fn, flush, ovh, rounds=24, budget=None):
        for name in caps:
            budget("timing %s round 0" % name)
            attempted.append(name)
        pytest.fail("must stop before finishing a round")

    monkeypatch.setattr(engine, "time_rounds", rounds)
    lines = []
    res = run(fake, spec(bounded=True, arms=[dict(name="good", apply="h_good")]), monkeypatch, lines)
    cell, = res["shapes"]
    assert cell["pick"] is None and not cell["aa"]["valid"]
    assert all("ratio" not in arm for arm in cell["arms"])
    assert res["verdict"]["kept"] == {}
    assert "BUDGET_EXHAUSTED" in res["verdict"]["flags"]
    assert attempted == ([] if stage == "capture good" else ["base"])
    assert lines[-1].startswith("CK_RESULT ") and report.evaluate(res)[0] == "INVALID"


def test_entry_off_measures_baseline_and_restores_original_candidate(fake, monkeypatch):
    state, ns, _ = fake
    state["variant"] = "good"

    def entry_off(st):
        old = st["variant"]
        st["variant"] = "entry"
        return lambda: st.update(variant=old)

    ns["h_entry"] = entry_off
    res = run(fake, spec(entry_off="h_entry", arms=[dict(name="good", apply="h_good")]),
              monkeypatch, [])
    assert [c["base_us"] for c in res["shapes"]] == [100.0, 100.0]
    assert res["verdict"]["kept"] == {} and state["variant"] == "good"
    for cell in res["shapes"]:
        candidate = next(a for a in cell["arms"] if a["name"] == "good")
        assert candidate["status"] == "valid"
        assert candidate["ratio"] == pytest.approx(100 / 90, abs=0.001)


def test_exception_in_a_cell_is_recorded_not_fatal(fake, monkeypatch):
    state, ns, _ = fake
    ns["h_good"] = lambda st, key: 1 / 0
    lines = []
    res = run((state, ns, None), spec(arms=[dict(name="good", apply="h_good")]), monkeypatch, lines)
    arm = {a["name"]: a for a in res["shapes"][0]["arms"]}["good"]
    assert arm["status"] == "raised" or "RAISED" in " ".join(res["verdict"]["flags"])


def test_compact_result_keeps_rejected_arm_reason(fake, monkeypatch):
    state, ns, _ = fake

    def rejected(st, key):
        raise ValueError("rejection detail" + "x" * 200)

    ns["h_good"] = rejected
    original = engine.trim
    calls = []

    def trim_with_budget_pressure(res):
        calls.append(None)
        if len(calls) == 1:
            # Perform normal note reduction before exercising the compact backstop.
            original(res, limit=1)
        return original(res)

    monkeypatch.setattr(engine, "trim", trim_with_budget_pressure)
    res = run(fake, spec(arms=[dict(name="good", apply="h_good")]), monkeypatch, [])
    rejected_arm = next(a for a in res["shapes"][0]["arms"] if a["name"] == "good")
    assert rejected_arm.get("note", "").startswith("ValueError('rejection detail"), rejected_arm
    assert len(rejected_arm["note"]) <= 96
    assert "TRIM_COMPACT" in res["verdict"]["flags"]


@pytest.mark.parametrize("stage", ["validation", "after_timing"])
def test_validation_exception_rejects_only_failed_arm(fake, monkeypatch, stage):
    seen = []

    def validate(cap, *args):
        if cap.variant == "good":
            seen.append(cap)
            if stage == "validation" or len(seen) == 2:
                raise RuntimeError("plan self-test failed")
        return True, "ok", {"routes": {"sum": 1}}

    monkeypatch.setattr(engine, "replay_valid", validate)
    lines = []
    res = run(fake, spec(raw=True, shapes=["s1"], arms=[dict(name="good", apply="h_good"),
                               dict(name="price", apply="h_price")]), monkeypatch, lines)
    assert len(res["shapes"]) == 1, res
    raw, = [json.loads(line.removeprefix("CK_RAW ")) for line in lines if line.startswith("CK_RAW ")]
    assert ("good" in raw["times_us"]) == (stage == "after_timing")
    if stage == "after_timing":
        assert raw["statuses"]["good"] == "invalid_after_timing"
    for cell in res["shapes"]:
        arms = {a["name"]: a for a in cell["arms"]}
        assert arms["good"]["status"] == ("raised" if stage == "validation" else "invalid_after_timing")
        assert "plan self-test failed" in arms["good"]["note"]
        assert "ratio" not in arms["good"]
        assert arms["price"]["status"] == "valid" and "ratio" in arms["price"]
    assert res["shapes"][0]["pick"] == "price" and res["verdict"]["kept"] == {}


@pytest.mark.parametrize("failed_shape", ["s1", "s2"])
def test_sticky_capture_fault_stops_sweep_and_keeps_partial_log(fake, monkeypatch, failed_shape):
    state, ns, _ = fake
    dead, attempted = [], []
    original_apply = ns["h_good"]

    def apply(st, key):
        st["shape"] = key
        return original_apply(st, key)

    ns["h_good"] = apply

    def sync():
        if dead:
            raise RuntimeError("CUDA illegal memory access")

    original = engine.Capture

    def capture(fn):
        attempted.append(state["variant"])
        if state["variant"] == "good" and state["shape"] == failed_shape:
            dead.append(True)
            raise RuntimeError("launch failed")
        return original(fn)

    monkeypatch.setattr(sys.modules["torch"].cuda, "synchronize", sync)
    monkeypatch.setattr(engine, "Capture", capture)
    lines = []
    with pytest.raises(RuntimeError, match="CUDA"):
        run(fake, spec(arms=[dict(name="good", apply="h_good"),
                            dict(name="price", apply="h_price")]), monkeypatch, lines)
    assert attempted == (["entry", "entry", "good"] if failed_shape == "s1" else
                         ["entry", "entry", "good", "price", "entry", "entry", "good"])
    assert state["variant"] == "entry"
    assert lines[-1].startswith("CK_RESULT ")
    res = report.extract("\n".join(lines))
    assert len(res["shapes"]) == (1 if failed_shape == "s1" else 2)
    if failed_shape == "s2":
        assert res["shapes"][0]["pick"] == "price"
        assert res["verdict"]["kept"] == {}
    cell = res["shapes"][-1]
    assert cell["pick"] is None and not cell["aa"]["valid"]
    failed = next(a for a in cell["arms"] if a["name"] == "good")
    assert failed["status"] == "raised" and "launch failed" in failed["note"]
    assert "CUDA_CONTEXT_UNUSABLE" in res["verdict"]["flags"]


def test_failed_control_has_no_pick_and_still_emits_result(fake, monkeypatch):
    def validate(cap, *args):
        if cap.variant == "entry":
            raise RuntimeError("control self-test failed")
        return True, "ok", {"routes": {"sum": 1}}

    monkeypatch.setattr(engine, "replay_valid", validate)
    lines = []
    res = run(fake, spec(shapes=["s1"], arms=[dict(name="good", apply="h_good")]), monkeypatch, lines)
    cell, = res["shapes"]
    assert cell["base_us"] is None and cell["pick"] is None and not cell["aa"]["valid"]
    assert res["verdict"]["kept"] == {}
    assert lines[-1].startswith("CK_RESULT ") and "AA-INVALID" in lines[-2]


@pytest.mark.parametrize("stage", ["arm", "reference"])
def test_failed_plan_restore_stops_before_other_arms(fake, monkeypatch, stage):
    state, ns, _ = fake
    attempted = []
    original = engine.Capture

    def capture(fn):
        attempted.append(state["variant"])
        return original(fn)

    def broken(st, key):
        st["variant"] = "good"

        def undo():
            raise RuntimeError("cannot restore original plan")
        return undo

    ns["h_good"] = broken
    ns["h_entry"] = lambda st: broken(st, None)
    monkeypatch.setattr(engine, "Capture", capture)
    lines = []
    with pytest.raises(RuntimeError, match="restore"):
        run(fake, spec(entry_off="h_entry" if stage == "reference" else None,
                       arms=[dict(name="good", apply="h_good"),
                            dict(name="price", apply="h_price")]), monkeypatch, lines)
    assert attempted == (["entry", "entry", "good"] if stage == "arm" else [])
    res = report.extract("\n".join(lines))
    assert res["verdict"]["kept"] == {} and "PLAN_RESTORE_FAILED" in res["verdict"]["flags"]
    if stage == "arm":
        assert "cannot restore" in res["shapes"][0]["arms"][-1]["note"]


def test_post_step_runs_last_and_cannot_break_the_result(fake, monkeypatch):
    state, ns, _ = fake
    seen = []

    def post(st, ctx):
        seen.append(sorted(ctx))
        ctx["emit"]("post line")
    ns["h_post"] = post
    lines = []
    res = run(fake, spec(post="h_post"), monkeypatch, lines)
    assert seen == [["device", "emit", "flush", "kernel_fn", "recipe"]]
    gate = max(i for i, l in enumerate(lines) if l.startswith("CK gate"))
    assert lines.index("CK post line") > gate and lines[-1].startswith("CK_RESULT ")
    assert res["kill"]["evaluated"] == "KILL"
    ns["h_post"] = lambda st, ctx: 1 / 0
    lines = []
    res = run(fake, spec(post="h_post"), monkeypatch, lines)
    assert any(l.startswith("CK post h_post raised ZeroDivisionError") for l in lines)
    assert lines[-1].startswith("CK_RESULT ") and "RAISED" not in res["verdict"]["flags"]


def test_probe_arm_is_timed_never_validated_never_picked(fake, monkeypatch):
    state, ns, seen = fake
    ns["h_probe"] = ns["h_price"]
    lines = []
    sp = spec(arms=[dict(name="good", apply="h_good"), dict(name="probe", apply="h_probe", kind="probe")],
              predeclared=dict(shape="s1", arm="probe", metric="saved_us", op="<", value=1e9, text="probe timed"))
    res = run((state, ns, seen), sp, monkeypatch, lines)
    arms = {a["name"]: a for a in res["shapes"][0]["arms"]}
    assert arms["probe"]["status"] == "probe" and "ratio" in arms["probe"]
    assert res["shapes"][0]["pick"] == "good"                          # a probe is never a candidate
    assert res["kill"]["evaluated"] == "KILL"                          # a rule may name the probe explicitly


def test_soak_result_discloses_sampling(fake, monkeypatch):
    import gpu

    monkeypatch.setattr(gpu, "poison", lambda out: None)
    checks = []
    recipe = fake[1]["h_recipe"](None)
    recipe._check = lambda *args: (checks.append(None) or True, "ok", {})
    res = run(fake, spec(soak=2000), monkeypatch, [])
    gate = next(g for g in res["gates"] if g["name"] == "clone_state")
    assert len(checks) == 20
    assert gate.get("soak") == "sampled 1/100", gate
