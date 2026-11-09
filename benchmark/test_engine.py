import pytest

import engine
from engine import Arm, cell_record, ck_plan_apply, ck_entry_off


def recs(*names):
    return [dict(name=n, kind="real", status="valid") for n in ("base", "twin", *names)]


def test_cell_record_picks_and_prices():
    base = [100.0, 100.0] * 4
    times = dict(base=base, twin=base, win=[90.0] * 8, price=[70.0] * 8)
    cell, pick = cell_record("d1s", 1, 3, times, dict(win="real", price="pricing", twin="control"), recs("win", "price"), 0.0, raw=True)
    assert pick == "win" and cell["pick"] == "win" and cell["aa"]["valid"]
    arms = {a["name"]: a for a in cell["arms"]}
    assert "base" not in arms and arms["win"]["saved_us"] == pytest.approx(100 * (1 - 0.9))
    assert arms["price"]["ratio"] > arms["win"]["ratio"]
    assert cell["raw"]["times_us"] == times and cell["raw"]["overhead_us"] == 0.0


def test_cell_without_twin_is_unusable_not_a_pick():
    cell, pick = cell_record("d1s", 1, 1, dict(base=[100.0] * 4), {}, recs(), 0.0)
    assert pick is None and not cell["aa"]["valid"]


def test_plan_apply_restores():
    state = {"swap": {(1, 1, True): ("old",)}}
    undo = ck_plan_apply((1, 1, True), ("a", "b"))(state)
    assert state["swap"][(1, 1, True)] == ("a", "b")
    undo()
    assert state["swap"] == {(1, 1, True): ("old",)}
    undo = ck_plan_apply((2, 1, True), ("a",))(state)
    undo()
    assert (2, 1, True) not in state["swap"]


def test_entry_off_restores():
    state = {}
    undo = ck_entry_off(state)
    assert state["swap_off"] is True
    undo()
    assert "swap_off" not in state


def test_arm_defaults_make_real_arms_need_kernel_evidence():
    a = Arm("x", engine.ck_noop)
    assert a.kind == "real" and a.expect == () and not a.fallback_ok and not a.single_input
