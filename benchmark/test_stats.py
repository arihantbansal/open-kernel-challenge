import math

import pytest

from stats import decide, eval_kill, near_verdict, overhead_from_times, pair_ratios, round_order, summarize


def test_pairs_subtract_overhead_before_geomean():
    assert pair_ratios([14, 44, 34, 14], [9, 14, 14, 14], 4) == pytest.approx([math.sqrt(8), math.sqrt(3)])


@pytest.mark.parametrize("base,other,ovh", [([4, 8], [8, 8], 4), ([8, 8], [8, 8], -1), ([8, 8], [8], 0),
                                            ([8, 8], [8, float("nan")], 0), ([8, 8, 8], [8, 8, 8], 0)])
def test_unusable_timings_raise(base, other, ovh):
    with pytest.raises(ValueError):
        pair_ratios(base, other, ovh)


def test_summary_and_aa_validity():
    t = dict(base=[14, 44, 34, 14], twin=[14, 44, 34, 14], a=[9, 14, 14, 14])
    s = summarize(t, 4)
    assert s["valid"] and s["band"] == 0 and s["base_us"] == pytest.approx(24 - 4 + 0)  # median 24
    t["twin"] = [24, 54, 44, 24]
    assert not summarize(t, 4)["valid"]


def test_decide_bar_and_kinds():
    base = [100.0] * 8
    t = dict(base=base, twin=base, win=[90.0] * 8, price=[80.0] * 8, tiny=[99.5] * 8)
    pick, ratios, aa, band, valid = decide(t, dict(win="real", price="pricing", tiny="real"), 0)
    assert pick == "win" and valid and ratios["price"] > ratios["win"] > 1
    assert decide(dict(base=base, twin=base, tiny=[99.5] * 8), dict(tiny="real"), 0)[0] is None  # under the 1.01 bar
    noisy = dict(base=base, twin=[100, 110] * 4, win=[90.0] * 8)
    assert decide(noisy, dict(win="real"), 0)[0] is None  # A/A band 0.1 swallows a 1.11 ratio? bar = 1.1
    assert decide(dict(base=base, twin=[130.0] * 8, win=[90.0] * 8), dict(win="real"), 0)[0] is None  # A/A invalid


def test_near_verdict():
    assert near_verdict(0.985, 0.0) and not near_verdict(0.97, 0.0)
    assert near_verdict(0.97, 0.02) and not near_verdict(0.95, 0.02)


def test_overhead_bounds():
    assert overhead_from_times([5, 5, 5], [6, 6, 6]) == 4
    with pytest.raises(ValueError):
        overhead_from_times([5], [20])
    with pytest.raises(ValueError):
        overhead_from_times([50], [20])


def test_round_order_is_position_balanced():
    names = ["a", "b", "c"]
    pos = {n: [0, 0, 0] for n in names}
    for r in range(6):
        for i, n in enumerate(round_order(names, r)):
            pos[n][i] += 1
    assert all(sorted(v) == [2, 2, 2] for v in pos.values())
    assert round_order(["a", "b"], 0) == ["a", "b"] and round_order(["a", "b"], 1) == ["b", "a"]


def test_eval_kill():
    shapes = [dict(key="d1s", aa=dict(valid=True), arms=[dict(name="x", status="valid", saved_us=3.0, ratio=0.99)])]
    rule = dict(shape="d1s", arm="x", metric="saved_us", op="<", value=4.0)
    assert eval_kill(rule, shapes) == "KILL"
    assert eval_kill(dict(rule, value=2.0), shapes) == "PASS"
    assert eval_kill(dict(rule, arm="y"), shapes) == "INVALID"
    assert eval_kill(dict(rule, shape="d2s"), shapes) == "INVALID_AA"
    shapes[0]["aa"]["valid"] = False
    assert eval_kill(rule, shapes) == "INVALID_AA"


def test_eval_kill_best_arm():
    arms = [dict(name="twin", status="valid", saved_us=9.0), dict(name="a", status="valid", saved_us=1.0),
            dict(name="b", status="valid", saved_us=5.0), dict(name="c", status="fallback"),
            dict(name="p", kind="control", status="valid", saved_us=8.0)]
    shapes = [dict(key="d32s", aa=dict(valid=True), arms=arms)]
    rule = dict(shape="d32s", arm="best", metric="saved_us", op="<", value=4.0)
    assert eval_kill(rule, shapes) == "PASS"  # b: the twin and control arms never count
    assert eval_kill(dict(rule, value=6.0), shapes) == "KILL"
    arms[2]["status"] = "dropped"
    assert eval_kill(rule, shapes) == "KILL"  # a is now the best valid arm
    arms[1]["status"] = "raised"
    assert eval_kill(rule, shapes) == "INVALID"
