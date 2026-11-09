"""Exercise the real layer comparator and replay seam on small CPU inputs."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from gpu import replay_valid, soak
from recipes.layer import LayerRecipe


@pytest.fixture(autouse=True)
def cpu_sync(monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)


def source(rows=128):
    return dict(stream=torch.ones(rows, 8), attn_input=torch.ones(rows, 8, dtype=torch.bfloat16),
                k_cache=torch.zeros(6, 1, 2, 4, dtype=torch.float8_e4m3fn),
                v_cache=torch.zeros(6, 1, 2, 4, dtype=torch.float8_e4m3fn),
                cache_loc=torch.tensor([1, 4]), local_start=torch.tensor([0]),
                positions=torch.arange(rows), config=dict(batch=rows, q_len=1))


STATE = dict(attnb_alpha=1.0, ffnb_alpha=1.0)


def valid_kernel(data, state):
    slots = data["cache_loc"].long()
    for name in ("k_cache", "v_cache"):
        data[name].view(torch.uint8).reshape(6, -1)[slots] = 56
    return (torch.ones_like(data["attn_input"]), data["stream"] + 1,
            torch.zeros(data["stream"].shape[0], 8, dtype=torch.long))


def compared(change=lambda out, live: None):
    recipe, src = LayerRecipe(valid_kernel), source()
    ref = recipe.reference(src, STATE)
    live = recipe.live(src)
    recipe.refresh(live, src)
    out = valid_kernel(live, STATE)
    change(out, live)
    return recipe.check(out, live, ref, STATE)


def test_valid_path_and_pristine_reference_inputs():
    src, recipe = source(), LayerRecipe(valid_kernel)
    before = deepcopy(src)
    ref = recipe.reference(src, STATE)
    assert torch.equal(src["k_cache"].view(torch.uint8), before["k_cache"].view(torch.uint8))
    live = recipe.live(src)
    assert live["config"] is not src["config"]
    recipe.refresh(live, src)
    assert recipe.check(valid_kernel(live, STATE), live, ref, STATE)[0]


def test_one_bad_row_fails_despite_small_aggregate_error():
    def change(out, live):
        out[0][0].add_(0.08)
    ok, note, info = compared(change)
    assert not ok, "one bad row was accepted"
    assert "row" in note and info["rel"] < 1e-2


@pytest.mark.parametrize("name,slot", [("k_cache", 1), ("v_cache", 4), ("k_cache", 0), ("v_cache", 5)])
def test_cache_target_and_unrelated_bytes_are_exact(name, slot):
    def change(out, live):
        live[name].view(torch.uint8)[slot, 0, 0, 0] ^= 1
    ok, note, _ = compared(change)
    assert not ok, "cache corruption was accepted"
    assert name in note


@pytest.mark.parametrize("name", ["stream", "attn_input", "positions", "cache_loc", "local_start", "config"])
def test_readonly_input_mutation_fails(name):
    def change(out, live):
        if name == "config":
            live[name]["batch"] += 1
        else:
            live[name].flatten()[0] += 1
    ok, note, _ = compared(change)
    assert not ok, "readonly input mutation was accepted"
    assert name in note


@pytest.mark.parametrize("slot,kind", [(0, "shape"), (0, "dtype"), (0, "finite"),
                                       (1, "shape"), (1, "dtype"), (1, "finite"),
                                       (2, "shape"), (2, "dtype")])
def test_output_contract(slot, kind):
    recipe, src = LayerRecipe(valid_kernel), source()
    ref = recipe.reference(src, STATE)
    live = recipe.live(src)
    out = list(valid_kernel(live, STATE))
    if kind == "shape":
        out[slot] = out[slot][:-1]
    elif kind == "dtype":
        out[slot] = out[slot].double()
    else:
        out[slot].flatten()[0] = float("nan")
    ok, note, _ = recipe.check(tuple(out), live, ref, STATE)
    assert not ok
    assert kind in note or kind == "finite" and "NaN" in note


@pytest.mark.parametrize("runner", ["replay", "soak"])
def test_replay_checks_actual_live_mutation(runner):
    recipe, src = LayerRecipe(valid_kernel), source()
    ref, live = recipe.reference(src, STATE), recipe.live(src)
    cap = SimpleNamespace(out=valid_kernel(live, STATE))
    def replay():
        fresh = valid_kernel(live, STATE)
        for dst, value in zip(cap.out, fresh):
            dst.copy_(value)
        live["positions"][0] += 1
    cap.replay = replay
    if runner == "replay":
        ok, note, _ = replay_valid(cap, live, [src], [ref], recipe, STATE)
        assert not ok, "replay validated pristine source instead of mutated live input"
        assert "positions" in note
    else:
        with pytest.raises(RuntimeError, match="positions"):
            soak(cap, "test", live, [src], [ref], recipe, STATE, n=1)


@pytest.mark.parametrize("bad_input", [0, 1, 2])
def test_replay_valid_checks_every_timed_input(bad_input):
    recipe, sets = LayerRecipe(valid_kernel), [source() for _ in range(3)]
    for i, src in enumerate(sets):
        src["positions"][0] = i
    refs = [recipe.reference(src, STATE) for src in sets]
    live = recipe.live(sets[0])
    cap = SimpleNamespace(out=valid_kernel(live, STATE))

    def replay():
        for dst, value in zip(cap.out, valid_kernel(live, STATE)):
            dst.copy_(value)
        if live["positions"][0] == bad_input:
            cap.out[0][0].add_(1)

    cap.replay = replay
    ok, note, _ = replay_valid(cap, live, sets, refs, recipe, STATE)
    assert not ok, "corrupt timed input %d was accepted" % bad_input
    assert "input %d" % bad_input in note


def test_fresh_refresh_poisons_only_allowed_cache_writes():
    recipe, src = LayerRecipe(valid_kernel), source()
    ref, live = recipe.reference(src, STATE), recipe.live(src)
    recipe.refresh(live, src)
    assert torch.equal(live["k_cache"].view(torch.uint8)[0], src["k_cache"].view(torch.uint8)[0])
    # A graph returning the right output while omitting cache writes must fail.
    out = (ref["out"].clone(), src["stream"] + 1, ref["ids"].clone())
    ok, note, _ = recipe.check(out, live, ref, STATE)
    assert not ok, "missing new-KV writes were accepted"
    assert "cache" in note


def test_baseline_readonly_mutation_is_rejected():
    def bad(data, state):
        out = valid_kernel(data, state)
        data["positions"][0] += 1
        return out
    with pytest.raises(ValueError, match="positions"):
        LayerRecipe(bad).reference(source(), STATE)


def test_baseline_unrelated_cache_mutation_is_rejected():
    def bad(data, state):
        out = valid_kernel(data, state)
        data["k_cache"].view(torch.uint8)[0, 0, 0, 0] = 1
        return out
    with pytest.raises(ValueError, match="outside cache_loc"):
        LayerRecipe(bad).reference(source(), STATE)


@pytest.mark.parametrize("slot", [0, 1])
def test_nonfinite_incumbent_cannot_define_an_agreement_bound(slot):
    def bad(data, state):
        out = valid_kernel(data, state)
        out[slot][0, 0] = float("nan")
        return out
    with pytest.raises(ValueError, match="incumbent output contains NaN"):
        LayerRecipe(bad).reference(source(), STATE)


@pytest.mark.parametrize("slot", [0, 1])
def test_worst_row_applies_to_output_and_residual_update(slot):
    def change(out, live):
        out[slot][0].add_(0.08)
    ok, note, info = compared(change)
    assert not ok and "worst row" in note
    assert info["rel"] < 1e-2 and info["worst_row"] > 5e-2


@pytest.mark.parametrize("error,accepted", [(0.004, True), (0.006, False)])
def test_near_zero_rows_use_fraction_of_mean_row_norm(error, accepted):
    def kernel(data, state):
        out = valid_kernel(data, state)
        out[0][0].zero_()
        return out
    recipe, src = LayerRecipe(kernel), source()
    ref, live = recipe.reference(src, STATE), recipe.live(src)
    out = kernel(live, STATE)
    out[0][0].fill_(error)
    ok, _, _ = recipe.check(out, live, ref, STATE)
    assert ok is accepted


def test_strict_per_expert_counts_remain_required():
    def change(out, live):
        out[2][0, 0] = 1
    ok, note, _ = compared(change)
    assert not ok and "routed counts differ" in note


@pytest.mark.parametrize("corruption", [None, "cache", "input"])
def test_assembled_recipe_and_replay_validate_live_integrity(corruption):
    import assemble
    from test_assemble import ENTRY, SPEC

    text, _ = assemble.assemble_text(ENTRY, assemble.sha(ENTRY), SPEC)
    ns = {"__name__": "layer_integrity_carrier"}
    exec(compile(text, "layer_integrity_carrier", "exec"), ns)
    recipe, src = ns["LayerRecipe"](valid_kernel), source()
    ref, live = recipe.reference(src, STATE), recipe.live(src)
    cap = SimpleNamespace(out=valid_kernel(live, STATE))
    def replay():
        for dst, value in zip(cap.out, valid_kernel(live, STATE)):
            dst.copy_(value)
        if corruption == "cache":
            live["v_cache"].view(torch.uint8)[5, 0, 0, 0] = 1
        elif corruption == "input":
            live["attn_input"][0, 0] += 1
    cap.replay = replay
    ok, note, _ = ns["replay_valid"](cap, live, [src], [ref], recipe, STATE)
    assert ok is (corruption is None), note
