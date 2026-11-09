"""The check engine: one same-run comparison (`compare`) and the matrix around it (`ck_run`).

Rules enforced here, not by convention: a pricing arm (or single_input=True) forces one input set; every arm is
captured against `state` and fallback is judged from executed kernel names; every timed replay is preceded by a
poison of outputs and declared scratch; the stale-snapshot keepalive is owned by one Retain; picks are re-validated
on a judge-style clone of state; one Emitter owns the log, one Budget owns prepare time; calibration is recorded.
"""

import os
import statistics
from dataclasses import dataclass

from budget import Budget, BudgetExpired, Emitter, priority_order
from xray import xray_shape
from gpu import (Capture, CudaContextError, Flush, Retain, calibrate, check_cuda_health, clocks, clone_state,
                 expect_kernels, kernels_of, poison, replay_valid, soak, time_rounds)
from recipes.attn import CallbackRecipe
from recipes.layer import LayerRecipe
from schema import new_result, table_rows, trim, validate
from stats import decide, eval_kill, near_verdict, round_order


@dataclass
class Arm:
    name: str
    apply: object  # apply(state) -> undo
    # "real" (may be picked), "pricing" (timed, never picked), "control", "probe" (timing-only: an instrumented
    # build whose output is knowingly wrong, e.g. decode removed; timed against the same base, never validated, never
    # picked, never a candidate)
    kind: str = "real"
    expect: tuple = ()  # regexes that must match executed kernel names; absent -> status "fallback"
    single_input: bool = False
    retain: tuple = ()  # callables (state) -> tensors an older graph may still read
    scratch: object = None  # callable (state) -> tensors poisoned before every replay
    fallback_ok: bool = False  # nearby gate: the arm may legitimately run the entry's own path


class PlanRestoreError(RuntimeError):
    """The temporary plan could not be undone; later arms cannot trust state."""


def ck_restore(undo, name):
    try:
        undo()
    except Exception as exc:
        raise PlanRestoreError("plan restore %s: %s" % (name, repr(exc)[:100])) from exc


def ck_noop(state):
    return lambda: None


def ck_entry_off(state):
    """The entry with the candidate switched off (swap components' `swap_off` switch)."""
    old = state.get("swap_off")
    state["swap_off"] = True

    def undo():
        if old is None:
            state.pop("swap_off", None)
        else:
            state["swap_off"] = old
    return undo


def ck_plan_apply(k, names):
    def apply(state):
        plan = state.setdefault("swap", {})
        old = plan.get(k)
        plan[k] = tuple(names)

        def undo():
            if old is None:
                plan.pop(k, None)
            else:
                plan[k] = old
        return undo
    return apply


def cell_record(label, tokens, nsets, times, kinds, recs, ovh, raw=False, bar=1.01):
    """Pure: the shape cell from timed rounds. `recs` are the arm records so far (status, note, ...); every arm that
    was timed gets ratio and saved_us. Opt-in raw samples are emitted separately before CK_RESULT rounding.
    -> (cell, pick)."""
    cell = dict(key=label, tokens=tokens, nsets=nsets, pick=None)
    if raw:
        cell["raw"] = dict(times_us=dict(times), overhead_us=ovh, nsets=nsets)
    if "base" not in times or "twin" not in times:
        cell.update(aa=dict(valid=False), arms=recs, base_us=None, note="base or twin unusable")
        return cell, None
    pick, ratios, aa, band, valid = decide(times, kinds, ovh, bar)
    base_us = statistics.median(times["base"]) - ovh
    cell.update(base_us=base_us, aa=dict(median=aa, band=band, valid=valid), pick=pick, arms=recs)
    for r in recs:
        if r["name"] in ratios:
            r["ratio"] = ratios[r["name"]]
            r["saved_us"] = base_us * (1 - 1 / ratios[r["name"]])
    cell["arms"] = [r for r in recs if r["name"] != "base"]
    return cell, pick


def compare(label, case, arms, *, recipe, kernel_fn, state, flush, ovh, rounds, device, seed, retain,
            base=None, nsets=3, raw=False, bar=1.01, counts=True, budget=None):
    """Time the entry (`base` applied), its A/A twin and each Arm on one machine in one run.
    -> (cell, pick name or None). Arms that fail validation, raise, or run the entry's kernels never get timed.
    An unusable CUDA context aborts with the incomplete cell attached for logging."""
    import torch
    base = base or ck_noop
    forced = any(a.kind == "pricing" or a.single_input for a in arms)
    if forced:
        nsets = 1  # equal work: a pricing arm snapshots the entry's output of one input
    if budget:
        budget("inputs " + label)
    undo = base(state)
    try:
        sets = recipe.sets(case, seed, device, state, nsets)
        refs = []
        for d in sets:
            if budget:
                budget("reference " + label)
            refs.append(recipe.reference(d, state))
    finally:
        ck_restore(undo, "reference")
    assert not forced or len(sets) == 1
    live = recipe.live(sets[0])
    caps, kinds, recs, scratch = {}, {}, [], {}

    def recover(exc):
        if isinstance(exc, BudgetExpired):
            exc.cell, _ = cell_record(label, case["batch"] * case.get("q_len", 1), len(sets), {}, {}, recs, ovh)
            exc.cell["note"] = "budget exhausted; timing incomplete"
            raise exc
        try:
            check_cuda_health(exc)
            if isinstance(exc, PlanRestoreError):
                raise exc
        except (CudaContextError, PlanRestoreError) as fault:
            fault.cell, _ = cell_record(label, case["batch"] * case.get("q_len", 1), len(sets), {}, {}, recs, ovh)
            fault.cell["note"] = "sweep aborted; timing incomplete"
            raise

    base_names = None
    for a in [Arm("base", base, "control"), Arm("twin", base, "control"), *arms]:
        rec = dict(name=a.name, kind=a.kind, status="raised")
        recs.append(rec)
        undo, cap = None, None
        try:
            try:
                if budget:
                    budget("capture " + a.name)
                undo = a.apply(state)
                recipe.refresh(live, sets[0])
                cap = Capture(lambda: kernel_fn(live, state))
                if budget:
                    budget("profile " + a.name)
                names = kernels_of(cap.replay)
                for r in a.retain:
                    retain.keep(*r(state))
                if a.scratch:
                    scratch[a.name] = a.scratch(state)
            finally:
                if undo is not None:
                    ck_restore(undo, a.name)
            if a.name == "base":
                base_names = names
            if a.expect:
                seen = expect_kernels(names, a.expect)
            elif a.kind == "control" or a.fallback_ok or base_names is None:
                seen = base_names is None or a.fallback_ok or sorted(names) == sorted(base_names)
            else:
                seen = sorted(names) != sorted(base_names)  # equal launches mean the entry ran, not this arm
            rec["kernels"] = dict(seen=seen, n=len(names))
            if not seen:
                rec.update(status="fallback", note="expected kernels absent" if a.expect else "launches equal the entry's")
                retain.keep(cap)
                continue
            if a.kind == "probe":
                rec.update(status="probe", note="timing-only arm: output not validated")
                caps[a.name], kinds[a.name] = cap, a.kind
                continue
            if budget:
                budget("validate " + a.name)
            ok, note, info = replay_valid(cap, live, sets, refs, recipe, state)
            if ok and counts and "routes" not in info:
                ok, note = False, "recipe returned no routed counts"
            rec.update(status="valid" if ok else "dropped", note=note, routes=info.get("routes"))
            if ok:
                caps[a.name], kinds[a.name] = cap, a.kind
            else:
                retain.keep(cap)  # later captures may still read storage owned by a rejected graph
        except Exception as exc:
            rec.update(status="raised", note=repr(exc)[:100])
            retain.keep(cap)
            recover(exc)

    def poison_fn(name):
        poison(caps[name].out)
        if name in scratch:
            poison(scratch[name])

    def refresh(r):
        recipe.refresh(live, sets[r % len(sets)])

    try:
        times = time_rounds(caps, refresh, poison_fn, flush, ovh, rounds,
                            **({"budget": budget} if budget else {})) if "base" in caps and "twin" in caps else {}
    except Exception as exc:
        recover(exc)
        raise
    raw_times = dict(times) if raw else None  # Preserve timed arms even if post-timing validation rejects them.
    for name in list(times):  # the graph must still be right after being replayed hundreds of times
        if kinds.get(name) == "probe":
            continue
        try:
            if budget:
                budget("post-timing validation " + name)
            ok, note, _ = replay_valid(caps[name], live, sets, refs, recipe, state)
        except Exception as exc:
            ok, note = False, repr(exc)[:100]
            next(r for r in recs if r["name"] == name).update(status="invalid_after_timing", note=note)
            recover(exc)
        if not ok:
            next(r for r in recs if r["name"] == name).update(status="invalid_after_timing", note=note)
            times.pop(name)
    cell, pick = cell_record(label, case["batch"] * case.get("q_len", 1), len(sets), times, kinds, recs, ovh, raw, bar)
    if raw:
        nrounds = len(raw_times.get("base", []))
        cell["raw"].update(times_us=raw_times, seed=seed, names=list(raw_times),
                           input_indices=[r % len(sets) for r in range(nrounds)],
                           order_indices=[round_order(list(range(len(raw_times))), r) for r in range(nrounds)],
                           statuses={r["name"]: r["status"] for r in recs if r["name"] in raw_times})
    if forced:
        cell["note"] = "pricing/single-input arm: one input set"
    retain.keep(*[c.out for c in caps.values()])
    caps.clear()
    return cell, pick


def ck_resolve_arms(spec, key, recipe, ns):
    arms = []
    for a in spec["arms"]:
        if a.get("shapes") and key not in a["shapes"]:
            continue
        if a.get("apply"):
            fn = ns[a["apply"]]
            apply = (lambda st, fn=fn: fn(st, key))  # noqa: E731
        elif a.get("plan") is not None:
            apply = ck_plan_apply(recipe.key(key), a["plan"])
        else:
            apply = ck_noop
        arms.append(Arm(a["name"], apply, a.get("kind", "real"), tuple(a.get("expect", ())),
                        bool(a.get("single_input")), tuple(ns[r] for r in a.get("retain", ())),
                        ns[a["scratch"]] if a.get("scratch") else None))
    return arms


def _cell_line(cell):
    if cell.get("base_us") is None:
        return "%s %s" % (cell["key"], cell.get("note"))
    cells = " ".join("%s %.3f(%+.1fus)" % (a["name"], a["ratio"], a["saved_us"]) if "ratio" in a
                     else "%s %s" % (a["name"], a["status"]) for a in cell["arms"] if a["name"] != "twin")
    aa = cell["aa"]
    return "%s base %.1f aa %.3f band %.3f%s pick %s | %s" % (
        cell["key"], cell["base_us"], aa["median"], aa["band"], "" if aa["valid"] else " AA-INVALID",
        cell["pick"] or "entry", cells)


def _recipe(spec, kernel_fn, ns):
    if spec.get("recipe", "layer") == "layer":
        return LayerRecipe(kernel_fn)
    return ns[spec["recipe_hook"]](kernel_fn)  # a CallbackRecipe built by the spec's hooks file


def ck_run(state, t_start, kernel_fn, spec, prov, ns):
    """Run the whole check at the end of prepare(); always ends by printing CK_TABLE and CK_RESULT."""
    import torch
    xr = bool(spec.get("xray"))
    emit = Emitter(human_cap=3500, raw_cap=0, xr_cap=4500) if xr else Emitter()
    # A replica can lift the stops (OKC_CK_STOP_S / OKC_CK_NEAR_STOP_S) to time more arms per prepare;
    # the judge never sets them, so official checks keep the 300 s-safe defaults.
    budget = Budget(t_start, float(os.environ.get("OKC_CK_STOP_S", spec.get("stop_s", 235.0))),
                    float(os.environ.get("OKC_CK_NEAR_STOP_S", spec.get("near_stop_s", 266.0))))
    guard = budget.require if spec.get("bounded") else None
    res = new_result(dict(prov, carrier_sha=None))
    flags = res["verdict"]["flags"]
    retain = Retain()
    kept = res["verdict"]["kept"]
    res["verdict"]["skipped"] = budget.skipped
    # Hooks can mutate global dispatch, so winners for different shapes cannot be composed safely.
    # Keep their measured picks as evidence, but leave the production entry unchanged.
    measurement_only = bool(spec.get("entry_off") or any(a.get("apply") for a in spec["arms"]))
    res["verdict"]["measurement_only"] = measurement_only
    try:
        if guard:
            guard("calibration")
        device = torch.device("cuda")
        recipe = _recipe(spec, kernel_fn, ns)
        flush = Flush(spec.get("flush_mib", 256), device)
        res["calib"] = calibrate(flush)
        ovh = res["calib"]["ovh_us"]
        flags += res["calib"]["flags"]
        res["env"] = dict(gpu=torch.cuda.get_device_name(0), sms=torch.cuda.get_device_properties(0).multi_processor_count,
                          torch=torch.__version__, clocks_start=clocks())
        emit.line("calib ovh %.2f ungated %.2f flush %dMiB %s" % (ovh, res["calib"]["ovh_ungated_us"], flush.mib,
                                                                 " ".join(flags)))
        eo = spec.get("entry_off")
        rounds, off = spec.get("rounds", 24), (None if not eo else ck_entry_off if eo == "swap_off" else ns[eo])
        for p_i, p in enumerate(spec.get("passes") or [dict(tag="main", nsets=3)]):
            for i, key in enumerate(priority_order(p.get("shapes") or spec["shapes"], spec.get("priority", []))):
                if not budget.ok("%s %s" % (p["tag"], key)):
                    continue
                try:
                    arms = ck_resolve_arms(spec, key, recipe, ns)
                    cell, pick = compare(key, recipe.cases[key], arms, recipe=recipe, kernel_fn=kernel_fn, state=state,
                                         flush=flush, ovh=ovh, rounds=rounds, device=device, seed=0x58524159 + 100 * i + 7 * p_i,
                                         retain=retain, nsets=p.get("nsets", 3), raw=spec.get("raw", False),
                                         bar=spec.get("bar", 1.01), base=off, counts=spec.get("counts", True),
                                         **({"budget": guard} if guard else {}))
                    cell["pass"] = p["tag"]
                    res["shapes"].append(cell)
                    emit.line(_cell_line(cell))
                    raw_cell = cell.pop("raw", None)
                    if raw_cell is not None:
                        raw_cell["pass"] = p["tag"]
                        if not emit.raw(key, raw_cell) and "RAW_TRUNCATED" not in flags:
                            flags.append("RAW_TRUNCATED")
                    for a in cell["arms"]:
                        if a.get("note") and a["status"] != "valid":
                            emit.line("%s %s %s: %s" % (key, a["name"], a["status"], a["note"]))
                    if pick and not measurement_only:
                        next(x for x in arms if x.name == pick).apply(state)  # kept for good; kernel_fn reads it per call
                        kept[key] = pick
                except Exception as exc:
                    if isinstance(exc, (CudaContextError, PlanRestoreError, BudgetExpired)) and getattr(exc, "cell", None) is not None:
                        exc.cell["pass"] = p["tag"]
                        res["shapes"].append(exc.cell)
                    if isinstance(exc, BudgetExpired):
                        raise
                    check_cuda_health(exc)
                    if isinstance(exc, PlanRestoreError):
                        raise
                    emit.line("%s raised %s" % (key, repr(exc)[:200]))
                    flags.append("RAISED:" + key)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        if spec.get("nearby") and off is not None:
            _nearby(spec, kept, recipe, kernel_fn, state, flush, ovh, device, retain, emit, budget, res, off)
        elif spec.get("nearby"):
            flags.append("NEARBY_NEEDS_ENTRY_OFF")
        fallbacks = [(c["key"], a["name"]) for c in res["shapes"] for a in c["arms"] if a["status"] == "fallback"]
        res["gates"].append(dict(name="kernels", **{"pass": not fallbacks}, detail="fallback arms: %s" % fallbacks[:6]))
        _clone_gate(spec, kept, recipe, kernel_fn, state, device, budget, res)
        for g in res["gates"]:
            emit.line("gate %s %s %s" % (g["name"], "ok" if g["pass"] else "FAIL", g.get("detail", "")))
        if spec.get("xray"):
            _xray(spec, recipe, kernel_fn, state, flush, ovh, device, emit, budget)
        if spec.get("post"):
            _post(spec, state, recipe, kernel_fn, flush, device, emit, budget, ns)
        res["verdict"].update(kept=kept, skipped=budget.skipped)
        res["env"].update(clocks_end=clocks(), prepare_s=round(budget.elapsed(), 1))
    except BudgetExpired as exc:
        flags.append("BUDGET_EXHAUSTED")
        emit.line(str(exc))
    except Exception as exc:
        flags.append("RAISED")
        try:
            check_cuda_health(exc)
        except CudaContextError as fault:
            flags.append("CUDA_CONTEXT_UNUSABLE")
            emit.line(str(fault))
            raise
        emit.line("raised %s" % repr(exc)[:300])
        if isinstance(exc, PlanRestoreError):
            flags.append("PLAN_RESTORE_FAILED")
            raise
    finally:
        res["env"]["prepare_s"] = round(budget.elapsed(), 1)
        retain.clear()
        if "CUDA_CONTEXT_UNUSABLE" not in flags:
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception:
                pass
        res["check"]["predeclared"] = spec["predeclared"]
        res["kill"] = dict(rule=spec["predeclared"].get("text"), evaluated=eval_kill(spec["predeclared"], res["shapes"]))
        if emit.raw_records:
            import hashlib
            res["verdict"]["raw_sha256"] = [hashlib.sha256(s.encode()).hexdigest() for s in emit.raw_records]
        try:
            line, _ = trim(res)
        except ValueError:  # last resort: decision fields and bounded rejection reasons
            for s in res["shapes"]:
                s["arms"] = [{k: a[k] for k in ("name", "status", "ratio", "saved_us", "note") if k in a} for a in s["arms"]]
            res["nearby"], res["verdict"]["flags"] = [], res["verdict"]["flags"][:8] + ["TRIM_COMPACT"]
            line, _ = trim(res)
        emit.final(table_rows(res), line)


def _nearby(spec, kept, recipe, kernel_fn, state, flush, ovh, device, retain, emit, budget, res, off):
    import torch
    fails = []
    keys = list(kept) or list(spec.get("nearby_shapes", ()))
    for i, key in enumerate(keys):
        for j, (label, case) in enumerate(recipe.nearby(key)):
            row = dict(label=label, status="skipped")
            res["nearby"].append(row)
            if not budget.ok(label, near=True):
                continue
            try:
                cell, _ = compare(label, case, [Arm("cand", ck_noop, fallback_ok=True)], recipe=recipe, kernel_fn=kernel_fn, state=state,
                                  flush=flush, ovh=ovh, rounds=12, device=device, seed=0x58524159 + 7000 + 100 * i + 10 * j,
                                  retain=retain, base=off, nsets=1, counts=spec.get("near_counts", True),
                                  **({"budget": lambda label: budget.require(label, near=True)} if spec.get("bounded") else {}))
                cand = next((a for a in cell["arms"] if a["name"] == "cand"), {})
                if cand.get("status") != "valid" or not cell["aa"].get("valid"):
                    row.update(status="FAIL", note=cand.get("note") or "A/A invalid")
                else:
                    ok = near_verdict(cand["ratio"], cell["aa"]["band"])
                    row.update(status="pass" if ok else "FAIL", entry_us=cell["base_us"], ratio=cand["ratio"],
                               band=cell["aa"]["band"])
            except Exception as exc:
                row.update(status="FAIL", note="raised " + repr(exc)[:100])
                if isinstance(exc, BudgetExpired):
                    raise
                check_cuda_health(exc)
                if isinstance(exc, PlanRestoreError):
                    raise
            if row["status"] == "FAIL":
                fails.append(label)
                emit.line("near %s FAIL %s" % (label, row.get("ratio", row.get("note"))))
            torch.cuda.empty_cache()
    res["verdict"]["near_ok"] = not fails
    emit.line("near summary: %s" % ("FAIL " + ", ".join(fails) if fails else "all cases pass"))
    if fails:
        state["swap_near_ok"] = False


def _clone_gate(spec, kept, recipe, kernel_fn, state, device, budget, res):
    """The scored path (picks applied) on clone_state(state), the way the judge hands it to kernel_fn, on a fresh
    input, poisoned, against the eager result on the original state."""
    import torch
    key = next(iter(kept), spec["shapes"][0])
    gate = dict(name="clone_state", **{"pass": False}, detail="not run")
    res["gates"].append(gate)
    if not budget.ok("clone gate", near=True):
        gate["detail"] = "skipped: budget"
        return
    try:
        guard = (lambda label: budget.require(label, near=True)) if spec.get("bounded") else None
        clone = clone_state(state)
        sets = recipe.sets(recipe.cases[key], 0x434C4F4E, device, state, 2)
        refs = [recipe.reference(d, state) for d in sets]
        live = recipe.live(sets[0])
        if guard:
            guard("clone capture")
        cap = Capture(lambda: kernel_fn(live, clone))
        ok, note, _ = replay_valid(cap, live, sets, refs, recipe, clone)
        if ok and spec.get("soak"):
            gate["soak"] = "sampled 1/100"
            soak(cap, key, live, sets, refs, recipe, clone, n=spec["soak"], **({"budget": guard} if guard else {}))
            note += " soak %d sampled 1/100" % spec["soak"]
        gate.update(**{"pass": ok}, detail="%s %s" % (key, note))
        clone = None
    except Exception as exc:
        gate["detail"] = "%s raised %s" % (key, repr(exc)[:150])
        if isinstance(exc, BudgetExpired):
            raise
        check_cuda_health(exc)


def _xray(spec, recipe, kernel_fn, state, flush, ovh, device, emit, budget):
    for i, key in enumerate(spec["xray"]):
        if not budget.ok("xray " + key):
            continue
        try:
            xray_shape(key, i, recipe, kernel_fn, state, flush, ovh, device, emit.line_xr,
                       **({"budget": budget.require} if spec.get("bounded") else {}))
        except Exception as exc:
            if isinstance(exc, BudgetExpired):
                raise
            check_cuda_health(exc)
            emit.line("xray %s raised %s" % (key, repr(exc)[:200]))


def _post(spec, state, recipe, kernel_fn, flush, device, emit, budget, ns):
    """The spec's untimed post-timing step: `post` names a hooks function (state, ctx) run after every timed cell, gate
    and X-ray, for diagnostics only (never part of a decision). ctx: recipe, kernel_fn, flush, device and emit (one
    CK line through the log budget)."""
    if not budget.ok("post"):
        return
    try:
        ns[spec["post"]](state, dict(recipe=recipe, kernel_fn=kernel_fn, flush=flush, device=device, emit=emit.line))
    except Exception as exc:
        if isinstance(exc, BudgetExpired):
            raise
        check_cuda_health(exc)
        emit.line("post %s raised %s" % (spec["post"], repr(exc)[:200]))
