"""Layer X-ray: the profiler timeline with exposed-time algebra, ported from build/xray_body.py. The trace analysis
(`_xr_*`, "XR " output lines, `_xr_mark` for entries whose kernel_fn carries stage marks) is unchanged and unit-tested
on synthetic Chrome traces; `xray_shape` drives it through the kit's recipe, Capture and Flush. Without marks in the
entry, ops are labelled "?" and only the roles, windows and gaps are meaningful. Diagnostic only."""

import time

from gpu import Capture, poison

import json as _xr_json
import math as _xr_math
import os as _xr_os
import re as _xr_re
import statistics as _xr_stats
import tempfile as _xr_tempfile

_XR = {"on": False, "rf": None}
_XR_REPS, _XR_PROF = 20, 10
_XR_MOE_LABELS = ("moe", "moe_meta", "moe_norm")


def _xr_mark(label):
    if not _XR["on"]:
        return
    import torch
    rf = _XR["rf"]
    if rf is not None:
        rf.__exit__(None, None, None)
        _XR["rf"] = None
    if label:
        rf = torch.profiler.record_function("xr:" + label)
        rf.__enter__()
        _XR["rf"] = rf


# ---- pure trace analysis ----

def _xr_short(name, width=30):
    n = _xr_re.sub(r"^void ", "", name).split("(")[0]
    n = _xr_re.sub(r"\b\w+::", "", n).replace(" ", "")
    return n[:width]


def _xr_stage(label, name):
    """The stage an op counts toward: its mark, with MoE ops split by kernel name."""
    if label not in _XR_MOE_LABELS:
        return label
    n = name.lower()
    if _xr_re.search(r"gemm<0|gemm1|_g1|w13", n):
        return "moe.g1"
    if _xr_re.search(r"gemm<1|gemm2|_g2|w2", n):
        return "moe.g2"
    if _xr_re.search(r"final|combine|weighted|wsum", n):
        return "moe.fin"
    if _xr_re.search(r"route|align|permute|sort|scatter|count|norm|memset", n):
        return "moe.route"
    return "moe.other"


def _xr_ops(events):
    """GPU ops (kernels, memsets, memcpys) of a Chrome trace, in start order."""
    ops = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memset", "gpu_memcpy")]
    return sorted(ops, key=lambda e: e["ts"])


def _xr_launches(events):
    """correlation id -> the host launch event (runtime or driver API) that issued it."""
    out = {}
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver"):
            c = e.get("args", {}).get("correlation")
            if c is not None:
                out[c] = e
    return out


def _xr_corr(e):
    return e.get("args", {}).get("correlation")


def _xr_label(events):
    """The labelled eager call: [(name, label, role)] in launch order, and how many ops had no mark."""
    ops, launches = _xr_ops(events), _xr_launches(events)
    marks = sorted((e["ts"], e["ts"] + e["dur"], e["name"][3:]) for e in events
                   if e.get("ph") == "X" and e.get("cat") == "user_annotation" and e.get("name", "").startswith("xr:"))
    rows = []
    for op in ops:
        launch = launches.get(_xr_corr(op))
        at = launch["ts"] if launch else None
        label = next((m[2] for m in marks if at is not None and m[0] <= at <= m[1]), "?")
        rows.append((at if at is not None else op["ts"], op["name"], label, op.get("args", {}).get("stream")))
    rows.sort(key=lambda r: r[0])
    main = next((r[3] for r in rows if r[2] == "qkv"), None)
    if main is None and rows:
        streams = [r[3] for r in rows]
        main = max(set(streams), key=streams.count)
    unlabelled = sum(r[2] == "?" for r in rows)
    return [(r[1], r[2], "m" if r[3] == main else "s") for r in rows], unlabelled


def _xr_replays(events, n_ops):
    """Per profiled replay, its GPU ops in start order. Primary: ops correlated with each cudaGraphLaunch.
    Fallback: ops not issued by a non-graph launch (input copies, flushes), chunked by n_ops."""
    ops, launches = _xr_ops(events), _xr_launches(events)
    graph = {c for c, e in launches.items() if "GraphLaunch" in e.get("name", "")}
    segs = {}
    for op in ops:
        c = _xr_corr(op)
        if c in graph:
            segs.setdefault(c, []).append(op)
    if segs and all(len(s) == n_ops for s in segs.values()):
        return [segs[c] for c in sorted(segs, key=lambda c: launches[c]["ts"])], "corr"
    rest = [op for op in ops if _xr_corr(op) not in launches or _xr_corr(op) in graph]
    if n_ops and len(rest) % n_ops == 0:
        return [rest[i:i + n_ops] for i in range(0, len(rest), n_ops)], "chunk"
    return [], "unsegmented (%d ops for %d per replay)" % (len(rest), n_ops)


def _xr_match(replay, labelled):
    """Replay ops -> eager indices by (name, occurrence); None where the names differ."""
    slots = {}
    for i, (name, _, _) in enumerate(labelled):
        slots.setdefault(name, []).append(i)
    seen, out = {}, []
    for op in replay:
        k = seen.get(op["name"], 0)
        seen[op["name"]] = k + 1
        idx = slots.get(op["name"], [])
        out.append(idx[k] if k < len(idx) else None)
    return out


def _xr_expose(ops):
    """ops: [(index, role, start, end)] of one replay. Returns ({index: exposed}, {"gap": us}, t0, window)."""
    t0 = min(o[2] for o in ops)
    t_end = max(o[3] for o in ops)
    exposed = {o[0]: 0.0 for o in ops}
    gap = 0.0
    main = sorted((o for o in ops if o[1] == "m"), key=lambda o: o[2])
    side = [o for o in ops if o[1] != "m"]
    prev = t0
    for idx, _, st, en in main:
        if st > prev:
            blockers = [s for s in side if prev < s[3] <= st + 0.5]
            wait = 0.0
            if blockers:
                b = max(blockers, key=lambda s: s[3])
                wait = min(b[3], st) - prev
                exposed[b[0]] += wait
            gap += st - prev - wait
        exposed[idx] += max(0.0, en - max(st, prev))
        prev = max(prev, en)
    if t_end > prev:  # a side op ends the replay (the final join)
        last = max(side, key=lambda s: s[3])
        exposed[last[0]] += t_end - prev
    return exposed, gap, t0, t_end - t0


def _xr_summary(labelled, replays):
    """Medians over replays: per eager op (start, end, exposed relative to the replay's first op), per stage the
    exposed sum, the gap and the window. Replays whose names do not match the eager call are dropped."""
    per_op = {i: [] for i in range(len(labelled))}
    stages, gaps, windows, dropped = {}, [], [], 0
    for rep in replays:
        idx = _xr_match(rep, labelled)
        if None in idx or len(idx) != len(labelled):
            dropped += 1
            continue
        ops = [(i, labelled[i][2], op["ts"], op["ts"] + op["dur"]) for i, op in zip(idx, rep)]
        exposed, gap, t0, window = _xr_expose(ops)
        sums = {}
        for i, _, st, en in ops:
            per_op[i].append((st - t0, en - t0, exposed[i]))
            key = _xr_stage(labelled[i][1], labelled[i][0])
            sums[key] = sums.get(key, 0.0) + exposed[i]
        for key in set(stages) | set(sums):
            stages.setdefault(key, []).append(sums.get(key, 0.0))
        gaps.append(gap)
        windows.append(window)
    med = lambda xs: _xr_stats.median(xs) if xs else float("nan")
    ops = [(i, *(med([r[j] for r in per_op[i]]) for j in range(3))) for i in range(len(labelled))]
    return {"ops": ops, "stages": {k: med(v) for k, v in stages.items()}, "gap": med(gaps), "window": med(windows),
            "replays": len(windows), "dropped": dropped}


def _xr_lines(key, labelled, summary):
    """The K and S lines of one shape."""
    out = []
    for i, st, en, ex in summary["ops"]:
        name, label, role = labelled[i]
        out.append("XR %s K %d %s %s %.1f %.1f %.1f %s" % (key, i, label, role, st, en, ex, _xr_short(name)))
    order = sorted(summary["stages"].items(), key=lambda kv: -kv[1])
    out.append("XR %s S win %.1f gap %.1f %s" % (key, summary["window"], summary["gap"],
                                                  " ".join("%s %.1f" % kv for kv in order)))
    return out


# ---- GPU side ----

def _xr_trace(prof):
    fd, path = _xr_tempfile.mkstemp(suffix=".json")
    _xr_os.close(fd)
    try:
        prof.export_chrome_trace(path)
        with open(path) as f:
            return _xr_json.load(f)["traceEvents"]
    finally:
        _xr_os.remove(path)


def _xr_timed(graph, refresh, flush):
    import torch
    torch.cuda.synchronize()
    refresh()
    flush.cold()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    graph.replay()
    e1.record()
    e1.synchronize()
    return e0.elapsed_time(e1) * 1e3


def xray_shape(key, index, recipe, kernel_fn, state, flush, ovh, device, say, budget=None):
    """Time and profile the captured graph of one scored shape; `say(line)` receives the "XR " lines."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    t0 = time.time()
    case = recipe.cases[key]
    sets = recipe.sets(case, 0x58524159 + 100 * index, device, state, 3)
    live = recipe.live(sets[0])
    if budget:
        budget("xray capture " + key)
    cap = Capture(lambda: kernel_fn(live, state))
    refresh = lambda r: recipe.refresh(live, sets[r % len(sets)])  # noqa: E731
    try:
        ref = recipe.reference(sets[1], state)
        recipe.refresh(live, sets[1])
        poison(cap.out)
        cap.replay()
        torch.cuda.synchronize()
        ok, note, info = recipe.check(cap.out, sets[1], ref, state)
        def timed(r):
            if budget:
                budget("xray replay " + key)
            return _xr_timed(cap.graph, lambda: refresh(r), flush)

        ev = [timed(r) for r in range(_XR_REPS)]
        evp = []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for r in range(_XR_PROF):
                evp.append(timed(r))
        rep_events = _xr_trace(prof)
        recipe.refresh(live, sets[0])
        torch.cuda.synchronize()
        if budget:
            budget("xray profile " + key)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _XR["on"] = True
            try:
                kernel_fn(live, state)
            finally:
                _xr_mark(None)
                _XR["on"] = False
            torch.cuda.synchronize()
        labelled, unlabelled = _xr_label(_xr_trace(prof))
    finally:
        del cap
    replays, how = _xr_replays(rep_events, len(labelled))
    summary = _xr_summary(labelled, replays)
    med = _xr_stats.median
    say("XR %s H ops %d unlabelled %d seg %s replays %d dropped %d in %.1fs" % (
        key, len(labelled), unlabelled, how, summary["replays"], summary["dropped"], time.time() - t0))
    say("XR %s V ev %.2f ev-ovh %.2f evprof %.2f win %.2f graph %s %s" % (
        key, med(ev), med(ev) - ovh, med(evp), summary["window"], "ok" if ok else "FAIL", note))
    lines = _xr_lines(key, labelled, summary)
    for line in lines[-1:] + lines[:-1]:  # the stage sums first: the Emitter drops from the end when the cap is hit
        say(line)
