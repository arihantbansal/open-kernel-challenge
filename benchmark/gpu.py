"""GPU primitives: capture, cold-L2 flush, overhead calibration, poisoned timed rounds, validation, soak, kernel names,
judge-faithful state clone. torch is imported inside functions so the CPU tests never need it."""

import math
import re
import statistics

from stats import round_order


class CudaContextError(RuntimeError):
    """The device cannot synchronize after a failure; no more arms may run."""

    def __init__(self, message):
        super().__init__(message)
        self.cell = None


def check_cuda_health(exc):
    """A host-side failure may be recoverable; a failed synchronization is not a reset."""
    import torch
    if isinstance(exc, CudaContextError):
        raise exc
    try:
        torch.cuda.synchronize()
    except Exception as fault:
        raise CudaContextError("CUDA context unusable after %s; synchronize: %s" %
                               (repr(exc)[:100], repr(fault)[:100])) from fault


class Capture:
    """fn captured once into a CUDA graph on a side stream after `warmups` eager calls; `.out` is fn's return value."""

    def __init__(self, fn, warmups=2):
        import torch
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(warmups):
                fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.out = fn()
        torch.cuda.synchronize()

    def replay(self):
        self.graph.replay()


class Flush:
    """Cold L2: `cold()` reads `mib` MiB (default 256, above the H200's 60 MB L2) into a fixed sink. Size is recorded."""

    kind = "read"

    def __init__(self, mib=256, device="cuda"):
        import torch
        self.mib = mib
        self.buf = torch.ones(mib * 1024 * 256, dtype=torch.float32, device=device)
        self.sink = torch.empty((), dtype=torch.float32, device=device)

    def cold(self):
        import torch
        torch.sum(self.buf, dim=0, out=self.sink)


def _event_time(graph, flush, events, gate_cycles):
    import torch
    flush.cold()
    if gate_cycles:
        torch.cuda._sleep(gate_cycles)  # keeps host enqueue ahead of the event interval, as the judge's queue is
    events[0].record()
    graph.replay()
    events[1].record()
    events[1].synchronize()
    return events[0].elapsed_time(events[1]) * 1000


def calibrate(flush, reps=20, gate_cycles=500_000):
    """Replay overhead 2 * one - two of trivial-kernel graphs, sleep-gated (the method that matches the judge's ~4 us)
    and ungated, both recorded. flags OVH_METHOD_SENSITIVE when they differ by more than 1 us."""
    import torch
    from_ = {}
    x = torch.zeros(1, device="cuda")
    x.add_(1)
    graphs = []
    for count in (1, 2):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(count):
                x.add_(1)
        graphs.append(g)
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    for label, gate in (("gated", gate_cycles), ("ungated", 0)):
        s = [[], []]
        for r in range(reps):
            for i in ((0, 1) if r % 2 == 0 else (1, 0)):
                s[i].append(_event_time(graphs[i], flush, events, gate))
        from_[label] = 2 * statistics.median(s[0]) - statistics.median(s[1])
    ovh = from_["gated"]
    if not math.isfinite(ovh) or ovh < 0 or ovh > 15:
        raise ValueError("invalid graph overhead %r us" % ovh)
    flags = ["OVH_METHOD_SENSITIVE"] if abs(from_["gated"] - from_["ungated"]) > 1.0 else []
    return dict(flush_mib=flush.mib, flush_kind=flush.kind, ovh_us=ovh, ovh_ungated_us=from_["ungated"], flags=flags)


def poison(x):
    """Fill every tensor in x (tensor, tuple, list, dict) with NaN, 0xFF bytes for FP8, or -1."""
    import torch
    if isinstance(x, torch.Tensor):
        if x.is_floating_point() or x.dtype == torch.bool:
            if x.element_size() == 1:
                x.view(torch.uint8).fill_(0xFF)  # NaN in E4M3FN
            else:
                x.fill_(float("nan"))
        else:
            x.fill_(-1)
    elif isinstance(x, dict):
        for v in x.values():
            poison(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            poison(v)


def time_rounds(caps, refresh, poison_fn, flush, overhead, rounds=24, gate_cycles=500_000, budget=None):
    """Cold-L2 event times in us per arm. Each replay is preceded by refresh(r), poison_fn(name) and the flush; the
    order rotates every pair of rounds and reverses on odd rounds (stats.round_order)."""
    import torch
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    names = list(caps)
    times = {n: [] for n in names}
    for r in range(rounds):
        for n in round_order(names, r):
            if budget:
                budget("timing %s round %d" % (n, r))
            torch.cuda.synchronize()
            refresh(r)
            poison_fn(n)
            times[n].append(_event_time(caps[n].graph, flush, events, gate_cycles))
    return times


def replay_valid(cap, live, sets, refs, recipe, state):
    """Poisoned replays of one captured graph on every input used for timing, including its capture input.
    -> (ok, note, info). The recipe checks the actual live inputs after replay, plus outputs against the eager
    reference and routed counts. Source sets stay pristine and only supply refresh values."""
    import torch
    worst, ids_eq = 0.0, True
    for i in range(len(sets)):
        recipe.refresh(live, sets[i])
        poison(cap.out)
        cap.replay()
        torch.cuda.synchronize()
        ok, note, info = recipe.check(cap.out, live, refs[i], state)
        if not ok:
            return False, "%s (input %d)" % (note, i), info
        worst, ids_eq = max(worst, info.get("rel", 0.0)), ids_eq and info.get("ids_eq", True)
    recipe.refresh(live, sets[0])
    return True, "rel %.1e ids %s" % (worst, "eq" if ids_eq else "DIFF(counts eq)"), info


def soak(cap, name, live, sets, expected, recipe, state, n=2000, every=100, budget=None):
    """n poisoned replays with output validation sampled 1/100 by default (every `every` replays and the final
    replay). Inputs rotate every `every` replays; sampled outputs are checked against expected[i], a clone of the
    entry's eager output. Raises RuntimeError naming the failed sample's replay index."""
    import torch
    for k in range(n):
        if budget:
            budget("soak %s replay %d" % (name, k))
        i = (k // every) % len(sets)
        if k % every == 0:
            recipe.refresh(live, sets[i])
        poison(cap.out)
        cap.replay()
        if k % every == every - 1 or k == n - 1:
            torch.cuda.synchronize()
            ok, note, _ = recipe.check(cap.out, live, expected[i], state)
            if not ok:
                raise RuntimeError("soak %s failed at replay %d: %s" % (name, k, note))


def kernels_of(run):
    """CUDA kernel names launched by one call of run() (a graph.replay or an eager call), in start order."""
    import torch
    from torch.profiler import ProfilerActivity, profile
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()
    from torch.autograd import DeviceType
    ev = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
    return [e.name for e in sorted(ev, key=lambda e: getattr(e.time_range, "start", 0))]


def expect_kernels(names, patterns):
    """True when every regex in patterns matches at least one executed kernel name."""
    return all(any(re.search(p, n) for n in names) for p in patterns)


def clone_state(x):
    """Clone dicts, lists, tuples and tensors using the challenge state contract."""
    import torch
    if isinstance(x, torch.Tensor):
        return x.clone()
    if type(x) is dict:
        return {k: clone_state(v) for k, v in x.items()}
    if type(x) in (list, tuple):
        return type(x)(clone_state(v) for v in x)
    return x


class Retain:
    """Keepalive owned by the engine: a snapshot an older graph still reads must not be freed until the run ends."""

    def __init__(self):
        self.items = []

    def keep(self, *objs):
        self.items.extend(objs)

    def clear(self):
        self.items.clear()


def clocks():
    """SM clock (MHz), power (W) and temperature of device 0 through NVML when torch exposes it; {} otherwise."""
    try:
        import torch
        out = {}
        for key, fn in (("sm_mhz", "clock_rate"), ("power_w", "power_draw"), ("temp_c", "temperature")):
            try:
                out[key] = getattr(torch.cuda, fn)()
            except Exception:
                pass
        return out
    except Exception:
        return {}
