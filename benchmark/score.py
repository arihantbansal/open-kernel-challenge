"""Challenge scoring arithmetic, implemented with the Python standard library.

For each shape, the speedup is the median of adjacent-pair geometric means of
baseline-to-candidate replay-time ratios after subtracting graph overhead. A
load time is the weighted sum of its shape times. The final score is the
geometric mean of baseline-to-candidate load-time ratios.
"""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import median


def replay_overhead_ns(one_ns: float, two_ns: float) -> float:
    """Fixed cost of replaying a graph, from timings of graphs with one and two trivial kernels."""
    return 2.0 * one_ns - two_ns


def rep_ratios(base_ns, cand_ns, overhead_ns: float = 0.0) -> list[float]:
    """Per-repetition SGLang time over candidate time, overhead removed from both. A time at or below the
    overhead means the measurement is broken (a ratio of two such times can look like any speedup), so
    it raises."""
    if len(base_ns) != len(cand_ns):
        raise ValueError("base and candidate repetition counts differ")
    for i, (b, c) in enumerate(zip(base_ns, cand_ns)):
        if b - overhead_ns <= 0 or c - overhead_ns <= 0:
            raise ValueError(f"repetition {i}: overhead-adjusted time is not positive "
                             f"(baseline {b:.0f} ns, candidate {c:.0f} ns, overhead {overhead_ns:.0f} ns)")
    return [(b - overhead_ns) / (c - overhead_ns) for b, c in zip(base_ns, cand_ns)]


def paired_speedup(ratios) -> float:
    """Median over adjacent repetition pairs (0,1), (2,3), ... of the pair's geometric mean.
    An odd trailing repetition is dropped."""
    pairs = [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios) - 1, 2)]
    if not pairs:
        raise ValueError("need at least two repetitions")
    return median(pairs)


def shape_result(base_ns, cand_ns, overhead_ns: float = 0.0) -> dict:
    """One shape's baseline time, speedup and candidate time from raw paired replay times (ns)."""
    speedup = paired_speedup(rep_ratios(base_ns, cand_ns, overhead_ns))
    baseline = median(base_ns) - overhead_ns
    return {"baseline_ns": baseline, "speedup": speedup, "median_ns": baseline / speedup}


def geomean(xs) -> float:
    xs = list(xs)
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def load_times(shapes) -> dict[str, tuple[float, float]]:
    """shapes: dicts with load, weight, baseline_ns, median_ns. Returns {load: (baseline, candidate)}
    where each is the weighted sum of shape times."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for s in shapes:
        w = float(s.get("weight", 1))
        acc[str(s["load"])][0] += w * s["baseline_ns"]
        acc[str(s["load"])][1] += w * s["median_ns"]
    return {k: (v[0], v[1]) for k, v in acc.items()}


def score(shapes) -> dict:
    """Geometric mean over loads of baseline load time / candidate load time, with per-load detail."""
    loads = load_times(shapes)
    per_load = {k: b / c for k, (b, c) in loads.items()}
    return {"speedup": geomean(per_load.values()), "per_load": per_load}


def displayed_time_us(sglang_us: float, speedup: float) -> float:
    """Convert a speedup to the displayed time using the published baseline time."""
    return sglang_us / speedup


def score_from_result(result: dict) -> dict:
    """Recompute the score from a judge result JSON (`benchmarks`: each with `case` holding load
    and weight, plus baseline_ns and median_ns)."""
    shapes = [{"load": b["case"]["load"], "weight": b["case"].get("weight", 1),
               "baseline_ns": b["baseline_ns"], "median_ns": b["median_ns"]} for b in result["benchmarks"]]
    return score(shapes)
