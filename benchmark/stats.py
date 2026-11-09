"""Pair math, decisions and kill rules. Pure Python (no torch), inlined flat into carriers."""

import math
import statistics


def pair_ratios(base, other, overhead):
    """Judge-style paired ratios: per round (base - ovh) / (other - ovh), geometric mean of rounds (2k, 2k+1).
    Strict: equal even lengths, finite overhead >= 0, every adjusted time finite and positive."""
    if len(base) != len(other) or len(base) < 2 or len(base) % 2:
        raise ValueError("equal, positive, even repetition counts required")
    if not math.isfinite(overhead) or overhead < 0:
        raise ValueError("invalid replay overhead")
    if any(not math.isfinite(x) or x <= overhead for x in [*base, *other]):
        raise ValueError("non-positive or non-finite adjusted replay time")
    ratios = [(a - overhead) / (b - overhead) for a, b in zip(base, other)]
    return [math.sqrt(ratios[i] * ratios[i + 1]) for i in range(0, len(ratios), 2)]


def summarize(times, overhead):
    """times {name: [us per round]} with "base" and "twin" -> base_us, A/A median and band, validity, median ratios."""
    pairs = {n: pair_ratios(times["base"], v, overhead) for n, v in times.items() if n != "base"}
    aa = statistics.median(pairs["twin"])
    return dict(base_us=statistics.median(times["base"]) - overhead, aa=aa,
                band=max(abs(x - 1) for x in pairs["twin"]), valid=0.97 <= aa <= 1.03,
                ratios={n: statistics.median(v) for n, v in pairs.items()}, pairs=pairs)


def decide(times, kinds, overhead, bar=1.01, aa_band=(0.97, 1.03)):
    """-> (pick or None, {name: median ratio}, A/A median, A/A band, A/A valid). Ratio > 1 is faster than the entry.
    An arm is picked when its kind is "real" and its ratio beats max(bar, 1 + band, 1 + 2 |A/A - 1|)."""
    s = summarize(times, overhead)
    valid = aa_band[0] <= s["aa"] <= aa_band[1]
    cut = max(bar, 1 + s["band"], 1 + 2 * abs(s["aa"] - 1))
    wins = {n: m for n, m in s["ratios"].items() if n != "twin" and kinds.get(n) == "real" and m > cut}
    return (max(wins, key=wins.get) if valid and wins else None), s["ratios"], s["aa"], s["band"], valid


def near_verdict(ratio, band, margin=0.02):
    """Nearby gate: pass when the candidate is not slower than the entry by more than max(margin, 2 x A/A band)."""
    return bool(ratio >= 1 - max(margin, 2 * band))


def overhead_from_times(one, two, limit_us=15.0):
    """The judge's replay overhead, 2 * median(one-kernel graph) - median(two-kernel graph), bounded to [0, limit]."""
    value = 2 * statistics.median(one) - statistics.median(two)
    if not math.isfinite(value) or value < 0 or value > limit_us:
        raise ValueError("invalid graph overhead %r us" % value)
    return value


def round_order(names, r):
    """Arm order for round r: rotate by r // 2, reversed on odd rounds, so each pair of rounds is position-balanced."""
    k = (r // 2) % len(names)
    order = names[k:] + names[:k]
    return order[::-1] if r % 2 else order


def eval_kill(rule, shapes):
    """Evaluate a predeclared rule against the result's `shapes`.
    rule: {shape, arm, metric "ratio"|"saved_us", op "<"|">", value, text}; arm "best" is the valid non-control arm
    with the largest metric. Returns "KILL", "PASS", "INVALID_AA" (the shape's A/A was out of band or missing) or
    "INVALID" (cell or arm absent, dropped, fallback or raised)."""
    cell = next((s for s in shapes if s.get("key") == rule["shape"]), None)
    if cell is None or not cell.get("aa", {}).get("valid"):
        return "INVALID_AA"
    if rule["arm"] == "best":
        arms = [a for a in cell.get("arms", []) if a["name"] != "twin" and a.get("kind") != "control"
                and a.get("status") == "valid" and rule["metric"] in a]
        arm = max(arms, key=lambda a: a[rule["metric"]], default=None)
    else:
        arm = next((a for a in cell.get("arms", []) if a["name"] == rule["arm"]), None)
    # a probe arm (timing-only, never validated) counts only when the rule names it
    measured = ("valid", "probe") if rule["arm"] != "best" else ("valid",)
    if arm is None or arm.get("status") not in measured or rule["metric"] not in arm:
        return "INVALID"
    hit = arm[rule["metric"]] < rule["value"] if rule["op"] == "<" else arm[rule["metric"]] > rule["value"]
    return "KILL" if hit else "PASS"
