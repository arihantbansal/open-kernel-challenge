"""Route phases for MoE checks: the same captures replayed on empty, tiny, observed and hot routes, so a check cannot
pass on easy routes only (persist check1 soaked empty routes). Pure Python."""

import json


def moe_load_observed(path):
    """p50 per-expert histogram (sorted descending) from a route-sample JSON report."""
    return [int(x) for x in json.load(open(path))["p50_hist"]]


def moe_phases(tokens, observed=None, top_k=8):
    """{phase: sorted-descending per-local-expert counts}. hot: top_k experts every token routes to; tiny: one route."""
    out = {"empty": [], "tiny": [1], "hot": [tokens] * top_k}
    if observed:
        out["observed"] = sorted(observed, reverse=True)
    return out


def moe_check_phase(phase, counts, tokens, top_k=8):
    """Assert an actual routed-count vector matches the phase's contract; -> (ok, note)."""
    total = sum(counts)
    if phase == "empty":
        return total == 0, "routes %d" % total
    if phase == "tiny":
        return total == 1, "routes %d" % total
    if phase == "hot":
        return max(counts, default=0) == tokens and total == tokens * top_k, "hot %d routes %d" % (max(counts, default=0), total)
    return total > 0, "routes %d" % total
