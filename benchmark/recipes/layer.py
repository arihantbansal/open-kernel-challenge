"""Input generation and output checks for the muse_decode_layer carrier recipe.

The recipe recreates public challenge inputs without importing challenge
reference code, and checks outputs, routed counts, cache writes and nearby cases.
"""

import math

LAYER_CASES = {
    "d1s": dict(batch=1, layer="sliding"), "d1f": dict(batch=1, layer="full"),
    "d16s": dict(batch=16, layer="sliding", spread="agentic"), "d16f": dict(batch=16, layer="full", spread="agentic"),
    "d32s": dict(batch=32, layer="sliding", spread="agentic"), "d32f": dict(batch=32, layer="full", spread="agentic"),
    "v1s": dict(batch=1, layer="sliding", q_len=8), "v1f": dict(batch=1, layer="full", q_len=8),
    "v16s": dict(batch=16, layer="sliding", q_len=8, spread="agentic"),
    "v16f": dict(batch=16, layer="full", q_len=8, spread="agentic"),
    "v32s": dict(batch=32, layer="sliding", q_len=8, spread="agentic"),
    "v32f": dict(batch=32, layer="full", q_len=8, spread="agentic"),
}
LAYER_SEED = 0x58524159


def layer_key(case_key):
    """(batch, q_len, sliding): the key the entry's per-shape tables use."""
    c = LAYER_CASES[case_key]
    return (c["batch"], c.get("q_len", 1), c["layer"] == "sliding")


def layer_nearby(key):
    """Neighbours of one scored shape: batch +-1, q_len +-1 (decode: 2), page size 64, ctx 80001."""
    case = LAYER_CASES[key]
    b, q = case["batch"], case.get("q_len", 1)
    out = [("%s+b%d" % (key, nb), dict(case, batch=nb)) for nb in (b - 1, b + 1) if nb >= 1]
    out += [("%s+q%d" % (key, nq), dict(case, q_len=nq)) for nq in ((2,) if q == 1 else (q - 1, q + 1))]
    out.append(("%s+p64" % key, dict(case, page=64)))
    out.append(("%s+c1" % key, dict(case, ctx=80001)))
    return out


def _layer_basis(device):
    import torch
    g = torch.Generator().manual_seed(5354)  # reference._stream_basis
    mean = torch.randn((8192,), generator=g)
    shared = torch.randn((4, 8192), generator=g)
    return mean.to(device), shared.to(device)


def layer_input(case, seed, device):
    """reference.generate_input for the scored shapes (ctx 80000, page 1, dist soft, kv_scale 1, spread fixed or
    agentic), with the same draws in the same order."""
    import torch
    batch, layer = case["batch"], case["layer"]
    q_len, spread = case.get("q_len", 1), case.get("spread", "fixed")
    ctx, page = case.get("ctx", 80000), case.get("page", 1)
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    if spread == "agentic":
        longest = math.ceil(ctx * 1.075)
        u = torch.rand((batch,), generator=gen, device=device)
        seq_lens = (ctx * (1 + 0.075 * (2 * u - 1))).long().clamp(1, longest)
    else:
        longest = ctx
        seq_lens = torch.full((batch,), ctx, dtype=torch.int64, device=device)
    pages_per = (seq_lens + q_len + page - 1) // page
    max_pages = (longest + q_len + page - 1) // page
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b in range(batch):
        n, o = int(pages_per[b]), int(offsets[b])
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    pos = positions.view(batch, q_len)
    cache_loc = (torch.gather(page_table.long(), 1, pos // page) * page + pos % page).reshape(-1)
    k_cache = torch.empty((total, page, 2, 64), dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.empty_like(k_cache)
    soft = torch.Generator(device=device).manual_seed(seed + 0x50F7)
    u = torch.randn((2, 64), generator=soft, device=device)
    u = u * torch.rsqrt(u.pow(2).mean(-1, keepdim=True))
    lo, hi = (0.3, 0.75) if layer == "sliding" else (0.5, 0.62)
    c2 = lo + (hi - lo) * torch.rand((2, 1), generator=soft, device=device)
    mix = (c2.sqrt() * u, (1 - c2).sqrt())
    for dst, keys in ((k_cache, True), (v_cache, False)):
        flat = dst.view(-1, 2, 64)
        for at in range(0, flat.shape[0], 1 << 20):
            n = min(1 << 20, flat.shape[0] - at)
            x = torch.randn((n, 2, 64), generator=gen, device=device)
            if keys:
                x = mix[0] + mix[1] * x
            flat[at:at + n] = x.to(torch.float8_e4m3fn)
    mean, shared = _layer_basis(device)
    coef = torch.randn((tokens, 4), generator=gen, device=device)
    scale = torch.ones((batch, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = 0.6
    scale = scale.reshape(tokens, 1)
    stream = (scale * (0.7 * mean + math.sqrt(0.9 / 4) * coef @ shared)
              + (1 - 0.9 * scale * scale).sqrt() * torch.randn((tokens, 8192), generator=gen, device=device))
    attn_input = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + 1e-5)).to(torch.bfloat16)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    meta = None
    if layer == "full" and q_len == 1:
        from sgl_kernel.flash_attn import get_scheduler_metadata
        meta = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=16, num_heads_k=2,
            headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=torch.float8_e4m3fn, cu_seqlens_q=cu_seqlens_q,
            page_size=page, causal=True, num_splits=0)
    one = lambda: torch.tensor(1.0, dtype=torch.float32, device=device)  # noqa: E731
    return {
        "stream": stream, "attn_input": attn_input, "positions": positions, "cache_loc": cache_loc,
        "cache_seqlens": cache_seqlens, "cu_seqlens_q": cu_seqlens_q, "scheduler_metadata": meta,
        "k_cache": k_cache, "v_cache": v_cache, "k_scale": one(), "v_scale": one(), "page_table": page_table,
        "seq_lens": seq_lens.to(torch.int32), "local_start": torch.tensor([0], dtype=torch.int32, device=device),
        "config": {"batch": batch, "q_len": q_len, "tokens": tokens, "hidden": 8192, "num_q_heads": 16,
                   "num_kv_heads": 2, "head_dim": 64, "page_size": page,
                   "window": 2048 if layer == "sliding" else 0, "rope_theta": 500000.0 if layer == "sliding" else 0.0,
                   "qk_scale_factor": 5.4730064863875, "sm_scale": 64 ** -0.5, "kv_scale": 1.0,
                   "n_experts": 256, "top_k": 8, "local_experts": 32},
    }


def layer_residual_alpha(state, weights):
    """Give an entry whose state lacks them (a donor layer) the residual alphas update() needs: reference.py's
    residual_coefficients of the judge's gates (temperature 0.3, eps 1e-3), the same per-column alpha1 / alpha2.
    An entry that already has them keeps its own."""
    import torch
    if not isinstance(state, dict) or "attnb_alpha" in state:
        return
    for key, gate in (("attnb_alpha", "attn_residual_gate"), ("ffnb_alpha", "ffn_residual_gate")):
        w = weights[gate].float() / 0.3
        state[key] = torch.sqrt(torch.clamp(torch.sigmoid(-w) * (1.0 + torch.sigmoid(w)), min=1e-3)).contiguous()


def layer_counts(ids, local_start):
    """Tokens routed to each of this rank's 32 local experts."""
    import torch
    local = (ids.long() - local_start.long()) % 256
    return torch.bincount(local[local < 32].flatten(), minlength=32)


def _layer_copy(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, dict):
        return {k: _layer_copy(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_layer_copy(v) for v in value)
    return value


def _layer_input_error(actual, expected, path="input"):
    """Byte comparisons also handle FP8 and preserve the exact readonly-input contract."""
    import torch
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            return "%s shape/dtype changed" % path
        if not torch.equal(actual.contiguous().reshape(-1).view(torch.uint8),
                           expected.contiguous().reshape(-1).view(torch.uint8)):
            return "%s modified" % path
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return "%s keys changed" % path
        for key in expected:
            error = _layer_input_error(actual[key], expected[key], "%s.%s" % (path, key))
            if error:
                return error
    elif isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return "%s changed" % path
        for i, value in enumerate(expected):
            error = _layer_input_error(actual[i], value, "%s[%d]" % (path, i))
            if error:
                return error
    elif actual != expected:
        return "%s changed" % path
    return None


def _layer_cache_error(data, before, targets, slots):
    """Check all bytes; bounded temporary comparisons avoid another retained full cache bank."""
    import torch
    for name in ("k_cache", "v_cache"):
        cache, original = data[name], before[name]
        if not isinstance(cache, torch.Tensor) or cache.shape != original.shape or cache.dtype != original.dtype:
            return "%s shape/dtype changed" % name
        width = cache.shape[-2] * cache.shape[-1] * cache.element_size()
        current = cache.contiguous().view(torch.uint8).reshape(-1, width)
        initial = original.contiguous().view(torch.uint8).reshape(-1, width)
        if not torch.equal(current[slots], targets[name]):
            return "%s new-KV bytes differ" % name
        for start in range(0, current.shape[0], 65536):
            same = (current[start:start + 65536] == initial[start:start + 65536]).all(dim=1)
            local = slots[(slots >= start) & (slots < start + same.numel())] - start
            same[local] = True
            if not bool(same.all()):
                return "%s bytes outside cache_loc modified" % name
    return None


def _layer_output_error(output, expected, label, tol):
    """Same-run incumbent tolerance, with the official row definition; not its dynamic oracle bound."""
    import torch
    if not isinstance(output, torch.Tensor):
        return "%s expected tensor" % label, float("inf"), float("inf")
    if output.shape != expected.shape or output.dtype != expected.dtype:
        return "%s shape/dtype mismatch" % label, float("inf"), float("inf")
    actual, reference = output.float(), expected.float()
    if not bool(torch.isfinite(actual).all()):
        return "%s contains NaN or Inf" % label, float("inf"), float("inf")
    diff = actual - reference
    rel = float(diff.norm() / reference.norm().clamp(min=1e-12))
    row_norm = reference.reshape(-1, reference.shape[-1]).norm(dim=-1)
    row_floor = (0.1 * row_norm.mean()).clamp(min=1e-12)
    row_error = diff.reshape(-1, reference.shape[-1]).norm(dim=-1) / torch.maximum(row_norm, row_floor)
    worst = float(row_error.max())
    if rel > tol or worst > 5.0 * tol:
        return "%s rel %.1e worst row %.1e (same-run tol %.1e)" % (label, rel, worst, tol), rel, worst
    return None, rel, worst


class LayerRecipe:
    """Recipe protocol the engine uses: cases, key, nearby, sets, live, refresh, reference, check, routes."""

    name = "layer"
    cases = LAYER_CASES
    tol = 1e-2  # Same-run incumbent agreement, not the official reference's case-dependent tolerance.

    def __init__(self, kernel_fn):
        self.kernel_fn = kernel_fn

    key = staticmethod(layer_key)
    nearby = staticmethod(layer_nearby)

    def sets(self, case, seed, device, state, n):
        import torch
        sets = []
        for i in range(n):
            d = layer_input(case, seed + i, device)
            ids = self.kernel_fn(self.live(d), state)[2]  # Keep generated inputs pristine, including new-KV slots.
            pairs = torch.bincount((ids.long() // 32).flatten(), minlength=8)
            d["local_start"].fill_(32 * int(pairs.argmax()))
            sets.append(d)
        return sets

    def live(self, src):
        return _layer_copy(src)

    def refresh(self, live, src):
        import torch
        for k, v in src.items():
            if isinstance(v, torch.Tensor):
                live[k].copy_(v)
            else:
                live[k] = _layer_copy(v)
        slots = src["cache_loc"].long().flatten()
        for name in ("k_cache", "v_cache"):
            cache = live[name]
            width = cache.shape[-2] * cache.shape[-1] * cache.element_size()
            cache.view(torch.uint8).reshape(-1, width)[slots] = 255

    def update(self, out, d, state):
        """The residual update the judge checks: output stream - alpha1 * alpha2 * input stream."""
        return out[1].float() - state["attnb_alpha"] * state["ffnb_alpha"] * d["stream"].float()

    def reference(self, d, state):
        import torch
        original = {k: _layer_copy(v) for k, v in d.items() if k not in ("k_cache", "v_cache")}
        baseline = self.live(d)
        o = self.kernel_fn(baseline, state)
        if not isinstance(o, (tuple, list)) or len(o) != 3 or any(not isinstance(v, torch.Tensor) for v in o):
            raise ValueError("incumbent expected three output tensors")
        if any(not bool(torch.isfinite(v).all()) for v in o):
            raise ValueError("incumbent output contains NaN or Inf")
        error = _layer_input_error({k: v for k, v in baseline.items() if k not in ("k_cache", "v_cache")}, original)
        if error:
            raise ValueError("incumbent " + error)
        slots = original["cache_loc"].long().flatten()
        targets = {}
        for name in ("k_cache", "v_cache"):
            cache = baseline[name]
            width = cache.shape[-2] * cache.shape[-1] * cache.element_size()
            targets[name] = cache.view(torch.uint8).reshape(-1, width)[slots].clone()
        # Source banks are owned by the pristine input sets; only baseline/live clones execute kernels.
        before = {name: d[name] for name in ("k_cache", "v_cache")}
        error = _layer_cache_error(baseline, before, targets, slots)
        if error:
            raise ValueError("incumbent " + error)
        ref = dict(out=o[0].clone(), ids=o[2].clone(), upd=self.update(o, d, state),
                   stream=o[1].clone(), counts=layer_counts(o[2], d["local_start"]),
                   immutable=original, cache_before=before, cache_targets=targets)
        torch.cuda.synchronize()
        return ref

    def check(self, out, d, ref, state):
        import torch
        if not isinstance(out, (tuple, list)) or len(out) != 3:
            return False, "expected three output tensors", {}
        for value, expected, label in zip(out, (ref["out"], ref["stream"], ref["ids"]), ("output", "stream", "ids")):
            if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
                return False, "%s shape/dtype mismatch" % label, {}
            if not bool(torch.isfinite(value).all()):
                return False, "%s contains NaN or Inf" % label, {}
        error = _layer_input_error({k: v for k, v in d.items() if k not in ("k_cache", "v_cache")}, ref["immutable"])
        if error:
            return False, error, {}
        error = _layer_cache_error(d, ref["cache_before"], ref["cache_targets"], ref["immutable"]["cache_loc"].long().flatten())
        if error:
            return False, error, {}
        counts = layer_counts(out[2], d["local_start"])
        if not bool(torch.equal(counts, ref["counts"])):
            return False, "routed counts differ (%d vs %d routes)" % (int(counts.sum()), int(ref["counts"].sum())), {}
        e0, r0, w0 = _layer_output_error(out[0], ref["out"], "output", self.tol)
        e1, r1, w1 = _layer_output_error(self.update(out, d, state), ref["upd"], "update", self.tol)
        info = dict(rel=max(r0, r1), worst_row=max(w0, w1), ids_eq=bool(torch.equal(out[2], ref["ids"])), routes=self.routes(counts))
        if e0 or e1:
            return False, e0 or e1, info
        return True, "ok", info

    def routes(self, counts):
        return dict(sum=int(counts.sum()), distinct=int((counts > 0).sum()), hot=int(counts.max()))

    def poison_targets(self, out):
        return out
