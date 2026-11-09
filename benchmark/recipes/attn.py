"""A recipe from callbacks, for attention, MoE and prefill boards whose entry owns its input generator (_gen,
_synthetic): the engine only needs sets, refresh, reference and check."""


class CallbackRecipe:
    name = "callback"

    def __init__(self, cases, sets, reference, check, nearby=None, key=None):
        self.cases, self._sets, self._reference, self._check = cases, sets, reference, check
        self.nearby = nearby or (lambda k: [])
        self.key = key or (lambda k: k)

    def sets(self, case, seed, device, state, n):
        return self._sets(case, seed, n)

    def live(self, src):
        import torch
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in src.items()}

    def refresh(self, live, src):
        import torch
        for k, v in src.items():
            if isinstance(v, torch.Tensor):
                live[k].copy_(v)

    def reference(self, d, state):
        return self._reference(d, state)

    def check(self, out, d, ref, state):
        return self._check(out, d, ref, state)

    def routes(self, counts):
        return {}
