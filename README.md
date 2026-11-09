# Open Kernel Challenge

My submissions for the [Open Kernel Challenge](https://openkernelchallenge.dev/),
optimizing Muse Spark Open inference on NVIDIA H200.

**At my peak, I ranked [#1 on the MoE leaderboard](https://x.com/arihantbansal/status/2113367627251032549).**

## Results

| Kernel | Speedup over SGLang |
|---|---:|
| [MoE](submissions/muse_moe_w4a16.py) | **2.37×** |
| [Attention](submissions/muse_attention_decode.py) | **2.33×** |
| [Decode layer](submissions/muse_decode_layer.py) | **2.23×** |
| [Prefill layer](submissions/muse_prefill_layer.py) | **1.91×** |
| [BF16 projections](submissions/muse_bf16_proj.py) | **1.52×** |

All five passed the challenge's correctness tests.

The MoE submission went from **1.31× to 2.37×** during the challenge;
attention improved from **2.12× to 2.33×**.
[Results and progress](results/README.md).

## Explore

- [submissions/](submissions/): five self-contained Python/CUDA submissions.
- [results/](results/): scores, timings and progress.
- [benchmark/](benchmark/): compare kernel implementations.
- [tools/](tools/): source checks and build tools.

## Run

The kernels target H200 (`sm_90a`) and the
[challenge environment](https://openkernelchallenge.dev/rules). Each file compiles
its CUDA on first use and exposes `kernel_fn`; all except attention also expose
`prepare(weights)`.

Check the saved files and results locally:

```sh
python3 tools/verify.py
```

See [tools/](tools/README.md) for setup and tests.
