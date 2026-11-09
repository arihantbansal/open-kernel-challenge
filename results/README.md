# Results

All five submissions passed the official correctness checks on an NVIDIA H200
SXM. Speedups are relative to the challenge's SGLang baseline.

| Kernel | Speedup | Tests passed | Job |
|---|---:|---:|---|
| [MoE](../submissions/muse_moe_w4a16.py) | 2.3698× | 5/5 | `268cf7dfbd7f455e` |
| [Attention](../submissions/muse_attention_decode.py) | 2.3285× | 16/16 | `300559f5d2484eb4` |
| [Decode layer](../submissions/muse_decode_layer.py) | 2.2315× | 13/13 | `6d010e3ba74841ac` |
| [Prefill layer](../submissions/muse_prefill_layer.py) | 1.9114× | 5/5 | `4d6d78725b744746` |
| [BF16 projections](../submissions/muse_bf16_proj.py) | 1.5180× | 9/9 | `e085bc90b3e043f1` |

[Full results](results.json) include confidence intervals, per-case timings and
source hashes. Durations ending in `_ns` are nanoseconds; `score_us` is microseconds.

## Progress

Selected scored runs during the challenge:

| Kernel | Earlier run | Latest run |
|---|---:|---:|
| MoE | 2.1098× · Oct 27 | 2.3698× · Nov 4 |
| Attention | 2.1156× · Nov 3 | 2.3285× · Nov 4 |
| Decode layer | 2.1713× · Nov 3 | 2.2315× · Nov 3 |
| BF16 projections | 1.4097× · Oct 27 | 1.5180× · Nov 3 |
| Prefill layer | — | 1.9114× · Nov 4 |

The MoE run in my [October 22 post](https://x.com/arihantbansal/status/2113367627251032549)
was **1.31×**, when I reached **#1 on the MoE leaderboard**.
[Progress records](progress.json) contain the later runs; [peak-rank.json](peak-rank.json)
records the leaderboard milestone.

## Source records

Comments and internal labels were cleaned up after the scored runs.
`measured_sha256` identifies the source that ran; `sha256` identifies the file
in this repo. Both are stored with each result.

From the repo root, run `python3 tools/verify.py` to check the files and records.
