# Compare kernels

Build a single-file comparison from a submission and an experiment definition:

```sh
python3 benchmark/build.py benchmark/example_aa.py
```

The generated file and its checksum go in `out/benchmark/`. Building uses only
Python's standard library; it does not execute the submission. The example
compares two captures of the same implementation to check timing noise.

On a compatible GPU, the generated comparison refreshes inputs, poisons outputs,
checks cloned state and alternates replay order. Define the candidate arms and
acceptance rule before running an experiment. Each comparison must fit within
1 MiB and the challenge's preparation limit.

Read a saved result and bind it to the local comparison and experiment:

```sh
python3 benchmark/report.py result.json \
  --carrier out/benchmark/muse_layer_aa.<sha8>.py \
  --spec benchmark/example_aa.py
```

The reader checks source identity, requested cases, unchanged controls and raw
timing records. It prints a verdict without modifying the result. A comparison
is evidence against its selected control; native correctness and an official
leaderboard score require separate checks.

Run the CPU tests with `python -m pytest benchmark`. They need pytest and
PyTorch, but no GPU.

[Measurement notes](NOTES.md) cover graph replay, timing noise and state ownership.
