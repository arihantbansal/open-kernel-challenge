# Measurement notes

## Graphs can read stale buffers

A captured CUDA graph keeps pointers. Refresh inputs, poison outputs and retain
buffers through the timed replays, then check the outputs again. A correct
warm-up does not cover the replay loop.

## Python copies can leave native state behind

A tensor copy can leave a ctypes helper or cuBLASLt plan pointing at the original
workspace. Keep plans and scratch under the same owner. Test a copied state and
first use from another thread.

## Check which kernel ran

A printed dispatch flag does not prove that a candidate executed. Inspect the
captured kernel names. A fallback result cannot qualify the intended candidate.

## Measure the instrument

Compare two captures of the unchanged implementation before trusting a small
improvement. Alternate execution order and retain raw observations. Repeated
replays on one GPU do not measure variation across GPUs.

## Keep the source with the result

Bind each result to the bytes that ran. A baseline ratio from one official run
and a paired comparison against an earlier implementation answer different
questions. File hashes and job IDs are recorded in [results/](../results/README.md).
