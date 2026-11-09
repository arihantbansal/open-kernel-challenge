# Tools

## Verify files and results

```sh
python3 tools/verify.py
```

This checks submission hashes, entrypoints and saved result records using only
Python's standard library.

## Run the tests

From the repo root:

```sh
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m pytest
```

The tests use CPU PyTorch and pytest.

## Check a submission

```sh
python3 tools/precheck submissions/muse_moe_w4a16.py --lint-only
```

This parses the source without running it. Add `--scout PATH` to check imports
against local dependency checkouts.

For compilation and meta tracing, build the Docker environment and omit `--lint-only`:

```sh
docker build -t okc-precheck:cu130-torch2.14.1 -f tools/Dockerfile tools
python3 tools/precheck submissions/muse_moe_w4a16.py
```

The image supports compilation on arm64. It checks native extensions, preparation,
meta-tensor calls and reached Triton specializations. `dock` wraps Docker compiler
commands with memory limits and serialization.

## GPU environment

The submissions were measured on H200 SXM (`sm_90a`) with CUDA 13.0.3,
PyTorch 2.14.1 and Triton 3.8.0. Use the
[challenge environment and problem inputs](https://openkernelchallenge.dev/rules)
for correctness and performance runs. The Docker precheck handles compilation
and tracing; GPU execution uses the challenge environment.
