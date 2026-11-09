# Compile-only stand-in for the judge image (lmsysorg/sglang:dev-muse-spark-open) on an
# arm64 host without a GPU. Used by tools/precheck; build it with
#   docker build -t okc-precheck:cu130-torch2.14.1 -f tools/Dockerfile tools
#
# Judge (x86_64)                         This image (arm64)
#   CUDA 13.0.3 toolkit, nvcc              CUDA 13.0.3 devel (same toolkit release)
#   Python 3.12 (/opt/sglang venv)         Python 3.12 (Ubuntu 24.04)
#   torch 2.14.1 (cu130)                   torch 2.14.1+cu130 aarch64 wheel (same headers)
#   triton 3.8.0 (judge logs print it)     triton 3.8.0 aarch64 wheel
#   sglang, sgl_kernel, flashinfer, ...    not installed: precheck stubs them
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04@sha256:b7ae301dea2c162444795462ce17a05f6a516e5a75944b57af5b88540a1a2266

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-dev python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    CUDA_HOME=/usr/local/cuda \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu130 "torch==2.14.1+cu130" \
    && pip install --no-cache-dir "triton==3.8.0" "ninja==1.13.0" "numpy==2.3.4"

# No GPU driver here: expose the toolkit's libcuda stub so anything that links or looks up
# libcuda.so.1 (Triton's driver helper) finds it instead of asserting.
RUN mkdir -p /opt/libcuda-stub && ln -s /usr/local/cuda/lib64/stubs/libcuda.so /opt/libcuda-stub/libcuda.so.1
ENV TRITON_LIBCUDA_PATH=/opt/libcuda-stub

# Fail the build if the pins drifted.
RUN python -c "import torch, triton; assert torch.__version__.startswith('2.14.1'), torch.__version__; \
assert triton.__version__ == '3.8.0', triton.__version__; assert torch.version.cuda.startswith('13.0'), torch.version.cuda; \
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__)" \
    && nvcc --version | tail -2

WORKDIR /work
