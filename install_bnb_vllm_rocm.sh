#!/bin/bash

git clone --recurse https://github.com/ROCm/bitsandbytes
cd bitsandbytes
git checkout rocm_enabled_multi_backend
uv pip install -r requirements-dev.txt
cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx90a;gfx942" -S .
make
uv pip install .
cd ..

git clone https://github.com/vllm-project/vllm.git
source .venv/bin/activate
uv pip install /opt/rocm/share/amd_smi
uv pip install setuptools_scm numba
cd vllm
uv pip install -r requirements-rocm.txt
PYTORCH_ROCM_ARCH="gfx90a;gfx942" python setup.py develop
cd ..
