# OpenVINO INTBIT Fork Guide

This document belongs on the `intbit-projection-groups-2026.2` branch of
`Wondernuttz/openvino`. It describes the fork-specific surface required by
`Wondernutts/Bonsai-27B-INTBIT-OpenVINO-Arc`.

## Scope

The branch is based on OpenVINO `2026.2.0`, commit
`52ddc07385712456dd9f8c5ecf05d7e49c6da329`. It extends the existing Intel GPU
`SimpleGPU` custom-layer path in four files:

The tested INTBIT implementation is commit
`be1b2f3f50947aa9a287833ee0851b2cfcca18b5`.

- `src/plugins/intel_gpu/include/intel_gpu/plugin/custom_layer.hpp`
- `src/plugins/intel_gpu/src/graph/impls/ocl/custom_primitive.cpp`
- `src/plugins/intel_gpu/src/plugin/custom_layer.cpp`
- `src/plugins/intel_gpu/src/plugin/ops/custom.cpp`

The changes add RAW custom-layer ports, multi-output custom operations, and the
`INTBITProjectionGroup` lowering used by the folded Bonsai graph. One primitive
owns one activation quantize/VNNI2-pack operation and all native signed-2-bit by
signed-8-bit DPAS projections that consume it.

This branch is an experimental downstream implementation. It is not official
OpenVINO support and has not been accepted upstream.

The tested package target is Linux x86-64. The source changes are not knowingly
Linux-specific, but no Windows plugin DLL has been built or validated. Do not
claim Windows support until a Windows build passes the same token-hash, serving,
and benchmark checks.

## Build the GPU Plugin

```bash
git clone --branch intbit-projection-groups-2026.2 --single-branch \
  https://github.com/Wondernuttz/openvino.git openvino-intbit
cd openvino-intbit
git checkout --detach be1b2f3f50947aa9a287833ee0851b2cfcca18b5
git submodule update --init --recursive

sudo ./install_build_dependencies.sh
sudo apt-get install -y patchelf

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON=OFF \
  -DENABLE_INTEL_GPU=ON
cmake --build build --target openvino_intel_gpu_plugin -j"$(nproc)"
cmake --install build --prefix "$PWD/install" --component gpu

PLUGIN="$PWD/install/runtime/lib/intel64/libopenvino_intel_gpu_plugin.so"
patchelf --set-rpath '$ORIGIN' "$PLUGIN"
sha256sum "$PLUGIN"
```

Build the plugin from this exact branch and use it with the OpenVINO 2026.2.0
Python wheel. Do not mix it with another OpenVINO minor version.

## Install into a Python Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install \
  openvino==2026.2.0 \
  openvino-genai==2026.2.0.0 \
  'huggingface_hub>=1.19,<2' \
  'numpy>=2,<3'

hf download Wondernutts/Bonsai-27B-INTBIT-OpenVINO-Arc \
  --local-dir ./Bonsai-27B-INTBIT-OpenVINO-Arc

python ./Bonsai-27B-INTBIT-OpenVINO-Arc/runtime/install_plugin.py "$PLUGIN"
```

Run the model using the scripts shipped in the model repository. The model card
contains device selection, generation, serving, benchmarks, checksums, and known
limitations.

## Runtime Contract

The pipeline must receive:

- Python `OpExtension` registrations for every extension type in the IR
- `CONFIG_FILE=custom_layers_u1_27b_projection_groups_v1.xml`
- `KV_CACHE_PRECISION=f16`
- `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
- A SchedulerConfig with prefix caching disabled for the measured path

The reference implementation is `runtime/intbit_runtime.py` in the model
repository.

For an HTTP server, execute all calls into one shared pipeline on one persistent
OS thread. A new request thread per `generate()` call causes cumulative latency
growth on this path even when requests are serialized with a mutex.

## Provenance

The measured lab plugin was built from this source state before portable RPATH
rewriting:

```text
3557b0d03d523effe7e871a0a9e05e34e00797714b74499cce6693c5363b09fe
```

Rebuilt binaries may have a different checksum because of compiler, linker, and
RPATH metadata. Record the source commit, compiler, OpenVINO version, and final
binary checksum for every distributed build.

## Validation

Before distributing a binary:

1. Install it into a clean OpenVINO 2026.2 virtual environment.
2. Confirm `ov.Core().available_devices` loads the plugin.
3. Compile the Bonsai INTBIT model using the reference runtime.
4. Run a fixed greedy token hash.
5. Run the 966-token benchmark three times on a cleared card.
6. Run six sequential HTTP requests and verify latency does not grow.

The published benchmark evidence and exclusions are in the model repository's
`BENCHMARKS.md`.
