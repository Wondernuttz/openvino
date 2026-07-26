# Wondernuttz Arc Xe2 INT4 GPU Path

This branch is the source used for the custom OpenVINO Intel GPU plugin tested on Intel Arc Pro B70.
It is based on OpenVINO 2026.2 and carries the low-bit projection work plus the MoE micro-GEMM selection changes used by the Qwen3.6-35B-A3B benchmark.

## Tested stack

- Ubuntu 24.04.4 LTS
- Linux 7.0.0 with the `xe` driver
- Intel Arc Pro B70, 32 GB
- Intel compute runtime 26.22.38646.6
- OpenVINO 2026.2.0
- OpenVINO GenAI 2026.2.0.0
- Python 3.12
- Implementation commit `6306353b0371c2bbb5fad05eea90e0fde3b45642`

The implementation commit adds the `MOE_MICRO_GEMM_N_HINT` override on top of the custom Arc GPU plugin work. For the published Qwen run, the selected value was `256`. The experimental Gemma A4W4 grouped primitive is present in the branch but is not used by the Qwen graph.

## Build the GPU plugin

```bash
git clone --branch arc-xe2-int4-2026.2 --single-branch \
  https://github.com/Wondernuttz/openvino.git openvino-arc-xe2
cd openvino-arc-xe2
git submodule update --init --recursive

sudo ./install_build_dependencies.sh
sudo apt-get install -y patchelf

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_PYTHON=OFF \
  -DENABLE_INTEL_GPU=ON
cmake --build build --target openvino_intel_gpu_plugin -j"$(nproc)"
```

Install the plugin into an OpenVINO 2026.2 Python environment:

```bash
PLUGIN="$(find "$PWD/build" "$PWD/bin" -type f \
  -name libopenvino_intel_gpu_plugin.so -print -quit)"
test -n "$PLUGIN"
patchelf --set-rpath '$ORIGIN' "$PLUGIN"

OV_LIBS="$(python - <<'PY'
from pathlib import Path
import openvino
print(Path(openvino.__file__).resolve().parent / "libs")
PY
)"

cp "$OV_LIBS/libopenvino_intel_gpu_plugin.so" \
  "$OV_LIBS/libopenvino_intel_gpu_plugin.so.stock"
cp "$PLUGIN" "$OV_LIBS/libopenvino_intel_gpu_plugin.so"
```

Do not install a plugin built from a different OpenVINO release into the 2026.2 Python package. Keep the stock backup until the custom build has passed a smoke test on the target machine.

## Qwen3.6-35B-A3B settings

The measured single-card configuration was:

```bash
export MOE_MICRO_GEMM_N_HINT=256
export MOE_USE_GROUPED_GEMM_PREFILL=0
```

```python
import openvino_genai as ov_genai

scheduler = ov_genai.SchedulerConfig()
scheduler.enable_prefix_caching = True
scheduler.cache_size = 9
scheduler.max_num_batched_tokens = 4096

pipe = ov_genai.VLMPipeline(
    "./qwen36-heretic-ov",
    "GPU",
    scheduler_config=scheduler,
    DYNAMIC_QUANTIZATION_GROUP_SIZE=0,
)
```

The 9 GB cache configuration used about 28 GB of device memory. Use a 32 GB card for this exact setup. Smaller cache settings need their own performance and coherence check.

## Clean B70 result

Date: 2026-07-25. One Arc Pro B70 was cleared before each fresh process. A 64-token request warmed compilation. Prefix cache was enabled, but each measured prompt was new and was not served from a cached prefix.

| Input | Prompt processing | Decode |
|---:|---:|---:|
| 2,048 | 3,392.8 tok/s | 106.2 tok/s |
| 6,144 | 4,108.0 tok/s | 106.0 tok/s |
| 16,384 | 3,914.6 tok/s | 106.0 tok/s |
| 32,768 | 3,128.2 tok/s | 102.8 tok/s on the separate short decode case |
| 31,000 retrieval prompt | 3,080.6 tok/s | 83.9 tok/s at long context |

The 31K test recovered three exact facts and completed the requested style continuation. All four checks passed.

## Current limits

- Linux x86-64 on Arc Pro B70 is validated. Windows is untested.
- These numbers are single-stream and single-card.
- The Qwen artifact is an INT4 model. It does not use the Bonsai INTBIT or INTERNARY projection kernels.
- `DYNAMIC_QUANTIZATION_GROUP_SIZE=0` is part of the tested configuration.
- The Gemma-4 31B StyleTune VLM graph is not release-ready on this stack. The plain pipeline faults after repeated requests, while continuous batching avoids the fault but fails coherence. No Gemma benchmark should be published until that is fixed.
