# Gemma 4 26B A4B on Arc Pro B70: faster long-prompt processing

September 6, 2026 — Wondernuttz OpenVINO fork, branch `arc-xe2-gemma4-pa-2026.4`.

Follow-up: [clean Release acceptance and per-checkpoint 24K/cache results](WONDERNUTTZ_GEMMA4_RELEASE_GATE_20260906.md), including the unresolved StyleTune RP difference and rollout hold. The measurements below retain their original binary and methodology.

An opt-in grouped-MoE token-row lookup optimization improves measured long-prompt processing without changing model weights, expert routing, KV precision, or floating-point accumulation order. **These measurements are for Gemma 4 26B A4B Heretic, not automatically for every fine-tune, dense 31B, or every Intel GPU.** Runtime changes do not require re-exporting the model.

## Matched results: one B70, 32 GB VRAM

Same model, GPU, library binary, DQ128, U4 KV, 8 GiB cache allocation, batch16384, one sequence, prefix reuse **disabled**. Only the lookup switch differs. Metric: input tokens divided by time to first token, not isolated kernel throughput.

| Input tokens | Original lookup PP tok/sec | Binary lookup PP tok/sec | Gain |
|---:|---:|---:|---:|
| 4,096 | 6,681.7 | 7,224.4 | 8.1% |
| 6,622 | 6,028.1 | 7,435.1 | 23.3% |
| 15,872 | 2,920.9 | 6,866.4 | 135.1% / 2.35x |
| 24,576 | 2,734.8 | 4,767.0 | 74.3% |

The first three rows use A/B/B/A process order and four warmed observations per configuration, excluding first shape observations. The 24K row uses one patched/control process pair with two warmed observations per configuration. These are measured points, not a universal gain curve or confidence interval.

At 24K, warmed TTFT falls from ~8.99 to **5.15 seconds**. Long-recall decode is 93.76 original / 93.29 patched tok/sec; short decode is approximately 111 tok/sec. **No meaningful decode improvement is established.** The patch changes prefill lookup work, not the decode kernel.

Sampled peak card VRAM is approximately 26.5 GiB at 24K. Five-second sampling is not an exact allocation high-water mark. The test has 24,576 input tokens plus generated output; a serving limit defined as total tokens must reserve output space.

## Prefix caching and longer context

With the same patched batch16384 profile and prefix caching enabled, an exact repeat of the 24,576-token chronicle reached first token in **172.8 ms**, decoding at 95.83 tok/sec. This is a cache-hit latency, not 142K raw prefill throughput. An updated-password probe used the new fact rather than the revoked password. It also answered older embedded questions, so fact freshness passed but strict answer-only formatting did not. The correction probe is 6,651 tokens, not a 24K updated conversation.

Separate patched-only runs at 32,768 input tokens passed the bounded recall checks: ~4,035.5 PP tok/sec with batch16384, ~89.5 tok/sec long decode, ~26.6 GiB sampled peak VRAM. That is not a matched baseline comparison at 32K and not a production multi-user stability certification.

## Mechanism and scope

Non-offloaded grouped MoE builds per-expert token lists in ascending order. The old scatter reduction repeatedly scans those lists to recover each output row. The new path uses one lane per selected expert and a bounded binary search. It retains missing-row checks, barriers, expert order, router weights, FP operations, and output stores.

The switch is **off by default** and only applies when grouped prefill is active and expert offloading is absent:

```bash
export MOE_USE_GROUPED_GEMM_PREFILL=1
export MOE_GROUPED_BINARY_LOOKUP=1
```

Set these before creating/compiling the pipeline. To disable the new lookup, unset `MOE_GROUPED_BINARY_LOOKUP` or set it to `0`, then recreate the pipeline. Stock wheels without this patch do not gain this kernel merely by setting an environment variable.

Important correction to earlier tuning attribution: `MOE_MICRO_GEMM_N_HINT` controls a separate generator bypassed by the grouped path. It must **not** be credited for an active grouped-GEMM tile change. All lookup results here retain the original grouped-matmul selector. A separate grouped-selector experiment was not successfully built or measured and is not claimed as an optimization.

## Reference text-only pipeline configuration

Use a matching OpenVINO 2026.4 build of this fork and compatible GenAI. The tested GenAI source revision is `79bc246970146922a385b6c0342f185b45478f4b`. Do not drop a 2026.4 plugin into a 2026.2 runtime. The model must already have a valid Gemma 4 export/tokenizer and the appropriate RoPE-LUT graph treatment; this lookup patch does not repair arbitrary broken exports.

```python
import os
os.environ["MOE_USE_GROUPED_GEMM_PREFILL"] = "1"
os.environ["MOE_GROUPED_BINARY_LOOKUP"] = "1"
import openvino_genai as g

s = g.SchedulerConfig()
s.max_num_batched_tokens = 16384
s.max_num_seqs = 1
s.cache_size = 8
s.enable_prefix_caching = True  # False for uncached comparisons
pipe = g.VLMPipeline(
    "/path/to/local/gemma4-26b-openvino-model", "GPU.N",
    ATTENTION_BACKEND="PA",
    DYNAMIC_QUANTIZATION_GROUP_SIZE=128,
    KV_CACHE_PRECISION="u4",
    scheduler_config=s,
)
```

Replace `GPU.N` with the verified device index/PCI mapping for your Arc card. Preserve the model's valid chat template, reasoning framing, and sampler. `cache_size` is a memory allocation, not a token-context setting. This example is not validation of native vision or audio. Multiple active conversations require separate capacity testing.

## Quality gates and limitations

* All 11 automatic factual/simple/boundary checks passed in each matched full-model run. The 24K pair produced identical generated strings, including the RP sample. A/B and C/D at shorter lengths also match; a free-form variation between pairs also occurs in the unmodified control.
* All 27 existing MoE CPU-reference smoke tests passed with lookup enabled, including shared-expert cases.
* One isolated expanded MoE test produced byte-identical GPU tensors with lookup off/on: 65,536 half values, SHA256 `f0f7d77dcdf6201d5feb11043149a6ae26b0c66a6abbead2b25f4fe87b463be4`.
* Four expanded CPU-reference shapes fail the existing absolute tolerance with lookup **off**. Their reference uses unquantized float weights. They are not reported as passes, and tolerances were not relaxed. A repeated expanded baseline sequence also triggered a GPU fault; subsequent isolated and full-model checks passed. This is not a claim that every baseline fault is fixed.
* Dense Gemma 31B does not use this MoE optimization. Its larger prefill batch delivered only ~1% at 6K with higher memory use. PA cache6/FP16 rejects 8K for insufficient cache budget. SDPA remains unqualified: its earlier repeated one-token-output failure is unresolved; an 8K attempt crossed a 29 GiB guard and a compute-engine reset was logged during termination. Do not apply 26B gains or its U4 cache validation to 31B.
* These are bounded regression/coherence probes, not an exhaustive intelligence evaluation or a multi-user RP soak.

## Provenance, build, and credits

The measured base before the two-file lookup patch is `31c98eec8ab8184820de55e7d443d12b7b89f40b`. It preserves this fork's earlier Gemma/Arc work and incorporates eight Intel upstream backports: DynamicQuantize deduplication, Gemma image/token mask handling, DQ inner-width cache keys, sliding-window decode skipping, 512-head and MIXED-stage micro-SDPA support, and U8/U4 cache boundary/zero-range corrections. Git history retains upstream authorship. No claim is made that Intel adopted code from this fork.

Measured lookup binary SHA256: `498c30cdff1b0e40abbcf4ba0355e766f808fce761eab9c244a198182b30c9da`. It is Release/O3 with debug/profiling capability compiled in, but counters and verbose tracing disabled for accepted speed measurements. A clean-release deployment gate is separate; do not deploy a mutable lab build directory or assume a different binary inherits these exact results.

Build from this branch with the matching dependencies and GenAI ABI. The [historical build notes](WONDERNUTTZ_GEMMA4_ARC.md#build) describe the GPU plugin target. For a production candidate explicitly set `CMAKE_BUILD_TYPE=Release`, `ENABLE_DEBUG_CAPS=OFF`, and `ENABLE_GPU_DEBUG_CAPS=OFF`; build matching core/plugin/frontend components and test in a separate runtime. Preserve the old runtime for rollback.

Base-model and fine-tune authors retain their respective credit; Wondernuttz's contribution here is the runtime lookup optimization, integration, and B70 testing. Model cards retain source links and licenses. New measurements are reference-stack results for related tunes until those exact checkpoints are benchmarked.
