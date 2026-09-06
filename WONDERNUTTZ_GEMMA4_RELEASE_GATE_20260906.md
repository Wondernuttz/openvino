# September 6 clean Release validation

Follow-up to [the matched grouped-MoE lookup study](WONDERNUTTZ_GEMMA4_PREFILL_20260906.md).

## Release build retained the gain

The GPU plugin, core, IR frontend and Python bindings were rebuilt as Release with debug/profiling capabilities disabled, then packaged in a separate runtime. The original runtime was retained for rollback. Kernel source is exactly the two-file patch published at `ca07f85b2de1d05f13980d8fdaa538f14b01c54e`; the lab version string retains base commit `31c98eec8ab8184820de55e7d443d12b7b89f40b` plus that patch. GenAI remains `79bc246970146922a385b6c0342f185b45478f4b`.

GPU plugin SHA256: `2818d86aa36e34cf7ce3bd5dd2ab27d3660c21baef39ebeb7a360a9c5dea9388`.

Core SHA256: `9792d89988f8813c0a489382298b9d2956d711070fa30a52a1e36616e72f428d`.

Matched Heretic test, 24,576 input tokens, same Release binary, DQ128 / U4 KV / cache8 GiB / batch16384, prefix reuse off, three requests each, median of the last two:

| Lookup | Warm PP tok/sec |
|---|---:|
| Original | 2,710.8 |
| Binary | 4,771.2 |

**76.0% higher prompt throughput** in this acceptance pair. Generated strings in this short-output speed gate matched. This supplements rather than replaces the earlier deeper quality and A/B/B/A study; it is not a new claim about decode or every prompt length.

## Per-checkpoint 24K/cache gates

All eight installed Gemma 26B A4B variants completed 11 automated factual/simple/boundary checks plus four prefix-repeat/corrected-fact checks using this Release binary. Each had a 24,576-token input recall request. Profiling and verbose tracing were disabled; prefix caching was enabled for this table.

| Checkpoint | Recall decode tok/sec | Exact-repeat TTFT ms | Sampled peak GiB |
|---|---:|---:|---:|
| Heretic | 93.87 | 172.77 | 26.48 |
| Chimera-X | 94.56 | 172.80 | 26.48 |
| StyleTune V2 | 94.47 | 172.60 | 26.48 |
| G4 Dark Soul | 94.61 | 101.90 | 26.48 |
| G4 Midnight Macaw | 93.95 | 172.67 | 26.48 |
| Phoenix-X | 94.24 | 172.63 | 26.48 |
| Luminous Mirror | 94.28 | 172.68 | 26.48 |
| Shadow Siren | 94.54 | 172.25 | 26.48 |

These are per-checkpoint measurements, **not per-checkpoint matched speedup ratios**. Exact-repeat TTFT is a cache hit, not raw prefill throughput. VRAM samples are five seconds apart, not true high-water marks. No memory guard tripped in these eight runs. The separate normal-server Chimera canary returned the correct location and corrected password without raw reasoning delimiters.

## Manual RP review is a separate gate

Automated recall success does not imply perfect prose or instruction following. The manually reviewed examples included a paragraph-count/output-cap miss in Dark Soul, Tomas/Thomas spelling drift in Midnight Macaw, and a small invented visitor action in Shadow Siren. These outputs remained grammatical and broadly coherent, but are not perfect agency/format adherence passes.

**StyleTune is held from rollout pending investigation:** its new-runtime 24K sequence produced broken phrasing and changed “yesterday” to “two days ago” in the later RP sample, despite passing the factual tests. A safe-context 6.6K comparison produced identical, grammatical RP text with new-runtime lookup off/on; the old runtime also produced grammatical text. A separate 24K lookup-off sequence did not reproduce the mangled phrasing of the earlier lookup-on sequence. That is an unresolved sample difference requiring repeated matched controls, not proof of a lookup, compression, or upstream-backport root cause. Do not label StyleTune a clean all-quality pass.

These are bounded text-only gates, not an exhaustive intelligence benchmark, native-vision qualification, or a multi-user RP soak. No weight, chat-template, sampler, reasoning, or vision setting was changed to obtain the results. Dense 31B and unrelated model families are outside this rollout.
