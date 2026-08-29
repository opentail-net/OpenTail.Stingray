# TTS performance: baseline + improvement plan (QwenTTS, CosyVoice3)

**Measured:** 2026-08-29, CPU backend (no CUDA/Vulkan involved), prompt
`"Hello, I will make some lunch, darling!"`, seed 42, 1 warmup run (excluded) + 3 timed runs each.
Harness: `tests/OpenTail.Stingray.Tests.Audio/TtsPerformanceBaselineDebugTest.cs`.

## Baseline numbers

| Pipeline | Audio produced | Runs (s) | Mean wall-clock | Real-Time Factor (RTF) |
|---|---:|---|---:|---:|
| QwenTTS (Talker + Code Predictor, Qwen3 backbone) | 2.16 s | 14.475, 14.422, 14.131 | **14.34 s** | **6.64×** slower than real-time |
| CosyVoice3 (LLM + flow/DiT + HiFT) | 2.44 s | 20.918, 21.200, 21.271 | **21.13 s** | **8.66×** slower than real-time |

RTF = wall-clock / audio-seconds; 1.0 would be real-time, lower is better. Both pipelines are
currently far from real-time on CPU (6-9x), which is expected for a from-scratch, non-batched,
CPU-only autoregressive TTS stack with no KV-cache reuse across runs — but there is real headroom.

Runs were tight (±2% run-to-run) after the warmup run, so these are stable baselines, not noisy
one-offs.

## What the numbers already tell us

- Both pipelines log `OpenBLAS: not found (fallback to sequential)` at startup. This is
  **expected, not a bug** — see [docs/done/openblas-elimination-findings-2026-08-20.md](done/openblas-elimination-findings-2026-08-20.md):
  OpenBLAS was deliberately removed from the source tree because it measured strictly worse than
  this codebase's own SIMD kernels on every shape tested. Don't reintroduce it as a "fix."
- CosyVoice3 is slower in both absolute time and RTF than QwenTTS, despite producing only
  slightly more audio (2.44s vs 2.16s) — consistent with it being a longer pipeline (LLM
  token generation *and* a flow-matching DiT with multiple ODE steps *and* a HiFT vocoder pass,
  vs QwenTTS's Talker + Code Predictor + DAC decode).
- Weight pre-faulting (`[ForwardPass] Pre-faulted ... GiB ... GiB/s`) is not the bottleneck —
  it's sub-millisecond per layer and happens once per `ForwardPass` construction, not per token.

## Where to look for real wins (in priority order)

1. **Profile before optimizing anything.** Neither pipeline has been profiled at a function
   level yet — the RTF numbers above tell us *that* it's slow, not *which stage* (LLM/Talker
   decode loop vs. DiT/CFM ODE steps vs. HiFT vocoder vs. vocoder/codec) dominates. Use a
   sampling profiler (e.g. `dotnet-trace collect` around the `Generate()` call, or coarse
   `Stopwatch` brackets around each pipeline stage) before touching any kernel. Per this
   project's own rule (CLAUDE.md "measure, don't assume"), no optimization below should be
   applied without confirming it's measurably faster on real weights afterward.

2. **CosyVoice3's ODE step count is a free, tunable lever.** `CosyVoice3ClipGenDebugTests.cs`
   already shows `odeSteps: 10` as a parameter to `Generate(...)`. Flow-matching quality/speed is
   usually fairly flat past a threshold step count — try 6-8 steps and A/B the audio quality
   (by ear + Whisper transcription, per this project's established practice) against wall-clock
   time. If quality holds at fewer steps, this is a pure win with no kernel work required.

3. **KV-cache reuse across the autoregressive decode loop.** Check whether `CosyVoice3Llm`'s and
   `QwenTtsTalkerGeneration`'s per-token forward calls are re-running attention over the full
   prefix each step (`PrefillCore`-style full recompute) vs. appending only the new token to a
   cached K/V buffer (`PagedKvCache` / the incremental decode path `ForwardPass.Decode.cs`
   already provides elsewhere in the engine for text LLMs). If these audio LLM call sites aren't
   using the incremental decode path, wiring them to it would turn an O(T²) decode loop into
   O(T) — likely the single biggest lever available, since a ~140-token generation quadratic
   in sequence length dominates real time quickly.

4. **Thread utilization in the elementwise/CPU-bound stages.** `Parallel.For` is already used
   throughout QwenTTS's DAC codec/upsample/transformer stages (`QwenTtsCodecDac.cs`,
   `QwenTtsCodecUpsample.cs`, `QwenTtsCodecTransformer.cs`) and `AudioResampler.cs`, but it's
   worth confirming (with the profiler from step 1) that the *actual* hot loop — likely
   matmuls inside the Qwen2/Qwen3 attention/FFN blocks in the shared `ForwardPass` engine, not
   the codec/resampler — is the one getting parallelized, and that `Environment.ProcessorCount`
   isn't being oversubscribed by nested `Parallel.For` calls (codec-level parallelism nested
   inside a per-token loop can cause thread-pool contention that a flat design wouldn't).

5. **GPU backends exist in this codebase (`OpenTail.Stingray.Cuda`, `OpenTail.Stingray.Vulkan`)
   but aren't wired to either audio pipeline** — both `CosyVoice3Llm` and the QwenTTS
   Talker/Code Predictor construct `Cpu.CpuBackend` directly. If CUDA hardware is available,
   wiring these pipelines to `CudaForwardPass` (as already done for the main text-LLM path)
   is likely a much larger win than any CPU kernel tuning, but is also the largest scope item
   here — treat it as a separate follow-up, not part of this initial pass, and validate numerically
   (golden comparison against the CPU path) before trusting the audio output.

6. **Batch the ODE/CFM steps and HiFT frames where possible.** If DiT/CFM currently processes one
   ODE step fully before starting the next serially (expected, since it's iterative refinement),
   there's no obvious batching win there — but check whether HiFT's mel→waveform synthesis or the
   codec's frame-by-frame upsampling could process multiple frames per SIMD-friendly call instead
   of one at a time, similar to how `QwenTtsCodecUpsample.cs` already parallelizes per-frame.

## Suggested next step

Re-run `TtsPerformanceBaselineDebugTest` after each individual change (one at a time, per
CLAUDE.md's performance-pass rule: several samples, keep only measured wins) and record the new
RTF in a table appended to this doc, so the improvement is traceable change-by-change rather than
retroactively.

## Test harness added this session

`tests/OpenTail.Stingray.Tests.Audio/TtsPerformanceBaselineDebugTest.cs` — a temporary debug test
(not part of the fast suite) that runs both pipelines on the same fixed prompt/seed, with one
untimed warmup run followed by 3 timed runs, and prints RTF to console. Re-run via:

```bash
STINGRAY_RUN_HEAVY_TESTS=1 dotnet test tests/OpenTail.Stingray.Tests.Audio -- --filter-class "*TtsPerformanceBaselineDebugTest*"
```
