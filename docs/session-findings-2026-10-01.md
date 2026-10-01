# Session findings — 2026-10-01

Everything below was measured or verified this session; commits are on `main`.

## Pushed
- `ef29abda`, `076dc8e1` — item 15 (LLaVA-NeXT AnyRes).

## Committed locally, not yet pushed
- `47f1d116` — item 21, `TriangularSolve.SolveLower`.
- `233d9519` — item 22 logged.
- the Phase 7 doc/League commit (see below), if it landed; check `git log`.

## Item 15 — LLaVA-NeXT AnyRes (closed 🟢 with caveats)
- Real defect: Stingray fed the overview image first; mtmd feeds slices row-major and the overview LAST. Fixed in `LlavaAdapter` (`LlavaImagePreprocessor.OverviewLast`).
- Evidence vs llama-server on LLaVA-1.6-mistral-7b: 672x336 mean first-token |dlogprob| 0.22 -> 0.07; 800x600 top-9 identical order. Token counts match exactly (1759 / 2909), no separator tokens, geometry matches `llama-mtmd-debug`.
- Granite Vision 3.2 shares the adapter and matched llama.cpp token-for-token overview-first on 2026-09-27, so it keeps that order (switch: presence of `clip.vision.feature_layer`). My 10-01 re-check of Granite was inconclusive (prompt differed: 30 vs 70 text tokens), so the 09-27 evidence stands; the new switch itself was only compile-checked, not re-run against Granite.
- Unverified: pixel-level resize/pad, per-view embeddings (no oracle), LLaVA-OneVision.
- Tests: `LlavaImagePreprocessorGeometryTests` 16/16 pass.

## Item 21 — generic SOLVE_TRI (closed 🟢, primitive only)
- `src/OpenTail.Stingray.Cpu/TriangularSolve.cs`: lower-triangular, non-unit diagonal, multi-RHS, batched, ggml semantics. 9 tests in `TriangularSolveTests` (independent double oracle, A.X=B, NaN-poisoned upper triangle, in-place, error cases) all pass. No consumer wired; GDN keeps its specialised solve.

## New item 22 — stale numpy-golden vision parity tests (open 🟡)
Full Vision run: 192 total, 4 failed, 47 skipped. Failing: `LlavaVisionEmbedderParityTests` (min cosine -0.21), `Qwen25Vl/MimoVl/Exaone4 ...EmbedderParityTests.Forward_MatchesNumpyReference`. Not caused by item 15 (encoders untouched). The LLaVA fixture dates from 2026-09-01; the encoder was deliberately changed 2026-09-27 to match llama.cpp. Suggested fix: replace numpy fixtures with `llama-mtmd-debug` oracles (no new Python) or retire the tests. Other three not individually investigated.

## docs/2-coverage
Only two files: `050` (superseded by `1-correctness/17`, which leaves RWKV6/RWKV7/SOLVE_TRI as the real gaps = items 19-21) and `058` (DeepSeek V4->V1; V3.2/V4 are alpha code never run on real weights). Next there: item 20 (RWKV7 WKV7 kernel, scalar-reference test first) then 19 (RWKV6); checkpoints are on `E:\_models`. Not started.

## docs/4-performance — Phase 7 (Mistral-7B CPU decode), measured
- Stingray 7.4-7.5 t/s (3 runs, 16 thr) vs llama-bench 9.34 t/s (8 thr) = 0.80x; llama.cpp at 16 thr = 8.58 -> 0.87x.
- Profile (`STINGRAY_PROFILE_DECODE=1`, needs `-g 0`; `-g -1` is the iGPU and yields an empty profile): FFN 72.7%, QKV 14.9%, out-proj 10.1%, attention 1.7%, rest 0.6%.
- Thread sweep: Stingray 8/12/16 thr = 6.0/7.6/7.5 t/s; llama.cpp is best at 8 threads. So Stingray needs all SMT threads to reach what llama.cpp gets from 8: weaker per-thread streaming.
- QKV and out-proj cost ~3.2 ms/Gparam vs ~2.2 for FFN (~1.45x less efficient). Closing that is worth about 8% decode (~0.87x). Lead only; no code change made, so no speed claim.
- Recorded as a dated row in `PerformanceLeague.md` (supersedes the 0.69x row of 2026-09-10) and in `perf-sweep-plan.md` Phase 7.
- Voxtral phase 1.x not runnable: its benchmark needs `models/_models/voxtral-mini-realtime/model.safetensors`, which is not on this machine (only the Q8_0 GGUF dir).

## Left alone
Scratch files (`docs/tts-benchmark-log.txt`, `stage_diagnostics_report.txt`, `higgs-tts-*`) untouched, per CLAUDE.md.
