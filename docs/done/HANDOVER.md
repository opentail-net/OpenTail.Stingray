# OpenTail.Stingray performance campaign — handover

**Written:** 2026-07-26. **For:** the next AI picking up this work.
**Read `docs/perf-loop-progress.md` first** — it is the source of truth, ~2800 lines, 46 numbered
iterations. This file is the orientation layer, not a replacement for it.

---

## 1. What this project is

`C:\git\opentail\extensions\OpenTail.Stingray` — a pure C#/.NET 10 LLM inference engine (a fork of
SharpInference). No native inference dependency; the whole point is that the kernels are C#.
It targets NativeAOT eventually.

Backends, each with its own forward pass in `src/OpenTail.Stingray.Engine/`:

| Backend | Forward pass | Notes |
|---|---|---|
| CPU | `ForwardPass.cs` | AVX2/FMA intrinsics via `src/OpenTail.Stingray.Cpu/SimdKernels.cs`. No VNNI on this box (Zen 3). |
| Vulkan | `GpuForwardPass.cs` | GLSL compute shaders as C# string constants in `src/OpenTail.Stingray.Vulkan/Shaders.cs`. |
| CUDA | `CudaForwardPass.cs` | **Cannot be measured here — no NVIDIA hardware.** Do not claim CUDA results. |

Reference model for all measurements: `models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf`
(llama arch, 24 layers, headDim 64, 32 heads, 8 KV heads, GQA).

**Hardware this was all measured on:** 6-core/12-thread Zen 3, integrated AMD Radeon (~8 CU),
measured memory ceiling **36.8 GB/s**. Every number in the progress log is from this box. A
different box invalidates the *magnitudes*, not usually the *rankings*.

---

## 2. The working agreement (non-negotiable — the user has restated these repeatedly)

1. **Never commit.** The user reviews and commits. Leave the tree dirty and say what is in it.
2. **Never touch `.git/` internals.** Never read paths matched by the repo-root `.claudeignore`.
   Runtime DBs (`opentail.db`, `*.sqlite`) and `opentail.local.json` / `opentail.*.local.json`
   are off-limits.
3. **Measure before optimizing.** Establish the ceiling by *ablation* before building the fix —
   this has repeatedly changed the decision (see §5).
4. **An isolated microbenchmark win is NOT sufficient to ship.** Iteration 24 is the standing
   lesson: a reproduced 2.4–2.6x single-threaded microbenchmark win was a real **11.9% end-to-end
   loss** under production's 12-way `Parallel.For` contention. Always confirm with a same-harness
   A/B (stash → rebuild → re-measure), multiple runs, honest spread.
5. **Noise floor is ±3–4%.** n=2 is a smoke test, not a result. Anything under ~8% needs n=6.
6. **CPU benchmarks require `DOTNET_TC_QuickJitForLoops=0`.** Without it tiered JIT corrupts the
   numbers outright (iteration ~12 chased a phantom for an hour because of this). The in-process
   xUnit runner does *not* set it.
7. **After ANY Vulkan shader edit, run `./scripts/gen-spirv.ps1`** (PowerShell) or
   `VulkanPrecompiledShaderTests` fails on SPIR-V drift.
8. **Keep `tests/OpenTail.Stingray.Tests.ForwardPass` green** (1165 tests at last run).
9. **Delete scratch/diagnostic test files before finishing.** Permanent tests stay; temp ones go.
10. **Record the outcome in `docs/perf-loop-progress.md` either way.** Negative results with
    reasons are explicitly as valuable as wins. Roughly half the 46 iterations are documented
    negatives — that is the point, not a failure.

**Communication style the user wants:** terse. No trailing summaries. No preamble. Do not claim
something is done without a clean build.

---

## 3. Current state of the tree

The user has committed iterations up to and including **46**. Iterations 47–48 are documented but
not committed. `git status` should show:

```
 M extensions/OpenTail.Stingray/docs/perf-loop-progress.md              <- iterations 47–48 results
?? extensions/LLamaSharp/                                           <- pre-existing, unrelated, ignore
?? extensions/OpenTail.Stingray/docs/HANDOVER.md                         <- this handover
```

**Known unrelated failure:** `ConcurrencyLimitTests.BoundedQueue_RejectsOnlyAfterActiveAndWaiting
CapacityIsConsumed` in the Server suite fails. I stash-tested this on a clean tree and confirmed it
is **pre-existing and unrelated**. Do not chase it.

---

## 4. TASK #8 RESOLVED — iteration 47, not shipped

The weighted-V prefetch and bit-identical loop-interchange experiments are complete and fully
reverted from `ForwardPass.cs`.

- Temporal prefetch distances 8/16/32/48 and NTA 16/32 were screened.
- PF16's apparent +11.4% at n=2 collapsed to **+3.14% at n=6**, inside the ±3–4% noise floor.
- A KV-group-fused loop interchange preserved every `(head,d)` accumulation order and tested
  `dSplit=1/2/4`; all were within noise.
- Combining group fusion with PF16 did not add a measurable win.

The full samples and verdict are in iteration 47 of `docs/perf-loop-progress.md`. Do not revive
prefetch or the same group-fusion design without new evidence.

---

## 5. Task #8 correction — why the +46% ceiling was misleading

**The problem.** CPU decode attention's weighted-V pass reads the KV cache with an 8 KB stride.
`Parallel.For` runs over heads, so head *h* touches only bytes `[h*hd, h*hd+hd)` of every KV row —
a jump of the whole row (`numKvHeads*headDim` floats = 8 KB) per iteration. Hardware prefetchers
do not follow strides past a page, so essentially every read exposes full memory latency.

**The measured make-V-free ceiling was +46%.** Iteration 46 ablated it — replaced
`ValueAt(rl, startLocal + i)` with a
fixed row `startLocal`, so identical instruction count and FLOPs but the row stays in L1. The delta
was **11.5 → 16.8 t/s** at 3239 tokens.

**Iteration 47 correction:** that ablation also removed all compulsory V traffic. It therefore
conflated stride, cache residency, and bytes moved; it was not a recoverable access-order ceiling.
Both bit-identical access-order fixes were tried and landed within noise.

**Iteration 48 tested the remaining bandwidth explanation directly.** A true half-width CPU bf16
cache with AVX2 reads was flat; after widening each K row once for all query heads and each V chunk
once for its four GQA heads, fp32 was 54.4/13.5 prefill/decode versus bf16 53.2/13.6. Reverted.

The +46% gap is therefore still not fully explained: V is only ~11% of nominal per-token bytes, and
neither hiding/reordering its access nor halving all KV bytes reproduced the ablation. It may expose
a latency effect or another consequence of repeatedly using one row, but it is **not a sizing model
for future work**. Do not reopen task #8 without a new, isolating ablation.

---

## 6. Everything else still open, ranked

### Tier 1
- **CLOSED by iteration 48 — bf16 KV for the CPU reference path.** True bf16 storage plus AVX2
  reads and GQA conversion reuse was flat in decode and slightly slower in prefill. Fully reverted.
  Revisit only as a memory-capacity feature or after a new substantially-longer-context ceiling.
- **(A) CPU prefill attention is still O(N²).** Iteration 33 tiled it (+56%) but did not change the
  complexity class; still ~65% of prefill at 6.4k tokens. The only true complexity fix left and the
  most work. The Vulkan flash kernel (iterations 31/44) is the template.

### Tier 2
- **(D) Q8_K activations for CPU prefill Q4_K** (~10–20%, untried). Our Q8_KS carries eight scales
  per super-block, forcing a `cvt`+`fma` per sub-block; Q8_K's single scale lets the int32
  accumulate across all eight. From the iteration 37 llama.cpp read.
- **(E) q8_0 flash variant.** Iteration 44 gave bf16 the Vulkan flash kernel; q8_0 still falls back
  to the O(N²) path and measured worst of the three (15.5 t/s prefill). Needs block dequant in the
  tile load — more than an accessor swap.

### Tier 3 — cheap, uncertain
- **(F) Vulkan split-K slice sweep.** llama.cpp sizes split-K by occupancy, which computes
  `split_k = 1` on this ~8-CU part; iteration 36 measured 512→256 as +42–64%. One of the two is
  wrong for this device. Sweep 1024/512/256/128 plus a no-split control. Cheap, and it resolves a
  live contradiction in the log.
- **(G) Drop LDS staging in the Vulkan flash kernel.** llama.cpp explicitly disables shared-memory
  K/V staging on AMD and reads through cache. We stage. One-parameter experiment — now needs doing
  in **both** flash variants (fp32 and bf16).
- **(H) `subgroupShuffleXor` reductions in the FA kernel.** Would correct something the log records
  as settled. Tempered: iteration 27 found no win in the matvec, though FA reduces far more.

### Tier 4 — low ceiling; say so rather than forcing them
- **(I) Q4_K repack default flip.** Kernel done and correct, +14%, shipped opt-in. Now *gateable*
  since iteration 45 built the Vulkan perplexity path (that was the blocker), but it costs a second
  copy of the weights (~5.6%) and loses mmap sharing.
- **(J) Vulkan 2048×2048 matvec.** Weakest shape (~60% of the 35.5 GB/s ceiling) but only ~20% of
  per-layer weight traffic — **arithmetically capped near +2%.** Not worth doing.
- **(K) Q6_K DP4A gate.** Sibling of iteration 43. The gate would say *no* on this device; only
  helps other people's GPUs.
- **(L) 16-token inner tile.** 32 live `Vector256` against 16 architectural YMM registers; RyuJIT
  spills far worse than clang. Flagged least valuable when found. Iterations 41/42 did 8.

**Also carried, not investigation:** CUDA's KV default stayed fp32 (no hardware to measure on), and
iteration 46 was validated end-to-end on llama arch only.

---

## 7. The one pattern worth internalising

**Six times now the win has been a fast path that already existed but was not being taken:**

| # | Iteration | The path that was silently skipped |
|---|---|---|
| 1 | 26 | per-token weight streaming |
| 2 | 31 | per-query K/V |
| 3 | 32 | SnapKV auto-enable disabled the batched Vulkan prefill trunk |
| 4 | 33 | per-token K/V on CPU |
| 5 | 34 | flash-decoding gated at 4096 |
| 6 | 44 | flash attention gated to fp32 KV only |

Every one was found by **measuring which path actually executes**, not by reading the kernel. When
a result is disappointing, check the dispatch condition before you optimise the arithmetic.

**Second pattern: confounded A/Bs.** Iteration 45's first bf16-vs-fp32 comparison was invalid
because fp32 auto-enabled SnapKV and bf16 did not — caught only by diffing the greedy output text.
When two configs differ in throughput, verify they did the *same work*. Greedy-decode text diff is
a cheap and effective confound detector.

**Third: gate design.** Iteration 43 built a DP4A capability probe that measured the intrinsic as
bit-exact in isolation (8192/8192) while the real kernel it gated was corrupt (1.0–4.15% error).
**Probe the real kernel, not the primitive.**

---

## 8. Practical mechanics

```powershell
# Build just the CLI (fast, ~6-14s). Use ABSOLUTE paths — the shell cwd resets to C:\git\opentail.
dotnet build C:\git\opentail\extensions\OpenTail.Stingray\src\OpenTail.Stingray.Cli\OpenTail.Stingray.Cli.csproj -c Release --nologo -v q

# Vulkan shader edits — MANDATORY after any change to Shaders.cs
./scripts/gen-spirv.ps1

# Tests (xUnit v3 / Microsoft.Testing.Platform — NOT the classic runner)
dotnet test <proj> -- --filter-method "*Name*" --output Detailed
```

**Gotchas that cost real time:**
- xUnit v3: there is no `Xunit.Abstractions`. `--logger` is invalid; use `-- --filter-method`.
- `Assert.True(false, msg)` trips analyzer xUnit2020 → use `Assert.Fail(msg)`.
- Passing tests suppress stdout. To capture diagnostics, temporarily insert `Assert.Fail("TEMP-DIAG")`
  — and remember to remove it.
- `TensorShape` takes `long[]`: `new TensorShape([count])`.
- The CLI's run verb is the **default command** — `opentail-llm-cli.exe -m ... -p ...`, no `run`.
- Shader constants: a `private const string` fragment is excluded from the SPIR-V precompiled table
  by an `!f.IsPrivate` filter in **both** `tools/SpirvGen/Program.cs` and
  `VulkanPrecompiledShaderTests.cs`. Split shaders into private fragments + internal complete
  variants; keep both filters in sync or the table test fails.

**Measurement env:** `DOTNET_TC_QuickJitForLoops=0` always; `STINGRAY_SNAPKV_BUDGET=0` when
isolating anything other than the default-path behaviour (SnapKV auto-enable will otherwise silently
change how much context is processed and confound the comparison).

---

## 9. Cumulative results so far (for context on what "good" looks like)

- Vulkan bf16 prefill: 20.3 → **54.9 t/s**
- Vulkan default path (fp32+SnapKV → bf16): prefill 47.6 → **52.2**, decode 6.1 → **9.4 t/s**
- Perplexity cost of the bf16 default: **+0.023%**
- CPU decode at long context (iteration 35, score pass): +15–22%, bit-identical
- CPU prefill phase-1 (`_4In`): closed the gap to llama.cpp to ~4.6x

The stop condition recorded in the progress log: only stop when every listed item has been tried
and either shipped or ruled out with a documented honest verdict, **and** no new concrete avenue is
identified after actually looking at the current state of the code. "We've tried the obvious things"
is explicitly not the bar.
