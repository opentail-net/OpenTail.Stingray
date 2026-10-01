# Plan: Q8_K activation quantization for the Q5_K decode matvec (GLM-4.5 PPL gap)

**Implementation record:** [closed bugstofix scope 09](09-glm45-q5k-activation-implementation.md).
**Open real-weight follow-up:** [item 19 in `docs/103-quickest-first-plan.md`](../103-quickest-first-plan.md).
Item 7 in that file is closed for implementation and synthetic validation. The archive and follow-up
carry the diagnostic history and exact numbers this plan continues from. Do not re-run the earlier
diagnostic steps (layer bisection, print-resolution caveat, etc.) — they are already done and their
conclusions are trusted.

## What's already been established (do not re-derive)

- Second-half wikitext PPL at `-c 2048`: ours 8.7753 vs `llama-perplexity` 8.6125 (1.9% worse).
- Generation is coherent, top-k order matches the reference — the gap is a diffuse numerical drift,
  not a broken op or a logic bug.
- Layer-by-layer bisection (2026-09-27, 2026-09-28) on the real checkpoint
  (`K:\_other_models\cerebras_GLM-4.5-Air-REAP-82B-A12B-Q2_K.gguf`), 326-token wikitext prompt
  (first 1400 bytes of `scripts/kvarn-gate/wiki.test.raw`, tokenized with our own tokenizer —
  already confirmed matching `llama-tokenize` exactly), last-token row:
  - Raw attention output (pre-`wo`, our `StageCapture` stage `attn_out` / llama.cpp's `kqv_out-N`)
    matches the reference **exactly** (last-digit difference is print rounding, not drift). This
    rules out RoPE, the QK score computation, and softmax.
  - The divergence (~2e-4, above the print-resolution noise floor established during the LFM2
    bisection — trust nothing at or below 1e-4) **first appears immediately after `wo`** (our stage
    `o_proj` / llama.cpp's `node_26`/`ffn_inp-N`), which is the model's first Q5_K-quantized matmul.
  - **Leading, now well-supported hypothesis:** our decode-path Q5_K matvec (`SimdKernels.DotQ5K`)
    keeps activations as F32 and dequantizes the weight per-element, while ggml quantizes
    activations to Q8_K for this matmul (`ggml_vec_dot_q5_K_q8_K`). Different intermediate rounding
    behavior, accumulating over 46 layers, is consistent with everything observed so far.
  - This hypothesis is **not yet confirmed as the actual cause** — implementing it and re-measuring
    is the point of this plan. If it doesn't close the gap, that itself is a real, useful result:
    document it and hand the investigation back rather than guessing further.

## Task

Add a Q8_K-activation variant of the Q5_K dot product, wire it into the decode path behind a
feature gate, verify it against the real ggml reference math, then re-measure both the layer-0
`wo` divergence and the full PPL to see whether it closes the gap.

### Step 1 — Read the real reference before writing any code

Per this project's CLAUDE.md (rule 8): a piece of quantization math that looks fiddly is often the
real, correct algorithm — read the actual C++ before "fixing" anything.

- **The exact function to port:** `ggml_vec_dot_q5_K_q8_K` in
  `examples/llama.cpp/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c` (line ~2216 as of this
  writing — confirm the current line, it may have moved). Read the whole function, not just the
  AVX2/AVX-512 intrinsic path — understand the scalar reference first if one exists in the same
  file, then the vectorized version.
- **The existing, working precedent to mirror the *shape* of (not copy the math from — Q5_K's
  layout differs from Q8_0's):** `SimdKernels.DotQ8_0_Q8K` in
  `src/OpenTail.Stingray.Cpu/SimdKernels.cs` (~line 7396). It has two overloads: one taking a
  pre-quantized `byte* scratch` (Q8_K format) and one taking `float* input` that quantizes to a
  local scratch buffer and calls the first. Follow the same two-overload shape for the new
  function. The Q8_K quantizer it uses already exists and should be reused as-is — do not write a
  second one.
- **The Q5_K block layout** (176 bytes per 256 elements: 2B fp16 `d`, 2B fp16 `dmin`, 12B packed
  6-bit scales/mins, 32B high bits (`qh`), 128B low nibbles (`ql`)) is documented in the doc comment
  above the existing `DotQ5K` function (`SimdKernels.cs`, search for "Q5_K Fused Dequant-Dot"). Read
  that comment and the function body — it already does the correct scale/min unpacking
  (`GetScaleMinK4`); the new function needs the same unpacking, paired against Q8_K's per-block
  scale and `bsums` (sum of quantized activations per sub-block) instead of raw float activations.
  Q5_K's asymmetric `d`/`dmin` (scale *and* min, not just scale) is the main place a naive port of
  the Q8_0-vs-Q8K pattern will go wrong — the `min` term needs to multiply the Q8_K block's `bsum`,
  not a naive per-element sum, exactly as the real ggml reference does it. Do not guess this; read
  it from the reference.

### Step 2 — Implement `DotQ5K_Q8K`

- Add to `src/OpenTail.Stingray.Cpu/SimdKernels.cs`, next to the existing `DotQ5K`/`DotQ8_0_Q8K`
  functions (there's a comment there already anticipating this: "this lives next to
  DotQ4K/DotQ5K/DotQ6K", ~line 3449).
- Two overloads, mirroring `DotQ8_0_Q8K`:
  - `DotQ5K_Q8K(byte* row, byte* q8kScratch, int cols)` — the real integer/fixed-point math.
  - `DotQ5K_Q8K(byte* row, float* input, int cols)` — quantizes `input` to a local Q8_K scratch
    (stack-alloc when small enough, heap fallback for large `cols` — follow the existing
    stack/heap-fallback discipline used elsewhere in this file for Q8_K scratch buffers) and calls
    the first overload.
- Do **not** modify `DotQ5K` itself or change what it returns. There is an existing test,
  `SimdKernelsQ8KSTests` (`tests/OpenTail.Stingray.Tests.ForwardPass.Fast/SimdKernelsQ8KSTests.cs`),
  that asserts `DotQ5K_2In` is bit-identical to two independent `DotQ5K` calls — changing `DotQ5K`'s
  numerics in place would break that invariant test for a reason unrelated to this fix. This is a
  new, additional function, not a replacement.
- A scalar (non-AVX2) fallback path is required too, following the pattern every other Dot* function
  in this file uses (`if (!Fma.IsSupported) return DotXxx_Scalar(...)`). Don't skip it even if you
  only benchmark on an AVX2 machine — `TreatWarningsAsErrors`/CI discipline aside, other real
  hardware in this project's test matrix may not have FMA.

### Step 3 — Wire it into the decode path behind a gate

- `SimdKernels.DotQ5K` is called from multiple places (`grep -rn "SimdKernels.DotQ5K\b" src`):
  `ForwardPass.Moe.cs`, `HybridGdnForwardPass.cs`, `VulkanHybridGdnForwardPass.cs`,
  `CudaHybridGdnForwardPass.cs`, plus `MatVecQ5K` and the paired-dot variants inside
  `SimdKernels.cs` itself. Do not infer the relevant dispatch from GLM-4.5-Air being a MoE model.
  Before editing, identify the exact function used for layer-0 attention output projection (`wo`),
  trace its call to `DotQ5K`, and record the call-site path. Modify only the confirmed path; do not
  blanket-replace other call sites.
- Add a feature gate, following the existing precedent (`SimdKernels.Q8PrefillEnabled`, a
  process-wide static bool, referenced in `PrefillDecodeSelfConsistencyTests.cs` for how a similar
  gate is tested/saved/restored). Something like `SimdKernels.Q5KDecodeQ8KActivations` (name it
  however reads best in context) — default **off**, so no existing model's behavior changes until
  explicitly enabled. This lets the fix be A/B tested against the F32 path without risking a silent
  regression on every other Q5_K checkpoint in this codebase.
- Wire the gate into whichever dispatch site(s) you confirmed in the previous bullet: when on, call
  `DotQ5K_Q8K` instead of `DotQ5K`.

### Step 4 — Verify the kernel math in isolation before touching real weights

- Before running anything on the 82B checkpoint, write a correctness test comparing
  `DotQ5K_Q8K` against a direct scalar translation of ggml's `ggml_vec_dot_q5_K_q8_K` reference
  formula. Both sides must consume the same encoded Q5_K row and the same Q8_K-encoded activation
  buffer. Do not use `DotQ5K` as the correctness oracle: it computes the F32-activation path and is
  useful only for measuring activation-quantization error.
- Use synthetic, seeded Q5_K rows and float activations at a few `cols` sizes (the existing
  `SimdKernelsQ3KQ8KTests.cs` and `SimdKernelsQ8KSTests.cs` show the established test pattern). For
  each seeded vector report F32 `DotQ5K`, Q8_K `DotQ5K_Q8K`, absolute error, and relative error; run
  enough seeds to establish and record the empirically observed error range rather than asserting a
  universal tolerance. Separately assert that the Q8_K kernel matches the scalar ggml translation
  within a numeric tolerance justified by floating-point accumulation/order differences. State the
  bound before evaluating results and report max absolute and relative error; do not choose a bound
  after seeing the output. Synthetic tests establish kernel correctness and characterize
  quantization error; add captured real GLM activation rows as a later, stronger validation when
  available.
- If the kernel disagrees with the scalar ggml translation, fix it here, in isolation, before
  spending any time on the 46GB real-checkpoint reload cycle.

### Step 5 — Run the focused four-arm comparison and verify layer 0

Use the following small matrix to separate the GLM hypothesis from generic Q5_K behavior:

| Model/path | Activation | Purpose |
| --- | --- | --- |
| GLM-4.5 | F32 (gate off) | Existing baseline |
| GLM-4.5 | Q8_K (gate on) | Proposed fix |
| Small known Q5_K model | F32 (gate off) | Regression baseline |
| Small known Q5_K model | Q8_K (gate on) | Generic behavior check |

For GLM-4.5, capture layer-0 `wo`, the first few logits, and (in Step 6) PPL in both activation
modes. Compare the layer output and logits against the corresponding reference outputs where
available, and compare gate-on/off results directly. Record raw values; the rounded callback output
is not an adequate measurement.

Reuse the exact harness already built for this (don't rewrite it):

- `tests/OpenTail.Stingray.Tests.ForwardPass/ZzLayerDumpTmp.cs` — dumps our engine's
  `attn_out`/`o_proj`/`post_attn_resid`/`post_ffn_resid`/`attn_norm` stages for the last token of a
  given prompt. Env vars: `ZZ_DUMP=1`, `ZZ_MODEL`, `ZZ_IDS_FILE`, `ZZ_OUT`.
- `tests/OpenTail.Stingray.Tests.ForwardPass/ZzMakeGlmIdsTmp.cs` — tokenizes the first N bytes of a
  text file with the real model's own tokenizer and writes the first K token ids to a file. Env
  vars: `ZZ_MKIDS=1`, `ZZ_MODEL`, `ZZ_FILE`, `ZZ_BYTES=1400`, `ZZ_COUNT=326`, `ZZ_OUT`.
- Enable the new gate (`SimdKernels.Q5KDecodeQ8KActivations = true`, or whatever it ends up named)
  for this run only — check how `ZzLayerDumpTmp.cs` is structured and add a way to flip the gate
  before the dump (an env var read at the top of the test, matching the existing `ZZ_*` convention,
  is the simplest fit).
- Compare the new `o_proj`/layer-0 output against the reference `node_26`/`ffn_inp-0` values already
  captured in this investigation (see the [archived implementation record](09-glm45-q5k-activation-implementation.md) for the exact reference numbers — no need
  to re-run `llama-eval-callback` unless you want a fresh capture; the reference values don't
  change). Compare raw values and report `max_abs`, `mean_abs`, RMS, relative L2, and cosine for F32
  and Q8_K against the reference, plus the change between activation modes. The ~1e-4 print-rounding
  floor is context only, not an acceptance criterion. If Q8_K does not reduce the layer-0 relative
  L2 error against the reference compared with F32, stop and report that finding rather than
  proceeding to the full PPL run.

### Step 6 — Full PPL re-measurement (only if Step 5 shows convergence)

- **Memory:** this checkpoint needs ~46 GB. Check free RAM first
  (`Get-CimInstance Win32_OperatingSystem | Select FreePhysicalMemory`) and run this alone — nothing
  else heavy in flight, per this project's standing rule on timing/memory under contention.
- Command: whatever this project's existing PPL command is for this checkpoint at `-c 2048`,
  second-half wikitext (see the [archived implementation record](09-glm45-q5k-activation-implementation.md) for the exact prior
  invocation, or `src/OpenTail.Stingray.Cli`'s `perplexity` command's own `--help`).
- Run both GLM activation modes under the same conditions and record each measured PPL, the
  reference PPL (8.6125), and the absolute/relative gap. Within ~0.3% of the reference is a target,
  not proof that the kernel hypothesis is correct or incorrect. The primary hypothesis test is
  whether Q8_K activation quantization reproduces the ggml reference `wo` computation; then assess
  whether the whole-model PPL gap shrinks correspondingly. Record the actual numbers and date them,
  regardless of outcome.

### Step 7 — Focused and then broader regression pass

`DotQ5K`/`MatVecQ5K` is a shared kernel, not GLM-specific — other checkpoints in this codebase
decode through Q5_K too. Even though the new path is gated off by default, **if this fix works and
you flip the gate on by default**, every model using Q5_K needs to be re-verified, not just GLM-4.5.

- Immediate scope: run the GLM-4.5 gate-on/gate-off arms and one small known Q5_K model gate-on/gate-off
  from the Step 5 matrix, comparing existing golden/parity expectations as well as the recorded
  GLM layer-0, logits, and PPL results. Any regression is a real finding — log it rather than
  silently reverting the change.
- Deferred until the new implementation is proven: inventory and re-run the wider set of real-weight
  Q5_K checkpoints (start from `grep -rln "Q5_K" tests --include=*.cs`, then narrow to tests that
  actually load real checkpoints) with the gate on, comparing against existing golden/parity
  expectations. Keep the gate off by default while this broader validation is outstanding.
- Existing kernel-level tests to re-run as-is (should still pass unchanged, since they test the old
  `DotQ5K` path specifically): `SimdKernelsQ8KSTests.cs`, `SimdKernelsQ3KQ8KTests.cs`.

### Step 8 — Close out

- Update [docs/103-quickest-first-plan.md item 19](../103-quickest-first-plan.md) and the
  [archived implementation record](09-glm45-q5k-activation-implementation.md) with the real
  outcome — whether the gap closed, by how much, and the measured PPL number, dated. If it didn't
  close the gap, that's still a real, useful result to record (rules out this specific hypothesis,
  narrows what's left).
- If it worked and the gate gets flipped on by default: update `docs/STATUS.md` if GLM-4.5's row
  status changes, and `docs/RUNNING.md` if its measured PPL/speed numbers change materially (per
  this project's own rule that those numbers must be dated and sourced, not asserted).
- This also affects the related, not-yet-investigated GLM-4.7-Flash entry in `bugstofix.md` — if
  this fix works for GLM-4.5, it's worth a
  quick check (not necessarily the full bisection again) on GLM-4.7-Flash too, since it may share
  the identical root cause.

## Constraints (this project's CLAUDE.md, apply throughout)

- `TreatWarningsAsErrors` — 0 warnings, always.
- No new Python reference scripts. `examples/llama.cpp` (vendored, real, working) is the oracle for
  the kernel math; do not write a Python re-implementation to "check" it.
- `dotnet test` — never `--nologo`. Prefer the fast test projects for iteration; only re-run the
  heavy real-weight suites when actually needed, and expect `--filter-class`/`--filter-method` to
  sometimes unreliably report "zero tests ran" for a real test — fall back to invoking the built
  `.exe` directly with `-class <FullyQualifiedName>` (must be fully namespace-qualified).
- Measure, don't assume, for every performance-adjacent claim (this is directly a correctness fix,
  not a perf one, but the same "write the real number down" discipline applies to the PPL result).
- Don't commit scratch/debug output to the repo root. The `Zz*Tmp.cs` scratch test files used here
  are an established, already-tracked convention in `tests/` — fine to extend, not something to
  clean up or "properly" refactor as part of this task.
