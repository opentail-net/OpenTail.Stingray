> **ARCHIVED 2026-08-08.** The SharpInference port audit is complete and its final full-suite
> run is recorded at the end of this file. **Carried forward:** the single known ForwardPass
> failure and the CI test-count floors, both now stated in
> [../00-current-work.md](../00-current-work.md). Note the stale line inside: the 1.0.3 release
> candidate described as unpushed has since been pushed; only the `stingray-v1.0.3` tag is
> still missing.

# Upstream port status — `examples/SharpInference08086` → OpenTail.Stingray

**Paused 2026-08-08 03:00.** Working state for whoever picks this up (including a later session of
mine, which will start cold).

## What the gap actually is

Measured directly (Stingray vs the 08086 snapshot, brand tokens normalised, whitespace-insensitive),
**not** taken on trust from the review document. Headline: **Stingray is ahead overall** — `src/`
133,887 lines / 227 files vs 109,483 / 152; `tests/` 80,955 / 290 vs 61,081 / 194. It is behind in
four narrow places only.

Three apparent gaps were false positives and must not be "ported":

| Looks behind | Reality |
|---|---|
| `Sampler.cs` | Identical logic at different indentation. Stingray 834 lines vs 686. |
| `Server.Host/Program.cs` | Stingray refactored env binding into `ServerEnvironmentOverrides.Apply` + `KnownEnvironmentVariables.FindUnknown()`. Strictly better. Per-file diffs cannot see code that moved between files. |
| `Shaders.Precompiled.g.cs` | Stingray has 183 precompiled entries vs 171. Ahead. |

`global.json` and `Directory.Packages.props` are **deleted upstream and present here**. Do not mirror
that deletion: without `global.json`, `dotnet test` falls back to VSTest, finds no adapter, and exits
0 having run nothing — a silent green across every suite.

## Theme C — prefix-cache correctness — **DONE, verified**

- `IForwardPass.MinRewindLength` (default 0) added; implemented on `ForwardPass`, `GpuForwardPass`,
  `HybridForwardPass`, `CudaForwardPass`, `CudaHybridForwardPass`.
- `TurboQuantKvCache.MaxTqLength` (max compressed length across layers; batched prefill advances
  per-layer counters independently so layer 0's `TqLength` understates it).
- `InferenceEngine`: prefix reuse gated on `candidate >= _fwd.MinRewindLength`; slot shadows keyed on
  a new `mutatedSlotIdx` (falls back to `OwnedSlotIdx` for MTP, which never binds a slot) instead of
  `activeSlotIdx`/hard-coded `[0]`; shadows downgraded to the retained prefix rather than nulled, and
  widened to the whole prompt once prefill completes.
- `tests/…ForwardPass/InferenceEnginePrefixCacheTests.cs` replaced wholesale with upstream's
  (1,443 lines; Stingray's was a strict subset, so nothing was lost). Compiled unmodified.
- **21/21 pass. Mutation-verified:** reverting the `MinRewindLength` gate fails
  `GenerateAsync_TqRewindFloor_ClampsPrefixReuse`; reverting `mutatedSlotIdx` fails
  `GenerateAsync_TwoSlot_MtpFailedMidDecode_DoesNotReuseDestroyedPrefix`.

## Theme B — chat-template hardening — **DONE, all 71 tests green**

### Done and verified

- `src/…Core/ChatTemplateException.cs`; `JinjaChatTemplate` `raise_exception`/`raise` throws it.
- All three endpoint families catch it → **400 instead of a bodyless 500**. OpenAI/Anthropic return a
  JSON `ErrorResponse`; Responses sets a plain 400, matching that endpoint's convention.
- `tests/…Server/TemplateRejectionTests.cs` ported. **3/3 pass, mutation-verified** — deleting the
  Anthropic catch fails exactly `Anthropic_NonAlternatingRoles_Returns400WithTemplateMessage`.

### Real engine bug found by porting that test

Stingray's Jinja had **no multiplicative precedence level at all** — `*`, `/`, `//` and `%` did not
parse. `1 % 2 == 0` evaluated to `1`. Because nothing checked that an expression was fully consumed,
the parser stopped at the `%` and silently discarded the rest, turning a role-alternation guard
`(message["role"] == "user") != (ns.index % 2 == 0)` into `… != ns.index`, which fires on a valid
conversation. That is why `AlternatingRoles_StillSucceed` failed on first port.

Fixed: added `ParseMul` between `ParseAdd` and `ParseUnary` (with `%}`-vs-`%` disambiguation),
evaluator entries for `*`/`%`/`//`/`/` with Python floor semantics, `FloorDiv`/`FloorMod` helpers,
and — the systemic part — `ParseExpr` now warns once when the parser stops early, so this whole class
of silent truncation becomes visible instead of producing a plausible-but-wrong AST.

Also added `JinjaChatTemplate.EosToken`, seeded unconditionally from `GgufTokenizer` (unlike BOS,
which is gated on `add_bos_token`). Mistral/Llama templates close assistant turns with
`{{ message["content"] + eos_token }}`; without it multi-turn history has no turn boundaries.

### Jinja engine port — **COMPLETE, all 71 tests green**

`JinjaChatTemplateTests.cs` (553 lines) and `MistralChatTemplateTests.cs` (159) ported from upstream;
Stingray's versions were strict subsets so nothing was lost. They went 17 red → 0 as each missing
engine feature landed:

| Added to `JinjaChatTemplate.cs` | Why it mattered |
|---|---|
| `ListExpr` — list displays in PRIMARY position only, so postfix `a[0]` stays a subscript | 5 tests |
| `FindTagEnd` — quote-aware scan for `}}`/`%}`, honouring backslash escapes. Comments (`#}`) deliberately NOT routed through it: Jinja2's comment state has no string rules | 5 tests. Mistral closes each `[AVAILABLE_TOOLS]` entry with `{{- "}}" }}`; a plain IndexOf ended the tag inside that literal and leaked the rest into the prompt, producing a tool block that is not valid JSON — silently, the only symptom being the model stops calling tools |
| `AttrFilter` + `TestValue` — `selectattr`/`rejectattr` | `messages \| selectattr('role','equalto','user') \| last` previously resolved to whatever message came last, of ANY role |
| Parameterised `is` tests (`x is equalto("user")`) — `IsTestExpr.Arg` | the argument list was left unconsumed and silently discarded along with whatever followed |
| Negative string indexing and string slicing (returns a string, not a list) | Mistral v3 uses `tool_call.function\|tojson` then `out[:-1]` to reopen the JSON and splice the call id; the slice yielded null and the tool-call body rendered empty |
| `ForNode.Filter` + `BindLoopVars` — `{% for k, v in … if cond %}`, filtered BEFORE the loop so `loop.index`/`length`/`last` describe the filtered set | the whole remainder had been parsed as a ternary (`else` is optional), evaluating the condition ONCE before the loop with loop vars undefined, so the filter never excluded anything |

Earlier in the port I reported these tests as passing when the build had actually failed and the run
used stale binaries. Any suite result quoted here is from a build verified at 0 errors first.

**Suite state after the port:** Core 470 (0 failed, 21 skipped), Server 261, Cli 367, Sessions 79,
TurboQuant 78, Vision 73, Pipeline 52 — all green.

## Theme A — Tekken / Mistral tokenizer — **DONE (split-pattern half verified; model half not run)**

Byte-level BPE added alongside the existing SentencePiece path, keyed on
`tokenizer.ggml.pre = "tekken"` (Mistral-Nemo / Ministral / Pixtral):

- `TokenizerSource.TokenizerPre` — new, carrying the GGUF `tokenizer.ggml.pre` value. The split is
  not derivable from the vocab, so it has to travel explicitly.
- `GgufTokenizer` is now `partial`, with a `[GeneratedRegex]` `TekkenPreTokenizer()` holding the
  split pattern **copied byte-for-byte** from upstream (it uses Unicode categories, and hand-retyping
  it through a script mangled `
`/`
` on the first attempt — splice it, never retype it).
- `_preTokenSplit` / `_byteBpeMerges` fields, threaded through the constructor as defaulted
  parameters so existing call sites are untouched; the merge rank table is now built for Tekken as
  well as SPM and routed to the right field.
- `EncodeByteLevelBpe` — splits the RAW text first, then byte-encodes each piece. That order is
  load-bearing: byte-encoding first would hand the split GPT-2 replacement characters instead of
  real letters and digits, so its Unicode categories would match the wrong things.
- The three GPT-2 helpers it needs (`EncodeToGpt2Bytes`, `EncodeByteToGpt2`, `SpmMergePieces`)
  already existed here and were reused unchanged.

**Evidence: `TekkenTokenizerTests` — 15 cases, 8 pass, 7 skip, 0 fail.** The 8 that pass are the
split-pattern theory cases, which need no model and are the actual specification of where Tekken
diverges from GPT-2. The 7 skips are model-backed and honest: no local GGUF declares
`tokenizer.ggml.pre = "tekken"`, and the finder scans for the *family* rather than naming a
checkpoint, so it will pick one up automatically if a Mistral-Nemo model ever lands in `models/`.

**Note:** upstream's version of those four model-backed tests used `if (t is null) return;`, which
counts as a PASS — the suite reported 15/15 green while four tests did nothing. Converted to
`Assert.SkipUnless` before recording any number. This is the same vacuous-gate pattern already
swept out of 644 sites; upstream still has it.

**Not verified here:** upstream's claim that this split pattern is a *correctness edge* over
llama.cpp (which reportedly carries an ASCII approximation because `std::regex` lacks Unicode
categories). The pattern is transcribed faithfully and its own tests pass, but no differential test
against llama.cpp was run. Do not repeat that claim as established.

## Theme D — CUDA — **DEFERRED, do not port blind**

`CudaTextKernels` +513/−478, `CudaForwardPass` +103/−41, `CudaMatMulBatchedTests` +78/−15,
`CudaKvarnPrefillTests` +39/−13, new `CudaVramHeadroomTests.cs`. Batched NORM RoPE (unblocks
llama-arch batched prefill), KVarN prefill for llama-arch, VRAM headroom warning.

This machine has no NVIDIA GPU, so none of it can be exercised beyond compiling. Porting it means
shipping unverifiable code — the same reasoning that left `cuda-fused-gate-up-plan.md` unimplemented.

## Tool-grammar tests — **DONE, and they found a hidden stale assertion**

Ported upstream's `ToolGrammarConstraintTests.cs` (7 methods; Stingray's 6 were a strict subset).
**7/7 pass against the real 7 GB `gemma-4-12b-it-qat-q4_0.gguf`.**

The interesting part was not the missing test. Stingray's copy hard-coded
`E:\models\gemma-4-12b-it-qat-q4_0.gguf`, a drive this machine does not have, so **every test in the
class had been skipping** — while the model sat in the repo's own `models/` directory. Replaced with
the repo-root-walking resolution `GgufModelTests.FindModelPath` already used.

With the fixture actually loading, `StringValue_FreeContentThenCloseQuote` **failed**: it asserted
that `,` is legal after the only declared key is satisfied. Stingray's implementation refuses it, and
is right to — `,` commits to another key, and with none declared the machine strands in `OExpectKey`
where `}` is no longer accepted. Upstream had already corrected the same assertion and documented the
reason against `GemmaToolArgumentConstraint.StepObject`; the corrected expectation came in with the
port.

So the assertion had been wrong for as long as the class existed and nothing caught it, because a
**path** — not an absent fixture — was suppressing the whole class. That is the failure mode the
`Assert.SkipUnless` conversion was meant to expose, and it only becomes visible if someone checks
*why* a suite is skipping rather than trusting the green.

Core skips accordingly dropped 21 → 15.

**Swept for the same pattern — bounded, not systemic.** Other classes do use absolute
`E:\models\…` / `C:\p\opentail-llm\models\…` paths, but of the 14 fixtures referenced that way only
three exist on this machine (OLMoE-1B-7B, Qwen3-8B-Q4_K_M, SmolLM2-1.7B), and the non-CUDA classes
referencing those already try a repo-relative candidate first (e.g. `ContinuousBatchingTests` walks
up looking for `models/<file>`). The rest name fixtures genuinely absent here, so their skips are
honest. `ToolGrammarConstraintTests` was the one class where a present fixture was being missed.

## Unrelated but open

- Release candidate `dbad0d7` is committed and **unpushed**; no `stingray-v1.0.3` tag exists.
  nuget.org has only 1.0.2. Packages are packed at `artifacts/nuget/` (1.0.3, all three + symbols).
- **`Tests.ForwardPass` completes locally and has a verdict** (first achieved 2026-08-08):
  **1,358 tests, ~18 min.** Steady state is **one real failure** —
  `ContinuousBatchingTests.PrefillWithCache_Chunked_MatchesFull`, deterministic across four runs
  with byte-identical values, pre-existing, and analysed in `docs/done/03-cpu-prefill-plan.md` item 5.
  - Two other failure sources were investigated and resolved as environmental, not defects: a
    once-seen flake under CPU contention that never recurred, and three `VulkanShaderTests` failures
    from unguarded `VulkanBackend` construction (now guarded — see the Vulkan section below).
  - The `--minimum-expected-tests 900` floor in `ci.yml` / `release.yml` /
    `verify-nuget-package.ps1` is **confirmed safe**: discovery finds 1,358, and discovery does not
    depend on fixtures, so a fixture-less CI runner discovers the same number. It could be raised
    toward ~1,200 once a CI run confirms the count there.
  - ~367 skips are dominated by the absence of an NVIDIA GPU, not missing models — see the
    corrected breakdown below. Do not read the other suites' green as covering this one.
  - **Lesson worth keeping:** never pipe a long test run through `tail`. An 18-minute run whose
    failure output is discarded costs another 18 minutes to redo.
- `docs/done/03-cpu-prefill-plan.md` item 4 **CLOSED 2026-08-08**: interleaved arms, warm-up and three
  samples per arm give **3.48x median** (F32 6.90 -> Q8 24.02 tok/s) with 1.0%/3.6% spread. The
  quality half closed the same day. Both halves of that plan are now evidenced.


## Fixture-resolution audit (2026-08-08) — skips that were hiding real coverage

Three separate cases now, all the same shape: **a test skipped for an environmental reason, not
because the fixture was missing.** A skip is only honest if the thing under test genuinely cannot
run here. Worth checking the *reason* for a skip, not just its count.

1. **`ToolGrammarConstraintTests`** — pinned `E:\models\gemma-4-12b-it-qat-q4_0.gguf`, a drive this
   machine lacks, while that exact model sat in the repo's `models/`. Every test in the class had
   never run. Fixed by repo-root walking; one test then failed on a stale assertion (see above).
2. **`TekkenTokenizerTests`** — upstream's four model-backed cases used `if (t is null) return;`,
   which counts as a PASS. Converted to `Assert.SkipUnless`.
3. **`Gemma4TokenizerTests`** — pinned `E:\models\gemma-4-E4B-it-Q8_0.gguf`: absolute path AND an
   exact quantisation, while everything it asserts (vocab size, BOS/UNK, special tokens,
   round-trip) is a property of the tokenizer, identical across quants. Now resolves any local
   `*E4B*.gguf` by family, excluding mmproj files.

   **What running it found:** Gemma-4 exports disagree about which token is the configured EOS.
   The local `gemma-4-E4B_q4_0-it.gguf` declares `tokenizer.ggml.eos_token_id = 1` (`<eos>`); the
   Q8_0 export the test was written against declares 106 (`<turn|>`). Both are legitimate. The test
   had hard-coded 106 and separately asserted that `EosTokenId` appears when encoding `"<turn|>"` —
   two assertions about one checkpoint dressed as assertions about Gemma 4.

   Rewritten to the export-independent contract: `EosTokenId` must be one of the two known end
   tokens, and — the part generation actually depends on — **`EogTokenIds` must contain BOTH**.
   Miss that and a model configuring `<eos>` runs straight through `<turn|>`, decoding the turn
   terminator as literal text. **That assertion passes**, so `BuildEogTokenIds` is correct on an
   export it had never been tested against.

**Left pinned deliberately:** `Gemma4E4BActualMetadataTests` and `Gemma4ToolTemplateTests` still
name the Q8_0 checkpoint. Unlike the tokenizer, per-export metadata is exactly what the first is
about, so family-resolution there would be wrong, not helpful.


## ~~ForwardPass skip audit~~ — **SUPERSEDED, DO NOT USE THE TABLE BELOW**

> **The reason-breakdown in this section is wrong.** It attributes 354 skips to missing model
> fixtures and 11 to hardware. The real split is **327 hardware / 35 fixture** — see
> "Corrected ForwardPass skip breakdown" further down. The cause was my own mislabelling of 323
> device gates. The *narrative* here (which classes want which models, and the Gemma fixture pin
> that was fixed) is still accurate; only the reason counts are not.

### Original section, retained for the reasoning

Analysed the 370 skips from the captured full run rather than re-running. Breakdown:

| Reason | Count |
|---|---|
| model fixture not present | 354 |
| no CUDA device | 11 |
| OpenBLAS not present | 2 |
| model fixture or CUDA device | 1 |
| glslc not found (Vulkan SDK compile-fallback test) | 1 |
| Flash-64 128/256 widths held back at the gate | 1 |

Only 11 skips are CUDA-hardware. The fixture-resolution problem found in Core is **not** systemic
here: cross-referencing the non-CUDA classes against models actually on disk, the large groups want
models this machine genuinely lacks — `PipelineStepTests` (11) needs Llama-4-Scout,
`HybridGdnForwardPassTests` (9) needs a GDN/qwen35moe model, the `VulkanMtp*` classes need an MTP
export. Those skips are honest and should stay.

**One real pin found and fixed:** `Gemma4CpuForwardPassTests` had two tests pinned to
`gemma-4-E4B-it-Q8_0.gguf` while asserting only architecture-level facts — the SWA layer map,
per-layer head dims, KV source layers, FFN activation, final logit softcap — none of which vary with
quantisation. The class already ran a third test against the local q4_0 export, so the pin was
history rather than intent. Repointed to a family finder (any local `*E4B*.gguf`, excluding mmproj).
**Class now 4/4 with 0 skips (was 2 skipped)**, so the gemma4 CPU forward and PLE paths execute here
for the first time.

**Left alone deliberately:** `Gemma4GpuForwardPassTests` wants `gemma4-v2-Q4_K_M.gguf`, a variant not
on this machine — a genuine absence, not a resolution bug, even though the Vulkan backend itself
works here.

**Negative results are worth recording too:** the earlier three fixture findings could have led to a
standing suspicion that every "fixture not present" skip is suspect. It is not. 354 of them are real.


## CORRECTION (2026-08-08): the ForwardPass skip breakdown above was wrong — my own doing

The table in the previous section reports 354 skips as "model fixture not present" and only 11 as
hardware. **That attribution is not trustworthy, and the cause was the mass `Assert.SkipUnless`
conversion earlier in this session.** That script rewrote every `if (X is null) return;` gate with a
single hard-coded message regardless of what `X` was, so a Vulkan or CUDA device-init failure came
out labelled *"model fixture not present in this environment"*.

Counted across the test tree: **323 device gates** carried the model-fixture message —
306 on `gpu`, 9 on `cuda`, 5 on `backend`, 3 on `vk` — against 253 genuine `path` gates. So a large
share of that 354 were GPU-availability skips wearing the wrong label.

**How it surfaced:** `GpuForwardPassKvDtypeTests.Q8Kv_GreedyDecode_Coherent` skipped in one full run
and passed in the next, with `Qwen3-8B-Q4_K_M.gguf` present throughout. Two hypotheses were tested
and discarded first — a resolver that swallows a transient open failure (it is pure `File.Exists`),
and CWD mutation by another test (`SetCurrentDirectory` appears nowhere). The real cause is
`Assert.SkipUnless(gpu is not null, …)`: the integrated Radeon failed to initialise transiently
under load during that run, and the skip claimed a missing model.

**Fixed:** all 323 relabelled — `gpu`/`vk` → "no usable GPU backend", `cuda` → "no CUDA device",
`backend` → "no usable compute backend". Gates on `path`, `tokenizer`, `mmproj` keep the
model-fixture wording, which is correct for them.

**Lesson:** a bulk mechanical fix that improves one property (skips now report as skips) can quietly
destroy another (the skip *reason*). The conversion was still right to do — 644 vacuous passes were
worse — but its output needed reading, not just counting. The corrected breakdown requires a fresh
full run to state; until then, treat the hardware-vs-fixture split as unknown rather than as the
numbers in the table above.


## Corrected ForwardPass skip breakdown (2026-08-08) — hardware, not fixtures

Fresh full run after relabelling the 323 mislabelled device gates. **1,358 tests, 1,101 s.**

| Reason | Count | Verdict |
|---|---|---|
| CUDA device unavailable (312 generic-GPU + 15 explicit; **all 312 verified to be in CUDA-named classes, zero Vulkan**) | **327** | legitimate — no NVIDIA GPU on this machine |
| model fixture not present | **35** | legitimate — models genuinely absent |
| OpenBLAS not present | 2 | legitimate |
| glslc not found (Vulkan SDK) | 1 | legitimate |
| Flash-64 128/256 widths held at the gate | 1 | deliberate |

**This inverts the earlier table.** It reported "354 model fixture / 11 hardware"; the truth is
**327 hardware / 35 fixture**. Local ForwardPass coverage is limited almost entirely by the absence
of an NVIDIA GPU — not by missing model files. Chasing more model fixtures would buy very little;
only CUDA hardware moves this number.

## Vulkan is intermittently unavailable under parallel load — and 54 tests turned that into failures

That run also showed **4 failures, up from 1**. Three were `VulkanShaderTests` throwing
`VkException [-9] ErrorIncompatibleDriver` from the `VulkanBackend` constructor — not a shader
defect. Root cause: that class (and two others) constructed `new Vulkan.VulkanBackend()` **directly,
unguarded**, so an environmental failure became a test FAILURE rather than a skip. The 300+ gates
elsewhere go through `TryCreate()` and skip cleanly, which is why only these three surfaced.

Evidence it is contention, not a permanent driver problem: **run `VulkanShaderTests` alone and it is
89/89 with 0 skips.** The same class loses 3 tests inside a full parallel run. It also explains the
earlier mystery of `GpuForwardPassKvDtypeTests.Q8Kv_GreedyDecode_Coherent` skipping one run and
passing the next with its model present throughout.

**Fixed:** 54 direct constructions across `VulkanShaderTests`, `VulkanPrecompiledShaderTests` and
`GpuFfnScratchGuardTests` now route through a `CreateBackendOrSkip()` helper that absorbs
**constructor** failures only — once a device exists, every shader assertion runs and fails
normally, so the guard cannot mask a correctness defect. `VulkanInitTests` is deliberately left
unguarded: there, bringing the device up *is* the thing under test.

**Standing caveat:** the 4th failure remains the known deterministic
`PrefillWithCache_Chunked_MatchesFull`. That one is a real defect, not an environmental flake.


## VERIFIED FINAL STATE (2026-08-08) — full suite, accurate labels

Run after the device-gate relabelling and the Vulkan constructor guard. This is the first
ForwardPass run whose skip reasons can be read at face value.

**1,358 tests · 1,072 s · 1 failed · 369 skipped.**

| Skip reason | Count |
|---|---|
| no CUDA device | **327** |
| model fixture not present | **35** |
| no usable GPU backend (transient Vulkan) | 2 |
| OpenBLAS not present | 2 |
| model fixture or CUDA device | 1 |
| glslc not found (Vulkan SDK) | 1 |
| Flash-64 128/256 widths held at the gate | 1 |

Three things this confirms:

1. **The Vulkan guard worked.** The three `ErrorIncompatibleDriver` failures are gone, and only 2
   Vulkan skips occurred — so Vulkan was available for essentially the whole run, and an environmental
   flake now reports as a skip instead of a failure.
2. **The labels are finally honest.** 327 CUDA vs 35 fixture, stated plainly. The original table
   claimed the reverse.
3. **Steady state is exactly one real failure:** `PrefillWithCache_Chunked_MatchesFull`, deterministic
   across five runs with byte-identical values (1.86363602 vs 1.5929451). Left red deliberately — it
   is the only automated detector for chunk-dependent Q8 activation scales.

**Whole-tree state at this point:** Core 488, Server 261, Cli 367, Sessions 79, TurboQuant 78,
Vision 73, Pipeline 52 — all green, 0 warnings under `TreatWarningsAsErrors` — plus ForwardPass
1,358 with the single known failure. Nothing committed, pushed, or tagged.
