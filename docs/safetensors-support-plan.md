# SafeTensors support plan

> ## HANDOFF — READ THIS SECTION FIRST
>
> This document is both the plan and the working log. The plan proper starts at
> "Decision and definition of 'full support'". **This section is the current state and the ordered
> next steps.** Status markers are also inlined in each phase below; where they disagree with this
> section, this section is newer.
>
> Last updated 2026-08-05.

## Where this stands

| Area | State |
|---|---|
| Phase 0 — capability contract | **DONE.** `ModelPackageCapability`, `ModelPackageInspector`, `inspect -m <dir>`, `capabilities` command, README table. 20 tests. |
| Phase 1 — package/tensor/tokenizer interfaces | **DONE.** Strict config reader (19 tests), `TokenizerSource` + `GgufTokenizer.FromSource` + `HuggingFaceTokenizerSource` (14 tests). |
| Phase 2 — CPU dense execution | **DONE.** Seam proven; differential parity tests (F32/F16/BF16 & tied embeddings) passing and mutation-verified. Real SmolLM2 package supported. |
| Phase 3 — product integration | **DONE 2026-08-05.** The CLI and server load and generate end-to-end from a real SafeTensors directory; R6-R9 release gates are closed. |
| Q8_0 on-the-fly quantization | **REMOVED 2026-08-05.** It was unreachable and unverified; SafeTensors stays high precision and GGUF is the quantized deployment route. |
| SentencePiece | **REFUSED, deliberately.** Detected and named; loading is not implemented. |
| Hardening / fuzzing | **DONE 2026-08-05** (R9). Mutation-verified parser, shard-index, config and tokenizer boundaries. |
| Phases 4-6 | **NOT STARTED.** |

**The R1-R10 SafeTensors acceptance items below are closed.** R11 remains a separate, pre-existing
CLI context-sizing correctness investigation; it is not a SafeTensors release blocker.

**SafeTensors CPU execution is proven end-to-end.** Inspection, configuration, tokenization, tied
embeddings and BF16 widening work against real models; differential GGUF parity tests verify
bit-identical F32 logits; and the CLI runs `models/SmolLM2-135M-Instruct` from the directory path.

Verified 2026-08-05 by direct execution, not inference:

```
$ opentail-llm-cli -m models/SmolLM2-135M-Instruct -p "The capital of France is" -n 16 --temp 0
Model loaded in 0.5s — 30L, 576d, headDim=64, 49152 vocab, ctx=8192
$ opentail-llm-cli -m models/SmolLM2-135M-Instruct -p "hi" -g 99
Error: GPU offload (--ngl / -g) is not yet supported for SafeTensors packages. …
```

### What exists, by file

| File | Purpose |
|---|---|
| `Core/ModelPackageCapability.cs` | Versioned profile schema + `DenseLlamaCpu` row + rejection types. |
| `Core/ModelPackageInspector.cs` | Verdict from files alone. Never loads tensors, never throws for an unsupported package. |
| `Core/SafetensorsConfigReader.cs` | `config.json` → `ModelHyperparams`, refusing anything unread. |
| `Core/TokenizerSource.cs` | Format-independent tokenizer metadata. |
| `Core/HuggingFaceTokenizerSource.cs` | `tokenizer.json` + sidecars → `TokenizerSource`. |
| `Core/IModelTensorSource.cs` | The engine seam: 5 members. `GgufModel` implements it unchanged. |
| `Core/SafetensorsTensorSource.cs` | **Proven.** Package → seam, over memory-mapped shards + native BF16->F32 buffers. |
| `Core/SafetensorsLoader.cs` | `TryGetMappedPointer` added; `FileStream` reads untouched. |
| `Cli/ModelPackageReporting.cs` | Renders reports for `inspect` / `capabilities`. |
| `tests/…ForwardPass/SafetensorsDifferentialFixtureTests.cs` | Differential GGUF vs SafeTensors parity fixture (F32, F16, BF16, tied embeddings). |
| `tests/…Core/RealPackageTouchpointTests.cs` | Runs against the downloaded model; skips when absent. |

### Test state — measured 2026-08-05, not carried forward from an earlier session

| Project | Result |
|---|---|
| Release build of `OpenTail.Stingray.slnx` | **0 errors, 0 warnings.** |
| `tests/OpenTail.Stingray.Tests.Cli` | **251/251.** |
| `tests/OpenTail.Stingray.Tests.Core` | **356/356** (last measured 2026-08-04). |
| `tests/OpenTail.Stingray.Tests.Server` | **208/209.** |
| `tests/OpenTail.Stingray.Tests.ForwardPass` | **1316/1318** (last measured 2026-08-04). |

The failures are all known and none belong to SafeTensors:

- `ConcurrencyLimitTests.BoundedQueue_RejectsOnlyAfterActiveAndWaitingCapacityIsConsumed` — **flaky
  under machine load.** Failed at 1m40s inside the full suite, passed in 770ms when run alone. Do not
  chase it; re-run it in isolation before believing a failure.
- `Gemma4VulkanPleE2ETests.Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent` and
  `Gemma4VulkanNarrowedKvE2ETests.Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax` — long-standing,
  pre-date this workstream.

**A test count measured while the tree is being edited is worthless.** A ForwardPass run during
concurrent edits reported 1045 tests and looked entirely credible; the real figure was 1318. If a
count moves by hundreds, suspect the runner before suspecting the code.

### Reference model

`models/SmolLM2-135M-Instruct` — `HuggingFaceTB/SmolLM2-135M-Instruct`, llama, 30 layers, hidden 576,
9 heads / 3 KV heads, vocab 49152, BF16, 257 MB. Gitignored. Re-fetch with:

```bash
D=models/SmolLM2-135M-Instruct; mkdir -p $D
B=https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main
for f in config.json generation_config.json tokenizer.json tokenizer_config.json \
         special_tokens_map.json merges.txt model.safetensors; do curl -sL -o $D/$f $B/$f; done
```

## What needs resolving

**This is the authoritative backlog.** Raised by review on 2026-08-05, ordered by how badly each one
bites. Every item states what "done" means; do not mark one resolved without meeting that bar. Items
R1-R4 are defects in code that is already committed — they are not new features, and they come first.

| # | Item | Kind | Blocks |
|---|---|---|---|
| R1 | Q8_0 tensor selection matched the wrong names | **CLOSED 2026-08-05.** Superseded by removing the unreachable Q8_0 path. | Q8_0 being trustworthy at all |
| R2 | Architecture feature probes always false for packages | **DONE 2026-08-05.** Probes use `IModelTensorSource`; package entry points pass their source. | Phase 5 (new architectures) |
| R3 | BF16 widening loop duplicated | **DONE 2026-08-05.** One locked widening/cache implementation. | nothing, but cheap to fix |
| R4 | Dimension-order `<remarks>` deleted | **DONE 2026-08-05.** Restored, including Q8_0 layout policy. | nothing, but it is the costliest trap |
| R5 | Q8_0 is unreachable from product code | **DONE 2026-08-05.** Option (b): removed until there is a product design and reference evidence. | advertising Q8_0 |
| R6 | `InferenceEngineLoader` never exercised | **DONE 2026-08-05.** Optional real-package server request smoke test. | Phase 3 sign-off |
| R7 | `--ctx-size` unconfirmed on the package path | **DONE 2026-08-05.** Package server path passes the effective context to `ForwardPass`; integration assertion covers it. | Phase 3 sign-off |
| R8 | Model format not persisted in session/plan state | **DONE 2026-08-05.** Typed discriminator persists in plans and session state/manifests. | Phase 3 sign-off |
| R9 | Hostile-package fixtures and fuzzing | **DONE 2026-08-05.** Bounded mutation corpus and full fixture matrix. | Phase 3 item 4, release |
| R11 | `--ctx-size` not passed to `ForwardPass` on the **CLI** CPU path, both formats | Minor, pre-existing | nothing here |
| R12 | Compatibility-key formula changed; v1 sessions could not match it | **DONE 2026-08-05.** GGUF hashes as before; only non-GGUF appends the format. | session restore |
| R10 | Q8_0 numerical accuracy never compared to GGUF | **CLOSED 2026-08-05.** No Q8_0 SafeTensors path remains to advertise. | advertising Q8_0 |

**R11. `--ctx-size` reaches the forward pass on the server but not the CLI.** R7 fixed
`InferenceEngineLoader`. `RunCommand` builds `ForwardPass` without `maxContextLength` on the package
path (`:671`) *and* the GGUF CPU path (`:812`), so both formats behave identically — not a SafeTensors
regression.

**This is smaller than it first sounds, and the fix is not free.** `ForwardPass` documents that
`ctxLen` "only governs scratch buffer sizes; PagedKvCache allocates pages lazily", so `--ctx-size` on
CPU is not silently mis-sizing the KV cache — it leaves scratch at `min(hp.ContextLength, 32768)`
instead of the requested bound. Wiring `ctxSize` in would shrink scratch to the requested context, and
scratch smaller than an oversized prompt is a heap overflow in `unsafe` code, so it requires first
proving the CLI clamps prompt length to `ctxSize`. **The server already carries that exposure via R7 —
check it there too.** Deliberately not done as a "quick" item; it is a correctness question wearing a
one-line-change costume.

**R12 — RESOLVED 2026-08-05. The compatibility-key formula no longer changes GGUF hashes.**

`ComputeCompatibilityKey` appended `abi.ModelFormat` unconditionally, so the same GGUF model hashed
differently than it did before the discriminator existed. The codec had carefully retained the ability
to *read* version-1 envelopes — but a v1 session's stored key could never equal a freshly computed one,
so any consumer that recomputed-and-compared would reject exactly the sessions back-compat was written
to preserve. The two halves of the change expressed opposite intentions.

Resolved in favour of v1 surviving: the format is folded into the hash **only for non-GGUF containers**.
GGUF hashes byte-identically to every historical build; SafeTensors still fences apart from GGUF over
the same weights. Pinned by `ComputeCompatibilityKey_Gguf_MatchesPreDiscriminatorHash`, which asserts
an **independently computed** SHA-256 of the literal `"fp:128:512"` rather than one obtained by calling
the method — a self-referential expectation would accept any future formula change silently. Verified
by mutation: inverting the condition fails that test and nothing else.

Nothing in `OpenTail.Stingray.Sessions` compares a stored key against a recomputed one yet. When restore
validation is wired, that comparison is what makes this test load-bearing.

Deliberately **not** on this list, because they are settled refusals rather than open questions:
SentencePiece loading, non-BPE tokenizer models, `rope_interleaved: true`, GPU/CUDA/Vulkan execution
from a package, and every architecture outside `llama`/`mistral`. Each is refused with a named reason.
Reopening one is a decision, not a bug fix — record it here first.

---

**R1 — CLOSED 2026-08-05. `SafetensorsTensorSource.IsQuantizableProjection` selected the wrong tensors.**

```csharp
return canonicalName.Contains("proj") || canonicalName.EndsWith(".weight", StringComparison.Ordinal)
    && !canonicalName.Contains("norm") && !canonicalName.Contains("embed");
```

Two defects. First, `||` binds looser than `&&`, so this reads `A || (B && C && D)` — almost certainly
not what was written, and the reader cannot tell whether the author knew. Second, the names reaching
this method are **canonical GGUF names**, not Hugging Face names: `SafetensorsTextModelPackage` maps
`self_attn.q_proj.weight` → `attn_q.weight` and `model.embed_tokens.weight` → `token_embd.weight`.
So `Contains("proj")` never matches anything, and `Contains("embed")` never excludes the embedding
table, because the canonical spelling is `embd`.

**Resolution:** this was briefly narrowed to an explicit canonical projection policy, then the whole
unreachable Q8_0 path was removed under R5. There is no SafeTensors quantization selector now.

**R2 — DONE 2026-08-05. `ModelHyperparams` feature probes were silently disabled for every package.**

The CLI calls the one-argument `FromGgufMetadata(stTensorSource.Metadata)` because the second
parameter is typed `GgufModel?` and a package has none. That parameter is what probes for
`blk.0.attn_q.bias`, `blk.0.attn_q_norm.weight` and `blk.0.ffn_gate_shexp.weight`, and
`ToOpenTailMetadata()` emits no `_opentailllm.has_*` keys to stand in for it. So `hasAttnBias`,
`hasQkNorm`, `perChannelQkNorm` and `hasSharedExpert` are **always false** for a SafeTensors package.

This is latent, not live: `DenseLlamaCpu` accepts only `llama` and `mistral`, and neither uses
attention bias or QK-norm. It becomes a silent-wrongness bug the moment a Qwen-family architecture is
added to the profile — the model would load, run, and emit nonsense.

**Resolution:** the probe parameter is widened from `GgufModel?` to `IModelTensorSource?`; both the
CLI and server pass the SafeTensors source. A synthetic `IModelTensorSource` carrying
`blk.0.attn_q.bias` asserts `HasAttnBias == true`. The fixture deliberately tests the shared contract
rather than widening the currently supported `llama`/`mistral` package profile. **Do this before
widening the architecture list, not after** — afterwards, the failure is silent.

**R3 — DONE 2026-08-05. `GetRawFloat32Ptr` duplicated the BF16 widening loop** that `GetTensorDataPtr` had,
including its own `_convertedBf16Buffers` bookkeeping. Two copies of a conversion that must agree
bit-for-bit is how they stop agreeing.

**Resolution:** `GetTensorDataPtr` delegates BF16 conversion to `GetRawFloat32Ptr`; that method is
the sole widening/cache implementation and locks its cache access. Differential BF16 tests remain the
regression oracle.

**R4 — DONE 2026-08-05. The `SafetensorsTensorSource` class `<remarks>` block was deleted** in the Q8_0 change. It carried
the dimension-order warning — the single most expensive trap in this workstream, and the one the plan
tells every reader to assert rather than eyeball.

**Resolution:** restored. It now explains the SafeTensors `[output,input]` to GGUF
`[input,output]` descriptor conversion, confirms that raw weight bytes are not transposed, and
states the Q8_0 projection-only policy.

**R5 — DONE 2026-08-05. Q8_0 was unreachable from product code.** `SafetensorsTensorSource.Open` took a
`SafetensorsQuantizationMode`, but `RunCommand.cs:648` and `InferenceEngineLoader.cs:76` both call the
one-argument overload, so `None` is the only mode any user can ever get. Q8_0 exists solely for
`SafetensorsOnTheFlyQuantizationTests`. Either wire a flag or delete the feature — a quantization path
that only tests can reach accrues maintenance cost and earns nothing, and it will quietly rot.

**Resolution:** option (b). The enum, descriptor rewriting, native Q8_0 buffers, quantizer, and its
test-only fixture were removed. A future quantized SafeTensors route needs an explicit product option,
capability row, independent GGUF/reference oracle, and a new plan entry.

**R6 — DONE 2026-08-05. `InferenceEngineLoader` had no real-package smoke test.** The CLI package
path was proven by execution; the server path was not.

**Resolution:** `SafetensorsProductIntegrationTests` now finds the optional
`models/SmolLM2-135M-Instruct` package, loads it through `InferenceEngineLoader`, completes one
greedy-token request, and asserts its usage and terminal stop chunks. It skips cleanly when the
downloaded model is absent.

**R7 — DONE 2026-08-05. `--ctx-size` was unconfirmed on the package path.** The branch computed
`stCtxSize` but originally constructed `ForwardPass` without it, leaving the configured value unused.

**Resolution:** the package server path now supplies `maxContextLength: stCtxSize` to
`ForwardPass`. The product integration fixture configures `ContextSize=128` against a 512-token
package and asserts the constructed pass uses a 128-token scratch context.

**R8 — DONE 2026-08-05. Model format was not persisted in session or plan state.** A resumed session
could not know it was a package. `ModelFormat` is now a typed, string-serialized execution-plan field
and part of the session ABI. Session-state envelopes and disk manifests write it explicitly; restoration
rejects a format mismatch. Their version-1 readers retain the historical GGUF interpretation for
pre-SafeTensors state. Regression tests cover SafeTensors persistence/mismatch rejection, manifest
round-trip, plan JSON, and legacy state decoding.

**R9 — DONE 2026-08-05. Hostile-package fixtures and fuzzing.** The first hardening slice
adds direct mutation fixtures for an oversized/truncated header length, invalid and mismatched tensor
offsets, duplicate tensor names, overlapping ranges, shard-index escape, and index/inventory mismatch.
The loader now bounds headers before allocation, validates descriptors/ranges/known-dtype byte sizes,
rejects duplicate or overlapping tensors, enforces package-root containment in its own index path, and
requires an index to agree with shard contents. Inspection also reports a non-object `config.json`
instead of throwing.

**Resolution:** a full dense-Llama package split across two indexed shards proves the product package
path. Mixed F16/BF16 weights are explicitly accepted and reported, while unsupported dtypes remain
refused. The deterministic corpus covers malformed/non-object config and tokenizer JSON plus malformed
index roots, maps and values. Together with existing supported, missing-tokenizer, tied-output and
unsupported-architecture fixtures, this meets Step 6's matrix. Treat all package metadata as untrusted
input.

**R10 — CLOSED 2026-08-05. Q8_0 numerical accuracy was never compared against GGUF.** The tests asserted the mode was
recorded and the buffers are shaped correctly. Nothing asserts the quantized weights *decode* to the
values GGUF's Q8_0 produces, which is the only property that matters — and this codebase's own
differential fixture is the obvious instrument.

**Resolution:** R5 removed the non-product Q8_0 path rather than expose an unverified quantizer. This
becomes a required acceptance gate if, and only if, a new SafeTensors quantization proposal is accepted.

## Rules that are not negotiable

1. **Do not commit.** The user commits. This has held for the entire workstream.
2. **No performance work.** No benchmarks, timings or throughput claims — the development machine runs
   other workloads and cannot produce a trustworthy number. Correctness only. The one performance
   obligation is negative and already satisfied by construction: the seam is a type swap, so GGUF runs
   the same code it did before.
3. **Clean Release build, 0 warnings.** `TreatWarningsAsErrors` is on. New `STINGRAY_*` variables
   must be added to `KnownEnvironmentVariables` or the drift test fails.
4. **Mutation-verify every new test.** Break the code deliberately, confirm the test fails, revert.
   An untested test is worse than none. `if (false)` mutations trip CS0162 — use a condition that
   compiles, e.g. flipping a key name or a bound.
5. **Never run two builds or test runs concurrently.**
6. **Refuse rather than run.** A model that executes and is subtly wrong is the worst outcome
   available. Every refusal so far exists because the alternative was plausible-looking garbage.
7. **Record decisions here**, including negatives and refusals, so a cold session can resume.

## Do these in order

> **Steps 1-4 are DONE.** They are kept below because they record *why* each decision went the way it
> did, and that reasoning is still load-bearing. Start at "What needs resolving" above, then Step 5.
>
> - **Step 1 DONE** — `SafetensorsDifferentialFixtureTests`, mutation-verified.
> - **Step 2 DONE** — tied embeddings implemented; the refusal is removed and the README row updated.
> - **Step 3 DONE — option (b) chosen**, with a correction: the widening is **lazy and per tensor**, on
>   first `GetTensorDataPtr`, not eager at load. `DType` gained no BF16 member, so no `switch` audit was
>   needed. `ModelPackageInspector` reports the inflated `EstimatedWorkingSetBytes` so the residency
>   cost is visible before load, which is what "must be reported rather than done silently" required.
> - **Step 4 DONE** — the published row matches what executes.

### Step 1 — Prove `SafetensorsTensorSource` with a differential fixture — **DONE**

This is the gate for everything after it. `SafetensorsTensorSource` is written and compiles but has
**never been verified**; do not build on it until this passes.

1. Add a fixture builder that emits the *same* tiny dense Llama twice — once as a GGUF F32 file, once
   as a SafeTensors package — from one set of deterministic pseudo-random weights. Keep it tiny
   (2 layers, hidden 64, 4 heads, 2 KV heads, vocab 128).
2. Assert `SafetensorsTensorSource` and `GgufModel` return **identical bytes** for every canonical
   tensor name, and identical `GgufTensorInfo` dimensions.
3. Then assert identical logits for a fixed prompt through `ForwardPass` on both sources.
4. Add an F16 SafeTensors variant against the F32 GGUF, asserting agreement within the tolerance the
   source dtype implies — not exact equality.

**The specific trap this catches.** `SafetensorsTensorSource.ToGgufDimensionOrder` reverses the shape:
Hugging Face stores row-major `[out_features, in_features]`, GGUF lists dimensions fastest-varying
first. The bytes are untouched; only the descriptor order differs. If that reversal is wrong the model
loads, runs, and emits nonsense — no exception, no obvious symptom. **Assert it; do not eyeball it.**

Acceptance: byte-identical tensors, identical F32 logits, F16 within tolerance, all mutation-verified.

### Step 2 — Tied output embeddings

Currently refused by `ModelPackageCapability.DenseLlamaCpu` and by `SafetensorsConfigReader`. The real
touchpoint proved this is **the norm, not an edge case** — SmolLM2-135M ties, and so do most small
Llama releases. Without it there is no real model to prove execution against.

1. Implement: when `tie_word_embeddings` is true, the output projection reuses the embedding matrix
   rather than expecting an `lm_head.weight`/`output.weight` tensor.
2. Verify with the Step 1 differential fixture built in a tied variant — the point is that tied and
   untied produce the same logits when the weights are the same.
3. Only then remove the refusal from both the profile's `Exclusions` and the config reader, and update
   the README row.

Do not relax the refusal before the fixture exists. Silently reusing the input embedding as the output
projection is exactly the "runs and is subtly wrong" failure this plan is built to avoid.

### Step 3 — Decide BF16, explicitly

`DType` has **no BF16 member** (GGUF numbers it 30). BF16 weights cannot be described to the engine at
all, so `SafetensorsTensorSource` refuses them. The reference model is BF16, as are most modern
releases. Choose one and record the choice here:

- **(a) Add `BF16 = 30` to `DType` and dispatch on it.** Correct and unblocks real models. Audit every
  `switch` over `DType` — any that silently falls through to a default rather than throwing is a place
  BF16 would be misread. This is the substantial option.
- **(b) Convert BF16 → F32 on load** into owned memory. Simple, but abandons the memory-mapping win and
  quadruples residency; a 7B model becomes ~28 GB. Acceptable only for small models, and it must be
  reported to the user rather than done silently — the plan forbids silent conversion.
- **(c) Keep refusing.** Honest, but the profile then cannot run the reference model or most others.

**Never** describe BF16 bytes as `Float16`. They share a width but not an exponent layout; that
substitution loads, runs, and produces nonsense.

### Step 4 — Narrow the published capability row to match reality

The row advertises `F32/F16/BF16`. Execution will support less than that until Step 3 lands. Nothing is
currently false because nothing executes — but the moment inference works, update
`ModelPackageCapability.DenseLlamaCpu.SourceDtypes`, the README table and
`ModelPackageInspector.RenderCapabilityTable` in the same change. **No broad format claim without a
capability row that is true.**

### Step 5 — Wire packages into model load (Phase 3) — **CLI DONE, remainder open**

Done in `RunCommand`: a directory or bare `.safetensors` path routes to a package branch that runs the
capability check first, then refuses `-g`/`--tq`/`--draft-*`/`--dspark-model`/`--image` with a named
reason, then opens the source and tokenizer and wires `forward`/`prefill`/`resetCache` exactly as the
GGUF CPU path does. Detection is by path shape, so no `--model-format` flag was needed; add one only if
a real ambiguity appears.

**How that branch is built, and why — do not "simplify" this.** It jumps to the existing
`backendConfigured:` label, a pattern the GGUF path already used twice. C# forbids a `goto` that jumps
past a `using` declaration (CS8648), so `model` and `cpuBackend` are plain nullable locals disposed in
the existing `finally`, not `using var`. Reverting either to `using var` breaks the build. Every local
the shared decode loop reads after the label must be definitely assigned on *both* paths — `nGpuLayers`
is set to 0 in the package branch for exactly this reason, and the compiler will tell you (CS0165) if a
future local is missed.

R6 (server smoke test), R7 (`--ctx-size`) and R8 (format persisted in session/plan state) closed on
2026-08-05. Phase 3 is signed off for the published `dense-llama-cpu` profile.

### Step 6 — Fixtures and hardening (Phase 3 item 4) — **DONE 2026-08-05** (R9)

Supported, corrupt, sharded, missing-tokenizer, mixed-dtype, tied-output and unsupported-architecture
packages. Fuzz SafeTensors headers, shard indexes, offsets, duplicate names and integer-overflow sizes.
Treat all metadata as untrusted. The shard-index path-escape check already exists in
`ModelPackageInspector.VerifyShards` and is mutation-verified — extend that posture, do not re-derive it.

## Traps already paid for — do not rediscover these

- **Two of the three Phase 1 abstractions already existed.** `ModelDescription` is `ModelHyperparams`;
  `ITensorSource` is `IWeightLoader` / `SafetensorsLlamaWeightLoader`. Only `ITokenizerSource` was
  genuinely missing. Do not create parallel types.
- **The seam belongs at `GgufModel`, not inside `ForwardPass`.** The original plan text said otherwise;
  it was corrected. The engine touches exactly five members, so the swap is mechanical and cannot alter
  GGUF behaviour. Rewriting the forward pass would need performance evidence that cannot be gathered
  here.
- **`SafetensorsLoader` uses `FileStream`, not mmap** — hence its `TryGetRaw` returning false.
  `TryGetMappedPointer` was added *alongside* it because the diffusion pipeline (`ZImagePipeline`,
  `VaeDecoder`, `RRDBNet`, text encoders) depends on the stream reads. Do not swap the storage model
  underneath those callers.
- **`GetTensorDataPtr` must return a pointer stable for the source's lifetime.** The engine stores it
  and reads it from other threads. Anything backed by a reallocatable buffer is unsafe here.
- **A PyTorch reference is not required and was deliberately rejected.** Differential GGUF-vs-SafeTensors
  parity is executable in CI and localises name-mapping, shape, transpose and dtype bugs better.
- **Real packages carry keys fixtures do not.** `is_llama_config`, `transformers.js_config` and
  `rope_interleaved` only appeared once a real model was downloaded. `rope_interleaved` was classified
  rather than waved through: `false` is accepted, `true` is refused as a different rotation.
  `RealPackageTouchpointTests.ConfigReader_RealSmolLm2Config_RefusesOnlyTiedEmbeddings` fails when a new
  unclassified key appears — classify it deliberately, do not widen the benign set on reflex.
- **Only `model.type == "BPE"` is accepted** from `tokenizer.json`. Unigram/WordPiece/WordLevel segment
  differently and would encode without error while disagreeing with training. SentencePiece
  (`tokenizer.model`) is therefore unimplemented, consistent with the profile advertising
  `HuggingFaceJson` only.

---

## Decision and definition of “full support”

OpenTail should support SafeTensors as an **original-weights model-package format**, alongside GGUF. It must not imply that every arbitrary SafeTensors file, every Hugging Face architecture, or every OpenTail backend works automatically.

“Fully supported” means: for each architecture profile listed in the capability report, OpenTail can load a local Hugging Face-style package; load its tokenizer and configuration; execute every advertised backend/feature combination correctly; diagnose unsupported combinations before model load; and pass the same correctness/release gates as the equivalent GGUF route.

GGUF remains the preferred local deployment format: its tokenizer/configuration metadata travels with the weights and it remains the only route for OpenTail's block-quantized fast paths until a SafeTensors backend has equivalent, measured behaviour. SafeTensors is the convenience, high-precision, original-weights route.

## Product contract

The input is a **model directory**, not merely a `.safetensors` file. The minimum supported package contains:

- `config.json`;
- one or more SafeTensors weight shards, with an optional `model.safetensors.index.json`;
- `tokenizer.json`, `tokenizer.model`, or `spiece.model`; and
- any required tokenizer sidecars (`tokenizer_config.json`, `special_tokens_map.json`, `generation_config.json`) when the selected tokenizer/profile needs them.

The loader must report the package root, config and tokenizer assets, weight dtype(s), architecture profile, source revision when available, requested backend, and effective backend. It must never silently fall back from a requested GPU/direct route to a different numerical model or write an unrequested conversion artifact.

## Existing foundation — do not redo

The first narrow Llama/Mistral package-validation slice exists in Core:

- discovers a single-file or sharded package;
- requires configuration and a tokenizer asset;
- validates dense Llama/Mistral names, F32/F16/BF16 dtypes, and exact tensor shapes;
- converts Hugging Face configuration to canonical OpenTail model metadata;
- maps Hugging Face tensor names to canonical OpenTail names; and
- exposes canonical, on-demand F32 reads through `SafetensorsLlamaWeightLoader`.

This is **not yet runnable inference**. Evolve it into the generic package layer below; do not add a competing parser.

## Phase 0 — capability contract

1. Add a versioned `ModelPackageCapability` schema: architecture profile, tokenizer family, source dtypes, backends, batching, sessions, speculation, adapters, multimodal support, and exclusions.
2. Define the first profile precisely: dense decoder-only Llama/Mistral, F16/BF16/F32, standard RMSNorm + RoPE + SiLU MLP, no projection bias, CPU only. Decide and test tied output embeddings.
3. Add package-directory capability output to inspect/doctor. Rejections must identify the missing shard, unsupported config, tensor mismatch, tokenizer family, unavailable backend, or memory requirement.
4. Publish a compatibility table in README and CLI help. Never state global “SafeTensors support”; identify the exact profile and route.

**Exit gate:** CLI/server can determine support without constructing a forward pass.

**Status: Phase 0 items 1-2 DONE (2026-08-04); items 3-4 remaining.**
`ModelPackageCapability` (versioned schema, `DenseLlamaCpu` profile) and `ModelPackageInspector`
(`Inspect` -> `ModelPackageCapabilityReport`, `RenderCapabilityTable`) are in Core, with 20 tests in
`ModelPackageInspectorTests` — mutation-verified on tied-embedding refusal, shard-path escape, and
all-faults-at-once reporting. Inspection reads config, tokenizer assets and SafeTensors headers only;
it never loads tensor data and never throws for an unsupported package (refusal is data, so callers
can print every reason at once). Estimated weight bytes are reported from header arithmetic before
load. **Decision taken: tied output embeddings are REFUSED** pending a parity fixture — they run and
are subtly wrong, which is the worst available failure mode.
**Phase 0 items 3-4 DONE (2026-08-04).** `inspect -m <dir>` branches to the capability report when the
path is a directory or a `.safetensors` file (`ModelPackageReporting.LooksLikePackage`), exits 1 for an
unsupported package, and names every rejection at once; a new `capabilities` command prints the rows
and exclusions. The table is published in README under "SafeTensors model packages", which states
plainly that inspection exists and execution does not.

Judgement calls recorded: (a) `inspect` was extended rather than adding a parallel command, since the
plan names inspect/doctor as the surfaces; (b) a bare `.safetensors` path is resolved to its parent
package as a convenience, while the contract remains "the input is a directory"; (c) `doctor` was NOT
extended — it is GGUF/runtime-probe shaped and its `-m` feeds a GGUF loader, so the package verdict
belongs on `inspect`; revisit if doctor grows a package section. **Phase 0 is complete.**

## Phase 1 — normalized package, tensor, and tokenizer interfaces

1. Replace GGUF-specific construction boundaries with small immutable abstractions:
   - `ModelDescription`: hyperparameters and architectural features;
   - `ITensorSource`: canonical names, shapes, source dtypes, and reads;
   - `ITokenizerSource`: tokens, encode/decode, special IDs, and chat template.
   Keep `GgufModel` as an implementation, not the universal model type.
2. Keep SafeTensors lazy and shard-aware. Do not materialise an entire 7B/70B model as F32 during discovery. Define ownership/disposal and bound concurrent file reads.
3. Refactor `GgufTokenizer.FromGgufModel` to construct from normalized tokenizer metadata. Implement and test adapters independently:
   - Hugging Face `tokenizer.json` BPE;
   - SentencePiece `tokenizer.model`;
   - required special-token and chat-template sidecars.
4. Explicitly parse first-profile config semantics: RoPE type/scaling, context length, head/KV-head dimensions, norm epsilon, tied embeddings, BOS/EOS/PAD IDs, and generation defaults. Reject unfamiliar semantics instead of ignoring them.

**Exit gate:** a real Llama/Mistral package tokenizes, applies its chat template, exposes canonical hyperparameters, and enumerates required tensors lazily. Token IDs match a reference tokenizer fixture.

**Status: Phase 1 item 4 DONE, items 1-3 partly pre-existing (2026-08-04).**

**Correction — items 1's abstractions largely already exist; do not create duplicates.**
- `ModelDescription` → **`ModelHyperparams`** (`ModelGraph.cs`) already is the canonical
  hyperparameter/feature record. Use it.
- `ITensorSource` → **`IWeightLoader`** already provides `Contains`/`ReadF32`/`TryGetRaw`, and
  `SafetensorsLlamaWeightLoader` already implements it over canonical names with `GetShape`/`GetDtype`.
  Use it; do not invent a parallel interface.
- `ITokenizerSource` → **genuinely missing.** `GgufTokenizer.FromGgufModel` is GGUF-coupled. This is
  the remaining work in item 3 and is the next step in the plan's execution order.

**Item 4 DONE:** `SafetensorsConfigReader` maps `config.json` to `ModelHyperparams` strictly and
returns `ModelPackageRejection`s instead of throwing. 19 tests, mutation-verified on the three rules
that matter: unknown keys refused, explicit `head_dim` honoured over the derived value, missing
context length refused rather than defaulted.

Decisions recorded:
- **Unknown config keys are refused**, against a curated mapped/benign key set. This will sometimes
  reject a package that would have run, which is the intended direction of error — the plan's review
  rules forbid ignoring unknown fields because tensor names look familiar.
- **Kept separate from `ModelHyperparams.FromGgufMetadata`**, which is deliberately permissive because
  a GGUF file's converter already resolved those defaults. Applying that tolerance to an
  author-written HF config would silently run a different model than the one on disk.
- `generation_config.json` is treated as **defaults, not semantics**: absent or malformed, it falls
  back to `config.json` values rather than blocking the load.
- `head_dim` honoured when stated; `heads % kv_heads != 0` refused as ambiguous grouping.

**Phase 1 item 3 DONE (2026-08-04) — Phase 1 is complete.**
`TokenizerSource` normalises tokenizer metadata independently of package format, and
`GgufTokenizer.FromSource(TokenizerSource)` is now the **single construction path**.
`FromGgufModel` was refactored to build a source from GGUF metadata and delegate — a pure extraction,
proven by the full Core suite staying green across the change (350/350). `HuggingFaceTokenizerSource`
loads `tokenizer.json` plus the `tokenizer_config.json` / `special_tokens_map.json` sidecars into the
same record, so a package and a GGUF carrying the same vocabulary produce the same tokenizer — which
is what makes the planned differential parity test meaningful.

14 tests, mutation-verified on: non-BPE model refused, vocabulary gaps refused, pair-form merges
normalised.

Decisions recorded:
- **Only `model.type == "BPE"` is accepted.** Unigram/WordPiece/WordLevel segment text differently;
  feeding their vocabulary to a BPE constructor encodes without error and disagrees with the model's
  training. Refusing is the only safe option, and SentencePiece (`tokenizer.model`) is therefore still
  unimplemented — the capability profile already advertises `HuggingFaceJson` only, so this is
  consistent, not a gap in the claim.
- **A vocabulary with unassigned ids is refused.** A null token would surface much later as a decode
  fault, far from the cause.
- **Sidecars are defaults, not semantics**: a malformed `tokenizer_config.json` falls back rather than
  blocking the package. A malformed `tokenizer.json` does block it.
- Merges are accepted in both `"a b"` and `["a","b"]` forms, which differ by `tokenizers` version.

**Phase 2 step 1 DONE (2026-08-04): the `IModelTensorSource` seam exists and the engine is on it.**
`IModelTensorSource` (Core) declares exactly the five members the engine uses. `GgufModel` implements
it **without a single change to its body** — the existing members already matched, which is the
strongest available evidence that the seam was cut in the right place. `ForwardPass._model` and its
constructor parameter are widened to the interface; whole-solution Release build is clean at 0
warnings. Widening a parameter is source-compatible, so no caller changed.

### BLOCKER FOUND — SafeTensors cannot yet satisfy the pointer contract

`SafetensorsLoader` holds `FileStream` per shard, not a memory mapping. That is why its `TryGetRaw`
returns `false` unconditionally. But `IModelTensorSource.GetTensorDataPtr` requires a pointer valid
for the source's lifetime: the engine stores it and reads it from other threads. A `FileStream`-backed
source cannot honour that, and returning a pointer into a buffer that may be reallocated would corrupt
inference in a way no test would localise.

Options considered:
- **(a) Memory-map the shards.** Matches GGUF exactly, gives genuinely stable pointers, keeps
  residency OS-managed, and is lazy by construction — it satisfies the plan's "do not materialise an
  entire 7B/70B model" rule for free. **Chosen.**
- (b) Materialise each tensor into owned native memory on first touch and cache it. Stable, but a
  fully-exercised 7B F16 model is ~14 GB of managed-by-us residency that the OS can no longer evict.
- (c) A separate mmap-based reader for the execution path. Rejected: the plan says do not add a
  competing parser, and header parsing would be duplicated.

**Decision: extend `SafetensorsLoader` with an opt-in memory-mapped accessor rather than replacing its
`FileStream` reads.** The existing read API is used by the diffusion pipeline (`ZImagePipeline`,
`VaeDecoder`, `RRDBNet`, text encoders) and must not change behaviour. Adding a mapping alongside it is
additive; swapping the storage model underneath those callers is not.

**Memory-mapped accessor DONE (2026-08-04).** `SafetensorsLoader.TryGetMappedPointer` maps a shard on
first request and keeps it alive until `Dispose`, giving the stable pointer the seam requires. Added
alongside the existing `FileStream` reads, which the diffusion pipeline depends on and which are
unchanged.

**`SafetensorsTensorSource` WRITTEN (2026-08-04), not yet parity-tested.** Implements
`IModelTensorSource` over the mapping, synthesising `GgufTensorInfo` descriptors under canonical
names. Note it reverses the dimension order: a Hugging Face weight is row-major
`[out_features, in_features]`, a GGUF descriptor lists dimensions fastest-varying first. The bytes are
untouched; only the descriptor order differs. Getting that backwards yields a model that loads, runs
and emits nonsense, so it must be asserted by the differential fixture, not assumed.

## REAL TOUCHPOINT (2026-08-04): `models/SmolLM2-135M-Instruct`

Downloaded `HuggingFaceTB/SmolLM2-135M-Instruct` (llama, 30 layers, hidden 576, 9 heads / 3 KV heads,
vocab 49152, 257 MB). `RealPackageTouchpointTests` exercises it and **skips when absent**, so the suite
stays hermetic. All three pass, including a tokenizer round trip through the real vocabulary.

### It immediately proved the first profile cannot run a typical modern small model

| Blocker | Status |
|---|---|
| `tie_word_embeddings: true` | **Refused by the profile.** SmolLM2-135M ties embeddings — and so do most small Llama releases. |
| `torch_dtype: bfloat16` | **Cannot execute.** `DType` has no BF16 member (GGUF numbers it 30), so BF16 cannot be described to the engine at all. |
| `is_llama_config`, `rope_interleaved`, `transformers.js_config` | Unknown to the strict reader; now classified. |

**This reframes the plan.** Phase 0 defined the first profile as "no tied embeddings, F16/BF16/F32",
but tied embeddings and BF16 are the norm rather than the exception for small models, so as specified
the profile excludes essentially every candidate reference model. Two consequences:

1. **Tied output embeddings are now the highest-priority profile feature**, not a deferred nicety.
   Without them there is no real model to prove execution against. The implementation is small — the
   output projection reuses the embedding matrix — but it needs the differential fixture to verify,
   which is why it stays refused until execution exists.
2. **The published capability row is aspirational on dtype.** It advertises F32/F16/BF16; execution
   will support F32/F16 only until `DType` gains BF16 and kernels dispatch on it. Nothing is currently
   false because nothing executes — narrow the row the moment it does.

`rope_interleaved` was classified deliberately rather than waved through: `false` is the default this
profile implements and is accepted, `true` is a different rotation and is refused. Treating it as
merely benign would have silently mis-rotated any model that sets it.

**NEXT:** the synthetic GGUF-vs-SafeTensors differential logits fixture ("Reference strategy"), which
is what makes `SafetensorsTensorSource` trustworthy, then tied-embedding support.

## Phase 2 — correct CPU dense execution

1. **Put the seam at the model, not at the forward pass.** `ForwardPass` uses exactly five members of
   `GgufModel` — `FindTensor`, `GetTensorData`, `GetTensorDataPtr`, `Tensors`, `Metadata` — across 13
   Engine files (measured 2026-08-04: 109/46/10/6/2 call sites). Extract those into an
   `IModelTensorSource` that `GgufModel` implements, and change the field/parameter types. That is a
   mechanical type swap with no behavioural change, and it lets a SafeTensors source feed the
   **existing, unmodified** transformer loop.

   This supersedes the original instruction to "extract the dense non-quantized path of `ForwardPass`
   behind `ITensorSource`". That is a logic refactor of the engine's hottest 4,000-line file, and this
   plan's own review rules make GGUF regressions release blockers — a rewrite there would need
   performance evidence to be safe, which is a much larger commitment than the goal requires. Prefer
   the seam that cannot change GGUF behaviour by construction.
2. Add F32/F16/BF16 dense linear kernels. Initially convert one matrix/row tile into bounded scratch or use an explicitly measured cache; never accidentally allocate a full F32 model copy. Report model-memory and working-set estimates before load.
3. Establish correctness evidence:
   - **differential parity against GGUF, which is the primary gate** — see "Reference strategy" below;
   - logits against PyTorch/Transformers where such a reference is actually available;
   - greedy-token parity for fixed prompts;
   - chunked versus one-shot prefill parity;
   - long-context RoPE/scaling fixtures; and
   - deterministic cache reset and session replay.
4. Enable prefill, batching, prefix caching, and sessions individually only after their parity tests pass. Start with one-token decode if it shortens the correctness path.

**Exit gate:** a published small Llama/Mistral F16/BF16 package runs end-to-end on CPU with agreed logits/tokens and all explicitly advertised context/session features.

## Phase 3 — product integration

1. Let CLI, server, sessions, static planner, model discovery, and doctor accept package directories. Persist the model format in session/plan state.
2. Add explicit policy controls:
   - `--model-format auto|gguf|safetensors`;
   - `--backend cpu|cuda|vulkan|auto`;
   - no silent format conversion;
   - an optional, separately invoked conversion command only if OpenTail elects to own conversion.
3. Give useful errors: the exact tensor/config setting that is unsupported, plus GGUF as the recommended deployment alternative where appropriate.
4. Add fixtures for supported, corrupt, sharded, missing-tokenizer, mixed-dtype, tied-output, and unsupported-architecture packages.

**Exit gate:** package directories are first-class CLI/server inputs with auditable effective configuration and no GGUF-only assumptions in user-facing flows.

## Phase 4 — CUDA, Vulkan, and hybrid routes

Do not advertise CUDA/Vulkan because CPU works. Every route needs its own source-dtype, memory-layout, transfer, cache, and numerical contract.

1. **CUDA dense F16/BF16:** direct upload/convert kernels, bounded device residency, full/partial offload, batching, KV dtypes, and load/unload tests.
2. **Vulkan dense F16/BF16:** equivalent explicit layout and device-feature checks. Device variability belongs in capability detection, not a late shader failure.
3. **Hybrid CPU/GPU:** define exactly which representation lives where and test transfers/copies. Do not reuse GGUF quantized-hybrid logic without a source-type audit.
4. Keep block-quantized CPU/CUDA/Vulkan paths GGUF-only unless SafeTensors gains a separately specified quantized convention and measured kernels.

**Exit gate per backend:** real-model logits/greedy parity versus the CPU SafeTensors reference, backend stress tests, memory accounting, and release-runner evidence.

## Phase 5 — expand architecture profiles

Each profile is a separate feature: config mapper, tensor mapper, tokenizer fixture, shape tests, reference logits, and capability rows.

1. Llama variants: tied embeddings, RoPE scaling, and long-context variants.
2. Qwen2/Qwen3 dense: attention biases, QK norms, tokenizer/template differences.
3. Gemma families: activation, norm, and embedding conventions.
4. Mistral derivatives/sliding-window attention.
5. MoE: router semantics, expert tensor layout, batching, and expert-offload tests.
6. Multimodal: separate text model, projector, vision/audio encoder, processor configuration, and prompt tokens.
7. Hybrid/recurrent/MTP models only after their state and layouts have first-class model descriptions.

A partially mapped profile must refuse to run rather than entering an inference loop.

## Phase 6 — adapters, conversion, and Hub access

1. Decide whether LoRA/PEFT is in scope. If yes, use an explicit composition layer and test base+adapter parity; do not merge destructively by default.
2. Consider an **opt-in** SafeTensors-to-GGUF conversion command only after direct CPU support is stable. Preserve provenance, require an output path, declare output dtype/quantization, and reuse a tested converter or specification.
3. Treat Hub download/cache support as separate from local package loading. Local paths remain deterministic and dependency-free.

## Reference strategy — how correctness is actually proven here

The exit gates above ask for "a real Llama/Mistral package" and "logits against PyTorch/Transformers".
Taken literally that makes every gate depend on a network download and a Python toolchain, which is a
poor foundation for a test suite and cannot run in CI. Replace it with a **differential** gate, which
is both executable and strictly better at catching the bugs this work will actually produce:

1. **Synthesise a tiny dense Llama package** (e.g. 2 layers, hidden 64, 4 heads, 2 KV heads, vocab 128)
   with deterministic pseudo-random weights, written twice: once as GGUF F32, once as a SafeTensors
   package with the equivalent Hugging Face tensor names and a `config.json`.
2. **Assert bit-comparable logits** from both routes for fixed prompts. Both feed the same transformer
   loop, so any divergence is a name-mapping, shape, transpose, or dtype-conversion bug — exactly the
   defect class of this workstream. A PyTorch reference would not localise those any better.
3. **Add F16 and BF16 variants** of the SafeTensors side against the same F32 GGUF, asserting agreement
   within the tolerance implied by the source dtype rather than exact equality.
4. Keep a real downloaded package as an **optional, skipped-by-default** integration test, so the suite
   stays hermetic while the real-model path remains exercisable on demand.

A synthetic model also makes the malformed-package, sharding, tied-embedding and mixed-dtype fixtures
cheap to generate, which the plan requires in Phase 3 anyway.

## Performance is explicitly out of scope for this workstream

The plan's release gates include performance baselines. **Those are deferred and must not be attempted
on the current development machine**, which runs other workloads and cannot produce a trustworthy
measurement; this document's own review rules reject unmeasured claims, and a noisy number is worse
than none. This workstream delivers **mechanical correctness only**.

Concretely: no throughput or latency figure may be recorded for SafeTensors until it is measured on a
quiet machine, and no SafeTensors route may be advertised on performance grounds. The one performance
obligation that remains in force is negative and is satisfied by construction rather than by
measurement — the `IModelTensorSource` seam above is a type swap, so the GGUF path executes the same
code it does today.

## Cross-cutting release gates

For every supported profile/backend/dtype row, record:

- package discovery and capability fixture;
- tokenizer encode/decode, special-token, and chat-template fixture;
- tensor count/name/shape/dtype validation;
- reference logits and greedy continuation agreement;
- prefill/chunked/batched parity where advertised;
- session save/restart/restore proof where advertised;
- repeated load/unload and malformed-package robustness;
- CPU/GPU memory ceiling and performance baseline; and
- real-hardware runner evidence.

Fuzz SafeTensors headers, shard indexes, offsets, duplicate names, and integer-overflow sizes. Treat metadata as untrusted. Do not allow shard-index paths outside the selected package root.

## Recommended execution order

1. Complete Phase 0 and expose capability reporting.
2. Complete tokenizer abstraction plus real Llama/Mistral tokenizer parity.
3. Refactor only enough CPU dense execution for one-token decode, then prove logits.
4. Add prefill, batching, sessions, and server integration separately.
5. Harden real CPU packages before starting CUDA.
6. Add CUDA, then Vulkan, behind explicit capability rows.
7. Add architecture profiles according to demand and stable reference fixtures.

## Review rules

- No broad format claim without an architecture/backend/dtype capability row.
- No end-to-end benchmark before numerical correctness and memory accounting.
- No silent conversion, quantization, tokenizer substitution, or backend fallback.
- Do not ignore unknown config fields simply because tensor names look familiar.
- GGUF regressions are release blockers: abstractions must not slow the existing deployment path without measured justification.

