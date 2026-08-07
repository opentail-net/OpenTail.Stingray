# Resolved: Gemma 4 E4B-it raw-prompt decode was an invalid coherence test

Resolved 2026-08-07. The KV-stride fix remains valid, but the remaining apparent degeneration was
not an inference defect: both tests supplied a bare continuation to an instruction-tuned checkpoint,
bypassing its own GGUF chat template. The production CLI renders the template and produces the
coherent answer: **“The capital of France is Paris.”**

The two Vulkan regressions now render the model's Jinja template through `Gemma4TestPrompt`, with
`enable_thinking=false` exactly as `RunCommand` does for stock Gemma 4 instruction checkpoints.
They retain the strict parity and coherence assertions, but on an in-distribution prompt.

---

## 1. The question

`gemma-4-E4B_q4_0-it.gguf`, prompt `"The capital of France is"`, greedy decode. Both the CPU and
Vulkan backends now agree with each other to cosine **0.999987**, and both produce:

```
[9079, 236761, 106, 236761, 106, 1, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106]
```

Token 106 is `<end_of_turn>`, token 1 is `<eos>`. The decode collapses to 4 distinct tokens.

**Answer:** it is expected behaviour for that unsupported raw-completion prompt; it is not evidence
of a shared CPU/Vulkan defect.

Two tests used to fail on exactly this, on their *coherence* assertion (their parity assertion
started passing once the KV-stride fix in §2.1 landed). Both now pass — see §7:

- `Gemma4VulkanPleE2ETests.Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent` (asserts ≥8 distinct tokens in 16 steps)
- `Gemma4VulkanNarrowedKvE2ETests.Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax`

---

## 2. Established (do not redo)

### 2.1 A real KV-stride bug was found and fixed

`PagedKvCache` stores V in a transposed region `[numKvHeads][PageSize][headDim]`, indexed by the
cache's `_headDim`. The cache is constructed with `_maxHeadDim` (512 for Gemma 4), but SWA layers
use `layerHd = 256`, while the Q/K/V projections pack heads **contiguously at layerHd**. So
`ScatterValue` read the source at `h × 512`: KV head 0 received both heads' data and KV head 1
received zeros. KV head 0 is at offset 0 and is correct under either stride, which is why query
heads 0–3 worked and 4–7 read unwritten memory.

Fixed by threading a `layerHeadDim` array into `PagedKvCache` and adding `HeadDimOf(layer)`, used
by `ValueAtHead`, `Bf16ValueAtHead`, `ScatterValue`, `ScatterValueBf16`, `GatherValue`. Pages are
pooled per layer, so no two layers share a page and a per-layer stride is safe.

Measured effect:

| metric | before | after |
|---|---|---|
| Gemma CPU↔Vulkan prefill cosine | 0.627859 | **0.999987** |
| CPU `attn_out` vs its own V (position 0) | max abs 3.248 | **0.000E+000** |
| CPU qHead→kvHead mapping | `[0,0,0,0,none,none,none,none]` | `[0,0,0,0,1,1,1,1]` |
| SmolLM2 / Qwen3 self-consistency | −0.448558 / 0.999721 | **unchanged** |

**This fix is not in question.** It is verified at every layer-0 stage (all cosine 1.000000) and
causes no change whatsoever for models without per-layer head_dim.

### 2.2 The fixture is NOT at fault

The hardcoded ids `[818, 5279, 529, 7001, 563]` decode to exactly `"The capital of France is"`,
and re-encoding the string returns the same ids. Verified with `GgufTokenizer.FromGgufModel`.
So the degenerate output was never a tokenization bug — the model really was being asked to
continue that exact English text.

**This is not a licence to tune the prompt until the assertion goes green.** The resolution in §7
replaced the raw continuation with the model's own chat template because a bare completion is
*out of distribution for an instruction-tuned checkpoint*, which is a statement about the model's
contract — not because the new prompt happened to produce nicer tokens. The assertions themselves
were left strictly intact. Changing the input to dodge a failing assertion, without an
independent reason the old input was invalid, would be cementing a bug.

### 2.3 It is not a backend-parity problem any more

Both backends are internally self-consistent (`prefill(N) == prefill(1)+decode`, cosine 1.000000
each) and agree with each other at 0.999987. Two independent implementations agreeing makes a
shared upstream cause far more likely than the same bug occurring twice.

### 2.4 Eliminated for the ORIGINAL CPU/Vulkan divergence

At position 0, softmax over a single key is 1.0, so attention output must equal V. That property
was used to eliminate: RoPE, `rope_freqs`, Q/K norms, the Q prescale, attention geometry,
embedding lookup (bit-identical), the V projection, the Gemma V-norm, and `k_eq_v`.

---

## 3. How it was resolved

### 3.1 Root cause: prompt was out of distribution — CONFIRMED

This is the **instruction-tuned** (`-it`) checkpoint. The tests feed a bare prompt with only BOS
and **no chat template** — no `<start_of_turn>user … <end_of_turn><start_of_turn>model`. An IT
model given a raw continuation prompt may legitimately emit `<end_of_turn>` immediately.

The model is fine. Note what this did NOT lead to: the ≥8-distinct-tokens assertion was kept
exactly as written, because the assertion was never the problem — the input was.

Rendered production prompt:

```text
<bos><|turn>system
<|think|>
<turn|>
<|turn>user
What is the capital of France?<turn|>
<|turn>model
```

The CPU CLI decoded: `The capital of France is **Paris**.` The fix is therefore to test the
production prompt contract, not to change Gemma math or weaken the coherence assertions.

### 3.2 If the template does not fix it — retired

The template did fix it, so this escalation path is retained only as historical diagnostic context.
If a future production-template regression occurs, investigate the shared surface in this order:

1. **Hyperparameter parsing** — `ModelHyperparams.FromGgufMetadata` for `gemma4`. Both backends
   consume the same `hp`, so a wrong `full_attention_interval`, SWA window, `layerHeadDim`
   pattern, RoPE theta pair, or `FinalLogitSoftcap` corrupts both identically. Cross-check every
   field against `llama.cpp`'s `src/models/gemma4.cpp` (referenced throughout `ForwardPass`).
2. **PLE (per-layer embeddings)** — Gemma-4-only and intricate. The build and injection sequences
   were verified to match *each other* across backends, but that only proves they agree, not that
   either matches the reference. Check against `llama.cpp` `build_inp_per_layer`.
3. **The sliding-window mask** — layer 0 is SWA. A wrong window is invisible at position 0 (one
   key) and would only bite during decode, which is exactly where the degeneracy appears.
4. **Weight loading for q4_0** — the model is q4_0 while most others here are K-quants.

### 3.3 What would settle it externally

`llama.cpp` **cannot** load this model: `unknown model architecture: 'gemma4'` (verified with the
binaries in `tools/llama.cpp`). There is no third-party oracle for Gemma 4 on this machine. If you
need one, either build a newer llama.cpp with gemma4 support, or use HF transformers with the
original safetensors checkpoint.

---

## 4. Tools available

### 4.1 `StageCapture` (src/OpenTail.Stingray.Engine/StageCapture.cs)

Internal, default-off diagnostic sink that **both** backends write to at identically named points,
so intra-layer state can be *diffed* rather than merely dumped. Stages: `embed`, `attn_norm`,
`v_proj`, `v_norm`, `attn_out`, `o_proj`, `post_attn_resid`, `post_ffn_resid`, `post_ple`,
`layer_out`. `Tests.ForwardPass` already has `InternalsVisibleTo`.

```csharp
StageCapture.Reset();
StageCapture.Enabled = true;
try { /* run CPU pass, then Vulkan pass */ }
finally { StageCapture.Enabled = false; }
var cpu = StageCapture.Find("cpu", layer, StageCapture.Stages.OProj);
var vk  = StageCapture.Find("vulkan", layer, StageCapture.Stages.OProj);
```

Adding a stage: add the name to `StageCapture.Stages`, then one `StageCapture.Record("cpu", …)` in
`ForwardPass` and one `RecordStage`/`RecordStageOf` in `GpuForwardPass` at the matching point.
Vulkan capture splits the command buffer (download must be host-visible), so it is slow — fine for
one token, not for a suite.

### 4.2 Hidden taps

`EnableHiddenTaps(layerIds)` / `HiddenTapsAt(position)` now work on CPU **and** Vulkan, including
Gemma 4, giving per-layer output vectors. Use for layer-granularity; use StageCapture for
intra-layer.

### 4.3 `STINGRAY_GEMMA4_PROBE`

Pre-existing env-gated Vulkan-only bisect: `embed`, `layers`, `stage0`. Reports magnitudes
(nonFinite / maxAbs / rms) and throws. Useful for blow-ups; it cannot distinguish "slightly off"
from "pointing elsewhere", which is why StageCapture exists.

---

## 5. Traps that cost real time here

- **Flag spellings differ by entry point** — both verified 2026-08-07. Through `dotnet test <proj>
  -- …`: `--filter-class` / `--filter-method` (and `--minimum-expected-tests`). Running the built
  `…/bin/Release/net10.0/<Proj>.exe` directly: **single-dash** `-class` / `-method`, plus
  `-class-` / `-method-` to exclude; it rejects the double-dash forms with `error: unknown
  option`. The exe is much faster for iterating, which is why this bites.
- **Never pass `--nologo`** to `dotnet test` — MTP rejects it and reports "Zero tests ran", which
  reads exactly like a discovery failure.
- **`models/` is populated (17 GB), so `Tests.ForwardPass` takes ~10 minutes**, not 35 seconds.
  Model-gated tests use `if (path is null) return;` — a **silent pass**, so a green run with
  `models/` absent proves nothing. ~616 such sites exist.
- **Do not assume the CPU backend is the oracle.** Twice in this investigation the CPU was the
  faulty side while a test treated it as the reference (this KV-stride bug, and int8 prefill
  quantisation `SimdKernels.Q8PrefillEnabled`). Prefer oracle-free invariants:
  - self-consistency: `prefill(N)` must equal `prefill(1) + decode`, same backend;
  - at position 0, attention output must equal V, broadcast across each GQA group.
  Both are in `PrefillDecodeSelfConsistencyTests` / the probe harness.
- **Run one test process at a time.** Overlapping runs on a 6-core box (kernels are internally
  `Parallel.For`'d) inflated a 35 s suite to 640 s and produced a phantom "regression".

---

## 6. Repo state

- Fix: `PagedKvCache.cs` (per-layer V stride), `ForwardPass.cs` (passes `layerHeadDim`, dead
  `slotStride` removed).
- Keep: `StageCapture.cs`, `PrefillDecodeSelfConsistencyTests.cs`.
- Deleted: `tests/OpenTail.Stingray.Tests.ForwardPass/_TempPleProbe.cs` — a throwaway harness
  committed by accident, removed 2026-08-07. Recoverable from git history if the ~20 probes are
  wanted for technique; it was never a test suite (several probes assert nothing).
- Added: `tests/OpenTail.Stingray.Tests.ForwardPass/Gemma4TestPrompt.cs` — renders the model's own
  GGUF chat template with `enable_thinking=false`, matching `RunCommand.ResolveThinkingOff`, which
  returns true (thinking off) for `arch == "gemma4"`.
- Both Gemma E2E tests now pass (§7) with their original parity and coherence assertions intact;
  only their input changed, via `Gemma4TestPrompt`. Do **not** weaken those assertions later to
  accommodate a failure — see the warning in §2.2.

**Verification status of the KV-stride fix.** Complete. Targeted evidence first (§2.1: every
layer-0 stage at cosine 1.000000, CPU `attn_out` bit-exactly equal to V, SmolLM2/Qwen3 numbers
unchanged to all printed digits), then the broad one that actually mattered: a full
`Tests.ForwardPass` run with `models/` populated — **1323 tests, 0 failed, 1 skipped, 661 s**
(2026-08-07). The single skip is the glslc-gated Vulkan compile-fallback test.

That run is what clears the blast radius. `PagedKvCache` backs every model and both the dense and
Gemma paths, and the change also touched `GatherValue`/`ScatterValue`, which only SnapKV
eviction/compaction exercises — those live in this suite and pass.

## 7. Resolution evidence

- Production CLI, the real QAT model, CPU backend, greedy decode: `The capital of France is Paris.`
- `Gemma4_E4B_Q4_0_VulkanForward_LongDecodeIsCoherent`: passed (1/1, 2026-08-07).
- `Gemma4_E4B_Q4_0_VulkanNarrowedKv_MatchesFp32Argmax`: passed (1/1, 2026-08-07).

The tests retain their original CPU/Vulkan parity and non-degeneracy requirements. Only their input
now follows the model's production chat-template contract.
