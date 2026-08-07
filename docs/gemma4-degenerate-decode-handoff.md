# Handoff: Gemma 4 degenerate decode (both backends agree, output is still wrong)

Investigation brief for whoever picks this up next. Everything in **Established** was measured,
not inferred — please don't re-derive it. Everything in **Open** is genuinely unknown.

---

## 1. The question

`gemma-4-E4B_q4_0-it.gguf`, prompt `"The capital of France is"`, greedy decode. Both the CPU and
Vulkan backends now agree with each other to cosine **0.999987**, and both produce:

```
[9079, 236761, 106, 236761, 106, 1, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106]
```

Token 106 is `<end_of_turn>`, token 1 is `<eos>`. The decode collapses to 4 distinct tokens.

**Is this correct behaviour for this model on this prompt, or is there a defect shared by both
backends?**

Two tests currently fail on exactly this, both on their *coherence* assertion (not their
parity assertion, which now passes):

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
Do not "fix" the tests by changing the tokens.

### 2.3 It is not a backend-parity problem any more

Both backends are internally self-consistent (`prefill(N) == prefill(1)+decode`, cosine 1.000000
each) and agree with each other at 0.999987. Two independent implementations agreeing makes a
shared upstream cause far more likely than the same bug occurring twice.

### 2.4 Eliminated for the ORIGINAL divergence (may still matter here)

At position 0, softmax over a single key is 1.0, so attention output must equal V. That property
was used to eliminate: RoPE, `rope_freqs`, Q/K norms, the Q prescale, attention geometry,
embedding lookup (bit-identical), the V projection, the Gemma V-norm, and `k_eq_v`.

---

## 3. Open — the actual investigation

### 3.1 Leading hypothesis: prompt is out of distribution

This is the **instruction-tuned** (`-it`) checkpoint. The tests feed a bare prompt with only BOS
and **no chat template** — no `<start_of_turn>user … <end_of_turn><start_of_turn>model`. An IT
model given a raw continuation prompt may legitimately emit `<end_of_turn>` immediately.

If true, the model is fine and the ≥8-distinct-tokens assertion is unreasonable as written.

**First experiment.** Render the model's own chat template (`JinjaChatTemplate` from the GGUF
metadata) around "What is the capital of France?", tokenize that, and decode greedily. If the
output becomes coherent, the hypothesis holds and the fix is to the tests, not the engine.
Do this before anything else — it is cheap and decisive either way.

### 3.2 If the template does not fix it

Then something upstream of both backends is wrong. Shared surface, in rough priority order:

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

- **The xunit runner takes `-class` / `-method` with a SINGLE dash**, plus `-class-` / `-method-`
  to exclude. `--filter-class` is rejected. (CLAUDE.md documents the wrong form.)
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
- **Delete before committing:** `tests/OpenTail.Stingray.Tests.ForwardPass/_TempPleProbe.cs` — a
  throwaway harness that was committed by accident. It contains ~20 probes worth reading for
  technique, but it is not a test suite and several probes assert nothing.
- Do **not** adjust the two failing Gemma tests to match current output. Their parity assertion is
  now correct and passing; only the coherence assertion fails. Changing them to accept a
  degenerate decode would cement whatever is left.

**Verification status of the KV-stride fix.** Targeted evidence is strong (§2.1: every layer-0
stage at cosine 1.000000, CPU `attn_out` bit-exactly equal to V, SmolLM2/Qwen3 numbers unchanged
to all printed digits). But a **full `Tests.ForwardPass` run has NOT been completed since the
fix** — it was started and cancelled. `PagedKvCache` backs every model and both the dense and
Gemma paths, so run the full suite (~10 min with `models/` populated) before trusting it broadly.
Pay particular attention to SnapKV tests: eviction/compaction calls `GatherValue`/`ScatterValue`,
which this change also touched.

## 7. Definition of done

Either:
- **(a)** the chat template explains it — then fix the tests to use the template, and say so
  explicitly in the test comment so the next person doesn't re-litigate it; or
- **(b)** a genuine defect is found — then the fix must keep Gemma CPU↔Vulkan at ~0.999987, keep
  SmolLM2/Qwen3 self-consistency bit-unchanged, and make both Gemma E2E tests pass on their
  original assertions.
