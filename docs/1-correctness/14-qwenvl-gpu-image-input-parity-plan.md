# Plan: Qwen3-VL / Qwen2.5-VL / PaddleOCR-VL GPU image-input parity

**Entry in:** `docs/1-correctness/bugstofix.md`, item **14**.

## 0. Current state

Three vision text decoders already work correctly on CPU: `qwen3vl`, `qwen2.5-vl`, and `paddleocr`.
The CPU image path has the required M-RoPE image-position machinery through:

- `src/OpenTail.Stingray.Engine/MRopeImageLayout.cs`;
- `src/OpenTail.Stingray.Engine/ForwardPass.Decode.cs`;
- `ForwardPass.AddMRopeImage(...)`.

Qwen3-VL additionally has deepstack injection through `ModelHyperparams.DeepstackMapping` and the
CPU trunk.

### Already completed: full Vulkan offload (`GpuForwardPass`)

Implemented 2026-09-28:

- per-pair M-RoPE positions;
- new Vulkan `RoPENeoxPairPos` shader;
- `MRopeImageLayout` shared with the CPU path;
- Qwen3-VL deepstack uploaded through `ForwardEmbedding`;
- image tokens and subsequent text tokens use the M-RoPE position layout;
- Qwen3-VL real-weight parity: final-logit cosine approximately `0.9995` vs CPU, same top token;
  controls without image registration/deepstack show measurable divergence.

Existing regression: `tests/OpenTail.Stingray.Tests.Vulkan/Qwen3VlVulkanMRopeParityTests.cs`.
Do not redesign or re-port the full Vulkan M-RoPE implementation.

### Remaining

1. CUDA full offload: `CudaForwardPass`.
2. CUDA hybrid offload: `CudaHybridForwardPass`.
3. Vulkan layer split: `VulkanLayerSplitForwardPass`.
4. Verify Qwen2.5-VL and PaddleOCR-VL on full Vulkan; current end-to-end proof is Qwen3-VL.
5. Verify the Qwen3-VL IMROPE fourth section remains unrotated on GPU text tokens: pairs 61–62
   must receive position `0`.
6. Ensure the CLI does not reject these models once the selected pass supports the required features.

## 1. Scope and non-goals

### In scope

| Model | CPU | Full Vulkan | Vulkan `-g N` | Full CUDA | CUDA hybrid |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL | proven | proven | not proven | not proven | not proven |
| Qwen2.5-VL | proven | implementation exists; needs E2E proof | not proven | not proven | not proven |
| PaddleOCR-VL | proven | implementation exists; needs E2E proof | not proven | not proven | not proven |

Include Qwen3-VL deepstack injection.

### Not in scope

- new vision encoder work;
- new model architecture support;
- changing CPU M-RoPE mathematics;
- changing the already-proven full Vulkan shader unless a test demonstrates an actual defect;
- server multimodal plumbing except what is strictly required to expose existing forward-pass
  capability;
- CUDA performance optimization;
- claiming CUDA verification on hardware this machine does not have.

## 2. Establish the feature contract

Before changing any backend, document the exact CPU semantics every backend must reproduce.

For an image token at KV slot `s`, `MRopeImageLayout.Position(s)` gives `(t, h, w)`, and
`FillPairPositions(...)` maps each rotation pair to its component:

- section 0 → `t`;
- section 1 → `h`;
- section 2 → `w`;
- section 3 → `0`.

For Qwen3-VL the sections are `[24, 20, 20, 0]`, so the final two pairs, 61 and 62 for a
128-wide head, must never rotate. Text positions are `(p, p, p)` with the appropriate image-token
compaction adjustment.

The Qwen3-VL decoder additionally receives vision deepstack slices through
`ModelHyperparams.DeepstackMapping`. Add those slices at their mapped transformer layers in the same
order as the CPU implementation.

**The CPU path is the semantic reference.** The full Vulkan path is the second implementation already
proven against it.

## Phase 0 — Freeze current baselines

Record exact real-weight baselines before touching the remaining passes. For each of Qwen3-VL,
Qwen2.5-VL, and PaddleOCR-VL record checkpoint, mmproj checkpoint, image, prompt, image-token count,
token-grid dimensions, prompt token IDs, greedy output, final-logit checksum/top token, CPU runtime,
and full Vulkan runtime where available.

For Qwen3-VL retain the existing `Qwen3VlVulkanMRopeParityTests` values as the frozen Vulkan
reference. Record a text-only Qwen3-VL run independently from the image path; the fourth M-RoPE
section affects ordinary text tokens even though it is primarily exposed by multimodal use.

## Phase 1 — Prove the Qwen3-VL fourth section on GPU

Before touching CUDA/hybrid, settle the existing GPU pair 61–62 concern. Do not assume a precomputed
GPU RoPE table and the new `RoPENeoxPairPos` path are equivalent.

Trace the actual GPU path:

- `MRopeImageLayout.FillPairPositions`;
- host-side `_mropePairPosHost`;
- `_mropePairPos`;
- `VulkanBackend.RoPEPairPositions`;
- `Shaders.RoPENeoxPairPos`.

For a normal Qwen3-VL text token verify `pairPositions[61] == 0` and `pairPositions[62] == 0`. Verify
the shader using deterministic input where the tested pair's only difference is position `0` versus a
non-zero position. Expected: pairs 0–60 rotate normally, while pairs 61–62 are bitwise or within test
noise unchanged when position is zero. Identify whether another GPU RoPE-frequency/table path is
involved. If a table is dead code for M-RoPE dispatch, document that rather than changing unused data.

### Regression test

Add a small backend-level regression pinning Qwen3-VL `[24,20,20,0]` mapping without requiring the
full multimodal checkpoint.

## Phase 2 — Verify full Vulkan on all three models

Verification first: the generic mechanism exists, but each model needs its own end-to-end proof. Run
Qwen3-VL, Qwen2.5-VL, and PaddleOCR-VL on full Vulkan. For each compare CPU to full Vulkan at
minimum: image embedding output, first post-image hidden state, one or more attention-layer hidden
states, final logits, and greedy output.

Qwen3-VL should retain the existing approximately `0.9995` parity target. For Qwen2.5-VL and
PaddleOCR-VL, measure against their CPU outputs first; do not invent architecture-specific tolerances
before measuring numerical noise.

Distinguish "generic M-RoPE implementation works" from "this model's full Vulkan image path is
proven". Do not infer the latter from Qwen3-VL alone.

## Phase 3 — Add M-RoPE state to CUDA full offload

`CudaForwardPass.cs` currently uses ordinary scalar-position RoPE via `_gpu.RoPE(... position ...)`.
Add CUDA's equivalent of Vulkan's per-pair path, preferably at the same semantic seam: shared
`MRopeImageLayout`, per-token pair-position buffer, and a CUDA kernel operating on NEOX pairs with one
position per pair. Do not create a Qwen-specific implementation.

Conceptually:

```text
ordinary model → scalar-position RoPE
M-RoPE model   → MRopeImageLayout → pair positions → per-pair CUDA RoPE
```

Preserve the existing ordinary-RoPE path unchanged.

## Phase 4 — Add deepstack to CUDA full offload

Qwen3-VL image embeddings can contain `[embedding][deepstack slice 1][deepstack slice 2]...`. CUDA
currently accepts only the ordinary embedding width at its `ForwardEmbedding` seam. Extend it so both
`width = embDim` and `width = embDim × (1 + NumDeepstack)` match `ForwardPass` and `GpuForwardPass`
semantics.

Inject each deepstack slice at the exact layer from `ModelHyperparams.DeepstackMapping`, in the same
order as CPU and full Vulkan. Do not invent a CUDA-specific deepstack mapping.

## Phase 5 — Make CUDA full-offload image registration explicit

Add `CudaForwardPass.AddMRopeImage(startSlot, nx, ny)` with the contract of
`ForwardPass.AddMRopeImage` and `GpuForwardPass.AddMRopeImage`. Image registration belongs to the
forward-pass instance, not the CLI. The CLI registers the image with whichever M-RoPE-capable forward
pass it selected, keeping position semantics backend-independent.

## Phase 6 — CUDA full-offload parity tests

Add hardware-gated CUDA tests. For Qwen2.5-VL and PaddleOCR-VL compare CPU and CUDA image embedding,
first decoder hidden state, final logits, and greedy result. For Qwen3-VL cover the same plus M-RoPE
pair mapping, deepstack layer injection, and explicit pairs 61–62 zero-position behavior.

Use `Assert.SkipUnless(...)` or the existing CUDA hardware gate when CUDA is absent. On this
 development machine, successful compilation is not CUDA runtime verification. Record CUDA as
implemented but hardware-unverified until an NVIDIA system runs the real-weight tests.

## Phase 7 — Vulkan layer-split support

`VulkanLayerSplitForwardPass` wraps `GpuForwardPass` for layers `[0, N)` and `ForwardPass` for
layers `[N, L)`. Add wrapper-level `AddMRopeImage(...)` that registers the image with both underlying
passes. Both sides must compute identical logical M-RoPE positions for the same sequence slots. Do
not reconstruct positions separately in the wrapper.

## Phase 8 — Vulkan layer-split embedding/deepstack seam

`VulkanLayerSplitForwardPass` currently exposes ordinary token forwarding only. For vision input add
the `ForwardEmbedding(...)` equivalent and preserve deepstack slices across the layer boundary.

```text
vision encoder → embedding + deepstack → GpuForwardPass [0, N) → hidden state
				→ ForwardPass [N, L) → logits
```

Deepstack must be consumed by whichever side owns each mapped layer. For example, a split before a
mapped deepstack layer means CPU consumes that slice; a split after it means GPU consumes the slice.
This must preserve the logical mapping. The CPU half should continue using its existing
`ForwardFromHidden(...)` mechanism where appropriate.

## Phase 9 — Vulkan split test matrix

Do not test only one arbitrary `-g N`. Qwen3-VL has multiple deepstack insertion points; place the
boundary before the first mapped layer, exactly at it, immediately after it, around the second and
third, and at one ordinary split away from them. Generate positions from the actual metadata-derived
mapping rather than hard-coding layer numbers in implementation. This catches GPU/CPU splits where
both sides work independently but the deepstack slice is injected on the wrong side.

For Qwen2.5-VL and PaddleOCR-VL there is no Qwen3-style deepstack requirement, so split tests
primarily exercise M-RoPE state propagation.

## Phase 10 — CLI/backend routing

`RunCommand` currently rejects image input for M-RoPE models when the selected pass is not
`ForwardPass` or `GpuForwardPass`. This was correct when those were the only safe implementations.
Once new paths exist, replace the type-specific restriction with a capability check. Allow image plus
M-RoPE only if the selected pass supports embedding input, M-RoPE image registration, and required
deepstack semantics. Do not maintain a CLI mapping from model names to backend eligibility; capability
belongs to the forward pass so future M-RoPE models do not repeat this issue.

## Phase 11 — End-to-end backend matrix

Once implementation is complete, run the real image path:

| Model | CPU | Full Vulkan | Vulkan split | Full CUDA | CUDA hybrid |
| --- | --- | --- | --- | --- | --- |
| Qwen3-VL | baseline | existing + regression | required | required | required |
| Qwen2.5-VL | baseline | required | required | required | required |
| PaddleOCR-VL | baseline | required | required | required | required |

For every runnable cell use the same checkpoint, mmproj, image, prompt, and token ordering; compare
final logits and greedy output. For unavailable CUDA hardware record
`implemented / build-verified / runtime-unverified`, not parity.

## Phase 12 — Reference comparison

Where practical, compare at least one complete image run for each model against llama.cpp. The
reference hierarchy is llama.cpp → CPU `ForwardPass` → Vulkan full/split and CUDA full/hybrid.
Existing CPU implementations are externally validated, so backend debugging should normally proceed
from GPU/split-versus-CPU divergence to first divergent layer and then backend repair, rather than
repeatedly comparing final responses. Retain existing llama.cpp-backed Qwen3-VL vision tests as
external evidence.

## Phase 13 — Regression coverage

Add permanent tests for actual failure modes, not only the happy path:

### M-RoPE

- image registration changes positions;
- image-token grid maps correctly to `(t,h,w)`;
- text after an image receives correct shifted positions;
- Qwen3-VL pairs 61–62 remain unrotated;
- full Vulkan retains existing parity.

### Deepstack

- correct number of slices;
- correct slice ordering;
- correct mapped-layer injection;
- no duplicate injection;
- no dropped slice.

### Layer split

- image embedding survives GPU-to-CPU boundary;
- M-RoPE registration survives the boundary;
- deepstack survives the boundary;
- split immediately before/after a deepstack layer remains correct.

### Routing

Cover CPU image, full Vulkan image, Vulkan split image, and CUDA image when hardware exists.

## Phase 14 — Smallest implementation principle

Prefer extending existing abstractions over creating parallel implementations. Reuse the shared
`MRopeImageLayout`, `AddMRopeImage(...)`, embedding-input width contract, and deepstack mapping
semantics. Backends should differ only in execution of the same mathematical operations.

Avoid separate Qwen3-specific CUDA logic, separate PaddleOCR M-RoPE logic, model-name checks in CLI,
duplicated `(t,h,w)` calculations, and duplicated deepstack layer mapping.

## Success criteria

Item 14 is complete when:

- Qwen3-VL image input remains correct on CPU and full Vulkan;
- Qwen2.5-VL image input is proven on full Vulkan, not inferred from generic code;
- PaddleOCR-VL image input is proven on full Vulkan;
- Vulkan `-g N` works for all three relevant models;
- Qwen3-VL deepstack works on both sides of a Vulkan layer split;
- CUDA full offload has the same M-RoPE semantics;
- CUDA full offload supports Qwen3-VL deepstack;
- CUDA hybrid carries image-position/deepstack state correctly;
- CLI permits image input based on forward-pass capability rather than backend-type/model-name special cases;
- Qwen3-VL pairs 61–62 have an explicit unrotated regression;
- no GPU path is called runtime-verified without real hardware;
- available end-to-end paths produce the same greedy answer as CPU/reference within the measured numerical envelope.

## Key rule

**Do not chase final-answer equality first.** For every backend trace image registration → pair positions → RoPE → embedding/deepstack → first divergent layer → final logits → greedy answer, then fix the first divergence.

The CPU `ForwardPass` defines the semantic contract; the already-verified full Vulkan implementation is the working GPU precedent. CUDA and layer-split work should extend those mechanisms rather than invent model-specific paths.
