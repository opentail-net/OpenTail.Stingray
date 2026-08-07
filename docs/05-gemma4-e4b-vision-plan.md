# Gemma 4 E4B Multimodal (Vision) — Research & Implementation Plan

Status: **V0 mmproj loader and V1 fixed-grid preprocessing implemented; encoder, projector,
embedding splice, and end-to-end image parity remain open.** Tracked by **issue #126**.

> ## ⚠️ Verification update (2026-06-15) — architecture confirmed from the real mmproj
> The §1 "verification debt" is now **retired**: the E4B mmproj header was dumped
> (`E:\models\gemma-4-E4B-it-mmproj.gguf`, ~992 MB). **The Gemma-3n / MobileNet-V5 assumption
> below is WRONG.** What E4B actually ships:
> - **Vision** `clip.vision.projector_type = gemma4v` — a **transformer ViT** encoder (NOT a
>   conv MobileNet): `block_count=16`, `embedding_length=768`, `head_count=12` (head_dim 64,
>   with QK-norm), GeGLU FFN (`feed_forward_length=3072`), conv patch-embed `v.patch_embd.weight
>   [16,16,3,768]`, learned 2D position table `v.position_embd.weight [768,10240,2]`, `image_size=224`,
>   `patch_size=16`, image_mean=[0,0,0]/std=[1,1,1].
> - **Audio** `clip.audio.projector_type = gemma4a` — a separate ~12-block conformer-style encoder
>   (`a.*` tensors, `num_mel_bins=128`); `clip.has_audio_encoder=True`.
> - Projectors: `mm.input_projection [768→2560]` (vision) and `mm.a.input_projection [1536→2560]`
>   (audio), into the E4B text embed dim 2560.
>
> **Net:** the plan's *conclusion* (E4B is encoder-FULL and needs a real encoder forward pass,
> unlike the 12B) holds; the *specifics* (MobileNet-V5 conv stack, gemma3n, 768² input, 256 fixed
> tokens, `<start_of_image>` markers) do not. The good news: a ViT reuses our existing
> attention/MLP/RMSNorm kernels almost directly — this is the plan's §4 "SigLIP fallback" path,
> which turns out to be the actual architecture. Phase V2 should target the `gemma4v` ViT, not
> MobileNet-V5; rewrite §1/§2/§3 hyperparameters against the header facts above before coding.
>
> **This is NOT the 12B path.** The Gemma 4 **12B** uses encoder-free `gemma4uv` (raw patches →
> linear projection, no ViT) and is implemented in `src/OpenTail.Stingray.Vision` (issue #250, see the
> gemma4uv section of `docs/OpenTail.Stingray-Design.md`). E4B (`gemma4v`+`gemma4a`) has only its
> `gemma4v` load/preprocessing boundary implemented; it is not yet usable for image inference.

> **Implementation update (2026-08-07): V0 is now complete.** `Gemma4VVisionModel` owns and
> validates the E4B `gemma4v` mmproj separately from the encoder-free 12B `gemma4uv` model. It
> resolves the patch/position/projector tensors plus the complete 13-tensor inventory for every
> one of the 16 ViT blocks; the real 992 MB mmproj fixture pins geometry and boundary tensor
> shapes. This is a load/ownership boundary only: preprocessing, ViT forward, reduction,
> embedding splice, and image-mask semantics remain open.

> **Reference-tool correction (2026-08-07):** the local `tools/llama.cpp` binary labelled build
> 8585 (`cad2d3884`) is **not** a usable E4B oracle. `llama-mtmd-debug` rejects the paired text
> GGUF before mmproj processing with `unknown model architecture: 'gemma4'`. Do not use this build
> for E4B preprocessing/encoder/parity evidence; acquire a llama.cpp build that actually admits
> `gemma4`, or capture reference intermediates from another confirmed implementation. V2 must not
> be guessed from tensor names.

This doc scopes adding **image input** to the already-working Gemma 4 E4B text path. Audio (the
other E-model modality) is noted but deferred. It is the multimodal counterpart to
`docs/done/gemma4-e4b-implementation-plan.md` (whose *text* phasing is now stale — the gemma4 text
trunk is implemented in `ForwardPass.cs`: embedding scale, PLE, dual-RoPE, SWA, cross-layer
KV-share, GeGLU, final-logit softcap are all present).

> **Reading rule:** the header facts in the verification update above are current. The detailed
> MobileNet/Gemma-3n draft retained below is historical research, not an implementation contract;
> do not revive its 768², 256-token, projector-type, or encoder-graph assumptions. Rewrite the
> implementation phases around the verified `gemma4v` ViT before coding.

## TL;DR

- Gemma 4 is **natively multimodal** (all sizes: text+image; E2B/E4B add audio). The user's
  instinct was right.
- E4B remains **text-only today** because its `gemma4v`/`gemma4a` encoders are not implemented.
  This is not a repository-wide absence of vision support: the separate 12B `gemma4uv` projector
  path is implemented in `OpenTail.Stingray.Vision` and should be reused where its abstractions fit.
- **Verified architectural finding:** E4B uses the `gemma4v` transformer ViT encoder (16 blocks,
  768-wide, 12 heads, QK-norm and GeGLU), not the historical MobileNet-V5/Gemma-3n assumption.
  Its core attention/MLP/RMSNorm operations should be evaluated against existing vision and decoder
  abstractions; it is not a direct reuse of the diffusion convolution pipeline.
- A Gemma-4-capable external reference implementation is required for parity debugging. The local
  llama.cpp build is not that oracle: it rejects `general.architecture=gemma4` before mmproj input.

## 1. How Gemma 4 multimodal works (the parts we must replicate)

A multimodal Gemma model is **two GGUF files**:

1. the text model we already load (`-m ...gemma-4-E4B-it-*.gguf`), and
2. a **multimodal projector** `mmproj-*.gguf` — the **vision encoder + projector** weights.
   (Available alongside the text GGUF, e.g. `ggml-org/gemma-4-E4B-it-GGUF`,
   `unsloth/gemma-4-E4B-it-GGUF`.)

Pipeline (image → text-model input):

1. **Preprocess** the image: decode → RGB float → resize to the encoder's fixed input
   (Gemma-3n/E-model: **768×768**, *to confirm against the E4B mmproj header*) → normalize.
   Optionally **Pan & Scan**: tile wide/tall images into extra crops + a global thumbnail, each
   encoded independently.
2. **Vision encoder** (MobileNet-V5-300M): conv stem → inverted-residual / depthwise-separable
   blocks → multi-scale feature fusion → a feature map that is pooled/projected to a fixed budget
   of **256 soft vision tokens per image/crop** (the Gemma-3n number; *confirm for E4B*).
3. **Projector MLP** (`mm.*` tensors): maps vision features into the **text embedding dim**
   (E4B `embedding_length` = 2560).
4. **Splice** the 256 embeddings into the token sequence: the chat template emits placeholder
   image tokens wrapped in `<start_of_image>` / `<end_of_image>`; those placeholder positions are
   **overwritten with the projected vision embeddings** (fed as raw input `embd`, not token IDs).
   The image then occupies 256 real positions in the KV cache; position IDs advance by 256.
5. The combined sequence flows through the **existing** gemma4 text decoder unchanged, except for
   attention masking (below).

**Attention over image tokens:** in the Gemma family, image soft-tokens attend **bidirectionally
within their own span**, while text remains causal. This interacts with `PagedKvCache` and the
causal mask and needs explicit handling (build a causal mask that is bidirectional inside each
image's 256-token block). *Confirm Gemma-4-E behavior matches Gemma-3 here.*

> **Historical verification debt (partly closed):** the real header now establishes a 224px,
> 16px-patch `gemma4v` ViT, mean=[0,0,0]/std=[1,1,1] preprocessing, and its
> 16×768/12-head/GeGLU geometry. Exact image-token reduction and mask semantics still need to be
> derived from runtime behaviour and checked against llama.cpp's Gemma 4 `clip`/`mtmd`
> implementation before the encoder is wired.

### mmproj GGUF structure (llama.cpp `clip` convention)

- Metadata: `clip.has_vision_encoder`, `clip.projector_type` (verified: `gemma4v`),
  `clip.vision.image_size`, `clip.vision.patch_size`, `clip.vision.embedding_length`,
  `clip.vision.projection_dim`, `clip.vision.block_count`, plus MobileNet-specific keys.
- Vision tensors: `v.*` (patch/stem conv, per-block conv/attn/norm weights, `v.post_ln`).
- Projector tensors: `mm.*` (the projection MLP / `mm.input_projection`).
- Audio (E-models): `a.*` + `clip.has_audio_encoder` — **out of scope for the first pass.**

## 2. What we already have to build on

| Asset | Location | Reuse |
|---|---|---|
| GGUF parser (mmap, multi-shard, metadata) | `Core/GgufModel.cs` | Load the `mmproj` as a 2nd model handle |
| Conv2d / activations / pixel-shuffle / upsample (GPU) | `Core/IImageOpsBackend.cs`; `Cuda/CudaBackend.cs:3621`, `Vulkan/VulkanBackend.cs:1584` | Patch embedding and image preprocessing support |
| Conv2D / GroupNorm / LayerNorm / Gelu / resize (CPU) | `Diffusion/DiffusionOps.cs` | CPU patch embedding + image resize |
| GEMM / RMSNorm / attention / GeLU SIMD kernels | `Cpu/SimdKernels.cs`, backends | ViT blocks and projector MLP |
| gemma4 text decoder (PLE, SWA, KV-share, dual-RoPE, softcap) | `Engine/ForwardPass.cs` | Unchanged consumer of spliced embeddings |
| **Embedding entry point** | `ForwardPass.cs:1212` `EmbedToken` call site / `1824` `EmbedToken` wrapper / `1885` `EmbedTokenInto(token, dest)` definition; scale at `:1215` | **The splice seam** — add an overload that writes a precomputed embedding instead of a `token_embd` lookup |

The essential pieces are a GGUF transformer runtime plus a patch-embedding/image-preprocessing
seam. The encoder itself is a transformer, so its attention/MLP/RMSNorm work should reuse decoder
and vision abstractions rather than be designed as a diffusion-convolution pipeline.

## 3. Phased implementation plan

Suggested new module: **`src/OpenTail.Stingray.Vision`** (mmproj loader, preprocessing, encoder,
projector), keeping vision concerns out of `Core`/`Engine` until the seam is stable.

### Phase V0 — mmproj/clip GGUF loader (low risk) — **DONE 2026-08-07**
`Gemma4VVisionModel` parses the verified `clip.*` geometry and validates the patch/position/
projector tensors plus the full 16×13 block tensor inventory. The real E4B mmproj smoke test
pins the loader to 224px / 16px patch / 768 wide / 12 heads / 3072 FFN / 16 blocks. Token
reduction and mask semantics are intentionally not inferred from this structural phase.

### Phase V1 — image preprocessing (low risk) — **fixed-grid core complete 2026-08-07**
`Gemma4VImagePreprocessor` now performs deterministic align-corners RGB resize to the
mmproj-declared fixed grid, packs planar CHW, and applies the header's three channel
mean/std values. Unit coverage pins channel order, interpolation, affine normalisation, and
invalid input handling. **This is a bounded implementation, not external parity evidence:** retain
the alignment/interpolation choice behind V2's reference gate until a Gemma-4-capable oracle
confirms it. PNG/JPEG decoding is already provided by `ImageIO`; **Pan & Scan remains open** until
its exact E4B crop policy is derived from the reference rather than guessed.

### Phase V2 — `gemma4v` ViT encoder forward pass (HIGH risk)
The load-bearing piece. Implement the verified 16-block, 768-wide, 12-head QK-norm/GeGLU
transformer encoder, preceded by its 16px convolutional patch embedding and learned 2D position
table. Start with a CPU reference that reuses existing RMSNorm, attention and MLP semantics, then
define GPU capability gates. **Parity-gate each stage** against llama.cpp's Gemma 4 `clip` path
using captured intermediate tensors. The main unknowns are exact vision attention/mask and token
reduction semantics, not a MobileNet convolution graph.

### Phase V3 — token reduction + projector MLP (medium risk)
Apply the verified model's token-reduction rule (do not hard-code the historical 256-token
assumption), then run the `mm.*` projector to the 2560-wide text embedding space. Parity-gate the
resulting embedding sequence against llama.cpp `mtmd_get_output_embd`.

### Phase V4 — embedding splice + bidirectional mask (HIGH risk)
- Add `ForwardPass`/`Prefill` support to accept **precomputed input embeddings** at given
  positions (overload of `EmbedTokenInto`; skip `token_embd` lookup; decide embedding-scale
  handling for image rows — text tokens get `× sqrt(2560)`, image embeddings come pre-scaled from
  the projector, *confirm*).
- Build the **causal-except-bidirectional-within-image** attention mask; verify interaction with
  SWA layers and cross-layer KV-share in `PagedKvCache`.
- Chat-template rendering of `<start_of_image>`/`<end_of_image>` + the soft-token placeholders
  (`GgufTokenizer` Jinja path).
- Acceptance: greedy-decode parity vs `llama-mtmd-cli` on a fixed image+prompt (e.g. "describe
  this image") for N tokens.

### Phase V5 — CLI + API surface (medium risk)
- CLI: `--image <path>` (and Pan & Scan toggle) on the existing `Spectre.Console.Cli` frontend.
- Server: image **content blocks** in `/v1/messages` (Anthropic) and `/v1/chat/completions`
  (OpenAI) — base64 / URL image parts → preprocess → encode → splice. Multi-image support.
- Smoke tests in `Tests.Server`.

### (Deferred) Phase V6 — audio
E2B/E4B audio via the `a.*` encoder (USM/conformer). Separate epic; not required for image.

## 4. Risks & de-risking

- **`gemma4v` semantics are the long pole** (Phase V2): QK-norm, learned 2D positions, vision
  attention/mask and token reduction must match llama.cpp. Mitigation: stage-by-stage tensor
  parity against llama.cpp; CPU-first before GPU.
- **Remaining verification debt** (§1): derive the token-reduction and mask rules from llama.cpp
  before V2; metadata already fixes the geometry and normalization.
- **Bidirectional mask × SWA × KV-share** (V4) is a delicate interaction in `PagedKvCache`.
- **Fallback / de-risking option:** if MobileNet-V5 parity proves too costly, the **SigLIP ViT**
  path (Gemma 3 4B, or Gemma 4 26B/31B big models) is a *much* simpler encoder (a plain
  bidirectional ViT that reuses our existing attention/MLP kernels almost directly, 896×896, 14px
  patches, 4×4 pool → 256 tokens). Phases V0/V1/V3/V4/V5 are largely shared; only the encoder
  (V2) differs. A SigLIP PoC would validate the whole splice pipeline end-to-end fastest, then the
  MobileNet-V5 encoder slots in for E4B. (User has chosen E4B-direct; this remains the documented
  fallback if V2 stalls.)

## 5. References

- Issue #82 — Gemma 4 family support (text); vision marked out of scope.
- `docs/done/gemma4-e4b-implementation-plan.md` — text trunk plan (phasing now stale; trunk landed).
- HF docs: [Gemma 4 (transformers)](https://huggingface.co/docs/transformers/main/model_doc/gemma4),
  [Gemma 3n](https://huggingface.co/docs/transformers/main/model_doc/gemma3n),
  [Welcome Gemma 4 (blog)](https://huggingface.co/blog/gemma4).
- [timm changelog](https://huggingface.co/docs/timm/changes) — "MobileNetV5 backbone … for Gemma
  3n image encoder" (the conv-encoder confirmation).
- llama.cpp: [multimodal.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/multimodal.md),
  [mtmd README + `mtmd.h`](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd),
  [gguf-py `constants.py`](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py)
  (`clip.*` keys, projector types, `v.*`/`mm.*`/`a.*` tensor names).
- GGUFs: `ggml-org/gemma-4-E4B-it-GGUF`, `unsloth/gemma-4-E4B-it-GGUF` (text + mmproj).

## Handover state 2026-08-08 — encoder forward is blocked on a missing reference

The mmproj loader and preprocessor are complete and validated against the real file
(`Gemma4V_Open_ResolvesCompleteE4BViTInventory` asserts the full tensor inventory and shapes of
`gemma-4-E4B-it-mmproj.gguf`). The next step — encoder forward — is **blocked**, and the blockers
are specific.

### Confirmed geometry (from the real mmproj)

`gemma4v`, 16 blocks, embedding 768, 12 heads (64/head), FFN 3072, projection 2560, image 224 /
patch 16 → **196 patches**, mean [0,0,0], std [1,1,1], layer-norm eps 1e-06. Blocks are
Gemma-shaped: `ln1`, q/k/v/out with per-head `attn_q_norm`/`attn_k_norm` (64-wide), `attn_post_norm`,
`ln2`, gated FFN (`ffn_gate`/`ffn_up`/`ffn_down`), `ffn_post_norm` — i.e. sandwich norms, not
CLIP-standard pre-norm only.

### Blocker 1 — the position embedding layout is undocumented

`v.position_embd.weight` is `[768, 10240, 2]` Float32 — 15.7M values, 60 MB — while the fixed
224/16 grid needs only **196** positions. No metadata key explains 10240 or the trailing 2.

Measured from the data rather than assumed: range [-0.414, 0.431], mean -0.000336, rms 0.0264, and
position 0 is **not** (1, 0). **It is a learned table, not a precomputed cos/sin RoPE table.** That
rules out the natural first guess. What it does not establish is the indexing convention — whether
the trailing 2 is (row, col) factorised axes, and what 10240 bounds (variable-resolution support,
multi-crop, or packed training positions). Indexing a learned position table incorrectly produces
embeddings that are wrong but entirely plausible-looking, which is the failure mode this repository
has repeatedly documented.

### Blocker 2 — no local reference implementation

`tools/llama.cpp` b8585 cannot serve as the reference: it rejects the text model outright with
`unknown model architecture: 'gemma4'`, so it implements neither the Gemma 4 decoder nor `gemma4v`.
There is therefore **no way on this machine to validate an encoder forward against a known-good
implementation**.

### Why the encoder should not be written yet

Both blockers point the same way. An encoder written now would run, produce plausible embeddings,
and be unverifiable — and two independent choices are already unpinned: the position-embedding
indexing above, and the preprocessor's `align_corners=True` bilinear resize (references commonly use
`align_corners=False`, and CLIP-family pipelines often use bicubic). Wrong resampling and wrong
position indexing both present as "the projector is broken".

### What unblocks it

1. A reference for `gemma4v`: the HuggingFace `google/gemma-4-E4B-it` implementation, or a llama.cpp
   build that knows `gemma4`. Either settles the position-embedding convention and the resize mode.
2. With a reference in hand, work bottom-up with parity at each stage — patch embedding, then
   position addition, then one block, then all 16, then the projection — rather than writing the
   whole encoder and comparing only final embeddings. A single end-to-end comparison cannot localise
   which of five stages is wrong.

Until then the honest status is: **loader and preprocessing done and verified; encoder deliberately
not started for want of a reference**, not merely unfinished.
