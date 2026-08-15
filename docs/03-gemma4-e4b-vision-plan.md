> **Reprioritized 2026-08-15 — runway position 3.** Architecture is fully reverse-engineered from
> the real mmproj + local llama.cpp source, and Phase V2 (the ViT encoder) is now IMPLEMENTED and
> passes a real-file structural sanity check. **The "blocked on oracle" status below is stale** —
> the absence of a Gemma-4-capable oracle blocks NUMERICAL PARITY, not implementation; those are
> different things. Do not re-derive the architecture from tensor names or resurrect the historical
> MobileNet-V5/Gemma-3n material further down this document — see the current contract immediately
> below instead.

# Gemma 4 E4B Multimodal (Vision) — Research & Implementation Plan

## CURRENT IMPLEMENTATION CONTRACT — 2026-08-15

Everything below is verified against the real `models/gemma-4-E4B-it-mmproj.gguf` (metadata +
tensor inventory) and the real reference graph (`examples/llama.cpp/llama.cpp`'s
`tools/mtmd/models/gemma4v.cpp` + the shared `clip_graph::build_vit`/`build_attn`/`build_ffn`/
`build_mm` helpers in `tools/mtmd/clip.cpp`), not inferred. Implemented in
`src/OpenTail.Stingray.Vision/Gemma4VVisionEncoder.cs` (Phase V2).

```
input:              224x224 RGB, preprocessed to [0,1] by Gemma4VImagePreprocessor
                     (encoder itself applies the range fix: x' = 2x - 1)
patch:               16x16 conv, stride 16, NO bias -> 14x14 = 196 patches, embedding 768
position:            learned lookup, two stacked [768,10240] tables (x-table then y-table on the
                     trailing axis) -- NOT a square grid, NOT RoPE, NOT interpolated
blocks:              16, each sandwich-normed (RMSNorm before AND after both sublayers):
                       RMSNorm(ln1)
                       -> separate Q/K/V projections, EACH a clamped linear (see below)
                       -> per-head RMSNorm on Q (weighted, attn_q_norm)
                       -> per-head RMSNorm on K (weighted, attn_k_norm)
                       -> 2D RoPE on Q and K only (NEOX, theta=100 -- gemma4v-specific constant,
                          NOT the paired text model's theta; leading half of each 64-wide head
                          rotated using the patch's COLUMN as position, trailing half using its ROW)
                       -> per-head RMSNorm on V (UNWEIGHTED, gemma4v-specific -- lives in the
                          shared build_vit helper gated on projector type, invisible in
                          gemma4v.cpp alone)
                       -> attention, scale = 1.0 (UNSCALED, not the usual 1/sqrt(head_dim))
                       -> output projection (clamped)
                       -> RMSNorm(attn_post_norm)
                       -> residual add
                       -> RMSNorm(ln2)
                       -> gated FFN: gate = QuickGelu(clamped ffn_gate(x)), up = clamped ffn_up(x),
                          ffn_out = clamped ffn_down(gate * up)
                       -> RMSNorm(ffn_post_norm)
                       -> residual add
heads:               12, head_dim = 64 (32+32 split for the 2D RoPE halves)
clamp:               ALL SEVEN per-block linear weights (attn_q/k/v/out, ffn_gate/up/down) --
                     confirmed via real per-tensor <name>.input_min/.input_max/.output_min/
                     .output_max scalar tensors in the mmproj, for every block. NOT the final
                     projection (mm.input_projection has no clamp tensors). Contract per linear:
                       x' = clamp(x, input_min, input_max)
                       y  = clamp(W @ x', output_min, output_max)
FFN activation:      Quick GELU: gelu_quick(x) = x * sigmoid(1.702*x) -- NOT the tanh-approximation
                     GELU used elsewhere in this codebase (mmproj declares neither use_gelu nor
                     use_silu, so clip.cpp's FFN_GELU_QUICK default applies)
post-blocks:         no final post-layernorm (no v.post_ln tensor for this projector)
token reduction:     3x3 average pool, stride 3, no padding (n_merge=3 -- NOT 4, a corrected
                     earlier misreading; confirmed absent from mmproj metadata so the hardcoded
                     default applies) -> 14x14 pools to 4x4 = 16 tokens, silently dropping the
                     last 2 patches per axis (14 is not a multiple of 3)
post-pool:           scale by sqrt(768)
projection:          unweighted RMSNorm (embedding_pre_projection_norm) -> linear 768->2560
                     (NOT clamped)
output:              16 soft-token embeddings, 2560-wide each
std_bias/std_scale:  absent from the real E4B mmproj (that branch is inert here)
audio (gemma4a):     deferred, out of scope for this contract
decoder-side splice/mask (Phase V4): NOT covered by this contract -- see the explicit note in
                     Phase V4 below. Do not assume Gemma-3's bidirectional-within-image mask
                     transfers without checking the actual Gemma-4 multimodal runtime; the vision
                     encoder above is unconditionally bidirectional, which is a separate question
                     from how the TEXT decoder should attend to the 16 spliced image positions.
```

## ADDENDUM 2026-08-15 — Gemma 3 SigLIP encoder also implemented (`clip.projector_type=gemma3`)

Separate architecture from the E4B `gemma4v` contract above — a DIFFERENT, simpler ViT family
(`llama.cpp`'s `clip_graph_siglip`, `tools/mtmd/models/siglip.cpp`'s `PROJECTOR_TYPE_GEMMA3`
branch), paired with the Gemma 3 4B text model rather than E4B. Both `gemma3` and `gemma4` text
architectures are already admitted in `ModelCompatibility.cs`, so this path — once Phase V4
(splice/mask) exists — could reach genuine end-to-end multimodal inference sooner than the E4B
path, which still needs V4 regardless. Implemented in `Gemma3VisionModel.cs` (loader) and
`Gemma3VisionEncoder.cs` (encoder), verified against the real
`models/mmproj-gemma-3-4b-it-f16.gguf` (851 MB, downloaded from `ggml-org/gemma-3-4b-it-GGUF`) and
its paired `models/gemma-3-4b-it-Q4_K_M.gguf` text model (2.49 GB).

**Genuinely simpler than `gemma4v`**: no 2D RoPE, no per-head QK-norm, no V-norm, no per-block QAT
clamp, a single learned position table `[1152,4096]` added once (not two stacked x/y tables looked
up per patch — confirmed one-to-one via `list-tensors`: 4096 = 64×64 patches exactly, added by
straight `ggml_add`, no `ggml_get_rows` lookup at all), a plain (non-gated) FFN with ordinary
tanh-GELU (`clip.use_gelu=true` in the real metadata — NOT quick-GELU, unlike `gemma4v`'s
metadata-absent default), the standard `1/sqrt(head_dim)` attention scale (no override), and a
REAL post-layernorm (`v.post_ln` exists here, unlike `gemma4v`). Confirmed geometry: 896×896
input, 14px patches → 4096 patches, embedding 1152, 16 heads (head_dim 72), FFN width 4304, 27
blocks, `n_merge=4` (default for this projector type — the plan's *original* draft here
misattributed the adjacent `PROJECTOR_TYPE_GEMMA3`-in-`clip.cpp`'s literal default to `gemma4v`;
this addendum's `n_merge=4` is for the actual `gemma3` SigLIP path and is correct as written)
→ 16×16 = 256 soft tokens (896/14=64, exactly divisible by 4, unlike `gemma4v`'s non-divisible
14/3 — no dropped patches here).

**Two new, genuinely surprising findings, both confirmed via direct verification against the real
file rather than guessed — the second one cost real debugging time and is worth remembering:**

1. **Different metadata key convention.** This mmproj declares `clip.projector_type` (no
   `.vision.` segment), NOT `clip.vision.projector_type` like `gemma4v`/`gemma4uv` use. Confirmed
   via `list-metadata`; the two conventions genuinely differ between exports and must not be
   assumed interchangeable.
2. **`ffn_up`/`ffn_down` tensor NAMES are swapped relative to their FUNCTIONAL role in this
   specific checkpoint export** — NOT a storage-transpose issue (unlike `mm.input_projection`,
   which genuinely IS stored transposed, matching `siglip.cpp`'s own explicit
   `ggml_cont(ggml_transpose(...))` before using it). Proven unambiguously via bias length (a bias
   vector's length is exactly its projection's output width — no axis-order ambiguity possible,
   unlike a weight matrix): the GGUF tensor named `ffn_up` has an `embeddingLength`-wide bias
   (1152) — it is actually the SECOND/reducing step (ffLen→embd) — and the tensor named `ffn_down`
   has a `feedForwardLength`-wide bias (4304) — it is actually the FIRST/expanding step
   (embd→ffLen). The reference C++ code works correctly regardless, because `build_ffn` never
   trusts the tensor's name — it just uses `layer.ff_up_w` as "whatever runs first" and
   `layer.ff_down_w` as "whatever runs second," and each tensor's OWN shape (not its name) is
   self-consistent with its actual position in the graph. `Gemma3VisionModel`'s loader now binds
   the GGUF tensors by FUNCTION (swapped) rather than by name, with both the swap and the
   evidence documented inline — do not "fix" this back to name-matching without re-reading that
   comment.

**Performance note (new, not present in the `gemma4v` contract):** at 4096 patches, unscaled
per-head attention is ~1 trillion MACs per `Forward()` call (~21× `gemma4v`'s 196-patch cost) —
measured single-threaded at ~75s/block (~34 min total for 27 blocks), impractically slow to
re-run routinely. `Gemma3VisionEncoder` parallelizes both the attention loop (across the 16 heads)
and the per-patch QKV/output-projection/FFN loops (across the 4096 patches) via `Parallel.For`,
each task given its own local scratch instead of sharing one buffer, which would otherwise race.
Net measured result: 562.5s (~9.4 min) end to end on this machine's 12 logical processors — a real
but modest win over the attention-only-parallelized version's 604.9s, well short of the naive
per-core-count expectation (per-task scheduling overhead across 4096 small `Parallel.For` work
items eats into the gain). Not pursued further — diminishing returns for a structural sanity check
that only needs to run occasionally, not routinely. `gemma4v`'s 196-patch encoder was never
parallelized and doesn't need to be (sub-second).

**Status: CONFIRMED PASSING, 2026-08-15.** Structurally implemented, passes the loader's strict
validation, and the end-to-end structural sanity test (`Gemma3VisionEncoderTests.cs` — shape, no
NaN/Inf, sane magnitude) is green: 1/1, 0 failures, 562.5s (~9.4 min, fully parallelized — see the
performance note above) against the real 851 MB mmproj. Further speedup was not pursued past this
point (diminishing returns, operator decision to stop). **Like `gemma4v`, still NOT numerically
parity-verified** — no Gemma-4-capable oracle exists locally (llama.cpp's own architecture is
`gemma4`; the paired `gemma3` architecture is a DIFFERENT, older, and — per `ModelCompatibility.cs`
— already-admitted text model, but this session did not attempt to build or acquire a llama.cpp
binary that supports `gemma3` multimodal inference specifically to use as an oracle).

Ignore all historical MobileNet-V5/Gemma-3n/768²/256-token/`<start_of_image>`-marker assumptions
in the sections below — they predate the real mmproj verification and are retained only as
research archaeology, not as an implementation contract.

Status: **V0 (mmproj loader), V1 (fixed-grid preprocessing), V2 (ViT encoder), and V3 (token
reduction + projector) implemented** — `Gemma4VVisionEncoder.Forward` runs blocks, pooling, and
the final projection as one call, so V2/V3 shipped together rather than as separate phases. Passes
a real-file structural sanity check (`Gemma4VVisionEncoderTests.cs`: correct shape, no NaN/Inf,
non-degenerate, sane magnitude) but is **NOT numerically parity-verified** — no Gemma-4-capable
oracle exists on this machine to compare against yet. Embedding splice into the text decoder,
image-token mask semantics, and CLI/API surface remain open (Phases V4-V5). Tracked by
**issue #126**.

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

### Phase V2 — `gemma4v` ViT encoder forward pass — **IMPLEMENTED 2026-08-15, NOT parity-verified**
`Gemma4VVisionEncoder.cs`. The verified 16-block, 768-wide, 12-head transformer encoder (real
per-head QK-norm, gemma4v-only V-norm, 2D RoPE theta=100, unscaled attention, per-block clamped
linears, quick-GELU FFN — see the contract at the top of this doc), preceded by its 16px
convolutional patch embedding and learned 2D position table. A CPU reference, as planned — no GPU
path yet. Passes a real-mmproj structural sanity test (shape, no NaN/Inf, non-degenerate, sane
magnitude — `Gemma4VVisionEncoderTests.cs`), but **stage-by-stage parity against llama.cpp's
`clip` path has NOT been done** — no working Gemma-4-capable oracle exists locally yet (see the
handover-state section below). Do not treat this as numerically verified.

### Phase V3 — token reduction + projector MLP — **IMPLEMENTED 2026-08-15, shipped inside V2**
`Gemma4VVisionEncoder.Forward` runs the 3×3 average pool, `sqrt(768)` scale, unweighted RMSNorm,
and `mm.input_projection` (768→2560) as the tail of the same call that runs the 16 blocks, rather
than as a separately invoked phase. Same parity caveat as V2 — not yet checked against
`mtmd_get_output_embd` or any other oracle output.

### Phase V4 — embedding splice + bidirectional mask (HIGH risk) — **NOT STARTED, do not begin
without re-reading this note**
- Add `ForwardPass`/`Prefill` support to accept **precomputed input embeddings** at given
  positions (overload of `EmbedTokenInto`; skip `token_embd` lookup; decide embedding-scale
  handling for image rows — text tokens get `× sqrt(2560)`, image embeddings come pre-scaled from
  the projector, *confirm*).
- **The "causal-except-bidirectional-within-image" mask below is NOT yet confirmed for Gemma 4 —
  do not assume it transfers from Gemma 3.** V2's ViT encoder is unconditionally bidirectional
  internally (that part IS settled — attention over image patches, before splicing), but how the
  TEXT DECODER should attend to the 16 spliced image positions once they're embedded in the token
  sequence is a genuinely separate, independent question this session has not investigated. Gemma
  4's PLE, SWA, cross-layer KV-share, dual-RoPE, and paged KV cache are all real interactions a
  wrong assumption here could break silently. **Confirm the actual Gemma-4 multimodal runtime's
  decoder-side mask semantics — via the real llama.cpp mtmd graph construction code (the same
  standard this whole doc has held V2 to) or a working oracle — before writing or modifying any
  `PagedKvCache`/global-causal-masking code for this.**
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


## CORRECTION 2026-08-08 — the reference DOES exist, and it answers both blockers

The previous section concluded there was no local reference for `gemma4v`. **That was wrong.** It
was true of the b8585 binary in `tools/llama.cpp`, and I generalised from that without checking
`examples/`. There is a full llama.cpp source checkout at
`examples/llama.cpp/llama.cpp` (HEAD `3653e6d6d`) which knows the `gemma4` architecture and
implements the projector in `tools/mtmd/models/gemma4v.cpp`. That file is the authority for
everything below.

### Blocker 1 resolved — the position embedding is two stacked lookup tables

`[768, 10240, 2]` is **not** a square grid and **not** RoPE. It is one table for x and one for y,
stacked on the trailing axis, each `[n_embd=768, 10240]`:

```c
tbl_x = view_2d(pos_embd, n_embd, pos_size, nb1, 0);
tbl_y = view_2d(pos_embd, n_embd, pos_size, nb1, pos_size * nb1);
inp = inp + get_rows(tbl_x, pos_x) + get_rows(tbl_y, pos_y);
```

Each patch adds its column embedding **and** its row embedding. 10240 is the per-axis maximum; a
224/16 image uses only the first 14 of each. Note this is a *different* scheme from the
`resize_position_embeddings` path used by siglip2-naflex in the same file, which interpolates a
square grid — applying that one here would be wrong, and `sqrt(10240)` not being an integer is the
tell.

### Encoder details the loader does not yet capture

- **Input range.** The graph applies `inp_raw * 2 - 1` before the patch convolution. With E4B's
  declared mean=[0,0,0], std=[1,1,1], `Gemma4VImagePreprocessor` emits **[0,1]**, and the encoder
  expects **[-1,1]**. The step legitimately lives in the graph rather than the preprocessor, but
  anyone wiring preprocessor output straight into an encoder will be wrong by a factor of two and an
  offset — silently, and in a way that looks like a weight problem.
- **No patch bias**, and the patch embedding is a `conv_2d` with stride = patch_size.
- **2D RoPE with neox ordering** is applied inside attention *in addition to* the learned position
  embeddings. Both are needed; neither substitutes for the other.
- **Token reduction is average pooling**, kernel `n_merge`, then a scale by `sqrt(n_embd)`.
  **Correction (2026-08-15) — the "defaults to 4 / 9 tokens" claim below was wrong.** `n_merge`/
  `rope_theta` are set by the case block for `PROJECTOR_TYPE_GEMMA4V` specifically in `clip.cpp`
  (`load_hparams`, ~line 1522), NOT the `PROJECTOR_TYPE_GEMMA3` block one case above it (where the
  wrong "defaults to 4" reading came from — easy to misattribute since they're adjacent in the same
  switch): `hparams.rope_theta = 100.0f; hparams.n_merge = 3;` — then optionally overridden by the
  `clip.vision.projector.scale_factor` metadata key (`KEY_PROJ_SCALE_FACTOR`) if present.
  **`rope_theta = 100.0f` is load-bearing and had no other source in this plan** — the 2D-RoPE
  frequency base for the ViT is NOT the text model's `rope_theta` and must be hardcoded to 100.0 for
  `gemma4v` unless mmproj metadata overrides it (verify the metadata key against the real file
  before trusting the hardcoded default). No `n_merge∈{2,4}` assertion applies to this projector
  type (that assertion belongs to a different case in the same switch). With the corrected default
  `n_merge=3`: a 14×14 patch grid pools with kernel=stride=3, no padding →
  `floor((14-3)/3)+1 = 4` per side → **4×4 = 16 soft tokens**, not 9 — and since 14 isn't a multiple
  of 3, the pool only covers the first 12 patches per axis (`4*3`), silently dropping patches 12-13
  (the last two rows/columns) per side, same as any non-overlapping pool over a non-divisible grid.
  Verify the real mmproj's `clip.vision.projector.scale_factor` key (present or absent) before
  implementing, since its presence would override 3 entirely.
- **Projection** is RMS norm (`embedding_pre_projection_norm`) followed by a *clippable* linear:
  `build_mm` consults a per-tensor clamp map and, when present, clamps input and output around the
  matmul.
- Optional `std_bias` / `std_scale` tensors are applied before projection when present. Verified
  absent from this mmproj, so that branch is inert here — but a differently exported projector could
  carry them, and the loader would silently ignore them.
- **Attention is unscaled.** `gemma4v.cpp` sets `kq_scale = 1.0f` explicitly, before calling the
  shared `build_vit()`. This is NOT the standard `1/sqrt(head_dim)` every other attention path in
  this codebase (and most generic attention kernels) defaults to — a reused kernel that doesn't
  accept an explicit override, or that silently falls back to the usual scale, will produce
  plausible-looking but wrong attention weights with no error.
- **V gets its own RMS-norm, gated on projector type — not visible in `gemma4v.cpp` at all.** The
  per-block ops (Q/K/V projection, QK-norm, 2D RoPE, attention, sandwich norms, FFN) live in a
  SHARED `clip_graph::build_vit()` (`tools/mtmd/clip.cpp:334`), called by every ViT-family model
  file including this one. Inside it, one branch is architecture-specific:
  `if (proj_type == PROJECTOR_TYPE_GEMMA4V) { Vcur = ggml_rms_norm(ctx0, Vcur, eps); }` — applied
  to V (not just Q/K) right before attention, after `add_pos` has already rotated Q/K. A ViT
  implementation written only from `gemma4v.cpp` (the model-specific file) would miss this entirely,
  since it's not called there — it's injected inside the shared helper based on the projector-type
  enum. Confirmed full per-block order from `build_vit` (clip.cpp:334-562): residual → `ln_1`
  (RMSNorm) → separate (not fused) Q/K/V projections → per-head QK-norm applied AFTER reshaping to
  `(d_head, n_head, n_pos, B)` (matches the mmproj's declared 64-wide per-head `attn_q_norm`/
  `attn_k_norm`) → 2D RoPE (neox, x/y split) on Q and K only → **V RMS-norm (gemma4v-only)** →
  unscaled attention → output projection → `attn_post_norm` (RMSNorm) → residual add → `ln_2`
  (RMSNorm) → gated FFN (GeGLU) → `ffn_post_norm` (RMSNorm) → residual add. This is the complete,
  load-bearing block structure for Phase V2 — every step is now sourced from the actual shared
  graph code, not inferred.

### Revised status

The encoder is no longer blocked on knowledge — it is blocked only on being written, and it can now
be built stage-by-stage against a real reference. The staged-parity recommendation stands and is now
actionable: patch conv → +pos(x,y) → one block with 2D RoPE → all 16 → pool/scale → norm+projection,
comparing at each stage rather than only at the end.

## CORRECTION 2026-08-15 — three more load-bearing findings, verified against the real local mmproj

Everything below was checked against the actual `models/gemma-4-E4B-it-mmproj.gguf` (991 MB,
present locally), not just read from source — `list-metadata`/`list-tensors` confirm every claim.

1. **Per-block linear layers carry a real INT8-range clamp, not just the final projection.**
   `clip_graph_gemma4v::build_mm` (the pasted-code function) checks a `clamp_info_map` for ANY
   weight tensor it's given — the plan previously only registered this for the final
   `mm.input_projection`. It is NOT special to that one tensor: `build_vit`'s attention block calls
   `build_mm(layer.q_w, ...)`, `build_mm(layer.k_w, ...)`, `build_mm(layer.v_w, ...)`, and
   `build_attn`'s output stage calls `build_mm(wo, ...)`; `build_ffn` calls `build_mm` for `up`,
   `gate`, AND `down`. **All seven per-block linear weights** (`attn_q/k/v/out`,
   `ffn_gate/up/down`) go through this path. Confirmed directly against the real file: EVERY one of
   those seven tensors in EVERY block has four accompanying scalar tensors —
   `<name>.input_min`/`.input_max`/`.output_min`/`.output_max` (e.g. `v.blk.0.attn_q.input_max`).
   The clamp is real and load-bearing: `x_clamped = clamp(x, input_min, input_max)`,
   `out = clamp(w @ x_clamped, output_min, output_max)`. Skipping it would silently produce wrong
   activations for every block, in a way that looks like a numerically-close-but-off encoder rather
   than a missing feature. `mm.input_projection` itself has NO such tensors (confirmed absent) —
   the clamp is per-block only, not on the final projector.
2. **FFN activation is quick-GELU, not plain GELU.** `hparams.ffn_op` defaults to `FFN_GELU_QUICK`
   (`clip.cpp` ~line 1321) unless the mmproj declares `use_gelu`/`use_silu` metadata keys — neither
   is present in the real file (confirmed via the full 37-key metadata dump). `FFN_GELU_QUICK`
   dispatches to `ggml_geglu_quick_split`, i.e. `gelu_quick(x) = x * sigmoid(1.702*x)` (the
   sigmoid/logistic approximation), NOT the tanh-approximation GELU
   (`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`) that `SimdKernels.GeluInPlace` already
   implements for other architectures in this codebase. Reusing `GeluInPlace` directly — the
   obvious, natural choice given it's the only GELU kernel already in the codebase — would have
   been silently wrong. A new quick-GELU kernel is needed.
3. **`n_merge=3`/`rope_theta=100.0` (the correction above) and no `scale_factor`/rope-override
   metadata key** are now directly confirmed absent from the real file, not just inferred from
   `clip.cpp` reading alone — both hardcoded defaults apply exactly as derived. `v.post_ln` is also
   confirmed absent (no post-layernorm step), matching the loader's tensor inventory.

Net: the architecture is now FULLY pinned down against the real file with no remaining unknowns
short of full numerical parity (which needs a working oracle to run end-to-end, still unavailable
locally) — but it took going two levels deeper than the model-specific `gemma4v.cpp` file alone
(into the shared `build_vit`/`build_attn`/`build_ffn`/`build_mm` helpers in `clip.cpp`) plus direct
verification against the real GGUF's tensor inventory and metadata to find the clamp mechanism and
activation variant. Neither would have been visible from `gemma4v.cpp` or the mmproj header summary
alone.
