# 087 — FLUX.2 real implementation plan (tensor inventory, ready to implement)

Follow-up to `docs/086-video-model-verification-plan.md` section 3: `Flux2DiT.cs` has zero
`IWeightLoader` wiring and `Flux2Pipeline.Generate` is a pure synthetic stub. This doc is the real
tensor inventory, checked directly against the downloaded checkpoint (`models/_models/
flux2-dev-Q4_K_S.gguf`, via `stingray list-tensors`) before any implementation — same discipline
`docs/055` used for LTX-Video. **Do not write a weight loader from `Flux2Params`'s current
defaults** — they are copied from FLUX.1 and are wrong on nearly every dimension (see table below).

## Real checkpoint stats

299 tensors, 17.97 GiB (Q4_K_S). **8 double-stream blocks, 48 single-stream blocks** (not FLUX.1's
19/38) — confirmed by counting `double_blocks.{0..7}`/`single_blocks.{0..47}` directly.

## Current `Flux2Params` defaults vs. real checkpoint (all wrong except AxesDim/QkvBias/GuidanceEmbed)

| Param | Current default | Real value | Evidence |
|---|---:|---:|---|
| `HiddenSize` | 3072 | **6144** | `img_attn.proj.weight [6144,6144]`, every block tensor |
| `NumHeads` | 24 | **48** | `HeadDim` confirmed 128 via `key_norm.scale [128]`; 6144/128=48 |
| `DepthDoubleBlocks` | 19 | **8** | direct count |
| `DepthSingleBlocks` | 38 | **48** | direct count |
| `ContextInDim` (text seq dim) | 4096 (T5) | **15360** | `txt_in.weight [15360, 6144]` — NOT T5's 4096; see note below |
| `VecInDim` (pooled CLIP) | 768 | **likely unused/0** | no `vector_in`/`y_in`-style tensor found anywhere in the checkpoint — FLUX.2 appears to have dropped CLIP pooled conditioning entirely (consistent with its README: Mistral-Small-24B is the only text encoder, no CLIP mentioned) |
| `AxesDim` | [16,56,56] | unconfirmed, plausible | not yet cross-checked against real RoPE source — same headDim=128 as FLUX.1 makes this a reasonable prior, not a verified fact |
| `Theta` | 10000.0 | unconfirmed | same caveat |
| `QkvBias` | true | **false** | no `.bias` tensor exists alongside any `.qkv.weight`/`.proj.weight` in the listing — FLUX.2's linears appear to be bias-free, unlike FLUX.1 |
| `GuidanceEmbed` | true | **true, confirmed** | `guidance_in.in_layer.weight`/`guidance_in.out_layer.weight` both present |

**`ContextInDim=15360` note**: this is almost certainly NOT a raw single-layer LLM hidden-state
width. `15360 = 3 * 5120` — a plausible concatenation of 3 of Mistral-Small-24B's own hidden-layer
activations (a known technique for extracting richer conditioning from a decoder LLM than any
single layer gives, used by some newer DiT text-conditioning schemes). **Do not assume this without
checking**: read `black-forest-labs`' own FLUX.2 reference code/paper (or the `diffusers` FLUX.2
pipeline if one has landed in the vendored `examples/diffusers` copy) for the real hidden-state
extraction recipe (which layer(s), concatenated how) before writing the text-encoder integration —
guessing this wrong would silently produce wrong-shaped-but-plausible conditioning, the worst kind
of bug to debug later.

## Real, distinctive architectural finding: SHARED (not per-block) modulation weights

Unlike FLUX.1 (and HunyuanVideo, and every other per-block-AdaLN DiT in this codebase), FLUX.2 has
exactly **one** `double_stream_modulation_img.lin.weight` / `double_stream_modulation_txt.lin.weight`
tensor each (`[6144, 36864]`, 36864 = 6144×6 — the usual 6-chunk shift/scale/gate ×2 AdaLN-Zero
split), shared across all 8 double blocks — not one set of modulation weights per block. Likewise
exactly one `single_stream_modulation.lin.weight` (`[6144, 18432]`, 18432 = 6144×3) shared across
all 48 single blocks. **This is a real simplification versus FLUX.1's per-block learned
modulation, not a checkpoint-loading quirk** (confirmed: no `double_blocks.N.img_mod.*`-style
per-block tensor exists anywhere in the listing).

**Open question, not resolved this pass**: how does a single shared modulation MLP produce
different actual shift/scale/gate values per block, if not from a per-block weight? Two plausible
mechanisms, not yet distinguished: (a) the modulation input itself is block-index-conditioned
(e.g. a learned per-block embedding is added to the timestep/guidance embedding before the shared
linear), or (b) blocks genuinely share identical modulation *values* derived only from
timestep+guidance, and per-block differentiation comes entirely from each block's own
attention/MLP weights. **Check the real FLUX.2 reference source for this before implementing** —
guessing wrong here would silently produce a structurally-plausible but numerically-wrong DiT.

## Full non-block tensor list (all BFloat16, all top-level — no block prefix)

```
img_in.weight                            [128, 6144]     -- Linear(128->6144): patchify projection, 128 = 2x2 patch x 32 latent channels (2x FLUX.1's 16-channel VAE assumption -- VERIFY against flux2-vae.safetensors's own latent_channels before assuming 32)
txt_in.weight                            [15360, 6144]   -- Linear(15360->6144): text conditioning projection (see ContextInDim note above)
time_in.in_layer.weight                  [256, 6144]     -- sinusoidal-256 -> 6144, timestep embed MLP layer 1
time_in.out_layer.weight                 [6144, 6144]    -- timestep embed MLP layer 2
guidance_in.in_layer.weight              [256, 6144]     -- guidance embed MLP layer 1 (CFG-distilled, same pattern as FLUX.1-dev)
guidance_in.out_layer.weight             [6144, 6144]    -- guidance embed MLP layer 2
double_stream_modulation_img.lin.weight  [6144, 36864]   -- SHARED across all 8 double blocks (see finding above)
double_stream_modulation_txt.lin.weight  [6144, 36864]   -- SHARED across all 8 double blocks
single_stream_modulation.lin.weight      [6144, 18432]   -- SHARED across all 48 single blocks
final_layer.adaLN_modulation.1.weight    [6144, 12288]   -- 12288 = 6144x2 (final shift/scale, standard)
final_layer.linear.weight                [6144, 128]     -- Linear(6144->128): final projection back to patch space
```

No `.bias` tensor exists for ANY of the above or any block-level linear (`img_in`, `txt_in`,
`time_in`/`guidance_in` MLPs, every block's `qkv`/`proj`/`mlp` linear, `final_layer.linear`) —
consistent with the `QkvBias: false` correction above; treat this as a global bias-free convention
for this architecture, not block-specific.

## Per-double-block tensors (identical shape across all 8, `double_blocks.{0..7}.*`)

```
img_attn.norm.key_norm.scale     [128]              -- per-head-dim RMSNorm gain (headDim=128, confirms NumHeads=48)
img_attn.norm.query_norm.scale   [128]              -- Q-RMSNorm present (checked -- do NOT repeat HunyuanVideo's missing-Q-norm bug, docs/086)
img_attn.qkv.weight              [6144, 18432]      -- Linear(6144->18432=6144x3): fused QKV
img_attn.proj.weight             [6144, 6144]       -- attention output projection
img_mlp.0.weight                 [6144, 36864]       -- FFN up (36864=6144x6, a 6x expansion -- NOT FLUX.1's 4x; verify MlpRatio=6.0 not 4.0 before implementing)
img_mlp.2.weight                 [18432, 6144]       -- FFN down -- NOTE: input dim 18432, not 36864; see gated-FFN note below
txt_attn.* / txt_mlp.*           (identical shapes, txt stream)
```

**FFN shape anomaly, not yet resolved**: `img_mlp.0.weight` is `[6144, 36864]` (up-projection,
6x expansion) but `img_mlp.2.weight` is `[18432, 6144]` (down-projection FROM 18432, not 36864).
18432 = 36864/2 exactly. This is the classic signature of a **gated FFN** (SwiGLU-style: up-proj
produces `2x` the hidden width, split into gate+value halves, elementwise-gated down to hidden
width before the down-projection) — NOT a plain GELU MLP like FLUX.1's. **Verify this against the
real FLUX.2 reference before implementing** — if confirmed, `FeedForward` needs a SiLU/GELU-gated
split, not a direct 2-layer MLP like every other block in this codebase's DiT ports assumes by
default.

## Per-single-block tensors (identical shape across all 48, `single_blocks.{0..47}.*`)

```
norm.key_norm.scale      [128]
norm.query_norm.scale    [128]
linear1.weight            [6144, 55296]   -- fused QKV+MLP-up, single big linear (FLUX.1-style single-block fusion) -- 55296 = 18432(qkv) + 36864(mlp-up) -- MATCHES the gated-FFN 6x-expansion hypothesis above (18432 qkv + 36864 = 6144x6 mlp portion)
linear2.weight            [24576, 6144]   -- fused output projection -- 24576 = 6144(attn-out) + 18432(mlp-down input) -- again consistent with an 18432-wide (not 36864) post-gate MLP hidden state
```

## Open questions RESOLVED AND IN-REPO VERIFIED (2026-09-18)

The three open questions above were first answered via an external report (ChatGPT), then the user
added the **real BFL FLUX.2 source** to this repo (`examples/flux2/`, plus `examples/diffusers`
gaining `src/diffusers/.../transformer_flux2.py`/`pipelines/flux2/pipeline_flux2.py`). **Every
single claim below has now been independently confirmed by direct grep against that real vendored
source** (`examples/flux2/src/flux2/model.py`, `text_encoder.py`, `system_messages.py`) — this is
no longer an externally-sourced lead, it is a verified fact per CLAUDE.md rule 8. Shape
corroboration that held even before this verification: `36864 = 2×18432` (gated-FFN split),
`55296 = 18432(qkv) + 36864(mlp-up)`, `24576 = 6144(attn-out) + 18432(mlp-down-in)`,
`15360 = 3×5120` (Mistral-Small-24B hidden size).

1. **Shared modulation — genuinely shared, no per-block differentiation via AdaLN at all.**
   `vec = time_in(timestep_emb) + guidance_in(guidance_emb)` (both 256→6144 MLPs) is computed ONCE
   per denoising step, then the three shared `Modulation` modules
   (`double_stream_modulation_img`/`_txt`, `single_stream_modulation`) are each called ONCE on that
   same `vec` to produce ONE set of shift/scale/gate values used identically by every block of that
   type. `Modulation.forward`: `SiLU(vec) -> Linear(dim, multiplier*dim) -> chunk(multiplier)`
   (multiplier=6 for double giving two (shift,scale,gate) triples for img/txt; multiplier=3 for
   single giving one triple). No per-block embedding, no block-index conditioning of any kind —
   blocks differentiate purely via their own attention/MLP weights, not via modulation.
2. **Gated FFN — confirmed SiLU-gated (SwiGLU-style), `mlp_hidden_dim = HiddenSize * 3 = 18432`.**
   `Linear(6144->36864, bias=false) -> chunk(2) -> [u1,u2] -> SiLU(u1)*u2 -> Linear(18432->6144,
   bias=false)`. Same structure for both `img_mlp` and `txt_mlp`. Single-stream blocks fuse QKV +
   this same gated-MLP-up into one `linear1: Linear(6144 -> 3*6144 + 2*18432 = 55296, bias=false)`,
   split into `qkv (18432)` and `mlp (36864)`, then `linear2: Linear(6144+18432=24576 -> 6144,
   bias=false)` on `concat(attn_out, SiLU(mlp)*mlp)`. This is an actual algorithmic simplification
   worth implementing as one fused op in C#, not three independent Q/K/V projections plus a
   separate conventional FFN, if aiming for reference-equivalent execution.
3. **`ContextInDim=15360` — 3 concatenated Mistral-Small-3.2-24B hidden-STATE SEQUENCES (not
   pooled), feature-dim-concatenated, from layers (10, 20, 30) of 40 (0-indexed into
   `output.hidden_states`).** Per token position `i`: `ctx[i] = hidden[10][i] || hidden[20][i] ||
   hidden[30][i]` (each 5120-wide, concatenated to 15360), preserving the full sequence length —
   NOT averaged/pooled/CLS. Reported default extraction layers `(10, 20, 30)`; **this specific
   triple, and the "which HF hidden_states index = which of Mistral's 40 layers" off-by-one
   convention, is exactly the kind of numeric fact that needs an in-repo cross-check (or a real
   golden-parity diff against actual Mistral layer outputs) before trusting for implementation** —
   do not substitute 9/19/29 or 11/21/31 without checking.
   FLUX.2 prepends a fixed BFL system message before the user prompt via Mistral's chat template,
   and pads/truncates the conditioning sequence to a fixed length — same discipline as FLUX.1's own
   T5-padding bug fixed in `docs/056` Round 9.

## 2026-09-18 UPDATE: real BFL FLUX.2 source now vendored (`examples/flux2/`) — every claim above IN-REPO CONFIRMED, plus 2 new real findings

The user added the real BFL FLUX.2 repo to this project (`examples/flux2/src/flux2/`), and
`examples/diffusers` separately gained a real FLUX.2 pipeline (`src/diffusers/pipelines/flux2/
pipeline_flux2.py`, `transformer_flux2.py`). Direct grep against `examples/flux2/src/flux2/
model.py`/`text_encoder.py`/`system_messages.py` confirms **every one of the three external-report
answers above exactly, verbatim**:
- Shared modulation: `model.py`'s `Flux2` class creates exactly one `double_stream_modulation_img`/
  `_txt`/`single_stream_modulation`, computes `vec` once, and passes the same modulation output
  into every block in a `for block in self.double_blocks:`/`self.single_blocks:` loop — no
  per-block modulation anywhere.
- Gated FFN: `mlp_ratio: float = 3.0` (all three `Flux2Params`-equivalent dataclasses), real
  `SiLUActivation` class (`x1, x2 = x.chunk(2); return SiLU(x1) * x2`), single-block `linear1`/
  `linear2` fusion exactly as reported.
- Text conditioning: `OUTPUT_LAYERS_MISTRAL = [10, 20, 30]` (0-indexed into `hidden_states`),
  `MAX_LENGTH = 512`, and `SYSTEM_MESSAGE` is the exact verbatim string reported. `Flux2Params.cs`
  now cites these directly instead of "reported."

`Flux2Params.cs` has been updated with the corrected values from the table above (HiddenSize 6144,
NumHeads 48, DepthDoubleBlocks 8, DepthSingleBlocks 48, ContextInDim 15360, MlpRatio 3.0,
QkvBias false, InChannels/OutChannels 128, VecInDim 0/unused, AxesDim/Theta per finding 1 below) —
`Flux2ConformanceTests` re-verified passing after the change.

**Two new real findings the external report never addressed, found by reading `model.py` directly:**

1. **RoPE is 4-axis `[t,h,w,l]`, not FLUX.1's 3-axis, and uses a different theta.** Real
   `axes_dim: list[int] = [32, 32, 32, 32]` (sum=128=HeadDim, confirmed), `theta: int = 2000` (NOT
   FLUX.1's 10000). The extra `t` axis is a frame/time coordinate used to temporally offset
   multiple reference images relative to the target image (`sampling.py`'s `encode_image_refs`:
   `t_off = [scale + scale*t for t in range(len(refs))]`, `scale=10`) — for plain text-to-image
   with zero reference images, `t` is effectively constant/zero for every token. `Flux2Params.cs`
   corrected; `Flux2RoPE.cs` (still built for the OLD 3-axis/FLUX.1-style scheme) needs a real
   rewrite to 4 axes before any numeric work, not just a params tweak — do not reuse the existing
   `BuildContextFreqs` unmodified.
2. **Attention is NOT FLUX.1's plain joint bidirectional attention — reference-image tokens are
   attention-isolated.** `model.py`'s `causal_attn_fn` (used by every block instead of a plain
   joint-attention call): sequence layout is `[txt, ref, img]`; **txt and img tokens attend to
   everything** (txt+ref+img keys), but **reference-image tokens only self-attend** (only to other
   ref tokens, never to txt or img) — implemented as two separate `scaled_dot_product_attention`
   calls per block (one over `[txt,img]` queries against `[txt,ref,img]` keys, one over `[ref]`
   queries against `[ref]` keys only), concatenated back together. This is a genuinely new
   mechanism, not present in FLUX.1's DiT at all, and is currently completely unimplemented in
   `Flux2DiT.cs`. For plain text-to-image (no reference images, `nRefTotal=0` in the existing C#
   signature), this degenerates to ordinary joint `[txt,img]` attention — **but `Flux2DiT.Forward`
   already has a `refLatents`/`refPositions` parameter in its signature, so implement the real
   `causal_attn_fn` split from the start rather than hardcoding the degenerate no-ref case**, since
   multi-reference conditioning is one of FLUX.2's headline real features (the class doc already
   says "Supports multi-image conditioning").

## Checkpoint status confirmed, 2026-09-18: Mistral-Small-24B is ready, no new architecture work needed

`models/_models/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf` IS present (13.5GB, an earlier
session check this same day wrongly reported it missing due to a `find`-vs-symlink bug -- see
`docs/088`). Checked its real metadata directly (`stingray list-metadata`): `general.architecture
= llama`, `llama.block_count = 40`, `llama.embedding_length = 5120` -- confirms `docs/087`'s
prediction exactly (`OUTPUT_LAYERS_MISTRAL=[10,20,30]` out of 40 layers, `15360 = 3×5120`). Plain
`llama` architecture is already fully allowlisted and supported by this engine's generic forward
pass -- **no new architecture code is needed to run Mistral-Small-24B itself**, only a way to
extract and capture its hidden states at three specific INTERMEDIATE layers (10/20/30 of 40)
simultaneously, not just the final logits. **Checked 2026-09-18: `ForwardPass.ExtractHiddenStates`
(`src/OpenTail.Stingray.Engine/ForwardPass.cs:1096`) only ever captures the FINAL transformer
layer's hidden state, not arbitrary intermediate layers** -- this is NOT a drop-in reuse for
FLUX.2's recipe (unlike Qwen Image's, which needs exactly this final-layer behavior -- see
`docs/089`). A real, new capability is needed here: either a new
`ExtractIntermediateHiddenStates(layers)`-style method, or three separate full forward passes each
truncated at a different depth (wasteful -- shares the first 10/20 layers of compute three times),
or a single pass that snapshots the residual stream at the three target layers as it runs (the
efficient approach, matching how `PrefillCore` already threads an `outAllHiddenStates` sink through
its layer loop -- extending that same mechanism to snapshot at N caller-specified layers instead of
only after the last one is the natural, minimal-diff way to do this). This significantly de-risks
FLUX.2's text-conditioning step (checkpoint ready, architecture already supported) but the
multi-layer capture itself is real, un-built work, not just a wiring exercise.

## 2026-09-18 UPDATE: steps 1-2 DONE -- first real-weight forward pass passes

`Flux2RoPE.cs` rewritten for the real 4-axis scheme (step 1) and `Flux2DiT` now has real
`IWeightLoader` wiring (step 2) — separate per-stream QKV with per-head RMSNorm, RoPE-after-norm,
joint `[txt,img]` attention, SiLU-gated FFN, fused single-block `linear1`/`linear2`, and shared
(not per-block) modulation, all implemented exactly per the confirmed recipe above, reusing
`DiffusionOps`'s existing `Linear`/`RmsNorm`/`LayerNormNoAffine`/`ModulateRows`/`MultiHeadAttention`
primitives (added additively — the old structural-only constructor is unchanged, conformance tests
unaffected). **Real, memory-safety bug hit and fixed along the way**: an initial per-tensor weight
cache got the test process OS-killed for low memory (the exact same bug class already found for
Qwen Image this session) — fixed identically by removing the cache entirely (`GetWeight` reads
fresh from `IWeightLoader` every call, matching `QwenImageModel.GetWeight`'s proven pattern).
`Flux2RealWeightsTests.Flux2DiT_RealWeights_ForwardPassProducesFiniteOutput` now passes: real
`flux2-dev-Q4_K_S.gguf` (18GB), 68s, finite non-NaN output — **the first successful real-weight
FLUX.2 forward pass in this codebase's history.** Text conditioning in that smoke test is still
synthetic (random floats, not real Mistral output) — not a coherence check yet, matches the same
"structurally sound, not yet coherent" milestone HunyuanVideo/Qwen Image reached before their own
text-conditioning wiring landed.

## Recommended next steps (implementation, NOT done this pass)

3. Wire a real Mistral-Small-24B forward pass (the checkpoint is downloaded: `models/_models/
   Mistral-Small-3.2-24B-Instruct-2506-Q4_K_S.gguf`) for the text-conditioning path, extracting and
   concatenating hidden-layer activations at the confirmed layer indices, with the confirmed system
   prompt/chat-template/padding convention. Build a standalone differential test comparing this
   port's layer-10/20/30 hidden states against a real independent Mistral reference BEFORE wiring
   into FLUX.2 at all — a wrong 15360-dim conditioning vector will make a perfectly correct DiT
   port look completely broken, exactly the failure mode `docs/056` spent 9 rounds chasing on
   FLUX.1's T5 conditioning.
4. Port a real FLUX.2 VAE decoder (checkpoint downloaded: `models/_models/flux2-vae.safetensors`)
   replacing the current raw-channel-repeat stub. Note InChannels=128 (32 latent channels × 2×2
   patch) confirms FLUX.2's VAE has 32 latent channels, double FLUX.1's 16 — verify this against
   `flux2-vae.safetensors`'s own tensor shapes before assuming parity with FLUX.1's VAE structure.
5. Only then attempt a real end-to-end run — per `docs/086`, on Vulkan GPU explicitly, per the
   user's stated requirement.

This is real, substantial implementation work (steps 2-5 each comparable in scope to one of this
session's other single-model fixes) — scoped here so it can be picked up as a focused task rather
than re-derived cold.
