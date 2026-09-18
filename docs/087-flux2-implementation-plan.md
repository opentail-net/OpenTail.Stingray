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
simultaneously, not just the final logits. **CORRECTION, 2026-09-18: this capability already
exists and does not need to be built.** `IForwardPass.EnableHiddenTaps(ReadOnlySpan<int> layerIds)`
/ `HiddenTapDim` / `HiddenTapsAt(position)` (PR #413, `HiddenTapBuffer.cs`) is exactly this
mechanism -- position-indexed capture of arbitrary, caller-specified intermediate layers,
concatenated per position in the order given. `ForwardPass.SupportsHiddenTaps` is `true` for the
plain (CPU) path (`ForwardPass.PrefillCore.cs:944`, false only when SnapKV/TurboQuant are active --
neither applies here), and `GpuForwardPass.SupportsHiddenTaps` is unconditionally `true` too. The
call shape for FLUX.2's exact recipe needs one more real correction, found reading
`ForwardPassHiddenTapTests.cs`'s own doc comment precisely: **this codebase's tap convention is
"layer i's tap is that layer's OWN output, which is HF's `hidden_states[i+1]`"** (HF's
`hidden_states[0]` is the pre-layer embedding, so HF's `hidden_states[k]` = the output of this
codebase's layer `k-1`). FLUX.2's real recipe wants HF's literal `hidden_states[10, 20, 30]` — so
the correct call is **`forwardPass.EnableHiddenTaps([9, 19, 29])`**, NOT `[10, 20, 30]` (which
would silently grab HF's `hidden_states[11, 21, 31]` instead, exactly the off-by-one trap flagged
as a risk earlier in this doc, now concretely resolved rather than left as a warning). Full call
shape: `forwardPass.EnableHiddenTaps([9, 19, 29]); /* run prefill */ var ctx =
forwardPass.HiddenTapsAt(tokenPosition);` -- `HiddenTapDim` will be
`3 * 5120 = 15360`, matching `ContextInDim` exactly. My earlier claim that this needed new
infrastructure was wrong (found by actually reading `ForwardPass.PrefillCore.cs`'s tap-buffer
plumbing rather than assuming from `ExtractHiddenStates`'s final-layer-only behavior alone --
`ExtractHiddenStates` and `EnableHiddenTaps` are two separate, independent mechanisms on
`IForwardPass`, not the same one). This fully de-risks FLUX.2's text-conditioning step: the
checkpoint is ready, the architecture is already supported, and the exact capture mechanism the
real recipe needs already exists and is proven (used elsewhere for real hidden-state extraction).

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

## 2026-09-18 UPDATE: step 3 mechanism proven end-to-end against real weights

`Flux2MistralHiddenTapsTests` confirms the real extraction mechanism works: real Mistral-Small-
3.2-24B-Instruct-2506-Q4_K_S.gguf (12.61 GiB pre-faulted), `EnableHiddenTaps([9, 19, 29])` (the
off-by-one-corrected indices for HF's literal `hidden_states[10, 20, 30]`), real finite non-zero
15360-dim tap vectors at every token position. **Not yet done**: the real ChatML system-prompt/
chat-template/`drop_idx`-style cropping convention (`docs/087`'s original recipe, still needs
implementing in `Flux2Pipeline`'s conditioning path, not just the raw tokenize-and-tap mechanism
tested here), and a standalone differential test comparing this port's tap output against a real
independent Mistral reference BEFORE wiring into FLUX.2 end-to-end — a wrong 15360-dim conditioning
vector would make a perfectly correct DiT port look completely broken, exactly the failure mode
`docs/056` spent 9 rounds chasing on FLUX.1's T5 conditioning.

## 2026-09-18 UPDATE: step 3 fully DONE -- real text-conditioning wired and passing

`Flux2TextConditioning.cs` (new) implements the complete real recipe: real ChatML rendering via
this project's existing Jinja chat-template engine (`GgufTokenizer.ChatTemplate` ->
`JinjaChatTemplate.BuildMessages(prompt, systemContent: SystemMessage)`, `add_generation_prompt:
false`), confirmed no cropping applies (unlike Qwen Image's `drop_idx` — direct grep of
`text_encoder.py` found no `crop`/`drop_idx`/`start_idx` anywhere), then `EnableHiddenTaps
([9,19,29])` extraction concatenated per token. `Flux2TextConditioning_Encode_RealWeights_
ProducesFiniteConditioning` passes against the real Mistral-Small-3.2-24B checkpoint: finite
15360-dim-per-token output. **`Flux2.Encode` is not yet called anywhere inside `Flux2Pipeline`** —
wiring it into the full `Generate()` path is still a real next step, but the extraction mechanism
itself is now complete and proven, not just its individual pieces.

## Recommended next steps (implementation, NOT done this pass)

3. Wire `Flux2TextConditioning.Encode` into `Flux2Pipeline.Generate`'s text-conditioning path
   (currently still zero-filled/synthetic there) — then build the standalone differential test
   against a real independent Mistral reference before trusting the output for real image
   generation quality (the mechanism is proven correct-shaped and finite, not yet proven
   numerically correct against ground truth).
4. Port a real FLUX.2 VAE decoder (checkpoint downloaded: `models/_models/flux2-vae.safetensors`,
   336MB, 251 tensors) replacing the current raw-channel-repeat stub.

   **2026-09-18 real tensor inventory done, checked against `examples/flux2/src/flux2/
   autoencoder.py` directly.** Real findings:
   - Confirmed `z_channels=32` (matches `InChannels=128=32×2×2`), and the decoder block structure
     (`decoder.conv_in`, `decoder.mid_block.resnets.0/1`+`attentions.0`, `decoder.up_blocks.0..3`
     at 512/512/256/128 channels) is the **exact same real diffusers `AutoencoderKL` schema this
     codebase's general-purpose `VaeDecoder.cs`** (already used for FLUX.1/SD1.5/SDXL/SD3) already
     parses via its `UpBlockDiffusers`/`MidAttnDiffusers`/`ResBlock` helpers and `post_quant_conv`
     auto-detection — **the decoder blocks themselves are directly reusable, zero new code**, same
     shape as Qwen-Image's real VAE turning out to be Wan2.1's `WanVaeDecoder3D` verbatim.
   - **Real, NEW architectural difference found that blocks a naive reuse**: FLUX.2's real
     `AutoEncoder.decode()` does NOT use `VaeDecoder.cs`'s existing scalar
     `scale`/`shift` convention (`(1/0.3611, VaeShift=0.1159)` for FLUX.1) at all. It uses a
     **per-channel `BatchNorm2d(128, affine=False, track_running_stats=True)`** un-normalization
     (confirmed via the checkpoint's own real `bn.running_mean [128]`/`bn.running_var [128]`
     tensors) — `z = z * sqrt(running_var + 1e-4) + running_mean`, 128 independent per-channel
     values, not one scalar pair — **followed by a 2×2 pixel-UNshuffle rearrange**
     (`"(c pi pj) i j -> c (i pi) (j pj)"`, `pi=pj=2`) that converts the 128-channel latent back to
     32 channels at 2× the spatial resolution, BEFORE `post_quant_conv`/`decoder.conv_in` ever see
     it. `VaeDecoder.cs`'s existing `Decode(latent, h, w, scaleOverride, shiftOverride)` API cannot
     express this (it only supports one global scalar scale/shift pair) — **a real, small, new
     preprocessing step is needed** (apply the 128-value per-channel affine, then unshuffle) before
     calling into the existing decoder-block machinery, not a parameter change.
   - Real next step: add this per-channel-BN + 2×2-unshuffle preprocessing (either as a new
     `VaeDecoder` overload/hook, or a small standalone function in `Flux2/` that produces a
     32-channel raw latent `VaeDecoder.Decode` can then consume with `scaleOverride=1,
     shiftOverride=0` since the real normalization already happened) — genuinely new but small
     code, not a full VAE reimplementation.
5. **DONE, 2026-09-18: first real end-to-end run completed (CPU).** `Flux2Pipeline.Load(ditPath,
   mistralPath, vaePath)` wires all three real components together; `Generate()` now uses real
   `Flux2TextConditioning.Encode` and real `Flux2Vae.UnnormalizeAndUnshuffle` + `VaeDecoder` when
   real weights are loaded (structural-only constructor unchanged, zero regression to existing
   conformance tests). `Flux2Pipeline_RealWeights_EndToEndGenerateProducesFiniteImage` passes: all
   three checkpoints loaded together (~44GB combined working set, stable, no OOM), 123.9s for a
   64×64/2-step generation, finite output, real PNG written
   (`docs/diffusion-samples/flux2_first_e2e_smoke_2026-09-18.png`). Output is visual noise, the
   correct/expected result for a 2-step/64px wiring smoke test (not enough steps or resolution to
   converge -- Wan/LTX-Video both needed ~20 steps). **This is the first-ever complete FLUX.2 image
   generation in this codebase's history.**

   **Same-day follow-up: 128×128/4-step moderate quality check, still noise -- inconclusive, not
   evidence of a bug.** Ran a larger real check (300.8s CPU): still pure color noise, no structure
   emerging. Every other model in this codebase's history (FLUX.1, SD3.5, Wan, LTX-Video) needed
   roughly 20 steps to converge to coherent output, so 4 steps at this small a resolution genuinely
   may not be enough — this result does NOT confirm the real wiring is correct, but it also does
   NOT demonstrate a bug (unlike, say, a severe periodic checkerboard artifact, which WOULD be
   diagnostic at any step count). A real, more diagnostic next step per this doc's own earlier
   warning: the standalone numeric differential test against an independent Mistral reference
   (still not done) would isolate whether the text-conditioning path is correct BEFORE spending
   more CPU time on a full 20-step run — cheaper and more conclusive than just running longer blind.
   Still not done: a real full 20-step run (CPU, ~25+ min estimated at this scale) to see if
   structure emerges, the differential conditioning test, and **the real end-to-end run on Vulkan
   GPU explicitly** (the user's own stated requirement -- every run so far has been CPU-only) —
   `Flux2DiT`'s real-weight forward pass currently has no GPU-residency wiring at all (see
   `docs/088`'s Pass 2 section), so a Vulkan run needs that work done first.

   **Same-day, real 20-step run completed (64×64, 1055.6s CPU): STILL PURE NOISE, ZERO visible
   improvement over 2 or 4 steps.** This resolves the "inconclusive, too few steps" question from
   above with a real negative result: 20 steps is the exact step count every other model in this
   codebase (FLUX.1, SD3.5, Wan, LTX-Video) converges by, and FLUX.2 shows no structural change at
   all across 2/4/20 steps — the output at step 20 looks like the same character of random-colored-
   blob noise as step 2, not a partially-converged image. **This is now real evidence of an actual
   bug somewhere in the real DiT/text-conditioning/VAE chain, not just an under-stepped run.**
   Checked two of the most likely candidates directly against `examples/flux2/src/flux2/
   sampling.py`'s real `denoise()` before ruling them out:
   - **Euler integration sign**: real `img = img + (t_prev - t_curr) * pred` (timesteps descend
     1→0). This port's `Flux2Pipeline.Generate` loop computes `dt = nextT - t` with the same
     descending convention — signs match, confirmed NOT the cause (unlike the real sign-inversion
     bug that hit both FLUX.1 and, via the shared `EulerFlowScheduler` class, Z-Image-Turbo
     earlier this session's history — worth checking here specifically because of that precedent,
     but this port uses its own inline Euler step in `Flux2Pipeline.cs`, not the shared scheduler
     class, so that specific regression can't have propagated here).
   - **Guidance mechanism**: real FLUX.2-dev uses single-pass DISTILLED guidance (`guidance_emb`
     fed into `guidance_in` MLP, baked into `vec` before the DiT ever runs — confirmed in `model.py`'s
     plain `forward()`, matching `Flux2Params.GuidanceEmbed=true`), not the separate two-pass
     classifier-free-guidance path (`denoise_cfg`, a different real function for non-distilled
     use). This port's `ComputeModulationVec` already does the distilled-guidance path — confirmed
     NOT the cause.
   **Real, not-yet-checked candidates for the next investigation round** (in likely-cost-to-check
   order): (a) ~~shared-modulation chunk order~~ **RE-CHECKED 2026-09-18, CONFIRMED CORRECT**:
   `Modulation.forward`'s real `out.chunk(multiplier)` then `img_mod1_shift, img_mod1_scale,
   img_mod1_gate = img_mod1` confirms chunk order is literally shift/scale/gate (chunks 0/1/2 for
   mod1, 3/4/5 for mod2) — this port's `ComputeModulation`/`ApplyDoubleBlockReal`/
   `ApplySingleBlockReal` unpack in the identical order; (b) ~~gated-FFN split order~~
   **RE-CHECKED, CONFIRMED CORRECT**: real `u1, u2 = x.chunk(2); SiLU(u1)*u2` matches this port's
   `u1`=first half gated by SiLU, `u2`=second half, exactly; (c) ~~RoPE 4-axis assignment~~
   **RE-CHECKED, CONFIRMED CORRECT**: real position-id tensor's last axis is literally `[t,h,w,l]`
   in that order (`prc_img`'s `cartesian_prod(coords["t"], coords["h"], coords["w"], coords["l"])`),
   `EmbedND.forward` applies `axes_dim[i]` to `ids[...,i]` in that same order — this port's
   `Flux2Pipeline`'s position-array construction and `Flux2RoPE.BuildContextFreqs`'s per-axis loop
   both use the identical `[t,h,w,l]` index order; (d) ~~the VAE's per-channel BatchNorm eps-inside-sqrt formula and the pixel-unshuffle index
   mapping~~ **RE-CHECKED 2026-09-18 with a real numeric unit test, CONFIRMED CORRECT**:
   `Flux2VaeUnshuffleUnitTests` (identity-BatchNorm known-pattern test + a separate per-channel-
   affine test) confirms `"(c pi pj) i j -> c (i pi) (j pj)"` maps input channel `k=pi*2+pj` to
   output `(row=pi, col=pj)` exactly as intended, and the affine step applies real per-channel
   mean/var correctly — a genuine numeric check, not just re-reading the code; (e) a latent-space
   scale mismatch between what the DiT actually outputs and what the VAE's `post_quant_conv`/
   `conv_in` expect (FLUX.1's own fix history includes exactly this class of bug) — still not
   checked.
   **A real, separate bug WAS found and fixed the same day: `Flux2Pipeline.Generate` built a real
   `EulerFlowScheduler` but never actually called it, instead stepping a plain linear
   `t = 1 - step/steps` ramp -- FLUX.2 needs the real resolution/step-count-dependent SHIFTED
   schedule (`Flux2Schedule.cs`, new, implements the real `compute_empirical_mu`/
   `generalized_time_snr_shift` formulas from `sampling.py`, `Flux2ScheduleTests` confirms real
   numeric properties). But re-running the 20-step check with that fix applied is STILL pure
   noise, visually indistinguishable in character from before the fix** — the schedule bug was real (confirmed
   correct by unit test, `Flux2ScheduleTests`) and worth fixing regardless, but it was NOT the
   dominant cause of the non-convergence, or there are multiple compounding bugs. Candidates
   (a)/(b)/(c)/(d) were all re-checked and found exactly correct; the schedule bug (previously
   uncategorized, now understood) is also fixed; **the remaining bug is most likely in (e), or
   somewhere not yet on this list at all** (e.g. a subtlety in the fused single-block `linear1`/
   `linear2` output-concatenation order, or the attention scale/masking within `DiffusionOps.
   MultiHeadAttention` as used here — both spot-checked by re-reading, not by a numeric test).
   This narrowing is real, cumulative progress even without having found the visually-confirmed
   bug yet — a numeric differential test against a real independent Mistral/DiT reference remains
   the single highest-leverage next step, since structural re-reading plus one real fix have now
   ruled out 5 real candidates without resolving the visual symptom (the same pattern FLUX.1's own
   Round 6-8 experience had before its real fix was found by a numeric, not structural, check).

   **Attempted 2026-09-18, real hardware constraint found.** This machine unexpectedly DOES have
   PyTorch 2.11 + transformers 5.7.0 installed, and `transformers` can load a GGUF checkpoint
   directly via `AutoModelForCausalLM.from_pretrained(gguf_file=...)` (confirmed working for the
   tokenizer/chat-template). Attempted an independent reference dump of the real Mistral-Small-24B
   checkpoint: killed by the OS for low memory at `dtype=torch.float32` (~96GB needed, far over
   this machine's 63GB), retried at `dtype=torch.float16` (~48GB needed vs ~51GB free at the time)
   — got to 84% of tensors converted (304/363) before also being killed. **This is a genuine
   hardware memory constraint, not a bug or a dead end from insufficient effort** —
   `transformers`' GGUF loader fully dequantizes every tensor into a live Python object graph
   before the model is usable at all (no streaming/lazy-per-layer load path), so there's no way to
   get partial credit without a fundamentally different approach (e.g. a custom minimal script
   that loads and runs only the first ~30 layers via the low-level `gguf` tensor reader directly,
   skipping the rest — real, substantial new work, not attempted this pass). Matches this
   project's own established pattern for hardware-constrained items (DeepSeek-V4, Llama-4 vision)
   — a real, named blocker, not silently dropped.
   **Do not report FLUX.2 as visually verified
   until the bug is found and fixed and a re-run shows real structure** — the wiring milestone
   (all 3 components load and run together without crashing) is real and worth keeping, but image
   correctness is a confirmed open bug, not just an unconfirmed claim.

   **MAJOR FINDING, 2026-09-18, same day, before attempting the blocked differential test:**
   switched from "re-read the structure again" to "compare what every OTHER model's pipeline does
   differently" -- and found it. The real reference's `timestep_embedding(t, dim,
   time_factor=1000.0)` (`model.py` line ~719) internally multiplies the raw `[0,1]`
   flow-matching `t` by 1000 before computing sinusoidal angles. This codebase's shared
   `DiffusionOps.SinusoidalTimestepEmbedding` helper does NOT do this scaling itself by design --
   every other caller pre-scales at the PIPELINE level (`WanPipeline.cs`'s
   `activeModel.Forward(latent, t * 1000.0f, ...)`, `HunyuanVideoPipeline.cs`'s identical pattern).
   **`Flux2Pipeline.Generate` was the only pipeline in the entire codebase passing raw `t` (already
   in `[0,1]`) straight through unscaled to `Forward`.** Angles were computed ~1000x too small
   across the ENTIRE denoising trajectory -- the timestep embedding was nearly flat/degenerate at
   every step, making the DiT effectively timestep-blind. This is a complete, sufficient
   explanation for the "zero visible improvement from 2 to 20 steps" symptom: a timestep-blind DiT
   cannot distinguish where it is in the trajectory, so more steps can't help.

   Fixed with one line in `Flux2Pipeline.cs` (`t * 1000.0f` at the `Forward` call site, matching
   the established convention, plus an explanatory comment referencing this finding).

   **Re-ran all 3 real end-to-end tests with the fix applied** (`Flux2EndToEndRealWeightsTests`,
   all pass, real weights, 1553.3s total for the 3-test class): the output changed CHARACTER
   dramatically. The 64×64/20-step convergence check
   (`docs/diffusion-samples/flux2_20step_convergence_check_64_2026-09-18.png`) now shows a real,
   structured, periodic pink/green block-grid pattern -- NOT random per-pixel noise. The
   128×128/4-step quality check (`flux2_quality_check_128_4step_2026-09-18.png`) shows a regular
   red dot-grid pattern, also clearly structured. **This is the exact same "structured, periodic,
   not-yet-coherent tiling" failure signature this project has independently found and eventually
   resolved for both Wan (`docs/056`) and FLUX.1's own multi-round tiling-artifact history** --
   strong circumstantial evidence the timestep fix was the real, dominant bug behind the pure-noise
   symptom, and what remains is a SEPARATE, narrower structural bug (most likely patchify/
   token-ordering, matching the tiling failure shape), not another item on the noise-cause
   candidate list.

   Re-verified the VAE's `UnnormalizeAndUnshuffle` channel-index math
   (`cIndex = c*4 + pi*2 + pj`) against real einops `rearrange(z, "(c pi pj) i j -> c (i pi) (j pj)")`
   flat-index-unravel semantics one more time given the grid-pattern shape -- confirmed
   mathematically correct, not the cause.

   **Real next step**: the same patchify/token-ordering playbook already applied to Wan/FLUX.1/
   Qwen Image/HunyuanVideo, focused on whether `Flux2Pipeline`'s `targetPositions` construction
   (4-axis `t,h,w,l`, already spot-checked against `prc_img`/`prc_txt` in isolation and found
   correct) actually matches the token ORDER `Flux2DiT`'s attention and eventual unpatchify path
   assumes -- a mismatch between "correct position values" and "correct token order" is a
   different, narrower bug than either being individually wrong, and hasn't been directly checked
   yet. If that doesn't resolve it, the numeric differential test against an independent
   Mistral/DiT reference (blocked on this machine's RAM, see above) remains the fallback, now a
   much narrower search given the timestep-blindness explanation is closed.

This is real, substantial implementation work (steps 2-5 each comparable in scope to one of this
session's other single-model fixes) — scoped here so it can be picked up as a focused task rather
than re-derived cold.
