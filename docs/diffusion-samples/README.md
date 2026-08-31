# Diffusion pipeline spot-check — real examples & timings

Generated this session while verifying the diffusion pipelines actually work end-to-end, per user
request. All images are real, weight-driven generations (no placeholders). CPU is an AMD Ryzen
(no CUDA); GPU is Vulkan on an integrated AMD Radeon iGPU (16GB shared VRAM budget), not a
dedicated card — timings reflect that.

## Results

| Pipeline | Prompt | Size / Steps | Backend | Time | Status |
|---|---|---|---|---|---|
| RealESRGAN x4 | (synthetic test pattern) | 256→1024px | CPU | 338.8s | ✅ Works |
| Z-Image-Turbo | "a red apple on a white table" | 256×256, 4 steps | **CPU** | 226.1s | ✅ Works |
| Z-Image-Turbo | "a serene mountain lake at sunrise" | 512×512, 9 steps | **Vulkan GPU** | 1193.5s | ❌ **Black image bug — GPU-path only** |
| SD1.5 | "a red apple on a white table" | 512×512, 20 steps | CPU | 3422.7s | ✅ Works (weak prompt adherence — expected for SD1.5) |
| SD1.5 | "a red apple on a white table" | 512×512, 20 steps | Vulkan GPU | 1077.2s | ✅ Works, ~3.2x faster than CPU |
| SDXL-Turbo | "a red apple on a white table" | 512×512, 4 steps | Vulkan GPU | 300.2s | ✅ Works |

## Bugs found and fixed this pass

1. **`ImageCommand.IsSdxl` false-positive on real SD1.5 checkpoints** (fixed, commit `72358e4`).
   Its content-sniffing heuristic checked for a UNet tensor key that also exists in real SD1.5
   checkpoints (not SDXL-exclusive as assumed), so every real SD1.5 file was silently misrouted
   into the SDXL loader and failed to load at all. Confirmed directly against real downloaded
   checkpoints before fixing. `v1-5-pruned-emaonly.safetensors` now correctly loads as SD1.5.

2. **Z-Image-Turbo produces a solid black image on the Vulkan GPU backend** (NOT fixed — flagged
   for follow-up). The exact same model/weights/prompt produces a correct image on CPU. SDXL-Turbo
   and SD1.5 both work correctly on the same Vulkan GPU backend, so this is specific to Z-Image's
   own Vulkan kernel path (its S3-DiT architecture), not a general Vulkan/GPU-backend bug. Real,
   reproducible, not yet root-caused.

## Round 2: Wan / LTX-Video / HunyuanVideo — a much bigger finding

Downloaded real checkpoints for Wan2.1 (VAE, converted from the official `.pth` to safetensors)
and LTX-Video (`ltx-video-2b-v0.9.1.safetensors`, 5.7GB), on top of the DiT GGUF already present
for Wan. Attempting to actually run them surfaced something more significant than a single bug:

3. **`ImageCommand.Execute`'s dispatch chain never routed to Wan, HunyuanVideo, or LTX-Video at
   all** (fixed, commit `6bf83d2`). `RunWan`/`RunHunyuanVideo`/`RunLtxVideo` all exist as real,
   implemented functions, and `IsStableDiffusion`'s own exclusion list already accounted for all
   three (implying this was the intent) — but `Execute()` itself never called `IsWan`/
   `IsHunyuanVideo`/`IsLtxVideo`. Any of their model paths silently fell through to the FLUX loader
   and failed with a misleading FLUX-shaped error instead. Fixed by adding the three missing checks.

4. **Fixing the dispatch surfaced that none of the three are actually finished end-to-end** (NOT
   fixed — these are real porting gaps, not quick wiring bugs, flagged for a dedicated follow-up
   rather than attempted here):
   - **Wan 2.1**: `WanModel` genuinely receives and applies real transformer weights. But running
     it for real against the DiT-only GGUF fails with `GGUF tensor not found: 'text_embedding.
     weight'` — the real UMT5 text encoder isn't baked into the DiT quant and isn't wired up
     anywhere in the CLI path, so there's currently no way to get a real text-conditioned Wan
     generation from this command. (Separately, `WanPipeline.Load()` also constructs the generic
     2D `VaeDecoder` with `vaePath` for its single-frame path, while `Generate()`'s multi-frame
     path builds an entirely separate `WanVaeDecoder3D` from `_weights` instead — two different,
     inconsistently-wired VAE code paths in the same pipeline.)

     **Update (commit `801c551`, following-up session)**: all of the above fixed. Wired a real
     UMT5-XXL text encoder (`UMT5Encoder.cs`, rewritten against the real Wan-shipped checkpoint's
     own tensor names, not HF's) through `--umt5-encoder`/`--umt5-tokenizer` CLI flags. Along the
     way, found and fixed three further real bugs in `WanModel.cs` itself once real (non-zero)
     text conditioning actually started flowing through it: `text_embedding` was a single Linear
     instead of the real 2-layer MLP; the AdaLN modulation was computing a fictitious per-block
     Linear instead of the real shared-projection-plus-additive-constant scheme; cross-attention
     was missing its pre-norm entirely. Unified the single/multi-frame VAE paths onto one real
     `WanVaeDecoder3D`, which itself needed a full rewrite (see below). **Real, end-to-end 1-step
     CPU run at 256×256 now completes and produces a real, non-degenerate image** (see
     `wan2.1-t2v_red-apple-white-table_CPU-256x256-1step_pipeline-runs-not-converged.png` in this
     directory) — expected "unconverged flow-matching" appearance at 1 step.

     **8-step follow-up run (same session): still NOT converging** (see
     `wan2.1-t2v_red-apple-white-table_CPU-256x256-8steps_NOT-CONVERGING-BUG.png`). At 8 real
     Euler steps with CFG guidance=6, a working flow-matching pipeline should show clear emerging
     structure toward the prompt (compare Z-Image-Turbo's real apple shape at just 4 steps,
     `z-image-turbo_red-apple-on-white-table_CPU-256x256-4steps_GOOD.png`) -- this output looks
     essentially unchanged from the 1-step image, no red/white/apple-shaped structure at all. This
     means a real bug remains beyond the ones already fixed (text_embedding MLP, AdaLN modulation,
     cross-attn pre-norm, VAE tensor names) -- most likely somewhere in the DiT's attention/RoPE,
     the CFG combination, or the flow-matching schedule itself, not yet root-caused. Flagged as an
     open, real gap rather than claimed fixed -- do not mark Wan green until a run actually
     converges.

   - **`WanVaeDecoder3D` was ALSO wrong** (found while fixing the above): assumed HuggingFace-
     `diffusers`-renamed tensor keys (`decoder.conv_in`, `decoder.mid_block.resnets.N`,
     `decoder.up_blocks.N`, split `to_q`/`to_k`/`to_v`, a custom "DupUp3D" upsample) that don't
     exist in the real downloaded `Wan2.1_VAE.safetensors` at all, and silently returned
     zero-filled tensors for any missing weight instead of erroring — which is exactly why this
     went unnoticed until the DiT-side bugs were fixed far enough to actually reach VAE decode.
     Fully rewritten against the real checkpoint's own tensor names/shapes, cross-checked against
     both `examples/stable-diffusion.cpp/src/model/vae/wan_vae.hpp` and
     `examples/diffusers/.../autoencoder_kl_wan.py`. Real, documented scope limit: the real
     architecture's temporal-upsample doubling only fires with a previous latent frame's
     causal-conv cache, which isn't threaded across frames yet — bit-exact for single-frame
     (image) output, a real gap for multi-frame video (would need real per-frame cache
     threading to get the correct `(t-1)*4+1` output-frame count).
   - **LTX-Video**: `RunLtxVideo` called `new LtxVideoPipeline()` (the bare parameterless
     constructor) instead of `LtxVideoPipeline.Load(modelPath, ...)`, so the user's `-m` checkpoint
     was never even opened. Beyond that CLI-level bug: `LtxVideoPipeline.Load()` itself constructs
     `new LtxVideoModel()` with no arguments, never applying the loaded `IWeightLoader` to the
     transformer at all, and `GenerateVideo`'s "text context" is `0.01f *
     (rng.NextSingle() - 0.5f)` — literal random noise, not a real text encoder. This pipeline is
     a structural placeholder, not a finished port.

     **Re-audited (follow-up session, 2026-08-31), real scope is bigger than the above implies**:
     `LtxVideoModel` has NO weight-loading capability anywhere in the class -- no
     `IWeightLoader` constructor parameter, no `GetWeight`/tensor-read calls at all. Its
     "patchify projection" (`Forward`'s step 2) is a hardcoded formula (`0.05f / (1 + i)`), and
     its timestep embedding (step 3) is a hardcoded sinusoidal formula with no learned MLP. This
     is not a wiring gap on top of a real port -- it's a from-scratch architectural stub with
     zero real weight consumption anywhere in the 28-layer transformer. Porting it for real
     (patchify Linear, timestep MLP, 28x real AdaLN+self-attn-with-RoPE+T5-cross-attn+FFN
     blocks, plus a real T5-XXL text encoder integration) is comparable in scope to this
     session's entire Wan DiT+VAE effort, not a quick fix -- a real reference does exist
     (`examples/stable-diffusion.cpp/src/model/diffusion/ltxv.hpp`, per this file's own doc
     comment) so it's tractable, just a genuinely separate, multi-session undertaking. Deferred
     rather than attempted mid-checklist alongside smaller, well-scoped items.
   - **HunyuanVideo**: `HunyuanVideoModel` does receive real weights (like Wan), but wasn't tested
     this pass — no local checkpoint (`hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` is
     large, not downloaded) and its own text-conditioning wiring wasn't audited.

     **Update (follow-up session, 2026-08-31): downloaded and structurally smoke-tested.**
     Downloaded the real 13.2GB checkpoint (`Kijai/HunyuanVideo_comfy`) and ran a 1-step, 256x256,
     zero-conditioning CPU smoke test. Real, clean finding: **the DiT forward pass runs completely
     through every layer with no tensor errors** -- `HunyuanVideoModel`'s real multi-candidate
     tensor-name/config detection (dim, head count, double/single block depth all auto-detected
     from real tensor shapes) is structurally correct against this real checkpoint, a meaningfully
     better starting state than Wan's DiT had before its fixes. The run fails only at VAE decode
     (`Safetensors tensor not found: 'decoder.conv_in.weight'`) -- the exact same bug CLASS Wan had
     before its VAE rewrite: `HunyuanVideoPipeline.Load()` falls back to the DiT checkpoint itself
     for VAE tensors when no separate `--vae` is given (this "diffusion_models"-only repackaged
     checkpoint has none), and even with a real separate HunyuanVideo VAE checkpoint, the generic
     SD-style `VaeDecoder` class wouldn't match its real tensor names either (HunyuanVideo's real
     VAE, like Wan's, needs its own dedicated decoder class).

     **Real remaining gaps for a working generation** (not attempted this pass -- each is its own
     Wan-VAE-sized or larger effort): (1) a real HunyuanVideo VAE checkpoint + a dedicated decoder
     class written against its real tensor names/architecture (same shape of work as
     `WanVaeDecoder3D`'s rewrite); (2) real dual text conditioning -- `HunyuanVideoPipeline.Generate`
     is never even called with a `textContext:` argument in `ImageCommand.cs`'s `RunHunyuanVideo`,
     same all-zero-conditioning gap Wan had, except HunyuanVideo's real reference needs BOTH a
     CLIP text encoder AND an LLM (LLaVA/similar) encoder ("Dual Text Conditioning"), a bigger
     integration than Wan's single UMT5. Deferred as a separate, future multi-session item -- the
     DiT itself being structurally sound is the real, valuable finding to keep from this pass.

## Round 3: five real Wan bugs fixed (RoPE, QK-norm, unpatchify, VAE scaling, VAE upsample) — still no convergence, real bug remains

Following up on Round 2's "real bug remains, not yet root-caused": two more real, reference-verified
DiT defects were found and fixed this session — interleaved ("GPT-J") RoPE pairing where the code
previously used split-half/NEOX pairing (`WanRoPE.cs`), and QK-norm computed per-head instead of
over the full concatenated projection width (`WanModel.RmsNormHeads`). Neither alone changed the
non-convergence symptom, but both are genuine, reference-confirmed fixes (confirmed against
`transformer_wan.py`), kept regardless.

A parallel session (same git author, commit `31bc129`) independently found and fixed three further
real bugs: `UnpackLatents`'s channel-ordering (spatial-sub-position must be the outer index with
channel inner, not the reverse — confirmed against `transformer_wan.py`'s unpatchify
`reshape`/`permute`), the VAE latent-unnormalization formula (`z = latent * std + mean`, not
`z = latent / std + mean`), and the VAE's nearest-2x-upsample+conv coordinate math. All 81 tests in
`OpenTail.Stingray.Tests.Diffusion` pass with the combined fix set.

**Verification run with all five fixes applied** (2 steps, 256×256, CPU, ~1234s,
`wan-unpatchify-fix-check.png` in this directory): still does **not** converge toward the prompt.
The output changed character, though — it's no longer unstructured noise, but a regular periodic
grid/checkerboard tiling artifact (visible vertical banding at what looks like the patch or VAE
upsample tile boundary). That's a real, measured negative result, and a more specific one than
Round 2's: a periodic tiling artifact at 2-step count points at a remaining bug in how tiles/patches
are stitched back together spatially — most likely still in the unpatchify path, the VAE's spatial
upsample tiling, or a patch-embedding/grid-arrangement mismatch — rather than in RoPE, QK-norm, or
the two VAE formula fixes, which are now independently confirmed correct. **Do not mark Wan green.**
The five fixes are kept (each individually reference-verified), but the root cause of
non-convergence is still open.

**Follow-up diagnostic (2026-08-31): bug localized to the VAE decoder, not the DiT.** Per another
AI's suggested split-test, `WanVaeDecoder3D.Decode` was fed a synthetic all-zero latent directly
(bypassing `WanModel`/`WanRoPE` entirely) — after the decoder's own `z = latent * std + mean`
un-normalization this becomes a perfectly flat, spatially-constant per-channel field with zero
input variation. The decode still produced visible periodic horizontal banding
(`wan-vae-synthetic-dc-latent-diagnostic.png` in this directory,
`WanVaeSyntheticLatentDiagnosticTests.Decode_AllZeroSyntheticLatent_IsolatesVaeFromDiT`). Since the
input carried no spatial signal at all, the banding can only have been introduced by the decoder
itself — this rules out RoPE, QK-norm, patchify/unpatchify, and the DiT generally as the source of
this specific artifact, and localizes it to `WanVaeDecoder3D`'s `ResampleSpatial` (nearest-upsample
+ conv phase) or `CausalConv3D` (replicate-pad edge handling) path. Not yet root-caused further —
the diagnostic's own numeric even/odd-column-parity metric measured the wrong axis (the real
artifact is row-wise/horizontal, not column-wise/vertical) and should be redone measuring row
parity before further narrowing.

**Correction after redoing the row-parity measurement properly (same day): the VAE is actually
clean, and the "periodic banding" read was a misread of a boundary-decay gradient.** Dumping real
per-row R-channel values (not just the even/odd split) showed the center of the decoded frame
(rows 124-131 of 256) identical to 5 decimal places — a perfectly flat interior — with rows near
the top edge (0-23) showing a smooth, monotonic gradient converging toward that flat center value.
Even/odd row split: 0.0000; even/odd column split: 0.0001. That is exactly the expected signature
of a CORRECT zero-padded conv stack (~20 stacked `padding=1` 3x3/3x3x3 convs between `conv1` and
`conv_out`) fed a spatially-uniform input: a boundary artifact that creeps roughly one pixel deeper
per layer, decaying to nothing well before the center, not full-frame periodic banding. (Convolving
a spatially-constant field with any fixed kernel yields a spatially-constant field at every interior
position, regardless of what the kernel's weight VALUES are — so genuinely periodic banding from a
uniform input would require a bug that breaks translation invariance, which this data rules out.)
The earlier "checkerboard" read of the small PNG thumbnail was very likely a moiré/downscaling
artifact from viewing a shrunk 256x256 image, not real per-pixel structure.

**Deep line-by-line re-verification of `WanVaeDecoder3D` and `WanModel`'s patchify/unpatchify/RoPE
against the real reference, following up on the other AI's four hypotheses — all four ruled out:**
1. *RoPE axis mislabeling (t/h/w swapped)*: checked `WanRotaryPosEmbed.forward` in
   `transformer_wan.py` directly — frequency axes are concatenated `[t_dim, h_dim, w_dim]` and the
   token grid is flattened `(ppf, pph, ppw)` row-major (t outermost, w innermost), and `WanRoPE.cs`
   assigns `dimT` first, `dimH` to the height position, `dimW` to the width position in the exact
   same order. No swap.
2. *VAE `ResampleSpatial` upsample phase offset*: `WanUpsample(mode="nearest-exact")` composed with
   the pad-1 3x3 conv was re-derived from PyTorch's actual `nearest-exact` index formula
   (`floor((dst+0.5)*scale)`); for an exact 2x scale factor this reduces to `dst // 2`, identical to
   the plain-`nearest` mapping already implemented. No phase bug, and the empirical DC-latent test
   now confirms it directly.
3. *VAE `CausalConv3D` single-frame temporal slice selection*: re-derived `WanCausalConv3d`'s real
   padding tuple (`_padding = (padW,padW,padH,padH, 2*padT, 0)`, confirmed in
   `autoencoder_kl_wan.py`) — for `kt=3, padding=1` that's exactly `padT=2` on the LEFT only (causal,
   zero-padded, not replicate — Wan differs from HunyuanVideo here), matching this port's own
   `CausalConv3D` exactly, including which single kernel tap survives at `t=1`.
4. *DiT `patch_embedding` weight layout mismatch between `PackLatents` and the real conv weight*:
   confirmed `patch_embedding` is a real `nn.Conv3d(16, dim, kernel_size=(1,2,2))`, whose weight's
   row-major layout is `[dim, 16, 1, 2, 2]` — channel outermost, patch-position innermost — and
   `PackLatents`'s own loop order (`c` outer, `dy`, `dx` inner) matches exactly. Separately,
   `proj_out`'s real output-feature order was re-derived from `transformer_wan.py`'s own
   `reshape(...,p_t,p_h,p_w,out_channels).permute(...)` (lines 726-730): channel is the FASTEST/
   innermost index there, which is the opposite convention from `patch_embedding`'s — but that's
   expected (they're different weight matrices with independently-real conventions, not required to
   match each other), and `UnpackLatents`'s `(dy*2+dx)*OutChannels + c` layout matches `proj_out`'s
   real order exactly (re-confirming the earlier `31bc129` fix, not a new bug).

**Working hypothesis, not yet tested: 2 denoising steps may simply be too few for ANY correct
flow-matching model to converge, and the "checkerboard" look could be the model's own 2x2 patch
grid showing through an under-denoised early-step latent** — normal behavior at 2 steps, not a bug
signature. All five previously-fixed bugs are real and correctly fixed; all four of the newest
hypotheses are now individually ruled out by direct reference comparison; the VAE is empirically
clean. A longer run (12+ steps) is the next real test of whether the pipeline was actually working
correctly the whole time and the low step count was simply misleading. See below for that run.

## Scope note

RealESRGAN, Z-Image-Turbo, SD1.5, and SDXL-Turbo have real local weights and confirmed real,
correct generations. Wan2.1 and LTX-Video have real local weights too, but real generation is
currently blocked by the porting gaps above (not just missing files). FLUX.1, SD3/3.5, and
HunyuanVideo were not attempted this pass — no local weight sets, and `scripts/download-model.ps1`
doesn't have download entries for them yet (only `z-image-turbo`/`z-image-turbo-q8`/
`realesrgan-x4`).
