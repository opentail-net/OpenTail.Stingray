# 056 — FLUX.1 tiling-artifact handoff

**Status: paused 2026-09-01, ready to hand off.** This doc is written as a self-contained brief
for whoever (human or AI) picks this back up — you should not need to re-read the whole session
history to continue.

## The task

Get `FluxDiT.cs` (FLUX.1-schnell's MM-DiT) producing a visually correct image. It had never been
run once before this pass — the code was a real, substantial, non-stubbed port (reads genuine
named weight tensors via `GgufModel`), but nobody had ever downloaded a checkpoint and pressed go.

## What's already fixed (four real, verified bugs)

All four are already applied in `src/OpenTail.Stingray.Diffusion/FluxDiT.cs` and
`src/OpenTail.Stingray.Diffusion/Flux2DRoPE.cs`. Do not re-investigate these — they're confirmed
against real reference source, not guessed, and each is covered by an inline doc comment
explaining the real vs. wrong behavior at its fix site:

1. **Tensor-name prefix mismatch.** `FluxDiT.cs` hardcodes `model.diffusion_model.*` tensor names
   (the real diffusers/safetensors convention), but `city96`'s GGUF converter strips that
   redundant prefix. Fixed via `FluxDiT.FindTensor`'s fallback (tries both conventions).
2. **Two missing `.weight` suffixes.** `SingleBlock`'s `linear2` and `FinalLayer`'s `linear`
   `MatQ` calls passed the bare tensor-name prefix instead of appending `.weight` like every other
   call site does via the `LinearNoBias`/`LinearBias` helpers.
3. **GEGLU vs. plain GELU.** `SingleBlock`'s MLP was gating a split hidden state (GEGLU); real
   FLUX (`FluxSingleTransformerBlock` in `transformer_flux.py`) applies plain
   `GELU(approximate="tanh")` to the FULL, un-split `mlp_hidden_dim = 4·d`. Confirmed against the
   real diffusers source AND independently against this checkpoint's own `linear2` weight shape
   (`[d, 5d]`, only consistent with the un-split 4d-wide MLP output concatenated with the d-wide
   attention output).
4. **2D RoPE, two compounding bugs, full rewrite.** Confirmed against real
   `black-forest-labs/flux`'s `flux/math.py` (`rope`/`apply_rope`) and `flux/modules/layers.py`
   (`EmbedND`):
   - Real FLUX uses THREE axes (`axes_dim=[16, 56, 56]`: identity time-axis, row, col), each
     independently theta-scaled by ITS OWN axis dim (56), not `head_dim` (128). The old code split
     `head_dim` evenly in half (64/64) with no identity portion.
   - Real FLUX rotates ADJACENT pairs `(x[2i], x[2i+1])` (GPT-NeoX/interleaved convention). The
     old code implemented "rotate-half" (`x[i]` paired with `x[i+head_dim/2]`) AND only ever read
     frequency slots `[0, head_dim/2)` regardless of axis — meaning column-axis frequencies were
     computed but never actually applied. Column position was silently ignored entirely.

After fix #4, the output is confirmed byte-different from the pre-fix run (verified via file
hash), so the fix has a real numeric effect — **but the visible artifact (see below) is
unchanged in character.**

## The remaining problem

Pipeline runs to completion cleanly (no exceptions), ~860s for 4 steps on CPU-only with a Q2_K
quant, but the output image is a periodic small-tile pattern repeated uniformly across the whole
512×512 frame — not remotely resembling the prompt. It looks like a regular textile/chevron
texture, not noise, not a blank/gray field, and not a recognizable-but-flawed image (contrast this
with LTX-Video's artifact, which does show real prompt-semantic structure with dithering on top —
FLUX's current artifact shows NO prompt-semantic structure at all).

**Why RoPE was ruled out as the (sole) cause**: if a purely positional signal were driving the
periodic pattern, fixing it should have changed the artifact's structure, not just its exact pixel
values. It didn't — same tiling character before and after the RoPE rewrite. That points away
from attention/positional wiring and toward something either upstream of the DiT's semantic
content (VAE latent conditioning) or downstream of it in a way that's structurally periodic
regardless of input.

**Already checked and confirmed correct this pass** (do not re-investigate unless new evidence
points back here): patchify/unpatchify (`EulerFlowScheduler.PackLatent`/`UnpackLatent`) — verified
against the real `"b c (h ph) (w pw) -> b (h w) (c ph pw)"` einops rearrange, channel-outer/
row/col patch layout and row-major patch-grid sequence order both match.

## Where to look next (suggested priority order, not mandatory)

1. **VAE latent shift/scale conditioning.** FLUX's real VAE decode applies a `shift_factor` and
   `scaling_factor` to the latent before decoding (`(latent / scaling_factor) + shift_factor`,
   real values `scaling_factor=0.3611`, `shift_factor=0.1159` per BFL's `ae.safetensors`
   metadata / diffusers' `AutoencoderKL` config for FLUX). Check whether `VaeDecoder`/
   `ImagePipeline.Generate` applies these at all, and with the right sign/order. A wrong or
   missing shift/scale on an otherwise-correct latent is a very plausible source of a structured,
   periodic-looking decode, since the VAE's conv stack would be decoding an out-of-distribution
   input it was never trained to handle gracefully, but whose statistics are still "plausible
   enough" to produce texture rather than blank noise.
2. **Q2_K-specific dequant/matvec kernel artifact.** This is the first time this project has ever
   run Q2_K weights through `FluxDiT.MatQ`'s CPU path at this scale. A periodic pattern that
   survives a real semantic fix could be a quantization-block-boundary artifact (Q2_K uses 16- or
   32-element sub-blocks; a wrong stride/offset in `SimdKernels.MatVecQ2K`/`QuantizeRowToQ8K`
   could produce exactly this kind of small-period repeating structure). Cheapest test: re-run
   with a higher-precision quant (Q4_K_S or Q8_0, both available from the same
   `city96/FLUX.1-schnell-gguf` repo) — if the artifact's period or character changes with quant
   level, that's strong evidence it's a kernel-precision issue, not an architecture bug.
3. **Attention QKV wiring inside `DoubleBlock`/`SingleBlock`.** Not yet re-checked line-by-line
   against real FLUX's `SelfAttention`/`DoubleStreamBlock` this pass. Check the QK-RMSNorm
   (`QKNorm` in `FluxDiT.cs`) is applied per-head with the real per-head `query_norm.scale`/
   `key_norm.scale` tensors (confirmed present in the GGUF, see `single_blocks.0.norm.*` in an
   earlier `list-tensors` dump), and that head-splitting/reshaping (`Reshape2Heads`) uses the same
   element ordering the attention scoring code expects.
4. **A numeric golden-parity pass against real diffusers**, one block at a time (img_in → first
   double block → first single block → final_layer), the same discipline
   `docs/055-ltx-video-implementation-plan.md` used for LTX-Video. This is the most thorough and
   most expensive option — only worth it if 1-3 above don't turn up the cause, since it requires
   running the real HuggingFace `diffusers` FLUX pipeline locally (Python, needs its own
   environment) to capture reference intermediate tensors.

## How to reproduce

Checkpoints are NOT vendored (deleted after every pass per this project's convention — see
`CLAUDE.md`). Re-download:

```
DiT:            city96/FLUX.1-schnell-gguf → flux1-schnell-Q2_K.gguf (4.01 GB)
VAE:            ffxvs/vae-flux → ae.safetensors (335 MB — an ungated mirror;
                 black-forest-labs/FLUX.1-schnell's own copy requires HF auth)
CLIP-L:         comfyanonymous/flux_text_encoders → clip_l.safetensors (246 MB)
T5-XXL:         comfyanonymous/flux_text_encoders → t5xxl_fp8_e4m3fn.safetensors (4.9 GB)
CLIP tokenizer: openai/clip-vit-large-patch14 → tokenizer.json
T5 tokenizer:   YuCollection/FLUX.1-schnell-Diffusers → tokenizer_2/tokenizer.json
                (plain google/t5-v1_1-xxl has no fast-tokenizer tokenizer.json of its own)
```

```bash
dotnet run --project src/OpenTail.Stingray.Cli -c Release -- image \
  -m flux1-schnell-Q2_K.gguf \
  --vae ae.safetensors \
  --clip-l clip_l.safetensors --clip-tokenizer tokenizer.json \
  --t5xxl t5xxl_fp8_e4m3fn.safetensors --t5-tokenizer tokenizer_2/tokenizer.json \
  -p "a red apple on a wooden table" \
  --steps 4 --seed 42 --verbose \
  --output flux-out.png
```

CPU-only, ~14-15 minutes for 4 steps. Prior runs are saved at
`docs/diffusion-samples/flux-schnell-first-run.png` (pre-RoPE-fix) and
`docs/diffusion-samples/flux-schnell-rope-fix.png` (post-RoPE-fix) for visual comparison — both
gitignored/local-only per this project's convention, so they only exist on whichever machine
generated them.

## House rules for whoever picks this up (from this project's `CLAUDE.md`)

- **No subagents** — do all work directly in the main session for this project.
- **Check the real reference before "fixing" code that looks wrong** — every bug fixed this pass
  was confirmed against real upstream source (`black-forest-labs/flux`'s actual `math.py`/
  `layers.py`, or the checkpoint's own real tensor shapes) before writing code, not guessed from
  first principles. Do the same.
- **Measure, don't assume**, and **verify before claiming victory** — run the pipeline for real
  after each change and actually look at the output image, the same discipline this pass used.
- **Delete transient checkpoints after use** — do not commit large model files.
- **Performance pass + DRY pass only once the port is actually complete and correct** — do not
  optimize or refactor `FluxDiT.cs` while it's still producing wrong output.
