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

## Scope note

RealESRGAN, Z-Image-Turbo, SD1.5, and SDXL-Turbo have real local weights and confirmed real,
correct generations. Wan2.1 and LTX-Video have real local weights too, but real generation is
currently blocked by the porting gaps above (not just missing files). FLUX.1, SD3/3.5, and
HunyuanVideo were not attempted this pass — no local weight sets, and `scripts/download-model.ps1`
doesn't have download entries for them yet (only `z-image-turbo`/`z-image-turbo-q8`/
`realesrgan-x4`).
