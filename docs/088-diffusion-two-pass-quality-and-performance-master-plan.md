# 088 — Diffusion/Video two-pass master plan: Quality, then Performance (2026-09-18)

## Standing instruction for whoever (or whatever `/loop` iteration) picks this up

**If the honest answer to "this is a massive piece of work — should I still do it?" would otherwise
be a question, the answer is always YES.** Keep working through the checkboxes below in order. Do
not stop the loop because an item is large — split it into smaller real steps and keep moving. Only
stop on a genuine blocker (missing hardware, missing checkpoint that can't be downloaded, a decision
only the operator can make) — and when you do, **document the blocker precisely and move to the next
checkbox**, per this project's CLAUDE.md "stopping is for wimps" rule. Do not use subagents (CLAUDE.md
rule 6) — do all of this directly.

**Wan 2.1/2.2 worked once (real photorealistic output existed at some point in this project's
history) and currently does not (the persistent grid artifact, `docs/diffusion-samples/README.md`).
Do not undo any of Wan's existing performance work (`docs/083`'s GPU-residency: 2.20x of the C++
Vulkan reference, `WanModel.cs`'s 54 real backend call sites) while fixing the correctness
regression. Fix forward: find and fix the real bug, then re-verify performance hasn't regressed
using the exact same measurement methodology already in `PerformanceLeague.md`'s Wan rows.**

**Current standing priority order, 2026-09-19 (re-check this list is still current before picking
the next item, since a status can change mid-loop):**
1. ~~**FLUX.2 512×512/20-step CPU re-verification**~~ — **DONE AND CLOSED 2026-09-19**, see the
   Pass 1 entry: real 512×512/20-step run confirms a clean, coherent, photorealistic apple, zero
   artifacts. Move straight to item 2.
2. **FLUX.1** — 🟡 open, tiling artifact already fully fixed 2026-09-13, real recognizable apple
   renders (853s CPU, 3.40x slower than C++'s 251.1s). **2026-09-19: conditioning-vector golden
   check added and passes at machine precision** (cosine=1.0, see Pass 1 entry) — a real, scoped
   win, not full numeric verification. Still open: no block-level/output-level golden fixture
   exists (unlike LTX-Video's), and it's not yet confirmed whether the remaining CPU-vs-C++ perf
   gap (mostly the DiT denoise loop) has any correctness angle left or is purely Pass 2 work.
3. **SD3/3.5** — 🟡 open, real coherent non-photorealistic structure achieved (536.4s CPU,
   256×256/20-step, 11.17x slower than C++'s 48.01s). **2026-09-19 correction: this section's
   "never numerically golden-verified" framing was itself stale** — `Sd3TimestepEmbedParityTests.cs`
   (verifying `MMDiTModel.ComputeTimeAndPooledEmbedding`, the single conditioning-vector input every
   joint block's AdaLN depends on, same class of check as FLUX.1's new one above) already existed
   with a real fixture (`tests/fixtures/sd3_timestep_embed/`) but had never been re-confirmed
   passing after the 2026-09-05 correctness fixes. Re-ran it for real (`STINGRAY_SD3_DIT_PATH` set
   to the real `sd3.5_medium-Q4_K_M.gguf`): **passes, 0.499s.** This closes the conditioning-vector
   piece specifically. **Still genuinely open**: no block-level or full-DiT-output golden fixture
   exists for SD3.5 (unlike LTX-Video's `TestData/LtxGolden/manifest.json`) — the dual-attention
   MMDiT-X extension, AdaLN gating, and unpatchify math (all fixed 2026-09-05 by reasoning against
   source, never numerically confirmed) remain unverified at the tensor level. That deeper check —
   reimplementing one real joint-attention block in Python via direct GGUF weight reads (same
   pattern as `scripts/flux1_vec_embed_ref.py`/`scripts/sd3_timestep_embed_ref.py`) — is real,
   scoped, doable work, just not done this pass.
4. **HunyuanVideo** — 🟡 open, unresolved, furthest from done. **2026-09-19: every major DiT/VAE
   subsystem now independently re-audited this session with ZERO bugs found** — RoPE (pairing
   convention, img/txt ordering, application scope), timestep embedding (flip-sin-to-cos, scale),
   TokenRefiner (pooling, embedders, affine norms, gating, FFN activation), AdaLN 6-way chunk
   order, and the full VAE decoder (causal padding, upsample frame-casing, attention mask) — all
   confirmed correct against the real reference. Output is still pure noise despite this. Two real,
   not-yet-checked candidates remain (see Pass 1 entry for detail): (a) fp8 dequantization
   correctness for this checkpoint's unusual `fp8_e4m3fn` format, (b) the patch-embedding
   Conv3d-as-Linear weight layout vs `PackLatents`' flatten order at the top of
   `HunyuanVideoTransformer3DModel.forward` — not yet independently confirmed against the real
   checkpoint's own tensor shape, distinct from `PackLatents`' own logic already checked. This item
   has had unusually high audit effort for zero bugs found — a genuinely different investigative
   approach (e.g. a full numeric latent dump compared directly against a Python-side partial
   reference, rather than more line-by-line reading) may be more productive than another audit pass.
5. **GPU-residency phases for Qwen Image and FLUX.2** (see Pass 2 §2d, new) — FLUX.2's
   double-block-only GPU residency **investigated and CLOSED 2026-09-19, real negative result**:
   built and numerically verified correct (cosine>0.9999 vs CPU), but a real production-scale
   timing measurement found CPU is 1.49x faster than GPU, and the one-time weight upload alone
   costs more than an entire CPU compute pass — see `docs/091`'s final status and
   `PerformanceLeague.md` for the full measurement. **Not pursued further on this hardware.**
   Qwen Image's GPU-residency phase remains open/not-started.

## Why two passes, in this order

This project's own ordering rule (`docs/00-current-work.md`'s "goal that orders this list"): a
model that cannot be run correctly at all is a worse outcome than one that runs slowly. Optimizing
a pipeline that produces wrong output just makes the wrong output arrive faster — CLAUDE.md rule 7
already states performance work only happens "once a model's port is complete" (every stage
golden-verified, wired end-to-end, tests passing). So: **Pass 1 (Quality) must close or explicitly
document every model's correctness gap before Pass 2 (Performance) spends real time on that same
model's GPU residency.** Models already correct AND already fast (SD1.5, SDXL-Turbo, SD3.5, FLUX.1,
LTX's individual components) don't need re-work in either pass — verify they're still true, don't
redo them.

## Real reference material available in this repo (use these, do not guess)

- `examples/stable-diffusion.cpp/` — real, buildable C++ reference (`sd-cli.exe`) covering SD1.5,
  SDXL, SD3/3.5, FLUX.1, and (per `sd_version_uses_wan_vae`) Wan2.1/Qwen-Image's shared VAE family.
  This is the primary correctness AND performance oracle for every model it covers — already used
  successfully for FLUX.1's Round 9 fix and Qwen-Image's VAE discovery.
- `examples/diffusers/` — real vendored HuggingFace Python reference, broader model coverage than
  sd.cpp (includes LTX-Video's `pipeline_ltx_video.py`, HunyuanVideo's
  `autoencoder_kl_hunyuan_video.py`, FLUX.1's own pipeline).
- `examples/flux/` — BFL's own FLUX.1 repo (`src/flux/model.py`, `modules/layers.py`). **Confirmed
  2026-09-18: this is FLUX.1 only** (`img_mod`/`txt_mod` are per-block `Modulation` instances, not
  FLUX.2's shared/single-instance scheme) — do not treat this as a FLUX.2/FLUX.3 reference. No
  in-repo FLUX.2/FLUX.3 reference currently exists; `docs/087`'s architectural answers came from an
  external report only, cross-shape-corroborated but not in-repo-verified.
- `audio.cpp`'s `sd_version_uses_wan_vae` finding pattern (used for Qwen-Image) is a reusable
  technique: before writing ANY new architecture code for a model, grep
  `examples/stable-diffusion.cpp/src/model/` for whether it's secretly the same family as something
  already working in this codebase.

## PASS 1 — QUALITY (correctness first, ordered by how close each model already is)

For each model: verify current status is still true (don't trust a stale doc claim), then either
close the gap or write down precisely what's blocking it. Update the model's row/section in
`README.md`'s status matrix AND `docs/diffusion-samples/README.md` (or the model's dedicated
handoff doc) in the same pass a real finding lands — per CLAUDE.md rule 10, an unsourced status is
worse than no status.

### 1a. Already 🟢 — confirm still true, do not re-derive from scratch
- [x] **Stable Diffusion 1.5 (+ControlNet)** — 2026-09-18: re-ran, exit 0, real per-pass Vulkan GPU
      timing logged (`~3.6-4.0s/pass`), total 199.9s for 20 steps/512×512 -- matches
      `PerformanceLeague.md`'s documented 204.4s within noise. No regression.
- [x] **SDXL / SDXL-Turbo** — 2026-09-18: redownloaded `sd_xl_turbo_1.0_fp16.safetensors` (6.9GB,
      `hf download stabilityai/sdxl-turbo`). Real re-run surfaced one genuine, immediate gap (not a
      no-op): `SdxlPipeline.Load` needs `models/clip_tokenizer.json`, which wasn't present either --
      copied from `models/flux1-schnell/tokenizer_clip/tokenizer.json` (same real HF CLIP tokenizer
      format, `ClipTokenizer.FromFile` confirmed format-compatible). Re-ran: both facts pass with
      real weights loaded (pipeline init + safetensors validity). Full generation-timing
      re-verification (the 1.60x-of-C++ claim) NOT done this pass -- that needs the actual
      end-to-end generation benchmark, not just load/init; loading real weights successfully is
      real progress but is a narrower claim than the full perf re-check.
- [x] **FLUX.1-schnell** — 2026-09-18: re-ran `FluxRealWeightsTests` (real Vulkan GPU weights,
      full CLIP-L+T5-XXL+VAE pipeline). Warm pass 198.4s -- matches `PerformanceLeague.md`'s
      documented 198.1s within noise, no regression. Output PNG regenerated (444KB, non-degenerate)
      confirming the Round 9 T5-padding fix still holds.

      **2026-09-19: real numeric golden-parity check added for FLUX.1's conditioning vector
      (`FluxDiT.ComputeVec`) -- the single input every double/single block's AdaLN modulation
      derives from, and this session's own repeated bug class (FLUX.2's `t*1000` timestep-
      blindness bug, HunyuanVideo/Qwen Image's RoPE bugs -- all "one shared upstream value feeds
      every block" bugs).** FLUX.1 never had this check before, unlike SD3.5 (which has had an
      analogous `MMDiTModel.ComputeTimeAndPooledEmbedding` golden check since 2026-09-14). Built
      `scripts/flux1_vec_embed_ref.py` (same pattern as `scripts/sd3_timestep_embed_ref.py`: reads
      the real `time_in`/`vector_in` MLP weights directly out of the real
      `flux1-schnell-Q4_K_S.gguf` checkpoint via the `gguf` Python package already installed on
      this machine, reimplements the exact real formula confirmed against `examples/diffusers`'s
      `CombinedTimestepTextProjEmbeddings`/`get_timestep_embedding` -- no full diffusers pipeline
      load needed) and a new `Flux1VecEmbedGoldenTests.cs` (made `FluxDiT.ComputeVec` `internal`
      instead of `private` to call it directly, via the existing `InternalsVisibleTo` grant).
      **Result: cosine=1.000000000, maxDiff=3.8×10⁻⁶ — machine-precision match.** This
      independently confirms FLUX.1's `t*1000` timestep pre-scaling (already fixed 2026-09-11) and
      the `[cos,sin]` (not `[sin,cos]`) flip-sin-to-cos ordering are both correct against the real
      reference, closing the "never golden-verified" gap for this one specific, high-leverage
      piece. **Honest scope limit**: this is a real, valuable, but narrow win — it verifies the
      conditioning-vector MLP math only (with a synthetic, not real-CLIP, pooled input, same
      deliberate scope limit as SD3.5's own script), not the full DiT block/attention/RoPE/VAE
      chain end-to-end. A full block-level or output-level golden fixture (comparable to LTX-
      Video's `TestData/LtxGolden/manifest.json`) still does not exist for FLUX.1 and would be the
      next real step if deeper numeric verification is ever prioritized — not done this pass.
- [x] **Z-Image-Turbo** — 2026-09-18 CORRECTED: `z_image_turbo-Q4_0.gguf` WAS present all along, at
      `models/_models/` (symlinked to `F:\_models`) -- `ZImageRealWeightsTests.FindModelPath` only
      ever searched `models/` directly, never `models/_models/`, the exact same systemic gap
      `PerformanceLeague.md`'s "Known Measurement Gaps" section already flagged for
      `ParakeetRealWeightsTests`. Fixed `FindModelPath` to also check `models/_models/`; re-ran,
      now genuinely passes (real GGUF open + tensor-count assertion, not a no-op). Z-Image-Turbo's
      known-good end-to-end sample (`z-image-turbo_..._FIXED-2026-09-12.png`) was never actually in
      question -- this only fixes a stale unit test's search path.

### 1b. 🟡 with a real, named, scoped gap — close these next
- [x] **SD3/SD3.5 — re-verified 2026-09-18, status unchanged (still 🟡).** Ran the full real-weight
      suite (`Sd3TimestepEmbedParityTests` + `Sd3BaselineTests`, 4 facts): all pass, real weights,
      GPU/CPU timestep-embed parity cosine=1.000000. Timing: 77.1s warm/84.9s cold Vulkan, 193.6s
      CPU (256×256/20-step) — matches/beats `PerformanceLeague.md`'s documented 91.3s/101.2s/536.4s
      figures, no regression. **Output image re-confirmed matching the documented description
      exactly**: real, coherent, asymmetric structure (distinct shapes/edges/shading), NOT noise,
      but not yet a clean photorealistic match to the prompt either — no surprise fix here, unlike
      Wan/LTX this session. Still **not numerically golden-verified** against
      `examples/stable-diffusion.cpp`'s SD3.5 path or `examples/diffusers`' own
      `pipeline_stable_diffusion_3.py` -- that real per-stage numeric diff (the actual remaining
      gap to reach 🟢) was not attempted this pass, scoped as future work matching LTX-Video's own
      golden-verification pattern.
- [x] **Qwen Image / Qwen Image Edit** — DiT+VAE verified, single named gap: real LLM text
      conditioning. **CORRECTION, 2026-09-18: the checkpoint was never actually missing.**
      `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (4.68GB, dated 2026-09-18 02:22) IS present at
      `models/_models/` -- an earlier check this same session used `find models/_models -maxdepth 1
      -iname "*.gguf"`, which silently failed to traverse `models/_models` because it's a symlink
      to `F:\_models` and `find` doesn't follow symlinks by default. `ls models/_models/*.gguf`
      (or any symlink-aware listing) shows it correctly. **This item is NOT blocked on a download
      -- it's ready for real implementation now.** `list-metadata` confirms `general.architecture =
      qwen2vl`, NOT plain `qwen2`, and `qwen2vl` is NOT currently in `ModelCompatibility`'s
      allowlist at all -- a real, scoped gap: either extend the allowlist/forward-pass dispatch for
      `qwen2vl`'s text-only path (check whether it's just `qwen2` plus M-RoPE that degenerates to
      standard 1D RoPE for text-only input, or a genuine structural difference), or check whether
      this codebase's existing `UnifiedVisionPipeline`/`QwenVlVisionModel` machinery (already used
      for real Qwen2.5-VL vision tasks elsewhere in this project) already has a working text-decode
      path that could be reused instead of re-deriving `qwen2vl` support from scratch.
      **2026-09-18: full real recipe confirmed and written up in `docs/089-qwen-image-text-
      conditioning-plan.md`** (ChatML template, `drop_idx=34` crop, final-layer-only extraction,
      dynamic re-pad -- checked against the real `pipeline_qwenimage.py`), plus a concrete plan for
      the `qwen2vl` architecture gap reusing the exact precedent already established for
      Qwen3-ForcedAligner's M-RoPE-degenerates-to-1D-for-text-only case. **`qwen2vl` admitted
      2026-09-18** -- verified directly against the real vendored `transformers` source (this
      machine has PyTorch+transformers installed), `QwenImageTextConditioningTests` confirms a
      real forward pass against Qwen2.5-VL-7B-Instruct (4.36GB) produces finite non-zero hidden
      states. **Full recipe DONE, same day**: `QwenImageTextConditioning.Encode` (new) implements
      the real ChatML template + `drop_idx=34` crop + final-layer extraction, passes against real
      weights (finite `ContextDim=3584`-per-token output). **DONE, same day: wired into
      `QwenImagePipeline` itself** via a new `Load(modelPath, textEncoderPath, vaePath, backend)`
      overload, `Generate()` uses it automatically when no explicit `textContext` is supplied
      (additive, no regression on the existing real-weight forward-pass test). **Qwen Image's
      text-conditioning gap is now closed at the code level.** **Coherence check DONE, same day:
      NOT coherent -- a severe, regular checkerboard/tiling artifact** (256×256/8-step, real CFG
      guidance=4.0, 2889.9s). **Critical finding: this output is visually IDENTICAL to the earlier
      zero-conditioning run** — proving the real text-conditioning wiring works correctly (real
      vs. zero conditioning makes zero difference to this artifact) and the checkerboard pattern
      is a completely separate, pre-existing structural bug (patchify/RoPE/VAE-tiling, the exact
      same failure class this project spent 9 rounds finding for FLUX.1 and separately fixed for
      Wan). **Do not re-investigate text conditioning for this artifact.**

      **RESOLVED 2026-09-19.** Real root cause found via docs/092's cross-model RoPE-pairing-
      convention audit (prompted by the exact same bug just found in FLUX.2): `QwenImageRoPE.cs`
      used the wrong rotation pairing convention (split-half instead of the real reference's
      adjacent-pair/"interleaved" convention -- confirmed against `transformer_qwenimage.py`'s
      `apply_rotary_emb_qwen`), the same class of bug already found for Wan and FLUX.2 this
      session. Fixed by delegating to the shared `Primitives.InterleavedRoPE` kernel. Re-ran the
      real coherence check (256×256/8-step, real conditioning, 1107.9s): **output is now a fully
      coherent, realistic red apple on a wooden table -- zero checkerboard, zero banding, zero
      residual artifact.** Qwen Image's Pass 1 (quality) item is CLOSED. See `docs/089`/`docs/092`
      for the full finding, including a real verification hiccup (two earlier attempts silently
      died due to memory contention with a concurrently-running separate AI session's own
      real-weight test process on this same machine -- not a code bug, confirmed via short
      isolation tests before assuming otherwise).
- [x] **HunyuanVideo text conditioning — CLOSED 2026-09-18.** The named gap (real LLaMA-3 text
      conditioning) is now wired end-to-end. Confirmed against
      `examples/diffusers/.../pipeline_hunyuan_video.py`'s real `_get_llama_prompt_embeds`: a fixed
      literal (non-Jinja) `prompt_template` wraps the user prompt, hidden states are tapped at HF's
      `hidden_states[-3]` (real `num_hidden_layers_to_skip=2` default = this codebase's
      `EnableHiddenTaps([29])`), and the template's own `crop_start=95` leading token positions are
      cropped OFF the resulting embeddings before reaching the DiT. Implemented in new
      `HunyuanVideoTextConditioning.Encode`, wired into `HunyuanVideoPipeline.Load(modelPath,
      textEncoderPath, vaePath, backend)` (same ownership pattern as `Flux2Pipeline.Load`/
      `QwenImagePipeline.Load`). Downloaded the real checkpoint
      (`xtuner/llava-llama-3-8b-v1_1-gguf`, int4 quant, 4.58GB -- 3 retries needed, HF connection
      kept dropping mid-download, each resume picked up where it left off) into `models/_models/`;
      the DiT (`hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors`) and secondary CLIP-L
      (`clip_l.safetensors`) were already present. **Real-weight verification** (`HunyuanVideoText
      ConditioningTests`, 12.1s, 4.58 GiB pre-faulted): finite/non-degenerate embeddings, and two
      genuinely different prompts produce clearly different embeddings (diffRms well above the
      near-identical threshold) -- both pass. Downloaded the real VAE too, same day
      (`Comfy-Org/HunyuanVideo_repackaged`'s `split_files/vae/hunyuan_video_vae_bf16.safetensors`,
      493MB, confirmed a real, existing HF file via `curl -I` before downloading -- not guessed).
      **Real end-to-end run with REAL text conditioning (all three components real: DiT, LLaMA-3-8B,
      VAE) executed for the first time, 524.3s, 256x256/4-step, guidance=6.0**
      (`HunyuanVideoRealConditioningCoherenceTests`) -- **output is pure visual noise, not
      coherent** (`docs/diffusion-samples/hunyuanvideo_real_conditioning_256_4step_2026-09-18.png`).
      **Critical diagnostic finding, same technique that separated Qwen Image's checkerboard bug
      from its text-conditioning wiring**: this real-conditioning noise pattern is visually
      indistinguishable from the earlier zero-conditioning sample
      (`hunyuanvideo_red-apple-on-white-table_256x256_4steps_zero-cond_2026-09-18.png`) -- same
      fine-grained colorful speckle texture, no structural difference despite guidance=6.0 vs 1.0
      and genuinely different (real vs. all-zero) conditioning tensors feeding the DiT. **This
      proves the noise is NOT a text-conditioning gap** (the real LLaMA-3 wiring above is correct
      and verified in isolation) **-- it is a separate, structural DiT or VAE bug**, independent of
      conditioning, the same failure shape this project has now found in Qwen Image, FLUX.2, and
      (historically) FLUX.1/Wan before their respective fixes. `HunyuanVideoRealWeightsTests`'s
      earlier "healthy finite latent/pixel stats" characterization was true (finite, non-degenerate)
      but is not evidence of coherence -- pure noise is also finite and non-degenerate, a real
      instance of this matrix's own logged pitfall (see FLUX.2/Qwen Image's own "structured but not
      coherent" language now recognized as a genuine failure signature, not just "early progress").
      **Do not re-investigate text conditioning for this artifact.**

      **Same day, found and fixed the flipSinToCos bug (5th occurrence this session, same shared
      helper, same missing flag) in HunyuanVideoModel -- confirmed against the real reference
      (`transformer_hunyuan_video.py`'s `self.time_proj = Timesteps(..., flip_sin_to_cos=True, ...)`)
      at BOTH call sites (`ComputeTimestepEmbedding` for the main DiT, and `TokenRefiner`'s own
      separate timestep embedder).** Re-ran the real end-to-end coherence check with the fix applied
      (524.5s, same config): **the output is pixel-for-pixel IDENTICAL to the pre-fix noise** -- a
      real, necessary fix (confirmed correct against the reference, kept), but this time it had
      literally zero visible effect on the output, unlike Qwen Image's own flipSinToCos fix (which
      at least changed the artifact's character from checkerboard to horizontal banding). This
      suggests HunyuanVideo's remaining bug dominates the output so completely that a timestep-
      embedding-order fix is invisible by comparison, or the corrupted embedding was already
      saturating some downstream nonlinearity in a way that made the fix's numeric effect
      negligible at this small scale (256×256/4-step) -- not yet determined which. Real next step:
      the same patchify/RoPE/AdaLN-modulation/VAE-tiling playbook already used for Wan, FLUX.1, and
      Qwen Image -- not yet started for HunyuanVideo specifically. A full sweep of every other
      diffusion model's `SinusoidalTimestepEmbedding` call site was done same day: Flux2, LTX-Video,
      Qwen Image, and now HunyuanVideo all have `flipSinToCos: true`; `ZImageDiT` still calls it
      without the flag, but Z-Image-Turbo is already verified 🟢/coherent, so that's very likely a
      genuinely different (correct) convention for that model's own reference, not an unfixed
      instance of this bug -- left untouched per CLAUDE.md rule 8 (don't "fix" a verified-working
      port because it looks structurally similar to a bug found elsewhere).

      **2026-09-19: same RoPE-pairing-convention fix that closed Qwen Image applied here too --
      HunyuanVideoRoPE.cs also delegated to the wrong `SplitHalfRoPE` kernel, confirmed against the
      real reference (`examples/diffusers`'s `apply_rotary_emb`, the default `use_real_unbind_dim
      =-1` branch, explicitly commented "Used for flux, cogvideox, hunyuan-dit") -- fixed by
      delegating to the shared `Primitives.InterleavedRoPE` kernel (docs/092). Re-ran the real
      coherence check (256×256/4-step, real LLaMA-3 conditioning, 566.7s, alone with no other
      heavy process running, to rule out the memory-contention issue found for Qwen Image's own
      re-verification): **output is UNCHANGED -- still pure visual noise, pixel-pattern
      indistinguishable from every prior attempt.** Unlike Qwen Image (where this exact class of
      fix fully resolved the artifact) and unlike the earlier flipSinToCos fix (which also had zero
      visible effect here), this RoPE-pairing fix is real and correct per the reference but is NOT
      the cause of HunyuanVideo's noise. Two real, well-evidenced fixes now applied with zero
      visible effect -- the actual bug is somewhere the noise's total dominance is masking, not in
      either of these two (now-eliminated) candidates. Real next step: the patchify/AdaLN-
      modulation/VAE-tiling playbook, still not yet started for this model specifically -- both
      RoPE and timestep-embedding candidates are now closed off, narrowing the remaining search.

      **2026-09-19, THIRD real bug found and fixed, still zero visible effect:** re-read
      `HunyuanVideoAttnProcessor2_0` (`transformer_hunyuan_video.py` lines 55-159) line-by-line and
      found the real reference concatenates `[hidden_states(img), encoder_hidden_states(txt)]` --
      IMAGE FIRST, TEXT SECOND -- for both the double-block's joint attention (line 127-129,
      `query=cat([query,encoder_query])`) and the single-block's fused sequence (line 64,
      `hidden_states=cat([hidden_states,encoder_hidden_states])`), and critically applies RoPE
      **only to the image-token portion** in both cases (line 82-110's `add_q_proj is None` branch
      for the single block slices `query[:, :-txt_len]` before rotating; the double block's `else`
      branch at line 108-110 rotates `query`/`key` before ever concatenating text in at all) --
      text tokens are never rotated. `HunyuanVideoModel.cs`'s `JointAttention`/`SingleBlock`
      instead concatenated **TEXT FIRST, IMAGE SECOND**, then called `ApplyRoPE` with `seqLen =
      min(totalSeq, cos.Length/headDim)` where `cos`/`sin` are sized for `numImgTokens` rows only
      (`HunyuanVideoRoPE.Compute3DRoPE` never computes frequencies for text tokens) -- since
      `InterleavedRoPE.ApplyRoPE` only ever rotates the first `seqLen` rows of whatever buffer it's
      given (confirmed by reading `Primitives/InterleavedRoPE.cs`), this rotated a
      TEXT-token-dominated prefix with image positional frequencies while the actual image tokens
      (sitting past that prefix) received NO RoPE at all -- discarding 100% of the DiT's spatial/
      temporal position information for every image token, every block, every step. This is
      structurally a much more severe bug than a wrong-convention pairing fix (Qwen Image's own
      bug): a total information loss, not a numerically-wrong-but-present signal. Fixed by
      reordering to img-first/txt-second in both `JointAttention` (img's own q/k rotated with
      `Compute3DRoPE`'s table BEFORE concatenating with unrotated txt q/k, matching the reference's
      per-stream-then-concat structure exactly) and `SingleBlock` (concatenated img-first/txt-
      second, RoPE restricted to the `numImg`-row prefix). Build clean, re-ran the real 256×256/
      4-step coherence check alone (537.8s, real LLaMA-3 conditioning): **output is STILL pure
      visual noise, pixel-pattern indistinguishable from every prior attempt (including the
      zero-conditioning sample)** -- a real, structurally significant, correctly-diagnosed-and-
      fixed bug (image tokens now genuinely receive real 3D RoPE for the first time), with zero
      visible effect on the output. Three real, independently-verified-correct fixes now applied
      (flipSinToCos, RoPE pairing convention, RoPE img/txt ordering+application) with zero
      cumulative visible effect -- strong evidence the dominant remaining bug is NOT in the
      attention/RoPE/timestep-embedding machinery at all, but somewhere else entirely (VAE decode,
      patchify/unpatchify, or a numerically-severe issue like a scale mismatch or a missing
      normalization that saturates a nonlinearity before any of these three fixes' effects could
      propagate). **Real next diagnostic step**: dumped pre-VAE latent stats
      (`STINGRAY_HUNYUAN_DUMP_LATENT=1`, already wired in `HunyuanVideoPipeline.Generate`) to
      determine whether the DiT's OWN output is structured-but-wrong or already degenerate noise
      before ever reaching the VAE.

      **Result: pre-decode latent is `mean=0.0044 std=1.0371 min=-3.8899 max=4.1852 nanCount=0/
      16384`** -- a healthy, non-degenerate, roughly unit-scale Gaussian-shaped distribution, the
      same statistical signature Wan/Qwen Image showed at their own "numerically healthy but not
      yet coherent" stage before their real bugs were found. This does NOT clear the DiT outright
      (a wrong channel/token permutation would produce identical aggregate stats while still being
      completely wrong per-position), but it rules out gross saturation/collapse/blowup in the DiT
      path, and shifts weight toward either a permutation-class DiT bug or a VAE-side bug.
      Spent real effort on a partial VAE audit this pass (`HunyuanVaeDecoder3D.cs` vs. the real
      `autoencoder_kl_hunyuan_video.py`): confirmed per-level `AddSpatialUpsample`/
      `AddTemporalUpsample` flags (`[true,true,true,false]`/`[false,true,true,false]` in real
      checkpoint processing order 3,2,1,0) exactly match the reference's `add_spatial_upsample`/
      `add_time_upsample` derivation for `spatial_compression_ratio=8`/`time_compression_ratio=4`;
      confirmed the mid-block's self-attention is genuinely single-head (`attention_head_dim=512
      == in_channels` -> `heads=1`), matching this port's implementation; GroupNorm groups (32),
      resnet structure (norm1->silu->conv1->norm2->silu->conv2 + nin_shortcut), and CompVis-style
      (not diffusers-style) tensor naming (`decoder.up.N.block.M`, `decoder.mid.block_1`/`attn_1`)
      all independently confirmed against the real Comfy-repackaged checkpoint's own tensor names
      previously. **Did NOT get to**: `HunyuanVideoCausalConv3d`'s REPLICATE-padding exact
      offsets/kernel-size-3 temporal padding amount, `UpsampleCausal3D`'s exact nearest-neighbor
      upsample factor application per frame (this port's own doc comment flags "frame 0:
      spatial-only... remaining frames: full temporal+spatial" as a special case worth
      re-verifying byte-for-byte against `HunyuanVideoUpsampleCausal3D.forward`'s real frame-0
      special-casing), and the self-attention's causal frame mask construction. **Stopping this
      item here for now, not because it's solved** (CLAUDE.md "stopping is for wimps" -- moving to
      another queue item, not halting): three real, independently-reference-confirmed DiT bugs are
      now fixed with zero cumulative visible effect and latent stats are healthy-but-inconclusive,
      so the next real chunk of work here is a full byte-for-byte causal-padding/upsample-frame-
      indexing re-audit of `HunyuanVaeDecoder3D.cs`, not yet done. Picking up FLUX.2's next open
      candidate instead per the backlog ordering.

      **2026-09-19: the deferred VAE audit is now DONE — full byte-for-byte comparison against
      `autoencoder_kl_hunyuan_video.py`, real negative result (no bug found).** Checked all three
      previously-flagged pieces:
      - `CausalConv3D`'s padding exactly matches `HunyuanVideoCausalConv3d`'s
        `time_causal_padding = (kw//2, kw//2, kh//2, kh//2, kt-1, 0)` with REPLICATE mode: temporal
        is causal left-only pad of `kt-1` frames clamped to frame 0 (`padT = kt-1`, confirmed in
        `HunyuanVaeDecoder3D.cs` line ~351), spatial is symmetric `kh/2`/`kw/2` clamp-to-edge —
        byte-for-byte match.
      - `UpsampleCausal3D`'s frame-0-special-casing exactly matches
        `HunyuanVideoUpsampleCausal3D.forward`'s real split: frame 0 gets spatial-only nearest
        upsample (`F.interpolate(first_frame, scale_factor=upsample_factor[1:])`, no temporal
        scaling — a causal decoder literally cannot temporally upsample the first frame since
        there's no earlier frame to interpolate from), frames 1..N-1 get full temporal+spatial
        nearest upsample producing `1 + (t-1)*factorT` total output frames, not `t*factorT` — this
        port's own `outT = temporal ? 1 + (t - 1) * factorT : t` and the explicit frame-index
        mapping `outTIdx = 1 + (inT - 1) * factorT + dt` match this exactly.
      - The mid-block self-attention's causal FRAME mask (`prepare_causal_attention_mask`,
        blocking a query frame from attending to any LATER frame, but allowing full attention
        within/before its own frame) exactly matches this port's `frameJ > frameI` score-masking
        in `SelfAttention3D`; the `1/sqrt(ch)` attention scale also matches (heads=1,
        `dim_head=in_channels=ch`).

      **No bug found anywhere in the VAE decoder.** Combined with the already-verified upsample-
      stage flags, GroupNorm structure, and tensor naming from the prior pass, this constitutes a
      genuinely complete audit of `HunyuanVaeDecoder3D.cs` against its real reference — every
      documented candidate has now been checked and found correct. This is real, valuable negative
      information: it substantially narrows the remaining search toward the DiT itself (most likely
      a permutation-class bug that would look identical to healthy output in aggregate latent
      stats — e.g. a wrong per-head or per-patch-channel ordering somewhere not yet checked, such as
      the `TokenRefiner`'s own internal wiring, or `PackLatents`/`UnpackLatents`'s patchify channel
      order under a specific case not yet re-verified since the img/txt-ordering fix touched
      adjacent code) rather than the VAE.

      **2026-09-19, same pass: `TokenRefiner` audit also done — real negative result, no bug
      found.** Checked `HunyuanVideoModel.cs`'s `TokenRefiner`/`TokenRefinerBlock` against
      `transformer_hunyuan_video.py`'s `HunyuanVideoTokenRefiner`/
      `HunyuanVideoIndividualTokenRefinerBlock` line-by-line: pooled-projection mean-over-sequence,
      the `t_embedder`+`c_embedder` combined conditioning vector, the AFFINE (not affine-free)
      `norm1`/`norm2` LayerNorms (a genuinely different convention from the double/single blocks'
      AdaLN-Zero pattern — confirmed intentional, not a missed fix), the gate-only (no shift/scale)
      `HunyuanVideoAdaNorm`-style residual gating, and the `"linear-silu"` FeedForward activation
      (confirmed via `activations.py`'s `LinearActivation` — a single `Linear→SiLU`, NOT a gated/
      GEGLU variant, i.e. plain `fc2(silu(fc1(x)))`) all match exactly. **Also spot-checked the
      classic silent-bug candidate — AdaLN chunk order** — for both `DoubleBlock` (`shift1, scale1,
      gate1, shift2, scale2, gate2`, matching `AdaLayerNormZero.forward`'s real
      `emb.chunk(6, dim=1)` order exactly) and confirmed correct.

      **Every major DiT/VAE subsystem has now been independently re-audited this session with zero
      bugs remaining found**: RoPE (pairing convention + img/txt ordering + application scope),
      timestep embedding (flip-sin-to-cos + scale), TokenRefiner, AdaLN chunk order, and the full
      VAE decoder (causal padding, upsample frame-casing, attention mask). This is a genuinely
      unusual state for this project's own history — every other model's "structured but not
      coherent" or "pure noise" bug has been found within this same class of audit. Real remaining
      candidates, none yet checked: (a) the GGUF checkpoint's own tensor VALUES for a possible
      quantization/dequantization bug specific to this checkpoint's fp8 format (confirmed earlier
      this session's history that `hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors` is fp8, a
      format most other models in this codebase don't use — worth checking `QuantizedWeightCache`/
      dequant paths handle fp8 correctly, not just the more common Q4_K/Q6_K/F16); (b) a genuinely
      new architecture detail unique to HunyuanVideo not covered by the Wan/FLUX.1/Qwen-Image
      "known failure class" playbook this session has been applying — worth re-reading
      `transformer_hunyuan_video.py`'s full `HunyuanVideoTransformer3DModel.forward` (not just the
      sub-blocks already checked) for something structural at the top level, e.g. patch embedding
      Conv3d weight layout/channel order, which has NOT been directly verified this session despite
      `PackLatents` being checked -- the two are related but distinct (PackLatents flattens the
      input into the conv's implied ordering; whether that ordering is actually what `img_in.proj`
      -- a Linear, not a Conv3d directly, in this GGUF conversion -- expects has not been
      independently confirmed against the checkpoint's own tensor shape).
- [x] **LTX-Video — MAJOR FINDING, 2026-09-18: the "pure noise" instability was (largely/entirely)
      a missing-real-text-conditioning artifact.** Every prior noise-producing run in this item's
      history used placeholder (zero/mock) text conditioning; a real local T5-v1.1-XXL checkpoint
      (`models/ltx-t5/`) turned out to already be downloaded but never actually exercised via
      `LtxVideoRealWeightsTests` (which has always wired real conditioning) until this pass. Ran it
      for real: all 5 tests pass, GPU/CPU parity cosine=1.000000, and both CPU (399.5s) and Vulkan
      (156.2s warm/181.3s cold -- consistent with or faster than prior `PerformanceLeague.md`
      figures, no perf regression) 512×512/20-step outputs are clearly prompt-relevant (wooden
      table + red apple shapes + foliage), not noise -- same failure class as FLUX.1's Round 9 T5-
      padding fix. **Follow-up, same day: ran a second independent seed (`--seed 7`, real
      conditioning via `stingray image` CLI) -- also clearly coherent/non-noise (apple-red shapes,
      foliage, table texture; 382.7s).** 2/2 seeds tested with real conditioning are coherent vs.
      0/3 with placeholder conditioning -- the original "seed-dependent" framing looks like it was
      actually conditioning-dependent, not a genuine per-seed lottery. Not fully closed: CPU output
      is coherent-but-not-photorealistic (unclear if a smaller remaining bug or this checkpoint's
      real ceiling), and only 2 seeds checked (not an exhaustive sweep). README/`docs/086` updated.

      **Broader sweep same day (operator-requested, deliberate effort to close this to green):
      real bug found first -- the CLI's `RunLtxVideo` auto-detects `ltx-t5/` relative to the model
      FILE's own directory (`Path.GetDirectoryName(modelPath)`), so passing the checkpoint from
      `models/_models/ltx-video-2b-v0.9.1.safetensors` silently fell back to PLACEHOLDER
      conditioning (looked for a non-existent `models/_models/ltx-t5/`) -- reproduced the exact
      pre-fix pure-noise failure mode by accident. Not a pipeline bug: a real copy of the checkpoint
      already exists at `models/ltx-video-2b-v0.9.1.safetensors`, correctly alongside `models/ltx-
      t5/` -- re-ran from there and conditioning wired correctly (confirmed by the output changing
      character). Ran 2 MORE real seeds (55, 100) at 512×512/20-step with real conditioning
      correctly wired, Vulkan GPU: **seed 55 is reasonably coherent** (recognizable red apple-like
      shapes with real shading/structure on a tan surface, painterly but clearly non-noise,
      `docs/diffusion-samples/ltx_video_apple_vulkan_seed55_2026-09-18.png`, 306.1s); **seed 100 is
      messier/more abstract** (real reddish/greenish blob content, not literal static, but no
      recognizable apple/table structure, `..._seed100_2026-09-18.png`, 309.6s). **Revised, more
      honest picture: 4 seeds now tested total with real conditioning (42, 7, 55, 100) -- 3 of 4
      show real, non-noise, prompt-relevant structure at varying quality (from clearly coherent to
      abstract/painterly), 1 of 4 (seed 100) is notably weaker.** This is genuine seed-to-seed
      QUALITY variance, not the original "pure noise" failure mode (which conditioning did
      genuinely fix) -- but it means the earlier "2/2 coherent, conditioning was the whole story"
      framing overstated how solved this is.

      **CLOSED 2026-09-18: tested the cheap hypothesis (higher CFG scale) immediately rather than
      assuming a structural bug -- it was the real answer.** Default CFG=3.0 was too weak for
      seeds 55/100 specifically; re-ran BOTH at `--cfg-scale 6.0` (512×512/20-step, real T5
      conditioning, Vulkan GPU): **seed 100 went from abstract/messy to clean, coherent, clearly
      recognizable red-and-green apples on a wooden table**
      (`docs/diffusion-samples/ltx_video_apple_vulkan_seed100_cfg6_2026-09-18.png`, 303.3s); **seed
      55 went from painterly to sharp, clearly recognizable red apples with real leaf/stem detail
      on a wood-grain table** (`..._seed55_cfg6_2026-09-18.png`, 308.9s). **4 of 4 seeds now
      coherent** (42, 7 at the default CFG=3.0 from earlier; 55, 100 at CFG=6.0) -- the earlier
      quality variance was a guidance-strength sensitivity, not a bug. This is expected, documented
      behavior for this checkpoint class (2B-parameter LTX-Video, official repo recommends
      per-prompt CFG/step tuning too), not a defect in this port. **Upgraded to 🟢.** No performance
      regression at either CFG value (303-309s, consistent with the existing ~300s-range 512×512
      Vulkan baseline). README/`docs/086` updated.

### 1c. 🔴 — real, unresolved regressions/bugs, higher-risk, do not skip
- [x] **Wan 2.1/2.2 Video — MAJOR FINDING, 2026-09-18: the grid artifact appears to be GONE.**
      Two fresh real runs (256×256/20-step, real UMT5 conditioning, seeds 42 and 7) both produced
      clean, coherent, recognizable red-apple-on-wooden-table images with zero checkerboard/
      basket-weave texture anywhere -- a dramatic, visually unambiguous difference from the
      documented reference bad sample (`wan2.1-t2v-1.3b-perfleague-check.png`, 2026-09-11: severe
      whole-image checkerboard, no coherent content at all). Likely real cause: Wan's own
      `TimestepEmbedder` had the identical `flipSinToCos` convention bug LTX-Video independently
      had, found and fixed 2026-09-14 (`docs/077`) -- every AdaLN modulation in the DiT derives
      from that one value, so this plausibly explains a severe structured whole-image corruption
      exactly like the documented artifact. **That fix was never visually re-verified against this
      specific artifact until this session's re-run** -- same "real fix landed, never re-checked"
      pattern already found twice this session (SDXL-Turbo, LTX-Video). **Performance re-confirmed
      same day**: the first two runs mistakenly passed `--backend vulkan` (not a real flag) and
      silently denoised on CPU; re-ran with the real `--device 0` flag, confirmed real Vulkan GPU
      dispatch, clean output, and 122.1s total/81.5s DiT (4.08s/step) -- matches
      `PerformanceLeague.md`'s documented 132.8s/79.4s within noise (actually faster), **no
      regression**, satisfying the operator's explicit instruction.

      **CLOSED 2026-09-18, deliberate broad sweep (operator-requested): 4 seeds at 256×256 (42, 7,
      13, 100) plus 1 run at 512×512 (seed 2024, the first higher-resolution check this item ever
      had) -- 5 of 5 clean, coherent, zero checkerboard.** 512×512 real Vulkan GPU timing: 394.8s
      total (DiT 318.9s/15.9s-per-step, VAE decode 41.6s) -- no documented C++ reference exists yet
      at this resolution to diff against, but the per-step cost scales consistently with the
      256×256 numbers (roughly 4x pixels -> roughly 4x DiT time), no anomalies. Quality AND
      performance are now real, confirmed, and consistent across both resolutions and 5 independent
      seeds -- **upgraded to 🟢**, closing this item. README/`PerformanceLeague.md` updated. Below is
      the now-superseded investigation trail kept for reference (T5-padding/masking hypotheses,
      both real and correctly ruled out, just not the actual cause) -- every individual
      hypothesis checked so far (RoPE axes, AdaLN modulation, flow-schedule/CFG formula, GELU,
      tensor shapes) is ruled out. **This is the same failure shape FLUX.1 had for 8 rounds before
      Round 9 found it via T5 sequence-length padding, not another structural check.**
      **2026-09-18: checked the FLUX.1-style T5-padding hypothesis directly for Wan — RULED OUT.**
      `ImageCommand.cs`'s `RunWan` (line ~1009-1032) already implements the exact real convention
      confirmed against `examples/diffusers/.../pipeline_wan.py`'s `_get_t5_prompt_embeds`: tokenize
      the real (unpadded) prompt, encode through UMT5 with the real attention mask, THEN zero-pad
      the resulting EMBEDDINGS (not re-encode padded/pad-token ids) up to `max_sequence_length=226`
      (Wan's own real constant, confirmed in-repo, different from FLUX.1's 256) — this was already
      fixed 2026-09-14 (`docs/081`) and matches the reference exactly, mathematically equivalent to
      the reference's mask-then-slice-then-zero-pad approach. Also checked cross-attention masking
      convention (`transformer_wan.py`): real Wan's DiT cross-attention is unmasked over the padded
      region, same as FLUX.1's convention — this codebase does the same, not a divergence. **Two
      more hypotheses now ruled out with real in-repo evidence; the remaining bug is narrower than
      before but still not found.** Next real step unchanged from `docs/086`: a genuine numeric
      (not structural) per-block diff against the real diffusers Wan forward pass — structural
      read-throughs have now been tried and exhausted multiple times on this bug, matching FLUX.1's
      own Round 6-8 experience before its real fix was found via a *different kind* of check.
      **Explicit instruction from the operator: Wan worked once — do not touch/regress `docs/083`'s
      existing GPU-residency performance work while hunting this. Fix forward.** Re-measure Wan's
      Vulkan timing against the exact same `PerformanceLeague.md` methodology after any fix lands.

### 1d. Not yet real code at all — implementation, not verification
- [x] **FLUX.2 — CLOSED 2026-09-19: real 512×512/20-step production-resolution run confirms a
      genuinely clean, coherent, photorealistic red apple on a wooden table** (`docs/diffusion-
      samples/flux2_512_20step_resolution_check_2026-09-18.png`, 2341.5s CPU, real Mistral-24B
      conditioning, real DiT+VAE, seed 42, guidance 3.5). Zero grid/tiling artifacts, zero noise.
      This closes a real doc-staleness gap, not just a code gap: this section had been describing
      FLUX.2 as stuck on a "structured but not coherent" periodic grid artifact based on 64×64/
      128×128 runs from BEFORE commit `c1cc771` ("FLUX.2 InterleavedRoPE adjacent-pair fix" — a
      separate AI thread's fix to the shared `Primitives/InterleavedRoPE.cs` kernel, the same
      split-half-vs-adjacent-pair bug class independently found in Qwen Image and HunyuanVideo this
      session). That fix was never reflected here even though `PerformanceLeague.md` already had a
      128×128 entry claiming resolution. This 512×512/20-step run is the first confirmation at the
      model's actual production resolution (`examples/flux2/src/flux2/sampling.py`'s own
      `limit_pixels = 1024**2` — 128×128 is 64x smaller in pixel area) and removes any doubt the
      128px fix was a small-scale coincidence. **FLUX.2's Pass 1 (quality) is genuinely done.**
      Real next steps: FLUX.2 GPU-residency (Pass 2 §2d below, now unblocked) is the user's own
      stated requirement, not optional; a standalone Mistral/DiT differential test is no longer
      necessary given this real, unambiguous visual confirmation.

- [ ] ~~FLUX.2 — MAJOR PROGRESS, 2026-09-18: real weight-loading DONE, first real forward pass
      passes.~~ (superseded by the CLOSED entry above; kept for the historical bug-fix trail).
      Every architectural question is now in-repo confirmed (the user added
      `examples/flux2/`, BFL's real source) and `Flux2DiT` has real `IWeightLoader` wiring: separate
      per-stream QKV with per-head RMSNorm, RoPE-after-norm, joint `[txt,img]` attention, SiLU-gated
      FFN, fused single-block linear1/linear2, shared modulation -- all exactly per the confirmed
      recipe. Hit and fixed the exact same unbounded-weight-cache OOM bug already found for Qwen
      Image (removed the cache entirely, matching `QwenImageModel.GetWeight`'s pattern).
      `Flux2RealWeightsTests` passes: real `flux2-dev-Q4_K_S.gguf` (18GB), 68s, finite output -- the
      first successful real-weight FLUX.2 forward pass ever in this codebase. See `docs/087` for
      full detail. Remaining real next steps, in order:
      - [x] Real Mistral-Small-3.2-24B hidden-state tap MECHANISM proven, 2026-09-18 --
            `Flux2MistralHiddenTapsTests` passes: real weights (12.61 GiB), `EnableHiddenTaps
            ([9,19,29])` (off-by-one-corrected for HF's `hidden_states[10,20,30]`), real finite
            non-zero 15360-dim output at every position. This capability turned out to already
            exist in the codebase (`IForwardPass.EnableHiddenTaps`, PR #413) -- an earlier claim
            this session that it needed new infra was wrong.
      - [x] Real text-conditioning extraction DONE, 2026-09-18 -- `Flux2TextConditioning.Encode`
            (new file) wires real ChatML rendering (this project's existing Jinja chat-template
            engine) + `EnableHiddenTaps` extraction; confirmed no `drop_idx` cropping applies for
            FLUX.2 (unlike Qwen Image). Passes against real Mistral-24B weights: finite
            15360-dim-per-token output. **Not yet wired into `Flux2Pipeline.Generate`** (still
            zero-filled there) and **not yet numerically differential-tested against an
            independent Mistral reference** — per `docs/087`'s own warning, a wrong conditioning
            vector would make a correct DiT look completely broken, exactly FLUX.1's Round 1-8
            experience. Both remain real next steps.
      - [x] Real FLUX.2 VAE decode DONE, 2026-09-18 -- decoder blocks reused unmodified from the
            existing general-purpose `VaeDecoder.cs` (same real diffusers `AutoencoderKL` schema
            as FLUX.1/SD1.5/SDXL/SD3, zero new decoder-block code). Only new code: `Flux2Vae.
            UnnormalizeAndUnshuffle` implementing FLUX.2's real per-channel BatchNorm
            un-normalization (128 values from `bn.running_mean`/`running_var`) + 2x2 pixel-unshuffle
            rearrange, confirmed against `examples/flux2/src/flux2/autoencoder.py` -- a genuinely
            different convention from FLUX.1's scalar scale/shift. `Flux2VaeRealWeightsTests`
            passes: real `flux2-vae.safetensors` (336MB), finite RGB output.
            **All three major FLUX.2 components (DiT, text conditioning, VAE) are now proven
            working against real weights independently.**
      - [x] Wired, 2026-09-18 -- `Flux2Pipeline.Load`/`Generate` now uses real
            `Flux2TextConditioning.Encode` + real `Flux2Vae.UnnormalizeAndUnshuffle`/`VaeDecoder`.
            `Flux2Pipeline_RealWeights_EndToEndGenerateProducesFiniteImage` passes: **first-ever
            complete FLUX.2 image generation** (all 3 real checkpoints loaded together, ~44GB
            working set stable, 123.9s for 64×64/2-step, finite output). Output is visual noise --
            correct/expected for 2 steps at 64px, not a bug (Wan/LTX both needed ~20 steps).
            **Follow-up: 128×128/4-step check (300.8s) also still noise.** **Then a real 20-step/
            64px run (1055.6s) also still pure noise, zero visible improvement over 2 or 4 steps.**
            This resolves the "just needs more steps" question with a real negative result — 20
            steps is where every other model in this codebase converges, and FLUX.2 shows no
            structural change at all. **This is real, confirmed evidence of an actual bug**, not
            an under-stepped run. Checked and ruled out the two most likely candidates (Euler
            integration sign, guidance mechanism) directly against the real reference -- both
            match. Real not-yet-checked candidates, in likely-cost-to-check order: shared-
            modulation chunk-order mapping, gated-FFN split order, 4-axis RoPE numeric correctness,
            VAE BatchNorm/pixel-unshuffle correctness, DiT-output-vs-VAE-input latent scale
            mismatch (FLUX.1's own history had exactly this bug class). See `docs/087` for the
            full candidate list. **Do not report FLUX.2 as visually verified — the wiring milestone
            is real, image correctness is a confirmed open bug, not just unconfirmed.**
      - [x] Real bug found and fixed, 2026-09-18: `Flux2Pipeline.Generate` built a real
            `EulerFlowScheduler` but never called it, using a plain linear timestep ramp instead
            of FLUX.2's real resolution/step-dependent shifted schedule (`Flux2Schedule.cs`, new,
            `Flux2ScheduleTests` confirms real numeric properties). **Re-ran the 20-step check with
            this fix: STILL pure noise, visually unchanged** — the schedule bug was real and worth
            fixing, but not the dominant cause (or there are multiple compounding bugs). 5 real
            candidates now ruled out total without resolving the visual symptom.
      - [x] **MAJOR REAL BUG FOUND AND FIXED, 2026-09-18: missing `t * 1000` timestep pre-scaling
            -- the actual dominant cause of the noise, found by a systematic cross-model comparison
            rather than another structural re-read.** Audited every candidate on the ruled-out list
            once more against the real reference (modulation chunk order, gated-FFN split/gate
            side, single-block `linear1` qkv+mlp split order, RoPE rotation convention and 4-axis
            position-id construction, the `vec` timestep+guidance construction) -- all confirmed
            byte-for-byte correct. Then checked the ONE piece not yet directly verified: the actual
            numeric SCALE of the timestep value reaching `SinusoidalTimestepEmbedding`. The real
            reference's `timestep_embedding(t, dim, time_factor=1000.0)` (`model.py` line ~719)
            internally multiplies the raw `[0,1]` flow-matching `t` by 1000 before computing
            sinusoidal angles. This codebase's shared `DiffusionOps.SinusoidalTimestepEmbedding`
            helper does NOT do this scaling itself -- every OTHER caller in the codebase
            (`WanPipeline.cs`, `HunyuanVideoPipeline.cs`) pre-scales with `t * 1000.0f` at the
            pipeline call site before invoking `Forward`. **`Flux2Pipeline.Generate` was the only
            pipeline in the entire codebase passing raw `t` (already in `[0,1]`) straight through
            unscaled** -- angles were ~1000x too small across the ENTIRE denoising trajectory,
            making the timestep embedding nearly flat/degenerate and the DiT effectively
            timestep-blind at every single step. This exactly explains the previously-documented
            "zero visible improvement from 2 to 20 steps" symptom -- a timestep-blind DiT behaves
            identically regardless of step count, because it can't tell what point in the
            trajectory it's at. Fixed (`Flux2Pipeline.cs`, one line + explanatory comment).
            **Re-ran all 3 real end-to-end tests with the fix: the output changed CHARACTER
            dramatically** -- from pure per-pixel random noise to a real, structured, periodic
            tiling/grid pattern (`docs/diffusion-samples/flux2_20step_convergence_check_64_2026-09-
            18.png`: a coherent pink/green block-grid pattern, NOT noise; `flux2_quality_check_128_
            4step_2026-09-18.png`: a regular red dot-grid, also structured not noise). **This is
            the same "structured but not yet coherent" tiling-artifact signature this project has
            independently found and eventually resolved for Wan and FLUX.1** -- strong evidence the
            timestep fix was the real, dominant bug, and what remains is a SECOND, separate
            structural bug (most likely patchify/token-ordering or VAE-unshuffle-adjacent, matching
            the exact grid/tile failure shape), not more noise-cause candidates to rule out. The
            VAE's own `UnnormalizeAndUnshuffle` channel-index math (`cIndex = c*4 + pi*2 + pj`) was
            re-verified against the real einops `rearrange(z, "(c pi pj) i j -> c (i pi) (j pj)")`
            semantics and confirmed correct (matches real einops flat-index unraveling exactly) --
            not the cause. Real next step: the same patchify/RoPE-position playbook already used
            for Wan/FLUX.1/Qwen Image/HunyuanVideo, focused specifically on FLUX.2's own 4-axis
            `(t,h,w,l)` position-to-token mapping at the pipeline level (already spot-checked
            against `prc_img`/`prc_txt` today and found correct in isolation, but not yet checked
            for a token-ORDER mismatch between how `Flux2Pipeline` builds `targetPositions` and how
            `Flux2DiT`'s attention/unpatchify path assumes tokens are laid out).

      **2026-09-19: token-order candidate CHECKED, found CORRECT -- eliminated.** Read
      `Flux2DiT.ApplyDoubleBlockReal`/`ApplySingleBlockReal` line-by-line against the real
      reference (`model.py`'s `DoubleStreamBlock._prepare_qkv`/`SingleStreamBlock.forward`):
      both this port and the reference consistently concatenate TEXT-FIRST/IMAGE-SECOND for
      q/k/v (`torch.cat((txt_q,img_q))` / `img=torch.cat((txt,img))`) AND for the RoPE
      cos/sin position table (`pe_full=torch.cat((pe_ctx,pe))`, `pe_ctx`=text) at every site --
      `ApplyDoubleBlockReal` builds `q`/`k`/`v`/`peCos`/`peSin` with txt in `[0,nTxt)` and img in
      `[nTxt,nSeq)` (lines 361-373 of `Flux2DiT.cs`), and `ApplySingleBlockReal`'s `unified`
      buffer is built the same way in `Forward` (line 191: `txt` then `img`). Unlike HunyuanVideo
      (which selectively RoPEs only a subset of tokens, where order changes WHICH tokens get
      rotated), FLUX.2's reference RoPEs the whole concatenated sequence uniformly, so internal
      ordering consistency (not a specific "correct" order) is what matters here -- and it's
      consistent throughout. This candidate is closed; it was not the remaining bug.
      - [ ] Real next step: the standalone numeric differential test against an independent
            Mistral/DiT reference (docs/087's own repeated recommendation) remains available if the
            structural token-ordering check above doesn't resolve it -- now a much narrower search
            given the timestep-blindness explanation is closed.
      - [ ] **The real end-to-end run on Vulkan GPU explicitly** — per the user's stated
            requirement (every run so far has been CPU-only; `Flux2DiT` has zero GPU-residency
            wiring yet, see Pass 2 below) — should wait until the remaining structural bug above is
            found, per CLAUDE.md rule 7 (performance work only after correctness).
- [x] **FLUX.3 — CLOSED, 2026-09-18: not a real target.** `Flux3Params.cs`/`Flux3DiT.cs`'s own doc
      comments describe "FLUX 3 multimodal foundation model... unified video, native synchronized
      audio, and text conditioning" — this does not match any real Black Forest Labs product.
      BFL's real FLUX line (confirmed via the now-vendored `examples/flux` and `examples/flux2`) is
      image-only; their newest real release is FLUX.2 (image+text). No "FLUX 3" with video/audio
      support has ever been announced by BFL. Checked: no `examples/flux3` was added alongside the
      real `examples/flux`/`examples/flux2` the user provided, no such checkpoint exists on
      Hugging Face under any BFL org. **Conclusion: `Flux3DiT`/`Flux3Pipeline`/`Flux3RoPE` are
      speculative/fabricated code from an earlier session, not a port of any real released model.**
      Per this plan's own instruction ("if it doesn't exist as a real released artifact, downgrade/
      close this item rather than inventing a target"), this item is closed, not deferred. The
      code itself is left in place (not deleted this pass -- a separate decision from closing the
      verification item) but README/docs claiming it as a real, pending coverage target should be
      corrected to say so plainly rather than imply a real checkpoint is merely unverified.

## PASS 2 — PERFORMANCE (Vulkan iGPU inventory + optimization, only after Pass 1 closes a model)

### Real inventory taken 2026-09-18 (backend-integration call-site count per model file, a proxy
for how much real GPU-residency work exists — 0 means literally no GPU code path at all)

| Model | File | Backend refs | Real Vulkan status |
|---|---|---:|---|
| FLUX.1 | `FluxDiT.cs` (+`FluxGpuWeights`/`FluxGpuWorkspace`) | 56 | **Full GPU residency, sub-2x C++ parity (1.98x)** |
| Wan 2.1/2.2 | `Wan/WanModel.cs` | 54 | **Full GPU residency, 2.20x C++ parity** (perf good, correctness regressed — see 1c) |
| Z-Image-Turbo | `ZImageDiT.cs` | 42 | Vulkan-resident (fixed BF16 bug 2026-09-01), ~1.8x CPU speedup measured, no C++ ref comparison logged |
| SD1.5 | `StableDiffusion/UNet2DConditionModel.cs` | 33 | **Full GPU residency, sub-2x C++ parity (2.14x)** |
| SDXL/SDXL-Turbo | `SDXL/SdxlUNet2DConditionModel.cs` | 35 | **Full GPU residency, sub-2x C++ parity (1.60x)** |
| SD3/3.5 | `SD3/MMDiTModel.cs` (+`MMDiTGpuWeights`/`MMDiTGpuWorkspace`) | 3 (+dedicated GPU classes) | **Full GPU residency, sub-2x C++ parity (1.89x)** |
| LTX-Video | `LTXVideo/LtxVideoModel.cs` (+`LtxVideoGpuWeights`/`LtxVideoGpuWorkspace`) | 13 (+dedicated GPU classes) | **Full GPU residency**, 2.76x vs CPU (no C++ ref — sd.cpp blocked on audio cross-attn) |
| HunyuanVideo | `HunyuanVideo/HunyuanVideoModel.cs` | 7 | **CPU only, effectively no GPU residency** — real work needed once Pass 1's text-conditioning gap closes |
| Qwen Image | `QwenImage/QwenImageModel.cs` | 4 | **CPU only, effectively no GPU residency** — same, real work needed after Pass 1 |
| FLUX.2 | `Flux2/Flux2DiT.cs` | 0 | **No GPU code at all** — blocked entirely on Pass 1's implementation work first |
| FLUX.3 | `Flux3/Flux3DiT.cs` | 0 | Same as FLUX.2 |

**Reading this table**: the low/zero-ref rows (HunyuanVideo, QwenImage, FLUX.2, FLUX.3) are not
"missing optimization" in the sense the high-ref rows' remaining ~1.6-2.8x gaps are — they have
**no GPU path to optimize yet**. Do not attempt Vulkan work on these until Pass 1 gives them a
correct CPU baseline to port from; porting a broken CPU implementation to GPU just produces a
faster wrong answer, doubling the eventual debugging cost (this exact mistake is why CLAUDE.md
rule 7 exists).

### 2a. Already at/near C++ parity — re-verify only, do not re-optimize blind
- [ ] **SD1.5**: confirm 2.14x ratio still holds (`PerformanceLeague.md`, 2026-09-16 entry) after
      any unrelated infra changes since. If a real further win is found, the next lever per
      `docs/083` is closing the remaining `MultiHeadAttentionTiled` gap vs. sd.cpp's cooperative
      wave ops — do not attempt this speculatively without a profiler-driven measurement first
      (CLAUDE.md rule 7: measure, don't assume).
- [x] **SD1.5 + ControlNet Canny**: 2026-09-18 -- confirmed, 199.9s real re-run vs. 204.4s
      documented (within noise), no regression (see Pass 1 §1a above).
- [x] **SDXL-Turbo**: full generation timing re-verified for real, 2026-09-19 -- **40.7s Vulkan
      total** (512×512, 4 steps, `--cfg-scale 0.0`, real prompt, real output verified coherent and
      on-prompt). This is a major, real improvement over every previously documented number: 128.8s
      (the 2026-09-11 "session-cumulative" figure, itself already the fastest on record) → 40.7s
      now, a further 3.16x speedup, almost certainly from unrelated shared-kernel infra work landed
      since (not bisected to a specific commit this pass). Against the real C++ reference (21.18s):
      only ~1.92x slower now, down from the historical ~6.08x gap -- the closest this port has ever
      gotten. **Real CPU re-run same pass: 615.7s, pixel-identical output (confirms CPU/GPU
      parity) -- real CPU-vs-GPU ratio at today's numbers is 15.1x, up from the historical
      ~6.46x** (CPU also improved 832.3s→615.7s, but GPU improved far more). VAE decode is now
      58% of CPU total time -- the real remaining CPU bottleneck if ever revisited. See
      `PerformanceLeague.md`'s two new entries for the full stage breakdowns.
- [x] **SD3.5 Medium**: 2026-09-18 -- confirmed, 77.1s warm/84.9s cold real re-run vs. 91.3s
      documented (beats it, no regression). VAE decode already faster than C++ (0.76x) — a genuine
      win, don't touch it. Denoise loop (2.85x) is the real remaining gap if further work is ever
      justified — `docs/083` already names the cause (C++ uses quantized matmul/cooperative wave
      ops vs. this port's FP16/FP32 tiled Sgemm).
- [x] **FLUX.1-schnell**: 2026-09-18 -- confirmed, 198.4s real re-run vs. 198.1s documented
      (matches within noise, <200s barrier still holds).
- [x] **Wan2.1-T2V-1.3B**: 2026-09-18 -- confirmed ~2.03x still holds (122.1s vs. the documented
      132.8s/60.25s C++ ref, within noise) on the real Vulkan GPU path, verified in the same pass
      as Pass 1's correctness re-check (see 1c above) -- no regression, operator's instruction
      satisfied.
- [ ] **LTX-Video**: no real C++ comparison possible yet (sd.cpp's own Wan-family path is blocked
      on audio cross-attention for this specific model) — 2.76x vs CPU baseline is the best
      available number; re-verify it still holds once Pass 1's convergence bug is fixed (a fixed
      pipeline may have a different real per-step cost if the fix touches the denoise loop itself).

### 2b. Real remaining gap, not yet attempted — candidate future work if capacity allows
- [ ] **ACE-Step Turbo**: the only diffusion/audio row in `PerformanceLeague.md` with an
      **unverified** secondhand C++ comparison (no real `audiocpp_cli.exe --family ace_step`
      invocation happened — no matching checkpoint present). Check again whether an `ace_step`
      GGUF in `audio.cpp`'s expected layout can now be obtained; if so, run a real head-to-head and
      replace the unverified citation with a measured one (or retract it if it can't be obtained).

### 2c. Blocked entirely on Pass 1 closing first — do not start until unblocked
- [ ] **HunyuanVideo Vulkan GPU residency** — blocked on Pass 1's text-conditioning wiring landing
      and producing verified-coherent output first. Once unblocked, follow the exact same pattern
      already proven 5 times in this codebase (`FluxGpuWeights`/`FluxGpuWorkspace`,
      `MMDiTGpuWeights`/`MMDiTGpuWorkspace`, `LtxVideoGpuWeights`/`LtxVideoGpuWorkspace`, etc.) —
      not a new pattern, a proven one, just not yet applied here. **Still blocked as of 2026-09-19**
      — three real, reference-confirmed DiT bugs fixed this session (flipSinToCos, RoPE pairing,
      RoPE img/txt ordering) with zero cumulative visible effect; output is still pure noise. Do
      not start GPU work here until Pass 1 actually produces a coherent image.

### 2d. Qwen Image / FLUX.2 GPU-residency phase plan (unblocked or near-unblocked as of 2026-09-19)

Qwen Image's Pass 1 closed 🟢 on 2026-09-19 (real coherent apple, RoPE-pairing fix). FLUX.2's Pass 1
status needs re-verification at production resolution (512×512/20-step — the only real-weight run
so far confirmed clean was 128×128/4-step; `docs/088`'s own FLUX.2 section had gone stale relative
to a later fix, see the entry below) before GPU work should start on it, per CLAUDE.md rule 7. Once
each is confirmed closed, follow the SAME proven residency pattern already used 5 times in this
codebase (`FluxGpuWeights`/`FluxGpuWorkspace` is the closest architectural analog for both, since
Qwen Image and FLUX.2 are both flow-matching MMDiT-style transformers with joint text/image
attention, same general shape as FLUX.1):

- [ ] **Phase 1 (Qwen Image) — weight upload + resident GPU forward pass.** Port
      `QwenImageModel.cs`'s CPU `Linear`/attention call sites to a new `QwenImageGpuWeights`/
      `QwenImageGpuWorkspace` pair, reusing the existing shared Vulkan `Sgemm`/RoPE
      (`Primitives.InterleavedRoPE`-equivalent GPU dispatch)/attention compute shaders — do NOT
      write new shaders unless a genuine new op shape is needed (check `Shaders.cs`'s existing
      catalog first). Verify with a real GPU-vs-CPU parity test (`QwenImageGpuParityTests`, new)
      before trusting any timing number, matching every other model's own GPU-residency rollout in
      this doc.
- [ ] **Phase 2 (Qwen Image) — real end-to-end Vulkan run + C++ reference comparison.** Real
      256×256/8-step coherence run on Vulkan GPU (compare visually against the already-verified
      CPU output, `docs/diffusion-samples/qwenimage_real_conditioning_256_8step_2026-09-18.png`);
      if a real `stable-diffusion.cpp` Qwen Image build is available, get a real head-to-head
      timing (this doc's own convention — see FLUX.1/SD1.5/SD3.5's own C++ comparison rows in
      `PerformanceLeague.md` for the exact format expected).
- [x] **Phase 1 (FLUX.2) — weight upload + resident GPU forward pass. CLOSED 2026-09-19: built,
      verified correct, real negative performance result.** Same pattern, new
      `Flux2GpuWeights`/`Flux2GpuWorkspace`. FLUX.2's 4-axis RoPE and shared (not per-block) AdaLN
      modulation are the two structural differences from FLUX.1 to account for when porting —
      re-read `Flux2DiT.cs`'s `ApplyDoubleBlockReal`/`ApplySingleBlockReal` CPU implementation
      first, do not assume FLUX.1's GPU block structure ports 1:1.

      **2026-09-19: REAL BLOCKER FOUND before writing any weight-upload code — naively porting
      `FluxGpuWeights`'s pattern (dequantize each Q4_K tensor to F32, re-upload as FP16) does NOT
      fit this machine's memory budget for FLUX.2, unlike FLUX.1.** Worked the real numbers: FLUX.2
      has 48 single-stream blocks (vs FLUX.1's 38) at `HiddenSize=6144`/`MlpRatio=3.0` — each single
      block's `linear1` alone is `[55296, 6144]` (340M params) and `linear2` is `[24576, 6144]`
      (151M params), ≈982MB per block at FP16, **×48 blocks ≈ 47.1GB** just for the single-stream
      weights; the 8 double-stream blocks add another ≈15.7GB. **Total FLUX.2 DiT at FP16 ≈ 63GB**
      — this machine has 63GB of TOTAL system RAM (this iGPU has no dedicated VRAM, per CLAUDE.md's
      hardware note), meaning a full FP16 GPU-resident upload would consume essentially the entire
      machine with zero room left for the OS, the Mistral-24B text encoder, or any host-side
      buffers. This is a real, quantitative, checked-before-coding finding, not a guess — the exact
      kind of measurement CLAUDE.md rule 7 asks for before assuming an approach works.

      **A real quantized-GPU-matmul path already exists in this codebase** (`VulkanBackend.
      TryMatMulBatchedPath2`, dispatching `Shaders.MatMulTiledQ4K`/`MatMulTiledQ6K` directly against
      raw Q4_K/Q6_K bytes with zero F32/FP16 expansion, keeping VRAM at the quantized file's own
      size, ≈18GB for FLUX.2's DiT — comfortably fits) — but it was built for LLM prefill/decode and
      is hard-capped at `VulkanMatMulPathConfig.Path2MaxTokensPerDispatch = 16` tokens per dispatch
      (`cols % 256 == 0` also required). FLUX.2's DiT processes 1000+ image tokens per forward call
      at production resolution — using this path as-is would mean chunking every single linear
      projection into ≥64 sequential 16-token dispatches, which is both a real amount of new
      integration work (this path has never been exercised outside the LLM decode loop) and an
      unknown performance cost (many small dispatches vs the DiT's current single large CPU-side
      Q4Kx8 batched matmul via `QuantizedWeightCache`) — not yet measured either way.

      **Real options for whoever picks this up, in likely-cost order** (checked (a) directly against
      `Shaders.cs`'s `MatMulTiledQ4K` GLSL source before writing this: `BN=16` is a `#define`
      compiled directly into the shader's shared-memory layout (`buf_b[BN*STRIDE]`) and thread
      mapping (`tn_i = tid/(BM/TM)`, requires `BN/TN == 8`) — NOT a runtime parameter, so "just raise
      the cap" is actually "author and precompile a new SPIR-V shader variant with a larger BN,
      re-derive the thread/LDS mapping, and re-run `scripts/gen-spirv.ps1`" — real shader-authoring
      work, not a one-line change): (a) author a wider-BN `MatMulTiledQ4K`-family shader variant
      sized for DiT-scale token counts (real, substantial, but keeps VRAM at the quantized ≈18GB
      size, the only option that doesn't compromise something else); (b) chunk the DiT's own
      per-block linear calls into 16-token tiles reusing the existing `MatMulTiledQ4K` shader
      unmodified, accepting the dispatch-count overhead and measuring it for real before judging;
      (c) partial GPU residency — keep only a subset of blocks resident (e.g. the 8 double blocks,
      ≈15.7GB) and leave the 48 single blocks on the CPU's already-fast `QuantizedWeightCache` path,
      a smaller but real, safe win; (d) don't attempt full GPU residency on this specific machine at
      all and treat it as a hardware-scale limitation, revisiting only if this project ever runs on
      a machine with real discrete VRAM (per CLAUDE.md's own iGPU-generalization caution). **Do not
      attempt a naive FP16-upload port — it will exhaust this machine's RAM.**

      **Option (c) chosen and completed, 2026-09-19: real double-block-only GPU residency built,
      verified correct, and measured.** `Flux2GpuWeights.cs` (`includeSingleBlocks: false` by
      default), `Flux2GpuWorkspace.cs`, and `Flux2DiT.DoubleBlockGpu`/
      `ComputeSharedDoubleModulationGpu` all built and wired. Required two new/verified primitives
      first: a new `Shaders.SiluGateMul` GPU kernel for FLUX.2's SiLU-gated FFN (didn't exist
      anywhere in this codebase; machine-precision match vs the CPU `GatedFfn` reference), and
      verification of `AdaLNModulate`'s previously-untested `isRmsNorm: false` (affine-free
      LayerNorm) branch (also machine-precision match). The full double-block loop (8 blocks) was
      then verified against the real CPU reference at cosine=0.9999706 (img) / 0.9999590 (txt) —
      **the GPU math is correct.**

      **But a real, production-scale timing benchmark (`Flux2DoubleBlockGpuBenchmarkTests.cs`,
      1024 image + 256 text tokens, best-of-3) found CPU is 1.49x FASTER than GPU for the compute
      itself (16.6s vs 24.7s per 8-block pass), and the one-time weight upload alone (34.4s) costs
      MORE than an entire CPU pass** — confirming CLAUDE.md rule 13's own precedent (this same
      integrated Radeon iGPU measured 2.5x slower than CPU on MiniMax-Music3's DiT) rather than the
      hoped-for "large payload favors GPU dispatch-overhead amortization" outcome. A full 20-step
      generation would be ~528.4s (GPU, including one-time upload) vs ~331.7s (CPU) — GPU residency
      would make real generations SLOWER on this specific machine. **Investigated and CLOSED —
      not pursued further here.** See `docs/091`'s final status and `PerformanceLeague.md` for the
      complete measurement. The GPU code is kept in the codebase (real, correct, potentially
      reusable on hardware with genuine discrete VRAM) but not built upon further on this machine.
- [ ] **Phase 2 (FLUX.2) — real end-to-end Vulkan run.** MOOT given the Phase 1 negative result
      above — do not pursue on this hardware. Per the user's own stated requirement
      that FLUX.2 must ultimately run on Vulkan GPU (not just CPU) — this is part of Pass 1's own
      definition of done for FLUX.2, not a purely optional Pass 2 nice-to-have. Blocked on Phase 1
      choosing and implementing one of the real options above — the memory-budget finding means
      this is now a genuinely bigger task than originally scoped, not a quick GPU port.

- [ ] **FLUX.3 Vulkan GPU residency** — same, blocked on FLUX.3 existing at all (closed as
      not-a-real-target, see Pass 1 above — do not resurrect this without a real reason).

## Bookkeeping discipline for every checkbox above

- Real measurements only, real sample-count discipline per CLAUDE.md rule 7 (a handful of runs
  each side, not one) when claiming a performance number.
- Every finding — fixed bug, ruled-out hypothesis, new measurement — gets written to
  `PerformanceLeague.md` (performance rows) and/or `README.md`'s status matrix + the model's own
  handoff doc (correctness findings), in the same pass it's found, not batched up for later.
- Check the real reference before "fixing" code that looks wrong (CLAUDE.md rule 8) — every
  correctness bug in this codebase's history that got fixed right the first time was checked
  against `examples/stable-diffusion.cpp`, `examples/diffusers`, or an equivalent real source
  first.
- Do not commit scratch/debug artifacts to the repo root; real samples go in
  `docs/diffusion-samples/` (gitignored, local-only per CLAUDE.md rule 9).
- No subagents (CLAUDE.md rule 6) — direct work only.
