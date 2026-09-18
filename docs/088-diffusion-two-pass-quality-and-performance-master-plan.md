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
- [ ] **Qwen Image / Qwen Image Edit** — DiT+VAE verified, single named gap: real LLM text
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
      Qwen3-ForcedAligner's M-RoPE-degenerates-to-1D-for-text-only case. Ready for implementation.
- [ ] **HunyuanVideo** — same shape of gap as Qwen Image: DiT+VAE numerically sound, single named
      gap is real LLaMA-3/Qwen2.5-VL text conditioning (`HunyuanVideoPipeline.Generate` currently
      defaults to all-zero context). **2026-09-18 scoping**: confirmed against
      `examples/diffusers/.../pipeline_hunyuan_video.py`'s real `_get_llama_prompt_embeds` --
      HunyuanVideo needs a real `LlamaModel` (specifically `xtuner/llava-llama-3-8b-v1_1-
      transformers`, an 8B decoder LLM) as PRIMARY text encoder plus a secondary `CLIPTextModel`
      (same secondary-CLIP-pooled-conditioning pattern as FLUX.1). Real subtlety confirmed in the
      reference worth getting right on the first attempt (same discipline as FLUX.2's system-
      message/layer-extraction recipe): a fixed `prompt_template` wraps the user prompt before
      encoding, and `crop_start` (computed by tokenizing the template alone first) crops the
      template's own tokens back OFF the final embeddings before they reach the DiT -- get this
      crop-after-encode convention right or conditioning will be silently offset. Neither the
      8B LLaMA checkpoint nor a CLIP-L is currently downloaded for this pipeline -- real,
      substantial (~16GB+) download needed before this item can proceed; not started this pass to
      avoid contending with the SDXL-Turbo download / Qwen Image's own pending redownload already
      queued.
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
      regression**, satisfying the operator's explicit instruction. Not fully closed: only 2 seeds
      checked, only at 256×256 (not production resolutions) -- but quality AND performance are now
      both real, confirmed, and consistent. README/`PerformanceLeague.md` updated. Below is
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
- [ ] **FLUX.2 — MAJOR PROGRESS, 2026-09-18: real weight-loading DONE, first real forward pass
      passes.** Every architectural question is now in-repo confirmed (the user added
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
      - [ ] Real next step: the standalone numeric differential test against an independent
            Mistral/DiT reference (docs/087's own repeated recommendation) — structural re-reading
            plus one real fix have both been tried without success, matching FLUX.1's own Round 6-8
            pattern before its real fix was found by a numeric, not structural, check.
      - [ ] **The real end-to-end run on Vulkan GPU explicitly** — per the user's stated
            requirement (every run so far has been CPU-only; `Flux2DiT` has zero GPU-residency
            wiring yet, see Pass 2 below) — should wait until the correctness bug above is found,
            per CLAUDE.md rule 7 (performance work only after correctness).
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
- [ ] **SDXL-Turbo**: only load/init re-verified this session (real weights load correctly after
      the redownload + clip_tokenizer.json fix), NOT full generation timing -- the 1.60x claim
      itself is unconfirmed this pass, narrower claim only.
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
      not a new pattern, a proven one, just not yet applied here.
- [ ] **Qwen Image Vulkan GPU residency** — same, blocked on Pass 1's text-conditioning wiring.
- [ ] **FLUX.2 Vulkan GPU residency** — blocked on Pass 1's entire implementation landing first
      (the user's own stated requirement is that FLUX.2 must ultimately run on Vulkan GPU, not
      just CPU — so this isn't purely a "later" item, it's baked into Pass 1's own definition of
      done for FLUX.2, listed here too so it isn't lost track of).
- [ ] **FLUX.3 Vulkan GPU residency** — same, blocked on FLUX.3 existing at all.

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
