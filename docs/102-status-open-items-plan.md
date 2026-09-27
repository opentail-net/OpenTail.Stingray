# STATUS open items — fixable on this PC (plan, 2026-09-26)

The STATUS status-matrix rows that are not fully green, minus what this machine cannot fix (no
checkpoint, too large for 64 GB RAM, upstream GGUF bug): DeepSeek-V3.2, DeepSeek-V4, Llama 4
vision, MobileNetV5 and MiMo-VL are out of scope.

Order: quickest expected fix first. The estimates are guesses; the log below records what
actually happened. Every fix follows CLAUDE.md: real weights, an independent reference (llama.cpp
/ vendored C++ / recorded reference outputs), timed test runs, and the STATUS row updated with the
dated evidence in the same pass.

| # | Status | Item | Why it is where it is | Expected size |
|---|---|---|---|---|
| 1 | ✅ DONE | SD3/3.5 (GPU/Vulkan) row: Status/Confidence columns swapped | table edit | minutes |
| 2 | ↪ merged into #11 | Stable Audio 3 Small Music + Medium still 🟡 though the rows say they match the reference | re-grade from the existing evidence, or name the open gap | minutes |
| 3 | ⏸ RE-SCOPED (Mamba-2 port, large) | Nemotron-Nano-12B-v2-VL: `nemotron_h` text backbone crashes (per-layer `feed_forward_length` array; 0 = pure Mamba layer) | root cause already known | small |
| 4 | ✅ DONE | dots.ocr: decode stops after one token | suspected prompt-format mismatch | small–medium |
| 5 | ✅ DONE (2026-09-27) | Kimi-VL + YoutuVL: split `attn_k_b` / `attn_v_b` MLA layout not read | known math (llama.cpp deepseek2 absorption path); unblocks two models | medium |
| 6 | ✅ DONE (2026-09-27) | PaddleOCR-VL: degenerate output | text architecture (`paddleocr`) has no validated forward pass | medium |
| 7 | ✅ DONE (2026-09-27; OCR v1 untested, no checkpoint) | DeepSeek-OCR / OCR2: garbled | text architecture (`deepseek2-ocr`) has no validated forward pass | medium |
| 8 | ✅ DONE (2026-09-27) | Step3-VL: garbled | unvalidated architecture on a Q2_K checkpoint | medium–large |
| 9 | ✅ DONE (2026-09-27) | IBM Granite Vision 3.2 / 4.0: output not image-grounded | investigation; 3.2 via LlavaAdapter, 4.0 via QFormer projector | large |
| 10 | ⛔ BLOCKED (needs an upstream CosyVoice2 reference) | CosyVoice 2: audio only partly right | investigation | large |
| 11 | ✅ DONE (2026-09-27) | Stable Audio 3 Small SFX: darker than the reference | investigation | large |
| 12 | ✅ DONE (2026-09-27) | Chronos-Bolt / Chronos-2: no numeric reference | needs an independent oracle without new Python reference scripts | large |
| 13 | 🟡 MOSTLY DONE (SD3, FLUX.2, Qwen Image verified with automated sd.cpp tests; HunyuanVideo BLOCKED: no independent v1 reference) | 🟢-but-⚪ diffusion rows (HunyuanVideo, FLUX.2, Qwen Image, SD3 CPU): not independently verified | needs reference outputs (vendored C++ / recorded) | large |
| 14 | ✅ DONE (CPU; granite-4.0-h 350M/1B admitted 2026-09-27) | New family: Mamba-2 hybrid layer + state cache, admitting IBM Granite 4.0 (`granitehybrid`, Apache-2.0, 1B-32B) first | missing family; one layer type unlocks #3, #15 and Falcon-H1 | large |
| 15 | ⬜ TODO | New family: NVIDIA Nemotron Nano v2 / Nemotron 3 Nano (`nemotron_h`) | missing family; reuses #14's Mamba-2 layer; also completes #3 | medium after #14 |
| 16 | ⬜ TODO | New family: GLM-4.5 / 4.6 / 4.7 incl. Air (`glm4moe`) | missing family, currently a top open family; GLM-4 dense already runs. Air is ~60 GB at Q4 | medium |
| 17 | ⬜ TODO | New family: Liquid LFM2 / LFM2-MoE (`lfm2`, 350M-8B, popular on-device) | missing family; skipped earlier because the LFM licence caps free commercial use at $10M revenue. **Decide the licence question first** | medium |

## Current state (paused 2026-09-26)

Done: #1, #4 (#2 merged into #11). In progress: #5. Re-scoped: #3. Not started: #6–#13.

#5 detail. Uncommitted-then-committed code:
- ModelGraph reads `key_length_mla` / `value_length_mla` / `q_lora_rank` / `expert_gating_func`;
  absorbed-MLA GGUFs get NumKvHeads = NumHeads.
- ForwardPass rebuilds `attn_kv_b` from `attn_k_b` / `attn_v_b` and adds the Q LoRA path
  (`attn_q_a` -> norm -> `attn_q_b`).
- `RouteExperts` adds DeepSeek-V3 sigmoid routing with the `exp_probs_b` selection bias.
- The GPU MLA pass is only used for the legacy layout.

Results so far: Kimi-VL text PPL 115.6 vs llama.cpp 114.1 (matches); first-token logprob " Paris"
-1.25 vs -1.20. Youtu-VL text: the 5.4% gap (14.02 vs 13.30) is confined to one passage and is
not systematic (2026-09-27; all runs use a 2048-token context unless noted):

| Scored window | Stingray | llama.cpp | Gap |
|---|---|---|---|
| wiki.test.raw [512,1024), 1024-token context | 7.2705 | 7.2839 | -0.2% |
| wiki.test.raw [1024,2048) | 13.97 (batched) / 14.02 | 13.30 (13.2990 with f32 KV too) | +5.0% / +5.4% |
| wiki.test.raw from byte 400000, [512,1024), 1024-token context | 25.70 | 26.24 | -2.1% |
| wiki.test.raw from byte 400000, [1024,2048) | 21.54 | 21.36 | +0.8% |

What this rules out:
- The KV cache precision: llama.cpp gives 13.2990 with an f32 or an f16 cache.
- The total context size: in our run, positions [512,1024) score the same whether the context
  is 1024 or 2048 tokens.
- A position-dependent (RoPE) bug: the second passage shows no growth with position.
- A broken Q LoRA path, which is position- and text-independent and would show everywhere.

The one +5% window is most likely a few outlier tokens. The earlier explanation (Q8_0 activation
quantization, F16 KV) does not fit: the KV part is disproven above, and the same effects give
0.4-1.3% on DeepSeek-V2-Lite and Kimi. To close it fully, diff per-token logprobs over
wiki.test.raw [1024,2048) against llama-server. Verdict: Youtu-VL text is close to llama.cpp,
with one unexplained window.
Vision end-to-end: Kimi Q2_K reads the OCR test image wrongly on BOTH llama.cpp and ours (checkpoint
too weak), so it is not a useful test.

## Tests for another model to run (hand-off)

Read CLAUDE.md first. Rules:
- Run test exes directly with the fully qualified `-class`.
- Heavy tests need `STINGRAY_RUN_HEAVY_TESTS=1`.
- Check the timing: a pass in ~0.1 s means the model was absent and nothing ran.
- Models live in `models/_models` (-> F:\_models).
- Oracles: `tools/llama.cpp/llama-server.exe` (use `--no-jinja`), `llama-perplexity.exe`,
  `llama-mtmd-cli.exe` / `llama-mtmd-debug.exe`.

1. Regression suites (must stay green):
   `dotnet test tests/OpenTail.Stingray.Tests.Core`,
   `tests/OpenTail.Stingray.Tests.Server.Fast` and `tests/OpenTail.Stingray.Tests.ForwardPass.Fast`
   (never pass `--nologo`).
   Vision project: `tests/OpenTail.Stingray.Tests.Vision/bin/Release/net10.0/OpenTail.Stingray.Tests.Vision.exe`
   (154 tests).
   Vulkan: `-class OpenTail.Stingray.Tests.Vulkan.VulkanArchLogitParityTests`,
   `VulkanLayerSplitParityTests`, `GptOssGpuParityTests`, `DeepSeek2GpuParityTests`,
   `VulkanRowOffsetMatVecTests` (all with STINGRAY_RUN_HEAVY_TESTS=1).
2. DeepSeek-V2-Lite regression from #5's changes (legacy MLA must be unchanged): wikitext
   second-half PPL, `stingray perplexity -m models/_models/DeepSeek-V2-Lite-Chat.Q2_K.gguf -f
   scripts/kvarn-gate/wiki.test.raw -c 2048` ([1024,+) bucket) vs `llama-perplexity ... -c 2048
   --chunks 1`. Before #5 it was 32.82 vs 32.68.
3. #5 Youtu-VL: find the 5.4% PPL gap. Same PPL commands on `youtu-vl-4b-Q8_0.gguf`. Then
   teacher-forced logprobs vs llama-server on a short prompt.
4. #5 vision: `llama-mtmd-debug.exe -m <text gguf> --mmproj <mmproj> -p encode -n 224 --image cb`
   vs our encoder on the same checkerboard (pattern: tests/OpenTail.Stingray.Tests.Vision/
   DotsocrVisionEmbedderParityTests.cs), for mmproj-youtuvl-4b-q8_0 and mmproj-kimivl-q8_0.
   Also Exaone4 / MiMo-VL encoders (the ApplyMRoPE section-reset follow-up from #4).
5. README recipes (docs-as-tests, manual for now): run every command/snippet in README.md from an
   empty folder with fresh downloads. They were verified 2026-09-26 on source; the quick start also
   against NuGet 1.0.7 plus Microsoft.Extensions.Logging.Abstractions.

## New-family items (#14-#17), added 2026-09-27

These are popular checkpoint families with no implementation at all. The popularity judgement is
qualitative, not from download statistics. Lower-priority missing families, not planned:
- AI21 Jamba (SSM hybrid) and Falcon-H1 (comes almost free with #14);
- Kimi Linear, RWKV-7, Arcee AFM, ServiceNow Apriel, Ant Ling (`bailingmoe2`);
- MiniMax-M2 (too large for this audience).

Each item is verified the usual way: `stingray admit-arch` against llama.cpp reference tokens,
second-half perplexity vs `llama-perplexity`, then an allowlist entry, a STATUS.md row and a
MODELS.md entry if it qualifies.

## Log

- 2026-09-27 #14 DONE (CPU): Mamba-2 hybrid layer; IBM Granite 4.0-H (`granitehybrid`) admitted.
  - **Mixer:** `ForwardPass.Mamba2.cs` follows llama.cpp `build_mamba2_layer` and the ggml CPU
    `ssm_conv`/`ssm_scan`:
    - in_proj splits into z / xBC / dt;
    - causal depthwise conv over `conv_kernel` taps, bias, SiLU;
    - `softplus(dt + dt_bias)` and `dA = exp(dt*A)`;
    - state `s = s*dA + B*x*dt`, `y = C.s + D*x`, then `silu(z)*y`;
    - grouped RMSNorm, then out_proj.
  - **State:** per recurrent layer, a conv state `[convDim][k-1]` and an SSM state
    `[heads][headDim][dState]`. Cleared by `ResetCache`; it cannot be rewound.
  - **Layer selection:** a layer is recurrent iff its `attention.head_count_kv` entry is 0.
    `NumKvHeads` = the max non-zero entry; `GetInt` used to take layer 0's 0.
  - **NoPE:** `rope.scaling.finetuned=false` gives NoPE on every attention layer (via
    `NoRopeLayerStep=1`).
  - **Excluded from the GDN probe:** `granitehybrid` is kept out of the GDN hybrid path that the
    `_opentailllm.is_hybrid_ssm` probe would otherwise select.
  - **Evidence**, wikitext second-half [1024,+) PPL (-c 2048) vs `llama-perplexity --chunks 1`:

    | Model | Ours | llama.cpp |
    |---|---|---|
    | 350M Q8_0 | 17.9003 | 17.9258 |
    | 1B Q8_0 | 8.7639 | 8.7563 |

    Tokenisation identical. `GraniteHybridGreedyParityTests` passes 4/4 with real weights (7.8 s):
    - 350M: " Paris." + EOS.
    - 1B short prompt: 16/16 tokens (EOS masked like `ignore_eos`).
    - 1B longer prompt: 15 tokens, then a documented near-tie (" equivalence" -1.288 vs
      " general" -1.352).
    - State reset.
  - **Perf pass, same day** (1B Q8_0, CLI, 40-token prompt + 128 decode, 3 runs each):
    - Scan heads now run in parallel.
    - The d_state loop is vectorised in ggml's AVX2 order: 4x8 accumulators, unfused
      `s*dA + B*x*dt`, FMA into the sums, ggml's reduce.
    - Decode 17.7 -> 21.6 tok/s (+22%); prefill 14.4 -> 17.1 tok/s. About 48 ms/token vs about
      52 ms to stream 1.56 GB at roughly 30 GB/s, so decode is near the bandwidth limit.
  - **Order sensitivity, measured:** second-half PPL moves about 0.3% with the scan's summation
    order alone.

    | Scan variant | 1B | 350M |
    |---|---|---|
    | Scalar | 8.7639 | 17.9003 |
    | Threaded scalar | 8.7639 (identical) | not run |
    | Plain SIMD | 8.7891 | 17.8688 |
    | ggml order | 8.7833 | 17.9578 |
    | llama.cpp (thread-count invariant) | 8.7563 | 17.9258 |

    So "matches llama.cpp" here means within about 0.3%, not closer; the committed numbers are
    the ggml-order ones.
  - **Regression:** `Tests.ForwardPass.Fast` 686/686.
    `GraniteGreedyParityTests`/`GraniteMoeGreedyParityTests` skipped: their checkpoints aren't
    local. Their code path only gains the `granitehybrid` family term.
  - **Known limits:**
    - CPU only.
    - Prefill runs token by token (1B 17 tok/s; llama.cpp batches it: about 110 tok/s scoring).
      Closing that gap means Mamba-2 support in the batched `PrefillCore`, a separate, sizeable
      item, not part of this perf pass.
    - Recurrent layers still append a zero KV row to satisfy `PagedKvCache` (wasted memory).
    - No partial rewind of the Mamba state: `SupportsPartialRewind` is false, so `InferenceEngine`
      disables prefix caching; `TruncateTo` resets at 0 and throws for any other earlier length.
    - MoE Granite-H (tiny/small) untested.

- 2026-09-27 #13 part 2: Qwen Image had a missing `txt_norm`, now fixed and verified against `sd-cli`.
  - **Bug:** neither the CPU nor the GPU path applied `txt_norm`, the RMSNorm (eps 1e-6) over the
    Qwen2.5-VL conditioning before `txt_in` (`qwen_image.hpp forward_orig`). The weight is in the
    checkpoint but was never read.
  - **Noise floor:** `sd-cli` defaults to Vulkan in this build (`SD_VULKAN=ON`). Its own CPU and
    Vulkan backends differ, so the comparison was run against both with `--backend cpu|vulkan0`.
    - One step, no guidance: sd.cpp CPU vs Vulkan is cosine 0.9984, relative L2 6.6%.
    - 8 steps, guidance 4: cosine 0.9957, 9.6%.
  - **Results:** 256², `--scheduler simple` (sd.cpp's default `discrete` spaces the steps
    differently), same injected noise, seed 42.

    | Case | Before the fix | After the fix |
    |---|---|---|
    | 1 step, no guidance, same conditioning, vs sd.cpp CPU | 0.9933 (11.9%) | **0.99968 (2.5%)** |
    | 8 steps, guidance 4, final latent, vs sd.cpp CPU | 0.981 (20%) | **0.99908 (4.3%)** |
    | Our Vulkan path, 8 steps, vs sd.cpp CPU | not measured | 0.9936 |
    | Our Vulkan path, 8 steps, vs sd.cpp Vulkan | not measured | 0.9961 |

    Both Vulkan results are within sd.cpp's own CPU-vs-Vulkan gap.
  - **Ruled out:**
    - **Text conditioning:** cosine 0.998 against `SD_DUMP_COND_PATH`, and feeding in the
      reference conditioning changes nothing.
    - **Activation rounding:** routing through the int8-activation path moves our output 0.18%.
  - **New env var:** `STINGRAY_QWENIMAGE_DUMP_COND_PATH`.
  - **Automated test:** `QwenImageSdCppParityTests` asserts cosine > 0.999 against the committed
    sd.cpp CPU fixture (`TestData/QwenImageSdCppGolden`: noise, conditioning, 1-step latent, 300 KB).
    Measured 0.999676 in a 16 s real-weight run.
  - **Regression:** of the 14 Qwen Image test classes, 13 passed with real weights.
    `QwenImageGpuParityTests` fell to 0.9883 because its synthetic context (uniform [0, 0.1]) is
    rescaled to unit RMS by `txt_norm`. It now uses the fixture's real noise and conditioning:
    GPU vs CPU 0.99655, passes.
  - **Caveat for part 1:** the SD3.5 and FLUX.2 references were most likely `sd-cli`'s default
    Vulkan backend too; their cosines are against that.
  - **Next:** HunyuanVideo.

- 2026-09-27 #13 FLUX.2 now has an automated reference test.
  - **Why:** part 1 marked it 🔬 from a manual comparison only.
  - **Test:** `Flux2SdCppParityTests` runs `Generate` with real Mistral conditioning and the DiT,
    one step, 256², guidance 3.5, with the committed noise injected. It compares against the
    `sd-cli --backend cpu` one-step latent in `TestData/Flux2SdCppGolden` (256 KB).
  - **Result:** cosine 0.999962, relative L2 0.9%, norm ratio 1.0014. A real 148 s run (9.7 GiB
    of weights pre-faulted).
  - **Fixture:** sd.cpp took 3m25s to generate it.
  - **Landscape sweep:** the heavy test sweep started at 09:16 was killed by Claude Code's
    low-memory reaper at 11:08. The Diffusion suite had logged no failures up to then; Audio,
    Vision and ForwardPass never ran. Not restarted.

- 2026-09-27 #13 SD3.5 Medium now has an automated reference test too.
  - **Test:** `Sd3SdCppParityTests` runs the real pipeline (CLIP-L/G + T5-XXL + MMDiT), one step,
    256², CFG 4.5, empty negative prompt, committed noise injected. It compares against a
    `sd-cli --backend cpu` fixture in `TestData/Sd3SdCppGolden` (128 KB; sd.cpp took 30 s).
  - **Results:** 29 s CPU run and 46 s for both variants, real weights.

    | Path | Cosine | Rel. L2 | Norm ratio |
    |---|---|---|---|
    | CPU | 0.998754 | 5.0% | 0.9961 |
    | Vulkan | 0.998767 | not recorded | 0.9956 |

  - **Threshold:** 0.998 for both. The residual is the small encoder differences (CLIP-G 0.996,
    T5 0.998) amplified by CFG.
  - **Where the aux files live:** in `models/sd35-medium-aux` and `models/flux1-schnell`, not
    `models/_models`.

- 2026-09-27 #13 HunyuanVideo: BLOCKED on numeric parity. No independent runnable v1 reference.
  - **Local sd.cpp patch** (`examples/stable-diffusion.cpp`, git-ignored, redo from this list):
    - `name_conversion.cpp`: single-block `q_norm`/`k_norm` mapped to `norm.query_norm`/`key_norm.scale`.
    - `hunyuan.hpp detect_from_weights`: v1 raw names for the head count and `qkv_bias`. It also
      resolves the heads after the loop, because the loop could see `key_norm` before `img_in` and
      divide the 2048 default by 128.
    - `SD_HUNYUAN_V1`: latent 16 channels at 8x (not 1.5's 32 at 16x), and `c_vector` feeds `y` /
      `vector_in` only, not 1.5's vision-token slot.
    - `SD_DIT_ONLY`: skip text-encoder/VAE tensor validation.
    - `SD_INJECT_COND_PATH` / `SD_INJECT_VEC_PATH`: conditioning from files, in both the image and
      video paths.
    - Noise and latent dumps in the video path.
    - Run with `-M vid_gen --video-frames 1 --guidance 6000` (v1 embeds guidance x1000; sd.cpp's
      time factor is 1).
    - With all that, sd.cpp detects v1 correctly (24 heads, 20+40 blocks, qkv bias, ctx 4096,
      vec 768, guidance) and runs.
  - **Result:** same injected noise and our own LLaMA-3 + CLIP-L conditioning
    (`STINGRAY_HUNYUAN_INJECT_NOISE_PATH` / `_DUMP_COND_PATH` / `_DUMP_VEC_PATH`, new).
    - One-step velocity cosine is only 0.187, not a layout permutation.
    - sd.cpp's x0 is noise-like: std 1.5, 0.52 correlated with the input noise.
    - At 8 steps, 256², our VAE decodes sd.cpp's latent to pure colour noise, while ours on the same
      noise is a clean red apple.
  - **Conclusion:** the v1.5-only sd.cpp is missing something v1 needs. Candidates: fp8 e4m3fn
    handling, token-refiner details, timestep conventions. Making it v1-correct would mean writing
    v1 into the reference ourselves, which destroys its independence.
  - **Status:** the row stays ⚪ with visual-only evidence.
  - **To unblock:** an independent v1 reference output. Either a ComfyUI/diffusers HunyuanVideo run
    recorded as data (noise + latent), like #10, or a C++ port that supports v1 upstream.

- 2026-09-27 #13 SD3.5 Medium now has an automated reference test too.
  - **Test:** `Sd3SdCppParityTests` runs the real pipeline (CLIP-L/G + T5-XXL + MMDiT), one step,
    256², CFG 4.5, empty negative prompt, committed noise injected. It compares against a
    `sd-cli --backend cpu` fixture in `TestData/Sd3SdCppGolden` (128 KB; sd.cpp took 30 s).
  - **Results:** 29 s CPU run and 46 s for both variants, real weights.

    | Path | Cosine | Rel. L2 | Norm ratio |
    |---|---|---|---|
    | CPU | 0.998754 | 5.0% | 0.9961 |
    | Vulkan | 0.998767 | not recorded | 0.9956 |

  - **Threshold:** 0.998 for both. The residual is the small encoder differences (CLIP-G 0.996,
    T5 0.998) amplified by CFG.
  - **Where the aux files live:** in `models/sd35-medium-aux` and `models/flux1-schnell`, not
    `models/_models`.

- 2026-09-27 #13 HunyuanVideo plan (not started; waiting for the landscape sweep to free RAM).
  - **Can sd.cpp run v1?** Possibly. STATUS says the vendored sd.cpp is HunyuanVideo 1.5 only, but
    `hunyuan.hpp HunyuanVideoConfig::detect_from_weights` reads the config from the checkpoint
    itself: depth, single blocks, `vector_in` dim, `guidance_in`, heads, MLP ratio and patch size.
    So it may run our v1 DiT (`hunyuan_video_720_cfgdistill_fp8_e4m3fn.safetensors`). Untested.
  - **Text conditioning won't match:** sd.cpp's conditioning is `LLMEmbedder`, the v1.5 recipe. It
    can't produce v1's LLaMA-3 `hidden_states[-3]` plus CLIP-L pooled `c_vector`.
  - **Method:** a DiT-level parity check.
    - Patch sd.cpp locally to inject `c_crossattn` / `c_vector` from files.
    - Feed it our own `HunyuanVideoTextConditioning` output.
    - Inject the same noise on both sides.
    - Compare 1-step velocity at guidance 1, then a short full run, on `--backend cpu` and
      `vulkan0` for the noise floor.
  - **If sd.cpp can't load the v1 checkpoint:** document it as blocked (no runnable v1 reference)
    and move on to #14.

- 2026-09-27 #13 part 1: latent-level parity against `examples/stable-diffusion.cpp` (`sd-cli`).
  - The reference is patched locally (git-ignored): `SD_DUMP_NOISE_PATH` (existing),
    `SD_DUMP_LATENT_PATH` (x_0 before VAE decode) and `SD_DUMP_COND_PATH` (cross-attention
    conditioning).
  - Ours: the shared `DiffusionParityHooks` plus per-model environment variables
    (`STINGRAY_SD3_INJECT_NOISE_PATH` / `_DUMP_LATENT_PATH`, `STINGRAY_FLUX2_INJECT_NOISE_PATH` /
    `_DUMP_LATENT_PATH` / `_DUMP_COND_PATH`).
  - SD3.5 Medium: verified; CPU 0.9971, Vulkan 0.9930.
  - FLUX.2: guidance embedding x1000 fix, system-message line break, 512-row zero padding.
    1-step 0.99986, 4-step 0.9982 (was 0.962).
  - Next: Qwen Image, HunyuanVideo.

- 2026-09-27 #12 DONE: both Chronos models now have an independent numeric oracle, community ONNX
  exports run through ONNX Runtime from C# (no Python).
  - Chronos-2 matches `OpenSTEF/chronos-2-onnx`: 2.5e-4 abs on values up to 45.7; multivariate
    group 1.7e-4.
  - The Chronos-Bolt encoder matches `light-curve/chronos-bolt-small`: 1.0e-6.
  - Downloads live in `models/_models/hf/OpenSTEF__chronos-2-onnx` and
    `models/_models/hf/light-curve__chronos-bolt-small`.
  - Note: the Chronos-2 export fixes the horizon at 42 x 16 = 672, so the test predicts 672 steps.

- 2026-09-27 #11 DONE: Stable Audio 3 (all three checkpoints) matches the vendored audio.cpp reference
  latent for latent.
  - Method: `examples/audio.cpp` `src/models/stable_audio/rf_dit.cpp` patched locally (git-ignored)
    to write `noise`, `cross`, `global`, `schedule`, `padding`, `local`, `tfeat0`, `out0` and `final`
    to `$SA3_DUMP_DIR`. Rebuilt with `ninja audiocpp_cli` in `examples/audio.cpp/build`, then run
    from `examples/audio.cpp`.
  - Ours takes the same noise via `STINGRAY_SA3_NOISE` and dumps via `STINGRAY_SA3_DUMP`
    (`StableAudioDebugHooks`). Note: the reference latent is channel-major, ours token-major.
  - Root cause: missing per-layer `to_local_embed`. Upstream `model.py` always passes zero
    inpaint conditioning, so a constant `W2 silu(b1) + b2` is added to latent tokens.
  - Results:
    - step-0 velocity cosine 0.9996 -> 0.999998 (the norm deficit of 4% is gone);
    - Small Music final 0.986 -> 0.9993 (CPU) / 0.9991 (Vulkan);
    - Medium 0.99994; SFX CFG 7 / 50 steps 0.9978.
    - SFX brightness now matches (HF 0.852 vs 0.858).
  - Also: V zeroing for the reference's padded token (valid = eff + floor(6 s x rate)), and GPU
    QK-norm eps 1e-6.
  - The component golden `StableAudioDiTGoldenParityTests` was recorded with `local_add_cond=None`
    and now runs with `IncludeLocalConditioning = false`.
  - Remaining small gap: the reference masks padded tokens inside the APG norm at CFG > 1; we
    don't (CFG-7 final 0.9978).

- 2026-09-27 #10 BLOCKED: CosyVoice2's garbled endings need an independent reference that is not
  available here.
  - Every vendored C++ port supports CosyVoice 3 only: `examples/cosyvoice.cpp`, `audio.cpp`
    (`include/engine/models/cosyvoice3`), CrispASR (`cosyvoice3-tts`).
  - New Python reference scripts are ruled out by project policy.
  - Already ruled out (docs/audio-review-new-progress.md, 2026-09-25):
    - the acoustic stack (resynthesis is almost exact);
    - the LLM forward pass (teacher forcing ranks real tokens near the top, with no decay by
      position);
    - the repetition penalty.
  - What would unblock it: one recorded run of upstream CosyVoice2-0.5B `inference_zero_shot`
    (made on any machine and checked in as data, not as a script), for the fixed test text,
    prompt wav and seed, giving:
    - the LLM input token ids (sos/eos, text, task_id, prompt speech tokens);
    - the generated speech-token sequence;
    - ideally the per-step top-k log-probs.
  - Diffing our generation against that pinpoints the step where the two diverge.

- 2026-09-27 #9 DONE: Granite Vision 4.0 matches llama.cpp.
  - Prompt of 473 tokens, same answer.
  - Encoder: per-patch positions, and the 8 QFormer streams concatenated per token.
  - Deepstack: added to the CPU `ForwardPass`; the Vulkan/CUDA passes are a follow-up.
  - Preprocessing: `llava_uhd` views, newline tokens, `<image>` prefix.
  - Tests: Vision 170 / 0 failed; ForwardPass.Fast 685 / 0 failed.

- 2026-09-27 #9 part 1: Granite Vision 3.2 fixed and matches llama-server token for token.
  - Fixes: feature-layer stack, anyres `llava_uhd` slicing, the model's own normalisation, no
    markers, and Granite `embedding_scale` on raw embeddings.
  - The `llava_uhd` path and the LLaVA feature-layer rule also apply to other LLaVA-style models;
    with no local LLaVA-1.5/1.6 checkpoint, those were not re-verified.
  - Part 2, Granite 4.0 Vision, is larger:
    - its text model uses deepstack (`granite.deepstack_mapping`): 8 x 2560 features per image
      token, 7 of them added into text layers;
    - its QFormer projector must emit all 8;
    - its `granite` preprocessor adds newline rows.
    - llama-server's own answer on the invoice is weak (":

The total is 38. EUR."), so the
      target there is prompt and embedding parity rather than a perfect reading.

- 2026-09-27 #8 DONE: Step3-VL encoder rewritten against `step3vl.cpp`.
  - Fused qkv, RoPE, layer scales, QuickGELU, conv downsamplers. Parity: sum 3957.5 vs 3959.4.
  - The slicing preprocessor and segmented image input (`IVisionEmbedder.LastSegments`) were
    ported too. The prompt equals llama-server's (520 tokens) and the model reads the invoice.
  - **Performance pass (flagged as extra cost, CLAUDE.md rule 11).** The first correct run took
    10m13s per image, because every vision linear was one weight pass per token.
    - Added `PackedSgemmF32.CanGemmStreaming` (F16 through the existing `GemmQuant` panel kernel).
      `VisionOps.MatVecAny` takes the GEMM path at >= 16 tokens: 10m13s -> 4m25s.
    - Added `VisionOps.AttentionGemm` (QK^T and PV as GEMMs) at >= 256 tokens: -> 2m54s-3m03s.
    - Step3 448-px encoder alone: 64 s -> 13.4 s.
    - GEMM activations are rounded the way ggml does (Q8_0 per-32 blocks with an F16 scale; F16).
      Without that, FP32 activations moved Kimi's sum by 3%; with it, Kimi matches llama.cpp more
      closely than before (-707.42 vs -707.47).
    - `STINGRAY_VISION_GEMM=0|nomm|noattn` switches the paths off for A/B checks.
  - dots.ocr: a new non-degenerate rainbow parity test shows layers match through 22, then drift
    in the very large late-layer activations in every variant. Tolerances were set from that
    measured drift; end-to-end OCR is unchanged.
  - Re-verified end to end after the change: PaddleOCR, Qwen2.5-VL, dots.ocr and DeepSeek-OCR2
    still read the invoice identically. The Vision suite has 168 tests, 0 failed.
  - Remaining cost: the text prefill of image tokens is per-token (~9 t/s here). A batched
    embedding prefill is a separate engine item.

- 2026-09-27 #7 DONE: DeepSeek-OCR2 768-px tiles added (bicubic position table, linear
  relative-position tables, 144 queries, llama.cpp grid choice).
  - Tile SAM output matches `llama-mtmd-debug`.
  - A 1400x900 page gives the same 1126-token prompt and text as llama-server.
  - DeepSeek-OCR v1 stays on the old encoder: no local checkpoint to verify it.

- 2026-09-27 #7: DeepSeek-OCR2 reads text and matches llama-server on the test invoice.
  - This needed a new encoder, larger than the "medium" estimate. The old one had no SAM stage.
    Ported: SAM ViT-B, neck, Qwen2 query encoder, projection.
  - Also: `deepseek2-ocr` admitted (NeoX), Pillow-bicubic fit-and-pad preprocessing, BOS for
    templates that omit it, and a plain-text close marker.
  - SAM is verified per block against `llama-mtmd-debug`. That tool segfaults in the Qwen2 stage,
    so that stage is verified end to end only.
  - Remaining for #7: 768-px tiles for pages larger than 768 px. This needs bicubic resizing of the
    SAM position table, linear interpolation of the relative-position tables, 144-query tiles, and
    grid selection. DeepSeek-OCR v1 has no local checkpoint.

- 2026-09-27 #6 DONE: PaddleOCR-VL reads text; OCR output is identical to llama-server.
  - Encoder, 4 bugs: LayerNorm with bias (it used RMSNorm), vision M-RoPE with section reset,
    antialiased position-table resize, projector eps.
  - Text decoder: NeoX rotation, and new 2D M-RoPE image positions (`ForwardPass.AddMRopeImage`,
    CPU `ForwardPass` only). Admitted to the allowlist.
  - Tokenizer: SPM byte-fallback tokens now decode to bytes.
  - The same session found Qwen2.5-VL was NOT reading images, despite its 🟢 row: its encoder had
    the dots.ocr no-section-reset RoPE bug. Fixed; its answer now equals llama-server's.
  - New parity tests: `LlamaMtmdVisionParityTests` (Kimi x2, Youtu, PaddleOCR, Qwen2.5-VL).
  - Known gaps:
    - CUDA/Vulkan forward passes and the server's image path (`InferenceEngine`, Gemma-style
      placeholders only) do not apply the 2D M-RoPE positions yet.
    - Qwen3-VL needs IMROPE plus the `qwen3vl` text architecture (not admitted).
  - The #4 follow-up is done: Exaone 4.5 and MiMo-VL encoders had the same missing section reset.
    Fixed and pinned: rainbow448 sums 1409.85 vs 1411.85 and -8360.5 vs -8317.8.
    - Every `GGML_ROPE_TYPE_VISION` encoder in `src/OpenTail.Stingray.Vision` now passes
      `independentSections: true`.

- 2026-09-27 #5 DONE: Kimi-VL and Youtu-VL vision now match llama.cpp.
  - `llama-mtmd-debug` stage fingerprints (checkerboard 224 and rainbow 448) found five encoder bugs.
    - Youtu: patch flatten order (HWC), missing window attention, and a preprocessor resize rule
      that gave 234 instead of 320 tokens.
    - Kimi: the 2D RoPE layout, and the position table indexed raw instead of resized with
      antialiased bilinear.
  - Pinned by `KimiYoutuVisionEmbedderParityTests` (3 tests, 29 s, real weights). The Vision suite
    has 157 tests, 0 failed.
  - End to end on `test-1.jpeg` against `llama-server`: prompts are identical (Youtu 348 tokens,
    Kimi 416).
    - Youtu reads the NYT moon-landing front page, including the date, like llama.cpp.
    - The Kimi Q2_K checkpoint misreads the page on both engines.
  - A checkerboard cannot test Youtu: every 16x16 patch is identical, and llama.cpp's own
    Q8_0/BF16 mmproj disagree by 3% on it. Use the rainbow pattern.
  - `llama-mtmd-cli.exe` now exits 127 with no output on any generate call (its `--help` works);
    `llama-server` with `--mmproj` was used as the end-to-end oracle instead.
- 2026-09-26 #1 DONE: SD3/3.5 (GPU/Vulkan) row had Status and Confidence swapped; now
  🟢 (closed 2026-09-24) / ⚪, matching the CPU row.
- 2026-09-26 #2 RE-SCOPED, not a re-grade: Stable Audio 3 Small Music and Medium are 🟡 for a real,
  named gap. At CFG 1 (conditional-only) ours is much darker than the reference; at the official
  CFG 7 they match. Item 11 (Small SFX "darker than the reference") is the same symptom, so #2 and
  #11 become one investigation, placed with the large items after #10.
- 2026-09-26 #3 RE-SCOPED (moved after #8): Nemotron-Nano's `nemotron_h` text backbone is a
  Mamba-2 hybrid. Every layer is Mamba-2 (`ssm_in` [5120->22656], `ssm_conv1d` k=4, per-head
  `ssm_a`/`ssm_d`/`ssm_dt.bias`, grouped `ssm_norm` [1280x8], `ssm_out`), an MLP-only block
  (`ffn_up`/`ffn_down`, no gate) or attention. This codebase has no Mamba-2 implementation at all.
  The "per-layer feed_forward_length" crash is only the first missing piece, so this is a new
  architecture port (reference: llama.cpp nemotron-h.cpp / build_mamba2_layer), not a small fix.
- 2026-09-26 #4 DONE dots.ocr. It was not a prompt-format problem: the vision encoder was wrong in
  five places, plus the image markers, all diffed against dotsocr.cpp / clip.cpp and
  `llama-mtmd-debug -p encode --image cb` stage fingerprints.
  - Fused `attn_qkv` never read: it looked for attn_q/k/v, which don't exist.
  - SwiGLU `ffn_gate` ignored: GELU was used.
  - Post-norm is `mm.post_norm`: it looked for v.post_ln.
  - Vision M-RoPE lacked ggml's per-section angle reset: `Qcur_pos-0` sum +622 vs ref -657.
  - Projector GELU is erf, not tanh.
  - Image markers are `<|img|>`/`<|endofimg|>` (from the llama-mtmd-cli log), not
    `<|vision_start|>`/`<|vision_end|>`.
  - After: projector sum 15843.43 vs ref 15843.74. The OCR test image reads "Invoice 4217 / Total:
    38.50 EUR", identical to llama-mtmd-cli. Vision test project 154 tests green.
  - FOLLOW-UP: Exaone4, MiMo-VL and YoutuVL also call VisionOps.ApplyMRoPE without the section reset
    (now an opt-in flag). Check each against llama-mtmd-debug before switching them.
- 2026-09-27 #5 Youtu-VL text PPL gap: not systematic. It is confined to one 1024-token window
  (+5%). The other windows are -2.1% to +0.8% (see the table under "Current state"), llama.cpp is
  unchanged with an f32 KV cache, and a Q LoRA or RoPE fault is ruled out. The regression suites
  and the DeepSeek-V2-Lite check are green (run by the hand-off model, 2026-09-26; timings show
  real weights ran). Next: the vision encoder checkerboard check (hand-off test 4).

