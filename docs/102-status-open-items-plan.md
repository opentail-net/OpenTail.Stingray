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
| 6 | ⬜ TODO | PaddleOCR-VL: degenerate output | text architecture (`paddleocr`) has no validated forward pass | medium |
| 7 | ⬜ TODO | DeepSeek-OCR / OCR2: garbled | text architecture (`deepseek2-ocr`) has no validated forward pass | medium |
| 8 | ⬜ TODO | Step3-VL: garbled | unvalidated architecture on a Q2_K checkpoint | medium–large |
| 9 | ⬜ TODO | IBM Granite Vision 3.2 / 4.0: output not image-grounded | investigation; 3.2 via LlavaAdapter, 4.0 via QFormer projector | large |
| 10 | ⬜ TODO | CosyVoice 2: audio only partly right | investigation | large |
| 11 | ⬜ TODO | Stable Audio 3 Small SFX: darker than the reference | investigation | large |
| 12 | ⬜ TODO | Chronos-Bolt / Chronos-2: no numeric reference | needs an independent oracle without new Python reference scripts | large |
| 13 | ⬜ TODO | 🟢-but-⚪ diffusion rows (HunyuanVideo, FLUX.2, Qwen Image, SD3 CPU): not independently verified | needs reference outputs (vendored C++ / recorded) | large |
| 14 | ⬜ TODO | New family: Mamba-2 hybrid layer + state cache, admitting IBM Granite 4.0 (`granitehybrid`, Apache-2.0, 1B-32B) first | missing family; one layer type unlocks #3, #15 and Falcon-H1 | large |
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

