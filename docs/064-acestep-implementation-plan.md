# ACE-Step 1.5 Turbo implementation plan

Status: **scoped and archaeology-complete, 2026-09-03 — implementation not started.** This is a
genuinely much larger architecture than MusicGen/AudioGen (a DiT + flow-matching + VAE stack with
a lyric encoder, a timbre encoder, and an FSQ audio tokenizer/detokenizer, not just a codec-LM
variant), so this doc captures the real, checkpoint-verified architecture before any code is
written — per this project's own rule of checking the real reference before implementing
suspicious-looking math, and per this plan's own recommendation not to start coding the
transformer blind.

## Why this is a bigger scope than MusicGen/AudioGen — read this before estimating effort

MusicGen and AudioGen are both "codec-LM" architectures: a single autoregressive Transformer over
discrete audio tokens, differing only in codec/dims. ACE-Step 1.5 is a hybrid diffusion system with
five real, independently-trained submodules that all need porting for a first working version:

1. **Qwen3 text encoder** (Qwen3-Embedding-0.6B, standalone checkpoint) — real Qwen3 (28 layers,
   GQA, RoPE, RMSNorm, SwiGLU), used purely as a hidden-state feature extractor (all positions fed
   downstream, not a pooled embedding).
2. **Lyric encoder** — a bidirectional (non-causal) transformer, 8 layers, over lyric text hidden
   states (also Qwen3-derived per the real pipeline, though the model file itself takes
   pre-computed `lyric_hidden_states` as input — the actual lyric tokenizer/encoder chain sits
   upstream, likely reusing the same Qwen3 model).
3. **Timbre encoder** — a bidirectional 4-layer transformer that extracts a reference-audio timbre
   embedding via a CLS-style special token, for the "cover" editing mode.
4. **The DiT itself** (`AceStepDiTModel`) — 24 layers alternating sliding-window/full self-attention
   (GQA, RoPE, RMSNorm — same building blocks as the encoders, all built on `transformers`' real
   Qwen3 primitives), AdaLN-style timestep modulation (6-way scale/shift/gate per layer from a
   sinusoidal timestep embedding, PLUS a second embedding for `t - r` meanflow-style
   conditioning), cross-attention to the packed condition sequence (with real encoder-decoder KV
   caching ACROSS DENOISING STEPS — the conditioning never changes between steps, so cross-attn
   K/V is computed once, unusual for a diffusion model and a real, deliberate optimization worth
   replicating), and Conv1d/ConvTranspose1d patchify/depatchify (patch_size=2) at the input/output.
5. **AutoencoderOobleck VAE** (real HF `diffusers` class, same one Stable Audio Open 1.0 uses —
   NOT the same as this project's existing `AcousticVae`, which is Stable Audio **3**'s
   different, bespoke SAME-transformer-resampling autoencoder, see below) — a real conv/Snake1d/
   residual-unit decoder, structurally similar to this project's already-working
   `Parler.DacDecoder`/`Primitives.EncodecDecoderKernels` (weight-normalized convs, residual
   units) but with a two-parameter Snake activation (`alpha` AND `beta`, confirmed from the real
   checkpoint's tensor names — DAC/EnCodec only ever needed a single-parameter Snake/ELU) that
   needs its own real-source-verified formula, not assumed identical to DAC's.

Plus a genuinely separate audio-token subsystem (FSQ `ResidualFSQ` tokenizer/detokenizer, used
for the "cover" mode's LM-hint mechanism) and a 1.7B "5Hz planner LM" that is explicitly a
second-stage convenience layer (auto-metadata/caption/lyric generation from minimal input), not a
prerequisite for basic text-to-music generation — confirmed from the real `generate_audio` method,
which takes already-encoded `text_hidden_states`/`lyric_hidden_states` directly and never touches
the planner LM.

**Estimate**: this is realistically a multi-session port even scoped down to Turbo-only,
text+lyrics-only, no cover/repaint/audio-conditioning. This doc's job is to make the next session's
work be "implement class N against a known-correct spec" rather than "re-derive the architecture."

## Real source obtained (not reconstructed from documentation or memory)

The `ACE-Step/Ace-Step1.5` HF repo bundles the actual reference PyTorch implementation as
`custom_code` (not just weights) — fetched and read directly, not guessed from the community
writeup this plan was originally drafted from:
- `acestep-v15-turbo/configuration_acestep_v15.py` — the real `AceStepConfig` class
- `acestep-v15-turbo/modeling_acestep_v15_turbo.py` — the real ~2140-line reference forward pass
  (`AceStepAttention`, `AceStepEncoderLayer`, `AceStepDiTLayer`, `AceStepDiTModel`,
  `AceStepLyricEncoder`, `AttentionPooler`, `AudioTokenDetokenizer`, `AceStepTimbreEncoder`,
  `AceStepAudioTokenizer`, `AceStepConditionEncoder`, `AceStepConditionGenerationModel`,
  including the real `generate_audio` Euler-ODE sampling loop with its exact hardcoded
  shift-1/2/3 8-step timestep tables)
- Real `config.json` for the turbo transformer, the VAE, the 5Hz LM, and Qwen3-Embedding-0.6B
- Real safetensors tensor headers for both `acestep-v15-turbo/model.safetensors` (677 tensors,
  confirmed to match every module name the modeling code predicts exactly: `decoder.layers.{i}`,
  `encoder.lyric_encoder.*`, `encoder.timbre_encoder.*`, `tokenizer.*`, `detokenizer.*`,
  `null_condition_emb`) and `vae/diffusion_pytorch_model.safetensors` (confirmed real
  `AutoencoderOobleck` naming: `decoder.block.{i}.res_unit{1,2,3}.{conv1,conv2,snake1,snake2}`,
  `decoder.block.{i}.conv_t1`, weight-norm `weight_g`/`weight_v` pairs throughout, BF16 storage —
  this project's `SafetensorsLoader.ReadF32` already handles BF16 upconversion, no new loader work
  needed there)

## Real config values (from the checkpoint's own config.json, not estimated)

**DiT (`acestep-v15-turbo/config.json`)**: `hidden_size=2048, head_dim=128,
num_attention_heads=16, num_key_value_heads=8` (GQA, 2:1 ratio), `intermediate_size=6144,
hidden_act=silu, num_hidden_layers=24, in_channels=192, audio_acoustic_hidden_dim=64,
patch_size=2, rms_norm_eps=1e-6, rope_theta=1000000, sliding_window=128,
use_sliding_window=true, layer_types` alternates `sliding_attention`/`full_attention` starting
with sliding (layer 0 = sliding, layer 1 = full, ...), `text_hidden_dim=1024` (matches
Qwen3-Embedding-0.6B's own `hidden_size`), `num_lyric_encoder_hidden_layers=8,
num_timbre_encoder_hidden_layers=4, num_attention_pooler_hidden_layers=2,
num_audio_decoder_hidden_layers=24, timbre_hidden_dim=64, pool_window_size=5,
fsq_dim=2048, fsq_input_levels=[8,8,8,5,5,5], fsq_input_num_quantizers=1, vocab_size=64003,
timestep_mu=-0.4, timestep_sigma=1.0, is_turbo=true, attention_bias=false`.

**`in_channels=192` derivation** (real, confirmed from `AceStepDiTModel.forward`, not assumed):
the DiT's actual input is `torch.cat([context_latents, hidden_states], dim=-1)` where
`hidden_states` is the noisy latent (`audio_acoustic_hidden_dim=64` channels) and
`context_latents` is itself `torch.cat([src_latents, chunk_masks], dim=-1)` (64+64 channels) —
64+64+64 = 192.

**Qwen3-Embedding-0.6B**: standard Qwen3, `hidden_size=1024, num_hidden_layers=28,
num_attention_heads=16, num_key_value_heads=8, head_dim=128, intermediate_size=3072,
hidden_act=silu, rms_norm_eps=1e-6, rope_theta=1000000`, all layers `full_attention` (no sliding).

**VAE (`vae/config.json`, real `AutoencoderOobleck`)**: `sampling_rate=48000, audio_channels=2`
(stereo), `decoder_channels=128, decoder_input_channels=64, channel_multiples=[1,2,4,8,16],
downsampling_ratios=[2,4,4,6,10]` (product = 1920; combined with the DiT's `patch_size=2`
downsampling this determines the exact samples-per-latent-frame relationship — needs a real
`prepare_condition`/pipeline-level check before assuming 25 Hz falls out automatically, don't
hardcode a frame-rate constant without verifying it against a real decode).

**Turbo inference (`AceStepConditionGenerationModel.generate_audio`, real, not estimated)**: NOT
ordinary CFG — the real Turbo checkpoint's default 8-step schedule uses fixed, hardcoded timestep
tables keyed by an integer "shift" of 1, 2, or 3 (only these three shifts are supported; any other
requested shift or explicit timestep list gets snapped to the nearest valid value from a
20-entry table spanning all three shifts' schedules) — e.g. `shift=3.0` (this checkpoint's
apparent default): `t_schedule = [1.0, 0.9545..., 0.9, 0.8333..., 0.75, 0.6428..., 0.5, 0.3]`
(8 values, no trailing 0). ODE (Euler) update: `x_{t+1} = x_t - v_t * (t_current - t_next)`; final
step instead computes `x0 = x_t - v_t * t_current` directly. An `"sde"` inference variant also
exists (re-noises the predicted clean sample each step) but ODE is the real default.

## What genuinely reuses existing Stingray infrastructure vs. needs new code

| Component | Reuse? | Why |
|---|---|---|
| GQA + RoPE + RMSNorm + SwiGLU attention/FFN math | **Likely reusable at the kernel level** | This is architecturally identical to Qwen3, which this engine's generic LLM `ForwardPass`/GGUF dispatch already runs for text generation — the underlying per-layer math (not the causal-LM-specific KV-cache/generation-loop wiring) should be extractable, though ACE-Step's own encoders are BIDIRECTIONAL (no causal mask) where the existing engine path assumes causal decoding, so this needs verification, not a blind copy |
| Weight-normalized Conv1d/ConvTranspose1d + residual-unit decoder shape | **Directly reusable pattern** | Same shape as `Parler.DacDecoder`/`Primitives.EncodecDecoderKernels`'s already-working `FullConv1d`/`ConvTranspose1dNoPad`/weight-norm-fold helpers — only the Snake activation's exact two-parameter formula and the channel/ratio schedule are new |
| `AcousticVae` (this project's existing Stable Audio 3 VAE) | **NOT reusable** | Confirmed a genuinely different real architecture (Stability AI's bespoke "SAME" transformer-resampling autoencoder) from `AutoencoderOobleck` (a conv/Snake/residual-unit design) despite both being "the VAE for a Stability-adjacent audio diffusion model" — do not assume shared code here without checking, which is exactly what this section did |
| Flow-matching Euler sampling loop | **New, but simple** | Same shape as MusicGen/AudioGen's delay-pattern generation loop in spirit (a real, config-driven stepping loop) — the real hardcoded shift-1/2/3 timestep tables must be transcribed verbatim, not re-derived from a generic flow-matching formula |
| AdaLN scale/shift/gate modulation, sinusoidal timestep embedding | **New** | Not present in MusicGen/AudioGen at all; real formula transcribed above from `TimestepEmbedding`/`AceStepDiTLayer.forward` |
| Cross-attention KV caching ACROSS denoising steps | **New, real optimization worth keeping** | Confirmed real in `generate_audio`'s `past_key_values` threading through the loop — conditioning K/V computed once, reused for all 8 steps, matches this project's "measure, don't assume" performance ethos already (it's free correctness-preserving speed, not a later optimization pass) |
| FSQ (`ResidualFSQ`) tokenizer/detokenizer | **New, deferred** | Only needed for "cover" mode (`is_covers` path) — the real `generate_audio` code shows a plain text-to-music call with `is_covers` all-zero never touches `self.tokenizer`/`self.detokenizer` at all, so V1 can skip this subsystem entirely, not just simplify it |

## V1 scope (matches this plan's own Stage 1 recommendation)

Text prompt (+ optional lyrics) → Qwen3 encoder → condition encoder (text + lyrics only, timbre
encoder gets a zero/empty reference so its packed contribution is trivial — needs verifying
against real `pack_sequences`/`AceStepTimbreEncoder` behavior for an empty reference batch, not
assumed to gracefully no-op) → 24-layer Turbo DiT (8-step Euler ODE, shift=3 default schedule) →
AutoencoderOobleck decode → 48kHz stereo WAV. No planner LM, no cover/repaint/extract/lego/complete,
no reference-audio timbre conditioning (V1 uses the null/silence path for it), no FSQ tokenizer.

## Suggested project layout (matches this plan's own recommendation, adapted to this project's existing per-domain-folder convention rather than a new top-level project)

```
src/OpenTail.Stingray.Diffusion/AceStep/
    AceStepConfig.cs              -- real config constants (this doc's numbers)
    AceStepGenerationParams.cs    -- public generation request (prompt, lyrics, duration, seed, shift, ...)
    AceStepModel.cs               -- weight bundle (DiT + VAE + Qwen3 + condition encoder)
    AceStepPipeline.cs            -- orchestration (encode -> condition -> Euler loop -> VAE decode)
    Text/
        AceStepQwen3TextEncoder.cs   -- bidirectional-vs-causal verified against real reference before writing
    Conditioning/
        AceStepConditionEncoder.cs   -- text_projector + lyric_encoder + timbre_encoder + pack_sequences
    Transformer/
        AceStepDiT.cs                -- the 24-layer DiT (GQA+RoPE+RMSNorm+AdaLN+cross-attn)
    Diffusion/
        AceStepFlowScheduler.cs      -- the real hardcoded shift-1/2/3 timestep tables + Euler update
    Vae/
        AceStepOobleckDecoder.cs     -- real AutoencoderOobleck decoder (Snake1d w/ alpha+beta, residual units)
tests/OpenTail.Stingray.Tests.Diffusion/AceStep/
    (golden tests per this plan's own 10-test ladder, once weights are downloaded)
```

Placed under `OpenTail.Stingray.Diffusion` (not a new top-level project) since this is a
diffusion/DiT model, matching where Stable Audio 3/FLUX/SD3.5/LTX-Video already live — the
audio-domain-specific pieces (Qwen3 text encoder, Oobleck VAE) still get their own
`AceStep/`-scoped files rather than being forced into `OpenTail.Stingray.Audio`'s MusicGen/AudioGen
codec-LM neighborhood, since the underlying mechanism (diffusion, not autoregressive token
generation) is genuinely different.

## Immediate next steps (in order)

1. Download the real weights (`acestep-v15-turbo/model.safetensors` ~4.79GB,
   `vae/diffusion_pytorch_model.safetensors`, `Qwen3-Embedding-0.6B/model.safetensors`) and
   confirm the full 677-tensor turbo inventory (only a partial range was fetched this session) plus
   full VAE inventory against the encoder side too (only decoder tensors were visible in this
   session's partial fetch).
2. Verify the real `AutoencoderOobleck` Snake1d formula from HF `diffusers` source (this
   environment has network + Python access, as demonstrated during the AudioGen investigation) —
   do not assume it matches DAC's single-parameter Snake.
3. Verify whether ACE-Step's Qwen3 text/lyric encoding is causal or bidirectional in the real
   pipeline code that CALLS this model (not visible in `modeling_acestep_v15_turbo.py` alone,
   since it only consumes pre-computed `text_hidden_states`) — check the `Ace-Step1.5` repo's own
   pipeline/inference script if one exists, or the real `transformers`/`diffusers` ACE-Step
   pipeline class once identified.
4. Scaffold the five core classes (`AceStepConfig`, `AceStepGenerationParams`, `AceStepModel`,
   `AceStepPipeline`, weight-loader stubs) with real constants — this session's deliverable.
5. Build outward from there per the golden-test ladder in the original plan (config → tokenizer →
   Qwen3 → condition encoder → one DiT block → full DiT → one scheduler step → full 8-step latent →
   VAE → end-to-end), each with a real golden/non-degeneracy check before moving on, matching how
   MusicGen/AudioGen were verified.
