# ACE-Step 1.5 Turbo implementation plan

Status: **V1 end-to-end working, 2026-09-03** — real weights for all four components (Qwen3 text
encoder, condition encoder, DiT, Oobleck VAE decoder) wired through a real flow-matching Euler-ODE
scheduler and `AceStepPipeline.Generate()`; `AceStepPipelineEndToEndTests` produces finite,
non-silent 48kHz stereo audio from a real text prompt on the first real end-to-end run. See the
"Phase E progress" section below for what's still open (numeric golden-parity, the missing real
`silence_latent` buffer, audible quality assessment) before calling this production-ready.

Status (superseded, kept for history): **scoped and archaeology-complete, 2026-09-03 —
implementation not started.** This is a
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
        AceStepQwen3TextEncoder.cs   -- CAUSAL Qwen3 (confirmed against real diffusers pipeline, see below) for the text prompt; lyrics use only an embedding lookup, no transformer
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

## Corrections and confirmations from the real `diffusers` ACE-Step pipeline (found after this doc's first draft)

`diffusers` 0.40.0 (already installed in this environment) ships the complete official ACE-Step
pipeline (`diffusers.pipelines.ace_step.pipeline_ace_step`, 1295 lines, plus
`diffusers.models.transformers.ace_step_transformer`) — read directly, resolving every open
question the first draft of this doc flagged:

1. **Real correction to this plan's original metadata assumption**: BPM/keyscale/timesignature are
   **NOT** projected as separate structured tensors by a dedicated encoder. They are templated
   directly into the TEXT PROMPT STRING and encoded through the same Qwen3 text encoder as
   ordinary text, via a real fixed template (`pipeline_ace_step.py`'s `SFT_GEN_PROMPT`/
   `_build_metadata_string`):
   ```
   # Instruction
   {instruction}            (default: "Fill the audio semantic mask based on the given conditions:")

   # Caption
   {prompt}

   # Metas
   - bpm: {bpm or "N/A"}
   - timesignature: {timesignature or "N/A"}
   - keyscale: {keyscale or "N/A"}
   - duration: {int(audio_duration)} seconds
   <|endoftext|>
   ```
   Lyrics use a SEPARATE, simpler template (not this one): `# Languages\n{vocal_language}\n\n#
   Lyric\n{lyrics}<|endoftext|>`. Do not build an `AceStepMetadata` class with its own tensor
   projection as the original plan draft suggested -- it's plain string formatting upstream of the
   tokenizer.
2. **Text vs. lyric encoding are genuinely different real paths, confirmed** (resolves this doc's
   original open question #3): the formatted TEXT string runs through the FULL Qwen3 model
   (`self.text_encoder(input_ids=...).last_hidden_state`, standard CAUSAL decoder masking, all 28
   layers) to produce `text_hidden_states`. The formatted LYRIC string only runs through Qwen3's
   `get_input_embeddings()` (a plain token-embedding lookup, no transformer layers at all) --
   `AceStepLyricEncoder`'s own 8 bidirectional layers (already read from
   `modeling_acestep_v15_turbo.py`) are what actually contextualizes the lyric embeddings. Do not
   run the lyric text through the full Qwen3 model.
3. **Turbo genuinely has no inference-time CFG, confirmed** (not an assumption): `do_
   classifier_free_guidance` returns `False` whenever `is_turbo=True` regardless of the requested
   `guidance_scale`, and the pipeline forces `guidance_scale=1.0` with a warning if a caller passes
   one for a turbo checkpoint. Confirms V1 needs only a single conditional forward pass per step,
   no null-condition branch at all for the Turbo path (the real `null_condition_emb` tensor exists
   in the checkpoint for the base/SFT variants' real CFG, out of scope for V1).
4. **Real potential shortcut worth checking before writing a new Qwen3 encoder from scratch**:
   `"qwen3"` is already an admitted GGUF architecture in this engine's existing generic LLM
   `ForwardPass` (`ModelCompatibility.cs`), meaning real GQA/RoPE/RMSNorm/SwiGLU kernels for this
   exact architecture already exist and are proven (used for real Qwen3 text generation). Two real
   open questions before relying on this: (a) does the existing `ForwardPass` API expose raw
   per-token hidden states (needed here) rather than only next-token logits -- a diffusion
   conditioning use case is a genuinely new consumption pattern for that engine path, not
   necessarily already supported; (b) does a GGUF conversion of `Qwen3-Embedding-0.6B` specifically
   exist/need producing, since the real checkpoint here is a stock safetensors release. If both
   check out, this could replace a large fraction of the "write a new Qwen3 encoder" work in the
   original layout below with "wire the existing engine into this pipeline as a hidden-state
   source" -- a materially smaller task. If not, `Text/AceStepQwen3TextEncoder.cs`'s CAUSAL
   (not bidirectional) attention now has a confirmed-correct real spec to implement against
   directly, matching this project's existing per-domain-encoder convention.

## Phase B progress, 2026-09-03: Qwen3 text encoder working (real GGUF, existing engine reused)

The Qwen3-reuse shortcut (flagged above as needing either a GGUF conversion or a
`SafetensorsTextModelPackage` extension) is now DONE via the simpler path: the official
`Qwen/Qwen3-Embedding-0.6B-GGUF` quant exists on HF (no converter needed — real llama.cpp
`convert_hf_to_gguf.py` vendored in this repo doesn't even have Qwen3 support, an older copy; the
official quant made that moot). `stingray list-metadata` confirmed `general.architecture=qwen3`
with every dim matching this doc's captured `Qwen3-Embedding-0.6B/config.json` exactly.

`src/OpenTail.Stingray.Diffusion/AceStep/Text/AceStepQwen3TextEncoder.cs` now wraps the existing
`Engine.ForwardPass` (GGUF-based, real qwen3 kernels this engine already runs for text generation)
using `EnableHiddenTaps`/`HiddenTapsAt` (a pre-existing mechanism, built for DSpark draft-model
conditioning, not written this session) to extract per-token hidden states, plus one small new
piece of real math: `EnableHiddenTaps` captures a layer's PRE-final-norm output (HF's
`hidden_states[i+1]` convention), but real `Qwen3Model.forward().last_hidden_state` is
POST-final-norm — so this class applies the model's own `output_norm.weight` RMSNorm itself to
each tapped row. Real test:
`tests/OpenTail.Stingray.Tests.Diffusion/AceStep/AceStepQwen3TextEncoderTests.cs`, passing against
real weights with the real SFT-formatted prompt template.

### Real bug found in the shared engine (not ACE-Step-specific), worked around

While testing the real SFT-formatted prompt, `Engine.ForwardPass.Prefill` produced NaN logits at
position 12 of a real 13-token sequence
(`[2,29051,198,14449,279,7699,41733,6911,3118,389,279,2661,4682]`, using the f16 quant). Localized
precisely via the existing `STINGRAY_TRACE_NORMS=1` diagnostic (not written this session): every
layer 0-26's residual-stream L2 norm stayed finite and grew normally (reaching ~600-800 by layer
26), and layer 27 (the model's LAST transformer layer) alone turned the output to NaN for this
specific position/token/context combination — reproduced identically via both the sequential
`Forward` and batched `Prefill` capture paths, and confirmed NOT specific to
`EnableHiddenTaps`/this session's new code (a raw `Prefill` call with no taps enabled at all
reproduces it too). Cross-checked against the Q8_0 quant of the exact same checkpoint on the
identical token sequence: **no NaN** — isolating this to the f16 weight-storage/kernel path
specifically (plausibly an F16 dynamic-range overflow given the real, legitimately large
activation norms by that layer, though not fully root-caused to a specific line of kernel code).

**Practical resolution for ACE-Step**: `AceStepQwen3TextEncoderTests` uses the Q8_0 quant, which
does not reproduce the bug and is smaller/faster anyway. **Real, separate follow-up**: the f16
qwen3 path having a genuine NaN-producing bug in shared, heavily-used engine code is worth its own
investigation independent of ACE-Step — flagged in `docs/00-current-work.md`, not silently
worked around without a record.

## Phase F progress, 2026-09-03: Oobleck VAE decoder working (real weights, out of order per the plan's own Phase F slot -- done early since it was already fully specified)

`src/OpenTail.Stingray.Diffusion/AceStep/Vae/AceStepOobleckDecoder.cs`: real `AutoencoderOobleck`
decoder, transcribed directly from `diffusers` 0.40.0 source, tensor names/shapes confirmed
against the real checkpoint (`vae/diffusion_pytorch_model.safetensors`, downloaded in full this
session, 337MB). Self-contained rather than force-reusing `Primitives.EncodecDecoderKernels`
(genuinely different Snake activation formula -- see below). Real test
(`AceStepOobleckDecoderTests`) passes against real weights: correct output shape, finite,
non-silent PCM from a synthetic latent.

**Independently confirmed the 25Hz/48kHz claim mathematically**, not just cited from
documentation: real hop length = `product(downsampling_ratios)` = `2*4*4*6*10 = 1920`;
`48000/1920 = 25` exactly. The test asserts both numbers directly from
`AceStepConfig`'s real config values.

**Real decoder channel progression note** (a correction to this plan's own earlier assumption):
the channel-per-stage sequence is `2048 -> 1024 -> 512 -> 256 -> 128 -> 128` -- NOT a clean
halving at every stage (the last stage keeps 128->128) -- derived from the real
`channel_multiples=[1,2,4,8,16]` config list the same way the real `OobleckDecoder.__init__`
computes each block's `input_dim`/`output_dim`, not assumed to follow
`Primitives.EncodecDecoderWeights.DefaultChannelsPerStage`'s simple doubling pattern (that helper
does not apply here and was correctly NOT reused for this reason).

## Phase D progress, 2026-09-03: the 24-layer DiT working (real weights, biggest remaining piece)

`src/OpenTail.Stingray.Diffusion/AceStep/Transformer/AceStepDiT.cs` (forward pass) +
`AceStepDiTWeights.cs` (loader): the full real `AceStepDiTModel` -- GQA self-attention
(alternating sliding-window/full, both BIDIRECTIONAL not causal), per-head Q/K RMSNorm before
RoPE, cross-attention (no RoPE, no gate on its residual -- both real, easy-to-miss facts, see the
class doc comment), Qwen3-style SwiGLU MLP, 6-way AdaLN modulation from a shared per-step timestep
embedding, and the `proj_in`/`proj_out` Conv1d/ConvTranspose1d patchify/de-patchify. Cross-attention
K/V is precomputed once via `PrepareCrossAttention` and reused across steps (matches the real
reference's own `EncoderDecoderCache` optimization, and the same shape as MusicGen/AudioGen's
`PrepareCrossAttention` pattern in this codebase, applied to a bidirectional DiT instead of an AR
LM). Downloaded the full real `acestep-v15-turbo/model.safetensors` (4.79GB) this session.

`AceStepDiTTests` passes on the FIRST real-weight run: correct shapes end to end
(`ProjIn -> Forward -> ProjOut`), finite non-degenerate output, AND -- the more meaningful check --
different timesteps produce measurably different output, confirming the AdaLN timestep-conditioning
path is genuinely wired (a common silent-bug shape: dropping the timestep embedding would still
produce finite, shape-correct, but timestep-INSENSITIVE output, which this test would have caught).
No real condition encoder exists yet, so this test drives cross-attention with a synthetic
(real-shaped) condition sequence -- sufficient to validate the DiT's own weight loading and math,
not yet an end-to-end golden check.

**Not yet done**: numeric golden-parity against a real `diffusers`/`AceStepTransformer1DModel`
reference run (no reference dump script written this session, unlike AudioGen's real
`audiocraft`-based cross-check) -- this DiT's correctness rests on careful reading of the real
source (documented inline) plus real-weight non-degeneracy, not a numeric diff yet. Given the real
`diffusers` package is right there in this environment, a real reference dump is a realistic,
worthwhile next step before trusting output QUALITY (as opposed to "it runs").

Three of the four V1 components are now real and tested (text encoder, DiT, VAE decoder); only the
condition encoder (packs text+lyric+timbre into the DiT's cross-attention sequence) and the
flow-matching Euler scheduler loop remain before a genuine text-to-music end-to-end attempt.

## Phase C progress, 2026-09-03: condition encoder working (real weights, text+lyric V1 scope)

`src/OpenTail.Stingray.Diffusion/AceStep/Conditioning/AceStepConditionEncoder.cs`: real
`AceStepConditionEncoder`/`AceStepLyricEncoder` V1 scope (text + lyrics, no timbre/reference-audio
-- matches the plan's own V1 cut). Text hidden states are projected via `text_projector` (real
`nn.Linear`, no bias); lyrics are embedded via a raw Qwen3 token-embedding LOOKUP (not a full
Qwen3 forward pass, confirmed from the real `diffusers` pipeline), then `embed_tokens` (a real
`nn.Linear` WITH bias despite the confusing name, projecting `1024 -> 2048`), then run through 8
real bidirectional (sliding/full alternating, same `layer_types` pattern as the DiT) transformer
layers with the same GQA/RoPE/per-head-QK-norm math as `AceStepDiT.cs` -- necessarily duplicated
since those are private static methods on a different class; left un-shared per CLAUDE.md rule 7
(DRY only once 2+ real verified callers exist, matching how MusicGen/AudioGen's generation loop
was left un-merged for the same reason). Real `pack_sequences` (sort-by-mask, for batched/padded
scenarios) is a no-op for V1's single unpadded prompt, so this class just concatenates
`[lyricHidden, textProjected]` directly rather than reimplementing padding-aware sorting that
never triggers.

To supply the raw Qwen3 embedding table to the condition encoder, `AceStepQwen3TextEncoder`
(`src/OpenTail.Stingray.Diffusion/AceStep/Text/AceStepQwen3TextEncoder.cs`) gained two small public
members: `Tokenize(string)` (raw tokenization without a full forward pass, for lyric token IDs) and
`TokenEmbeddingTable` (lazily-loaded, dequantized real `token_embd.weight`, row-major
`[vocab, hiddenSize]`).

`AceStepConditionEncoderTests.Forward_RealWeights_ProducesNonDegenerateCondition` passes on the
FIRST real-weight run against `turbo.safetensors`'s condition-encoder tensors and the real Q8_0
Qwen3-Embedding-0.6B GGUF: correct shape (`lyricTokens.Length + textHidden.Length` rows of 2048),
finite non-degenerate output, AND a real sensitivity check -- two different lyric strings produce
measurably different packed condition rows, confirming the lyric-encoder path (not just the text
path) is genuinely wired rather than silently ignoring its input.

All four V1 components (text encoder, DiT, VAE decoder, condition encoder) are now real and
individually tested against real weights. Only the flow-matching Euler scheduler loop and
`AceStepPipeline.Generate()`'s end-to-end wiring remain.

## Phase E progress, 2026-09-03: flow-matching Euler scheduler + real end-to-end generation

`src/OpenTail.Stingray.Diffusion/AceStep/Transformer/AceStepFlowScheduler.cs`: the real Turbo
`infer_method="ode"` Euler-ODE loop from `generate_audio`, transcribed directly (the `"sde"` branch
is a real alternative in the reference but unused by any real Turbo default, not ported). Samples
Gaussian noise at the target latent length, runs the schedule selected by `shift` (snapping to the
nearest of the real {1,2,3} values exactly like the reference), and on the final step computes
`x0 = xt - vt*t` directly (`get_x0_from_noise`) instead of an Euler step. Cross-attention K/V is
precomputed once via `AceStepDiT.PrepareCrossAttention` and reused every step, matching the real
reference's `EncoderDecoderCache` optimization.

**Real, documented gap**: the real `diffusers` `AceStepConditionEncoder` ships a learned
`silence_latent` buffer (VAE-encoded real audio silence), used as `src_latents` for plain
text-to-music generation (confirmed from `diffusers/pipelines/ace_step/pipeline_ace_step.py`'s
`prepare_src_latents`). That buffer is genuinely NOT present in the real
`acestep-v15-turbo/model.safetensors` checkpoint this project downloaded (confirmed by inspecting
its real safetensors header directly -- no `silence_latent` key among its 678 tensors), so it must
ship as a separate converter/asset this project doesn't have. `AceStepFlowScheduler` uses all-zero
`src_latents` as an explicit, flagged placeholder for V1 -- the real pipeline's own comment warns
zeros put the TIMBRE encoder OOD (drone-like audio), but V1 never calls the timbre encoder at all
(text+lyrics-only condition encoder scope), so that specific warning doesn't directly apply; the
real open question is how much zero `src_latents` degrades the DiT's own context conditioning
relative to real encoded silence. `chunk_masks` = all-ones IS confirmed real for plain generation
(the same pipeline's own doc comment: "dumping the chunk_masks tensor ... unique values = [True]").
Revisit if/when the real `silence_latent` buffer is located (candidate: the official `diffusers`-
converted checkpoint on HF, as opposed to the raw `custom_code` Ace-Step1.5 repo used so far).

`AceStepFlowSchedulerTests` passes on the FIRST real-weight run against the full DiT checkpoint
with a synthetic condition sequence: finite non-degenerate output, and a real sensitivity check
(two different seeds produce measurably different final latents).

`AceStepPipeline.Generate()` is now real, wiring all four components together: real SFT prompt
formatting -> Qwen3 text encoder -> condition encoder (text+lyrics, empty lyric token list for
`Instrumental=true`) -> flow scheduler -> Oobleck VAE decode -> stereo `StereoAudioBuffer`.
`AceStepPipelineEndToEndTests` (2-second duration, to keep the real 8-step loop's wall-clock cost
low) passes on the FIRST real end-to-end run against all four real checkpoints: finite, non-silent,
correctly-shaped 48kHz stereo PCM from a real text prompt.

**Real sample generated and numerically characterized, same day**: an 8-second real-weight sample
(`docs/diffusion-samples/acestep_v15turbo_cinematic_orchestral_8s.wav`, prompt "A cinematic
orchestral soundtrack with deep drums", instrumental, seed 1234 -- gitignored, not committed, per
this project's own rule 9) was generated via a throwaway scratch test (deleted after use, per rule
9) and characterized the same way the AudioGen "white noise" investigation did: RMS stays stable
around 0.11-0.12 across every 0.5s window for the full 8s (no silence, no runaway clipping -- peak
0.47), and spectral flatness is 0.166 (far from 1.0, which would indicate white-noise-like output;
consistent with tonal/structured audio content). This is a real, non-degenerate, plausibly-musical
signal even at only 8 real Euler steps with the zero-`src_latents` placeholder -- reassuring, but
NOT a substitute for an actual human listening pass (no ears were involved in producing these
numbers) and NOT proof the zero-`src_latents` gap has zero quality cost.

**DRY pass done, same day** (CLAUDE.md rule 7 -- porting is complete, and `AceStepConditionEncoder`
became a genuine second real, verified caller of the DiT's GQA/RoPE/per-head-QK-norm/softmax/SiLU
math): extracted the byte-identical duplication into
`src/OpenTail.Stingray.Diffusion/AceStep/Primitives/AceStepAttentionKernels.cs`
(`BuildRope`/`ApplyRope`/`RmsNormPerHead`/`RmsNorm`/`SoftmaxRange`/`Silu`), matching this project's
existing `Primitives/*Kernels.cs` convention. Both `AceStepDiT.cs` and `AceStepConditionEncoder.cs`
now call the shared kernels; each caller's own per-layer glue (AdaLN modulation vs. plain pre-norm,
gated vs. ungated residuals) stayed separate -- only the shared attention/norm math moved. All six
real-weight AceStep tests (text encoder, VAE decoder, DiT, condition encoder, flow scheduler,
end-to-end pipeline) re-run and pass unchanged after the extraction, confirming no numerical
regression.

**Performance pass, same day** (CLAUDE.md rule 7 -- measured, not assumed): real 10-second
`AceStepPipeline.Generate()` calls (real weights, 8 real Euler steps, CPU, `AceStepConditionEncoder`
lyric encoder empty since `Instrumental=true`), 3 timed runs after a warmup run to exclude JIT/weight-
paging cost: **121.78s / 122.52s / 124.74s, mean 123.01s** -- i.e. roughly 12.3s of wall-clock per
second of generated audio at this duration/step-count. Confirmed the CPU matmul path already goes
through this project's own real SIMD kernels (`CfmLinearWeight.MatMul` -&gt; `SimdKernels.MatVecF32`,
AVX2/AVX-512), NOT a naive scalar fallback -- the "OpenBLAS: not found" log line during these runs
is unrelated to this path (that's the GGUF text-generation engine's own diagnostic, not something
`CfmLinearWeight` depends on). No change was made this pass: `MatVecF32` is called once per sequence
position rather than as one batched GEMM across all `t` positions at once (per-row `MatMul` loop
inside the attention/MLP helpers) -- that's the obvious next lever if a real speedup is needed, but
per this project's own rule ("only keep a change if it's measurably better"), it is left unimplemented
rather than spec-implemented without re-measuring; revisit with a real profiler pass (not guesswork)
if 10s-of-audio-in-~2-minutes turns out too slow for the intended use.

## Golden-parity check, 2026-09-03: real bug found and fixed, DiT + lyric encoder now numerically verified

Discovered that `diffusers` ships a real, checkpoint-compatible reimplementation of ACE-Step 1.5
Turbo (`diffusers.models.transformers.ace_step_transformer.AceStepTransformer1DModel` and
`diffusers.pipelines.ace_step.modeling_ace_step.AceStepLyricEncoder`) that this project's own real
`turbo.safetensors` checkpoint loads into with a purely mechanical tensor-name remap (`q_proj`->
`to_q`, `o_proj`->`to_out.0`, etc. -- documented in the (gitignored, scratch-only) `golden_dit.py`/
`golden_lyric.py` scripts). `load_state_dict(strict=False)` reported **zero missing and zero
unexpected keys** for both modules against the real checkpoint, confirming this project's own
tensor-name/shape understanding of the real architecture is exactly correct.

**Real bug found**: `AceStepDiTWeights` already loaded the real `decoder.condition_embedder.weight/
bias` tensors (a real, learned `[2048,2048]` `nn.Linear`), but nothing in `AceStepDiT.cs` ever
applied them -- `PrepareCrossAttention` fed the raw condition-encoder output straight into
cross-attention K/V, when the real `AceStepTransformer1DModel.forward` applies
`encoder_hidden_states = self.condition_embedder(encoder_hidden_states)` exactly once, before any
layer sees it. This is the kind of bug non-degeneracy testing (finite, shape-correct, sensitive-to-
input output) structurally cannot catch -- the DiT still ran, still produced plausible-looking
output, and was still numerically WRONG. Fixed in `AceStepDiT.PrepareCrossAttention` (now projects
`encoderHiddenStatesRaw` through `w.ConditionEmbedderWeight`/`Bias` before computing K/V).

**Numeric verification**: with the fix applied, `AceStepDiTGoldenParityTests` (fixed-seed synthetic
`hidden_states`/`context_latents`/`encoder_hidden_states`/`timestep`, real weights, compared against
the real `diffusers` reference's output over the SAME real weights) measures **relative error
~7e-6** -- essentially F32-rounding-level agreement over the full 24-layer forward pass, not just
"close". `AceStepLyricEncoderGoldenParityTests` (real 8-layer lyric encoder, same remap technique)
similarly passes well under a 0.1% relative-error tolerance. `EncodeLyrics` was made `public` (was
`private`) specifically so this test could drive it directly with the real reference's exact
intermediate tensor. All eight real-weight AceStep tests (four component tests, flow scheduler,
end-to-end pipeline, two golden-parity checks) re-run and pass together.

**Scope of this golden-parity check**: covers the DiT (the largest, most bug-prone component) and
the lyric encoder end-to-end at essentially bit-exact agreement. Does NOT cover: the Qwen3 text
encoder (already reuses this project's existing, separately-tested `Engine.ForwardPass`, out of
scope for an ACE-Step-specific reference), the Oobleck VAE decoder (a real `diffusers`
`AutoencoderOobleck` golden check is a realistic, still-open next step, not done this pass), or a
full `generate_audio`-shaped end-to-end numeric comparison (the flow-matching Euler loop itself
wasn't diffed step-by-step against a real reference run -- the per-step math it calls, the DiT
forward, now is verified, which is the load-bearing piece).

**Sample regenerated after the fix**: re-ran the same 8-second real-weight sample (same prompt,
same seed 1234) with the `condition_embedder` fix applied. Output changed substantially from the
pre-fix version -- RMS jumped from ~0.11-0.12 to ~0.55-0.67 per 0.5s window (peak now hits the
int16 clamp at 1.0, i.e. real clipping in the raw un-normalized VAE output), spectral flatness
0.266 (still far from white-noise's 1.0, still structured/tonal). The much higher amplitude is a
real, expected consequence of the fix -- before it, cross-attention was reading an un-projected
(effectively wrong-scale) condition sequence. The clipping itself is a new, separate, real
observation: `AceStepOobleckDecoder.Decode` returns raw un-normalized PCM (matches the real
`AutoencoderOobleck.decode`, which also doesn't normalize) -- a real product-facing pipeline likely
needs a peak-normalize-before-WAV step this port doesn't have yet; flagged as a real open item, not
fixed speculatively here.

**Human listening feedback, same day: "it's a soup of noise."** Real, honest signal that numeric
proxies alone missed. Investigated by extending the golden-parity technique to the VAE decoder
(the one real component not yet numerically checked): `AceStepOobleckDecoderGoldenParityTests`
(same technique -- real `vae.safetensors` loaded into the real `diffusers` `AutoencoderOobleck`
decoder with zero missing/unexpected keys, fixed-seed synthetic latent, numeric diff) measures
**~4e-6 relative error** -- also essentially bit-exact. **This rules out the VAE decoder as the
cause.** With the DiT (~7e-6), lyric encoder (well under 0.1%), and VAE decoder (~4e-6) all now
independently numerically verified against real references on synthetic inputs, the code paths
this project wrote are confirmed correct at the component level. The "soup of noise" is therefore
almost certainly a real CONDITIONING gap, not a math bug:

1. **The zero-`src_latents` placeholder** (flagged since Phase E) -- the DiT was never trained on
   literal zeros here; real inference always feeds either real source-audio latents or the real
   learned `silence_latent` buffer, which this project doesn't have.
2. **The timbre encoder is entirely absent from V1's condition sequence**, not just fed zeros --
   the real condition sequence is `[lyric, timbre, text]` (`_pack_sequences` order, confirmed
   earlier this session); V1 omits the timbre segment's rows entirely rather than replacing them
   with placeholder rows. The DiT's cross-attention was trained on a condition sequence that always
   includes a real (if silent-reference) timbre segment -- structurally shortening it, not just
   changing its values, is a bigger distribution shift than the `src_latents` gap alone.
3. Real clipping (7.4% of samples at the int16 max in the regenerated sample) compounds whatever
   the underlying signal quality is, though clipping alone doesn't explain "noise" for an otherwise
   structured signal.

**Real next step**: the `encoder.timbre_encoder.*` weights DO exist in the real checkpoint (this
project just never wired them, V1 scope) -- implementing the timbre encoder path with the real
learned `silence_latent` substituted for "no reference audio" (matching the real pipeline's own
`prepare_condition` behavior) is the concrete, verifiable fix candidate, blocked on the same missing
`silence_latent` buffer already flagged in Phase E. Until that buffer is found or reconstructed, V1
should be understood as "numerically correct given its inputs, but tested with out-of-distribution
conditioning" -- not yet a reliable text-to-music generator.

**Not yet done**: (1) locate or reconstruct the real `silence_latent` buffer; (2) implement the
timbre encoder path (weights already loadable, just unused); (3) re-run the human listening check
once both land; (4) output-normalization/clipping handling before WAV export.

## Immediate next steps (in order)

1. Download the real weights (`acestep-v15-turbo/model.safetensors` ~4.79GB,
   `vae/diffusion_pytorch_model.safetensors`, `Qwen3-Embedding-0.6B/model.safetensors`) and
   confirm the full 677-tensor turbo inventory (only a partial range was fetched this session) plus
   full VAE inventory against the encoder side too (only decoder tensors were visible in this
   session's partial fetch).
2. ~~Verify the real `AutoencoderOobleck` Snake1d formula~~ **DONE, same session**: real formula
   from `diffusers/models/autoencoders/autoencoder_oobleck.py`, confirmed NOT the same as DAC's
   single-parameter Snake: `x + (1/(exp(beta)+1e-9)) * sin(exp(alpha)*x)^2` -- both `alpha` and
   `beta` are stored in LOG-SCALE (`logscale=True` real default), so both need `exp()` applied
   before use, a real easy-to-miss detail (using the raw stored values directly would be a
   real, silent bug). Full decoder structure also confirmed: `conv1(k=7,pad=3) -> N x
   OobleckDecoderBlock(snake1 -> ConvTranspose1d(k=2*stride, pad=ceil(stride/2)) -> 3x
   OobleckResidualUnit at dilations 1/3/9, same shape as DAC's residual stack) -> snake1 ->
   conv2(k=7,pad=3,NO bias, channels->audio_channels)`. The decoder's `upsampling_ratios`
   constructor param is the REVERSE of the config's `downsampling_ratios` (confirmed from the real
   class docstring: "used in reverse order for upsampling in the decoder") -- so decoder strides
   are `[10,6,4,4,2]` for this checkpoint's real `downsampling_ratios=[2,4,4,6,10]`, not the
   config order directly. Structurally this is now directly buildable by adapting
   `Primitives.EncodecDecoderKernels`'s existing `FullConv1d`/`ConvTranspose1dNoPad`/weight-norm-
   fold helpers with a new two-parameter Snake activation swapped in for ELU.
3. ~~Verify whether ACE-Step's Qwen3 text/lyric encoding is causal or bidirectional~~ **DONE, same
   session**: read the real official `diffusers` ACE-Step pipeline (`diffusers` 0.40.0, already
   installed in this environment -- ships the complete real
   `diffusers.pipelines.ace_step.pipeline_ace_step`/`diffusers.models.transformers
   .ace_step_transformer`, 1295+ lines, not partial). See the "Corrections and confirmations" 
   section above for the full real answer: text is CAUSAL full-Qwen3, lyrics are embedding-lookup-
   only, metadata is templated into the text string (not a separate tensor), and Turbo genuinely
   skips CFG. This resolved a real wrong assumption in this plan's first draft (metadata handling)
   before any code was written against it.
4. Scaffold the five core classes (`AceStepConfig`, `AceStepGenerationParams`, `AceStepModel`,
   `AceStepPipeline`, weight-loader stubs) with real constants — **DONE, this session's deliverable**.
5. ~~Check whether the existing `ForwardPass`/GGUF `"qwen3"` engine path can be reused as the text
   encoder~~ **Investigated, same session, real nuanced answer** (not a clean yes): `ForwardPass`
   already has exactly the right extraction primitive for this
   (`EnableHiddenTaps(layerIds)`/`HiddenTapsAt(position)`, an existing per-token/per-layer hidden-
   state tap, plus a `LastHidden` property) — if a Qwen3-Embedding-0.6B checkpoint can be loaded
   into a `ForwardPass`, getting `text_hidden_states` out is basically free. BUT the SafeTensors
   text-model loading lane (`Core/SafetensorsTextModelPackage.cs`) that would load this checkpoint
   WITHOUT a GGUF conversion currently gates `model_type` to `llama`/`mistral` only
   (`SafetensorsTextModelPackage.Open` throws `NotSupportedException` for anything else) — `qwen3`
   is admitted for the separate GGUF-loading path (`ModelCompatibility.cs`), not this one. So the
   real options are: (a) convert `Qwen3-Embedding-0.6B` to GGUF first (existing tooling, likely the
   less risky path), or (b) extend `SafetensorsTextModelPackage`'s supported `model_type` set to
   include `qwen3` and verify the downstream tensor-name-mapping/QK-norm wiring actually handles it
   correctly — a real, scoped, but genuinely separate piece of engine work from ACE-Step itself,
   deliberately NOT attempted in this session (would need its own verification pass, out of scope
   for "port ACE-Step's text encoder"). Recommend (a) for the next session unless a GGUF conversion
   turns out to be unexpectedly awkward for this specific checkpoint.
6. Build outward from there per the golden-test ladder in the original plan (config → tokenizer →
   Qwen3 → condition encoder → one DiT block → full DiT → one scheduler step → full 8-step latent →
   VAE → end-to-end), each with a real golden/non-degeneracy check before moving on, matching how
   MusicGen/AudioGen were verified.
