# AudioGen implementation plan

Status: **first real end-to-end generation working, 2026-09-02**, same day as MusicGen. Full
pipeline (T5-large text conditioning -> 48-layer delayed-pattern decoder with classifier-free
guidance -> 16kHz EnCodec decode) runs against real `facebook/audiogen-medium` weights and
produces non-degenerate, listenable audio
(`docs/audio-samples/audiogen-medium-first-real-sample.wav`, gitignored/local — not golden-parity
verified yet). Implementation in `src/OpenTail.Stingray.Audio/AudioGen/`; tests in
`tests/OpenTail.Stingray.Tests.Audio/AudioGen/`.

## Why this was fast: proving MusicGen's infrastructure was reusable

The premise for doing AudioGen right after MusicGen was that AudioCraft's own real AudioGen
implementation explicitly reuses MusicGen's LM architecture (single-stage autoregressive
Transformer over delayed EnCodec codebooks) — differing mainly in the audio codec (16kHz,
environmental-sound-trained EnCodec vs MusicGen's 32kHz music EnCodec). That held up under real
inspection, but two real, checkpoint-format surprises meant this wasn't a drop-in reuse of
MusicGen's *files* — it required a DRY pass to extract genuinely shared kernels first, then new
per-model glue on top. What ended up shared vs. new, checked against real evidence rather than
assumed:

| Component | Shared? | Evidence |
|---|---|---|
| Delay pattern (`MusicGen.DelayPattern`) | **Yes, unchanged** | Real AudioCraft `codebooks_patterns.py`'s `DelayedPatternProvider` read directly (pip-installed `audiocraft`, not guessed) — confirmed byte-for-byte identical algorithm and `delays: [0,1,2,3]` in AudioGen's own real training config |
| CFG combination formula | **Yes, unchanged** | Real `audiocraft.models.lm`: `logits = uncond + (cond - uncond) * cfg_coef` — identical to MusicGen's |
| CFG null-condition (all-zero embedding) | **Yes, and now independently CONFIRMED for both** (was a guess for MusicGen; verified for both this session) | Real `T5Conditioner.forward`/`.tokenize` source: empty-string prompt's embedding gets multiplied by a zeroed attention mask -> always all-zero in practice |
| T5 encoder math (non-gated FFN) | **Extracted to `Primitives.T5EncoderKernels`** | Byte-identical algorithm, only dims differ (MusicGen's `t5-base` 768d/12L/12H vs AudioGen's `t5-large` 1024d/24L/16H) |
| EnCodec decoder forward pass | **Extracted to `Primitives.EncodecDecoderKernels`** | Byte-identical layer skeleton (init conv -> 2-layer LSTM+residual -> 4x[upsample+residual block] -> final conv, `n_filters=64`/`compress=2`/`kernel=7`/`trim_right_ratio=1.0`/no final activation in both real checkpoints), only per-stage upsample ratios differ (`[8,5,4,4]` vs `[8,5,4,2]`) |
| Transformer attention/FFN low-level code | **New (`AudioGenTransformer`)** | Real, structural differences: fused `in_proj_weight` (Q/K/V concatenated) vs MusicGen's separate matrices; computed sinusoidal position embedding (`cos` first half, `sin` second) vs MusicGen's precomputed `[sin,cos]` buffer; no linear-layer bias anywhere vs MusicGen's LayerNorm-only-bias; different norm-application order naming (`norm1`/`norm_cross`/`norm2`) |
| Generation loop (delay-pattern stepping, CFG batching, sampling) | **Same shape, separate files** (`AudioGenGenerator` vs `MusicGenGenerator`) | Deliberately NOT merged into one shared `AudioTokenGenerator` yet — with only two real callers whose per-model `KvCache`/`Step` signatures still differ, a premature shared interface would likely need reshaping once a third AudioCraft-family model arrives. Revisit once there's a third real caller (CLAUDE.md rule 7: DRY once duplication is proven real, not ahead of it) |

Net effect: not "70-90% plumbing reuse" in the sense of literally running MusicGen's files
unmodified, but two genuinely reusable primitives (`T5EncoderKernels`, `EncodecDecoderKernels`)
now serve both models, MusicGen's own files got measurably smaller/cleaner in the process (see
git history), and the delay-pattern/CFG *algorithm* was proven identical rather than assumed.

## The checkpoint-format surprise the plan didn't anticipate

`facebook/audiogen-medium` has NO official HuggingFace `transformers`-format release (unlike
MusicGen) — it ships as `library_name: audiocraft`, two raw native PyTorch checkpoints
(`state_dict.bin`, `compression_state_dict.bin`), no safetensors anywhere, and no official
`config.json`. This meant:

1. **No safetensors loader could touch it directly.** Resolved by using this environment's
   already-available Python/torch (confirmed present, same as noted for prior numeric-comparison
   work) to `torch.load` both checkpoints and re-save every real tensor via
   `safetensors.torch.save_file`, preserving native AudioCraft tensor names verbatim (no
   remapping) — see the conversion script referenced in git history. NOT a hand-rolled PyTorch
   pickle/zip parser in C#, which would have been a much larger, riskier undertaking for
   uncertain payoff.
2. **The real archaeology data lives inside the checkpoint, not a repo file.** Both
   `state_dict.bin` and `compression_state_dict.bin` are real AudioCraft "Solver" checkpoints: a
   dict with `best_state` (the actual weights) and `xp.cfg` (the full OmegaConf training config as
   a YAML string) — every real hyperparameter used in this port (dims, layer count, delay pattern,
   EnCodec ratios, T5 variant, CFG coefficient) was read directly from that embedded config, not
   assumed from MusicGen's numbers or general AudioCraft documentation.
3. **T5-large is genuinely external and frozen**, unlike MusicGen's bundled `t5-base`. Confirmed
   from real `T5Conditioner.__init__` source: `finetune=False` means the T5 model is stored via
   `self.__dict__['t5'] = ...`, bypassing `nn.Module` parameter registration entirely — so it
   never appears in the checkpoint's own state dict. Downloaded the real stock `t5-large`
   safetensors + tokenizer.json separately. (First download attempt of this file was silently
   truncated by a background-task time limit and produced a corrupt safetensors header —
   caught by a real parse failure, not a silent wrong-shape bug — re-downloaded clean.)

## Real-checkpoint config used (from `xp.cfg`, not memory)

LM (`transformer_lm`): `dim=1536, num_heads=24, num_layers=48, hidden_scale=4 (ffn=6144), n_q=4,
card=2048, activation=gelu, norm_first=true, bias_ff=false, bias_attn=false, bias_proj=false,
positional_embedding=sin, causal=true, qk_layer_norm=false, kv_repeat=1`.
Conditioning: `conditioners.description.model=t5, t5.name=t5-large`.
`classifier_free_guidance.inference_coef=3.0`. `codebooks_pattern.modeling=delay,
delays=[0,1,2,3]`.

EnCodec (`compression_state_dict.bin`'s own `xp.cfg`): `seanet.dimension=128, n_filters=64,
n_residual_layers=1, ratios=[8,5,4,2], kernel_size=7, residual_kernel_size=3, last_kernel_size=7,
dilation_base=2, compress=2, lstm=2, causal=false, disable_norm_outer_blocks=0`.
`decoder.trim_right_ratio=1.0, decoder.final_activation=null`. `rvq.n_q=4, rvq.bins=2048`.
`sample_rate=16000, channels=1` -> `16000 / (8*5*4*2) = 50` frames/sec, same frame rate as
MusicGen despite the different sample rate.

## Known gaps (same shape as MusicGen's, not yet closed)

- No numeric golden-parity reference against real AudioCraft output (no reference dump script run
  yet — `audiocraft` IS pip-installed in this environment now, so a real Python-side golden dump
  is more reachable here than it was for MusicGen's HF-only checkpoint; worth doing next).
- No performance pass (CLAUDE.md rule 7) — same two-serial-forward-pass CFG as MusicGen, and a
  48-layer/1536-dim model is meaningfully slower per step than MusicGen's 24-layer/1024-dim (a 5s
  real-sampled generation currently takes on the order of a minute+ on this machine, untuned).
- No DRY pass on the generation LOOP itself yet (see table above) — only the T5/EnCodec math was
  extracted this round.
- Real per-codebook top-k/temperature sampling and greedy decoding are implemented; nucleus
  (top-p) sampling is not (same as MusicGen).

## Suggested next steps

1. A real Python-side golden dump using the now-installed `audiocraft` package (LM logits at a
   few real steps, T5 conditioning vector, EnCodec decode of a known token sequence) — this
   environment can actually do this for AudioGen, unlike MusicGen where no local reference model
   was ever run.
2. CFG batch-2 performance pass once correctness is verified — same shape of work noted in
   MusicGen's plan doc.
3. Revisit the shared-generation-loop question (`AudioTokenGenerator`) once/if a third
   AudioCraft-family model (e.g. MAGNeT, a MusicGen melody variant) is ported — not before.
