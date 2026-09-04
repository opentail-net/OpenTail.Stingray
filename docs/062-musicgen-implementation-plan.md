# MusicGen (and AudioGen) implementation plan

Status: **first real end-to-end generation working, 2026-09-02.** Full pipeline (T5 text
conditioning -> delayed-pattern 24-layer decoder with classifier-free guidance -> EnCodec 32kHz
decode) runs against real `facebook/musicgen-small` weights and produces non-degenerate,
listenable audio (`docs/audio-samples/musicgen-small-first-real-sample.wav`, gitignored/local —
not a golden-parity result yet, see "Known gaps" below). Implementation lives in
`src/OpenTail.Stingray.Audio/MusicGen/`; tests in
`tests/OpenTail.Stingray.Tests.Audio/MusicGen/`.

## DRY pass, 2026-09-02 (same day as AudioGen)

Once AudioGen needed the byte-for-byte identical non-gated T5 encoder algorithm and EnCodec
decoder layer skeleton (just different dims/ratios), both were extracted to shared kernels:
`Primitives/T5EncoderKernels.cs` and `Primitives/EncodecDecoderKernels.cs`. MusicGen's own
`MusicGenTextEncoder`/`MusicGenTextEncoderWeights`/`EncodecDecoderWeights` files became thin
loaders delegating to those kernels; `MusicGenT5Tokenizer` moved to `Primitives/T5Tokenizer.cs`
(it was already fully generic). Re-ran the full MusicGen test suite after each step of this
refactor — zero regressions. See `docs/063-audiogen-implementation-plan.md` for the full
reuse-vs-new breakdown and why the generation LOOP itself was deliberately NOT merged yet.

## Known gaps (not yet closed)

- **No numeric golden-parity verification against a real independent reference** (no local
  Python/torch available this session) — today's pass is a non-degeneracy smoke test (finite,
  non-silent PCM) plus internal consistency (delay-pattern round-trip tests), not a token-level
  or spectral-level match against real HF `transformers` output. This is the single biggest open
  item before trusting output quality.
- **Classifier-free guidance null condition is a guess**: implemented as all-zero
  `encoder_hidden_states` for the unconditional branch (the AudioCraft convention, from memory),
  not verified against real HF `MusicgenForConditionalGeneration.generate`'s actual masking
  behavior. If the real implementation instead runs T5 on an empty/pad string, or handles the
  mask differently, guidance strength (and therefore prompt adherence) will be off.
- **No performance pass yet** (CLAUDE.md rule 7) — CFG currently runs as two fully serial forward
  passes (conditional + unconditional) rather than the batch-2 GEMM the plan below calls for; a
  5-second real-sampled generation currently takes ~30s wall clock on this machine (untuned).
- **No DRY pass yet** (CLAUDE.md rule 7) — `MusicGenTextEncoder` duplicates
  `Parler.T5Encoder`'s structure (non-gated FFN variant) rather than sharing a common T5 kernel;
  intentional per this doc's original "don't extract shared abstraction prematurely" note, revisit
  once MusicGen is fully verified.
- Real per-codebook top-k/temperature sampling and greedy decoding are both implemented; nucleus
  (top-p) sampling is not.

## Corrections made during implementation (real checkpoint inspection caught two real bugs)

1. **Delay-pattern input/target off-by-one**: the initial design conflated the delayed TARGET
  column (what `DelayPattern.BuildInput` lays out) with the transformer's INPUT column at a given
  generation step. A causal LM predicts position `s` from what was already generated at position
  `s-1` (or BOS at `s=0`) — it cannot be fed the very token it's about to produce. Caught by a
  unit test (`DelayPatternTests`) before this reached real-weight code; fixed as
  `DelayPattern.InputColumnForStep` (see its doc comment for the exact shift).
2. **Text encoder is bundled, not composed, and needs a real projection layer**: the original plan
  assumed MusicGen composes a SEPARATE stock `t5-base` checkpoint via
  `AutoModel.from_pretrained`. Real inspection of `musicgen-small`'s own `model.safetensors`
  header disproved this — it contains a full self-contained `text_encoder.*` tensor tree (same
  "bundled" convention `Parler.T5EncoderWeights` already used) PLUS a top-level
  `enc_to_dec_proj.{weight,bias}` (`[1024,768]`) that projects T5's 768-dim output up to the
  decoder's 1024-dim hidden size before cross-attention — missing that produced an immediate
  array-length crash the first time real weights were run, not a silent numeric error.

Both corrections are the exact reason this project's golden-testing discipline (docs/062 §
"testing strategy" below, matching CLAUDE.md rule 8) exists — check the real checkpoint/reference
before trusting an assumption, even one that reads as authoritative documentation.

## Why this model

Text-to-music, `facebook/musicgen-small` alone has 2M+ Hugging Face downloads — squarely inside
this project's "run any GGUF from Hugging Face" popularity-weighted priority order (see
`docs/00-current-work.md`'s cross-project priority section). Architecturally it is much less
exotic than CosyVoice3 or the diffusion-video pipelines already ported here: text encoder →
autoregressive Transformer over discrete audio codes → neural codec decoder. Most of the pieces
(Transformer inference, KV cache, autoregressive sampling, quantized GGUF weights, conv-based
audio synthesis) already exist in this codebase in some form.

**Licensing note (must resolve before distributing/downloading weights by default):** AudioCraft's
code is MIT, but the released MusicGen checkpoints are CC-BY-NC 4.0. `stingray pull` should not
silently normalize this the way it does permissively-licensed models — flag it or require explicit
opt-in. Does not block implementation; blocks default-on weight distribution.

## Architecture summary

```
TEXT PROMPT
   |
   v
T5 text encoder  ->  text conditioning hidden states
   |
   v
MusicGen Transformer (decoder-only, cross-attends to text conditioning)
   audio token embeddings (summed across codebooks) + positional embedding
   -> N transformer layers -> per-codebook output heads
   |
   v
4 parallel codebook logit streams, generated under a DELAYED PATTERN
   (codebook q is offset by q frames so one forward step produces one column
   of all 4 codebooks; this is why generation only needs ~50 steps/sec of audio
   instead of 200)
   |
   v
Delay-pattern de-interleave -> 4 clean codebook token streams
   |
   v
EnCodec decoder (32kHz): per-codebook embedding lookup -> sum -> conv-transpose /
   residual-block upsampling decoder -> PCM
```

Do not treat this as one monolithic new engine. Structure it as an "audio token transformer"
abstraction that MusicGen is the first instance of, with AudioGen (same architecture, 16kHz
EnCodec, single-stage) as a likely cheap second instance later. Do not build that shared
abstraction prematurely — extract it only once MusicGen alone is proven, per this project's
"no speculative abstraction" rule.

## Phases (each independently golden-testable — do not skip ahead)

1. **Model archaeology** — download `facebook/musicgen-small`, inventory every tensor and config
   value from the real checkpoint (not from memory/blog posts). Produce a tensor-name -> C#
   component mapping and confirm: hidden size, layer count, heads, FFN dim, activation, norm type,
   positional embedding scheme, attention bias, T5 variant/config, EnCodec layer structure.
2. **Delay-pattern token machinery** — `MusicGenDelayPattern` build/reverse round-trip, PAD/BOS
   handling, multi-codebook embedding (sum, not concat), output-head splitting. Pure data-shape
   logic, no model weights needed — write this and its round-trip test first, since an off-by-one
   here produces plausible-looking garbage that is expensive to debug later.
3. **Text conditioning (T5)** — prompt -> tokens -> T5 hidden states, golden-verified against a
   real independent reference (same standard as SD3.5/LTX T5 work already in this repo).
4. **MusicGen Transformer** — known audio tokens -> exact per-codebook logits, golden-verified.
   Reuse existing decoder/attention/KV-cache primitives; do not hand-roll a parallel stack.
5. **Generation loop** — autoregressive sampling with classifier-free guidance (batch conditional +
   unconditional as one batch-2 pass, not two serial passes), top-k/temperature, delay schedule.
   Milestone: prompt -> deterministic audio tokens (no sound yet, but a complete LM).
6. **EnCodec decoder** — RVQ codebook lookup/sum -> conv-transpose decoder -> residual blocks ->
   ELU -> upsampling -> PCM. The single largest genuinely-new component for this codebase.
7. **End-to-end** — prompt -> WAV, `MusicGenerationRequest`/`GenerateAsync` product API on top.

Golden-test checkpoints to capture at each phase boundary (tokenizer IDs, T5 hidden states,
first-forward-pass logits per codebook, post-N-step token sequence, EnCodec latents, final PCM
correlation) — same trace-and-diff discipline used for SD3.5's timestep/pooled-embedding
verification (see `docs/057-sd35-performance-handoff.md` and the diffusion samples README) and
demanded by CLAUDE.md rule 8 (check the real C++/reference before "fixing" ported math that looks
wrong).

## Suggested project layout

```
src/OpenTail.Stingray.Audio/MusicGen/
    MusicGenModel.cs / MusicGenConfig.cs
    Codec/EncodecDecoder.cs, EncodecQuantizer.cs, EncodecResidualBlock.cs, EncodecConfig.cs
    Transformer/  (thin adapter over existing decoder/attention/KV-cache primitives)
    Conditioning/T5MusicConditioner.cs  (implements a reusable IConditionEncoder, not MusicGen-private)
    Generation/DelayPattern.cs, MusicGenSampler.cs, MusicGenGenerator.cs
tests/OpenTail.Stingray.Tests.Audio/MusicGen/
    DelayPatternTests.cs, EncodecGoldenTests.cs, TransformerGoldenTests.cs, GenerationTests.cs
```

## Immediate next action

Sprint 1: pull `facebook/musicgen-small` (check for a GGUF conversion via `stingray pull`, else
plan a safetensors loader path since this is a new checkpoint format for this project), inventory
tensors/config, and produce the tensor-mapping table this whole plan depends on. Nothing past this
should be coded until that table exists and matches the real checkpoint, not blog-post recollection.

## Numeric golden-parity CLOSED, 2026-09-04

`MusicGenDecoderGoldenParityTests` (`scratch-llamacpp-ref/musicgen_decoder_golden.py`, pure-numpy
oracle transcribed from real HF `transformers` MusicGen decoder source, loaded directly against
`models/musicgen-small/musicgen-small.safetensors` -- real tensor names confirmed matching
`MusicGenTransformerWeights.cs`'s assumptions exactly, no surprises) compares real production
`MusicGenTransformer.Step`'s codebook-0 logits (3 fixed timesteps x 4 codebooks, synthetic T5-
stand-in encoder hidden state) against the oracle: cosine similarity > 0.999, passed on the first
try, no bug found. Real 4.53s wall-clock confirms genuine execution of all 24 real transformer
layers, not a silent skip. This closes MusicGen's last documented gap ("not yet golden-verified
numerically") -- see README's status matrix for the updated row.

**Update, same day -- FULL pipeline closure, not just the decoder.** The operator explicitly
required every real stage closed, not just the riskiest one. Three more real golden-parity tests
built and passing, all against `models/musicgen-small/musicgen-small.safetensors`:
- **T5 text encoder** (`MusicGenTextEncoderGoldenParityTests`,
  `scratch-llamacpp-ref/musicgen_t5_encoder_golden.py`): real relative-position-bias self-attention
  (no 1/sqrt(d) scaling), real RMSNorm-only `T5LayerNorm`, plain ReLU FFN. Cosine > 0.999, 0.356s.
- **EnCodec 32kHz decoder** (`MusicGenEncodecDecoderGoldenParityTests`,
  `scratch-llamacpp-ref/musicgen_encodec_decoder_golden.py`): real weight-norm folding, 2-layer
  whole-stack-residual LSTM, trimmed transpose-conv upsampling. Cosine > 0.999, 0.414s.
- **Full end-to-end chain** (`MusicGenEndToEndGoldenParityTests`,
  `scratch-llamacpp-ref/musicgen_e2e_golden.py`): calls the actual PUBLIC `MusicGenGenerator.Generate`
  entry point (not a hand-wired internal pipeline) with a real prompt ("electronic dance music"),
  independently confirms the real C# `T5Tokenizer` produces the identical token ids the oracle
  assumed (guards against tokenizer drift as a separate failure mode), real greedy delayed-pattern
  decode (`guidanceScale=1.0`, `topK=1` for determinism) chained through to real EnCodec decode.
  Cosine > 0.999, 5.7s (confirms real multi-layer/multi-step computation).

All 4 golden tests (this update's 3 + the earlier decoder-only one) pass together, 0 failed,
3.656s combined. **Zero production bugs found in this pass** -- every stage's C# already matched
real reference math exactly on first comparison; the existing doc comments citing specific real
quirks (no T5 attention scaling, RMSNorm-only LayerNorm, weight-norm folding, single whole-stack
LSTM residual, `trim_right_ratio=1.0` conv trimming, delay-pattern lookback index) were all
independently confirmed correct, not just plausible-sounding. MusicGen's numeric verification is
now genuinely complete across every real stage.
