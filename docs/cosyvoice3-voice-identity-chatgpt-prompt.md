# CosyVoice3 zero-shot voice cloning produces a "foreign"/wrong-timbre voice — request for advice

## Context

I'm porting CosyVoice3's TTS pipeline from a real C++ reference implementation
(`cosyvoice.cpp`, a llama.cpp-style GGUF/GGML port) to a native C# engine
(OpenTail.Stingray). The pipeline is: LLM speech-token generation → flow
encoder (mu/spks) → DiT/CFM flow-matching mel decoder → HiFT NSF-HiFiGAN
vocoder.

I already found and fixed one real bug: my `SineGen` (NSF harmonic source
generator in the HiFT vocoder) had been "simplified" to a continuous
per-sample cumulative phase, when the real reference algorithm
(`SineGen2::build_cgraph` in `cosyvoice.cpp/src/cosyvoice-graph.cpp:667`)
accumulates phase at **frame rate** and then upsamples the resulting phase to
sample rate with a **nearest-neighbor hold** (`GGML_SCALE_MODE_NEAREST`), not
a continuous per-sample ramp. I reverted my C# port back to match that exactly
(frame-rate cumsum, offset added once at t=0 before the cumsum, then the
scalar `phaseVal * 2π * upsampleScale` held constant across each
`upsampleScale`-sample block). That's now numerically structured the same way
as the reference.

## The remaining problem

Independent of the above: **with zero-shot voice cloning (real reference
audio + campplus speaker embedding + prompt speech tokens), the output voice
does not sound like the reference speaker at all** — it's intelligible speech,
correct-ish prosody, no more buzzing/wobble, but the timbre/identity is
generic/"foreign" rather than matching the cloned voice. Text-only synthesis
(no reference audio, zero speaker embedding) obviously can't match a specific
voice either, but even *with* a real reference clip and real campplus
embedding extracted, the identity doesn't transfer convincingly.

## Pipeline pieces involved in speaker conditioning (C#, my port)

1. **Speaker embedding extraction** (`CosyVoice3Pipeline.ExtractSpeakerEmbedding`,
   `src/OpenTail.Stingray.Audio/CosyVoice/CosyVoice3Pipeline.cs:216`): reads the
   reference wav, resamples to `CamPlusSpeakerEncoder.SampleRate`, runs it
   through `CamPlusSpeakerEncoder.Extract` (an ONNX CAM++ model,
   `models/campplus.onnx`), returns a fixed-size embedding vector (falls back
   to an all-zero vector if no reference audio is given).
2. **Reference mel** (`ExtractReferenceMel`) and **prompt speech tokens**
   (`ExtractPromptTokens`, via `CosyVoiceSpeechTokenizer.Extract`, ONNX) are
   also extracted from the same reference clip.
3. **LLM conditioning**: `CosyVoice3Llm.GenerateSpeechTokens` is called with
   `promptText`/`promptSpeechTokens` so the newly generated speech tokens are a
   real continuation of the reference's own tokens (this part I already fixed
   in an earlier session and verified).
4. **Flow encoder**: `CosyVoice3FlowEncoder.ComputeMuAndSpks(_flowWeights,
   jointTokens, speakerEmbedding)` — `jointTokens` is
   `[...promptTokens, ...speechTokens]` (prompt + newly generated, spliced),
   and `speakerEmbedding` is the raw campplus vector from step 1, fed straight
   in (no normalization applied on my side that I'm aware of — need to check
   whether the reference L2-normalizes or otherwise transforms this vector
   before use).
5. **DiT/CFM decode**: conditions on `mu`/`spks` plus the reference mel spliced
   in as a prompt prefix (`cond` array), then the newly-synthesized portion is
   trimmed off before returning.
6. **HiFT vocoder**: takes the resulting mel, no additional speaker
   conditioning at this stage (source-excitation + conv/ISTFT decode only).

## What I have NOT yet done (please tell me where to focus)

- I have **not** numerically diff'd the campplus embedding vector itself
  (mine vs. the real C++/Python reference) on the exact same reference wav —
  don't know yet if the embedding extraction itself is correct.
- I have **not** checked whether CosyVoice3's flow encoder expects the speaker
  embedding L2-normalized, mean-centered, or passed through any projection
  before being combined with `mu` — I'm currently passing the raw CAM++
  output through unchanged.
- I have **not** verified that `jointTokens` splicing (prompt tokens +
  newly-generated tokens, concatenated) matches how the reference actually
  conditions the flow encoder's token embedding — possible off-by-something in
  how prompt-vs-generated boundaries are handled downstream of the LLM stage.
- I have **not** ruled out that the DiT/CFM stage's own prompt-mel-prefix
  splicing (`cond` array in `CosyVoice3Pipeline.Generate`) is where speaker
  identity actually gets carried across (rather than, or in addition to, the
  `spks` embedding) — if that splicing has an indexing/length bug, the model
  could be falling back to a generic/averaged voice while everything else
  keeps working.
- CFG rate (`cfgRate`, currently defaulted around 0.7) and sampling
  temperature (`temperature`, currently 0.8) are both hand-tuned values from
  earlier debugging sessions, not verified against the reference's actual
  defaults — a mismatch here could plausibly wash out speaker identity even if
  every embedding is numerically correct.

## What I'd like help with

Given a real, working `cosyvoice.cpp` (or the original PyTorch CosyVoice3)
reference to compare against, and env-var-gated tensor dumps already wired up
in my C# port for F0/excitation/mel-stage checkpoints (pattern established in
an earlier debugging session, see `docs/cosyvoice3-hift-buzz-chatgpt-prompt.md`
for that pattern) — **what is the most likely single place a
zero-shot-cloning pipeline like this loses speaker identity while still
producing fluent, non-buzzy speech**, and what's the fastest checkpoint to
numerically diff first (campplus embedding, `spks` after any
normalization/projection, or the DiT prompt-mel-prefix splicing) to localize
it before I go instrument all three?
