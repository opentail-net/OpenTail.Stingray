Parler-TTS is actually a very worthwhile one for Stingray, and I would approach it somewhat differently from Fish Speech.

The big architectural attraction is that Parler-TTS is much more obviously separable into text/description conditioning → autoregressive audio-token generation → DAC audio decoding. That gives you a nice opportunity to make the Stingray speech stack more reusable.

Parler-TTS → OpenTail.Stingray
1. Start with the exact Parler-TTS family

First lock down which checkpoint/generation you're targeting.

Don't make the implementation assume that every Parler-TTS checkpoint has identical:

transformer dimensions
tokenizer
audio-token vocabulary
DAC configuration
conditioning scheme
speaker/description handling

The initial investigation should produce:

Parler-TTS checkpoint
        │
        ├── text tokenizer
        ├── description/prompt conditioning
        ├── autoregressive transformer
        ├── audio-token representation
        └── DAC decoder
                  │
                  ▼
                PCM
2. The really important distinction: Parler-TTS is description-conditioned

This is one of the things I'd make central to the plan.

Instead of simply:

"Hello world"
      ↓
speech

the model can use a natural-language description of the desired voice/audio characteristics.

Conceptually:

Text:
"Hello world."


Description:
"A female speaker with a warm, clear voice,
speaking slowly and naturally."


             ↓


        Parler-TTS


             ↓


           speech

So the Stingray API shouldn't reduce everything to voiceId.

I'd allow something like:

var audio = await parler.GenerateAsync(
    text: "Hello world.",
    description:
        "A warm female voice speaking clearly and naturally.");

That is a fundamentally useful capability for OpenTail.

3. Separate text from description conditioning

Internally:

                ┌───────────────┐
Text ──────────►│               │
                │   Parler-TTS  │
Description ──► │  Transformer  │
                │               │
                └───────┬───────┘
                        │
                 audio tokens
                        │
                        ▼
                       DAC
                        │
                        ▼
                       PCM

I'd explicitly model these as separate inputs:

public sealed record ParlerTtsRequest
{
    public required string Text { get; init; }


    public string? Description { get; init; }


    public ParlerTtsGenerationParams Generation { get; init; } = new();
}

That will make the API much more faithful to the model.

4. The transformer is the obvious Stingray reuse opportunity

The core implementation should look like:

                    GGUF
                      │
                      ▼
             Stingray GGUF loader
                      │
                      ▼
            Parler-TTS adapter
                      │
          ┌───────────┼───────────┐
          │           │           │
       attention     FFN       embeddings
          │           │           │
          └───────────┼───────────┘
                      │
                 KV cache
                      │
                      ▼
              audio token stream

Don't create a separate tensor/runtime stack if the underlying transformer can use your existing machinery.

This is exactly where the OpenTail architecture starts paying off.

5. DAC should be treated as a separate native model

This is probably the most important technical component after the transformer.

Parler-TTS generates audio codes/tokens, and those then need to be decoded by DAC (Descript Audio Codec) into waveform audio.

So:

Parler transformer
       │
       ▼
 audio codes
       │
       ▼
    DAC decoder
       │
       ▼
    waveform

I'd introduce a generic codec abstraction:

public interface IAudioCodecDecoder
{
    int SampleRate { get; }


    AudioBuffer Decode(
        ReadOnlySpan<int> codes);
}

Then:

public sealed class DacDecoder : IAudioCodecDecoder
{
}

This is potentially reusable beyond Parler-TTS.

6. DAC is where I'd do the most early investigation

Before implementing the transformer, establish:

exact DAC version/configuration
sample rate
number of codebooks
codebook dimensions
codebook vocabulary
decoder architecture
tensor names
whether existing GGUF tooling can represent it
whether Stingray's tensor operations are sufficient
whether decoder state can be made incremental

The desired result is:

DAC weights
    ↓
Stingray tensor loader
    ↓
native DAC implementation
    ↓
PCM

rather than pulling in a heavyweight Python audio stack.

7. Codebooks need a dedicated abstraction

Again, don't assume the audio codes are equivalent to normal LLM tokens.

I'd have something along the lines of:

public readonly record struct AudioCode(
    int Codebook,
    int Token);

and potentially:

public sealed class AudioCodeFrame
{
    public ReadOnlyMemory<int> Codes { get; init; }
}

Then the Parler generation loop can produce frames without knowing how DAC consumes them.

8. Voice control is different from Fish Speech

This is a particularly useful distinction.

For Fish Speech/CosyVoice/QwenTTS you may have:

reference audio
       ↓
voice conditioning
       ↓
speech

For Parler-TTS the interesting path is:

natural-language description
       ↓
conditioning
       ↓
speech

So I'd not force Parler-TTS into the same voice-cloning abstraction.

Instead, make Stingray support both:

ISpeechConditioning
       │
       ├── ReferenceAudioConditioning
       ├── SpeakerEmbeddingConditioning
       └── TextDescriptionConditioning

That could become a very useful common architecture.

9. Description conditioning could be cached

If you're repeatedly using the same description:

"A warm male voice with a British accent,
speaking clearly and conversationally."

you don't necessarily want to recompute its conditioning every time.

Potential API:

var style = await parler.CreateConditioningAsync(
    "A warm male voice speaking clearly.");


await parler.GenerateAsync(
    "Hello.",
    style);


await parler.GenerateAsync(
    "How are you?",
    style);

This is analogous to your reusable voice-conditioning idea, but without pretending that the conditioning necessarily represents an actual speaker identity.

10. This is another excellent use of prefix/KV caching

Potentially:

Description
     │
     ▼
tokenisation
     │
     ▼
conditioning prefix
     │
     ▼
cached KV
     │
 ┌───┼─────────────┐
 ▼   ▼             ▼
text A            text B
 ▼                  ▼
audio              audio

That could be particularly useful in OpenTail where an assistant may repeatedly speak using the same style.

Your existing radix prefix cache / session KV work could therefore potentially benefit Parler-TTS directly.

11. Streaming should be designed for

The ideal pipeline is:

Text
 ↓
Parler Transformer
 ↓
audio code frame
 ↓
DAC
 ↓
AudioChunk
 ↓
Playback

API:

IAsyncEnumerable<AudioChunk> StreamAsync(
    ParlerTtsRequest request,
    CancellationToken cancellationToken = default);

But there's an important caveat:

DAC streaming must be investigated

If DAC requires the entire code sequence before decoding, you can't simply pretend it is streaming.

So Phase 0 should establish whether incremental DAC decoding is practical.

12. Audio representation

Keep the internal result independent of WAV.

Something like:

public sealed record AudioBuffer(
    ReadOnlyMemory<float> Samples,
    int SampleRate,
    int Channels);

Then:

Parler-TTS
    ↓
AudioBuffer
    ├── WAV
    ├── PCM
    ├── streaming playback
    └── OpenTail.Audio

That keeps the model layer clean.

13. Generation parameters

Map the actual Parler implementation rather than creating arbitrary TTS settings.

Potentially:

public sealed record ParlerTtsGenerationParams
{
    public float Temperature { get; init; }
    public float TopP { get; init; }
    public int? Seed { get; init; }


    public int? MaxAudioTokens { get; init; }
}

And where the model supports them, things such as:

repetition handling
guidance
generation length
sampling configuration

should remain explicit.

14. Quantisation is particularly interesting

I'd test:

F16
Q8
Q6
Q5
Q4
Q3
Q2

but split the testing between:

Transformer

Likely a good candidate for aggressive GGUF quantisation.

DAC

Potentially much more sensitive.

I'd therefore not automatically quantise the DAC identically to the transformer.

A sensible first matrix might be:

Transformer     DAC
-----------     ---
Q4              F16
Q5              F16
Q6              F16
Q8              F16

and only then investigate quantised DAC.

15. Golden-token testing

This would be extremely useful.

Run:

Official Parler
       │
       ├── same text
       ├── same description
       ├── same seed
       └── same parameters
              ↓
         audio tokens


Stingray
       │
       ├── same text
       ├── same description
       ├── same seed
       └── same parameters
              ↓
         audio tokens

Compare the audio token sequence first.

Then:

Official tokens ──► official DAC
                  │
                  ├── waveform
                  │
Stingray tokens ──► Stingray DAC

That lets you isolate:

transformer problem
token-format problem
DAC problem

instead of debugging all three simultaneously.

16. Suggested Stingray API

I'd aim for something along these lines:

public interface IParlerTtsModel
{
    ParlerTtsSession CreateSession();
}

Then:

public interface IParlerTtsSession
{
    Task<AudioBuffer> GenerateAsync(
        string text,
        string description,
        ParlerTtsGenerationParams parameters,
        CancellationToken cancellationToken = default);


    IAsyncEnumerable<AudioChunk> StreamAsync(
        string text,
        string description,
        ParlerTtsGenerationParams parameters,
        CancellationToken cancellationToken = default);
}

And eventually:

var session = parler.CreateSession();


var audio = await session.GenerateAsync(
    "Welcome to OpenTail.",
    "A warm, friendly British male voice speaking naturally.",
    parameters);

That's a very OpenTail-native API.

17. Implementation phases

I'd make the actual plan:

Phase 0 — Architecture reconnaissance
Identify target Parler-TTS checkpoint
Transformer architecture
tokenizer
description conditioning
audio-token format
codebooks
special tokens
DAC version
DAC architecture
sample rate
streaming possibilities
GGUF feasibility
Phase 1 — GGUF/model loading
metadata
tensor mapping
architecture registration
quantisation
model validation
Phase 2 — Parler transformer
text conditioning
description conditioning
embeddings
attention
positional encoding
FFN
KV cache
generation
Phase 3 — Audio-token pipeline
codebook representation
frame handling
EOS
stopping
sampling
deterministic generation
Phase 4 — Native DAC
DAC model loader
tensor implementation
decoder
waveform reconstruction
sample-rate handling
Phase 5 — Conditioning cache
description tokenisation
conditioning representation
prefix KV cache
reusable style/conditioning object
Phase 6 — Streaming
incremental audio-code generation
DAC streaming investigation
AudioChunk
cancellation
backpressure
Phase 7 — OpenTail integration
IParlerTtsModel
sessions
ModelRuntimeManager
resource accounting
metrics
FlightDeck
Stingray.Audio
Phase 8 — Optimisation
KV paging
prefix caching
quantisation
CPU optimisation
batching
potentially speculative decoding
Phase 9 — Conformance
golden audio-token tests
DAC tests
waveform tests
intelligibility
speaker/style fidelity
latency
RAM
model-size benchmarks
And I'd make one architectural change to the overall TTS work

After CosyVoice + QwenTTS + Fish Speech + Parler-TTS, I think you have enough different architectures to justify making a proper:

OpenTail.Stingray.Audio
        │
        ├── SpeechTokenGenerator
        │
        ├── SpeechConditioning
        │
        ├── AudioCodec
        │
        ├── AudioCodecDecoder
        │
        ├── Voice / Style
        │
        ├── AudioBuffer
        │
        └── StreamingAudio

Then the four model ports become adapters:

CosyVoice ────────┐
QwenTTS ──────────┤
Fish Speech ──────┼──► Stingray Speech Runtime
Parler-TTS ───────┘

Parler-TTS is especially valuable because it adds a different conditioning model — natural-language voice/style description — rather than just another reference-audio cloning system. That makes it a good test of whether the abstraction is genuinely generic rather than accidentally designed around one TTS family.