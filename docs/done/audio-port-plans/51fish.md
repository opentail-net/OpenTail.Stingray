Yes — and Fish Speech is a particularly interesting one for OpenTail.Stingray, because its architecture has some similarities to the QwenTTS/CosyVoice problem, but the speech-token pipeline is different.

I'd structure the pointers like this:

Fish Speech → OpenTail.Stingray
1. Start with the exact Fish Speech generation

Don't treat "Fish Speech" as one fixed architecture. Establish which generation/checkpoint you're targeting and lock the plan to that.

The first investigation should establish:

Fish Speech checkpoint
        │
        ├── text/token representation
        ├── semantic/acoustic speech tokens
        ├── autoregressive transformer
        ├── codec / decoder
        └── waveform

The important question for Stingray is:

How much of Fish Speech can be expressed using the existing GGUF transformer infrastructure, and what genuinely needs a new native audio component?

2. The likely high-level Stingray architecture

I'd aim for:

                 OpenTail
                    │
              FishSpeech API
                    │
        ┌───────────┴────────────┐
        │                        │
 Fish Speech Transformer    Speech Codec
        │                        │
        └──────────┬─────────────┘
                   │
                PCM audio
                   │
             Stingray.Audio

Rather than introducing a completely separate inference runtime.

3. Fish Speech's biggest issue: the codec

This is probably the first serious technical investigation.

Fish Speech uses a neural audio codec/tokenisation stage, so you need to understand precisely:

text
 ↓
language/text tokens
 ↓
Fish Speech language model
 ↓
speech tokens
 ↓
codec decoder
 ↓
waveform

The codec shouldn't be hidden inside the transformer implementation.

I'd create something conceptually like:

public interface ISpeechCodec
{
    int SampleRate { get; }


    AudioBuffer Decode(
        ReadOnlySpan<int> speechTokens);
}

Then:

public sealed class FishSpeechCodec : ISpeechCodec
{
}

That gives you a reusable abstraction for Fish Speech, QwenTTS, CosyVoice, etc.

4. Don't assume the speech tokens are ordinary tokens

This deserves its own implementation phase.

Investigate:

number of codebooks
codebook vocabulary sizes
token ordering
special tokens
EOS semantics
delayed/interleaved codebooks
frame structure
relationship between token count and audio duration

You want an internal representation like:

public readonly record struct SpeechCode(
    int Codebook,
    int Token);

or, if Fish Speech has a more efficient representation:

public sealed class SpeechTokenFrame
{
    public ReadOnlyMemory<int> Codes { get; }
}

Don't flatten this prematurely into a normal int[] token stream.

5. Reuse Stingray's transformer machinery

This is where your existing work becomes valuable.

Ideally:

GGUF
 │
 ▼
Stingray GGUF loader
 │
 ▼
Fish Speech architecture adapter
 │
 ├── tensor mapping
 ├── model configuration
 ├── attention
 ├── positional encoding
 ├── embeddings
 └── generation

You don't want:

FishSpeechRuntime
FishAttention
FishKVCache
FishTensorLoader
FishQuantizer
...

if the existing infrastructure can handle it.

The Fish-specific implementation should be as thin as possible.

6. KV caching should work particularly nicely here

Because the speech generator is autoregressive, your existing KV infrastructure should be directly relevant.

I'd explicitly make Fish Speech a test case for:

paged KV cache
prefix caching
session-native inference
quantised KV
session continuation
speculative decoding, if the token structure permits it

Potentially:

Fish Speech session
        │
        ├── text prefix KV
        ├── voice/prompt KV
        └── generated speech KV

That could make repeated speech generation substantially cheaper.

7. Voice cloning should be designed as conditioning

If the particular Fish Speech model supports reference audio/voice cloning, don't make that an incidental API.

Something like:

var voice = await fish.CreateVoiceAsync(
    referenceAudio);


var audio = await fish.GenerateAsync(
    "Hello world",
    voice);

Internally:

reference.wav
     │
     ▼
audio preprocessing
     │
     ▼
speech codec
     │
     ▼
reference speech tokens
     │
     ▼
Fish Speech conditioning
     │
     ▼
generation

That means the expensive reference processing can be cached.

8. Make voice conditioning reusable

This fits OpenTail particularly well:

public interface ISpeechVoice
{
    string Id { get; }
}

Then:

var voice = await fish.LoadVoiceAsync("my-voice");


await fish.SpeakAsync("First sentence", voice);
await fish.SpeakAsync("Second sentence", voice);
await fish.SpeakAsync("Third sentence", voice);

The model can remain resident while the voice conditioning remains separately cached.

9. Streaming should be a first-class feature

For OpenTail this is important.

Don't only implement:

Task<AudioBuffer> GenerateAsync(...)

Design for:

IAsyncEnumerable<AudioChunk> StreamAsync(...)

Potential pipeline:

                 Fish Speech
                     │
             speech token frames
                     │
                codec decoder
                     │
              AudioChunk #1
              AudioChunk #2
              AudioChunk #3
                     │
                     ▼
              OpenTail.Audio

Then FlightDeck/assistant applications can start playback before the whole sentence has been generated.

10. Audio decoding needs incremental support

This is an important distinction.

A decoder that only accepts:

Decode(allSpeechTokens)

is easy.

But streaming wants:

Decode(nextSpeechFrames)

So I'd investigate whether the Fish codec can maintain decoder state:

public interface IStreamingSpeechCodec
{
    void Reset();


    AudioChunk Decode(
        ReadOnlySpan<SpeechCode> codes);


    AudioChunk Flush();
}

If it can't be made genuinely incremental, the first release can still provide non-streaming generation and add streaming later.

11. Fish Speech should have its own model adapter

Something along these lines:

public interface IFishSpeechModel
{
    FishSpeechSession CreateSession();
}

and:

public interface IFishSpeechSession
{
    Task<GeneratedAudio> GenerateAsync(
        string text,
        FishSpeechGenerationParams parameters,
        CancellationToken cancellationToken = default);


    IAsyncEnumerable<AudioChunk> StreamAsync(
        string text,
        FishSpeechGenerationParams parameters,
        CancellationToken cancellationToken = default);
}

This keeps Fish-specific behaviour out of the generic audio API.

12. Generation parameters

Map the actual Fish Speech parameters rather than inventing a generic TTS parameter set.

Potential categories to investigate:

public sealed record FishSpeechGenerationParams
{
    public float Temperature { get; init; }
    public float TopP { get; init; }
    public int? Seed { get; init; }


    // Only if supported by the checkpoint:
    public float RepetitionPenalty { get; init; }
}

And keep model-specific parameters model-specific.

13. A really useful optimisation: cache the text/voice prefix

This is where your existing radix/prefix-cache work could become very interesting.

Imagine:

Voice conditioning
        │
        ▼
┌───────────────────────┐
│ cached KV prefix      │
└───────────────────────┘
        │
        ├── "Hello..."
        ├── "How are you?"
        └── "The weather..."

If Fish Speech's conditioning can be represented in the transformer prefix, you may be able to avoid recomputing it for every utterance.

That is exactly the sort of thing your Stingray architecture can exploit better than a simple Python reference implementation.

14. Quantisation needs its own test matrix

I'd test at least:

F32
F16
Q8
Q6
Q5
Q4
Q3
Q2

but don't assume every layer should be quantised identically.

Speech generation can be surprisingly sensitive to errors.

You should measure:

intelligibility
pronunciation
speaker similarity
prosody
artefacts
duration
generation speed
RAM

rather than just perplexity.

15. The golden-reference tests are especially important

I'd make the Fish implementation conform against the official implementation.

First compare:

same text
same seed
same voice
same parameters
        │
        ├── official
        └── Stingray

Then compare speech tokens first.

Official:
  [12, 483, 91, ...]


Stingray:
  [12, 483, 91, ...]

If those match, you've isolated the transformer.

Then test:

speech tokens
      ↓
official codec ──┐
                 ├── waveform comparison
Stingray codec ──┘

This is much easier to debug than starting with "the voices sound slightly different".

Suggested implementation phases

I'd make the actual plan:

Phase 0 — Fish Speech architecture reconnaissance
Exact supported checkpoint
Model architecture
GGUF availability/conversion
Tensor inventory
Speech-token format
Codebooks
Special tokens
Codec architecture
Codec tensor format
Reference/voice conditioning
Phase 1 — GGUF integration
GGUF metadata
tensor mapping
architecture registration
quantised loading
model validation
Phase 2 — Fish Speech transformer
embeddings
attention
positional encoding
FFN
KV cache
forward pass
autoregressive generation
Phase 3 — Speech-token engine
codebook representation
token/frame conversion
EOS
stopping
sampling
deterministic seed handling
Phase 4 — Fish codec
codec loader
codec model
decoder
waveform generation
sample-rate handling
Phase 5 — Voice cloning / conditioning
reference audio
reference encoding
conditioning
voice cache
reusable voice objects
Phase 6 — Streaming
incremental generation
incremental codec decoding
audio chunks
cancellation
backpressure
Phase 7 — OpenTail integration
IFishSpeechModel
sessions
ModelRuntimeManager
residency
memory accounting
metrics
FlightDeck
Stingray.Audio
Phase 8 — Optimisation
KV paging
prefix caching
quantisation
batching
speculative approaches
CPU optimisation
Phase 9 — Conformance
token-level golden tests
codec tests
waveform tests
voice similarity
intelligibility
CPU/RAM benchmarks
The really nice end result

I'd ultimately want all three to converge on common Stingray primitives:

                 OpenTail.Stingray.Audio
                          │
             ┌────────────┼────────────┐
             │            │            │
         CosyVoice     QwenTTS     Fish Speech
             │            │            │
             └────────────┼────────────┘
                          │
                  Common abstractions
                          │
              ┌───────────┼───────────┐
              │           │           │
          Speech       Voice       Codec
          Tokens     Conditioning  Decoder
              │           │           │
              └───────────┼───────────┘
                          │
                    Stingray Core

That's the bit I'd emphasise in the Fish Speech plan. Don't just port three TTS models independently. Use CosyVoice/QwenTTS/Fish Speech to establish a native speech-generation substrate in Stingray, with the model-specific adapters sitting on top.

And Fish Speech is a very good candidate for that because it forces you to properly solve speech tokens + codebooks + codec decoding + voice conditioning, rather than just adding another transformer.