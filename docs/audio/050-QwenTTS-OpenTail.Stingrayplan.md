QwenTTS → OpenTail.Stingray, I’d structure the implementation plan around the same principles as the CosyVoice work, but there are a few important QwenTTS-specific things to watch.

1. First identify exactly which QwenTTS generation

Don't design around "QwenTTS" as a single architecture. The plan should explicitly separate:

Qwen3-TTS / current Qwen TTS family
Base text-to-speech
Voice cloning / reference-audio conditioning
Custom voice / speaker conditioning
Streaming generation, if supported by the particular checkpoint
The acoustic/token generator
The vocoder / audio decoder

The key architectural question for Stingray is:

Can we represent QwenTTS as a GGUF-backed token-generation model + a small amount of deterministic audio decoding, without dragging in Python/PyTorch?

That should be the central design goal.

2. Don't treat it like ordinary LLM inference

QwenTTS is attractive for Stingray precisely because the language-model-looking part can potentially fit your existing infrastructure, but the output isn't ordinary text.

Conceptually:

Text
  ↓
Text / phoneme / semantic conditioning
  ↓
QwenTTS Transformer
  ↓
Speech / acoustic tokens
  ↓
Speech decoder / codec
  ↓
Waveform
  ↓
PCM / WAV

So I'd introduce something along the lines of:

IQwenTtsModel
IQwenTtsSession
IQwenTtsConditioning
IQwenTtsDecoder
IQwenTtsVoice

rather than trying to make it pretend to be ITextGenerationModel.

3. Reuse Stingray's existing GGUF infrastructure aggressively

This is probably the biggest opportunity.

If QwenTTS contains a transformer that is sufficiently close to architectures already handled by Stingray, don't create another inference engine.

Instead:

OpenTail.Stingray
        │
        ├── GGUF loader
        ├── tensor backend
        ├── KV/cache infrastructure
        ├── quantisation
        ├── CPU execution
        └── scheduler
                │
                ▼
          QwenTTS adapter

The Qwen-specific layer should ideally contain:

architecture metadata
tensor-name mapping
input preparation
positional encoding
generation loop
speech-token handling
stopping conditions
decoder integration

rather than another copy of the transformer runtime.

4. Pay particular attention to multiple codebooks

This is one of the areas I'd investigate before writing much code.

Speech-token models commonly don't behave like:

token → token → token → token

They can instead have something closer to:

time step 1:
    codebook 0
    codebook 1
    codebook 2
    ...


time step 2:
    codebook 0
    codebook 1
    codebook 2
    ...

or use specialised interleaving/flattening.

That affects the generation loop enormously.

I'd therefore make a dedicated abstraction:

public interface ISpeechTokenStream
{
    int CodebookCount { get; }


    bool TryDecode(
        ReadOnlySpan<int> tokens,
        out SpeechFrame frame);
}

Don't bake the token layout into the generic generation engine.

5. Voice/reference audio should be a first-class conditioning object

For voice cloning, I'd avoid an API such as:

Generate("Hello", "voice.wav");

Internally you want:

var voice = await tts.CreateVoiceConditioningAsync(
    referenceAudio);


var audio = await tts.GenerateAsync(
    text,
    voice);

because the reference audio may require expensive processing.

Potential pipeline:

reference.wav
     ↓
audio preprocessing
     ↓
speaker / voice embedding
     ↓
prompt speech tokens
     ↓
QwenTTS conditioning
     ↓
cached conditioning object

That means the same voice can be reused:

var voice = await tts.LoadVoiceAsync(...);


await tts.SpeakAsync("Hello", voice);
await tts.SpeakAsync("How are you?", voice);
await tts.SpeakAsync("Goodbye.", voice);

Much better for a local assistant.

6. Separate voice conditioning from model residency

This fits your ModelRuntimeManager particularly well.

You don't want:

model loaded
reference processed
voice embedding generated
model unloaded
...

to happen repeatedly.

Think:

QwenTTS model
      │
      ├── Voice A conditioning
      ├── Voice B conditioning
      ├── Voice C conditioning
      └── Voice D conditioning

with explicit memory accounting.

This could eventually become a generic Stingray concept:

IModelConditioning

which CosyVoice and QwenTTS could both use.

7. The vocoder is probably the major architectural question

I'd make this an explicit Phase 0 investigation, rather than assuming it can be handled by the same GGUF transformer path.

You want to determine:

What exactly does QwenTTS emit?
What decoder turns it into waveform?
Is that decoder:
Transformer?
ConvNet?
VAE?
codec decoder?
custom neural audio decoder?
Does it have GGUF support?
Can its tensors be loaded by Stingray?
Does it require another runtime?
Can it be implemented directly using Stingray primitives?

The ideal architecture is:

QwenTTS Transformer
        ↓
 speech tokens
        ↓
Stingray-native decoder
        ↓
 PCM

rather than:

C# → Python → PyTorch → Qwen → Python vocoder → WAV
8. Don't overlook sample-rate handling

The public API should probably return something richer than byte[].

Something like:

public sealed record GeneratedAudio(
    ReadOnlyMemory<float> Samples,
    int SampleRate,
    int Channels);

Then conversion to WAV/PCM is an output concern.

This gives you:

QwenTTS
   ↓
float PCM
   ↓
AudioBuffer
   ├── WAV
   ├── streaming PCM
   ├── audio player
   └── OpenTail voice pipeline
9. Streaming should be designed in from the beginning

This is particularly important for an assistant.

Don't make the API:

Task<AudioBuffer> GenerateAsync(...)

only.

I'd have:

IAsyncEnumerable<AudioChunk> StreamAsync(...)

or an equivalent callback abstraction.

Potentially:

text
 ↓
speech tokens
 ↓
decoder
 ↓
AudioChunk
 ↓
consumer

So FlightDeck could start playing audio before the complete utterance has finished.

10. Add a speech-specific session

Your existing session machinery could become particularly useful here.

Something like:

var session = tts.CreateSession();


await session.SetVoiceAsync(voice);


await session.SpeakAsync(
    "Welcome to OpenTail.");


await session.SpeakAsync(
    "What would you like to do?");

The session could retain:

voice conditioning
generation parameters
speaker configuration
language
style
audio state
cached prefixes/conditioning
decoder state where possible
11. Sampling needs speech-specific controls

Don't just expose ordinary LLM temperature.

I'd investigate and expose whatever QwenTTS actually supports, potentially:

public sealed record TtsGenerationParams
{
    public float Temperature { get; init; }
    public float TopP { get; init; }
    public int? Seed { get; init; }


    public float RepetitionPenalty { get; init; }


    // Qwen-specific where applicable
    public float SpeechRate { get; init; }
    public float Pitch { get; init; }
}

But don't invent controls that QwenTTS doesn't actually implement. The plan should map these against the real model implementation.

12. Build a conformance test against the official implementation

This is especially important for TTS.

I'd make a golden test:

Official QwenTTS
       │
       ├── text
       ├── voice
       ├── seed
       └── parameters
             ↓
        reference audio


Stingray
       │
       ├── same text
       ├── same voice
       ├── same seed
       └── same parameters
             ↓
        candidate audio

Then compare:

generated speech tokens
token count
codebook structure
waveform duration
sample rate
RMS
spectral characteristics
eventually perceptual similarity

The speech-token comparison is especially valuable because waveform-level floating-point differences can be misleading.

13. Suggested implementation phases

I'd make the actual plan roughly:

Phase 0 — Architecture reconnaissance

Identify exact QwenTTS checkpoint architecture
Map model files
Identify transformer
Identify speech-token representation
Identify codec/vocoder
Establish whether GGUF exists/is practical
Establish tensor names and dimensions

Phase 1 — GGUF/model loader

metadata
tensor mapping
quantised loading
architecture registration
model validation

Phase 2 — QwenTTS transformer

forward pass
attention
RoPE/position handling
generation
sampling
speech-token emission

Phase 3 — Speech-token pipeline

codebook representation
token interleaving/deinterleaving
stopping
EOS handling
deterministic generation

Phase 4 — Audio decoder

native decoder
tensor loading
decoder inference
PCM generation

Phase 5 — Voice conditioning

reference audio
speaker embedding / prompt tokens
conditioning cache
voice abstraction

Phase 6 — Streaming

incremental token generation
incremental decoding
IAsyncEnumerable<AudioChunk>
cancellation

Phase 7 — Stingray integration

IQwenTtsModel
model manager integration
resource accounting
sessions
metrics
FlightDeck integration

Phase 8 — Conformance + benchmarks

official implementation comparison
token-level golden tests
audio quality tests
CPU benchmarks
memory benchmarks
quantisation tests
14. One thing I'd specifically not do

Don't make the first version dependent on:

Python
PyTorch
Transformers
CUDA
ONNX Runtime

unless the architecture investigation proves that a particular component genuinely cannot reasonably be brought native.

Given what you've been building with OpenTail.Stingray, I'd aim for:

                 OpenTail
                    │
              QwenTTS API
                    │
        ┌───────────┴───────────┐
        │                       │
 QwenTTS Transformer       Audio Decoder
        │                       │
        └──────────┬────────────┘
                   │
              Native audio
                   │
              OpenTail Audio

That is the really interesting target.

And compared with CosyVoice, I would expect QwenTTS to be a somewhat more involved port—not necessarily because the transformer itself is harder, but because the speech-token/codebook + codec/decoder boundary needs to be understood very precisely.