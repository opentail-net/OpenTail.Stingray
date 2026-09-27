having looked at Orpheus specifically, I actually think this one is one of the easier and more attractive TTS ports for OpenTail.Stingray.

The reason is that Orpheus is unusually close to something Stingray already understands:

It's essentially a Llama-family causal LM that has been taught to emit audio codec tokens, followed by SNAC decoding.

The upstream architecture describes Orpheus around a Llama-3B backbone plus SNAC, and there are already GGUF versions of the model and separate GGUF SNAC codec weights available.

Orpheus → OpenTail.Stingray
1. This should be a high-priority port

I'd actually rank the difficulty roughly:

Orpheus
   ↓
Llama-3.2-3B
   ↓
existing Stingray Llama infrastructure
   ↓
small Orpheus-specific generation layer
   ↓
SNAC decoder

That is a much nicer starting point than implementing an entirely new transformer architecture.

The published GGUF architecture describes the talker as Llama-3.2-3B with 28 layers, 3072 hidden size, 24 attention heads and 8 KV heads.

So I'd make the first question:

How much of Orpheus can literally run through Stingray's existing Llama implementation unchanged?

Potentially: a lot.

2. The architecture

The native Stingray target should be:

                 Text
                  │
                  ▼
             Orpheus prompt
                  │
                  ▼
        ┌───────────────────┐
        │ Llama-3.2-3B      │
        │                   │
        │ existing Stingray │
        │ transformer       │
        └─────────┬─────────┘
                  │
             audio tokens
                  │
                  ▼
        ┌───────────────────┐
        │ SNAC 24 kHz       │
        │ decoder           │
        └─────────┬─────────┘
                  │
                  ▼
              24 kHz PCM

That's wonderfully clean.

3. Don't create an OrpheusTransformer

This is probably the most important implementation recommendation.

If Stingray already supports the required Llama-3 architecture, do not fork it.

Instead:

LlamaModel
    │
    ├── normal text generation
    │
    └── Orpheus speech generation

Orpheus should mostly be an application/model adapter around the existing Llama runtime.

Something like:

public sealed class OrpheusModel
{
    private readonly ILlamaModel _model;
    private readonly ISnacDecoder _codec;
}

That's exactly the kind of reuse your architecture is designed for.

4. The output tokens are the interesting bit

Orpheus doesn't generate conventional text.

It generates SNAC codec tokens. The published GGUF description says the model has a vocabulary consisting of the normal Llama vocabulary plus 7 × 4096 codec-token slots.

The structure is particularly interesting:

generated token stream


       ↓


7 tokens = one super-frame


       ↓


┌──────────────────────┐
│ codebook 0 : 1 code  │
│ codebook 1 : 2 codes │
│ codebook 2 : 4 codes │
└──────────────────────┘


       ↓


      SNAC

That 1+2+4 structure is explicitly documented for Orpheus.

So I'd create:

public readonly record struct SnacCode(
    int Codebook,
    int Value);

and:

public sealed class OrpheusTokenDecoder
{
    public bool TryDecodeFrame(
        ReadOnlySpan<int> tokens,
        out SnacFrame frame);
}
5. There is a VERY important token-offset trap

This is something I'd put in the plan in bold.

The codec tokens aren't simply:

0..4095

The different codebooks occupy different ranges in the generated LM vocabulary.

The published implementation describes offsets such as:

codebook 0 → 0
codebook 1 → 4096
codebook 2 → 8192
...

and the generated seven-token pattern needs to be redistributed into the three SNAC codebooks.

One implementation report found that getting those position-specific offsets wrong resulted in silence rather than merely slightly degraded audio.

So:

Make the codec-token mapping a tested standalone component.

Not buried inside the generation loop.

6. Prompt construction is another critical area

There is a deceptively important detail here.

The Orpheus prompt isn't simply:

text → tokenizer → Llama

The published GGUF documentation gives a specific structure involving audio-start, BOS, the text prompt, EOT and audio control tokens.

And notably, it reports that the Llama BOS position is critical.

So create:

public sealed class OrpheusPromptBuilder
{
    public IReadOnlyList<int> Build(
        string text,
        OrpheusVoice voice);
}

and test it independently.

Don't scatter magic token IDs throughout GenerateAsync().

7. Voice handling is pleasantly simple

Orpheus has a predefined voice mechanism rather than requiring the same sort of heavyweight reference-audio conditioning that some other TTS systems use.

The ecosystem exposes voices such as:

tara
leah
jess
leo
dan
mia
zac
zoe
bob
rebeca
lisa

and also supports paralinguistic elements such as:

laugh
chuckle
sigh
cough
sniffle
groan
yawn
gasp

in existing integrations.

That gives you a nice API:

var audio = await orpheus.GenerateAsync(
    "Hello, welcome to OpenTail!",
    voice: OrpheusVoice.Tara);
8. Paralinguistic tokens are worth exposing

This is one of the fun parts of Orpheus.

Rather than treating TTS as:

text → boring speech

you can have:

await orpheus.GenerateAsync(
    "That's hilarious!",
    new OrpheusGenerationParams
    {
        Voice = OrpheusVoice.Tara,
        ParalinguisticElement = OrpheusElement.Laugh
    });

But I'd keep the generic API capable of expressing this without making the common ISpeechSynthesizer interface Orpheus-specific.

Something like:

SpeechStyle
SpeechEvent
SpeechExpression

could eventually become useful across models.

9. SNAC should become a reusable Stingray component

This is where I'd go beyond just "port Orpheus".

SNAC is a multi-scale residual vector quantised audio codec. The 24 kHz version uses three codebooks with 4096 entries and is relatively small compared with the 3B talker.

And there is already evidence that SNAC can be implemented without Python/PyTorch: a pure-Go implementation reports a native decoder verified against the Python implementation.

So Stingray should aim for:

OpenTail.Stingray.Audio.Codecs
              │
              └── SNAC
                    │
             ┌──────┴──────┐
             │             │
          Orpheus       future models

That's potentially much more valuable than an Orpheus-only codec.

10. Keep SNAC F32 initially

This is another case where I would not go crazy with quantisation.

The published SNAC GGUF is only about 25 MB and is deliberately kept F32 because quantising such a small codec isn't considered worth the quality risk.

So:

Orpheus 3B → Q4/Q5/Q6/etc.
SNAC      → F32

is a very sensible first implementation.

That also keeps debugging much easier.

11. Streaming is actually a major opportunity

The official Orpheus repository contains a realtime streaming example.

That makes streaming a much more important target here than "maybe we'll support it later".

Desired pipeline:

Llama
  │
  ├── codec tokens
  │
  ▼
7-token super-frame
  │
  ▼
SNAC
  │
  ▼
AudioChunk
  │
  ▼
Playback

So:

IAsyncEnumerable<AudioChunk> StreamAsync(...)

should be part of the initial architecture.

12. But streaming SNAC needs careful design

Don't assume:

snac.Decode(allTokens)

can simply become:

snac.Decode(nextTokens)

SNAC is hierarchical and reconstructs audio across multiple temporal scales.

I'd explicitly investigate the minimum amount of lookahead/state needed to decode progressively.

Potential interface:

public interface IStreamingSnacDecoder
{
    void Reset();


    bool TryDecode(
        ReadOnlySpan<SnacFrame> frames,
        out AudioChunk audio);


    AudioChunk Flush();
}
13. Orpheus is a fantastic test of your KV infrastructure

Because the backbone is Llama-like, your existing:

KV cache
paged KV
prefix caching
session management
quantisation
scheduler

can potentially be reused almost directly.

The generation looks like:

prompt
  ↓
Llama prefill
  ↓
KV cache
  ↓
audio token
  ↓
KV update
  ↓
audio token
  ↓
KV update
  ...

That's exactly the sort of workload your Stingray inference runtime is designed around.

14. Prefix caching could be particularly useful

The voice/prompt prefix could potentially be cached:

Voice + Orpheus prompt
          │
          ▼
      KV prefix
          │
     ┌────┼────┐
     ▼    ▼    ▼
   text1 text2 text3

Then repeated utterances don't have to redo the same conditioning work.

I'd make this an explicit future optimisation rather than complicating V1.

15. Model size is very attractive

A 3B Orpheus model is a very different proposition from some of the enormous modern speech models.

One published GGUF conversion reports roughly:

FP16       ~6.3 GB
Q4_K_M     ~2 GB

for the model.

Then the SNAC decoder is tiny by comparison.

So this fits your local-PC / small-binary / GGUF-first philosophy extremely well.

16. CPU performance needs honest benchmarking

There is a catch.

One reported CPU deployment of Orpheus achieved only around 7.5 tokens/sec, with a short sentence taking roughly 95 seconds; the same setup on a T4 reportedly reached around 65 tokens/sec.

I wouldn't treat those numbers as universal benchmarks, but they highlight something important:

Getting Orpheus to run natively is not the same thing as making it pleasant on CPU.

So I'd benchmark:

Q4_K_M
Q5_K_M
Q6_K
Q8_0

against:

tokens/sec
audio seconds generated/sec
time-to-first-audio
RAM
SNAC decoding time
total latency

The metric that matters most for TTS is probably:

realtime factor
generated audio duration
─────────────────────────
wall-clock generation time

You want:

RTF < 1.0

for realtime speech.

17. The conformance test should be extremely precise

I'd build:

Official Orpheus
       │
       ├── text
       ├── voice
       ├── seed
       └── generation parameters
                │
                ▼
          codec tokens


Stingray
       │
       ├── same text
       ├── same voice
       ├── same seed
       └── same parameters
                │
                ▼
          codec tokens

Then compare:

Stage 1

Prompt tokens identical.

Stage 2

First N LM logits identical.

Stage 3

Generated codec tokens identical.

Stage 4

SNAC codebooks identical.

Stage 5

PCM output identical/near-identical.

This should make the port much easier to debug.

18. Suggested API

I'd keep it pleasantly small:

public interface IOrpheusTtsModel
{
    OrpheusSession CreateSession();
}
public interface IOrpheusSession
{
    Task<AudioBuffer> GenerateAsync(
        string text,
        OrpheusGenerationParams parameters,
        CancellationToken cancellationToken = default);


    IAsyncEnumerable<AudioChunk> StreamAsync(
        string text,
        OrpheusGenerationParams parameters,
        CancellationToken cancellationToken = default);
}

with:

public sealed record OrpheusGenerationParams
{
    public string Voice { get; init; } = "tara";


    public float Temperature { get; init; } = 0.6f;
    public float TopP { get; init; } = 0.95f;
    public float RepetitionPenalty { get; init; } = 1.1f;


    public int? Seed { get; init; }


    public string? ParalinguisticElement { get; init; }
}

Those defaults are consistent with parameters exposed by existing Orpheus integrations, though the exact generation surface should follow the target checkpoint/upstream implementation.

19. Implementation phases

I'd actually make the plan shorter than QwenTTS/Fish Speech.

Phase 0 — Orpheus reconnaissance
Target checkpoint
Llama architecture compatibility
prompt format
special tokens
codec-token ranges
7-token super-frame layout
SNAC version
streaming behaviour
voice mechanism
Phase 1 — Existing Llama integration
load Orpheus GGUF
verify architecture metadata
reuse Stingray Llama forward path
KV cache
quantisation
Phase 2 — Orpheus generation adapter
prompt builder
voice tokens
audio-start/end handling
codec-token filtering
stopping rules
sampling
Phase 3 — SNAC
native SNAC loader
GGUF codec loading
hierarchical codebooks
decoder
24 kHz PCM
golden decoder tests
Phase 4 — End-to-end TTS
text
 ↓
Orpheus
 ↓
SNAC tokens
 ↓
SNAC
 ↓
AudioBuffer
Phase 5 — Streaming
super-frame buffering
incremental SNAC
AudioChunk
cancellation
playback integration
Phase 6 — OpenTail integration
ModelRuntimeManager
residency
sessions
metrics
FlightDeck
Stingray.Audio
Phase 7 — Optimisation
paged KV
prefix caching
quantisation
CPU SIMD
batching
latency optimisation
Phase 8 — Conformance
prompt golden tests
logits
codec tokens
SNAC output
waveform
audio quality
RTF benchmarks
And this one changes my view of the whole TTS effort

After looking specifically at Orpheus, I'd definitely not implement four independent codec systems.

I'd make the Stingray architecture something like:

OpenTail.Stingray.Audio
│
├── SpeechGeneration
│     │
│     ├── CosyVoice
│     ├── QwenTTS
│     ├── Fish Speech
│     ├── Parler-TTS
│     └── Orpheus
│
├── SpeechTokens
│
├── Conditioning
│     ├── Voice
│     ├── ReferenceAudio
│     └── Description
│
├── Codecs
│     ├── SNAC
│     └── other codecs
│
├── AudioBuffers
│
└── Streaming

And Orpheus is probably the cleanest proof-of-concept for this architecture.

You've effectively got:

Orpheus
  = existing Llama inference
  + special token vocabulary
  + speech-token deinterleaving
  + SNAC decoder

The existence of GGUF Orpheus weights and separate GGUF SNAC weights makes it particularly well aligned with the direction you've been taking Stingray.

In fact, I'd put Orpheus near the front of the actual implementation queue, before some of the more complicated TTS families. It gives you a relatively contained way to prove that "ordinary GGUF LLM infrastructure + native neural audio codec = fully local TTS" works inside OpenTail without introducing another heavyweight runtime.