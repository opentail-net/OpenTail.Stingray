# Plan 049 — Native Qwen3-ASR Support / Completion in OpenTail.Stingray

**Target repository:** `opentail-net/OpenTail.Stingray`  
**Target area:** `src/OpenTail.Stingray.Audio/QwenASR/`  
**Primary targets:** `Qwen3-ASR-0.6B` first, then `Qwen3-ASR-1.7B`  
**Secondary target:** `Qwen3-ForcedAligner-0.6B`  
**Implementation model:** native C#/.NET, using OpenTail's existing inference/audio/runtime infrastructure  
**Status:** **Existing implementation present — this plan completes and makes it numerically faithful.**

---

# 0. Critical Framing

This is **not a greenfield Qwen3-ASR port**.

OpenTail.Stingray already contains a QwenASR implementation with:

```text
src/OpenTail.Stingray.Audio/QwenASR/
    QwenAsrWeights.cs
    QwenAsrDecoder.cs
    QwenAsrPipeline.cs
    QwenAsrMelExtractor.cs
    QwenAsrAudioEncoder.cs
    QwenAsrTokenizer.cs
    QwenAsrForcedAligner.cs
    QwenForcedAlignerWeights.cs
    ...
```

The existing pipeline already exposes:

```csharp
public sealed class QwenAsrPipeline : ISpeechToTextPipeline
```

with:

```csharp
public string Architecture => "Alibaba-Qwen3-ASR";
public int SampleRate => 16000;
```

and already composes:

```text
MelExtractor
Tokenizer
AudioEncoder
Decoder
ForcedAligner
Weights
```

The current pipeline also has:

```csharp
Load(string ggufPath)
Transcribe(...)
Align(...)
TranscribeStreamAsync(...)
```

So the correct objective is:

```text
CURRENT OPEN TAIL QWEN ASR
        │
        ├── audit existing implementation
        ├── retain useful boundaries
        ├── replace synthetic approximations
        ├── make GGUF/weights genuinely faithful
        ├── validate audio frontend numerically
        ├── validate AuT encoder numerically
        ├── validate Qwen3 decoder numerically
        ├── validate tokenizer/prompt formatting
        ├── validate streaming
        ├── validate forced alignment
        └── integrate runtime/residency/benchmarks
                │
                ▼
        NATIVE QWEN3-ASR
```

The existing code is therefore a **scaffold/prototype that should be completed, not thrown away**.

---

# 1. Ground Truth

Use these in descending order of authority.

## 1.1 Official Qwen3-ASR repository

Official source:

`https://github.com/QwenLM/Qwen3-ASR`

The project was released on **2026-01-29**, and native Transformers support was added on **2026-06-26**.

The official package supports:

- Qwen3-ASR-0.6B
- Qwen3-ASR-1.7B
- Qwen3-ForcedAligner-0.6B
- offline inference
- streaming inference
- long audio
- language identification
- multilingual recognition
- forced alignment

The official repository should be the primary behavioural reference.

---

# 2. Model Targets

## 2.1 Qwen3-ASR-0.6B

This should be the first target.

Reasons:

- much smaller
- easier CPU validation
- easier golden tests
- good fit for OpenTail's local-first positioning
- useful as the first complete implementation before scaling to 1.7B

Known architecture:

```text
Audio encoder:
    d_model       = 896
    layers        = 18
    heads         = 14
    FFN           = 3584

Qwen decoder:
    hidden        = 1024
    layers        = 28
    Q heads       = 16
    KV heads      = 8
    head_dim      = 128
    intermediate  = 3072

Vocabulary:
    151,936
```

## 2.2 Qwen3-ASR-1.7B

Second target:

```text
Audio encoder:
    d_model       = 1024
    layers        = 24
    heads         = 16
    FFN           = 4096

Qwen decoder:
    hidden        = 2048
    layers        = 28
    Q heads       = 16
    KV heads      = 8
    head_dim      = 128
    intermediate  = 6144

Vocabulary:
    151,936
```

The architecture should therefore be **configuration driven**, not duplicated in hard-coded model-specific classes.

---

# 3. Current OpenTail Implementation Audit

## 3.1 `QwenAsrPipeline`

The existing pipeline is structurally good.

Current flow:

```text
PCM
 ↓
QwenAsrMelExtractor
 ↓
QwenAsrAudioEncoder
 ↓
audio soft tokens
 ↓
QwenAsrTokenizer.FormatPrompt()
 ↓
QwenAsrDecoder.Generate()
 ↓
QwenAsrTokenizer.DecodeWithTimestamps()
 ↓
SpeechToTextResult
```

It also has a separate:

```text
Align()
 ↓
audio encoder
 ↓
QwenAsrForcedAligner
```

This overall decomposition should be retained.

### Problem

The current streaming implementation accumulates two-second chunks and calls the complete offline `Transcribe()` path on each chunk:

```csharp
int chunkFrames = SampleRate * 2;

...

var req = baseRequest with
{
    AudioSamples = buffer.ToArray()
};

var res = Transcribe(req);
```

This is **chunked re-transcription**, not true Qwen3-ASR streaming.

It should eventually become an incremental encoder/decoder implementation with appropriate audio-context and token rollback/state management.

---

# 4. `QwenAsrMelExtractor` — Good Starting Point, Needs Exact Verification

The current class already captures the important Qwen3-ASR frontend:

```csharp
public const int SampleRate = 16000;
public const int NumMels = 128;
public const int WindowSize = 400;
public const int HopLength = 160;
public const int NFft = 512;
```

and implements:

```text
16 kHz audio
→ STFT
→ power spectrum
→ 128-bin Slaney mel
→ log10
→ dynamic max-8 clamp
→ (x+4)/4
```

That is directionally correct.

## Critical issue: verify every numerical detail

The reference implementation describes:

```text
sample rate      = 16000
mel bins         = 128
window           = 400
hop              = 160
STFT n_fft       = 400/512 depending reference implementation details
log10
dynamic max - 8
(x + 4) / 4
```

Do **not** rely on a prose description alone.

Build a Python golden harness and compare:

```text
PCM
→ STFT
→ mel
→ normalized mel
```

element by element.

### Golden test

```csharp
var actual = extractor.ExtractMel(pcm16k);

AssertTensorClose(
    actual,
    expected,
    atol: 1e-5f,
    rtol: 1e-4f);
```

The exact tolerance should be established empirically from FFT implementation differences.

---

# 5. Important Long-Audio Behaviour

Qwen3-ASR is not simply:

```text
audio → one giant transformer
```

The reference implementation uses important chunk/window behaviour.

Key facts to preserve:

```text
100 mel frames
≈ 1 second

8× temporal downsampling
≈ 12.5 audio tokens/sec

attention window
≈ 8 seconds / 800 mel frames
```

The audio encoder uses windowed attention.

This means a naive implementation:

```csharp
encoder.Forward(allMelFrames)
```

with unrestricted global attention is **not equivalent**.

Long audio must preserve:

```text
chunking
+
per-window attention
+
per-chunk position handling
```

---

# 6. `QwenAsrAudioEncoder` — Current Implementation Is Mostly a Placeholder

The existing class has the right public configuration:

```csharp
public int InMelChannels { get; init; } = 128;
public int EncoderDim { get; init; } = 896;
public int NumLayers { get; init; } = 18;
public int NumHeads { get; init; } = 14;
public int QwenHiddenDim { get; init; } = 1024;
public int WindowSizeInfer { get; init; } = 800;
```

But its forward pass currently synthesizes the convolution output using arithmetic over mel bins.

For example, it effectively does:

```csharp
int mChan = (d + s * 16) % numMels;
float mVal = mel[mChan * numMelFrames + mFrame];
sum += mVal;
```

rather than loading and applying learned Conv2D weights.

The attention similarly uses:

```csharp
float posBias = MathF.Cos(relPos * 0.15f + d * 0.05f);
```

rather than learned Q/K/V projections.

The projection uses:

```csharp
float w = MathF.Cos((q * 19 + d * 11) * 0.05f);
```

rather than checkpoint weights.

Therefore this component is:

```text
API / architecture scaffold: YES
real neural inference:       NO
```

This is the highest-priority replacement after the mel frontend.

---

# 7. Implement the Real Conv2D Stem

The real audio frontend should become:

```text
128-channel mel
        │
        ▼
Conv2D
        │
        ▼
Conv2D
        │
        ▼
Conv2D
        │
        ▼
8× temporal reduction
        │
        ▼
flatten frequency/channel dimensions
        │
        ▼
linear / projection into encoder representation
```

Do not write three special-purpose CosyVoice-style convolution classes.

Reuse OpenTail's existing tensor/convolution primitives if available.

If no suitable primitive exists, add a generic reusable implementation under the shared audio/tensor infrastructure rather than burying it inside `QwenAsrAudioEncoder`.

---

# 8. Real Audio Transformer

The encoder should then implement:

```text
input
 ↓
Conv2D stem
 ↓
positional encoding
 ↓
18 or 24 Transformer blocks
 ↓
projection
 ↓
Qwen hidden dimension
```

Each block should be checkpoint-faithful.

Conceptually:

```csharp
for (int layer = 0; layer < config.NumLayers; layer++)
{
    x = block[layer].Attention(
        x,
        attentionMask,
        positionIds);

    x = block[layer].FeedForward(x);
}
```

Not:

```csharp
x += FakeAttention(x);
x += FakeFFN(x);
```

---

# 9. Windowed Attention

This is a major correctness point.

For each inference window:

```text
mel window
    ↓
Conv2D
    ↓
audio tokens
    ↓
windowed Transformer
```

The model should not accidentally attend across unrelated windows.

Implement a reusable windowing abstraction:

```csharp
public readonly record struct AttentionWindow(
    int StartToken,
    int Length);
```

Then:

```csharp
foreach (var window in windows)
{
    EncodeWindow(
        hidden,
        window,
        positionIds);
}
```

This also becomes useful for streaming.

---

# 10. Per-Chunk Positional Encoding

Do not use one global monotonically increasing position index if the reference implementation resets positions per audio chunk/window.

Golden test:

```text
audio 0–8 sec
audio 8–16 sec
```

Compare:

```text
position IDs
sin/cos tables
encoder outputs
```

between Python and C#.

---

# 11. `QwenAsrDecoder` — Current Implementation Must Be Replaced

The current decoder is clearly synthetic.

It calculates:

```csharp
float audioEnergy = 0.0f;

for (...)
    audioEnergy += MathF.Abs(audioSoftTokens[i]);
```

and then creates token IDs from:

```csharp
int candidateBase =
    1000 +
    ((int)(audioEnergy * 17.0f) + step * 31) % 50;
```

This is not Qwen3 inference.

The replacement must be a genuine Qwen3 decoder.

---

# 12. Real Qwen3 Decoder Architecture

For 0.6B:

```text
hidden = 1024
layers = 28
Q heads = 16
KV heads = 8
head dim = 128
FFN = 3072
vocab = 151936
```

For 1.7B:

```text
hidden = 2048
layers = 28
Q heads = 16
KV heads = 8
head dim = 128
FFN = 6144
vocab = 151936
```

The decoder should reuse OpenTail's Qwen transformer infrastructure wherever possible.

Do not fork an entire Qwen3 implementation just for ASR if the repository already has equivalent Qwen3 attention/RMSNorm/RoPE/GQA/SwiGLU code.

---

# 13. Multimodal Audio Injection

The important conceptual flow is:

```text
audio
 ↓
AuT
 ↓
projected audio embeddings
        │
        ▼
Qwen prompt + audio embeddings
        │
        ▼
Qwen3 decoder
        │
        ▼
text tokens
```

The audio representation is not merely:

```text
audioEnergy = ...
```

It must be inserted at the same semantic location and with the same embedding/position behaviour as the reference implementation.

Golden-test:

```text
prompt token IDs
audio embedding tensor
combined embedding sequence
position IDs
attention mask
first decoder hidden state
```

---

# 14. Qwen3 Attention

The decoder requires GQA:

```text
16 query heads
8 KV heads
```

and the correct Qwen3 positional/normalization behaviour.

Conceptual OpenTail implementation:

```csharp
var q = QProjection(x)
    .Reshape(batch, seq, numQHeads, headDim);

var k = KProjection(x)
    .Reshape(batch, seq, numKvHeads, headDim);

var v = VProjection(x)
    .Reshape(batch, seq, numKvHeads, headDim);

ApplyQKNorm(q, k);
ApplyRoPE(q, k, positions);

var output = GroupedQueryAttention(q, k, v, mask);
```

Use the existing Qwen implementation if OpenTail already has it.

---

# 15. KV Cache

The decoder is autoregressive.

Therefore:

```text
prompt
 ↓
prefill
 ↓
KV cache
 ↓
one token at a time
 ↓
decode
```

Do not recompute the entire prompt for every generated token.

This should plug into OpenTail's existing KV cache/session infrastructure.

Potential shape:

```csharp
public sealed class QwenAsrDecodeState
{
    public required IKvCache KvCache { get; init; }
    public int Position { get; set; }
}
```

The exact type should follow existing OpenTail infrastructure.

---

# 16. Sampling / Termination

ASR normally wants deterministic or near-deterministic decoding.

The implementation should support:

```text
temperature = 0
```

as greedy decoding.

But do not hard-code:

```csharp
if (step > 20)
    EOS;
```

Termination must use actual model special-token IDs.

Support:

```text
EOS
language tokens
timestamp tokens
task tokens
special multimodal tokens
```

from the real tokenizer/model configuration.

---

# 17. `QwenAsrTokenizer`

The tokenizer must be checkpoint-exact.

The Hugging Face model repository contains:

```text
vocab.json
merges.txt
tokenizer_config.json
chat_template.json
preprocessor_config.json
model.safetensors
```

for the released model.

Do not construct a miniature vocabulary.

The model vocabulary is:

```text
151,936 tokens
```

for the Qwen3-ASR checkpoints.

The tokenizer should load the actual assets.

---

# 18. Prompt Formatting

The official inference code must be treated as the ground truth for:

```text
system message
audio placeholder
task instruction
language instruction
timestamp mode
```

The OpenTail API should expose useful semantic options:

```csharp
public sealed record SpeechToTextRequest
{
    public float[] AudioSamples { get; init; }
    public int SampleRate { get; init; }

    public string? Language { get; init; }

    public SpeechTask Task { get; init; }

    public bool EnableTimestamps { get; init; }

    public string? Context { get; init; }

    public float Temperature { get; init; }
}
```

But conversion into Qwen3 ChatML should remain internal.

---

# 19. Language Identification

Qwen3-ASR supports language identification.

Do not fake it by:

```csharp
language = request.Language ?? "en";
```

The current pipeline does exactly that for the result language.

Instead:

```text
audio
 ↓
Qwen3 output
 ↓
language annotation / special token
 ↓
detected language
```

If the user explicitly supplies a language:

```text
forced language
```

should override automatic detection where the reference implementation does so.

---

# 20. Timestamps

There are two separate concepts.

### ASR timestamp output

Model-generated timestamp tokens / segment timing.

### Forced alignment

Separate:

```text
Qwen3-ForcedAligner-0.6B
```

which is specifically intended to align provided text to audio.

Do not merge these into one implementation.

---

# 21. Forced Aligner

The repository already has:

```text
QwenAsrForcedAligner
QwenForcedAlignerWeights
```

This should be treated as a second model target.

Architecture:

```text
audio
 ↓
same / compatible AuT
 ↓
alignment conditioning
 ↓
ForcedAligner model
 ↓
token/word timing
```

The official model supports alignment in 11 languages and up to roughly 5 minutes of speech.

Implement after basic ASR is numerically correct.

---

# 22. `QwenAsrWeights`

The current pipeline loads:

```csharp
var weights = new QwenAsrWeights(ggufPath);
```

and derives:

```csharp
EncoderDim
AudioLayers
AudioHeads
LlmDim
LlmLayers
LlmHeads
LlmKvHeads
LlmVocabSize
```

from the loaded file.

This is a good design direction.

However, the loader must be checked against the actual Qwen3-ASR GGUF tensor naming and metadata.

The architecture should never be silently inferred from a wrong default.

Add:

```csharp
public sealed record QwenAsrModelMetadata(
    string Architecture,
    int EncoderDim,
    int EncoderLayers,
    int EncoderHeads,
    int DecoderDim,
    int DecoderLayers,
    int DecoderHeads,
    int DecoderKvHeads,
    int VocabularySize);
```

and validate it against the checkpoint.

---

# 23. GGUF Strategy

The first-class OpenTail target should remain GGUF if the existing `QwenAsrWeights` path is intended to be the native distribution format.

However, ground truth should be established from the official:

```text
Safetensors
```

checkpoint.

Recommended pipeline:

```text
Official HF Safetensors
          │
          ├── Python reference
          │
          └── conversion / tensor mapping
                    │
                    ▼
                 GGUF
                    │
                    ▼
             OpenTail loader
```

Never validate a GGUF conversion only by whether it produces vaguely plausible transcripts.

Validate:

```text
tensor names
tensor shapes
tensor values
quantized reconstruction
```

where practical.

---

# 24. Strong Recommendation: Keep a Safetensors Verification Path

Even if GGUF is the shipping format, maintain a developer-only verification route:

```text
Safetensors → reference comparison
GGUF       → OpenTail comparison
```

This isolates:

```text
model implementation error
```

from:

```text
GGUF conversion error
```

This is particularly important for a new architecture.

---

# 25. Golden Reference Harness

Create:

```text
tools/
    Qwen3AsrGolden/
```

The Python harness should run the official model and dump:

```text
01_pcm.wav
02_mel.npy
03_conv_stem.npy
04_encoder_layer_0.npy
05_encoder_final.npy
06_audio_projection.npy
07_prompt_ids.npy
08_position_ids.npy
09_combined_embeddings.npy
10_decoder_layer_0.npy
11_decoder_final.npy
12_logits.npy
13_generated_ids.json
14_final_text.json
```

For forced alignment:

```text
20_aligner_input.npy
21_aligner_logits.npy
22_alignment.json
```

---

# 26. Golden Tensor Comparison

Implement reusable OpenTail tests:

```csharp
static void AssertTensorClose(
    ReadOnlySpan<float> actual,
    ReadOnlySpan<float> expected,
    float atol,
    float rtol)
{
    Assert.Equal(expected.Length, actual.Length);

    for (int i = 0; i < actual.Length; i++)
    {
        float tolerance =
            atol + rtol * MathF.Abs(expected[i]);

        Assert.True(
            MathF.Abs(actual[i] - expected[i]) <= tolerance,
            $"Mismatch at {i}: actual={actual[i]}, expected={expected[i]}");
    }
}
```

Use progressively stronger tests:

```text
frontend
→ stem
→ encoder block
→ projector
→ decoder block
→ logits
→ tokens
→ text
```

Do not start with end-to-end transcript equality.

---

# 27. First Golden Test

Use a very short deterministic WAV.

For example:

```text
16 kHz mono
1–3 seconds
known reference phrase
```

Then compare:

```text
Mel[0]
Mel[127]
Mel[last]
```

and aggregate statistics:

```text
min
max
mean
RMS
```

before comparing the complete tensor.

This makes FFT/filterbank bugs much easier to isolate.

---

# 28. Encoder Golden Tests

For the 0.6B model:

```text
input:
    [128, T]

after Conv2D:
    expected shape from reference

after encoder:
    [N, 896]

after projection:
    [N, 1024]
```

Test:

```text
N
shape
mean
variance
selected rows
full tensor
```

---

# 29. Decoder Golden Tests

Start with one prefill pass.

Do **not** start with 256-token generation.

Capture:

```text
embedding output
layer 0 output
layer 1 output
layer 27 output
final RMSNorm
logits
```

Then test:

```text
greedy next-token ID
```

Only after that implement the full autoregressive loop.

---

# 30. End-to-End Golden Test

Finally:

```csharp
var result = pipeline.Transcribe(request);

Assert.Equal(
    expectedText,
    result.Text);
```

But transcript equality should be the final test, not the only test.

---

# 31. Current Streaming Implementation — Replace

Current approach:

```text
2 sec PCM
 ↓
full Transcribe()
 ↓
emit result
```

This causes:

- repeated encoder work
- duplicated context
- unstable boundaries
- poor latency
- incorrect behaviour for speech crossing chunk boundaries

The official Qwen3-ASR stack explicitly supports streaming inference.

---

# 32. Target Streaming Architecture

Use:

```text
microphone / PCM
       │
       ▼
audio ring buffer
       │
       ▼
incremental mel
       │
       ▼
incremental / windowed AuT
       │
       ▼
decoder state
       │
       ▼
partial tokens
       │
       ▼
stable-prefix detection
       │
       ▼
SpeechSegment
```

Potential API:

```csharp
public async IAsyncEnumerable<SpeechPartialResult>
    TranscribeStreamAsync(
        IAsyncEnumerable<ReadOnlyMemory<float>> audio,
        SpeechToTextRequest request,
        ...)
```

The existing `IAsyncEnumerable<SpeechSegment>` API can be retained initially for compatibility.

---

# 33. Streaming Should Use Prefix Stability

Do not emit:

```text
"the ca"
```

then:

```text
"the cat"
```

as two final segments.

Maintain:

```text
previous hypothesis
current hypothesis
```

and identify:

```text
stable prefix
```

Only emit stable text as final.

The exact rollback/context policy should follow the official Qwen3-ASR streaming implementation.

---

# 34. Long-Form Offline ASR

Do not simply feed arbitrary hours of audio into one decoder.

Implement a long-audio strategy:

```text
long audio
 ↓
energy-aware segmentation
 ↓
model-sized chunks
 ↓
ASR
 ↓
timestamp correction
 ↓
merge
```

The official tooling uses long-audio segmentation and supports very long recordings.

OpenTail should keep the generic segmentation machinery reusable for other ASR models.

---

# 35. Context / Prompt Biasing

Qwen3-ASR supports contextual prompting.

Expose:

```csharp
public string? Context { get; init; }
```

but ensure it is inserted exactly where the reference implementation expects it.

Useful examples:

```text
"Company names: OpenTail, Stingray, Alibaba."

"Technical terms: GGUF, Safetensors, Qwen."

"Names: Dmitri, Max, Alex."
```

This should be tested as a behavioural feature.

---

# 36. Audio Input Integration

Reuse OpenTail audio infrastructure:

```text
WAV reader
resampler
PCM conversion
spectral kernels
audio buffers
```

Do not make QwenASR responsible for:

```text
MP3 decoding
WAV parsing
general audio container handling
```

Its model-facing boundary should be:

```text
float PCM
sample rate
channels already normalized
```

---

# 37. 16 kHz Normalization

The model expects:

```text
16 kHz
mono
float PCM
```

Pipeline:

```csharp
float[] pcm16k =
    request.SampleRate == 16000
        ? request.AudioSamples
        : AudioResampler.Resample(
            request.AudioSamples,
            request.SampleRate,
            16000);
```

The existing pipeline already does this.

Retain it, but add tests for:

```text
8 kHz
16 kHz
22.05 kHz
24 kHz
44.1 kHz
48 kHz
```

and verify against the Python reference.

---

# 38. Model Residency

Qwen3-ASR consists of:

```text
audio encoder
+
Qwen decoder
+
tokenizer
+
optional forced aligner
```

Do not create a separate QwenASR-specific residency manager.

Use OpenTail's model/runtime manager.

Potential accounting:

```text
Qwen3-ASR-0.6B
    encoder weights
    decoder weights
    KV cache
    activation buffers
    audio buffers

Qwen3-ASR-1.7B
    encoder weights
    decoder weights
    KV cache
    activation buffers
```

Forced aligner should be independently resident/evictable if the runtime supports component-level residency.

---

# 39. Memory Strategy

The HF 0.6B checkpoint is roughly 1.88 GB in BF16/FP representation according to the published model repository.

The native OpenTail footprint will differ depending on:

```text
GGUF quantization
tensor packing
runtime allocations
KV cache
temporary activations
audio length
```

Therefore benchmark:

```text
Q4
Q5
Q8
F16/BF16 where supported
```

rather than assuming the raw checkpoint size equals runtime memory.

---

# 40. Quantization

The decoder is the obvious largest target.

Prioritize:

```text
Qwen decoder linear weights
```

then:

```text
audio encoder linear weights
```

Do not blindly quantize:

```text
LayerNorm/RMSNorm
small projection layers
sensitive embedding/output tensors
```

without a golden/perplexity/ASR comparison.

Measure:

```text
WER
CER
RTF
RAM
```

for each quantization.

---

# 41. CPU-First Optimization

The implementation should initially prioritize correctness.

Then optimize:

```text
1. matrix multiplication
2. QKV projection
3. FFN
4. Conv2D stem
5. mel extraction
6. attention
7. memory copies
```

Reuse OpenTail optimized kernels.

Potential architecture:

```text
QwenAsrAudioEncoder
    ↓
shared MatMul
    ↓
shared attention
    ↓
shared RMSNorm
    ↓
shared SwiGLU
```

not custom scalar loops in every model.

---

# 42. SIMD / Vectorization

After correctness:

```csharp
Vector<float>
```

or existing OpenTail SIMD kernels should be used for:

- mel operations
- normalization
- elementwise activation
- projection
- quantized dequantization

But do not prematurely optimize the model before golden parity.

---

# 43. Architecture Metadata Validation

At load time:

```csharp
if (weights.Architecture != "qwen3_asr")
    throw new InvalidDataException(...);
```

Then validate:

```text
encoder dimensions
decoder dimensions
layer count
head count
KV head count
vocabulary size
```

Example:

```csharp
Debug.Assert(
    weights.LlmHeads % weights.LlmKvHeads == 0);
```

And:

```csharp
int groupSize =
    weights.LlmHeads / weights.LlmKvHeads;
```

---

# 44. Do Not Trust Filenames

These should not determine architecture:

```text
qwen3-asr-0.6b.gguf
```

Instead:

```text
GGUF metadata
+
tensor inventory
+
known model configuration
```

should establish the model.

---

# 45. Tensor Inventory Tool

Create:

```text
tools/Qwen3AsrInspect/
```

that outputs:

```text
tensor name
shape
dtype
bytes
quantization
```

Example output:

```text
audio.encoder.layers.0.self_attn.q_proj.weight
    [896, 896]
    Q8_0

model.layers.0.self_attn.q_proj.weight
    [1024, 1024]
    Q4_K_M
```

This is invaluable when mapping the GGUF.

---

# 46. Weight Mapping Table

Create a checked-in mapping document:

```text
docs/qwen3-asr-tensor-map.md
```

with:

| Official tensor | GGUF tensor | OpenTail field |
|---|---|---|
| audio encoder conv | ... | `ConvStem[0]` |
| audio encoder q_proj | ... | `Encoder.Layer[0].Q` |
| decoder q_proj | ... | `Decoder.Layer[0].Q` |
| lm_head | ... | `OutputHead` |

Do not rely on undocumented string replacement forever.

---

# 47. Test Structure

Existing repository already has Qwen ASR test coverage including:

```text
tests/OpenTail.Stingray.Tests.Audio.Fast/QwenAsrTests.cs
tests/OpenTail.Stingray.Tests.Audio/QwenAsrRealWeightsTests.cs
tests/OpenTail.Stingray.Tests.Audio/QwenForcedAlignerRealWeightsTests.cs
```

These should become the primary completion points.

Add:

```text
QwenAsrMelGoldenTests
QwenAsrEncoderGoldenTests
QwenAsrDecoderGoldenTests
QwenAsrTokenizerGoldenTests
QwenAsrStreamingTests
QwenAsrLongAudioTests
QwenAsrQuantizationTests
```

---

# 48. Test Levels

## Level 0 — structural

```text
loads
disposes
correct architecture
correct dimensions
```

## Level 1 — frontend

```text
PCM → mel
```

## Level 2 — encoder

```text
mel → audio embeddings
```

## Level 3 — decoder

```text
audio embeddings + prompt → logits
```

## Level 4 — token

```text
logits → token IDs
```

## Level 5 — text

```text
tokens → transcript
```

## Level 6 — streaming

```text
PCM stream → stable transcript
```

## Level 7 — forced alignment

```text
audio + reference text → timestamps
```

---

# 49. Acceptance Dataset

Use a small fixed set of recordings covering:

```text
English
Chinese
Japanese
German
French
Spanish
Cantonese
accented English
music/background
noise
quiet speech
long speech
very short speech
```

Include:

```text
0.5 sec
2 sec
10 sec
30 sec
2 min
10+ min
```

The official model supports many languages/dialects, so a single English WAV is not sufficient validation.

---

# 50. Metrics

Track:

```text
WER
CER
language accuracy
timestamp error
RTF
TTFT
tokens/sec
RAM
peak activation memory
```

For streaming:

```text
first partial latency
stable-prefix latency
finalization latency
```

---

# 51. Regression Corpus

Keep a tiny deterministic corpus in the repository or test fixture storage.

Each test should have:

```text
audio hash
model hash
expected text
expected language
expected token IDs
```

Do not store giant model weights in the repository.

---

# 52. Reproducibility

Record:

```text
model revision
GGUF conversion revision
quantization
OpenTail commit
test audio hash
runtime settings
```

This fits particularly well with the existing model-provenance work in OpenTail.

---

# 53. `QwenAsrPipeline` Final Architecture

Target:

```text
ISpeechToTextPipeline
        │
        ▼
QwenAsrPipeline
        │
        ├── AudioInputNormalizer
        │
        ├── QwenAsrMelExtractor
        │
        ├── QwenAsrAudioEncoder
        │       ├── Conv2D stem
        │       ├── positional encoding
        │       └── windowed Transformer
        │
        ├── QwenAsrTokenizer
        │
        ├── QwenAsrDecoder
        │       └── shared Qwen3 runtime
        │
        └── optional QwenAsrForcedAligner
```

---

# 54. Streaming Architecture

Target:

```text
IAsyncEnumerable<PCM>
          │
          ▼
QwenAsrStreamingSession
          │
          ├── mel state
          ├── encoder window state
          ├── decoder KV cache
          ├── emitted token prefix
          └── rollback context
                    │
                    ▼
             SpeechPartialResult
```

This should eventually become a reusable ASR streaming abstraction.

---

# 55. Context Safety

Long audio can create enormous decoder context.

Do not allow:

```text
unbounded audio
+
unbounded prompt
+
unbounded generated tokens
```

to silently exceed the decoder context.

Use OpenTail's existing context-window safety mechanisms where available.

Conceptually:

```csharp
if (requiredContext > maxContextTokens)
{
    throw new ContextWindowExceededException(...);
}
```

or use controlled segmentation.

---

# 56. Cancellation

All streaming and long-form operations should support:

```csharp
CancellationToken
```

Cancellation must propagate into:

```text
audio processing
encoder
decoder
KV cache
forced aligner
```

and should release temporary buffers promptly.

---

# 57. Error Handling

Fail clearly for:

```text
unsupported GGUF architecture
missing tensors
incorrect dimensions
invalid tokenizer
unsupported sample format
empty audio
invalid forced-alignment text
context overflow
```

Do not silently fall back to synthetic inference.

This is particularly important because the current implementation contains placeholder computations.

---

# 58. No Synthetic Fallback

This deserves a hard rule:

```text
REAL WEIGHTS
    ↓
REAL INFERENCE

otherwise
    ↓
FAIL
```

Do not keep fake outputs for "demo mode" inside the production pipeline.

If a test double is needed:

```text
QwenAsrPipelineFake
```

should be explicit and separate.

---

# 59. Ground-Truth Sources

Use:

### Official Qwen3-ASR

`https://github.com/QwenLM/Qwen3-ASR`

Primary source for:

```text
model architecture
inference
streaming
tokenization
prompt format
forced alignment
```

### Hugging Face checkpoints

`Qwen/Qwen3-ASR-0.6B`  
`Qwen/Qwen3-ASR-1.7B`  
`Qwen/Qwen3-ASR-0.6B-hf`  
`Qwen/Qwen3-ASR-1.7B-hf`

Use for:

```text
config
tokenizer
weights
metadata
```

### Technical report

`arXiv:2601.21337`

Use for:

```text
architecture rationale
training/inference details
performance
```

### Independent minimal implementations

A useful secondary reference is the `antirez/qwen-asr` implementation, which documents the architecture and provides a standalone Python implementation. It should **not** override official Qwen behaviour, but it is useful for cross-checking low-level tensor operations.

---

# 60. Particularly Useful Independent Reference

The independent implementation describes the pipeline as:

```text
WAV
 ↓
16 kHz
 ↓
Mel
 ↓
Conv2D ×3
 ↓
Transformer encoder
 ↓
Projection
 ↓
Qwen3 decoder
 ↓
tokens
```

and provides a standalone implementation without Transformers.

This is useful when debugging OpenTail because it reduces the number of framework abstractions between:

```text
checkpoint
```

and:

```text
tensor operation
```

Use it as a debugging oracle, not as the authoritative specification.

---

# 61. Recommended Development Sequence

## Phase 0 — Audit

Inspect:

```text
QwenAsrPipeline
QwenAsrWeights
QwenAsrMelExtractor
QwenAsrAudioEncoder
QwenAsrDecoder
QwenAsrTokenizer
QwenAsrForcedAligner
QwenForcedAlignerWeights
existing tests
shared Qwen infrastructure
```

Classify every operation:

```text
real
approximate
synthetic
missing
```

---

## Phase 1 — Model inventory

Implement:

```text
GGUF tensor inspector
model metadata validator
tensor mapping table
```

---

## Phase 2 — Exact frontend

Make:

```text
PCM → mel
```

golden-correct.

---

## Phase 3 — Real Conv2D

Replace synthetic stem.

---

## Phase 4 — Real AuT Transformer

Implement:

```text
attention
RMS/LayerNorm as appropriate
FFN
position encoding
windowing
```

---

## Phase 5 — Audio projection

Make:

```text
encoder → Qwen hidden
```

golden-correct.

---

## Phase 6 — Real tokenizer

Load exact Qwen assets.

---

## Phase 7 — Real Qwen decoder

Reuse shared OpenTail Qwen3 infrastructure.

---

## Phase 8 — KV cache

Implement proper autoregressive decoding.

---

## Phase 9 — End-to-end offline ASR

First:

```text
0.6B
short English WAV
```

Then expand.

---

## Phase 10 — Multilingual

Validate:

```text
English
Chinese
Japanese
German
French
Spanish
```

then broader language/dialect coverage.

---

## Phase 11 — Streaming

Replace the current 2-second re-transcription implementation.

---

## Phase 12 — Long audio

Add segmentation/window/context handling.

---

## Phase 13 — Forced aligner

Complete `Qwen3-ForcedAligner-0.6B`.

---

## Phase 14 — Quantization

Start:

```text
Q4
Q5
Q8
```

and compare WER/RTF/RAM.

---

## Phase 15 — Performance

Optimize:

```text
matmul
attention
FFN
Conv2D
memory movement
KV cache
```

---

# 62. Definition of Done

Qwen3-ASR is complete in OpenTail when:

```text
✓ Qwen3-ASR-0.6B loads real weights
✓ no synthetic audio encoder
✓ no synthetic decoder
✓ exact tokenizer assets
✓ exact mel frontend
✓ real Conv2D stem
✓ real AuT transformer
✓ real audio projection
✓ real Qwen3 decoder
✓ real KV cache
✓ real EOS/token handling
✓ language identification works
✓ multilingual transcription works
✓ offline long audio works
✓ streaming works without full re-transcription
✓ timestamps work where supported
✓ forced aligner works
✓ GGUF path is validated against Safetensors ground truth
✓ real-weight tests pass
✓ quantized models have measured WER
✓ CPU benchmark exists
✓ memory usage is measured
✓ OpenTail runtime/residency is reused
✓ no hidden synthetic fallback remains
```

---

# 63. What Should Be Reused vs Rewritten

| Component | Current status | Decision |
|---|---|---|
| `QwenAsrPipeline` | good orchestration | **KEEP / COMPLETE** |
| `QwenAsrMelExtractor` | mostly real frontend | **VERIFY / FIX** |
| `QwenAsrAudioEncoder` | synthetic internals | **REWRITE INTERNALLY** |
| `QwenAsrDecoder` | synthetic | **REWRITE INTERNALLY** |
| `QwenAsrTokenizer` | existing implementation | **AUDIT / MAKE EXACT** |
| `QwenAsrWeights` | GGUF integration exists | **AUDIT / COMPLETE** |
| `QwenAsrForcedAligner` | existing | **AUDIT / COMPLETE** |
| `QwenForcedAlignerWeights` | existing | **AUDIT / COMPLETE** |
| audio resampling | existing OpenTail | **REUSE** |
| spectral kernels | existing OpenTail | **REUSE** |
| Qwen3 transformer code | if already present | **REUSE** |
| KV cache | existing OpenTail | **REUSE** |
| runtime/residency | existing OpenTail | **REUSE** |
| streaming infrastructure | existing OpenTail | **REUSE / EXTEND** |

---

# 64. Most Important Difference from a Generic Qwen-ASR Port

The implementation should **not** become:

```text
Qwen ASR
    ↓
private runtime
    ↓
private tensor library
    ↓
private KV cache
```

It should become:

```text
                    OpenTail shared runtime
                            │
            ┌───────────────┼───────────────┐
            │               │               │
       Tensor ops        Qwen3 core       KV cache
            │               │               │
            └───────────────┼───────────────┘
                            │
                       QwenAsr
                            │
              ┌─────────────┼──────────────┐
              │             │              │
           AuT encoder   Qwen decoder   aligner
```

That is the architectural value of doing this inside Stingray.

---

# 65. First PR Recommendation

Do **not** make the first PR "implement Qwen3-ASR".

Make it:

> **Qwen3-ASR: establish model provenance, tensor inventory and golden frontend/encoder harness**

It should:

```text
1. identify exact checkpoint
2. inspect all tensors
3. validate metadata
4. establish Python golden runner
5. validate current mel extractor
6. identify every synthetic operation
7. add first golden fixtures
8. add architecture consistency tests
```

Then the implementation PRs can be much smaller and much safer.

---

# 66. Final Engineering Principle

The current OpenTail QwenASR implementation already has the **right shape**.

The problem is not that OpenTail needs another ASR architecture.

The problem is that some of the existing classes currently simulate the architecture rather than execute the checkpoint.

Therefore the goal is:

> **Turn the existing QwenASR scaffold into a numerically faithful native Qwen3-ASR implementation using OpenTail's existing tensor, Qwen, audio, KV-cache, runtime, residency and testing infrastructure.**

The most important early deliverable is therefore:

```text
official Qwen3-ASR
        │
        ▼
golden tensors
        │
        ▼
OpenTail implementation
        │
        ▼
same tensors
```

Once that works for the 0.6B model, scaling the same implementation to 1.7B should be primarily a configuration/weight-size problem rather than a second model port.
