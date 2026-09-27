# Plan — Native CosyVoice TTS Support for OpenTail.Stingray

**Reference implementation:** `FunAudioLLM/CosyVoice` / `QwenAudio/CosyVoice`  
**Primary target:** Fun-CosyVoice3-0.5B-2512, with a compatibility path for CosyVoice2-0.5B  
**Target:** `opentail-net/OpenTail.Stingray` (`src/OpenTail.Stingray.Audio/`)  
**Execution model:** **100% native managed C# (.NET 10) — no Python, no external binaries, no P/Invoke, no sidecar process**  
**Recommended strategy:** reuse OpenTail's existing native transformer/LLM primitives where they genuinely match the CosyVoice graph; implement the speech-token, flow-matching and vocoder pieces natively rather than trying to force the whole system through a generic text LLM abstraction.

---

# Status

**PLANNED — Native C# implementation**

CosyVoice is materially more complicated than a conventional TTS model. It is not a single Transformer followed by a small waveform decoder.

For CosyVoice 3 the production inference graph is:

```text
                         ┌──────────────────────────────┐
                         │             TEXT             │
                         │ text + optional instruction │
                         │ + optional prompt text      │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         CosyVoice3 / Qwen tokenizer
                                        │
                                        ▼
                         ┌──────────────────────────────┐
Prompt WAV ─────────────►│ speech tokenizer / frontend │
                         │ 25 Hz discrete speech       │
                         │ tokens + speaker embedding  │
                         │ + prompt mel                │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ CosyVoice3LM                 │
                         │ Qwen-based autoregressive    │
                         │ LM                           │
                         │                              │
                         │ text + prompt speech +       │
                         │ speaker/instruction context   │
                         │          ↓                   │
                         │ discrete speech tokens @25Hz │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Conditional Flow Matching    │
                         │ CausalMaskedDiffWithDiT      │
                         │                              │
                         │ speech tokens + prompt mel   │
                         │ + speaker embedding          │
                         │          ↓                   │
                         │ generated mel @50 Hz         │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
                         ┌──────────────────────────────┐
                         │ Causal HiFT / HiFi-GAN        │
                         │ vocoder                       │
                         │                              │
                         │ mel + conditioning            │
                         │          ↓                   │
                         │ 24 kHz PCM waveform           │
                         └──────────────────────────────┘
```

The current CosyVoice repository explicitly maps these stages to:

* `cosyvoice/llm/llm.py` → `CosyVoice3LM`
* `cosyvoice/flow/flow.py` → `CausalMaskedDiffWithDiT`
* `cosyvoice/flow/flow_matching.py` → `CausalConditionalCFM`
* `cosyvoice/flow/DiT/dit.py` → DiT estimator
* `cosyvoice/hifigan/generator.py` → `CausalHiFTGenerator`
* `cosyvoice/cli/frontend.py` → preprocessing / prompt preparation
* `cosyvoice/cli/model.py` → orchestration and streaming

This is the ground-truth implementation map and should be treated as the primary porting map.

---

# 1. Ground Truth Sources — Do Not Implement From Memory

The most important part of this port is **not guessing the architecture from papers or model-card descriptions**.

Use the following hierarchy.

## 1.1 Primary ground truth

### A. CosyVoice source repository

Repository:

```text
https://github.com/QwenAudio/CosyVoice
```

Clone for local investigation:

```bash
git clone --recursive https://github.com/QwenAudio/CosyVoice.git
```

The repository's own roadmap records the release of:

* CosyVoice2-0.5B in December 2024
* CosyVoice2 vLLM support in May 2025
* Fun-CosyVoice 3.0 evaluation material in July 2025
* Fun-CosyVoice3-0.5B-2512 in December 2025

Use the repository at the exact model revision being implemented, not an arbitrary fork.

### B. Exact CosyVoice 3 configuration

Primary configuration:

```text
examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
```

Important values exposed by the released configuration include:

```yaml
sample_rate: 24000
llm_input_size: 896
llm_output_size: 896
spk_embed_dim: 192
token_frame_rate: 25
```

Do not hard-code these values until they have been confirmed against the model checkpoint and configuration being loaded.

### C. CosyVoice architecture document

Use:

```text
docs/architecture.md
```

This is especially useful because it explicitly maps the paper architecture onto the actual production Python modules.

### D. Exact Python inference path

The most important source files are:

```text
cosyvoice/cli/cosyvoice.py
cosyvoice/cli/frontend.py
cosyvoice/cli/model.py

cosyvoice/tokenizer/tokenizer.py

cosyvoice/llm/llm.py

cosyvoice/flow/flow.py
cosyvoice/flow/flow_matching.py
cosyvoice/flow/DiT/dit.py
cosyvoice/flow/DiT/modules.py

cosyvoice/transformer/upsample_encoder.py

cosyvoice/hifigan/generator.py
cosyvoice/hifigan/f0_predictor.py
cosyvoice/hifigan/hifigan.py
```

The port should follow these files function-by-function where numerical behavior matters.

---

# 2. What Makes CosyVoice Different From Existing OpenTail TTS Engines

Do **not** model CosyVoice as simply:

```text
Text → Transformer → Mel → Vocoder
```

The actual system is closer to:

```text
Prompt Audio
    │
    ├──► speech tokenizer ──► discrete speech tokens
    │
    ├──► speaker encoder ────► 192-D speaker embedding
    │
    └──► acoustic frontend ──► prompt mel

Text / instruction
    │
    └──► Qwen tokenizer

Text + speech prompt + speaker embedding
    │
    ▼
Qwen/CosyVoice3 autoregressive LM
    │
    ▼
25 Hz speech token stream
    │
    ▼
Conditional Flow Matching / DiT
    │
    ▼
50 Hz mel
    │
    ▼
Causal HiFT
    │
    ▼
24 kHz waveform
```

This means the implementation should introduce a **CosyVoice-specific pipeline abstraction**, while reusing OpenTail's lower-level Transformer, attention, KV-cache, sampling and audio primitives.

---

# 3. Scope

## 3.1 First target

Implement:

```text
Fun-CosyVoice3-0.5B-2512
```

with:

* text-to-speech
* zero-shot voice cloning from reference WAV
* prompt text
* prompt speech
* speaker embedding
* multilingual generation where supported by the released model
* instruction conditioning
* streaming generation
* deterministic / configurable sampling
* native 24 kHz PCM output

## 3.2 Compatibility target

Design the interfaces so that:

```text
CosyVoice2-0.5B
```

can be supported without duplicating the whole engine.

CosyVoice2 and CosyVoice3 share the high-level idea:

```text
text → speech tokens → acoustic generation → vocoder
```

but the tokenization, Qwen integration, control tokens and acoustic decoder differ.

## 3.3 Explicit non-goals for the first implementation

Do not initially implement:

* training
* LoRA training
* DPO / RL training
* TensorRT
* Triton
* vLLM
* CUDA-specific kernels
* Python compatibility layer
* external `ffmpeg`
* external `sox`
* external `onnxruntime`
* arbitrary ONNX execution

The native runtime should own inference.

---

# 4. Proposed OpenTail Directory Structure

Target:

```text
src/OpenTail.Stingray.Audio/
│
├── ITtsPipeline.cs
├── TtsRequest.cs
├── TtsResult.cs
├── AudioBuffer.cs
├── WavReader.cs
├── WavWriter.cs
│
└── CosyVoice/
    │
    ├── CosyVoiceConfig.cs
    ├── CosyVoiceModel.cs
    ├── CosyVoicePipeline.cs
    ├── CosyVoiceStreamingSession.cs
    │
    ├── Tokenizer/
    │   ├── CosyVoiceTokenizer.cs
    │   ├── CosyVoice3Tokenizer.cs
    │   ├── SpecialTokens.cs
    │   └── PronunciationTags.cs
    │
    ├── Frontend/
    │   ├── CosyVoiceFrontend.cs
    │   ├── TextNormalizer.cs
    │   ├── PromptAudioProcessor.cs
    │   ├── MelExtractor.cs
    │   └── SpeakerEmbeddingExtractor.cs
    │
    ├── SpeechTokens/
    │   ├── SpeechTokenizer.cs
    │   ├── SpeechTokenizerConfig.cs
    │   └── SpeechTokenEncoder.cs
    │
    ├── Llm/
    │   ├── CosyVoice3Lm.cs
    │   ├── CosyVoice2Lm.cs
    │   ├── SpeechTokenEmbedding.cs
    │   ├── SpeechTokenHead.cs
    │   └── CosyVoiceSampler.cs
    │
    ├── Flow/
    │   ├── ConditionalFlowMatching.cs
    │   ├── CausalMaskedDiff.cs
    │   ├── DiT/
    │   │   ├── CosyVoiceDiT.cs
    │   │   ├── DiTBlock.cs
    │   │   ├── Attention.cs
    │   │   └── FeedForward.cs
    │   ├── UpsampleEncoder.cs
    │   └── FlowScheduler.cs
    │
    ├── Vocoder/
    │   ├── CausalHiFtGenerator.cs
    │   ├── HiFiGan.cs
    │   ├── F0Predictor.cs
    │   ├── ResBlock.cs
    │   └── VocoderConfig.cs
    │
    └── ModelIO/
        ├── CosyVoiceModelLoader.cs
        ├── SafetensorsReader.cs
        ├── TensorMap.cs
        └── CosyVoiceManifest.cs
```

If OpenTail already has equivalent primitives, do not create duplicate implementations. Prefer:

```text
OpenTail.Stingray.Audio
        │
        └── CosyVoice
              │
              ├── existing Transformer primitives
              ├── existing tensor primitives
              ├── existing sampling
              ├── existing audio utilities
              └── CosyVoice-specific components
```

---

# 5. Public Pipeline Contract

The existing TTS abstraction should be extended rather than exposing CosyVoice internals.

Conceptually:

```csharp
public sealed record TtsRequest(
    string Text,
    string? Voice = null,
    string? Language = null,
    float Speed = 1.0f,
    float Temperature = 1.0f,
    int? Seed = null,
    ReadOnlyMemory<float>? PromptAudio = null,
    string? PromptText = null,
    string? Instruction = null);

public sealed record TtsResult(
    ReadOnlyMemory<float> Samples,
    int SampleRate,
    int Channels,
    TimeSpan Duration,
    string? VoiceId = null);
```

CosyVoice-specific options should be represented separately:

```csharp
public sealed record CosyVoiceOptions(
    int FlowSteps = 10,
    float Temperature = 1.0f,
    float TopP = 0.7f,
    int TopK = 20,
    int? Seed = null,
    bool Streaming = false,
    bool ZeroShot = false,
    bool RepetitionAwareSampling = true);
```

Do not put every CosyVoice-specific parameter into the global TTS interface.

---

# 6. End-to-End Native Pipeline

The core pipeline should look like:

```csharp
public async IAsyncEnumerable<AudioChunk> SynthesizeAsync(
    TtsRequest request,
    CosyVoiceOptions options,
    [EnumeratorCancellation] CancellationToken cancellationToken = default)
{
    var text = _frontend.NormalizeAndTokenize(
        request.Text,
        request.Language,
        request.Instruction);

    CosyVoicePrompt? prompt = null;

    if (request.PromptAudio is not null)
    {
        prompt = await _frontend.BuildPromptAsync(
            request.PromptAudio.Value,
            request.PromptText,
            cancellationToken);
    }

    var speechTokens = _llm.GenerateSpeechTokens(
        text,
        prompt,
        options,
        cancellationToken);

    await foreach (var chunk in _flow.GenerateMelAsync(
        speechTokens,
        prompt,
        options,
        cancellationToken))
    {
        var audio = _vocoder.Generate(chunk.Mel);

        yield return new AudioChunk(
            audio,
            24000);
    }
}
```

This is deliberately a pipeline rather than a monolithic `Generate()` method.

The reason is important:

```text
LLM token stream
       ↓
flow chunks
       ↓
vocoder chunks
       ↓
audio stream
```

can eventually support true low-latency streaming.

---

# 7. Phase 1 — Repository and Model Forensics

Before writing C# model code, create a small forensic harness.

The goal is to answer:

```text
What tensors exist?
What are their names?
What are their shapes?
What dtype are they?
What config values produced them?
Which Python operation consumes each tensor?
```

Create:

```text
tools/CosyVoiceInspect/
```

The tool should inspect:

```text
*.safetensors
*.yaml
*.json
*.onnx
```

and output a machine-readable manifest:

```json
{
  "model": "Fun-CosyVoice3-0.5B-2512",
  "sampleRate": 24000,
  "tokenFrameRate": 25,
  "llmInputSize": 896,
  "llmOutputSize": 896,
  "speakerEmbeddingDimension": 192
}
```

For every tensor:

```json
{
  "name": "llm.model.layers.0.self_attn.q_proj.weight",
  "shape": [896, 896],
  "dtype": "F16"
}
```

This becomes the mapping document for the native loader.

---

# 8. Phase 2 — Model Format

Do not make the first implementation dependent on converting the entire model into GGUF.

CosyVoice is a composite model.

A single GGUF representation is not necessarily the natural representation for:

```text
Qwen LLM
+
speech tokenizer
+
speaker encoder
+
flow/DiT
+
HiFT vocoder
```

The initial native loader should therefore support the checkpoint representation actually released by CosyVoice.

Recommended:

```text
Safetensors
+
YAML/JSON configuration
+
tokenizer assets
+
small binary resources
```

Then optionally introduce:

```text
CosyVoice native packed format
```

later.

A native packed manifest could look like:

```json
{
  "format": "opentail.cosyvoice",
  "version": 1,
  "sampleRate": 24000,
  "components": {
    "llm": "llm.safetensors",
    "speechTokenizer": "speech_tokenizer.safetensors",
    "speakerEncoder": "speaker_encoder.safetensors",
    "flow": "flow.safetensors",
    "vocoder": "hift.safetensors"
  }
}
```

---

# 9. Phase 3 — Text Tokenizer

CosyVoice 2 uses a Qwen tokenizer with additional special tokens.

The repository explicitly defines tokens including:

```text
<|im_start|>
<|im_end|>
<|endofprompt|>
[breath]
<strong>
</strong>
[noise]
[laughter]
[cough]
[clucking]
[accent]
[quick_breath]
<laughter>
</laughter>
[hissing]
[sigh]
[vocalized-noise]
[lipsmack]
[mn]
```

CosyVoice 3 extends this further with pronunciation controls including Chinese Pinyin and English phoneme tags.

Important rule:

**Do not strip these tokens as ordinary punctuation.**

They are model control inputs.

Ground truth:

```text
cosyvoice/tokenizer/tokenizer.py
```

The repository's tokenizer code should be mirrored by tests.

Example:

```csharp
var ids = tokenizer.Encode(
    "Hello [breath] world!");

Assert.Contains(
    tokenizer.SpecialTokens.Breath,
    ids);
```

For CosyVoice3:

```csharp
var ids = tokenizer.Encode(
    "你好 [laughter] <strong>世界</strong>");

Assert.Contains(
    tokenizer.SpecialTokens.Laughter,
    ids);
```

---

# 10. Phase 4 — Prompt Audio Frontend

Zero-shot voice cloning is one of the important CosyVoice capabilities.

The prompt WAV is not simply fed directly to the LLM.

It supplies several conditioning signals:

```text
prompt WAV
   │
   ├──► speech tokenizer ──► prompt speech tokens
   │
   ├──► acoustic frontend ─► prompt mel
   │
   └──► speaker encoder ───► speaker embedding
```

Create:

```csharp
public sealed record CosyVoicePrompt(
    ReadOnlyMemory<int> SpeechTokens,
    ReadOnlyMemory<float> Mel,
    ReadOnlyMemory<float> SpeakerEmbedding,
    int SampleRate,
    int TokenCount,
    int MelFrames);
```

This structure should be cached.

For repeated synthesis using the same voice:

```text
reference.wav
     ↓
CosyVoicePrompt
     ↓
cache
     ↓
many text generations
```

Avoid recalculating the speech tokenizer and speaker embedding for every sentence.

---

# 11. Phase 5 — Speech Tokenizer

This is a major implementation boundary.

CosyVoice3 uses a speech tokenizer as the interface between waveform audio and discrete speech tokens.

The architecture documentation describes the CosyVoice3 speech tokenizer as an FSQ-based discrete speech representation operating at approximately:

```text
25 speech tokens / second
```

Thus:

```text
4 seconds audio
≈
100 speech tokens
```

The tokenizer is not equivalent to Whisper.

Do not reuse `WhisperEncoder` just because both consume audio.

Instead create:

```csharp
public interface ISpeechTokenizer
{
    SpeechTokenResult Encode(
        ReadOnlySpan<float> samples,
        CancellationToken cancellationToken = default);
}
```

Implementation:

```csharp
public sealed class CosyVoice3SpeechTokenizer
    : ISpeechTokenizer
{
    public SpeechTokenResult Encode(
        ReadOnlySpan<float> samples,
        CancellationToken cancellationToken = default)
    {
        // 1. audio feature extraction
        // 2. voice encoder
        // 3. projection
        // 4. FSQ/discrete bottleneck
        // 5. token IDs
    }
}
```

Important:

The current CosyVoice repository consumes the released speech tokenizer through an ONNX resource (`speech_tokenizer_v3.onnx` / batch form in the runtime path). The tokenizer training code is not simply present as a normal C#-portable module.

Therefore this phase requires **more forensic work than the LLM**.

Ground truth must come from:

```text
cosyvoice/cli/frontend.py
cosyvoice/llm/llm.py
released speech_tokenizer_v3 model
ONNX graph
```

and, where necessary, the CosyVoice3 paper.

Do not invent the tokenizer architecture.

---

# 12. Phase 6 — Speaker Embedding

CosyVoice uses a speaker embedding as global identity conditioning.

The released CosyVoice3 configuration specifies:

```yaml
spk_embed_dim: 192
```

The native component should expose:

```csharp
public interface ISpeakerEmbeddingExtractor
{
    ReadOnlyMemory<float> Extract(
        ReadOnlySpan<float> audio,
        CancellationToken cancellationToken = default);
}
```

Normalize the embedding exactly as the reference implementation does.

The CosyVoice LLM code contains the important pattern:

```python
embedding = F.normalize(embedding, dim=1)
embedding = self.spk_embed_affine_layer(embedding)
embedding = embedding.unsqueeze(dim=1)
```

Native equivalent:

```csharp
embedding = NormalizeRows(embedding);
embedding = _speakerAffine.Apply(embedding);
embedding = embedding.Reshape(1, 1, _config.LlmInputSize);
```

This is an important numerical-conformance test.

---

# 13. Phase 7 — CosyVoice3 Qwen Language Model

CosyVoice3 uses a Qwen-based causal language model.

The reference implementation defines:

```python
class CosyVoice3LM(Qwen2LM):
```

and creates a speech-token output head.

The critical architectural point is:

```text
Qwen hidden state
       ↓
speech-token projection
       ↓
speech token logits
```

rather than ordinary text-token logits.

The reference code creates:

```python
self.llm_decoder = nn.Linear(
    llm_output_size,
    speech_token_size + 200,
    bias=False)
```

and reserves a special speech-token range including:

```text
SOS
EOS
TASK
FILL
```

The native implementation should therefore be:

```csharp
public sealed class CosyVoice3Lm
{
    private readonly QwenCausalModel _llm;
    private readonly Embedding _speechEmbedding;
    private readonly Linear _speechTokenHead;

    public int SpeechTokenSize { get; }

    public int[] GenerateSpeechTokens(
        CosyVoiceTextInput input,
        CosyVoicePrompt? prompt,
        CosyVoiceOptions options,
        CancellationToken cancellationToken)
    {
        // Build exact reference input layout.
        // Run incremental Qwen inference.
        // Project hidden state to speech-token logits.
        // Sample one speech token at a time.
    }
}
```

---

# 14. Critical CosyVoice3 Input Layout

The reference implementation constructs the LLM input from multiple embedding streams.

Conceptually:

```text
[SOS]
[optional speaker embedding]
[text tokens]
[TASK_ID]
[prompt speech tokens]
```

CosyVoice3 additionally has instruction tokens and uses the Qwen text embedding path.

The reference code contains logic equivalent to:

```python
sos_emb = self.speech_embedding.weight[self.sos]
task_id_emb = self.speech_embedding.weight[self.task_id]

prompt_speech_token_emb = self.speech_embedding(
    prompt_speech_token)

lm_input = torch.concat([
    sos_emb,
    text_emb,
    task_id_emb,
    prompt_speech_token_emb
], dim=1)
```

The exact branch for CosyVoice3 includes instruction tokens and the Qwen model's native text embedding.

This must be reproduced exactly.

Do not simplify it to:

```text
Qwen(text)
```

because the conditioning layout is part of the trained model.

---

# 15. LLM Generation Loop

Native generation should reuse OpenTail's existing:

* KV cache
* sampling parameters
* session state
* speculative decoding infrastructure where applicable
* token streaming
* cancellation
* deterministic RNG

Conceptual implementation:

```csharp
while (!finished)
{
    var hidden = _llm.DecodeNext(
        inputEmbedding,
        kvCache);

    var logits = _speechTokenHead.Project(hidden);

    var token = _sampler.Sample(
        logits,
        options.Temperature,
        options.TopK,
        options.TopP);

    if (IsEndToken(token))
        break;

    yield return token;

    inputEmbedding =
        _speechEmbedding[token];
}
```

The key optimization is:

```text
first token:
    process full conditioning sequence

subsequent tokens:
    process one token + KV cache
```

Do not recompute the entire sequence.

---

# 16. Repetition Aware Sampling

CosyVoice explicitly introduced Repetition Aware Sampling (RAS) for stability.

Therefore it should not be silently omitted from a high-quality native implementation.

Create:

```csharp
public sealed class CosyVoiceRepetitionAwareSampler
{
    public int Select(
        ReadOnlySpan<float> logits,
        ReadOnlySpan<int> recentTokens,
        CosyVoiceSamplingOptions options)
    {
        // Reference RAS algorithm.
    }
}
```

Do not approximate the algorithm from the README.

Retrieve the exact implementation from the CosyVoice inference code and port it literally first.

Only optimize after golden-output tests exist.

---

# 17. Phase 8 — Speech Token → Mel

This is the second major model.

The reference CosyVoice3 architecture uses:

```text
speech tokens
     │
     ▼
upsample / conditioning encoder
     │
     ▼
DiT / conditional flow matching
     │
     ▼
mel spectrogram
```

The architecture documentation identifies:

```text
cosyvoice/flow/flow.py
cosyvoice/flow/flow_matching.py
cosyvoice/flow/DiT/dit.py
cosyvoice/flow/DiT/modules.py
cosyvoice/transformer/upsample_encoder.py
```

as the relevant implementation files.

---

# 18. Conditional Flow Matching

This is **not ordinary diffusion sampling** and should not be implemented by copying a Stable Diffusion scheduler.

The reference implementation uses a conditional flow-matching decoder.

Conceptually:

```text
random noise x0
     │
     │ ODE / flow integration
     ▼
mel distribution x1
```

conditioned on:

```text
speech tokens
+
prompt mel
+
speaker embedding
+
mask
```

The reference code passes:

```python
decoder(
    mu=h.transpose(1, 2).contiguous(),
    mask=mask.unsqueeze(1),
    spks=embedding,
    cond=conds,
    n_timesteps=10,
    streaming=streaming
)
```

The native API should therefore expose:

```csharp
public ReadOnlyMemory<float> GenerateMel(
    ReadOnlySpan<int> speechTokens,
    ReadOnlySpan<float> promptMel,
    ReadOnlySpan<float> speakerEmbedding,
    CosyVoiceFlowOptions options);
```

---

# 19. Flow Scheduler

Implement the scheduler as a separate class:

```csharp
public sealed class FlowScheduler
{
    public ReadOnlySpan<float> Timesteps =>
        _timesteps;

    public FlowState Initialize(
        TensorShape shape,
        Random rng);

    public FlowState Step(
        FlowState state,
        ReadOnlySpan<float> velocity,
        float timestep);
}
```

Start with the exact number of reference inference steps.

The released reference currently shows:

```text
n_timesteps = 10
```

in the relevant CosyVoice3 flow path.

Make this configurable:

```csharp
FlowSteps = 10
```

but default to the reference value.

---

# 20. DiT Implementation

The DiT model should be implemented from:

```text
cosyvoice/flow/DiT/dit.py
cosyvoice/flow/DiT/modules.py
```

Do not substitute a generic Transformer block.

The native structure should be explicit:

```csharp
public sealed class CosyVoiceDiT
{
    private readonly DiTBlock[] _blocks;

    public Tensor Forward(
        Tensor x,
        Tensor timestep,
        Tensor condition,
        Tensor speakerEmbedding,
        Tensor mask)
    {
        foreach (var block in _blocks)
        {
            x = block.Forward(
                x,
                timestep,
                condition,
                speakerEmbedding,
                mask);
        }

        return x;
    }
}
```

The exact:

* normalization
* timestep embedding
* attention
* masking
* conditioning
* residual structure
* feed-forward activation

must be copied from the reference code and verified tensor-by-tensor.

---

# 21. Causal / Streaming Flow

CosyVoice supports streaming.

The native implementation should not begin with streaming complexity everywhere.

Implement:

```text
Phase A:
offline flow

Phase B:
chunked causal flow

Phase C:
true streaming audio output
```

For chunking:

```text
speech tokens
    ↓
chunk
    ↓
flow
    ↓
mel chunk
    ↓
vocoder
    ↓
audio chunk
```

The reference code's `static_chunk_size`, masks and `streaming` arguments are the ground truth for the exact chunk semantics.

Do not invent chunk boundaries.

---

# 22. Phase 9 — Causal HiFT Vocoder

The final waveform generator is a substantial native component.

CosyVoice's production code uses:

```text
cosyvoice/hifigan/generator.py
cosyvoice/hifigan/f0_predictor.py
```

and related HiFi-GAN / HiFT components.

The architecture is conceptually:

```text
mel
 │
 ▼
upsampling
 │
 ▼
residual blocks
 │
 ▼
harmonic / F0 conditioning
 │
 ▼
HiFT synthesis
 │
 ▼
24 kHz waveform
```

Do not use a generic OpenTail HiFi-GAN merely because the name is similar.

The exact convolution topology, upsampling ratios, residual blocks, F0 processing and conditioning have to match the CosyVoice checkpoint.

---

# 23. Vocoder API

Implement:

```csharp
public interface IWaveformVocoder
{
    ReadOnlyMemory<float> Generate(
        ReadOnlySpan<float> mel,
        CancellationToken cancellationToken = default);
}
```

CosyVoice:

```csharp
public sealed class CausalHiFtGenerator
    : IWaveformVocoder
{
    public ReadOnlyMemory<float> Generate(
        ReadOnlySpan<float> mel,
        CancellationToken cancellationToken = default)
    {
        // Exact native port of generator.py
    }
}
```

For streaming:

```csharp
public ReadOnlyMemory<float> GenerateChunk(
    ReadOnlySpan<float> melChunk,
    VocoderState state);
```

`VocoderState` must retain the exact convolutional history needed for causal continuity.

---

# 24. Phase 10 — Mel / Audio Conventions

CosyVoice3 configuration specifies:

```text
sample rate = 24000 Hz
token frame rate = 25 Hz
```

The acoustic side operates at a higher frame rate.

The architecture documentation describes:

```text
25 Hz speech tokens
        ↓
50 Hz mel
        ↓
24 kHz waveform
```

These rates must become explicit configuration rather than magic numbers:

```csharp
public sealed record CosyVoiceAudioConfig(
    int SampleRate = 24000,
    int SpeechTokenRate = 25,
    int MelFrameRate = 50);
```

Every conversion should have tests.

For example:

```csharp
Assert.Equal(
    expectedFrames,
    SecondsToMelFrames(seconds));
```

and:

```csharp
Assert.Equal(
    expectedTokens,
    SecondsToSpeechTokens(seconds));
```

---

# 25. Phase 11 — Text Normalization

CosyVoice3 deliberately supports text normalization for:

* numbers
* symbols
* mixed text
* multilingual input

The native implementation should not assume English-only text.

Create:

```csharp
public interface ITextNormalizer
{
    string Normalize(
        string text,
        string? language,
        string? instruction);
}
```

Do not attempt to port every training-time text-processing dependency immediately.

First reproduce the inference frontend behavior for common inputs.

Add explicit tests:

```text
123
£10.50
3.14
2026-08-21
URLs
mixed Chinese/English
punctuation
phoneme tags
Pinyin tags
```

---

# 26. Phase 12 — Instruction Conditioning

CosyVoice3 supports instruction-style controls.

Examples include concepts such as:

```text
language
dialect
emotion
speed
volume
style
```

The instruction is not merely a UI parameter.

It becomes part of the model conditioning sequence.

Expose:

```csharp
TtsRequest.Instruction
```

but preserve it through:

```text
TtsRequest
    ↓
CosyVoiceFrontend
    ↓
instruction tokens
    ↓
Qwen embedding
    ↓
CosyVoice3LM
```

---

# 27. Phase 13 — Streaming Architecture

Streaming should be designed around three asynchronous boundaries:

```text
LLM
 │
 │ speech token stream
 ▼
Flow
 │
 │ mel chunk stream
 ▼
Vocoder
 │
 │ PCM chunk stream
 ▼
Client
```

Use:

```csharp
IAsyncEnumerable<AudioChunk>
```

rather than:

```csharp
byte[] Generate(...)
```

for the streaming path.

Example:

```csharp
await foreach (var audio in pipeline.SynthesizeStreamingAsync(
    request,
    options,
    cancellationToken))
{
    await output.WriteAsync(
        audio.Samples,
        cancellationToken);
}
```

This integrates naturally with OpenTail's existing cancellation and session architecture.

---

# 28. Phase 14 — Model Residency

CosyVoice is composite.

Treat the components as independently measurable memory residents:

```text
CosyVoice3
├── LLM
├── speech tokenizer
├── speaker encoder
├── flow/DiT
└── vocoder
```

This should integrate with the existing `ModelRuntimeManager`.

A model residency record could expose:

```csharp
public sealed record CosyVoiceResidency(
    long LlmBytes,
    long SpeechTokenizerBytes,
    long SpeakerEncoderBytes,
    long FlowBytes,
    long VocoderBytes)
{
    public long TotalBytes =>
        LlmBytes +
        SpeechTokenizerBytes +
        SpeakerEncoderBytes +
        FlowBytes +
        VocoderBytes;
}
```

This is important because a TTS request should not blindly load every component if only a subset is needed.

---

# 29. Phase 15 — Quantization

Do not start by quantizing everything.

Recommended order:

```text
1. F32 reference
2. F16/BF16 where numerically safe
3. LLM quantization
4. DiT quantization
5. vocoder optimization
6. speech-tokenizer optimization
```

The most likely high-value target is the Qwen-based LLM.

The flow and vocoder may be more sensitive.

Every quantized component needs:

```text
FP32/F32 golden output
        vs
quantized output
```

comparison.

---

# 30. Phase 16 — Existing OpenTail Primitive Reuse

Before implementing any low-level kernel, search the OpenTail codebase for:

```text
RMSNorm
LayerNorm
RoPE
Qwen
Qwen2
CausalAttention
KV cache
Linear
Conv1D
ConvTranspose1D
GroupNorm
SiLU
GELU
MultiHeadAttention
FlashAttention
Softmax
Embedding
Safetensors
```

The correct implementation strategy is:

```text
CosyVoice-specific graph
        │
        ├── existing Qwen implementation
        ├── existing attention
        ├── existing KV cache
        ├── existing tensor operations
        ├── existing sampling
        └── new CosyVoice-only layers
```

Avoid creating:

```text
CosyVoiceQwen2
CosyVoiceAttention
CosyVoiceTensor
CosyVoiceLinear
```

if OpenTail already has equivalent components.

---

# 31. Phase 17 — Numerical Ground Truth Harness

This is essential.

Create:

```text
src/OpenTail.Stingray.Tests.Audio/CosyVoice/
```

and:

```text
tools/CosyVoiceGolden/
```

The golden harness should run the official Python model and dump intermediate tensors.

For example:

```text
golden/
  text_tokens.bin
  prompt_speech_tokens.bin
  speaker_embedding.bin
  llm_input.bin
  llm_hidden_0.bin
  speech_logits_0.bin
  speech_tokens.bin
  flow_condition.bin
  flow_step_00.bin
  flow_step_01.bin
  ...
  mel.bin
  waveform.bin
```

Then C# runs the same input and compares.

---

# 32. Golden Tensor Comparison

Use relative + absolute tolerance.

Example:

```csharp
static void AssertClose(
    ReadOnlySpan<float> expected,
    ReadOnlySpan<float> actual,
    float atol = 1e-4f,
    float rtol = 1e-3f)
{
    Assert.Equal(expected.Length, actual.Length);

    for (int i = 0; i < expected.Length; i++)
    {
        var diff = MathF.Abs(
            expected[i] - actual[i]);

        var tolerance =
            atol +
            rtol * MathF.Abs(expected[i]);

        Assert.True(
            diff <= tolerance,
            $"Mismatch at {i}: expected={expected[i]}, actual={actual[i]}, diff={diff}");
    }
}
```

Do not compare only final WAV output.

If the waveform differs, you need to know whether the error began in:

```text
tokenizer
LLM
flow
vocoder
```

---

# 33. Phase 18 — Deterministic Seed

The reference configuration explicitly seeds:

```python
random.seed(...)
numpy.random.seed(...)
torch.manual_seed(...)
torch.cuda.manual_seed_all(...)
```

Native C# must have a clearly defined RNG.

```csharp
var rng = new CosyVoiceRandom(seed);
```

Use it for:

* speech-token sampling
* flow initialization
* any stochastic frontend component

A deterministic test should be:

```csharp
var a = Generate(request, new CosyVoiceOptions(Seed: 1234));
var b = Generate(request, new CosyVoiceOptions(Seed: 1234));

Assert.Equal(a.Samples.ToArray(), b.Samples.ToArray());
```

---

# 34. Phase 19 — First End-to-End Milestone

Do not make the first milestone zero-shot streaming.

First target:

```text
Text
 ↓
CosyVoice3 tokenizer
 ↓
Qwen/CosyVoice3 LM
 ↓
speech tokens
 ↓
flow
 ↓
mel
 ↓
HiFT
 ↓
WAV
```

with:

```text
no prompt audio
no streaming
deterministic seed
F32
```

This reduces the debugging surface.

---

# 35. Phase 20 — Zero-Shot Voice Cloning

Second milestone:

```text
reference.wav
+
reference text
+
new text
        ↓
prompt preparation
        ↓
speech tokens + speaker embedding + prompt mel
        ↓
CosyVoice3 LM
        ↓
flow
        ↓
vocoder
        ↓
cloned voice
```

Test with a fixed reference clip.

Store the reference-derived tensors as fixtures:

```text
Fixtures/
  speaker_embedding.bin
  prompt_speech_tokens.bin
  prompt_mel.bin
```

This allows debugging without repeatedly processing audio.

---

# 36. Phase 21 — Streaming

Third milestone:

```text
text
 ↓
LLM tokens
 ↓
flow chunks
 ↓
vocoder chunks
 ↓
PCM chunks
```

Measure:

```text
time-to-first-token
time-to-first-mel
time-to-first-audio
real-time factor
```

Expose:

```csharp
CosyVoiceMetrics
```

with:

```csharp
public sealed record CosyVoiceMetrics(
    TimeSpan TimeToFirstSpeechToken,
    TimeSpan TimeToFirstMel,
    TimeSpan TimeToFirstAudio,
    double RealTimeFactor,
    int SpeechTokensGenerated,
    int MelFramesGenerated);
```

---

# 37. Phase 22 — CLI

Add a command such as:

```bash
stingray tts cosyvoice \
    --model Fun-CosyVoice3-0.5B-2512 \
    --text "Hello from OpenTail." \
    --output output.wav
```

Zero-shot:

```bash
stingray tts cosyvoice \
    --model Fun-CosyVoice3-0.5B-2512 \
    --prompt voice.wav \
    --prompt-text "This is the reference recording." \
    --text "This is the generated sentence." \
    --output cloned.wav
```

Instruction:

```bash
stingray tts cosyvoice \
    --model Fun-CosyVoice3-0.5B-2512 \
    --instruction "Speak warmly and slowly." \
    --text "Hello." \
    --output warm.wav
```

Streaming:

```bash
stingray tts cosyvoice \
    --model Fun-CosyVoice3-0.5B-2512 \
    --text "A long sentence..." \
    --stream
```

---

# 38. Phase 23 — Server

Expose an OpenAI-compatible TTS surface where possible.

For example:

```http
POST /v1/audio/speech
Content-Type: application/json
```

Request:

```json
{
  "model": "cosyvoice3",
  "input": "Hello from OpenTail.",
  "voice": "default"
}
```

CosyVoice-specific extension:

```json
{
  "model": "cosyvoice3",
  "input": "Hello from OpenTail.",
  "voice": "clone",
  "prompt_audio": "...",
  "prompt_text": "Reference text",
  "instruction": "Speak warmly."
}
```

Keep OpenAI compatibility at the HTTP boundary.

Do not distort the internal CosyVoice model around the compatibility API.

---

# 39. Phase 24 — Tests

Minimum test groups:

```text
CosyVoiceTokenizerTests
CosyVoiceFrontendTests
CosyVoiceSpeechTokenizerTests
CosyVoiceSpeakerEmbeddingTests
CosyVoiceLmTests
CosyVoiceSamplingTests
CosyVoiceFlowTests
CosyVoiceDiTTests
CosyVoiceVocoderTests
CosyVoicePipelineTests
CosyVoiceStreamingTests
CosyVoiceModelLoaderTests
CosyVoiceGoldenTests
```

---

# 40. Required Unit Tests

## Tokenizer

```csharp
[Fact]
public void EndOfPrompt_IsPreserved()
{
    var ids = _tokenizer.Encode(
        "Hello <|endofprompt|>");

    Assert.Contains(
        _tokenizer.EndOfPromptId,
        ids);
}
```

## Speaker embedding

```csharp
[Fact]
public void SpeakerEmbedding_IsNormalizedLikeReference()
{
    var actual =
        _speaker.Extract(_fixtureAudio);

    AssertClose(
        _goldenEmbedding,
        actual.Span,
        atol: 1e-4f,
        rtol: 1e-3f);
}
```

## LLM first step

```csharp
[Fact]
public void Llm_FirstHiddenState_MatchesGolden()
{
    var actual =
        _llm.DebugFirstStep(_fixtureInput);

    AssertClose(
        _goldenHidden,
        actual,
        1e-4f,
        1e-3f);
}
```

## Speech token generation

```csharp
[Fact]
public void SpeechTokens_AreDeterministic()
{
    var actual = _pipeline.GenerateTokens(
        _fixtureRequest,
        new CosyVoiceOptions(Seed: 1234));

    Assert.Equal(
        _goldenTokens,
        actual);
}
```

## Flow

```csharp
[Fact]
public void Flow_FirstStep_MatchesGolden()
{
    var actual =
        _flow.DebugStep(
            _fixtureFlowInput,
            timestep: 0);

    AssertClose(
        _goldenFlowStep,
        actual);
}
```

## Vocoder

```csharp
[Fact]
public void Vocoder_Output_MatchesGolden()
{
    var actual =
        _vocoder.Generate(_fixtureMel);

    AssertClose(
        _goldenWaveform,
        actual.Span,
        atol: 2e-4f,
        rtol: 2e-3f);
}
```

---

# 41. Phase 25 — End-to-End Acceptance Test

Use one fixed reference prompt and one fixed sentence.

Example:

```text
Reference:
reference.wav

Reference text:
"Hello, this is a reference recording."

Target:
"Hello, this is a native OpenTail CosyVoice test."
```

Acceptance criteria:

```text
✓ model loads without Python
✓ model loads without P/Invoke
✓ text token IDs match reference
✓ prompt speech tokens match reference
✓ speaker embedding matches reference
✓ first LLM hidden state matches
✓ speech tokens match under deterministic sampling
✓ flow intermediate matches tolerance
✓ mel output matches tolerance
✓ waveform is valid 24 kHz PCM
✓ cloned speaker identity is preserved
✓ streaming produces continuous audio
```

---

# 42. Phase 26 — Performance Benchmarks

Record:

```text
model load time
prompt processing time
LLM tokens/sec
flow steps/sec
vocoder real-time factor
total RTF
peak RAM
resident RAM by component
time-to-first-audio
```

Example benchmark record:

```csharp
public sealed record CosyVoiceBenchmark(
    TimeSpan ModelLoad,
    TimeSpan PromptEncode,
    double LlmTokensPerSecond,
    double FlowStepsPerSecond,
    double RealTimeFactor,
    long PeakBytes,
    TimeSpan TimeToFirstAudio);
```

Benchmark:

```text
10 seconds generated audio
100 seconds generated audio
short sentence
long paragraph
with prompt
without prompt
streaming
offline
```

---

# 43. Phase 27 — CPU-First Optimization

The initial implementation should be correct on CPU before attempting accelerator specialization.

Prioritize:

```text
1. KV-cache reuse
2. memory reuse
3. contiguous tensor layouts
4. avoiding allocations inside token loop
5. fused QKV where existing primitives support it
6. batched linear layers
7. efficient Conv1D
8. flow step reuse
9. vocoder state reuse
```

Do not prematurely introduce unsafe SIMD.

The correctness reference remains the scalar/native tensor implementation.

---

# 44. Phase 28 — Memory Optimization

Avoid materializing unnecessary copies of:

```text
text embeddings
speech token embeddings
flow conditions
prompt mel
speaker embedding
KV cache
vocoder history
```

Use pooled buffers where appropriate.

Example:

```csharp
using var workspace =
    _workspacePool.Rent(requiredBytes);

var flowState =
    new FlowState(workspace);
```

The existing OpenTail memory-management approach should be reused.

---

# 45. Important Numerical Risks

The highest-risk areas are:

## 45.1 Qwen compatibility

Small differences in:

```text
RoPE
RMSNorm
attention scaling
masking
embedding layout
KV cache position
```

will alter every later speech token.

## 45.2 Speech-token embedding space

Do not accidentally use the ordinary Qwen vocabulary for speech-token generation.

## 45.3 Flow integration

Small numerical differences accumulate across flow steps.

## 45.4 Vocoder convolution state

Streaming boundaries can introduce audible discontinuities.

## 45.5 Mel conventions

A wrong:

```text
window
hop
normalization
mel filter
log scaling
padding
```

can produce a completely different acoustic result.

---

# 46. Debugging Strategy

When audio sounds wrong, classify the failure.

### No / nonsense speech tokens

Investigate:

```text
text tokenizer
conditioning layout
Qwen model
speech embedding
sampling
KV cache
```

### Correct speech tokens but bad mel

Investigate:

```text
token upsampling
flow conditioning
DiT
speaker embedding
flow scheduler
mel normalization
```

### Correct mel but bad waveform

Investigate:

```text
HiFT
F0 predictor
upsampling
residual blocks
convolution padding
```

### Offline good, streaming bad

Investigate:

```text
chunk masks
KV state
flow state
vocoder state
overlap / boundary handling
```

---

# 47. Do Not Use Final Audio as the First Debug Signal

This is a major rule.

Bad workflow:

```text
Run model
↓
sounds wrong
↓
change random layer
↓
try again
```

Correct workflow:

```text
input tokens
↓
prompt tensors
↓
speaker embedding
↓
LLM hidden
↓
speech logits
↓
speech tokens
↓
flow condition
↓
flow step 0
↓
flow step N
↓
mel
↓
vocoder
↓
waveform
```

Compare each boundary against Python ground truth.

---

# 48. Model Download / Distribution

Official CosyVoice documentation currently exposes Hugging Face model downloads such as:

```text
FunAudioLLM/Fun-CosyVoice3-0.5B-2512
FunAudioLLM/CosyVoice2-0.5B
FunAudioLLM/CosyVoice-300M
```

and ModelScope equivalents.

The OpenTail model registry should describe the components instead of assuming one file:

```json
{
  "id": "Fun-CosyVoice3-0.5B-2512",
  "type": "tts",
  "sampleRate": 24000,
  "components": [
    "llm",
    "speech-tokenizer",
    "speaker-encoder",
    "flow",
    "vocoder"
  ]
}
```

---

# 49. Licensing / Third-Party Notices

The CosyVoice source files carry Alibaba copyright and Apache License 2.0 notices.

Before merging code:

```text
inspect every copied/reference-derived source file
record copyright
record license
record source URL
```

Add an appropriate section to:

```text
THIRD_PARTY_NOTICES.md
```

Do not copy Python code verbatim into C# without recording the source and license.

A useful notice entry:

```text
CosyVoice
Copyright Alibaba Inc. / FunAudioLLM contributors
Apache License 2.0
Source: https://github.com/QwenAudio/CosyVoice
```

Verify the exact license at the revision being ported.

---

# 50. Recommended Implementation Order

This is the order that minimizes wasted work.

```text
PHASE 0
Repository + checkpoint forensic inspection

PHASE 1
Safetensors/config/model manifest support

PHASE 2
CosyVoice tokenizer + special-token handling

PHASE 3
Reuse/validate OpenTail Qwen2 infrastructure

PHASE 4
CosyVoice3 LM conditioning + speech-token head

PHASE 5
Deterministic speech-token generation

PHASE 6
Speech tokenizer / prompt encoder

PHASE 7
Speaker embedding

PHASE 8
Mel frontend / prompt mel

PHASE 9
Flow-matching infrastructure

PHASE 10
CosyVoice DiT

PHASE 11
Token-to-mel end-to-end

PHASE 12
Causal HiFT vocoder

PHASE 13
Text → waveform offline

PHASE 14
Zero-shot voice cloning

PHASE 15
Streaming

PHASE 16
CLI/server

PHASE 17
Quantization

PHASE 18
Performance tuning
```

---

# 51. Suggested Commit Structure

Keep the port bisectable.

```text
feat(audio): add CosyVoice model manifest support

feat(audio): add CosyVoice3 tokenizer

feat(audio): add CosyVoice3 Qwen speech-token head

feat(audio): add CosyVoice3 conditioning layout

feat(audio): add CosyVoice3 speech-token generation

feat(audio): add CosyVoice3 prompt speech tokenizer

feat(audio): add CosyVoice3 speaker embedding

feat(audio): add CosyVoice3 flow matching

feat(audio): add CosyVoice3 DiT

feat(audio): add CosyVoice3 HiFT vocoder

feat(audio): add CosyVoice3 offline pipeline

feat(audio): add CosyVoice3 zero-shot cloning

feat(audio): add CosyVoice3 streaming

feat(cli): add cosyvoice tts command

feat(server): add CosyVoice TTS endpoint

test(audio): add CosyVoice golden tensor suite

perf(audio): optimize CosyVoice CPU inference
```

---

# 52. Definition of Done

CosyVoice support should not be considered complete merely because a WAV file is produced.

## Functional

```text
✓ CosyVoice3 model loads natively
✓ no Python runtime
✓ no external executable
✓ no P/Invoke
✓ text synthesis works
✓ 24 kHz PCM output
✓ prompt voice cloning works
✓ speaker embedding works
✓ CosyVoice3 control tokens work
✓ deterministic generation works
✓ configurable sampling works
✓ streaming works
```

## Numerical

```text
✓ tokenizer matches reference
✓ speaker embedding matches reference
✓ LLM first-step tensors match
✓ speech-token sequence matches under fixed seed
✓ flow intermediate tensors match tolerance
✓ mel tensors match tolerance
✓ vocoder output matches tolerance
```

## OpenTail integration

```text
✓ model residency integrates with ModelRuntimeManager
✓ cancellation works
✓ session lifecycle is clean
✓ no per-token allocations in hot path
✓ audio streaming uses existing abstractions
✓ CLI works
✓ server works
✓ tests pass
```

---

# 53. Critical Design Decision — Do Not Force CosyVoice Into a Generic TTS Shape

The generic API should remain simple:

```text
Text
 ↓
TTS
 ↓
Audio
```

but the internal runtime should expose the real structure:

```text
CosyVoiceRuntime
│
├── Frontend
├── SpeechTokenizer
├── SpeakerEncoder
├── LLM
├── Flow
└── Vocoder
```

This makes future models much easier to add.

For example:

```text
CosyVoice
F5-TTS
VibeVoice
IndexTTS
Spark-TTS
```

can eventually share:

```text
ITtsPipeline
IAudioPromptEncoder
ISpeechTokenizer
IFlowDecoder
IWaveformVocoder
```

without pretending their graphs are identical.

---

# 54. Particularly Valuable OpenTail Reuse

CosyVoice is a good stress test for the OpenTail architecture because it combines several capabilities in one model family.

Reuse existing OpenTail work for:

```text
Qwen model execution
GGUF / tensor loading where applicable
Safetensors loading
KV caching
session state
sampling
memory residency
streaming
audio buffers
model lifecycle
CLI
server
benchmarking
golden tests
```

New CosyVoice-specific work should mainly be:

```text
speech tokenizer
speaker encoder
conditioning format
speech-token head
flow matching
DiT
HiFT
CosyVoice frontend
```

---

# 55. Ground-Truth Reading Checklist

Before coding each component, inspect these exact sources.

## LLM

```text
cosyvoice/llm/llm.py
```

Questions:

```text
What is the exact input embedding order?
What IDs are reserved?
How are prompt speech tokens embedded?
Where is speaker embedding inserted?
How is EOS represented?
What exactly is sampled?
How does streaming update KV cache?
```

## Frontend

```text
cosyvoice/cli/frontend.py
```

Questions:

```text
How is text normalized?
How is prompt audio processed?
What tensors are created?
What sample rate is assumed?
How is prompt text represented?
```

## Flow

```text
cosyvoice/flow/flow.py
cosyvoice/flow/flow_matching.py
```

Questions:

```text
What is mu?
What is cond?
What is spks?
What is mask?
What is the initial noise?
What timestep schedule is used?
What does streaming change?
```

## DiT

```text
cosyvoice/flow/DiT/dit.py
cosyvoice/flow/DiT/modules.py
```

Questions:

```text
What normalization?
What positional encoding?
What attention?
What timestep conditioning?
What residual layout?
What activation?
```

## Vocoder

```text
cosyvoice/hifigan/generator.py
cosyvoice/hifigan/f0_predictor.py
```

Questions:

```text
What is the exact mel shape?
What is the hop size?
What is the upsampling ratio?
How is F0 obtained?
What state is retained in streaming?
```

---

# 56. Final Recommended Architecture

The completed native implementation should look approximately like:

```text
OpenTail.Stingray.Audio
│
├── ITtsPipeline
│
└── CosyVoice
    │
    ├── CosyVoicePipeline
    │
    ├── Frontend
    │   ├── TextNormalizer
    │   ├── CosyVoiceTokenizer
    │   ├── PromptAudioProcessor
    │   └── MelExtractor
    │
    ├── Prompt
    │   ├── SpeechTokenizer
    │   └── SpeakerEncoder
    │
    ├── LLM
    │   ├── Qwen runtime
    │   ├── SpeechEmbedding
    │   ├── SpeechTokenHead
    │   └── RAS sampler
    │
    ├── Flow
    │   ├── ConditionalFlowMatching
    │   ├── UpsampleEncoder
    │   ├── DiT
    │   └── FlowScheduler
    │
    ├── Vocoder
    │   ├── CausalHiFT
    │   ├── F0Predictor
    │   └── HiFiGAN blocks
    │
    └── Runtime
        ├── StreamingSession
        ├── ModelResidency
        ├── Metrics
        └── ModelLoader
```

The important architectural principle is:

```text
             CosyVoice
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
      LLM       Flow     Vocoder
       │         │         │
       ▼         ▼         ▼
  speech IDs    mel      PCM
       ▲         ▲
       │         │
 speech tokenizer
       ▲
       │
 prompt WAV
```

That is the native implementation boundary to preserve.

---

# 57. Short Version — What Actually Has To Be Written

The minimum serious native port is **not** just a Qwen TTS wrapper.

It requires these major new native components:

```text
1. CosyVoice3 tokenizer/control vocabulary
2. prompt audio frontend
3. speech tokenizer
4. speaker embedding
5. CosyVoice3 Qwen conditioning
6. speech-token output head
7. CosyVoice sampling/RAS
8. token → mel upsampling encoder
9. conditional flow matching
10. CosyVoice DiT
11. causal HiFT vocoder
12. streaming state for flow + vocoder
13. composite model loader
14. golden tensor test harness
```

The good news is that a substantial portion of the infrastructure should already exist in OpenTail:

```text
Qwen execution
attention
KV cache
sampling
tensor primitives
audio buffers
model management
streaming
CLI
server
```

So the project should be treated as a **native composite TTS model port**, not as a new general-purpose inference engine.

---

# 58. Key References

Official CosyVoice repository:

```text
https://github.com/QwenAudio/CosyVoice
```

Architecture guide:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/docs/architecture.md
```

CosyVoice3 LLM:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/llm/llm.py
```

CosyVoice flow:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/flow.py
```

Flow matching:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py
```

DiT:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/flow/DiT/dit.py
```

Vocoder:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/hifigan/generator.py
```

Tokenizer:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/cosyvoice/tokenizer/tokenizer.py
```

CosyVoice3 configuration:

```text
https://github.com/QwenAudio/CosyVoice/blob/main/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml
```

Model:

```text
https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
```

---

# 59. Final Implementation Recommendation

Implement **CosyVoice3 first**, not CosyVoice1.

CosyVoice3 gives OpenTail a modern architecture with:

```text
Qwen-based LLM
+
25 Hz discrete speech tokens
+
speaker conditioning
+
conditional flow matching / DiT
+
causal HiFT
+
streaming
+
instruction conditioning
```

It is therefore a substantially more useful architectural addition than porting the older 300M implementation first.

However, the port should be staged so that each major component can be independently verified against the official Python implementation.

The governing rule for the entire project is:

> **Every native component gets a Python-reference tensor test before it gets optimized.**

That will make this port considerably safer than trying to reproduce the audible output directly.

