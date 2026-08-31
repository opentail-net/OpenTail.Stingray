# 053 — Vulkan `--backend` audit of the remaining audio engines (TTS + ASR)

## User instruction (same standing instruction as 052, extended to the rest of the audio stack)

> "good, do the others. make sure you make a SIMILAR plan for those. still only as a parameter
> option plz"

Following [052](052-vulkan-backend-for-tts-engines-plan.md)'s pattern exactly: audit each engine's
real hot-path call shape first (per CLAUDE.md rule 8 — check the real code, don't assume the
Chatterbox/CosyVoice3/F5-TTS recipe transfers), port what's genuinely tractable as an explicit
`--backend vulkan` opt-in (never gated on this machine's weak iGPU winning), and honestly document
what isn't tractable and why, rather than forcing a port that's likely wrong or actively harmful.

## Result up front

**Orpheus-TTS is the only remaining engine with real GPU support, and it already had it before
this session** (`OrpheusPipeline.cs` already tries CUDA then falls back to real
`GpuForwardPass`/`VulkanBackend` — the working reference example this whole audit was checked
against). Every other remaining engine hits one of three real, structural blockers below. None
were ported this pass — not for lack of time, but because each would either need a real
restructuring the CPU-optimized code deliberately avoided, or a change to a shared core engine
class (`GpuForwardPass`) too invasive for a single pipeline's `--backend` option.

## The three blockers, found across the audit

### Blocker A — per-row `MatVec`-inside-`Parallel.For`, not batched `Linear`/`Sgemm`

The Chatterbox/CosyVoice3/F5-TTS port (052) worked because those DiT/UNet kernels already called
a single batched `Linear(x, t, inDim, weight, bias, outDim)` per weight — trivial to swap for one
`Sgemm` dispatch. Most of the remaining engines instead parallelize over rows on the CPU and call
a single-vector `MatVec` per row:

```csharp
Parallel.For(0, t, i => output[i] = SomeKernels.MatVec(x[i], weight, ...));
```

Turning this into GPU dispatch either means *t* separate GPU calls (far worse than not porting —
this is exactly Parler's T5 encoder blocker from 052 §5), or restructuring the kernel to batch all
`t` rows into one call first — a real, non-trivial refactor, and **`FunAsrEncoder.cs`'s own code
comment states this per-row `Parallel.For` pattern was already measured faster than a naive
batched approach on CPU** (`src/OpenTail.Stingray.Audio/FunASR/FunAsrEncoder.cs:88-89`) — meaning
the restructure needed for GPU could plausibly *regress* the CPU path too if done carelessly.

**Confirmed present in**: `FunAsrEncoder.cs`, `ParakeetConformerEncoder.cs`,
`WhisperEncoder.cs`/`WhisperDecoder.cs`, `KokoroDecoder.cs`/`KokoroProsodyPredictor.cs`,
`MeloGenerator.cs`/`MeloDurationPredictor.cs`, `XttsGptTrunk.cs`, `Piper*.cs` (VITS-family,
additionally conv-heavy — see Blocker C).

### Blocker B — `ForwardPass.LastHidden`/`EnableHiddenTaps` has no `GpuForwardPass` equivalent

`IForwardPass.LastHidden` is declared on the interface, but confirmed (via
`QwenTtsTalkerGeneration.cs`'s own doc comment, and by reading `GpuForwardPass.cs` directly — no
`LastHidden` member exists there at all) unimplemented on `GpuForwardPass`. Any pipeline that
bridges the LLM trunk's per-position hidden state into a second, smaller model this way cannot
safely swap to `GpuForwardPass` without that bridging silently breaking.

**Confirmed present in**: FishSpeech's fast-AR bridge (already documented, 052 §4) and QwenTTS's
Talker → Code Predictor bridge (`QwenTtsPipeline.cs`/`QwenTtsCodePredictorGeneration.cs`, same
`ForwardPass.LastHidden` dependency, same real blocker).

### Blocker C — `GpuForwardPass`'s constructor only accepts a raw `GgufModel`, not the
`IModelTensorSource` abstraction some pipelines need

`GpuForwardPass`'s only constructor is `GpuForwardPass(GgufModel model, VulkanBackend gpu, ...)` —
unlike `ForwardPass(IModelTensorSource model, IComputeBackend backend, ...)`, which accepts any
tensor source. Orpheus's Talker LM is a plain GGUF checkpoint with no special embedding logic, so
passing `_model` (the raw `GgufModel`) directly works for it. QwenASR's decoder, however, needs
`QwenAsrLlmTensorSource`'s real audio-conditioned embedding remap (`EnableAudioConditioning`
replaces `<|audio_pad|>` placeholder rows with the AuT encoder's real per-frame projections before
the LLM ever sees them) — `GpuForwardPass` has no way to accept that. Making it accept a generic
`IModelTensorSource` would be a real change to a shared core engine class used by every other
GPU-backed text pipeline in the repo, not a change scoped to one audio pipeline's `--backend`
option — out of scope here.

**Confirmed present in**: `QwenAsrDecoder.cs`'s `GenerateFromSource` (otherwise a clean,
low-risk-looking `ForwardPass.Prefill`/`.Forward` loop with no `LastHidden` dependency — this was
the one candidate that looked promising until this constructor mismatch was found).

## Per-engine disposition

| Engine | Blocker | Notes |
|---|---|---|
| Orpheus-TTS | — (already done, pre-existing) | Real `CudaForwardPass`/`GpuForwardPass` fallback chain already in `OrpheusPipeline.cs`. Nothing to do. |
| QwenASR (decoder) | C | Otherwise the cleanest candidate audited — blocked only by the tensor-source/constructor mismatch above. |
| QwenTTS (Talker/Code Predictor) | B | `LastHidden` bridge, same as FishSpeech. |
| FishSpeech (fast-AR) | (see 052 §4, different reason: call size/frequency, not A/B/C) | Already documented; not re-litigated here. |
| Parler-TTS | A (T5 encoder) + own bad-target reasoning (decoder) | Already documented in 052 §5. |
| Whisper (encoder + decoder) | A | Encoder is the one candidate in this whole audit large/batched enough (full-utterance, non-autoregressive) that a *real* batched restructure might actually pay off on stronger GPUs — flagged as the best candidate for a **future, separate** restructure-then-port effort, not attempted here since the restructure itself is real, untested work. |
| Parakeet (Conformer encoder) | A | Same shape as Whisper's encoder; same future-candidate note. |
| FunASR (encoder) | A | Same shape; additionally has the CPU-was-already-faster caveat from its own code comment. |
| Kokoro | A | StyleTTS2 decoder/prosody predictor, per-row pattern. Also already the fastest-or-near-fastest engine on CPU (README: 0.933x RTF) — low motivation regardless. |
| MeloTTS | A | Same per-row pattern; also already fast on CPU (1.337x RTF). |
| XTTS-v2 (GPT trunk) | A | Per-position `LinearWithBias` inside a `for (i < t)` loop, same pattern. |
| Piper | A + conv-heavy (same "no Vulkan SDK, can't add conv shaders" constraint as 052's Blocker for CfmUNetKernels' Conv1d) | Also already the single fastest engine in the whole benchmark table (0.188x RTF) — essentially no real-world motivation to GPU-accelerate the fastest thing in the suite. |
| MMS-TTS | A (VITS-family, same shape as Piper/MeloTTS) | Not individually re-audited in depth — same family, same expected blocker; skipped rather than re-deriving the same finding a third time. |

## What would make Whisper/Parakeet/FunASR's encoders real candidates later

If this is picked up again: these three are architecturally the best remaining fit (large,
non-autoregressive, full-sequence batched attention+FFN, similar in spirit to CosyVoice3/F5-TTS's
DiT blocks) — the blocker is purely that the current CPU code parallelizes over rows instead of
batching them into one call. A real next step would be: restructure one encoder's attention/FFN to
build a `[t, dim]` batched buffer once per layer (instead of `t` separate row buffers), benchmark
that CPU-side change alone first (per CLAUDE.md rule 7 — could regress or improve the CPU path
independent of GPU), and only then add the `LinearGpu`-style dispatch on top, following 052's
established pattern exactly.

## Do not (same standing rules as 052)

- Do not force a port through Blocker A by issuing `t` separate GPU dispatches per row — that is
  strictly worse than not porting, confirmed by the exact reasoning that killed Parler's T5
  encoder in 052.
- Do not swap any `LastHidden`-dependent pipeline to `GpuForwardPass` without first adding real
  `LastHidden`/`EnableHiddenTaps` support there (a shared-engine change, needs its own review).
- Do not widen `GpuForwardPass`'s constructor to accept `IModelTensorSource` as a side effect of
  one audio pipeline's `--backend` flag — that's a core Engine change with blast radius across
  every GPU-backed text pipeline in the repo, and needs to be a deliberate, reviewed change on its
  own, not smuggled in here.
