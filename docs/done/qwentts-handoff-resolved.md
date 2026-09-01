# Archived: QwenTTS portion of the QwenTTS/CosyVoice3 handoff

This is the QwenTTS half of `docs/qwentts-cosyvoice3-handoff.md`, archived 2026-08-31 because
QwenTTS is now fixed and re-enabled: commit `05a4152` ("Fix QwenTTS's Talker/Code Predictor: same
missing-tensor-source bug as CosyVoice2/3, re-enable engine") found and fixed the real root cause
— the same missing-tensor-source defect class also affecting CosyVoice2/3 — closing the "golden
verified, precisely localized, no fix found" blocker this doc describes below. The CosyVoice3
portion of the original handoff remains active; see
[../qwentts-cosyvoice3-handoff.md](../qwentts-cosyvoice3-handoff.md).

The content below is preserved verbatim as it stood before the fix, for investigation history.

---

## 1. QwenTTS — currently disabled, `-e qwentts` throws immediately

### Current state

`src/OpenTail.Stingray.Cli/TtsCommand.cs`'s `qwentts` dispatch case is a `throw`, with the real
working dispatch line kept as a `//`-commented-out line directly above it (restore that line to
re-enable). Main pipeline code:
- `src/OpenTail.Stingray.Audio/QwenTTS/QwenTtsPipeline.cs` — the Talker (slow-AR-equivalent)
  generation loop.
- `src/OpenTail.Stingray.Audio/QwenTTS/QwenTtsTalkerTensorSource.cs` — tensor-name remapping for
  the Talker, reusing the shared `ForwardPass` engine (same reuse pattern Fish Speech's slow-AR
  used successfully).
- `src/OpenTail.Stingray.Audio/QwenTTS/QwenTtsCodePredictorGeneration.cs` — secondary/acoustic
  codebook expansion (this model's equivalent of Fish Speech's fast-AR).
- `src/OpenTail.Stingray.Audio/QwenTTS/QwenTtsCodePredictorTensorSource.cs`.
- `src/OpenTail.Stingray.Audio/QwenTTS/QwenTtsTalkerLm.cs` — **DEAD CODE, do not use or trust**:
  a synthetic placeholder (`GenerateCode0`) that fabricates plausible-looking output via
  `MathF.Sin`/`Exp` formulas, never wired to real weights. Confirmed the real pipeline
  (`QwenTtsPipeline.cs`) never calls it. Misleading if read cold — flagged for cleanup, not yet
  removed.

Model: `models/qwen-talker-0.6b-base-Q8_0.gguf`. Real C++/GGML reference:
**`examples/qwentts.cpp`** (confirmed present locally, NOT yet built — no `build/` directory
exists yet, unlike `examples/s2.cpp` which was already built this session). Also available: a
real PyTorch reference at `examples/qwen-tts-py` (used for an earlier golden-verification pass,
see below) — prefer the C++ reference as source of truth per the methodology above, but the
Python one is there if the C++ build proves difficult.

### What's already known (real, not guessed) — four bugs already found and fixed

1. **Missing sampling** (fixed): both the Talker's semantic-code loop and the code-predictor's
   acoustic-codebook loop used plain `ArgMax`. Replaced with real Qwen3-TTS sampling
   (temperature=0.9, top-k=50, repetition_penalty=1.05 for the talker; temperature=0.9, top-k=50,
   no penalty for the subtalker — sourced from `examples/qwen-tts-py`'s real model config, not
   guessed). This fixed an audible tonal "drill noise" collapse.
2. **Missing acoustic-codebook feedback** (fixed): the real generation loop feeds ALL 16
   codebooks (semantic + 15 acoustic) back into the next talker step, summed via their respective
   embedding tables. The port only fed back the semantic code, silently dropping every acoustic
   code. Fixed by summing all 16 codebook embeddings for the next-step input.
3. **Stale-pointer bug, talker side** (fixed): `QwenTtsTalkerTensorSource.SetPromptEmbedding`
   allocated a brand-new buffer at a new address on every call, but `ForwardPass` captures a
   tensor's raw data pointer ONCE at construction and never re-resolves it — every talker step
   after the first was silently conditioned on a stale first-prompt-row embedding. Fixed with one
   persistent buffer, written in place. **This exact bug shape (stale pointer from
   reallocate-instead-of-write-in-place) is worth checking for anywhere else in this pipeline that
   might do the same thing, and worth checking in CosyVoice3 too if its debugging gets this far.**
4. **Same stale-pointer bug, code-predictor side** (fixed): same root cause as #3, in
   `QwenTtsCodePredictorTensorSource`.

### The real, still-unsolved blocker (AS IT STOOD BEFORE THE FIX): golden-verified, precisely localized, no fix found

A real golden-verification harness was built: loads the real Q8_0-dequantized GGUF weights
directly into the actual `Qwen3TTSTalkerModel` PyTorch class from `examples/qwen-tts-py`, feeds it
the IDENTICAL input embedding the C# pipeline composed, and compares hidden states/logits
numerically. Needed some `transformers` 5.7.0 API compatibility patches (documented in
`docs/audio-review-progress.md`'s QwenTTS section, not re-derive-worthy, just re-apply if the
harness needs rebuilding).

**Precise result** (real measured cosine similarities):

| Test | Cosine similarity | Verdict |
|---|---|---|
| T=1 (single token, no cross-attention), 1 layer | 0.999959 | Correct |
| T=2 (two tokens), 1 layer | 0.760090 | Wrong |
| T=11 (a real short prompt), 1 layer | 0.560 | Wrong |
| T=11, 28 layers (full model) | 0.005994 | Wrong (near-random) |

This precisely localizes the bug: **everything involving only a SINGLE position is proven
correct** (Q/K/V projection, QK-RMSNorm, RoPE rotation at position 0 specifically — which is
literally the identity rotation regardless of convention, so this test alone doesn't distinguish
NEOX vs interleaved — FFN, residual/norm structure). **Everything involving MULTIPLE positions is
where the bug lives**: causal attention across cached positions, RoPE rotation at position > 0, or
the KV-cache write/read path.

**Ruled out already** (checked against both `examples/qwen-tts-py` and `examples/qwentts.cpp`,
agreement between the two references was itself informative):
- Hyperparameters (HeadDim=128, NumHeads=16, NumKvHeads=8, EmbeddingDim=1024, RopeTheta=1e6,
  RmsNormEps=1e-6) — confirmed exactly correct via direct `ModelHyperparams` dump.
- NEOX RoPE selection — `"qwen3-tts"` is explicitly in `ModelGraph.cs`'s NEOX architecture switch,
  confirmed taken (not silently falling back to interleaved).
- The NEOX RoPE rotation formula itself (`SimdKernels.ApplyRoPECachedNeox`) — byte-for-byte
  matches both references' `rotate_half`.
- The RoPE frequency table (`SimdKernels.BuildRopeTable`) — matches both references' formula.
- RoPE dispatch consistency — every call site (single-step decode AND batched prefill) branches on
  `_hp.IsNeoxRope` correctly and uses the same table, no path-specific mismatch found.
- YaRN / partial-RoPE — confirmed NOT accidentally triggered (the relevant metadata keys are
  absent from this GGUF, defaults resolve to "off").
- GQA head-to-KV-head grouping — confirmed matches the real `repeat_kv`'s consecutive-repeat
  convention exactly.
- `mrope_interleaved`/`mrope_section` metadata — confirmed genuinely irrelevant for plain-text TTS
  (no vision/video input means all 3 multimodal position axes are identical, so both code branches
  of `apply_multimodal_rotary_pos_emb` produce numerically identical cos/sin regardless).

### Also flagged, not blockers

- `QwenTtsPipeline`'s real run re-initializes `ForwardPass` roughly once per generated
  frame/codebook (~28 `[ForwardPass] Pre-faulted...` log lines for one short utterance), each
  re-pre-faulting its weight set from scratch — a real, avoidable performance issue, worth a perf
  pass (same category of work just done for Fish Speech) once correctness is fixed, not before.
- `QwenTtsTalkerLm.cs`'s dead code (see above) should be deleted once someone confirms nothing
  else references it, to stop it misleading future readers.
