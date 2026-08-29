# Handoff: finishing QwenTTS and CosyVoice3 (OpenTail.Stingray)

## Context: how to work on this project's audio pipelines (read first)

This is `OpenTail.Stingray`, a C#/.NET port of several GGUF-based inference engines. Fish Speech
S2 Pro (a sibling TTS engine in this same codebase) was JUST taken from completely broken to
"sonically 100% spot on, flawless" (user's own words) using a specific, repeatable methodology.
Both QwenTTS and CosyVoice3 have the exact same shape of problem Fish Speech had, and this
methodology is directly applicable to both. Read `docs/audio-review-progress.md` for the full
Fish Speech investigation trail before starting either of these — it's long, but it's the proof
that this approach works, and it documents dead ends worth not repeating.

**The methodology that actually worked, in order:**

1. **Read the real C++/GGML reference source line by line. Never guess "what the model does"
   from general knowledge of the upstream Python project** — these C++ ports have their own
   specific implementation choices that can differ from what you'd assume. Fish Speech's biggest
   bug (a whole missing 8-layer transformer in the codec) was found by an earlier session
   incorrectly concluding "it's just a bare RMSNorm" from a shallow tensor-name check, and only
   caught later when someone actually read `build_quantizer_decode_stage`/`build_transformer` in
   the reference source in full.
2. **Build and run the actual reference binary** on the identical prompt/model as ground truth.
   This proves whether the checkpoint/architecture is even capable of clean output (it always
   was, for Fish Speech) before assuming the port itself must be broken in some fundamental way.
3. **Numeric verification before audio generation, every time.** Add temporary, env-var-gated
   debug dumps to the reference C++ source (a pattern used repeatedly and successfully — dump
   logits/hidden-states/codes to a file, read them back in a C# test, compute cosine similarity /
   rank comparisons). This is FAR cheaper than generating audio and listening, and repeatedly
   caught or ruled out real bugs before wasting a listen on a guess. Revert the C++ debug
   instrumentation once you're done with it (these reference repos are gitignored/their own git
   repos — instrumentation never needs to be committed, just don't leave it lying around
   uncommitted forever either).
4. **The listener's ear is the final authority, not any numeric metric.** Several
   "numerically-verified-correct" changes still sounded wrong. Don't declare victory from cosine
   similarity alone — but also don't skip the numeric step, since it's what actually finds bugs
   fast.
5. **When stuck, write down everything proven/ruled-out/unresolved and hand off to a fresh
   perspective** rather than continuing to guess-and-check the same few ideas. This is literally
   how Fish Speech's final, real bug (a missing input position in the fast-AR sequence) was found
   — a second AI, reading the same reference source with fresh eyes, spotted a structural mistake
   the original investigation had missed after weeks of temperature/sampling-strategy tweaking
   that never addressed the real cause.

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

### The real, still-unsolved blocker: golden-verified, precisely localized, no fix found

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

### What's NOT yet been tried (concrete next steps, in priority order)

1. **Rule out a T=2-specific edge case first.** The T=2 test used `Prefill([0])` (a length-1
   prefill) followed by one `Forward` call — real usage never prefills fewer than ~10 tokens.
   Before trusting the T=2 result as representative of the real bug, re-run it with a longer
   leading prefill (e.g. 5 real tokens, then the position under test) to rule out a
   prefill-length-1-specific artifact being a SECOND bug layered on top of the real one.
2. **Dump post-RoPE Q/K vectors directly, not just the final hidden state or logits.** Add a debug
   hook inside the attention/RoPE application code (or a temporary standalone kernel-level test)
   to capture the rotated Q/K vectors at position 1 from both C# and the Python reference, and
   diff those directly. This narrows "somewhere in attention" down to a specific tensor: Q after
   RoPE? K after RoPE? the raw attention scores? the weighted V-sum? the output projection?
3. **KV-cache write/read consistency.** Verify the cached K vector for position 0 (written during
   a `Prefill([0])` call) is byte-identical to what a FRESH, uncached computation of position 0's
   K would produce. This is the exact test that (in a different but structurally similar
   investigation) never got run for Fish Speech either, and is a good candidate for "the bug that
   only shows up with 2+ positions but is invisible in a position-0-only test."
4. **Given `examples/qwentts.cpp` exists but isn't built yet**: build it (same MSVC Developer
   environment requirement as `examples/s2.cpp` — plain `cl.exe` from a bare shell fails on a
   missing `<cstdint>`; use
   `vcvars64.bat` first, e.g. `cmd /c '"C:\Program Files (x86)\Microsoft Visual
   Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cd /d <repo>\examples\qwentts.cpp &&
   cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build --config Release --parallel
   8'`, adjust for whatever this repo's actual CMake target/flags turn out to be), then apply the
   EXACT Fish-Speech-proven methodology: add a temporary env-var-gated dump of the real
   reference's per-position hidden states/logits for a real multi-token prompt, and diff against
   the C# port position by position. This is likely to be much faster than continuing with the
   Python harness alone, and is how Fish Speech's actual remaining bugs were eventually found —
   the C++ reference is closer to "the same kind of code" as this port than the Python reference
   is, so bugs found by comparing against it tend to be more specific/actionable.
5. Only after 1-4 narrow the failure to a specific tensor/operation should a fix be attempted —
   the QwenTTS investigation deliberately did NOT guess-and-check further changes without that
   evidence, and that discipline should continue.

### Also flagged, not blockers

- `QwenTtsPipeline`'s real run re-initializes `ForwardPass` roughly once per generated
  frame/codebook (~28 `[ForwardPass] Pre-faulted...` log lines for one short utterance), each
  re-pre-faulting its weight set from scratch — a real, avoidable performance issue, worth a perf
  pass (same category of work just done for Fish Speech) once correctness is fixed, not before.
- `QwenTtsTalkerLm.cs`'s dead code (see above) should be deleted once someone confirms nothing
  else references it, to stop it misleading future readers.

---

## 2. CosyVoice3 — currently runs, but produces a "dentist drill" sound

### Current state

`-e cosyvoice`/`cosyvoice3`/`cosy` on the CLI routes to `CosyVoice3Pipeline`
(`src/OpenTail.Stingray.Audio/CosyVoice/CosyVoice3Pipeline.cs`). It does NOT throw — it runs to
completion and produces audio, but that audio sounds wrong (described as "dentist drill",
consistent across multiple attempts including with the real-speaker-embedding fix below).

Other real, working, already-verified components (per `docs/audio-review-progress.md`, cosine
>0.999 golden-verified against real oracles for each): `CosyVoice3FlowEncoder.cs` (flow encoder),
`CosyVoice3DiTModel.cs`/`CosyVoice3DiTWeights.cs` (the DiT backbone — confirmed to literally BE
F5-TTS's DiT architecture, tensor-for-tensor, ported and verified earlier this project),
`CosyVoiceHiftVocoder.cs`/`CosyVoiceHiFT.cs` (HiFT vocoder, shared with the already-good
Chatterbox pipeline, F0-predictor bugs found and fixed earlier). `CosyVoice3Llm.cs` (speech-token
generation) already uses real sampling (top-k=25, top-p=0.8, sliding-window repetition penalty,
ported from `examples/cosyvoice.cpp`'s reference sampler) — this is NOT a greedy-decode-collapse
bug like Parler-TTS/Fish Speech/QwenTTS all had.

Model: `models/cosyvoice3/CosyVoice3-2512_F16.gguf` (or `_Q4...`, check what's present). Real
C++/GGML reference: **`examples/cosyvoice.cpp`** (confirmed present locally, NOT yet built).

### What's already been fixed this session (real, verified, but insufficient alone)

**Real x-vector speaker embedding wired in.** Earlier assumption ("needs a whole new CamPlus
neural-net port, much bigger job") was WRONG — re-scoped after actually reading
`examples/cosyvoice.cpp/src/cosyvoice-frontend.cpp`: the reference doesn't reimplement CAM++
natively either, it just runs a pre-exported ONNX graph (`models/campplus.onnx`, already present
locally, no download needed) via ONNX Runtime. This codebase already has a generic ONNX host
(`OpenTail.Stingray.Core.OnnxModelSession`). Added `src/OpenTail.Stingray.Audio/CosyVoice/
CamPlusSpeakerEncoder.cs`: a real Kaldi-compatible 80-bin fbank feature extractor (Povey window,
per-frame DC removal + pre-emphasis, 20-8000Hz mel filterbank, per-utterance cepstral mean
normalization — ported tensor-for-tensor from the reference's real SIMD implementation, reusing
this codebase's existing `SpectralKernels.ComputePowerSpectrum` FFT helper), feeding
`campplus.onnx` (confirmed via a direct ONNX shape probe: input `[1, T, 80]` -> output `[1, 192]`).
`CosyVoice3Pipeline.Generate` now extracts a real per-reference speaker embedding when
`--ref-audio` is supplied, replacing the previous all-zero placeholder. Verified via
`tests/OpenTail.Stingray.Tests.Audio/CamPlusSpeakerEncoderTests.cs` (real, non-degenerate 192-dim
output from real audio) and an end-to-end CLI run.

**This did NOT fix the "dentist drill" sound.** Two other pieces of real, missing conditioning
remain (already correctly scoped in an earlier entry, not yet attempted):

### What's still missing (concrete, scoped, not yet attempted)

1. **`cond` (the reference conditioning mel) is still all-zero.** CosyVoice3 is architecturally a
   zero-shot voice-CLONING model (confirmed via `list-metadata` — no baked-in voice-preset
   metadata keys, unlike Kokoro-style engines). The DiT's real input expects a reference audio's
   OWN mel-spectrogram (a different, CosyVoice3-specific mel filterbank than CamPlus's fbank —
   the reference's own mel-basis construction was already read in full this project:
   `build_mel_basis(24000.f, 1920, 80, 0.0f, 12000.0f)` in `examples/cosyvoice.cpp`'s
   `cosyvoice-frontend.cpp` — 24kHz, 80-bin, `n_fft=1920`, `fmin=0`, `fmax=12000`, DISTINCT
   parameters from CamPlus's own fbank, already spec'd, just not yet ported). This mel needs to be
   computed from the same `--ref-audio` file already being loaded for the speaker embedding, then
   fed into the DiT alongside (masked/prepended to) the DiT's own input the way the reference's
   real `frontend_zero_shot` function does — read that function in full in
   `cosyvoice-frontend.cpp` before implementing, don't guess the exact masking/prepending
   mechanics.
2. **The DiT's classifier-free-guidance (CFG) refinement step is omitted entirely.** Flagged in
   `CosyVoice3DiTModel.SolveFlowMatchingOde`'s own doc comment. Needs the real CFG scale/mechanics
   read from the reference source and ported — this is a standard diffusion-model technique
   (unconditional + conditional forward passes, extrapolated), but get the exact scale factor and
   whether it's applied at every ODE step or only some from the real source, don't assume.

Given a real speaker embedding is now wired in and BOTH of these are still missing, the honest
expectation is that fixing #1 alone might not be enough either — try #1 first (bigger of the two,
most likely to matter most since a diffusion model with zero conditioning input is fundamentally
out-of-distribution), re-verify with the same "numeric check before audio" discipline, THEN
attempt #2 if #1 alone doesn't fix it.

### Also worth checking, not yet investigated

- `CosyVoice2Pipeline.cs` and `CosyVoicePipeline.cs` (v1) have the exact same real gap (their own
  doc comments say so) — if CosyVoice3's fix pattern turns out reusable, it's worth checking
  whether v1/v2 need the identical fix or already have a different, real speaker-encoder path.
  Not investigated this pass, don't assume either way.
- `speech_tokenizer` (real reference audio -> speech tokens, needed for TRUE zero-shot voice
  cloning where the model continues from a real reference utterance, not just conditions on its
  speaker identity) — `models/cosyvoice_speech_tokenizer.onnx`/`_v2.onnx` sit locally, unclear if
  already wired to anything useful. Flagged, not investigated.

---

## General reminders for whoever picks either of these up

- **Never guess at "what the model does" from general knowledge** — always cite the specific line
  in the specific reference C++/GGML file. Confirm, don't assume.
- **Numeric verification before audio generation, every time** — cosine similarity, rank/
  probability comparisons, per-stage golden tests. Audio generation + listening is the expensive,
  limited resource — the person listening has explicitly asked not to be given many clips to A/B
  test in a row ("almost no point in running hundreds of tests... 1 test - ME").
- **The listener's ear is the final authority, not any numeric metric or how "correct" a change
  looks on paper.**
- **Revert on demand, without arguing the point first**, if a change isn't trusted or doesn't
  sound better.
- **CLAUDE.md's project-wide rules apply**: no `--nologo` on `dotnet test`; heavy/real-weight
  tests need `STINGRAY_RUN_HEAVY_TESTS=1`; use `dotnet test ... -- --filter-class`/`--filter-method`
  or the built `.exe` with single-dash `-method`/`-class`; do a performance pass and a DRY pass
  once a model's port is genuinely complete and correct, not before.
