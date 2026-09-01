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

## 1. QwenTTS — RESOLVED, moved to done/

QwenTTS is fixed and re-enabled (commit `05a4152`, same missing-tensor-source bug class as
CosyVoice2/3). The original investigation content (four earlier bug fixes, the golden-verification
harness, the precisely-localized multi-position blocker, and the fix) is archived at
[done/qwentts-handoff-resolved.md](done/qwentts-handoff-resolved.md).

---

## 2. CosyVoice3 — runs end-to-end with real conditioning; speech is intelligible; identity-transfer awaiting a listen

### Current state (updated 2026-09-01 — see below, most of this section describes an earlier, now-resolved state)

`-e cosyvoice`/`cosyvoice3`/`cosy` on the CLI routes to `CosyVoice3Pipeline`
(`src/OpenTail.Stingray.Audio/CosyVoice/CosyVoice3Pipeline.cs`). Per `docs/audio-review-progress.md`
(2026-08-30), speech is real and intelligible, non-buzzing. The "dentist drill" description below
was this project's EARLIER state, before both the mel-conditioning and CFG fixes landed.

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

**UPDATE (2026-09-01): both items below were already fixed by an earlier session (commit
`4cb189f`, "fix real HiFT source-injection padding bug + wire LLM zero-shot conditioning"), this
section was simply left stale.** Re-verified directly by reading the current source and running a
real end-to-end generation with `--ref-audio`:

1. **`cond` is no longer all-zero.** `CosyVoiceMelExtractor.cs` is a complete, real port of
   `extract_speech_feat` (720-sample reflect pad, periodic Hann window, FFT-magnitude spectrum,
   Slaney mel filterbank, `log(max(1e-5, energy))`) — confirmed matching the reference's exact
   formula (`build_mel_basis(24000, 1920, 80, 0, 12000)`, the same `hz_to_mel`/`mel_to_hz` Slaney
   constants) line-for-line against `cosyvoice-frontend.cpp`'s current source. `CosyVoice3Pipeline`
   also now extracts real prompt speech TOKENS (`CosyVoiceSpeechTokenizer`, a separate real ONNX
   model, not just the CAM++ speaker embedding) and prepends them to the LLM's generation, matching
   the reference's real zero-shot mechanism where the LLM continues from the reference utterance's
   own tokens, not just conditions on a speaker vector.
2. **Real CFG is now implemented** in `CosyVoice3DiTModel.SolveFlowMatchingOde` (a genuine
   conditional/unconditional dual forward pass with `zeroCond`/extrapolation, `cfgRate` parameter
   defaulting to 0.7, threaded through the CLI).

**Verified working end-to-end** (2026-09-01, `STINGRAY_DEBUG_COSYVOICE3=1`, real ref-audio):
`speechTokens=87 promptTokens=65 numFrames=304 promptFrames=130`, real non-zero speaker embedding,
`refMel.Length=10400` (130 frames × 80 mel, exactly matching promptFrames), plausible mel output
range (`min=-14.41 max=5.64 mean=-4.22`) — every stage is producing real, non-degenerate data.
**Not yet judged by ear** — per this doc's own "listener is the final authority" rule, whether the
cloned voice actually sounds like the reference speaker is a human-judgment call this session
cannot make. Output: `docs/audio-samples/cosyvoice3-identity-check.wav` (ref:
`docs/audio-samples/fishspeech-lunch-REFERENCE.wav`, text: "The quick brown fox jumps over the lazy
dog."). **Next step for whoever picks this up: listen to that file and report whether the speaker
identity actually transferred** — if not, the numeric pipeline is real and correct enough that any
remaining gap is likely a genuine subtle bug (worth the same "numeric verify" discipline against
the C++ reference build), not a wholesale missing feature anymore.

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
