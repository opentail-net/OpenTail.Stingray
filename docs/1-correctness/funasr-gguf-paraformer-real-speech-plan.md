# Plan: FunASR GGUF Paraformer returns an empty transcript on real speech

**Bug logged:** 2026-09-27 (`docs/103-quickest-first-plan.md` item 10; see the tracked entry in
`docs/1-correctness/bugstofix.md`).

## Goal and scope

Find and fix why the real Paraformer GGUF pipeline returns an empty transcript for real Mandarin
speech. Establish the first divergent stage using the exact real checkpoint and WAV, compare with a
trusted reference on the same audio, and add a real-speech regression test without weakening existing
stage goldens.

There are **two separate defects** to track and never conflate:

1. The real Paraformer inference path returns `""` for the real Mandarin WAV.
2. Some real-weight tests resolve `paraformer-q8.gguf` from `models/` before
   `models/_models/`, and therefore load a different Fun-ASR-Nano checkpoint.

Fixing the lookup trap does not prove the inference bug is fixed. Conversely, an inference result
from the wrong checkpoint is not evidence about Paraformer.

## Confirmed facts — do not re-derive without new evidence

| Finding | Status |
| --- | --- |
| Correct real Paraformer GGUF | `models/_models/paraformer-q8.gguf`, about 237 MB; metadata includes `general.architecture=paraformer` and `pf.vocab`. |
| Conflicting wrong file | `models/paraformer-q8.gguf`, about 1.0 GB; metadata identifies `general.name=Fun-ASR-Nano-2512`, `general.architecture=audiocpp`. |
| Real Mandarin WAV | `docs/audio-samples/paraformer-zh-test-0.wav`. |
| ONNX control on same WAV | Produces fluent Mandarin: `对我做了介绍啊那么我想说的是呢大家如果对我的研究感兴趣呢嗯`. |
| GGUF pipeline on same WAV | Returns an empty transcript. |
| Existing encoder, predictor/CIF, and decoder stage goldens | Pass on their dedicated controlled fixtures; they do not prove the stages agree on this real WAV. |
| Existing GGUF end-to-end test | Uses a 440 Hz tone, not speech; an empty transcript is plausible and it is not a real-speech correctness test. |
| Local `examples/paraformer.cpp` implementation | Not a trustworthy full encoder oracle: its FSMN memory addition is explicitly disabled. Do not use it to judge encoder parity. |

The production path is:

```text
PCM → FunAsrRealMelExtractor → FunAsrEncoder → FunAsrPredictor/CIF
   → FunAsrRealDecoder → argmax → FunAsrTokenizer.Decode → SpeechToTextResult
```

In the current `FunAsrPipeline.TranscribeReal()` path, an empty result can return before decoding when
features are empty or `tokenCount == 0`; otherwise, the decoder and tokenizer can still produce empty
text. Instrument and identify the first failure instead of inferring it from the final string.

## Phase 0 — Eliminate the wrong-checkpoint trap before interpreting tests

Do this before relying on any real-weight test result.

- Inspect both local files and their GGUF metadata. Do not infer architecture from the filename or
  file size alone.
- If renaming the incorrect `models/paraformer-q8.gguf`, use an explicit Nano name such as
  `models/fun-asr-nano-2512.gguf`; update Nano consumers to that path. Never overwrite or delete the
  Nano checkpoint. If the target already exists, preserve it and use explicit paths/metadata checks
  instead of replacing it.
- Make `FunAsrWeights` reject a checkpoint unless `general.architecture` is exactly `paraformer` and
  `pf.vocab` exists with the expected type. Fail with a clear error that reports the actual
  architecture and missing/invalid metadata.
- The current `FunAsrPipeline.Architecture` property reports the constant
  `Alibaba-FunASR-Nano`, even for the Paraformer GGUF path. Audit this identity reporting and tests;
  do not use that property as proof of the loaded GGUF architecture. Make the reported identity
  accurate if its public/API contract is intended to describe the active model.
- For the immediate Paraformer real-weight run, use the explicit
  `models/_models/paraformer-q8.gguf` path or an architecture-validating lookup helper.

**Acceptance:** no Paraformer test can fail with `Paraformer GGUF missing 'pf.vocab'` because it
silently selected the Nano file. Re-run the relevant FunASR tests and record the actual path,
`general.architecture`, `general.name`, and presence of `pf.vocab` for the loaded checkpoint.

## Phase 1 — Freeze the exact real-speech failure

Run only the verified Paraformer checkpoint:

- Model: `models/_models/paraformer-q8.gguf`
- Audio: `docs/audio-samples/paraformer-zh-test-0.wav`
- Language: `zh`

Do not change inference math in this phase. Record:

- canonical model path, file identity, `general.architecture`, `general.name`, and `pf.vocab` count;
- WAV sample rate, sample count, duration, channels, and raw PCM min/max/RMS;
- frontend frame count, encoder frame count, CIF token count, decoded token count;
- final transcript and wall time;
- ONNX control transcript for this same WAV and its input audio details.

The baseline must demonstrate, reproducibly:

```text
verified Paraformer GGUF + real Mandarin WAV + real pipeline = empty transcript
```

The ONNX result is a control that the clip contains transcribable speech, not a promise of byte-for-
byte output parity between independently implemented pipelines.

## Phase 2 — Add compact stage diagnostics, not fixes

Instrument one failing `TranscribeReal()` run. Report summaries and shapes; do not dump full tensors
to the console. Write a compact binary/CSV diagnostic only if full values are needed for parity work.

Record at minimum:

```text
[paraformer]
audio: samples, rate, duration, channels, PCM min/max/RMS
frontend: logmel [T,80]; LFR/CMVN [T,560], feature min/max/mean/RMS
encoder: [T,512], min/max/mean/RMS, finite
predictor: tokenCount, sum(alpha), alpha min/max/mean
predictor: acousticEmbeds [N,512], finite, min/max/RMS
decoder: logits [N,8404], finite
decoder: first token argmax IDs, token strings, special/non-special counts
final: transcript
```

Use the actual metadata-derived dimensions if the loaded model differs; confirm the expected values
against this checkpoint rather than hardcoding log labels. Add alpha tail/sum and CIF fire counts as
useful details, but keep the diagnostic bounded.

Classify the first observable failure:

- **A — CIF first failure:** frontend and encoder are plausible; reference has fires/tokens but
  Stingray has `tokenCount == 0` or a materially different count. Investigate encoder-to-CIF inputs,
  alpha generation, and integrate/fire logic.
- **B — decoder first failure:** token count and embeddings are finite, but decoder logits or IDs are
  non-finite, pathological, or almost entirely special tokens. Investigate decoder inputs/context or
  decoder computation.
- **C — tokenizer first failure:** raw decoder IDs include ordinary vocabulary IDs but decode to an
  empty string. Investigate ID-to-vocabulary mapping and token handling.
- **D — frontend/encoder first failure:** features are empty/pathological, or encoder output already
  diverges substantially. Continue with frontend then encoder parity before touching CIF/decoder.

Do not change formulas, CIF thresholds, or tokenizer behavior based only on the final empty string.

## Phase 3 — Verify frontend preprocessing on the actual WAV

The existing real-mel golden uses a dedicated short fixture. It does not prove that the real WAV is
read and preprocessed identically to the trusted Paraformer reference. Compare this complete chain:

```text
real WAV → WavReader → waveform convention/scaling → fbank/log-mel → frame handling
→ LFR splice → CMVN
```

The current `FunAsrRealMelExtractor.ExtractLogMel()` default includes `waveformScale = 32768f`.
Verify the expected PCM amplitude convention from the real Paraformer preprocessing/reference and
compare against it. Do not change the scale merely because another pipeline uses normalized floats,
or preserve it merely because a synthetic frontend golden passes.

For the failing WAV, record and compare against the reference:

- raw PCM min/max/RMS and the exact scaling applied;
- log-mel frame count and shape;
- LFR frame count and final feature dimension (expected 560 for this checkpoint);
- feature min/max/mean/RMS and first, middle, and last feature rows.

A frontend mismatch is a stop condition: resolve it before investigating encoder or later stages.

## Phase 4 — Obtain real-audio reference tensors

The existing stage goldens validate operators on controlled fixtures, not this full real-audio input.
Produce reference tensors for the same WAV and intended Paraformer checkpoint family:

```text
frontend features
encoder output
predictor alphas
predictor acoustic embeddings
decoder logits
raw decoder argmax IDs
```

Reference rules:

1. Locate and reuse the existing `scratch-llamacpp-ref/funasr_golden_*.py` oracle machinery if it is
   available; extend it to accept the WAV rather than creating a second implementation.
2. Otherwise use an already available, real FunASR/PyTorch reference implementation in the current
   environment, after confirming its checkpoint/configuration corresponds to this Paraformer family.
3. Do not write an independent Python reimplementation. Do not treat the local `paraformer.cpp`
   encoder as authoritative for FSMN behavior; its memory addition is disabled.
4. If no trustworthy real-audio reference is available, record that blocker and establish how to
   obtain one before changing stage mathematics. Do not substitute a synthetic golden or the ONNX
   transcript for tensor-level reference parity.

Keep the exact WAV, reference checkpoint/configuration, sample convention, feature settings, tensor
shapes, and reference-generation command with the capture.

## Phase 5 — Compare in order and stop at the first divergence

For every comparable tensor, report shape and raw-value metrics: max absolute error, mean absolute
error, RMS error, relative L2, and cosine. Preserve compact dumps/captures needed to reproduce the
comparison.

### 5.1 Frontend

Compare reference features against Stingray features for the same real WAV. Shapes and preprocessing
must agree before continuing. A mismatch means **stop**; do not investigate encoder/CIF/decoder code
yet.

### 5.2 Encoder

Compare the complete sequence from the reference encoder with `FunAsrEncoder.Forward(...)`. Report
first, middle, and last frames as well as global metrics. If the short encoder golden passes but the
real sequence diverges, investigate composition and variable-length behavior, especially:

- sequence-length-dependent assumptions;
- FSMN depthwise convolution boundaries;
- padding and first/last frame handling;
- composition of the special input layer and the main encoder stack.

Do not use the incomplete local C++ encoder implementation to adjudicate FSMN parity.

### 5.3 Predictor/CIF

Compare, in order:

1. reference alpha sequence against Stingray alpha sequence;
2. reference token count against Stingray token count;
3. reference acoustic embeddings against Stingray acoustic embeddings.

Record `sum(alpha)`, `floor(sum(alpha))`, tail alpha values, fire/token count, and norms of the first
and last emitted embeddings. The synthetic predictor golden does not prove that the real speech
sequence produces the correct number of fires.

If the reference has `N` tokens and Stingray has zero or a materially different count, the failure
lies no later than the encoder-to-CIF path; the decoder is not the first cause. Use the preceding
encoder comparison to distinguish encoder output from CIF generation/integration.

## Phase 6 — Isolate the decoder with real acoustic embeddings

If predictor parity is sufficient to continue, run this four-arm experiment:

| Acoustic embeddings | Decoder | Purpose |
| --- | --- | --- |
| Stingray | Stingray | Existing failing path |
| Reference | Stingray | Isolates the decoder using trusted inputs |
| Stingray | Reference | Isolates predictor/input effects using trusted decoder |
| Reference | Reference | Reference control |

Use captured real-audio embeddings; keep the encoder context/memory inputs paired and explicit as
required by the decoder. The first two arms should be runnable in C#. Use the trusted reference
implementation for its decoder arm; do not assume the local incomplete C++ encoder is a valid source
for reference decoder inputs.

If reference embeddings through the Stingray decoder produce correct IDs but Stingray embeddings do
not, the decoder is not the initial culprit. If the Stingray decoder fails even with reference
embeddings/context, investigate decoder implementation and wiring.

## Phase 7 — Inspect decoder logits and raw token IDs

For the first approximately 10 predicted positions, record:

```text
position
argmax ID and vocabulary string
top-5 IDs and vocabulary strings
blank, <s>, </s>, and <unk> logits
argmax margin
finite/logit-range status
```

Determine whether Mandarin IDs are competitive, special tokens dominate, logits are finite but
nearly constant, or valid IDs are being lost after argmax. This distinguishes an unhealthy
acoustic/decoder state from a token interpretation issue.

## Phase 8 — Audit tokenizer behavior only after validating decoder output

The tokenizer intentionally strips `<blank>`, `<s>`, `</s>`, and `<unk>` and applies the vocabulary's
`@@` continuation convention. Do not change this behavior because the transcript is empty.

First establish the raw argmax IDs, then inspect each `ID → vocab string` mapping for this checkpoint:

- ordinary Mandarin IDs with empty text → tokenizer/vocabulary handling is implicated;
- only special IDs → upstream model state is implicated;
- out-of-range IDs or unexpected vocab size → decoder/checkpoint vocabulary mismatch is implicated.

## Phase 9 — Add a real-speech end-to-end regression

Add `Paraformer_GgufRealSpeech_ProducesNonEmptyMandarinTranscript` using
`docs/audio-samples/paraformer-zh-test-0.wav` and the verified Paraformer GGUF.

A first regression gate should assert nonempty `result.Text`, nonempty `result.Segments`, and a
positive decoded token count (for example, a nonempty segment token array). Optionally assert
plausible Chinese characters. Do not require exact transcript equality with ONNX unless
cross-pipeline determinism and wording are established.

If a deterministic real-reference capture is available, the stronger assertion is parity of raw
output token IDs. Keep the existing 440 Hz test as a no-speech/synthetic sanity test; it is not a
substitute for this real-speech regression.

## Phase 10 — Harden every fixture/model lookup

After the correct checkpoint can be loaded unambiguously, remove the structural lookup trap across
FunASR tests:

- Replace ambiguous `FindModelPath("paraformer-q8.gguf")` lookups with an explicit expected path or a
  helper that validates `general.architecture == "paraformer"` and `pf.vocab` before returning a
  candidate.
- Give Nano tests an explicit Nano checkpoint path or a helper validating `general.architecture ==
  "audiocpp"`; do not make Paraformer tests pass by redirecting Nano tests to the other file.
- Audit all test classes from the FunASR landscape-sweep failures, including weight, encoder,
  predictor, decoder, mel-extractor, Nano golden, performance, and real-weight tests. Confirm each
  test loads the intended architecture and reports the resolved path/identity when appropriate.
- A matching filename alone must never select or certify a checkpoint.

Re-run the relevant FunASR test set and record that the real Paraformer file was selected. Keep the
inference correctness result separate from these test-discovery results.

## Hypotheses — investigate in evidence order

1. **Predictor/CIF output is zero or unusable (high).** Directly explains a zero-token early return;
   compare encoder output and alpha/token counts first.
2. **Real-audio frontend mismatch (high).** Synthetic frontend fixture is short; check PCM scaling,
   framing/boundaries, LFR padding, and CMVN against the reference.
3. **Real-audio encoder composition or sequence-length issue (medium-high).** The existing encoder
   golden is only 10 frames; inspect long-sequence boundaries, FSMN temporal handling, and the special
   input layer.
4. **Decoder receives incorrect acoustic embeddings or context (medium).** Use the four-arm decoder
experiment if token count is correct.
5. **Tokenizer strips valid output (low).** Prove from raw decoder IDs before changing the tokenizer.
6. **Audio loading/scaling issue (low-medium).** Record sample rate, channels, PCM range, and RMS.
The ONNX control lowers this probability but does not prove identical audio values reach both
frontends.

## What not to do

- Do not modify decoder math just because the final transcript is empty.
- Do not modify CIF thresholds without comparing real alpha sequences first.
- Do not change special-token handling without inspecting raw decoder IDs.
- Do not use the 440 Hz test as evidence that real-speech transcription works.
- Do not trust `paraformer-q8.gguf` as an identity; validate GGUF architecture metadata and `pf.vocab`.
- Do not overwrite or delete the actual Nano checkpoint while correcting the filename collision.
- Do not use the incomplete local `paraformer.cpp` encoder as the authority for FSMN memory behavior.
- Do not create a new independent Python reference implementation.
- Do not loosen existing stage-golden tolerances to make the real-audio investigation pass.
- Do not make multiple stage-math changes before identifying the first divergence.

## Decision tree

```text
Verified 237 MB Paraformer GGUF + real Mandarin WAV
↓
Frontend comparison
Mismatch: fix frontend and stop; match: continue to encoder
↓
Encoder comparison
Mismatch: fix encoder/composition and stop; match: continue to CIF
↓
CIF comparison
Alpha/token-count mismatch: fix encoder-to-CIF/predictor path and stop; match: continue to decoder
↓
Decoder isolation/logits
Bad logits/IDs: fix decoder or inputs; ordinary valid IDs: continue to tokenizer
↓
Tokenizer mapping
Strips valid IDs: fix tokenizer/vocabulary handling
Nonempty plausible transcript: inference bug resolved
```

Separately, checkpoint-path validation must prove that Paraformer tests load the `paraformer`
architecture and Nano tests load `audiocpp`; success in either branch does not imply success in the
other.

## Success criteria

Close the real-speech bug only when all are true:

1. Real-weight tests unambiguously select the 237 MB `general.architecture=paraformer` checkpoint
   with valid `pf.vocab` metadata.
2. The real WAV's frontend, encoder, predictor, and decoder have been traced with real-audio evidence
   against a trusted reference, locating the first divergence.
3. The corresponding defect is fixed without weakening existing stage goldens.
4. The real Mandarin end-to-end regression produces nonempty, plausible Chinese text and positive
   decoded token count.
5. The existing 440 Hz synthetic/no-speech test still passes.
6. FunASR test lookup cannot silently load Nano when Paraformer is requested (and Nano tests still
   select Nano explicitly).
7. `docs/1-correctness/bugstofix.md`, `docs/STATUS.md`, and relevant FunASR test documentation record
   the actual root cause, model identities, and real-speech result.

Preferred evidence is real-audio frontend, encoder, predictor, and decoder parity followed by a
nonempty real-WAV transcript — not merely a non-null result object or a passing synthetic tone test.
