# Plan: Voxtral Realtime audio.cpp GGUF support

**Entry in:** `docs/1-correctness/bugstofix.md`, item **05** (Voxtral Realtime audio.cpp GGUF support).
Found during the 2026-09-28 `RUNNING.md` verification.

This is primarily a **Voxtral ASR format and entry-point coverage gap**, not a generic text-generation
architecture-admission bug.

## Goal

Make the self-contained audio.cpp-packed Voxtral Mini 4B Realtime 2602 GGUF usable through the
existing native Voxtral ASR pipeline. The intended user-facing command is:

```text
stingray stt -m voxtral --model-file models/Voxtral-Mini-4B-Realtime-2602-GGUF/voxtral-mini-4b-realtime-2602-q8_0.gguf -i input.wav
```

The generic text-generation command must continue to reject this GGUF:

```text
stingray -m <model.gguf> -p "..."
```

Do not add `audiocpp` to `ModelCompatibility.IsTextGenerationArchitectureSupported`. Do not route
Voxtral through the generic `ForwardPass` text-generation path. GGUF weights and embedded resources
must feed the existing Voxtral audio encoder, multimodal projector, text decoder, and Tekken tokenizer;
SafeTensors remains a supported input to the same inference implementation.

## Existing implementation and controls

| Component | Current state |
| --- | --- |
| Generic GGUF architecture gate | Conservative; rejects unadmitted text-generation architectures, including `audiocpp`. Keep unchanged. |
| `RunCommand -m` | Generic text-generation entry point; validates via `ModelCompatibility.ValidateForTextGeneration`. |
| `SttCommand -m voxtral` | Exists, but `ResolveVoxtralDir` and its branch require a SafeTensors directory. |
| `VoxtralPipeline.Load` | Loads `model.safetensors` and filesystem `tekken.json`; transcription logic is format-independent after construction. |
| `VoxtralAudioEncoder` / `VoxtralTextDecoder` | Existing real-weight-verified ASR math and decoding path; preserve as the control. |
| Existing end-to-end control | SafeTensors Voxtral plus `examples/audio.cpp/assets/resources/a.wav` matches the reference transcript exactly. |
| Packed GGUF infrastructure | `RvcPackedTensorSource` resolves `audiocpp.tensor_names` to actual `GgufTensorInfo`, including on-disk dtype and raw data pointer. `IWeightLoader` already separates `ReadF32` from `TryGetRaw`. |
| Embedded sidecar precedent | Audio pipelines already read `audiocpp.embedded_files.*`; reuse the established representation/behavior rather than creating another metadata parser. |

The published model card describes Q8_0 and Q4_K Voxtral Realtime GGUF files as single-file packages
with embedded model spec and required sidecars. This is background only. The actual local GGUF is the
source of truth for metadata, names, shapes, dtypes, and embedded contents.

## Phase 0 — Classify the original failure; do not implement yet

### 0.1 Reproduce the generic text-generation behavior

Run the original generic command against the exact local Voxtral GGUF. Record absolute path, file
size, SHA-256, `general.architecture`, `audiocpp.*` metadata, and tensor count. Expected behavior is
`NotSupportedException` for architecture `audiocpp` before inference. Record this as **intentional
generic text-generation admission behavior**, not as the bug to fix.

### 0.2 Reproduce the dedicated ASR behavior

Run:

```text
stingray stt -m voxtral --model-file <exact-gguf-path> -i <known-good-audio>
```

Use `examples/audio.cpp/assets/resources/a.wav` as the initial known-good audio. Record the actual
failure and where it occurs: CLI path resolution, package validation, tensor loading, tokenizer
loading, inference, or output comparison.

### 0.3 Keep the bug classifications separate

- **Not a defect:** generic `stingray -m <Voxtral GGUF> -p ...` rejection.
- **Actual gap:** `stt -m voxtral --model-file <Voxtral GGUF>` does not yet select a packed-GGUF loader
  and route it into the existing Voxtral ASR pipeline.

Do not describe the expected generic rejection as evidence that the text-generation architecture gate
should be broadened.

## Phase 1 — Inventory the exact packed GGUF

Use the actual Q8_0 checkpoint first. Do not infer its internal representation from its filename or
the model card.

### 1.1 Architecture and packed name map

Verify and record:

- `general.architecture == "audiocpp"`;
- exact `audiocpp.tensor_name_format`, if present;
- `audiocpp.tensor_names` exists, has valid entries, and its length equals `GgufModel.Tensors.Count`;
- the packed physical tensor entries resolve to unique canonical names.

Build the mapping once from the metadata table. Do not infer canonical names from tensor ordering
alone or silently accept duplicate/malformed name entries.

### 1.2 Canonical tensor inventory

Resolve and record a complete inventory of every tensor consumed by the Voxtral implementation,
including:

- `audio_tower.*`;
- `multi_modal_projector.*`;
- `language_model.*`;
- token embeddings, final norm, and output/tied-embedding representation;
- all 32 audio-tower layers and all 26 text-decoder layers.

For every required tensor record canonical/source name, element count, GGUF dimensions, matrix
orientation, and dtype. Confirm the required mappings against the existing SafeTensors loaders and the
actual local GGUF. Do not map by an assumed positional naming scheme after the metadata map exists.

### 1.3 Embedded resources

Inspect and record the full contents of:

- `audiocpp.embedded_files.names`;
- `audiocpp.embedded_files.offsets`;
- `audiocpp.embedded_files.data`.

Verify whether this exact GGUF includes `tekken.json`, `config.json`, tokenizer/config sidecars, and
any other resources required by existing Voxtral code. Validate offset bounds, lengths, and names
before extracting or parsing bytes. Do not assume the sidecars are present or absent from the phrase
"self-contained" in the model card.

### 1.4 Dtype inventory

Record actual on-disk dtypes for representative and required tensors, including audio-tower
Q/K/V/O and FFN matrices, projector matrices, decoder Q/K/V/O and FFN matrices, embeddings/output
weights, norms, and biases. This determines whether raw native Q8_0 access can be used by the current
matvec kernels or a conversion is required.

**Phase 1 acceptance:** a reproducible mapping from packed storage to the complete set of tensors
consumed by the Voxtral implementation, and an exact embedded-resource inventory. No tensor name,
orientation, dtype, or sidecar requirement is guessed; do not declare the mapping complete until every
consumed tensor is accounted for.

## Phase 2 — Reuse the existing packed-GGUF infrastructure

Do not implement a second `audiocpp` metadata parser.

- First evaluate `RvcPackedTensorSource` as the name/raw-info layer for Voxtral. Reuse it locally if
  doing so does not introduce RVC-specific semantics into Voxtral code.
- If using that type creates an inappropriate cross-namespace dependency, extract only the genuinely
  generic packed tensor-name map/raw-info/pointer behavior into a narrowly named shared source (for
  example `AudioCppPackedTensorSource`) and migrate existing consumers mechanically.
- Avoid a general audio.cpp framework redesign.
- Preserve and validate the real GGUF dtype/shape returned by the source. Do not dequantize a large
  matrix merely to discover its metadata.

## Phase 3 — Introduce a format bridge for Voxtral weights

The weight-access boundary should allow SafeTensors and packed GGUF to populate the same canonical
`VoxtralAudioEncoderWeights` and `VoxtralTextDecoderWeights` contracts, which then feed the same
inference code. This means matching tensor semantics and dimensions, not necessarily identical
quantized values:

```text
SafeTensors loader ──┐
                    ├──> canonical Voxtral weights ──> existing Transcribe path
Packed GGUF loader ─┘
```

The current Voxtral weight constructors accept concrete `SafetensorsLoader` instances, so adjust only
the storage boundary; do not duplicate the audio encoder or decoder math. Prefer the existing
`IWeightLoader` contract where it fits: `ReadF32` for small tensors and `TryGetRaw` for supported raw
matrices. Add a narrow Voxtral-specific adapter/source only if needed to preserve tensor naming and
validation.

### Quantization and memory constraints

Do not blindly perform this path for every large matrix:

```text
GGUF Q8_0 → full F32 tensor → Q8_0 requantization → inference
```

Prefer small-tensor F32 materialization where needed and raw native Q8_0 matrix access when the
source dtype, orientation, and kernel layout are compatible. Use existing quantization only when
conversion is genuinely required. Do not widen the engine API or materialize multi-gigabyte F32
intermediates to solve Voxtral.

The current Voxtral weight classes store quantized matrices in owned `byte[]` fields, and the
SafeTensors path currently creates those arrays by reading F32 and quantizing. For GGUF Q8_0 matrices,
the first implementation may copy the native Q8_0 tensor bytes into owned `byte[]` storage when that
is the smallest safe change. Do not route native Q8_0 through F32 and requantize it. Direct
memory-mapped Q8_0 pointers are preferable only if the existing weight ownership and pipeline
lifetime model can support them cleanly; do not redesign all Voxtral weight classes solely to eliminate
this copy.

### Validate tensor contracts

For each required tensor validate canonical/source name, element count, dimensions, orientation, and
dtype. Fail early with an explicit Voxtral-specific error naming the missing or malformed tensor and
expected/actual shape or dtype; do not defer these failures to a SIMD kernel.

## Phase 4 — Bridge the audio encoder and projector weights

Refactor `VoxtralAudioEncoderWeights` only at its weight-access boundary. Keep
`VoxtralAudioEncoder.Forward` mathematics unchanged.

Validate all canonical SafeTensors names for the convolutional embedder, all 32 audio layers,
`audio_tower.norm.weight`, and both projector matrices, including the names enumerated in the existing
SafeTensors loader and in the original issue notes. Pay particular attention to projection biases:
the existing loader expects selected Q/V/O and FFN biases while K has no bias; verify that the packed
GGUF matches this contract rather than guessing from architecture conventions.

**Acceptance:** both formats produce tensors with the same canonical semantics and verified
dimensions/orientation, with each format's storage and conversion path explicitly recorded. GGUF
Q8_0 and SafeTensors F32 followed by Stingray Q8_0 quantization may have different quantized block
representations; measure and report those differences rather than assuming byte identity. No
audio-tower formula changes in this phase.

## Phase 5 — Bridge the text-decoder weights

Apply the same source abstraction to the existing 26-layer Voxtral text decoder. Validate the canonical
embedding, final norm, per-layer input/post-attention norms, Q/K/V/O, gate/up/down, and
`ada_rms_norm.linear1/linear2` tensors against the actual packed inventory.

Preserve the existing Q8_0-oriented matvec semantics. Where the GGUF has native Q8_0 matrices,
prefer a compatible raw representation or only the minimum required copy; avoid a multi-gigabyte F32
round trip. Preserve tied embedding/output behavior exactly and validate vocabulary size and matrix
orientation. Compare canonical tensor semantics and dimensions; measure quantized representation
differences between the GGUF's native Q8_0 and SafeTensors-to-Stingray-Q8_0 paths rather than
assuming byte identity.

## Phase 6 — Load the embedded Tekken vocabulary safely

The existing filesystem SafeTensors loader uses `TekkenVocab.Load(tekkenPath)`. For GGUF, prefer
loading the verified embedded `tekken.json` bytes through the existing `TekkenVocab` parser, rather
than creating another tokenizer or writing bytes to disk and reading them back.

- Add a byte/span/stream-based loader only if required by the current parser API.
- Retain the filesystem path for SafeTensors packages.
- If scratch extraction is required by an existing tokenizer dependency, validate the embedded names
  and offsets, use a content-addressed/safe location, and document cleanup/lifetime behavior.
- Fail explicitly if `tekken.json` or another required sidecar is missing or malformed.

## Phase 7 — Define loader ownership and lifetime

If any weight uses a raw pointer into the memory-mapped GGUF, keep the `GgufModel` and its backing
mapping alive for the complete lifetime of the `VoxtralPipeline`. Do not dispose it immediately after
weight construction.

If weights are copied into owned buffers instead, prove that all fields—including any raw quantized
matrix storage—remain valid after disposing the GGUF. Make pipeline disposal release resources exactly
once and safely. Add a test that exercises transcription or weight access after loader construction
and verifies the intended lifetime boundary; do not infer ownership safety from successful loading.

## Phase 8 — Add a format-specific Voxtral load boundary

Keep `Transcribe()` format-independent. Expose a clear loader API such as:

```text
VoxtralPipeline.Load(safetensorsDirectory)
VoxtralPipeline.LoadGguf(ggufPath)
```

or a single verified dispatching `Load`. Both paths must populate the same canonical audio-weight,
text-weight, and `TekkenVocab` contracts and then use the existing `Transcribe` implementation.
Quantized values may differ by source/conversion path as specified above. Do not add a second GGUF
transcription implementation or make the inference method inspect file formats.

The GGUF loader must verify that the file is the expected Voxtral package, not merely an arbitrary
`audiocpp` file. Validate enough architecture metadata, canonical tensor inventory/shapes, expected
audio and decoder structure, and required tokenizer resources to reject unrelated packed checkpoints
with a useful Voxtral-specific error.

## Phase 9 — Route `SttCommand` to the dedicated ASR loader

In `SttCommand`'s `voxtral` branch, distinguish an existing SafeTensors directory/package from a
`.gguf` file:

```text
SafeTensors directory/file → existing Voxtral loader
verified GGUF file         → Voxtral GGUF loader
other input                → explicit unsupported Voxtral checkpoint error
```

Update `--model-file` help/error text to document the GGUF file form. Validate package identity
before selecting the GGUF pipeline; do not accept an arbitrary `audiocpp` file. Preserve existing
SafeTensors discovery and behavior.

## Phase 10 — Keep generic text-generation admission unchanged

Add a separate regression test for the intentional routing boundary:

- Generic `stingray -m voxtral.gguf -p "Hello"` remains rejected by
  `ModelCompatibility.ValidateForTextGeneration` for `audiocpp`.
- `stingray stt -m voxtral --model-file voxtral.gguf -i input.wav` routes to
  `VoxtralPipeline.LoadGguf` and does not invoke generic text generation.

Do not modify `ModelCompatibility`'s supported text-generation architecture set for this feature.
Do not treat one audio.cpp package containing a Mistral-style decoder as a text-generation
architecture admission precedent.

## Phase 11 — Numerical, functional, and resource validation

Use the existing SafeTensors implementation as the control and the same reference audio:

```text
examples/audio.cpp/assets/resources/a.wav
```

Expected transcript:

```text
This little work was finished in the year 1803, and intended for immediate publication.
```

### 11.1 Package metadata and tensor tests

On the actual GGUF, assert `general.architecture == audiocpp`, valid packed tensor-name map, expected
Voxtral tensors and shapes, and required embedded resources. Also test malformed name-table length,
missing required tensor, wrong architecture, invalid dimensions, and missing vocabulary to ensure clear
errors rather than incidental kernel exceptions.

### 11.2 Representative tensor parity

Compare SafeTensors and GGUF materializations for audio convolution, audio attention, audio FFN,
projector, text embedding, text attention, text FFN, AdaLN weights, and final norm. The invariant is
the same canonical tensor semantics and dimensions, not identical quantized representations. Report
shape, storage dtype, conversion path, min/max/mean/RMS, and max absolute difference in a common
comparison representation. For quantized tensors, report on-disk bytes/dtype separately and measure
the differences between native GGUF Q8_0 and SafeTensors F32 quantized by Stingray; do not claim raw
byte parity.

### 11.3 Audio-stage and end-to-end parity

With the same audio and existing preprocessing, compare mel output, audio-tower output, and projector
output before comparing decoder results. Then compare prefill logits, first generated token, several
subsequent token IDs, and the final transcript. Report stage shapes and raw metrics so any divergence
can be localized; do not change inference math to mask storage conversion differences.

Require the Q8_0 GGUF route to match the existing exact transcript and use Q8_0 as a higher-precision
control, not as a near-lossless oracle. Measure and report known quantization and conversion-path
differences separately from inference-math errors. Record revision, input, and checkpoint alongside
comparison metrics so results have clear provenance. Q8_0 is the scope for closing this bug. Q4_K is a
separate coverage extension after Q8_0 support is validated; upstream transcript results are not
evidence of Stingray behavior and must not substitute for its own tests.

### 11.4 Memory and lifetime

Measure peak memory or at minimum verify that the loader does not create a full-size F32 copy of all
Q8_0 matrices. Exercise inference while the GGUF mapping is live and disposal after pipeline use.

## Success criteria

The feature is complete when:

1. The local Q8_0 GGUF has a recorded, validated packed tensor map, dtype inventory, and embedded-file
   inventory; no schema is assumed from the model card.
2. The GGUF-backed weights satisfy the same canonical contracts as the SafeTensors weights without
   changing encoder/decoder mathematics or creating an unnecessary full-F32 checkpoint copy.
3. Embedded Tekken vocabulary loading and GGUF mapping lifetime/ownership are tested.
4. `stt -m voxtral --model-file <gguf> -i <audio>` loads through the dedicated Voxtral ASR pipeline
   and matches the existing real-audio reference transcript.
5. Existing SafeTensors Voxtral tests continue to pass.
6. Generic text generation continues to reject `general.architecture=audiocpp`; no generic architecture
   allowlist change is made.
7. Unrelated `audiocpp` checkpoints fail with a useful Voxtral-package validation error.
8. `docs/1-correctness/bugstofix.md` and any relevant command/test documentation describe the ASR
   GGUF support and preserve the intentional generic text-generation rejection.
