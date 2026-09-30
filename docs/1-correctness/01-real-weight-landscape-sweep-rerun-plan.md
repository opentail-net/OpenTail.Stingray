# Real-Weight Landscape Sweep Rerun and Failure Triage Plan

**Tracker:** [`bugstofix.md`, item 01](bugstofix.md)  
**Original sweep:** 2026-09-28, `scripts/sweep-tests.ps1`, default suites `Diffusion,Audio,Vision,ForwardPass` (649 classes resumed/run).  
**Purpose:** establish a current, evidence-backed disposition for every historical failure attributed to item 01. This is a triage-and-closure plan, not a mandate to change every component that was red in the old sweep.

## Governing principle

The resumed sweep is historical evidence, not current truth. Current source, the exact current test binary, verified model identity, and a reproducible current reference are the basis for classification. Do not change model code or numerical thresholds solely to make a historical test pass.

The execution order below is an **investigation order**, not a ranking of likely causes. For multimodal tests, explicitly compare real-input preprocessing in the production path with the reference path; synthetic fixtures alone do not prove that image/audio preprocessing matches.

## Required disposition for each reported failure

Every historical failure receives exactly one final disposition:

| Disposition | Apply when |
| --- | --- |
| **FIXED** | A real engine or test defect was reproduced, corrected, and protected by a regression. |
| **VERIFIED** | Current behavior is correct; the historical result came from stale or incorrect test/oracle data, demonstrated with current evidence. |
| **ASSET-BLOCKED** | Correctness cannot be established because the required checkpoint or reference asset is unavailable. |
| **ENVIRONMENT** | The result is explained by an incomplete download, machine setup, or other non-engine environment issue, with evidence. |
| **DUPLICATE** | The same investigation is owned by a separate numbered item/plan; link it and avoid duplicating the work. |
| **STALE** | The historical failure does not reproduce on current `main` with the required assets. |
| **DEFERRED** | The failure reproduces but is intentionally parked with a specific rationale and follow-up condition. |

Do not use an unqualified “failed” as a final state. A skipped test is not a pass; record the skip reason and classify it only when the asset/environment evidence supports that disposition.

## Evidence and run protocol

For each current targeted run, record:

- Git commit SHA and working-tree state.
- Test executable path and SHA-256 (or equivalent identity), test class and exact method.
- Exact command, including `STINGRAY_RUN_HEAVY_TESTS=1` where required; retain the repository's direct test-executable convention from `CLAUDE.md` and do not add `--nologo`.
- Model/reference path, file size and SHA-256; GGUF architecture/name/tensor identity or ONNX graph identity as applicable.
- Whether the test ran, failed, passed, or skipped; exact assertion/error and actual/expected values; duration.
- Reference implementation/version/commit, precision/dtype, input fixture identity, preprocessing path, and metric definition for numerical comparisons.

First inspect the test's asset discovery and identity checks. For a missing asset, capture the exact candidates checked and skip/error output; do not spend time diagnosing model math without the required asset. Run still-open test classes individually before considering a fresh broad sweep. Do not reuse `%TEMP%\stingray-sweep\state.jsonl` results as current unless their execution identity matches the current commit, binary, class/method, and model asset.

## Investigation sequence

### 1. Freeze current baseline and rebuild the failure matrix

1. Record current commit and working tree. Inspect the old sweep state/logs only as historical evidence.
2. Enumerate every #01 failure in the tracker and original sweep evidence, including any failure present in sweep logs but omitted from the current tracker. The continuous-batching generic failure is one known omission and must be accounted for.
3. Run each still-open, asset-ready test class directly against current `main`. Identify the exact failing test method; do not rerun all 649 classes first.
4. Build a matrix with historical result, current result, asset identity/availability, disposition, evidence link, and next action. Preserve the distinction between a non-reproduction and an asset-blocked run.

### 2. Reconcile already-addressed tracker entries

These historical cases are not invitations to reopen code. Verify that the existing evidence supports the recorded closure, and reopen only if a current targeted run fails:

- **Wan layout:** encoder Conv3D channel-outer packing and decoder Linear spatial-outer unpacking are separate contracts; no model change is indicated by the old shared-order assertions.
- **EXAONE 4.5 long-prompt fixture:** expected token count was corrected to 157 and verified against the documented llama.cpp token sequence.
- **Dequant-cache prefill parity:** the F32 diversion was disabled to preserve the quantized prefill/decode invariant; related self-consistency work is tracked separately as item 02.
- **Qwen ASR synthetic smoke:** a synthetic tone can verify pipeline structure and duration, not transcript quality. Real-speech correctness remains a separate evidence question.
- **PersonaPlex memory ranking:** mark pre-zero-copy-Q8 rankings historical/invalid for the current baseline; do not reuse them as current performance evidence.

### 3. FunASR checkpoint identity and test lookup audit

Treat the similarly named GGUF files as different models. The repository notes identify the root `models/paraformer-q8.gguf` as Fun-ASR-Nano-2512 (`general.architecture=audiocpp`) and `models/_models/paraformer-q8.gguf` as Paraformer (`general.architecture=paraformer`, with `pf.vocab`). Verify these facts against the actual local metadata before relying on them.

Audit all of the following together: `FunAsrEncoderTests`, `FunAsrWeightsTests`, `FunAsrPredictorTests`, `FunAsrRealDecoderTests`, `FunAsrRealMelExtractorTests`, `FunAsrNanoDecoderGoldenTests`, `FunAsrPerfBenchTests`, `FunAsrRealWeightsTests`, and `GgufAudioAndEmbeddingRealWeightsTests`.

- Replace filename-only selection with architecture-validating Nano and Paraformer selection (or equivalent explicit model identity). A Paraformer test must reject a Nano GGUF, and a Nano test must reject a Paraformer GGUF.
- Validate expected metadata and at least one architecture-specific tensor/key where available; do not treat mere GGUF readability or nonempty metadata as identity.
- Review `FunAsrRealWeightsTests.Paraformer_GgufRealModelFile_LoadsAndTranscribes`: its current lookup searches the root location first and its assertion expects the Nano architecture, despite the Paraformer test name. Correct both the lookup and the assertion contract.
- Review every helper's root/nested and hard-coded absolute candidate ordering. A candidate's filename does not establish its model identity.
- Keep the synthetic-tone test limited to pipeline structure; do not require a transcript from non-speech audio.

**Acceptance:** each FunASR test either loads the explicitly intended architecture and asserts that identity, or skips with a specific missing-asset reason. Record the exact path and metadata in test output where practical.

### 4. Fun-ASR-Nano decoder golden

Reproduce `FunAsrNanoDecoderGoldenTests` on current `main` with the verified Nano checkpoint and reference fixture. The recorded repeated-token output is historical until reproduced. The fixture isolates the text LLM path using fixed audio embeddings; retain that useful isolation.

If it still fails, capture the first numerical divergence—not just the final token IDs—at embedding, each decoder layer, and final logits. For the first failing boundary, collect concise statistics (mean, standard deviation, min/max, max absolute value, selected elements, argmax) for hidden state, Q/K/V, attention output, FFN output, residual, and logits as applicable. Verify actual GGUF tensor names/shapes (`token_embd.weight`, attention projections, FFN tensors, output projection) and checkpoint hyperparameters against the reference metadata. Confirm the audio-token offset and embedding vocabulary layout against the reference, rather than assuming the test's offset formula is correct.

**Acceptance:** exact expected greedy token IDs from the real decoder fixture, with the regression retained. If the checkpoint/reference cannot be verified, record the precise blocker; “output no longer repeats” is not sufficient evidence.

### 5. FunASR real-speech checkpoint dependency

The Paraformer real-speech WAV case is separate from Nano decoder isolation. Verify that the actual Paraformer GGUF is present and metadata-valid before running the WAV. If absent, classify the evidence gap **ASSET-BLOCKED**, record the WAV and expected model identity, and continue under the dedicated Paraformer real-speech investigation (the tracker/repository's referenced item #07 plan, if present). Do not debug a Nano checkpoint as if it were Paraformer.

### 6. MeloTTS asset integrity

Before engine diagnosis, establish that the local ONNX file is the expected complete model: filename/source, size and hash, valid ONNX container, graph/tensor count, input/output names, and expected architecture. Replace size-only evidence with graph/identity checks if supported by existing model APIs. Only with a verified complete checkpoint, run load/generation checks for expected sample rate, nonempty finite output, and sensible duration.

Classify an incomplete or missing checkpoint as **ENVIRONMENT** with captured evidence. Do not infer engine correctness or failure from the 50 MB threshold alone.

### 7. Parler decoder parity

Keep this as a decoder-isolation comparison: real decoder weights and codebook IDs, deterministic encoder hidden input, C# decoder output versus a regenerated independent reference. Record checkpoint identity, reference revision/package, precision/dtype, input IDs, hidden-input construction, and activation formula. Compare embedding, position embedding, each layer, and final normalization to find the first divergence; verify attention, positions, normalization, GELU, and the exact `LinearQ8_0` conversion path against the reference.

Do not lower the `0.99` threshold without evidence. If quantized conversion is implicated, distinguish conversion-path discrepancies from mathematical errors, include Q8_0 as the higher-precision control, and state exact metric provenance. Either fix a demonstrated defect (**FIXED**) or document an independently reproduced precision/reference explanation and evidence-backed tolerance (**VERIFIED**). Missing weights/reference mean **ASSET-BLOCKED**, not a guessed conclusion.

### 8. Qwen TTS Code Predictor

Reproduce the narrow real-weight forward test with the intended Talker/Code Predictor checkpoint. Record finite/non-finite status, location, and repeatability. If non-finite output remains, identify the first non-finite stage and layer through embedding, normalization, Q/K/V, QK-RMSNorm, RoPE, attention, FFN, and residual; record tensor shapes and checkpoint hyperparameters. Compare deterministic hidden/logit outputs with an independent reference where available.

**Acceptance:** at minimum, all logits finite with a permanent regression; the final correctness target is numerical parity against an independent reference, not merely finite output. Missing checkpoint/reference evidence is **ASSET-BLOCKED**.

### 9. Vision oracle reconciliation

For Exaone4, LLaVA, MimoVL, and Qwen2.5-VL, compare three things separately: current C# output, a current llama.cpp/MTMD reference where supported, and the project NumPy fixture. Record llama.cpp revision, model/projector hashes, image bytes, crop/tile layout, selected layer, preprocessing operations and resulting tensor identity, precision, and per-token metric provenance.

Before interpreting a real-image mismatch, verify that reference preprocessing matches the production path for the same real input bytes; synthetic Rainbow fixtures only cover their stated tensor path. Do not treat an old NumPy fixture as authoritative if it predates documented architecture or RoPE corrections, and do not alter the encoder or relax tolerances merely to satisfy it.

Use the evidence to identify whether the discrepancy is in production C#, current llama.cpp comparison, preprocessing, or a stale project oracle. Correct/regenerate an oracle only after reconciling the exact semantics. If the current reference/model/image assets are unavailable, record **ASSET-BLOCKED**. Preserve current llama.cpp-backed regressions, including the existing LLaVA 1.5 MTMD parity test, where applicable.

### 10. Continuous batching

Rerun `ContinuousBatchingTests` by class and identify the exact failing method; the old “values differ” message is not enough to classify it. For the failing deterministic greedy scenario, compare the promised-equivalent modes: unchunked/single request, chunked/single request, unchunked/two identical requests, and chunked/two identical requests. Capture token IDs (stronger invariant than strings), generated text, cache lengths, positions, `wantLogits`/`takes`, and completion ordering.

Bisect the packed-prefill seam (`PrefillWithCache`, `PrefillPackedMulti`, `BatchForwardMulti`) before changing the scheduler. Separate accumulation-order noise from wrong request/cache state; do not widen tolerance until the mismatch is characterized. Add a regression for the exact invariant/method that fails. If the behavior is not promised equivalent, document the contract instead of asserting an unsupported equivalence.

### 11. Sweep cache hardening and freshness

Prevent cached `done` entries from silently representing another execution. Persist and compare at least commit SHA, test assembly hash/timestamp, test class/method, model path, size, and preferably model hash. Invalidate relevant entries after changes to vision, tokenization, prefill, quantization, or model loading. Keep historical entries for audit but never present a mismatched entry as current. Avoid a wholesale harness rewrite; implement the smallest identity/invalidation change that closes this gap.

### 12. Final regression and tracker closure

For each demonstrated defect, retain a permanent regression: exact Nano token IDs; evidence-backed Parler decoder parity; QwenTTS finite/numerical checks; architecture-appropriate vision parity/oracle tests; token-level batching equivalence for the promised modes; and FunASR architecture identity tests. Run the affected test classes directly and record commit, binary, model, result, and duration.

Update item 01 in `bugstofix.md` with a dated closure note and exactly one disposition for every historical failure, linking the relevant detailed investigation when available. Include asset/environment blockers explicitly. Do not close #01 with “all tests pass” alone: closure means every historical red has a current explanation and evidence, not that unavailable assets were tested.

## Completion checklist

- [x] Every historical item from the sweep is enumerated, including failures missing from the current tracker.
- [x] Every item has exactly one final disposition and supporting current evidence or an explicit asset/environment blocker.
- [x] FunASR tests cannot silently load the wrong architecture under a shared filename.
- [x] Nano decoder has a reproduced exact-token mismatch and a specific evidence-backed defer reason.
- [x] QwenTTS historical non-finite result is non-reproducing with the intended, identity-checked checkpoint.
- [x] Continuous batching's historical exact invariant is identified and assigned to item 02; current class passes.
- [x] Four vision outcomes are reconciled as far as available checkpoints and reference paths allow; unavailable exact checkpoints and preprocessing gaps are explicitly recorded.
- [x] Parler's historical cosine reproduces on the verified checkpoint; the F32 control currently becomes non-finite in the FFN, so precision is not established and the issue is explicitly deferred.
- [x] MeloTTS is exercised using the verified complete ONNX model; placeholder lookup was fixed.
- [x] Stale sweep cache entries cannot masquerade as current runs after the identity hardening change.
- [x] `bugstofix.md` links this plan and records the dispositions.

## 2026-09-30 rerun record and final dispositions

### Baseline and scope

- Git `HEAD`: `a62447246fd0c5120484d42161394bb730d92048`; the checkout is dirty. All results below are from the current dirty worktree and newly built Release test assemblies, not attributed to the clean commit. Do not discard unrelated pre-existing changes.
- The 2026-09-28 `state.jsonl` contains 22 historical failed class entries across the four suites. Each is accounted for in the matrix below. Old cache rows were treated as historical only.
- Release builds completed with zero warnings/errors for Audio, Vision, ForwardPass, and Diffusion test projects. Audio was rebuilt again after the Parler diagnostic change. Direct runs followed the repository's `.exe -class` convention with `STINGRAY_RUN_HEAVY_TESTS=1`.
- Representative current test assembly identities (SHA-256): Audio `DDCAF473CF841576F653BDCF867242BF55EBBBA8C61696E7ED2480DC9C827D2C`; Vision `F213058816C3DED979402CFEAC580E6B59944D969DF28329D81EE9516BB09189`; ForwardPass `69030BE2F6923A0ACED64FBE88F6B9621CB722B2B846E0C669FC7411C818165A`; Diffusion `035CA1E811EAE03A72A20A283803056E43334A4A2FD5BCE4F2C64CE2F03A93C9`. Current Audio test host executable SHA-256: `8BCE8B8F35691003A02491016F54AF7B65215B3A8DFA52A7127804DE1382A104`.

### Historical failure matrix

| Historical class / result | Current evidence | Disposition |
| --- | --- | --- |
| `Diffusion.WanTests` (4/8 failed layout expectations) | Current class: 11/11 pass. Encoder Conv3D channel-outer and decoder Linear spatial-outer contracts are distinct; current regressions cover both. | **VERIFIED** |
| `Audio.Fast.FunAsrPerfBenchTests` (failed) | Current class passes with architecture/name-validated Paraformer path; historical run selected the wrong same-named GGUF. | **FIXED** (lookup/identity) |
| `Audio.Fast.FunAsrRealWeightsTests` (failed) | Current class 3/3 pass; correct Nano and Paraformer identities logged and asserted. Pure-tone path checks structure/duration, not speech quality. | **FIXED** (lookup/assertion contract) |
| `Audio.Fast.MeloTtsRealWeightsTests` (failed) | Current direct class 1/1 pass, 8.753s. Selected `_models/melotts-zh_en.onnx`, 170,429,550 bytes, SHA-256 `BF30582EB1B012250A35B1A4A80E7DFBCF8485E7BB9DE0D95EFBBEEF0E4AD86D`; loaded/generation checks pass. Empty root placeholder is ignored. | **FIXED** (asset lookup) |
| `Audio.Fast.QwenAsrRealWeightsTests` (1/2 failed) | Current direct class 2/2 pass, 3.029s with heavy gate. | **STALE** |
| `Audio.FishSpeechCodecTests` | Not part of item 01's open investigations; separately tracked Fish Speech work owns it. | **DUPLICATE** (Fish Speech tracker) |
| `Audio.FishSpeechFastArTests` | Separately tracked Fish Speech work owns this failure. | **DUPLICATE** (Fish Speech tracker) |
| `Audio.FunAsrEncoderTests` | Correct Paraformer selected; still fails cosine due to NaN. | **DEFERRED** (separate real Paraformer path investigation, item 07) |
| `Audio.FunAsrNanoDecoderGoldenTests` | Correct Nano identity confirmed (`audiocpp`, `Fun-ASR-Nano-2512`); repeats `[33108,33108,33108,33108,33108]` vs `[56568,1773,151645]`, including text-only diagnostic. | **DEFERRED** (GGUF tensor-source/forward-pass first-divergence work remains) |
| `Audio.FunAsrPredictorTests` | Correct Paraformer selected; still expected 1, actual 0. | **DEFERRED** (item 07) |
| `Audio.FunAsrRealDecoderTests` | Correct Paraformer selected; still `IndexOutOfRangeException` in `FunAsrKernels.FsmnDepthwiseConv`. | **DEFERRED** (item 07) |
| `Audio.FunAsrRealMelExtractorTests` | Current correct-model run passes. | **FIXED** (lookup) |
| `Audio.FunAsrWeightsTests` | Current correct-model run 3/3 passes; validates architecture-specific `pf.vocab` length 8404. | **FIXED** (lookup/identity) |
| `Audio.ParlerDecoderTests` | Current direct command: `$env:STINGRAY_RUN_HEAVY_TESTS='1'; .\tests\OpenTail.Stingray.Tests.Audio\bin\Release\net10.0\OpenTail.Stingray.Tests.Audio.exe -class OpenTail.Stingray.Tests.Audio.ParlerDecoderTests`. Verified `_models` safetensors, 3,511,490,560 bytes, SHA-256 `BC430EB6752B96FFB3F67036D1A6E207FBD031575A775716FFA64EF1EEB03692`. Q8 cosine remains `0.9896934984794502` vs old `>0.99` gate. F32 control returns NaN due to an arithmetic exception in FFN/GELU. No tolerance relaxation is justified. | **DEFERRED** (first-divergence/precision investigation required) |
| `Audio.QwenTtsCodePredictorForwardPassTests` | Current direct class 1/1 passes using `_models/qwen-talker-0.6b-base-Q8_0.gguf`, 992,615,488 bytes, SHA-256 `D54DBAF10591421FA764ED630D764EFA717AE40CD959BD48C66D4EB1AF226426`; output finite. | **STALE** (historical wrong/missing model selection) |
| `Vision.Exaone4VisionEmbedderParityTests` | Main checkpoint is now confirmed at `K:\_other_models\EXAONE-4.5-33B-Q4_K_M.gguf`, 20,047,839,424 bytes, SHA-256 `5BA3839B67DCEE5618EA7B2206CEDC8F9E2EC90FBCEC3C95A8CC8B33967F6BAF`. Current NumPy fixture test still fails (minimum token cosine 0.825366). Fresh `llama-mtmd-debug` run with this exact model and `mmproj-exaone-4.5-q8_0.gguf` produced the rainbow448 projector output (256×5120, sum 1411.853271). C# `LlamaMtmdVisionParityTests.Exaone45_Rainbow448_MatchesLlamaMtmdDebug` passes with sum 1409.8544 and row0 `[0.2004,0.0131,-0.2530]`, within the recorded aggregate gate. Real-image preprocessing comparison remains open. | **DEFERRED** |
| `Vision.LlavaVisionEmbedderParityTests` | NumPy fixture still fails (minimum token cosine -0.210073). Matching LLaVA 1.5 7B main GGUF absent; available LLaVA-3 checkpoint is not a substitute. | **ASSET-BLOCKED** |
| `Vision.MimoVlVisionEmbedderParityTests` | NumPy fixture still fails (minimum token cosine 0.249123). Current C# aggregate agrees with a fresh `llama-mtmd-debug` run within the existing sum gate (reference sum -8317.800781); per-token/preprocessing reconciliation remains incomplete. | **DEFERRED** |
| `Vision.Qwen25VlVisionEmbedderParityTests` | NumPy fixture still fails (minimum token cosine 0.282964). Fresh `llama-mtmd-debug` aggregate is 18550.607422, within the existing current-reference gate; per-token/preprocessing reconciliation remains incomplete. | **DEFERRED** |
| `ForwardPass.ContinuousBatchingTests` | Historical exact failure was `PrefillWithCache_DequantCacheOnOff_BitIdentical`, expected `1.03415108`, actual `0.823490143`; it belongs to item 02's dequant-cache work. Current class passes 12/12 after fixing async test mutex ownership (8.764s initial post-fix run). | **DUPLICATE** (item 02; current class green) |
| `ForwardPass.Exaone45GreedyParityTests` | Historical long-prompt expected token count corrected to 157 against documented llama.cpp token sequence; full required EXAONE model is unavailable for a current full-weight rerun. | **ASSET-BLOCKED** |
| `ForwardPass.PrefillDecodeSelfConsistencyTests` | Current direct class 10/10 passes (7.064s); historical seam is covered by item 02's prefill/dequant-cache work. | **DUPLICATE** (item 02) |

### Focused investigation notes

- **FunASR identities:** `models/paraformer-q8.gguf` is Nano (`general.architecture=audiocpp`, name `Fun-ASR-Nano-2512-hf-854d88f`), size 1,045,334,432, SHA-256 `4D727357574B079B7F43336B2930F39DA086CA02F5D8D50872090B4C1C3D5E0A`. `models/_models/paraformer-q8.gguf` is Paraformer (`general.architecture=paraformer`, 8,404-entry `pf.vocab`), size 236,929,024, SHA-256 `42BF76EA1575A336AACA4C1B7C01A82B79113E6D04D0D6B799561BFCF07EE011`. The locator rejects wrong architecture and checks architecture-specific metadata; tests print selected path, size, and identity. The real-speech WAV is present (179,646 bytes; SHA-256 `668BF8DF51A10027B84D5D8816A1CE11AE93545538DC05CFE2AA6811D399C250`) and the Paraformer ONNX reference is present (81,828,675 bytes; SHA-256 `3EF6C19369B912F7CAF3CEF8E545C5CCD1A33D9D7EC792A46668DC41C4B229EC`). The GGUF WAV run still returns an empty transcript while the ONNX real-weight run recognizes speech; see item 07 for the dedicated stage-by-stage investigation. This plan does not claim ASR correctness from the tone smoke.
- **Nano decoder:** checkpoint tensor identity includes `blk.0.attn_q.weight` shape `[1024,2048]` and `token_embd.weight` shape `[1024,151938]`. Current evidence narrows the mismatch to the isolated text decoder path but does not capture first divergent layer/stats; that work remains explicitly deferred.
- **Vision references:** fresh `llama-mtmd-debug` binary SHA-256 `94874105ADDE0C2662F89D7B0451E8C7F9FCF496807D60167B5F4AD904297435`. Full MimoVL and Qwen2.5-VL checkpoints/projectors are present and their synthetic Rainbow448 reference runs exit successfully. The EXAONE main model is available on `K:\_other_models` and its fresh rainbow448 reference plus current C# aggregate gate now pass as detailed above. The original EXAONE NumPy fixture still fails. These checks do not establish real-image preprocessing equivalence or NumPy-oracle freshness. LLaVA 1.5 exact main weights remain unavailable.
- **Sweep cache:** `scripts/sweep-tests.ps1` now keys reusable completion on commit, test host executable identity, test assembly path/hash/timestamp, suite/class, plus stdout-reported model path/size/hash when available. Legacy rows without identity are rerun. Verified with a scratch state: first invocation reran the class and captured SmolLM2 asset hash `77665EA4815999596525C636FBEB56BA8B080B46AE85EFEF4F0D986A139834D7`; the next same-identity invocation reused it. ISO timestamps are normalized after JSON round-trip. Asset identity is available only when a test emits its asset path, so the cache cannot claim a model hash for silent asset consumers.
- **Other closures:** Wan class 11/11 pass. Qwen ASR 2/2 pass. `PrefillDecodeSelfConsistencyTests` 10/10 pass. Melo and Qwen TTS exact checkpoint identity and current run results are above. Qwen ASR's synthetic tone is not treated as speech-quality evidence.

The classifications above close the historical sweep triage, not every separate unresolved correctness issue. Deferred items retain an explicit owner/follow-up; asset-blocked rows must be reopened when the matching checkpoints become available. No numerical threshold was lowered in this rerun.
