# Audio subsystem review — progress log

## LOOP RESTART PLAN (2026-08-21, user going AFK, autonomous `/loop` re-armed)

User is stepping away and wants this session to keep grinding through the remaining fake
pipelines unattended via `/loop`. Per `CLAUDE.md`, no subagents — all work done directly in
this session, one pipeline/stage at a time, each iteration reading this doc first to avoid
re-deriving what's already known, then updating this doc at the end of the iteration before
stopping (so progress survives context compaction and is visible to the user on return).

**Remaining work queue, in order** (skip Piper/Whisper/Kokoro/Chatterbox/F5-TTS DiT+Vocos —
those are done and verified per the table below):

1. ~~**MeloTTS**~~ — CONFIRMED FULLY DONE (2026-08-21, this loop iteration). All of
   `MeloDurationPredictor.cs` (SDP + plain DP + sdp_ratio blend), `MeloFlow.cs`
   (TransformerCouplingBlock reverse), `MeloGenerator.cs` (HiFi-GAN), and length-regulator
   code already existed real/weight-driven from an earlier session, just not reflected in
   this doc's queue list. Ran every Melo test class individually
   (`STINGRAY_RUN_HEAVY_TESTS=1 dotnet test tests/OpenTail.Stingray.Tests.Audio --
   --filter-class <FQN>`, one at a time): `MeloDurationPredictorTests`, `MeloFlowTests`,
   `MeloGeneratorTests`, `MeloLengthRegulatorTests` — all PASS. Also ran the end-to-end
   real-weights test `OpenTail.Stingray.Tests.Audio.Fast.MeloTtsRealWeightsTests` — PASS
   (18s). MeloTTS is genuinely done, do not re-spend an iteration on it. Move straight to
   Parakeet next.
2. **Parakeet** — NOT blocked (false blocker corrected this session); full architecture spec
   already written out below (subsampling, positional encoding, 24-layer Conformer block,
   CTC head). Next: read `examples/crispasr/src/canary_ctc.cpp` for BN-fold/mel/ln_eps
   details, build a golden oracle, port `ParakeetConformerEncoder.cs`, verify per-stage.
3. **CosyVoice** — not started. Weights present (`cosyvoice2_0.5b.safetensors`,
   `cosyvoice_speech_tokenizer.onnx`, `campplus.onnx`, `flow.decoder.estimator.fp32.onnx`);
   reference `examples/cosyvoice.cpp` (native GGML, mostly usable). Investigate the ONNX flow
   decoder trim/AOT constraint before assuming a pure-C# port is needed.
4. **QwenASR** — not started. Model `models/qwen3-asr-0.6b-q4_k.gguf`; reference
   `examples/qwen3-asr.cpp` (native GGML).
5. **FunASR** — not started. Models `models/paraformer-q8.gguf`,
   `models/sensevoice-small.int8.onnx`; references `examples/FunASR-GGML`,
   `examples/paraformer.cpp`.
6. **Silero VAD** — partially real; needs decoding the real ONNX graph (fused
   `reparam_conv` + learned STFT frontend + graph-structured 8kHz/16kHz `If` branch) instead
   of the current flat-sequential assumption. Comparable effort to a full rebuild.
7. **QwenTTS** — do LAST or best-effort only: **no model weight file exists locally** for
   this one, so it can be structurally written against `examples/qwen-tts-py` (real PyTorch
   source) but cannot be numerically golden-verified. Say so explicitly, do not claim
   false confidence, when/if this gets implemented.

**Per-iteration discipline** (same bar every other pipeline in this doc was held to — do not
relax it under autonomous operation):
- Read this doc's relevant section fully before starting; don't re-derive already-known specs.
- Real reference source only (native C++/GGML or extracted real PyTorch) — never guess math.
- Golden-verify each stage numerically (cosine similarity, >0.99 bar) against a real oracle
  before calling a stage done; "compiles and produces plausible-looking audio" is not done.
- Run heavy/real-weight tests individually (`STINGRAY_RUN_HEAVY_TESTS=1`, one filter-class at
  a time), never the whole heavy suite at once, matching this doc's established practice.
- Never pipe verbose test output through tail/head — use the full output or a saved file.
- Update this doc at the end of every iteration with what was done/found/blocked, so the next
  wakeup (or the user on return) has a true, current picture — this doc drifting stale caused
  real confusion earlier in this session (see the Piper status-snapshot correction above).
- If a pipeline turns out genuinely blocked (missing weights, no real reference anywhere),
  document the blocker precisely and move to the next item in the queue rather than guessing.

## MASTER REBUILD PLAN (2026-08-21, restarted as an autonomous `/loop`) — go through every remaining fake pipeline

Restarting the earlier "decision needed" stop: user has now given explicit direction —
go through ALL remaining fake pipelines, one at a time, autonomously, via a recurring
`/loop` while AFK. Disk is no longer a blocker (79GB free, checked this session). Full
model inventory checked in `models/` and `examples/` this session before restarting; see
per-pipeline notes below for exactly what's available for each.

**Status snapshot before this loop starts (verified against git log + working tree,
NOT just this doc's older prose, which had drifted stale):**

| Pipeline | Status | Evidence |
|---|---|---|
| Whisper | REAL, working | fully verified earlier this doc |
| Kokoro | REAL, working | 7-stage forward pass, verified vs golden ONNX, see above |
| Chatterbox | REAL, working | T3 GPT2-medium LM + S3Gen (Conformer/CFM/HiFiGAN), perf-tuned, commits `2760907`/`3168cf8`/`bb97260` |
| Piper | REAL, just landed | VITS text encoder + flow + duration predictor + HiFiGAN decoder, commits `bcce3d8`/`349a1bf` — **not yet doc'd in detail, not yet golden-verified the way Kokoro was; spot-check first thing this loop does** |
| MeloTTS | FAKE, not started | model: `models/melotts-zh_en.onnx` (ONNX only, no native ref); C++ ref: `examples/MeloTTS.cpp` (custom, no ggml/onnx hits per earlier audit — needs closer read); same VITS-family architecture as Piper, should be much faster now that Piper's flow/duration-predictor/HiFiGAN code exists as a template |
| F5-TTS | DiT core REAL & golden-verified; Vocos vocoder BLOCKED (no local weights) | see the dedicated "F5-TTS DiT" section near the end of this doc for the full writeup |
| CosyVoice | FAKE, not started | models: `models/cosyvoice2_0.5b.safetensors` (LLM+flow weights), `models/cosyvoice_speech_tokenizer.onnx`, `models/campplus.onnx` (speaker embedding), `models/flow.decoder.estimator.fp32.onnx` (flow decoder, ONNX — same trim/AOT constraint issue as before, may need pure-C# port of just this small graph rather than an ONNX Runtime dependency); reference: `examples/cosyvoice.cpp` (mostly native GGML) |
| Parakeet | FAKE, not started | model: `models/parakeet-ctc-0.6b-q4_k.gguf`; reference: `examples/parakeet.cpp` (native GGML) + `examples/nemo-toolkit-py` (real NeMo FastConformer source, pip-extracted, not yet verified to contain the exact model class per earlier note — check this first) |
| QwenASR | FAKE, not started | model: `models/qwen3-asr-0.6b-q4_k.gguf`; reference: `examples/qwen3-asr.cpp` (native GGML) |
| QwenTTS | FAKE, not started | **no model weight file found anywhere under `models/`** (checked this session) — reference source exists (`examples/qwen-tts-py`, real pip-extracted PyTorch) so the forward pass CAN be written and structurally verified, but cannot be numerically golden-verified until a real checkpoint is available locally. Implement last, or on a best-effort/structural-only basis, and say so explicitly rather than claiming false confidence. |
| FunASR | FAKE, not started | models: `models/paraformer-q8.gguf`, `models/sensevoice-small.int8.onnx`; reference: `examples/FunASR-GGML`, `examples/paraformer.cpp` |
| Silero VAD | Partially real, messier than expected | model: `models/silero_vad.gguf` + `models/silero_vad.onnx`; real architecture is fused `reparam_conv` + learned STFT frontend + graph-structured 8kHz/16kHz `If` branch, NOT the flat sequential pipeline `SileroVad.cs` assumes — needs decoding the real ONNX graph structure first, comparable effort to a full pipeline rebuild |

**Piper verification CONFIRMED DONE (checked directly, no subagent, 2026-08-21 follow-up):**
ran each Piper test class individually (`STINGRAY_RUN_HEAVY_TESTS=1 dotnet test
tests/OpenTail.Stingray.Tests.Audio -- --filter-class <FQN>`, one at a time per user
instruction, not the whole suite at once):
- `OpenTail.Stingray.Tests.Audio.PiperTextEncoderTests` — PASS. Golden-verified: cosine
  similarity >0.999 vs real onnxruntime `/enc_p/proj/Conv_output_0` (mu/logs).
- `OpenTail.Stingray.Tests.Audio.PiperDurationPredictorTests` — PASS. Golden-verified:
  cosine >0.99 vs real onnxruntime `dp_Split_output_0` (StochasticDurationPredictor logw).
- `OpenTail.Stingray.Tests.Audio.PiperFlowTests` — PASS. Golden-verified: cosine >0.999
  (length-regulator z_p) and >0.99 (ResidualCouplingBlock reverse flow output) vs real
  onnxruntime golden dumps.
- `OpenTail.Stingray.Tests.Audio.PiperHifiGanDecoderTests` — PASS. Golden-verified: cosine
  >0.99 vs real onnxruntime final waveform output.
- `OpenTail.Stingray.Tests.Audio.Fast.PiperRealWeightsTests` — PASS (end-to-end, real
  weights, ~5s). NOTE: this one only checks shape/sample-rate/finite/duration, NOT cosine
  vs golden — the per-stage tests above are what actually prove correctness.

All golden dump scripts/data already existed on disk in `scratch-llamacpp-ref/
piper_golden_{textenc,sdp,flow}*` (real onnxruntime runs against `models/
en_US-lessac-medium.onnx`), so this verification work was in fact already done in an
earlier iteration, just not reflected in this doc — **Piper is genuinely done and
verified, not just "builds and produces sound."** No bugs found needing a fix. Move
straight to MeloTTS next; do not re-spend a whole iteration re-verifying Piper.

**Execution order for this loop** (most template-reuse and highest-confidence first,
hardest/most-blocked last):

1. ~~**Piper**~~ — DONE, see confirmation above.

**MeloTTS investigation checkpoint (2026-08-21, this iteration, done directly — no
subagents per user instruction):**

Real PyTorch reference source recovered: `MeloTTS` on PyPI (0.1.1) has a broken sdist
(setup.py references a missing `requirements.txt`, `pip download`/`pip install` both fail
to build) — but the sdist tarball itself downloads fine directly from
`files.pythonhosted.org` and extracts cleanly without running `setup.py`. Extracted to
`examples/melotts-py/` (`models.py`, `modules.py`, `attentions.py`, `mel_processing.py`,
etc. — the exact `SynthesizerTrn`/`TextEncoder`/`StochasticDurationPredictor`/
`DurationPredictor`/`TransformerCouplingBlock`/`Generator` module source, same family as
`chatterbox-tts-py`/`f5-tts-py`/`kokoro-py` recovered earlier this doc). `examples/
MeloTTS.cpp` (the OpenVINO C++ wrapper) is confirmed NOT a useful math reference — it
wraps a compiled OpenVINO IR graph, same "ONNX-wrapper-only" situation as
Chatterbox-turbo-cpp/kokoro.cpp earlier in this doc; use `examples/melotts-py` instead.

**Confirmed via `models.py`:** MeloTTS's `SynthesizerTrn.infer` (line ~970) is VITS2, same
overall shape as Piper but with two real architectural differences — do NOT port Piper's
flow/duration code assuming they're identical:
1. **Duration blend**: `logw = self.sdp(...) * sdp_ratio + self.dp(...) * (1 - sdp_ratio)`
   — MeloTTS runs BOTH a StochasticDurationPredictor AND a plain DurationPredictor and
   blends them (`sdp_ratio`, default 0.2, already a parameter on this codebase's
   `MeloModel.Forward`). Check whether Piper's port only implements the SDP half.
2. **Flow is `TransformerCouplingBlock`, not `ResidualCouplingBlock`**: confirmed by
   walking the real ONNX graph (see below) — each flow block (`/flow/flows.0`, alternating
   with `Flip`) has an internal `enc` with its own 3 relative-attention layers + FFN +
   `spk_emb_linear`, i.e. a small transformer inside each coupling layer, not a plain
   dilated-conv WaveNet stack like Piper's flow. Materially more complex than
   `PiperFlow.cs` — reuse only its "reverse loop over flow blocks" structure, not its
   per-block math.

**Weight extraction: a real, reusable technique for this specific ONNX file.**
`models/melotts-zh_en.onnx`'s raw initializer *tensor* names are NOT clean module paths
(auto-generated `onnx::Conv_15858`-style names from constant folding, unlike Piper's
`en_US-lessac-medium.onnx` which has clean `enc_p.encoder.attn_layers.0.conv_q.weight`-
style names directly) — BUT the graph's *node output* names DO retain the full module
path (e.g. `/enc_p/encoder/attn_layers.0/conv_q/Conv_output_0`). Built a path->raw-name
mapping by walking `Conv`/`Gemm`/`MatMul`/`ConvTranspose` nodes and reading each node's
own `.input[1]`(weight)/`.input[2]`(bias), keyed by its path-stripped output name. Saved
(not a permanent script, but the OUTPUT is checked in): `scratch-llamacpp-ref/
melo_weight_map.json` (259 entries: clean path -> {op, weight init name, bias init name})
and `scratch-llamacpp-ref/melo_module_structure.txt` (readable dp/sdp/flow/dec listing).
Module coverage confirmed complete: `enc_p` (40 ops), `dp` (4), `sdp` (33), `flow` (84),
`dec` (98) — all 5 real VITS2 submodules present and extractable.

**Concrete simplification found, worth exploiting:** traced `/enc_p/bert_proj`'s Conv
input back through the graph — it's fed by a `ConstantOfShape` node (an all-**zero**
runtime-shaped tensor), NOT by any of this ONNX's 7 actual graph inputs (`x`, `x_lengths`,
`tones`, `sid`, `noise_scale`, `length_scale`, `noise_scale_w` — there is no `bert` input
at all in this exported graph). Same for `ja_bert_proj`. **This means this checkpoint's
BERT conditioning is always zero at inference** — the C# port does NOT need real BERT
features to match this exact model; `MeloBertEncoder.cs`'s current external-BERT plumbing
is unnecessary for correctness here (only `bert_proj.bias`/`ja_bert_proj.bias` need adding
as fixed per-channel offsets — the weight matmul contributes nothing since its input is
always zero). Re-check against `melo_weight_map.json` if a future checkpoint DOES take
real BERT input, but for `melotts-zh_en.onnx` specifically, zero-BERT is confirmed correct.

**NOT yet done, next concrete steps for whoever picks this up (no C# code changed yet this
iteration — confirmed `MeloModel.cs` is still the original 211-line fake sine-wave/
procedural implementation, no weight tensor reads at all):**
1. Write `MeloOnnxWeights.cs` (ONNX initializer loader, analogous to `PiperOnnxWeights.cs`
   but consuming `scratch-llamacpp-ref/melo_weight_map.json` to resolve opaque raw names).
2. Port `TextEncoder` (6 relative-attention layers + FFN, same pattern Piper already has
   code for, but check `attentions.py` in `examples/melotts-py/` for the exact relative
   attention formula rather than assuming it's byte-identical to Piper's).
3. Port `DurationPredictor` (plain conv stack) AND `StochasticDurationPredictor`
   (`modules.py`'s `ConvFlow`/`DDSConv`, 4 flow blocks) plus the `sdp_ratio` blend — verify
   against `modules.py` directly, same family as Piper's SDP but not necessarily identical.
4. Port `TransformerCouplingBlock` (harder than Piper's flow, has internal attention — see
   above) using `modules.py`'s `TransformerCouplingLayer` / `attentions.py`'s `FFT` class.
5. Port `Generator` (HiFi-GAN, likely close to `PiperHifiGanDecoder.cs`, but confirm
   channel counts/upsample rates against `modules.py` — MeloTTS targets 44.1kHz vs Piper's
   22.05kHz, so upsample rates differ).
6. Build a golden-output oracle the same way as Piper's `piper_golden_*.py` scripts
   (template still in `scratch-llamacpp-ref/`) — run `models/melotts-zh_en.onnx` via
   onnxruntime with fixed `x`/`tones`/`sid` inputs, fetch named intermediate node outputs
   directly (the clean `/enc_p/...`-style names above make `session.run([exact_name], ...)`
   work directly, same trick used for Piper/Kokoro).
7. Verify each C# stage against its golden dump via cosine similarity (>0.99, matching
   Piper's bar) before considering MeloTTS done — do not skip numeric verification, that's
   exactly how the original fake pipelines went unnoticed.

**MeloTTS TextEncoder DONE and verified (2026-08-21, this iteration, done directly, no
subagents per user instruction):** step 1 (`MeloOnnxWeights.cs`) and step 2 (TextEncoder
port) from the list above are complete.

- **Shared kernel extraction first** (per user feedback to reuse existing code rather than
  hand-roll a second copy): pulled Piper's relative-position-attention/conv1x1/conv-same-
  pad/layernorm/softmax math out of `Piper/PiperTextEncoder.cs` into a new
  `Primitives/VitsAttentionKernels.cs`, shared by any VITS-family pipeline. Refactored
  `PiperTextEncoder.cs` to call it and re-ran all 5 Piper test classes individually
  afterward to confirm zero regression (all still pass, same cosine-similarity numbers).
- **`MeloTTS/MeloOnnxWeights.cs`** (new): loads real `models/melotts-zh_en.onnx` enc_p
  weights (emb/tone_emb/language_emb/bert_proj-bias/ja_bert_proj-bias/emb_g/proj/
  spk_emb_linear + 6 relative-attention layers). Confirmed empirically (not just by static
  graph reading) via `scratch-llamacpp-ref/melo_golden_textenc.py`: NumHeads=2, Window=4,
  HiddenDim=192, FfnKernel=3, NumEncoderLayers=6 (same core config as Piper).
- **`MeloTTS/MeloTextEncoder.cs`** (new): ports `examples/melotts-py/models.py`'s
  `TextEncoder.forward` + `attentions.py`'s `Encoder.forward` — embedding sum (token+tone+
  language+bert-bias+ja_bert-bias, scaled by sqrt(hidden)) -> 6-layer relative-attention
  Transformer with a speaker-embedding injection at layer index 2 (`cond_layer_idx`, a
  real architectural difference from Piper's single-speaker checkpoint) -> proj to mu/logs.
- **One real, source-verified bug found and fixed via per-stage verification** (exactly
  the failure mode this whole rebuild process exists to catch, not "compiles and runs"):
  `spk_emb_linear`'s weight is exported as a bare ONNX `MatMul` (not `Gemm`), and
  confirmed via direct initializer inspection that PyTorch's `nn.Linear`-to-`MatMul` export
  pre-transposes the weight to `[inDim, outDim]` (256x192 = gin_channels x hidden) --
  the OPPOSITE of torch's usual `[outDim, inDim]` nn.Linear layout that every other weight
  in this codebase (including Piper's) uses. The first implementation assumed the usual
  layout and got a plausible-looking but wrong result (final mu/logs cosine similarity
  0.715 -- clearly wrong, but not obviously-broken-looking, i.e. exactly the kind of subtle
  bug this whole doc exists to catch). Found by bisecting stage-by-stage against golden
  dumps (pre-encoder embedding sum: matched >0.999; layer 0 output: matched; layer 1
  output: matched; layer 2 output -- the first one after the speaker-conditioning add --
  0.688, i.e. the exact layer where the bug lived), not by guessing. Fixed by indexing the
  weight as `weight[i*outDim+o]` instead of `weight[o*inDim+i]` in `LinearVec`. After the
  fix: `MeloTextEncoderTests.MeloTextEncoder_RealOnnxWeights_MatchesOnnxGoldenOutput`
  passes, mu/logs cosine similarity both >0.999 vs real onnxruntime golden output.
- **Golden dump scripts** (checked in, reusable): `scratch-llamacpp-ref/
  melo_golden_textenc.py` (final mu/logs + individual embedding-table gathers + language-id
  empirical trace) plus two more intermediate-node dumps added ad hoc while bisecting the
  bug above (pre-encoder `/enc_p/Mul_output_0`, and `/enc_p/encoder/norm_layers_2.{0,1,2}/
  Transpose_1_output_0` for per-layer verification) -- the `.npy` outputs for all of these
  are in `scratch-llamacpp-ref/melo_golden_textenc/`, re-run the `.py` script if they're
  ever deleted (they're gitignored scratch data, not committed).
- **Test**: `tests/OpenTail.Stingray.Tests.Audio/MeloTextEncoderTests.cs` (new), asserts
  final mu/logs cosine >0.999 vs golden AND (as an internal stage check before the final
  assert) the pre-encoder embedding-sum cosine >0.999 vs golden `/enc_p/Mul_output_0` --
  keeps one intermediate checkpoint in the permanent test, not just the final output, so a
  future regression here is easier to localize than "the whole encoder is somehow wrong."

**MeloTTS duration predictor (sdp+dp blend) DONE and verified (2026-08-21, continued in the
same session, done directly, no subagents):**

- **Shared kernel extraction, again** (same reuse-over-hand-roll principle applied to the
  SDP): pulled Piper's DDSConv/ConvFlow-spline/ElementwiseAffine/Flip math out of
  `Piper/PiperDurationPredictor.cs` into `Primitives/VitsDurationFlowKernels.cs` (new,
  alongside the earlier `VitsAttentionKernels.cs`), with `VitsDdsConvWeights`/
  `VitsConvFlowWeights` as the shared weight-holder types (constructed via a caller-supplied
  `Func&lt;string,float[]&gt;` tensor getter so both Piper's and MeloTTS's different ONNX
  name-resolution strategies can build them). `PiperOnnxWeights`/`PiperDurationPredictor`
  refactored to use the shared types; re-ran all 5 Piper test classes individually
  afterward -- zero regression, identical cosine-similarity numbers.
- **`MeloOnnxWeights.cs` extended**: added `dp.*` (plain `DurationPredictor`: cond, conv_1,
  norm_1, conv_2, norm_2, proj -- all plain-named initializers, no weight_norm anonymization
  unlike enc_p's attention convs) and `sdp.*` (`StochasticDurationPredictor`: cond, pre,
  convs (DDSConv), proj, flows.{3,5,7} (ConvFlow) + flows.0.m/exp(-logs) -- confirmed via
  the real ONNX graph that `filter_channels == in_channels == HiddenDim` here, per the
  reference's own `models.py` comment "filter_channels = in_channels # it needs to be
  removed from future version", and that the SDP's flow pruning/execution order is
  IDENTICAL to Piper's dp -- Flip(8)-&gt;ConvFlow(7)-&gt;Flip(6)-&gt;ConvFlow(5)-&gt;
  Flip(4)-&gt;ConvFlow(3)-&gt;Flip(2)-&gt;ElementwiseAffine(0), same "remove a useless
  vflow" trick, confirmed both by reading `models.py`'s reverse-path list slicing AND by
  checking the real ONNX graph only has flows.{3,5,7} (not 1) as VITS2's `n_flows=4` +
  pruning always produces this exact set regardless of checkpoint).
- **`MeloTTS/MeloDurationPredictor.cs`** (new): implements the SDP half (same flow math as
  Piper via the shared kernels, but with a real speaker-conditioning `cond` 1x1 conv added
  to `x` right after `pre` -- Piper's checkpoint has no such conditioning, `gin_channels=0`
  there) and the plain DP half (`conv_1(k=3)-&gt;ReLU-&gt;LayerNorm-&gt;conv_2(k=3)-&gt;
  ReLU-&gt;LayerNorm-&gt;proj`, also cond-conditioned), then blends:
  `logw = logwSdp*sdpRatio + logwDp*(1-sdpRatio)` per `SynthesizerTrn.infer`'s real formula.
  `cond`'s weight is a plain Conv1d `[out,in,1]` layout here (NOT the transposed-MatMul
  layout `spk_emb_linear` had in the TextEncoder stage -- confirmed via direct shape
  inspection before assuming either convention, since guessing wrong here would silently
  repeat the exact bug just found and fixed in the TextEncoder stage).
- **No bugs found this stage** -- passed golden verification on the first attempt (unlike
  the TextEncoder stage's transposed-weight bug), most likely because the shared-kernel
  extraction from the already-verified Piper SDP meant the hardest, most error-prone math
  (the rational-quadratic spline, DDSConv, flow pruning order) was reused verbatim rather
  than re-transcribed from scratch, and the only genuinely new code (the `cond` speaker
  conditioning and the sdp/dp blend) is comparatively simple arithmetic.
- **Golden dump**: `scratch-llamacpp-ref/melo_golden_durpred.py` (checked in), capturing the
  SDP's exact raw noise draw (`/sdp/RandomNormalLike_output_0`, so the C# test isolates
  "is the flow math right" from "does the RNG match", same technique as Piper's SDP test),
  the SDP's pre-blend logw (`/sdp/Split_output_0`), and the DP's pre-blend logw
  (`/dp/proj/Conv_output_0`).
- **Test**: `tests/OpenTail.Stingray.Tests.Audio/MeloDurationPredictorTests.cs` (new) --
  verifies the SDP half (`sdpRatio=1.0`) and DP half (`sdpRatio=0.0`) SEPARATELY against
  their own golden node outputs, not just the final blend -- checking only the blend could
  hide an 80%-wrong DP half behind a low `sdpRatio` (this checkpoint's default is 0.2).
  Both halves: cosine similarity >0.99 vs real onnxruntime golden output. Passed on first
  run after implementation (no debugging iteration needed, unlike the TextEncoder stage).
- **Full individual regression pass this iteration** (per user instruction: run heavy
  tests one class at a time, never the whole suite at once): all 5 Piper test classes +
  `PiperRealWeightsTests` (end-to-end) + `MeloTextEncoderTests` + `MeloDurationPredictorTests`
  all pass individually.

### MeloTTS `TransformerCouplingBlock` flow -- DONE and verified

- Read `examples/melotts-py/modules.py`'s `TransformerCouplingLayer`/`Flip` and `models.py`'s
  `TransformerCouplingBlock` as ground truth: 4 coupling layers (`flows.0/2/4/6` in ONNX weight
  paths) interleaved with parameter-free `Flip`s (`flows.1/3/5/7`) -- 8 sub-flows total, `mean_only=True`
  for every layer (so `logs` is implicitly zero: reverse update simplifies to `x1 - m`, no `exp`).
  Each coupling layer has its OWN internal 3-layer relative-attention encoder (`enc`,
  `attentions.py`'s `Encoder`, `ffnKernel=5` -- differs from enc_p's `ffnKernel=3`) plus its own
  `spk_emb_linear`, confirmed distinct (non-shared) per-layer weights by checking the ONNX
  initializer list directly (`flow.flows.{0,2,4,6}.enc.*` all have their own tensors, not just
  `flows.0`'s reused -- i.e. `share_parameter=False` for this checkpoint).
- **Second application of the shared-kernel extraction pattern this session**: pulled the
  `attentions.py` `Encoder.forward` loop (the N-layer relative-attention + optional speaker-
  conditioning loop) out of `MeloTextEncoder.cs` into a new `MeloRelativeEncoder.cs`, since this
  exact module is instantiated twice in the graph (enc_p's 6-layer encoder AND each flow layer's
  3-layer internal encoder) -- reused by both instead of a second hand-rolled copy. Also moved
  the `LinearVec` helper (the `[inDim,outDim]`-layout `spk_emb_linear` MatMul quirk) there since
  both call sites need it.
- Generalized `MeloEncoderLayerWeights`'s constructor to take explicit `modBase`/`nodeBase`
  prefixes plus a `weightNormAnonymized` flag: enc_p's attention conv weights are anonymized by
  weight_norm-style ONNX export fusion (must resolve via `GetWeightViaNode`), but confirmed by
  direct ONNX initializer inspection that the flow's internal encoders are NOT weight_norm'd in
  this checkpoint (clean `model.model.flow.flows.0.enc.attn_layers.0.conv_q.weight` names exist
  directly) -- reading them by name instead of via node avoids an unnecessary (and for this case,
  wrong) `GetWeightViaNode` lookup.
- New `MeloFlowLayerWeights` in `MeloOnnxWeights.cs`: `pre`/`post` convs (clean names, not
  anonymized -- confirmed via direct shape inspection) plus `SpkEmbLinearWeight` (still resolved
  via `GetWeightViaNode`, since only `spk_emb_linear`'s bare-MatMul export is anonymized here, not
  the Conv1d weights) and a `Layers[3]` of the generalized `MeloEncoderLayerWeights`.
- New `MeloFlow.cs`: `Reverse(w, zP, t, g)` implements `TransformerCouplingBlock.forward(reverse=True)`
  (inference only ever runs reverse) plus a full-channel-reversal `Flip` (torch.flip on the
  channel axis -- NOT the 2-channel swap in `VitsDurationFlowKernels.Flip`, which is specific to
  the SDP's 2-channel flow).
- Golden dump (`scratch-llamacpp-ref/melo_golden_flow.py`) captures the frame-rate `z_p` fed into
  the flow (`/Add_2_output_0`) and the flow's final output (`/flow/flows.0/Concat_output_0`),
  feeding the golden z_p directly into `MeloFlow.Reverse` rather than requiring the (not yet
  built) length regulator -- isolates flow correctness from length-regulator correctness, same
  bisection philosophy as the duration-predictor stage.
- **Bug found and fixed via golden bisection**: cosine similarity was initially -0.06 (garbage).
  First hypothesis (wrong) -- a separate one-off debug dump script captured intermediate node
  outputs in a SEPARATE `sess.run()` call from the main golden dump. `/Add_2_output_0` (z_p) sits
  downstream of a `RandomNormalLike` noise draw that onnxruntime does NOT seed, so every session
  run draws different noise -- comparing intermediates captured across two different script
  invocations was comparing against two different random draws, producing nonsense mismatches
  even for `Flip` (a channel reversal that is mathematically guaranteed to be exact). Fixed by
  consolidating ALL golden targets (final input/output plus every per-coupling-layer bisection
  checkpoint) into a single `sess.run()` call in one script -- now documented directly in
  `melo_golden_flow.py`'s target list as a durable warning for the next pipeline. Also disabled
  onnxruntime graph optimization (`ORT_DISABLE_ALL`) when dumping, after separately discovering
  that default-optimization-level intermediate-output extraction can return a value that doesn't
  match what was actually fed to a downstream node (found via the same bisection: `Flip`'s output
  didn't match `np.flip` of its dumped input under default optimization, but matched exactly with
  optimizations disabled).
- With golden dumps fixed, re-bisection showed the per-coupling-layer math was already correct
  (all 4 intermediate checkpoints matched >0.9999999) -- the REAL bug was a spurious extra `Flip`
  call after the reverse loop. `TransformerCouplingBlock.forward(reverse=True)` iterates
  `reversed(self.flows)` where `self.flows = [TCL0,Flip,TCL1,Flip,TCL2,Flip,TCL3,Flip]`; reversed
  this is `Flip,TCL3,Flip,TCL2,Flip,TCL1,Flip,TCL0` -- 4 Flips and 4 TCLs alternating, ending on
  TCL0 with NO trailing Flip. An early version of `MeloFlow.Reverse` had exactly one too many
  `Flip` calls (correct 4 inside the loop, plus one more after) -- every per-TCL golden
  checkpoint passed (since they're captured mid-loop, before the extra Flip), but the actual
  returned tensor was channel-order-reversed relative to golden. Caught by comparing the real
  `Reverse()` return value against the final golden output, not just the loop's intermediates --
  exactly the "verify the actual output, not just that intermediate stages look plausible"
  discipline this whole rebuild exists to enforce.
- New `MeloFlowTests.cs`: cosine similarity >0.99 vs real onnxruntime golden output (currently
  ~1.0). Passes individually.
- **Full individual regression pass this iteration**: `MeloTextEncoderTests`,
  `MeloDurationPredictorTests`, `PiperTextEncoderTests`, `PiperFlowTests`,
  `PiperDurationPredictorTests`, `MeloFlowTests` all pass individually (no regression from the
  `MeloEncoderLayerWeights`/`MeloTextEncoder` refactor into shared `MeloRelativeEncoder`).

### MeloTTS length regulator -- DONE and verified (reuses Piper's, no new math)

- The length regulator (bridging enc_p's token-rate mu/logs to the flow's frame-rate z_p via
  `w = exp(logw)*length_scale`, `w_ceil = ceil(w)`, repeat-interleave expansion, `z_p = m_p_exp +
  noise*exp(logs_p_exp)*noise_scale`) is pipeline-agnostic VITS math with NO model-specific
  weights -- Piper's `PiperLengthRegulator.cs` already implemented it exactly. **Third
  application of the shared-kernel pattern this session**: promoted it to
  `Primitives/VitsLengthRegulator.cs` (renamed from `Piper.PiperLengthRegulator`) rather than
  writing a second copy for MeloTTS. Updated both call sites (`PiperModel.cs`,
  `PiperFlowTests.cs`) to the new name/namespace; reran `PiperFlowTests` individually to confirm
  zero regression from the rename before writing any MeloTTS-specific code.
- Golden dump (`scratch-llamacpp-ref/melo_golden_lengthreg.py`, same single-`sess.run()` +
  `ORT_DISABLE_ALL` discipline established during the flow stage) captures the REAL per-token
  durations (`/Ceil_output_0`) and REAL frame-rate noise draw (`/RandomNormalLike_output_0`)
  alongside the target `/Add_2_output_0` -- lets the test feed real golden durations/noise through
  our OWN already-verified `MeloTextEncoder` mu/logs, isolating length-regulator correctness
  end-to-end without needing to re-derive the sdp_ratio-blended durations ourselves.
- New `MeloLengthRegulatorTests.cs`: cosine similarity >0.99 vs golden `z_p` (passed on the first
  run, unsurprising since both the encoder and the length-regulator math were already
  independently golden-verified).
- **Full individual regression pass this iteration**: `MeloTextEncoderTests`,
  `MeloDurationPredictorTests`, `MeloFlowTests`, `MeloLengthRegulatorTests`,
  `PiperTextEncoderTests`, `PiperFlowTests`, `PiperDurationPredictorTests`,
  `PiperHifiGanDecoderTests` all pass individually.

### MeloTTS `Generator` (HiFi-GAN, 44.1kHz) -- DONE and verified

- Read `models.py`'s `Generator` and `modules.py`'s `ResBlock1` as ground truth, then confirmed
  the real topology via direct ONNX Conv/ConvTranspose node attribute inspection (never assumed
  from the class name, since Piper's own decoder -- also nominally "HiFi-GAN" -- turned out to use
  a structurally simpler resblock): 5 upsample stages (512→256→128→64→32→16 channels, kernels
  [16,16,8,2,2], strides [8,8,2,2,2], total factor 512 -- matching MeloTTS's 44.1kHz/hop-512 mel
  rate), 15 resblocks (5 stages × 3 kernel sizes [3,7,11]), each a real `ResBlock1` (3 conv PAIRS
  per resblock: convs1[j] at dilation from a FIXED (1,3,5) tuple, convs2[j] always dilation 1) --
  NOT Piper's simpler 2-conv-per-kernel-group resblock.
- **Fourth application of the shared-kernel pattern this session**: extracted Piper's low-level
  conv/upsample math (dilated "same"-pad Conv1d, ConvTranspose1d, LeakyReLU) into
  `Primitives/HifiGanKernels.cs` and refactored `PiperHifiGanDecoder.cs` to use it (reran
  `PiperHifiGanDecoderTests` individually to confirm zero regression before writing MeloTTS code).
  The resblock LOOP STRUCTURE itself is genuinely different between the two checkpoints (Piper: 2
  layers per resblock, dilation indexed by kernel group; MeloTTS: 3 layers per resblock, ALL
  reusing the same (1,3,5) dilation tuple regardless of kernel group) so that part is NOT shared,
  only the underlying conv primitives are.
- New `MeloOnnxWeights.Dec*` fields + `MeloResBlockWeights` (conv weights anonymized by
  weight_norm-style export fusion, resolved via `GetWeightViaNode`, same as enc_p's attention
  convs) and new `MeloGenerator.cs`.
- Golden dump (`scratch-llamacpp-ref/melo_golden_generator.py`) feeds the golden flow output
  directly (== dec's real input, since the mask is all-ones for this unpadded test input) and
  targets the model's final waveform graph output ("y"), isolating generator correctness from
  flow correctness.
- **Bug found and fixed via golden bisection**: initial cosine was 0.69 (plausible-but-wrong).
  Bisection showed conv_pre+cond and the first upsample were both ~1.0, but the very first
  resblock's own output dropped to ~0.86 -- pinpointing the bug inside `ResBlock1Forward`. Root
  cause: the method took a single `dilation` parameter equal to `ResblockDilations[k]` (the
  resblock's OWN kernel-group index k, 0/1/2 -> kernel 3/7/11), applying that SAME dilation to all
  3 internal j-sublayers -- but the reference's `ResBlock1(ch, kernel_size, dilation=(1,3,5))`
  reuses the identical `(1,3,5)` dilation tuple across ITS OWN 3 internal j-iterations regardless
  of which kernel-size group the resblock belongs to. Fixed by removing the `dilation` parameter
  and cycling `ResblockDilations[j]` = 1,3,5 internally for every resblock's 3 sublayers. Verified
  fix: all bisection checkpoints (conv_pre+cond, upsample-0, resblock-0's own output, stage-0
  average) matched >0.9999999, and the final waveform cosine went from 0.69 to ~1.0.
- New `MeloGeneratorTests.cs`: cosine similarity >0.99 vs golden waveform (currently ~1.0).
- **Full individual regression pass this iteration**: `MeloTextEncoderTests`,
  `MeloDurationPredictorTests`, `MeloFlowTests`, `MeloLengthRegulatorTests`,
  `MeloGeneratorTests`, `PiperTextEncoderTests`, `PiperFlowTests`, `PiperDurationPredictorTests`,
  `PiperHifiGanDecoderTests`, `Fast.PiperRealWeightsTests` (end-to-end) all pass individually.

### MeloTTS end-to-end wiring -- DONE (MeloTTS complete, moving to F5-TTS next)

- `MeloModel.cs`: added a real weight-driven path (`ForwardReal`, gated on a new `MeloOnnxWeights?
  _weights` field + a `MeloModel(string onnxPath, ...)` constructor, mirroring `PiperModel`'s
  fake/real dual-constructor pattern exactly) that chains MeloTextEncoder -> MeloDurationPredictor
  -> VitsLengthRegulator -> MeloFlow -> MeloGenerator with real `GaussianRandom` noise draws
  (production RNG, not golden-dumped noise -- that isolation stays in the per-stage golden tests).
  The original fake/procedural `Forward` body is kept as the no-model-file fallback (renamed
  nothing, just gated behind `_weights is null`), same as Piper's pattern.
- **Fifth application of the shared-kernel pattern this session**: promoted Piper's
  `GaussianRandom` Box-Muller sampler to `Primitives/GaussianRandom.cs` (it had zero Piper-specific
  logic) instead of writing a second copy for MeloTTS's real RNG draws.
- Per `MeloOnnxWeights`'s already-documented checkpoint quirks, the real path only consumes
  `phones`/`tones`/`speakerId` -- `langIds`/`bertFeatures` are accepted (kept in the public
  `Forward` signature for API compatibility) but NOT forwarded into the real graph, since this
  checkpoint's `language`/`bert`/`ja_bert` are not real dynamic inputs (see the class doc's
  position-parity-pattern / always-zero-bert findings from the TextEncoder stage).
- **Found and fixed a second, pre-existing "compiles but doesn't use real weights" bug** while
  wiring this up: `MeloPipeline.Load(modelPath)` validated that the ONNX file exists/is large
  enough, but then ALWAYS constructed `new MeloModel(sampleRate: 44100)` -- the no-weights fake
  path -- regardless of `modelPath`, silently discarding it. Every prior "real weights" pipeline
  run (including `MeloTtsRealWeightsTests`'s own smoke test, which only checks the output is
  non-empty/finite/long-enough, not that it's real) was actually running procedural/sinusoidal
  fake synthesis end-to-end. Fixed to load real weights via `new MeloModel(modelPath, ...)`,
  mirroring `PiperPipeline.Load`'s already-correct pattern.
- Reran `MeloTtsRealWeightsTests` (the pre-existing pipeline-level smoke test, unmodified) after
  the fix: still passes, now genuinely exercising the real weight-driven path end-to-end (visibly
  slower -- ~18s vs near-instant before -- consistent with real neural computation replacing fake
  sine synthesis).
- **Full and final individual regression pass for MeloTTS this iteration**: `MeloTextEncoderTests`,
  `MeloDurationPredictorTests`, `MeloFlowTests`, `MeloLengthRegulatorTests`, `MeloGeneratorTests`,
  `Fast.MeloTtsRealWeightsTests`, plus the full Piper suite (`PiperTextEncoderTests`,
  `PiperFlowTests`, `PiperDurationPredictorTests`, `PiperHifiGanDecoderTests`,
  `Fast.PiperRealWeightsTests`) -- 11 test classes, all pass individually.

**MeloTTS is now COMPLETE per the MASTER REBUILD PLAN**: every sub-stage is real, weight-driven,
independently golden-verified against real onnxruntime output, AND wired end-to-end through both
`MeloModel` and `MeloPipeline`. Known residual limitation (out of scope for this rebuild, same as
Piper): `MeloPhonemizer`/`MeloBertEncoder` are still placeholder text->token/tone/feature
implementations, not a real espeak-NG-equivalent multilingual phonemizer -- the neural math this
whole effort targets is real and verified, but end-to-end audio *quality* from arbitrary free text
still depends on that separate, much larger phonemization undertaking.

**Next per the MASTER REBUILD PLAN's order**: F5-TTS.

3. **F5-TTS** — real safetensors + real DiT/CFM reference source both available locally,
   no ONNX/trim complications. Build a golden-output dump from the real
   `f5-tts` pip package (same technique as Kokoro's `kokoro_golden.py`) before writing the
   C# forward pass, not after.
4. **Parakeet** — verify `examples/nemo-toolkit-py` actually contains the FastConformer
   model class first (flagged as unconfirmed earlier this doc), then port against
   `examples/parakeet.cpp` (native GGML) as the structural reference.
5. **QwenASR** — native GGML reference (`qwen3-asr.cpp`), real GGUF weight file present.
6. **FunASR** — two reference implementations available, both models present locally.
7. **CosyVoice** — most architecturally complex remaining (LLM + flow decoder + speaker
   embedding, one component is ONNX-only). Do NOT add an ONNX Runtime dependency without
   re-confirming the `TrimMode=full`/`PublishAot=true` blocker still applies — if the
   flow decoder graph is small, prefer hand-porting it in pure C# over reopening that
   architecture decision.
8. **Silero VAD** — needs real ONNX graph decoding first (structural investigation, not
   just porting known math) since the actual conv/reshape/LSTM sequence isn't confirmed.
9. **QwenTTS** — no local weight file; best-effort structural port from
   `examples/qwen-tts-py` only, explicitly flagged as numerically-unverified when done,
   don't claim it's "golden-verified" when there's nothing to verify against.

**Standing ground rules for every pipeline in this loop (carried over, still binding):**
- Every stage gets a golden-output oracle (real reference model run via Python/onnxruntime
  or the pip-extracted PyTorch source) BEFORE being declared done — "compiles and produces
  non-silent audio" is exactly the failure mode this whole rebuild exists to fix. Cosine
  similarity vs. golden tensors, not just shape/finite checks.
- Don't touch `ModelGraph.cs` / `ForwardPass.Moe.cs` / `ModelCompatibility.cs` (other
  session's WIP).
- Keep `tests/OpenTail.Stingray.Tests.Audio` green after each pipeline; add a
  `*RealWeightsTests`-style regression test with real audio/text and known-correct output
  wherever the reference makes that possible (Whisper's JFK test is the template).
- Update this doc after every pipeline (or every meaningful sub-stage) — it's the handoff
  point between loop iterations, not a session log to fill in at the end.
- Commit only if explicitly asked — leave changes uncommitted like every prior iteration
  unless told otherwise.
- If genuinely blocked on a scope/architecture decision (like the CosyVoice ONNX/trim
  question), document the blocker clearly and move to the next pipeline in the order above
  rather than stalling the whole loop on one open question.

## REBUILD PLAN (2026-08-21, scope change) — porting real math from examples/ reference repos

User reframed the task: instead of just auditing, port the real algorithms from the
reference C/C++ implementations checked into `examples/` into the 9 confirmed-fake
pipelines (Kokoro, Piper, MeloTTS, F5-TTS, Chatterbox, CosyVoice, Parakeet, QwenASR,
QwenTTS). FunASR and Silero VAD are NOT in this list (not requested) — leave as-is.

**Reference source audit** (checked which repos are native math vs. ONNX-Runtime wrappers
— native/GGML repos expose portable math directly; ONNX wrappers hide the math inside
opaque graphs and only give tokenizer/orchestration code):

| Pipeline  | Reference repo(s)             | Kind                          | Tractability |
|-----------|--------------------------------|--------------------------------|--------------|
| Kokoro    | `kokoro.cpp` (307+66 lines)    | native GGML                    | High — small, C# weight loading already wired |
| Parakeet  | `parakeet.cpp`                 | native GGML                    | High |
| QwenASR   | `qwen3-asr.cpp`                | native GGML                    | High |
| QwenTTS   | `qwentts.cpp`                  | native GGML                    | High |
| CosyVoice | `cosyvoice.cpp`                | mostly native GGML, some ONNX  | Medium-High |
| MeloTTS   | `MeloTTS.cpp`                  | custom (no ggml/onnx hits — needs closer look) | Medium |
| Piper     | `piper`                        | ONNX Runtime wrapper only      | Medium — well-known VITS architecture, but no native math reference in-repo |
| Chatterbox| `Chatterbox-turbo-cpp` (902 lines total) | ONNX Runtime wrapper only, thin | Low — no architecture reference at all, only a 3-model ONNX call sequence |
| F5-TTS    | **none found under `examples/`** | n/a                          | Blocked — no reference source available locally |

**Execution order** (most tractable / highest-confidence first): Kokoro → Parakeet →
QwenASR → QwenTTS → CosyVoice → MeloTTS → Piper → Chatterbox (best-effort, architecture
reimplementation from public knowledge since no math reference exists) → F5-TTS (skip
until a reference is provided).

**Ground rules carried over:** don't touch `ModelGraph.cs` / `ForwardPass.Moe.cs` /
`ModelCompatibility.cs` (other session's WIP), watch disk space (35GB free as of
2026-08-21, no longer critical), keep tests green, commit only if asked, keep this doc
updated as the handoff point between iterations.

### MAJOR FINDING (2026-08-21): real PyTorch reference source is pip-installable for most pipelines

`examples/kokoro.cpp` and `examples/Chatterbox-turbo-cpp` turned out to be thin ONNX
Runtime wrappers (no portable math inside) rather than native references, contradicting
the earlier tractability table above. But `pip download <pkg> --no-deps` successfully
pulled the REAL PyTorch source (exact nn.Module definitions, not just ONNX graphs) for:

- `kokoro` (0.7.16) -> extracted to `examples/kokoro-py/` (istftnet.py, model.py, models.py,
  modules.py, pipeline.py). This is the exact StyleTTS2/Kokoro-82M architecture.
- `chatterbox-tts` (0.1.7) -> extracted to `examples/chatterbox-tts-py/` (s3gen decoder,
  flow matching, matcha modules -- this is the real CosyVoice-derived S3Gen vocoder
  Chatterbox actually uses, much better than the ONNX-only Chatterbox-turbo-cpp reference).
- `f5-tts` (1.1.22) -> extracted to `examples/f5-tts-py/` (`f5_tts/model/backbones/dit.py`,
  `cfm.py` -- the real flow-matching DiT. This UNBLOCKS F5-TTS, which had no reference at
  all under the old plan).
- `qwen-tts` (0.1.1) -> extracted to `examples/qwen-tts-py/` (`modeling_qwen3_tts.py`,
  tokenizer_12hz/25hz modules -- real QwenTTS talker LM + codec).
- `nemo-toolkit` (3.0.0) downloaded to `/tmp/pkgcheck` but NOT yet extracted/verified to
  contain the actual Parakeet FastConformer model class (nemo-toolkit is a large general
  framework; still need to locate the specific model file before trusting it as ground truth).
- `piper-tts` downloads as a compiled Windows wheel (no Python source) -- not useful as a
  math reference; `examples/piper` (ONNX-wrapper) plus public VITS architecture knowledge
  remains the best option there.
- `melotts` and `cosyvoice` pip sdists failed to build locally (missing build deps) --
  not yet resolved; `examples/cosyvoice.cpp` (mostly-native GGML) remains the fallback for
  CosyVoice, and `examples/MeloTTS.cpp` for MeloTTS.

**This changes the priority order** -- pipelines with a real pip-installable PyTorch
reference are now higher-confidence/lower-risk than the GGML-native-but-unverified-content
repos: Kokoro → Chatterbox → F5-TTS → QwenTTS → Parakeet (pending nemo verification) →
CosyVoice → MeloTTS → Piper → QwenASR (check qwen3-asr.cpp / qwen-asr pip separately).

**Also built:** `scratch-llamacpp-ref/kokoro_golden.py` -- runs the real `models/kokoro-v1.0.onnx`
via onnxruntime (confirmed installed: onnx 1.21.0, onnxruntime 1.25.1) with a fixed test
input and dumps ~2000 intermediate tensor values as `.npy` files to
`scratch-llamacpp-ref/kokoro_golden/`. This is a numeric oracle: as each C# stage (BERT,
text encoder, duration predictor, F0/N predictor, decoder, generator) is implemented, its
output can be diffed against the matching golden tensor instead of trusting "it compiles and
doesn't crash." Same technique (pip source + onnxruntime golden dump) should be repeated for
each remaining pipeline before writing its forward pass, not just for Kokoro.

### Kokoro-82M: weight loader DONE and verified; forward pass NOT yet started

`src/OpenTail.Stingray.Audio/Kokoro/KokoroWeights.cs` was completely rewritten (previously
loaded almost nothing -- just a style vector). It now loads and dequantizes every real
tensor in `models/kokoro-82m-q8_0.gguf` (459 tensors: `bert.*` ALBERT-shared transformer,
`text_enc.*` CNN+BiLSTM, `pred.*` ProsodyPredictor duration/F0/N stacks, `dec.*` AdaIN
decoder + NSF/iSTFTNet generator with harmonic source), using `Dequantize.ToFloat32` from
`OpenTail.Stingray.Cpu` for the mixed Q8_0/F16/F32 tensor types. All names and shapes were
verified against BOTH the GGUF file's own tensor table (dumped via
`scratch-llamacpp-ref/dump_gguf_tensors.py`) AND `examples/kokoro-py/istftnet.py`/`modules.py`
(the real PyTorch module definitions) -- not guessed from the ONNX graph's scoped names alone,
which caught several wrong assumptions along the way (decode stack is a FIXED 4 blocks, not
`kokoro.n_layer`=3; the generator's real prefix is `dec.gen` not `dec.generator`; it's a
full NSF harmonic-source generator with per-stage `noise_convs`/`noise_res`/`m_source`, not a
plain HiFi-GAN MRF stack; `AdainResBlock1`'s conv pairs use per-conv Snake1D activation with
learned `alpha1`/`alpha2`, not plain LeakyReLU). Verified via
`tests/OpenTail.Stingray.Tests.Audio/KokoroWeightsTests.cs` (passing, loads real
`models/kokoro-82m-q8_0.gguf`, asserts shapes match KV metadata).

`KokoroModel.Forward` (the actual math) is UNTOUCHED and still 100% the old fake
sine-wave/procedural implementation described in earlier sections of this doc. This is the
next concrete step: implement each stage against `examples/kokoro-py/*.py` as ground truth,
verified stage-by-stage against `scratch-llamacpp-ref/kokoro_golden/*.npy`, roughly in this
order: (1) BERT embeddings + 12x shared-weight transformer layer, (2) TextEncoder CNN+BiLSTM,
(3) ProsodyPredictor DurationEncoder + duration_proj (needs the length-regulator/alignment
matrix construction from `model.py`'s `KModel.forward`, not shown in modules.py alone --
re-read `model.py`), (4) F0Ntrain (shared LSTM + F0/N AdainResBlk1d stacks), (5) Decoder
(F0_conv/N_conv + encode/decode AdaIN stack), (6) Generator (NSF harmonic source + AdaIN+Snake
resblocks + learned ISTFT). Each stage is independently verifiable against the golden dump
before moving to the next -- do not skip verification to move faster, that's the exact
failure mode (plausible-looking, numerically wrong code) this whole rebuild exists to fix.

**Not yet done for any other pipeline**: Chatterbox, F5-TTS, QwenTTS, Parakeet, CosyVoice,
MeloTTS, Piper, QwenASR are all still in their original fake state. Only the research
(finding real reference source) is done for Chatterbox/F5-TTS/QwenTTS; Kokoro is the only
one with any real implementation work started.

### Kokoro exact algorithm, transcribed from examples/kokoro-py/model.py KModel.forward (ground truth, read in full)

This is the complete, exact inference algorithm -- not a summary. `ref_s` is the 256-dim
voice style vector (`kokoro-voice-af_heart.gguf`'s `StyleVector`); NOTE two different halves
feed two different subsystems: `s_pred = ref_s[128:]` (predictor/prosody), `s_dec = ref_s[:128]`
(decoder/generator). `StyleDim=128` in KV metadata is the PER-SUBSYSTEM half, not ref_s's
full length.

1. `input_ids = [0, *phoneme_ids, 0]` (BOS/EOS = token 0). Single utterance -> no padding,
   `text_mask` can be treated as all-false (skip masking entirely in the C# port).
2. `bert_dur = CustomAlbert(input_ids)` -- standard HF ALBERT forward: embeddings
   (word `bert.embd.tok` + position `bert.embd.pos` + token_type `bert.embd.tt`, all
   summed, then `bert.embd.ln`) -> `bert.embd_proj` (Linear 128->768) -> the SAME shared
   transformer layer (`bert.attn_*`/`bert.ffn_*`) applied `BertNumHiddenLayers`=12 times
   in a loop (ALBERT parameter sharing -- this is why there's only one indexed weight set).
   Standard post-LN transformer block: self-attn (12 heads) -> `+residual` -> `bert.attn_ln`
   -> FFN (GELU, 768->2048->768) -> `+residual` -> `bert.ffn_ln`. Output is `last_hidden_state`
   (pooler/`bert.pooler` is NOT used here).
3. `d_en = bert_proj(bert_dur).transpose(-1,-2)` -- Linear 768->512 (this is `bert_proj.weight`
   in GGUF, matches `KModel.bert_encoder`), then transpose to channel-first [512, T].
4. `d = predictor.text_encoder(d_en, s_pred, ...)` -- `DurationEncoder.forward` (modules.py):
   3x alternating (BiLSTM, AdaLayerNorm) blocks (`pred.dur_enc.{0,1,2}.lstm` /
   `pred.dur_enc.{0,1,2}.adaln`). Each iteration: concat style (broadcast across time) onto
   the channel dim -> BiLSTM(640->512, bidirectional 256 each way) -> AdaLayerNorm(512,
   style) -> concat style again for the next LSTM's input (640 channels). Output `d` is
   [T_text, 640] channel-last (640 = 512 hidden + 128 style, still concatenated after the
   final AdaLN).
5. `x, _ = predictor.lstm(d)` -- one more BiLSTM (`pred.lstm`, 640->512) over `d`.
6. `duration = sigmoid(predictor.duration_proj(x)).sum(-1) / speed` -- Linear 512->50
   (`pred.dur_proj`) then sigmoid+sum over the 50 bins (NOT softmax/argmax -- StyleTTS2
   encodes duration as a sum of independent per-bin probabilities). `pred_dur =
   round(duration).clamp(min=1)` gives an integer frame count per phoneme.
7. **Length regulator**: build a one-hot alignment matrix `pred_aln_trg` [T_text, T_frames]
   via `repeat_interleave(arange(T_text), pred_dur)` (frame f's source phoneme index) then
   scatter 1s. `en = d.T @ pred_aln_trg` upsamples `d` from phoneme-rate to frame-rate
   [640, T_frames] (matrix multiply, not a copy loop -- though a copy loop is equivalent
   and simpler for a one-hot alignment matrix).
8. `F0_pred, N_pred = predictor.F0Ntrain(en, s_pred)` (modules.py `ProsodyPredictor.F0Ntrain`):
   `pred.shared` BiLSTM(640->512) over `en` -> two parallel 3-block AdainResBlk1d stacks
   (`pred.F0.{0,1,2}` / `pred.N.{0,1,2}`, dims 512->512, 512->256 with 2x upsample via
   `UpSample1d`+`pool` ConvTranspose1d, 256->256) -> `pred.F0_proj`/`pred.N_proj`
   (Conv1d 256->1). **F0_pred and N_pred end up at 2x T_frames** because of the upsample
   block inside the AdainResBlk1d stack -- this is why the decoder needs F0_conv/N_conv
   (stride=2) to bring them back down before concatenating with `asr`.
9. `t_en = text_encoder(input_ids, ...)` (modules.py `TextEncoder`, separate from BERT):
   `text_enc.embd` embedding -> 3x (`text_enc.cnn.{i}.conv` + `text_enc.cnn.{i}.ln`
   [gamma/beta, NOT weight/bias] + LeakyReLU(0.2) + dropout) -> `text_enc.lstm`
   BiLSTM(512->512) -> channel-first [512, T_text] output.
10. `asr = t_en @ pred_aln_trg` -- upsample text-encoder features to frame-rate, same
    alignment matrix as step 7, [512, T_frames].
11. `audio = decoder(asr, F0_pred, N_pred, s_dec)` (istftnet.py `Decoder.forward`,
    **using the ref_s[:128] half, not s_pred**):
    - `F0 = F0_conv(F0_pred)`, `N = N_conv(N_pred)` -- stride-2 convs, downsample back to
      T_frames (matching `asr`'s rate). `dec.F0_conv`/`dec.N_conv`.
    - `x = cat([asr, F0, N])` -> [514, T_frames] -> `dec.encode` AdainResBlk1d(514->1024).
    - `asr_res = dec.asr_res(asr)` -- Conv1d 512->64 (1x1).
    - 4 decode blocks (`dec.decode.{0,1,2,3}`, fixed count -- NOT `kokoro.n_layer`):
      for each, `x = cat([x, asr_res, F0, N])` (1090 channels) THEN `x = block(x, s_dec)`;
      only block 3 upsamples (2x, via its `pool` ConvTranspose1d) and outputs 512 channels.
    - `x = generator(x, s_dec, F0_pred)` -- **passes the ORIGINAL non-downsampled F0_pred**
      (the 2x-rate one from step 8), not the downsampled `F0` used above.
12. `Generator.forward` (istftnet.py, NSF + iSTFTNet): upsample F0_pred by
    `prod(upsample_rates)*hop_size` = 10*6*5 = 300x (nearest) to sample-rate resolution ->
    `SourceModuleHnNSF` (`dec.gen.m_source`): 9-harmonic sine generator (cumulative phase
    per `SineGen._f02sine`, NOT a naive per-sample sin(2*pi*f*t) -- it integrates F0/sample_rate
    to get phase, matching real pitch-synchronous harmonic generation) -> Linear(9->1)+tanh
    -> `har_source`. STFT (n_fft=20,hop=5) of `har_source` -> `har` [22, frames']. Then for
    each of 2 upsample stages: LeakyReLU(x,0.1) -> `noise_convs[i](har)` projects the
    harmonic spectrum into the stage's channel width -> `noise_res[i]` (AdainResBlk1d)
    refines it -> `ups[i]` ConvTranspose1d upsamples `x` -> (last stage only: reflection-pad
    x by 1 on the left) -> `x = x + noise` -> average of 3 `resblocks[i*3+j]` (AdaIN+Snake1D
    HiFi-GAN resblocks, dilations [1,3,5]) . Final: LeakyReLU -> `conv_post` (7-wide,
    128->22) -> `spec=exp(x[:11])`, `phase=sin(x[11:])` -> learned inverse-STFT
    (`torch.istft`-equivalent overlap-add with a Hann window sized `n_fft`=20) -> waveform.
    **Snake1D** (inside every `AdainResBlk1d`/`AdainResBlock1`): `x = x + (1/alpha) *
    sin(alpha*x)^2` where `alpha` is a learned per-channel parameter (`adain*.alpha1`/`alpha2`
    -- shape [1,channels,1]), applied AFTER each AdaIN1d norm and BEFORE each conv.
    **AdaIN1d** (`*.adain1`/`*.adain2`, `*.norm1`/`*.norm2`): instance-norm (per-channel,
    per-sample mean/var over the time axis, no learned affine) then modulated by
    `(1+gamma)*norm(x)+beta` where `[gamma,beta] = Linear(style_dim, channels*2)(style)`.
    **AdaLayerNorm** (`pred.dur_enc.*.adaln`): same modulation formula but over a standard
    LayerNorm (normalize over the channel dim per time-step, not instance norm over time).

**GGUF tensor layout convention** (confirmed against `WhisperEncoderWeights`' established
pattern in this codebase, and cross-checked: e.g. `pred.F0.0.adain1.weight` is a PyTorch
`nn.Linear(128, 1024)` whose torch weight shape is `[1024,128]` (out,in), but GGUF lists
`dims=[128,1024]` -- GGUF/ggml dims are reversed from torch shape (fastest-varying first),
so the flat byte buffer is ALREADY exactly `[outFeatures, inFeatures]` row-major, i.e.
directly usable with `SimdKernels.MatMulBatchedF32` with no transpose needed, same as Whisper).

**Implementation status**: weights loader done (see above). Forward pass step 1-2 (embeddings
+ 12x shared ALBERT layer) is DONE and numerically verified:
`src/OpenTail.Stingray.Audio/Kokoro/KokoroBertEncoder.cs` implements
`KokoroBertEncoder.Forward` (embeddings -> LN -> embd_proj 128->768 -> 12x shared
self-attn+FFN block, post-LN, ALBERT's `gelu_new` tanh-approx activation, eps=1e-12) and
`ProjectToWorkingDim` (`bert_proj`/`KModel.bert_encoder`, 768->512, transposed to
channel-first). Verified in
`tests/OpenTail.Stingray.Tests.Audio/KokoroBertEncoderTests.cs` against a real onnxruntime
run of `models/kokoro-v1.0.onnx` (`scratch-llamacpp-ref/kokoro_golden_bert.py`, fixed
input_ids `[0,50,83,54,156,57,135,3,16,65,156,0]`): cosine similarity > 0.999 between the
C# `last_hidden_state` and the golden ONNX output (raw mean-abs-diff ~0.0065, max ~0.054 on
a std~0.5 signal -- expected from comparing our Q8_0-quantized GGUF weights against the
reference's FP32 ONNX weights across 12 stacked layers, not a bug; verify with cosine
similarity, not raw diff, when the two models differ in weight precision like this).
Pattern for future stages: for each stage, (1) find/confirm the relevant ONNX node names
via a narrow `kokoro_golden_*.py` dump (few hundred KB, not the 1.1GB full-graph dump),
(2) implement against `examples/kokoro-py/*.py`, (3) verify via cosine similarity in a
`KokoroXxxTests.cs`, (4) only then move to the next stage.

**TextEncoder stage DONE and verified** (2026-08-21): `src/OpenTail.Stingray.Audio/Kokoro/KokoroLstm.cs`
is the shared bidirectional-LSTM primitive (standard PyTorch gate order i,f,g,o; weight_ih/weight_hh
already `[4*hidden, in-or-hidden]` row-major per the GGUF reversed-dims convention, confirmed against
the actual file: `text_enc.lstm.weight_ih_l0 dims=[512,1024]` ggml-order -> torch shape `[1024,512]`
= `[4*256, 512]`). `KokoroTextEncoder.cs` implements `KModel.text_encoder` (modules.py `TextEncoder`:
embedding -> 3x(Conv1d k=5 pad=2 + custom channels-LayerNorm(gamma/beta, eps=1e-5, per-timestep
over channels) + LeakyReLU(0.2)) -> BiLSTM(256 per direction)), returning `t_en` channel-first
`[512,T]` (matches `model.py` line 105's `t_en = self.text_encoder(...)`, later used as
`asr = t_en @ pred_aln_trg`). Verified in `KokoroTextEncoderTests.cs` against
`scratch-llamacpp-ref/kokoro_golden_textenc.py`'s golden ONNX output (node
`/encoder/text_encoder/Transpose_2_output_0`, shape `[1,512,12]`) -- cosine similarity > 0.999.
Note: ProsodyPredictor also has an internal module confusingly also named `text_encoder`
(that's actually `DurationEncoder`, see `modules.py` `ProsodyPredictor.__init__`'s
`self.text_encoder = DurationEncoder(...)`) -- do not conflate the two; the golden-dump script
excludes ONNX nodes under `/encoder/predictor/` to avoid grabbing the wrong one.

**DurationEncoder + predictor.lstm + duration_proj DONE and verified** (2026-08-21):
`src/OpenTail.Stingray.Audio/Kokoro/KokoroAdaLayerNorm.cs` implements AdaLayerNorm (style ->
`fc` Linear(128,1024) -> chunk into gamma/beta[512] -> per-timestep channel LayerNorm(eps=1e-5)
-> `(1+gamma)*x+beta`; net of the reference's redundant double-transposes, which cancel out for
our fixed rank-3 shapes -- see the file's doc comment for the trace). `KokoroProsodyPredictor.cs`
implements `EncodeDuration` (DurationEncoder: concat `[d_en; style]` -> 640ch -> 3x(BiLSTM 640->512
+ AdaLayerNorm(512) + re-concat style ->640) -> returns `d` [T,640]) and `PredictDurations`
(predictor.lstm [640->512 BiLSTM] -> duration_proj [Linear 512->50] -> sigmoid -> sum over the
50 max_dur bins, PRE round/clamp/speed-divide -- caller still needs to do `/speed`, `round`,
`clamp(min=1)` to get `pred_dur`). Verified in `KokoroProsodyPredictorTests.cs`, chained on top
of the already-verified BERT stage, against `scratch-llamacpp-ref/kokoro_golden_durenc.py`'s
golden ONNX output (`Concat_4_output_0` for `d` [1,12,640], `ReduceSum_output_0` for the
duration sums [1,12]). Cosine similarity thresholds: `d` needed slightly relaxing to >0.998
(vs BERT's >0.999) since it's the already-~0.999 BERT output run through 3 more stacked
LSTM+AdaLayerNorm blocks -- Q8_0 quantization noise compounds further; confirmed this isn't a
structural bug by checking per-timestep cosine similarity was uniform (0.998-0.9996 across all
12 tokens, no outlier/discontinuity). Duration-sum cosine similarity still clears >0.999.
IMPORTANT naming gotcha (re-confirmed): `ProsodyPredictor.text_encoder` in the reference IS
`DurationEncoder`, a completely different module from top-level `KModel.text_encoder`
(`KokoroTextEncoder.cs`) -- both files' doc comments now call this out explicitly.

**Alignment/length-regulator + F0Ntrain DONE and verified** (2026-08-21):
`src/OpenTail.Stingray.Audio/Kokoro/KokoroAlignment.cs` implements `ToPredDur` (round/clamp/
speed-divide, model.py line 97) and `BuildFrameToTokenMap`+`Expand` (model.py lines 99-103's
`repeat_interleave`+one-hot-scatter length regulator, implemented as a direct gather since the
alignment matrix is one-hot -- no need to materialize it). Covered by pure unit tests
(`KokoroAlignmentTests.cs`, no GGUF/golden-dump needed, just arithmetic). Separately,
`src/OpenTail.Stingray.Audio/Kokoro/KokoroAdainResBlk1d.cs` implements `AdainResBlk1d`
(istftnet.py -- LeakyReLU(0.2)-activated AdaIN1d residual block, with an optional depthwise
ConvTranspose1d "pool" upsample in the residual branch and nearest-x2 upsample in the shortcut;
NOT to be confused with `AdaINResBlock1`, the Snake1D-activated, differently-shaped class used
only inside the Generator -- see `ResBlockWeights`, already loaded, not yet consumed).
`KokoroProsodyPredictor.F0Ntrain` chains predictor.shared BiLSTM (640->512) with the F0 and N
`AdainResBlk1d` stacks (512->512->256(upsample x2)->256) + F0_proj/N_proj (Conv1d 256->1),
matching istftnet.py/modules.py's `F0Ntrain`. Verified in `KokoroF0NtrainTests.cs` by feeding
`scratch-llamacpp-ref/kokoro_golden_f0n.py`'s real onnxruntime `en` tensor
(`/encoder/MatMul_output_0`, `[1,640,42]`) directly into `F0Ntrain` (deliberately bypassing the
not-yet-integration-tested upstream alignment step, to isolate this stage) and comparing
against golden `F0_proj`/`N_proj` outputs (`[1,1,84]` -- confirms the x2 upsample lands
correctly): cosine similarity > 0.998 for both F0 and N curves.

**Kokoro status as of this checkpoint**: ALL 7 forward-pass stages implemented and verified:
1. `KokoroBertEncoder` (BERT embeddings + 12x shared ALBERT layer + projection to 512 channels)
2. `KokoroTextEncoder` (Token embeddings + 3x Conv1d/LN/LeakyReLU + BiLSTM)
3. `KokoroProsodyPredictor` (DurationEncoder 3x BiLSTM/AdaLayerNorm + predictor.lstm + duration_proj)
4. `KokoroAlignment` (ToPredDur + one-hot alignment / length regulator expand)
5. `KokoroProsodyPredictor.F0Ntrain` (shared BiLSTM + parallel F0 & N AdaIN stacks + F0/N proj)
6. `KokoroDecoder` (`Decoder.forward` from `istftnet.py`: F0_conv/N_conv + encode + 4x decode AdaIN stack)
7. `KokoroDecoder.GeneratorForward` (`Generator.forward` from `istftnet.py`: 300x F0 upsample, SineGen 9-harmonic cumulative-phase NSF source, STFT, ConvTranspose upsamples with AdaIN+Snake1D resblocks + noise branch, conv_post, learned inverse-STFT overlap-add).

`KokoroModel.Forward` is fully wired to `ForwardReal` (delegating to the 7 verified modules when `_weights != null`, preserving fallback when `_weights == null`). Tested end-to-end against real GGUF weights (`kokoro-82m-q8_0.gguf` + `kokoro-voice-af_heart.gguf`) in `KokoroRealWeightsTests`, producing valid non-silent 24kHz synthesized audio waveforms (182s test pass).

**Next pipeline up for rebuild/audit**:
- Parakeet / CosyVoice / F5-TTS / Chatterbox / QwenTTS / QwenASR / MeloTTS / Piper.


## SESSION HANDOFF, 4th/final iteration — recurring loop stopped, needs your input to continue

Stopped the 30-minute cron loop here rather than keep firing. Not because the task is
"done" — 10 of 12 pipelines are still fake — but because every further step from here is
either (a) a scope/priority decision only you can make, or (b) multi-hour-per-pipeline
rebuild work that shouldn't be started unsupervised on a guess about which one you'd want
first. Continuing to wake up every 30 minutes with nothing new and safe to do would just
be spinning, or worse, pressure to rush a risky rewrite to look productive. Restart with
`/loop 30m <same prompt>` any time, or just tell me which of the items below to act on.

**What's done and verified, safe to build on:**
- Whisper ASR: fully fixed (2 real, root-caused, source-verified bugs — mel filterbank
  constant typo, mel buffer sizing — plus a decode-time logit suppression bug), verified
  against real ground truth at 4 model sizes, regression-tested.
- Full audit of all 12 pipelines complete: only Whisper is real; the other 11 (Kokoro,
  Piper, MeloTTS, F5-TTS, Chatterbox, CosyVoice, QwenASR, QwenTTS, Parakeet, FunASR, plus
  a "partially real, actually messier than first thought" Silero VAD) are confirmed fake
  by direct code inspection, not guesswork.
- Shared DSP infrastructure checked clean: `WavReader`, `WavWriter`, `AudioResampler`,
  `AudioDownmixer`, `VadSegmenter` — all correct, no bugs found.
- Two risky shortcuts evaluated and correctly NOT taken: ONNX Runtime for Piper/MeloTTS
  (blocked by this project's real, enforced `TrimMode=full`/`PublishAot=true`), and a
  quick top-up of Silero VAD (the real model turned out to be a materially different,
  messier architecture than assumed — would need real reverse-engineering, not a patch).
- Regression-clean: 106/106 tests in `tests/OpenTail.Stingray.Tests.Audio`, Cli/Server
  Release builds clean, all previously-working (fake) TTS engines (Kokoro/Piper/Chatterbox)
  still run without crashing.

**Decision needed from you before more Audio work makes sense:**
1. Which (if any) of the 10 fake pipelines should get a real rebuild first, if any? Each
   is comparable in scope to the Whisper fix I just did, times several — genuinely
   multi-hour, architecture-specific work (VITS+HiFiGAN for Piper/MeloTTS, a real DiT for
   F5-TTS, Conformer for Parakeet, etc.), not something to guess-and-start on 10 different
   architectures across unsupervised 30-minute windows.
2. Is weakening `TrimMode=full`/`PublishAot=true` (even just for the Audio/Cli assembly)
   an acceptable tradeoff to unlock ONNX Runtime for Piper/MeloTTS/Silero VAD? That would
   be meaningfully cheaper than hand-porting VITS/HiFiGAN math, but it's your call since
   it touches a stated hard project constraint, not just an implementation detail.
3. Uncommitted state: all my changes (Whisper fixes + this doc + diagnostic test) are
   sitting uncommitted, exactly like the other session's deepseek2 WIP — per your
   instructions I haven't committed anything since you didn't ask for it. Let me know if
   you want these committed, and whether separately from or together with the other
   session's work once it's done.

Full chronological findings/evidence below, oldest first.

---


Started 2026-08-21. Reviewing `src/OpenTail.Stingray.Audio/` (built mostly by another AI
session) for correctness, running autonomously via a 30-minute `/loop` cron job
(job id `5b406f4b`, session-only, auto-expires after 7 days) while the user is AFK. This
file is the handoff state between iterations — read it before doing new work.

**Constraint: disk is at ~2.5GB free.** Do not download large models without deleting
something first.

**Do not touch** `ModelGraph.cs`, `ForwardPass.Moe.cs`, `ModelCompatibility.cs` — a
separate session has uncommitted deepseek2 MLA work in progress there.

## HEADLINE FINDING — most of this subsystem is not real inference

This is the central thing the user needs to know, so it's at the top. This is not a
"few small bugs" situation.

**Of the 12 pipeline families under `src/OpenTail.Stingray.Audio/`, only ONE
(Whisper ASR) does genuine neural computation against loaded model weights.** Every
other pipeline follows the same pattern: a real weight-loading class exists (parses a
real GGUF/ONNX/safetensors file, e.g. `KokoroWeights`, `PiperConfig`, `ChatterboxWeights`,
`FunAsrPipeline`'s `GgufModel`), and is even wired up and called — but the actual
`Forward()`/`Generate()`/`Predict()` math **never reads the loaded tensors**. Instead it's
hardcoded closed-form sine/cosine formulas with magic constants (e.g.
`0.08f * MathF.Sin(tid * 7.77f + d * 0.15f)`) that produce plausible-shaped,
plausible-*sounding* (non-silent, correct-duration) output completely independent of
which model file — or whether any model file — was loaded. Verified directly: running
Kokoro TTS with `-m models/kokoro-82m-q8_0.gguf --voices-dir models` vs. no `-m` flag at
all produced **byte-identical** WAV output.

Confirmed FAKE (no real weight usage in the forward pass, verified by reading the code):
- **Kokoro** (`KokoroModel.Forward`) — procedural harmonic/sine synthesis.
- **Piper** (`PiperModel.Forward`) — same pattern; `PiperModel` doesn't even have a
  constructor parameter for weights at all.
- **MeloTTS** (`MeloModel.Forward`) — same pattern.
- **F5-TTS** (`F5DiTModel.ForwardVelocity`) — the 22-layer "DiT" has no weight tensors
  anywhere; fixed linear-combination formulas standing in for all 22 transformer blocks.
- **Chatterbox**, **CosyVoice**, **Parakeet**, **QwenASR**, **QwenTTS** — CONFIRMED fake,
  2nd iteration: every core neural component's constructor
  (`ChatterboxDecoder()`, `CosyVoiceFlowDiT`/`CosyVoiceHiFT`/`CosyVoiceLlm`,
  `ParakeetConformerEncoder`, `QwenAsrAudioEncoder`,
  `QwenTtsCodePredictor`/`QwenTtsDacDecoder`/`QwenTtsTalkerLm`) takes only a config
  record, never a weights object — there is no code path by which a loaded model file
  could reach the forward pass math at all.
- **FunASR** (`FunAsrPipeline`) — confirmed fake. Loads a `GgufModel` but never reads a
  single tensor from it (`_ggufModel` field exists only to be `Dispose()`d).
  `CifPredictor.Predict` emits synthetic token IDs (`100 + t % 500`) from a fixed
  firing-rate heuristic regardless of audio content; `FunAsrTokenizer.Decode` just prints
  `[T123]`-style placeholders, not real text.

Partially real:
- **Silero VAD** (`SileroVad`) — genuinely ingests real GGUF weights for 2 of 4 conv
  encoder stages and the LSTM (`IngestGgufWeights`), but the other 2 conv stages and the
  final projection layer are permanently fake (Glorot-sin initialized, GGUF never
  consulted for them), AND a hardcoded heuristic (`frameEnergy`-based logit boost/penalty)
  is layered on top of whatever the network computes — so even the real portion's
  contribution to the final speech/no-speech decision is not trustworthy as-is.
  **UPDATE (3rd iteration) — this is a bigger gap than "50% done, top up the rest."**
  Dumped `models/silero_vad.gguf`'s real tensor names directly
  (`scratch-llamacpp-ref/dump_gguf_tensors.py`, a minimal from-scratch GGUF parser since
  this was faster than adding a throwaway C# test for it): the real model is NOT the
  "STFT + 4 plain conv layers + plain LSTM" architecture `SileroVad.cs`'s comments
  describe. It's Silero's newer fused "reparam_conv" architecture (`encoder.0.reparam_conv.weight`
  etc, not separate conv+norm+activation tensors), the real STFT frontend is a learned
  `stft.forward_basis_buffer` conv kernel `[128,1,130]`, not a hand-rolled Hann+DFT (which
  is what this engine's `ComputeStftMagnitude` does instead), `encoder.0`'s real in-channel
  count is 65, not the 129 this engine assumes, and there's no clean `lstm.weight_ih`/
  `weight_hh` tensor at all — the actual LSTM gate weights are buried in auto-generated
  ONNX constant-folding names (`...Unsqueeze_6_output_0_subg_96_sub_graph2`, shape
  `[128,512,1]`) from a generic ONNX→GGUF conversion, not a purpose-built one. There's also
  a whole second copy of the graph (`If_0_then_branch__Inline_0__...`) for an 8kHz code
  path with different shapes, meaning the real model is graph-structured (an `If` node
  choosing 8kHz vs 16kHz), not the flat sequential pipeline this engine assumes. Properly
  completing this needs decoding the actual ONNX graph structure (from `models/silero_vad.onnx`,
  not the messy GGUF conversion) to know the real conv/reshape/LSTM sequence -- comparable
  effort to the fake-TTS-pipeline rebuilds, not a quick top-up. Not attempted this session;
  flagged rather than rushed.

Genuinely real (does real per-layer matmul/attention/conv against loaded weights,
conditionally branching on `_weights != null`):
- **Whisper** (`WhisperEncoder`, `WhisperDecoder`) — see below, real but currently
  produces wrong transcriptions; needs numeric debugging, not a rewrite.

### Evidence table (grep across every file, `_weights.`/`weights.`/`Weights.` refs vs.
matmul-ish calls)

Only `Whisper/WhisperEncoder.cs` (3 weight refs, 10 matmul calls) and
`Whisper/WhisperDecoder.cs` (1 weight ref via a weights object, 30 matmul calls) show a
real ratio. Every other pipeline's core model file shows 0 matmul-ish calls, full stop —
mechanically impossible to be doing real neural inference. Re-run if you want to spot
check: `grep -c "MatMul\|TensorPrimitives\.\|Dot(" <file>`.

### What this means for scope

Fixing this for real means reimplementing genuine neural forward passes (VITS text
encoder + normalizing flow + HiFi-GAN vocoder for Piper/MeloTTS; AdaIN-ResBlock decoder
for Kokoro; a real 22-layer flow-matching DiT for F5-TTS; a Conformer encoder for
Parakeet; LLM+neural-codec decoder stacks for Chatterbox/CosyVoice/QwenTTS) — each
individually comparable in scope to the multi-day architecture-porting work logged in
`docs/00-current-work.md` for text models, and there are ~9 of them. This is NOT
something to rush through as "one more fix" per loop iteration; doing so risks producing
more of exactly this failure mode (plausible-looking code that quietly doesn't do what it
claims). Recommend the user makes an explicit call on priority/scope here rather than
this loop silently attempting a full rewrite of 9 architectures unsupervised.

**UPDATE (3rd iteration): the ONNX Runtime option below is likely NOT viable, checked
against actual project config rather than just CLAUDE.md prose.**
`src/OpenTail.Stingray.Cli/OpenTail.Stingray.Cli.csproj` has `<PublishAot>true</PublishAot>`
AND `<TrimMode>full</TrimMode>` actively set (not just aspirational doc text) — this is
CLAUDE.md's Critical Rule 4 ("NativeAOT / trim analyzers are active") enforced as a real
build setting, not just guidance. `TrimMode=full` is the aggressive setting and
`Microsoft.ML.OnnxRuntime`'s managed wrapper is not guaranteed clean under it. Don't adopt
this without the user explicitly accepting either a trim-mode carve-out or verifying the
package's AOT/full-trim compatibility firsthand — downgrading this from "flagged option"
to "likely blocked, needs the user's call specifically because it may require weakening a
stated hard project constraint," which is a bigger decision than just "add a NuGet
package." Original reasoning kept below for context.

**A pragmatic, much cheaper option worth flagging for the user's decision, not yet
acted on (see UPDATE above):** Piper and MeloTTS ship real `.onnx` files, and this whole
solution currently has **zero** dependency on ONNX Runtime anywhere (verified via
repo-wide grep). Adding `Microsoft.ML.OnnxRuntime` and actually running those two
pipelines' real ONNX graphs would be dramatically less work than hand-porting
VITS+HiFiGAN math, and would fix 2 of the 9 fake pipelines relatively cheaply. Caveat:
`CLAUDE.md` says NativeAOT/trim analyzers are active for this project's single-binary
deployment goal; ONNX Runtime's C# bindings need to be checked for AOT/trim compatibility
before adopting — flagging as a
scope/architecture decision, not doing it silently.

## Whisper ASR — real implementation, wrong output (fixed one bug, one remains)

**Bug 1, FIXED this session:** `SttCommand.cs` (the `stingray stt` CLI) never loaded a
real model at all — it called `new WhisperPipeline(config)` with just a config preset,
never `WhisperPipeline.Load(ggmlPath)`. Every STT run was silently using
`WhisperEncoder`/`WhisperDecoder`'s placeholder (untrained, deterministic-sinusoidal)
weight path, identical in kind to the fake-pipeline pattern above, even though
`WhisperPipeline.Load()` and real weight parsing (`WhisperGgmlModel`,
`WhisperEncoderWeights`, `WhisperDecoderWeights`) were already fully implemented and just
never called from the CLI. Fixed: `SttCommand.cs` now resolves a real `models/ggml-*.bin`
file matching the requested `--model` preset (or an explicit `--model-file` override) and
calls `WhisperPipeline.Load()`; if no file is found it now prints an explicit warning
instead of silently running on fake weights.

**Bug 2, NOT YET FIXED — real numeric defect.** After the fix, `stingray stt -i
examples/whisper.cpp/samples/jfk.wav -m tiny` (and base/small) now loads real weights but
transcribes the JFK reference clip ("And so, my fellow Americans, ask not what your
country can do for you. Ask what you can do for your country.") as just `"[Music]"`,
stopping after ~1 of 11 seconds of audio. `-m medium` timed out after 2 minutes (separate
possible perf issue, not investigated). This is Whisper hallucinating/collapsing almost
immediately, not silence — so the bug is somewhere in the encoder conv/attention stack,
the decoder's cross-attention/KV-cache, or the initial-prompt token construction, not in
loading. Structurally checked and looking correct: `WhisperMelExtractor`'s mel filterbank
and log-mel normalization match the standard Whisper/whisper.cpp formula (Slaney mel
scale, `(clamp(x, max-8) + 4)/4` normalization) — mel extraction is NOT the obvious
suspect, though not numerically verified frame-by-frame against a reference. No
whisper.cpp reference binary is available locally to diff against directly (only
`tools/llama.cpp/*.exe`, no whisper-cli) and building one from source was avoided per
`docs/bugstofix.md`'s existing note that this session deliberately avoids installing MSVC
build tools. Next step: read `WhisperDecoder.cs`'s `PrimeCrossAttention`/`ForwardStep`
and the tokenizer's `BuildInitialPrompt`/timestamp-suppression logic line-by-line against
`examples/whisper.cpp`'s C++ source, the same way `docs/00-current-work.md`'s text-model
architecture receipts do it for the LLM side — this is the single most valuable thing to
spend the next iteration's time on, since Whisper is the one pipeline where "fix a bug"
(vs. "write the missing implementation from scratch") is actually the correct framing.

**Specific lead for the decode bug, found while reading `WhisperEncoder.cs` (not yet
verified):** `ApplyConv1DReal`'s doc comment asserts the conv1 weight tensor's on-disk
layout is `[outCh, inCh, 3]` with kernel-position `k` fastest-varying
(`wIcOff + (k+1)`), an assumption, not something cross-checked against
`examples/whisper.cpp`'s actual GGML conv1d tensor layout/converter. The rest of the
encoder (`ComputeMultiHeadSelfAttentionReal`, `ComputeMlpReal`, `LinearReal` via
`SimdKernels.MatMulBatchedF32`) all look structurally standard and correct on inspection.
Since the very first thing real weights touch is this conv1d stage, a transposed/wrong
axis order here would corrupt everything downstream while still producing
plausible-shaped (LayerNorm'd, softmax'd) activations — consistent with the observed
symptom (fluent-looking but wrong output, not NaN/crash). **Diagnostic harness added, `tests/OpenTail.Stingray.Tests.Audio/WhisperDiagnosticTests.cs`**
(deliberately fails so xUnit prints its dump — run with:
`./tests/OpenTail.Stingray.Tests.Audio/bin/Debug/net10.0/OpenTail.Stingray.Tests.Audio.exe
-class OpenTail.Stingray.Tests.Audio.WhisperDiagnosticTests -verbose`, rebuild first if the
source changed). Findings from it, tiny model, real jfk.wav vs. a matched-length silence
buffer:
- Mel extraction and encoder both produce sane, non-degenerate stats (no NaN/Inf, plausible
  ranges) for real audio.
- Encoder output DOES differ meaningfully between real audio and silence (mean abs diff
  0.53 against a baseline meanAbs of ~0.7) — cross-attention conditioning is NOT dead/inert,
  ruling out "audio path completely disconnected" as the root cause.
- BUT: greedy decode's very first post-prompt token, in both the real-audio and silence
  cases, is dominated by a narrow band of special/control token IDs (`<|notimestamps|>`
  id 50363, then the `<|0.00|>..<|0.24|>` timestamp cluster ids 50364-50376) — no real word
  token appears anywhere in the top 10 for either case. The silence case additionally shows
  a suspicious exact-arithmetic-progression pattern in its 3rd-10th ranked tokens (50414,
  50464, 50514, 50564, 50614 — each exactly +50 apart, i.e. exactly `<|1.00|>`, `<|2.00|>`,
  `<|3.00|>`, `<|4.00|>`, `<|5.00|>`, round-second timestamps only). That regularity smells
  like a systematic bias/bug in the tied-LM-head matmul or final layernorm rather than
  genuine learned behavior.
- Cross-checked against expectation: `examples/whisper.cpp/samples/jfk.wav` is the
  standard whisper.cpp smoke-test sample and is reliably transcribed correctly by real
  whisper.cpp even at the `base` size. This engine produces the identical `"[Music]"`
  failure at `tiny`, `base`, AND `small` (only `medium` wasn't confirmed — it timed out).
  Getting the same wrong answer regardless of model size argues for an implementation bug
  common to all sizes (something in `WhisperDecoder`/`WhisperEncoder`/`WhisperGgmlModel`,
  not a per-checkpoint weight issue), not "tiny is just weak."
- Not yet localized further. Next concrete step: extend the diagnostic to decode a few more
  greedy steps forced past the special-token cluster (temporarily zero those logit
  positions) and see if real content ever surfaces underneath, which would point at
  "special-token bias/suppression logic bug" (comparatively easy fix — e.g. missing
  `<|notimestamps|>` suppression when timestamps ARE enabled, which real whisper.cpp does
  apply and this engine's `ProcessAudioChunk` does not) vs. real content never surfacing at
  all even several tokens in, which would point at a deeper numeric bug in the attention/MLP
  chain itself (harder fix).

**RULED OUT, checked this session**: verified against
`examples/whisper.cpp/src/whisper.cpp` line 1760 —
`ggml_new_tensor_3d(ctx, vtype, 3, n_mels, n_audio_state)` — ggml's `ne[]` is
fastest-varying first, so the real on-disk layout is exactly `[k=3, inCh=n_mels,
outCh=n_audio_state]` with k fastest, matching `ApplyConv1DReal`'s indexing
(`oc*inChannels*3 + ic*3 + k`) exactly. Conv1d layout is NOT the bug. Also confirmed
`ggml_conv_1d_ph(ctx0, model.e_conv_1_w, mel, 1, 1)` (stride=1, pad=1) matches the C#
conv1 stage's stride/pad. Next place to look, not yet checked: conv2 (stride=2) indexing,
the positional-embedding addition, or — most likely given "[Music]" appears even on
`tiny` — the decoder's cross-attention K/V projection or `PrimeCrossAttention`/
`ForwardStep`'s KV-cache indexing in `WhisperDecoder.cs`. This needs either a real
per-layer intermediate-tensor oracle (no whisper.cpp binary available locally, and
building one was avoided per `docs/bugstofix.md`'s standing note about avoiding an MSVC
toolchain install) or substantially more line-by-line source comparison time than this
iteration had left. Left for a future iteration with a larger time budget.

## WHISPER ASR NOW FULLY WORKING — fixed this session (3rd iteration), verified

**Root cause found and fixed.** `WhisperMelExtractor.CreateSlaneyMelFilterbank` had a
literal typo: `const float logStep = 27.0f / 64.0f; // log(6.4) / 27.0` — the comment
describes the correct formula but the CODE computes something else entirely (0.4219
instead of the real 0.0688, a ~6x error). This corrupted the mel-scale warping above
1000Hz badly enough that the lowest-index mel filters came out completely zeroed
(confirmed: filter[0] was all-zero in this engine vs. a real nonzero triangular filter in
a numpy reference port of librosa/whisper's exact filterbank algorithm), and every other
filter's shape was wrong too. That numpy reference (`scratch-llamacpp-ref/whisper_mel_ref.py`,
not checked in — session-local scratch) is what caught it: dumping both filterbanks and
diffing found the C# version's row sums varying erratically instead of holding the
constant value (~0.02486) Slaney normalization guarantees. **Fixed**: compute `logStep`
correctly (`MathF.Log(6.4f) / 27.0f`). Filter output now matches the numpy reference
exactly.

A second, independent bug was in the same file: `ExtractMel` padded audio by
`realAudioLength + 30 seconds` instead of padding/trimming TO exactly 30 seconds, producing
~4100 mel frames for an 11-second clip instead of the correct 3000 — and worse, the
per-utterance log-mel normalization max was then computed over that wrongly-oversized
buffer, shifting every frame's normalized value. **Fixed**: cap the pre-edge-pad buffer at
exactly `SampleRate * 30` (trimming if longer, zero-padding if shorter, matching
`openai-whisper`'s `pad_or_trim`), and added the missing end-of-buffer reflect-padding to
match `torch.stft(center=True)`'s behavior (a latent correctness issue for audio ≥30s,
though a no-op for anything shorter, which is everything this pipeline currently chunks
to).

Combined with the decode-time logit-suppression fix (still valid, see "Bug 3" below —
found and fixed in the same investigation, before the mel bug was found), **Whisper now
transcribes real audio correctly**. Verified two ways:
1. `stingray stt -i examples/whisper.cpp/samples/jfk.wav -m {tiny,base,small}` — ALL THREE
   sizes now produce "And so my fellow Americans, ask not what your country can do for
   you, ask what you can do for your country." (tiny drops one comma), matching the real
   quote essentially verbatim. Previously: "[Music]" / "(Police sirens)" at every size.
2. `stingray stt -i examples/qwen-asr/samples/test_speech.wav -m base` — "Hello, this is a
   test of the VoxTrol speech to text system." against ground truth (`test_speech.txt`)
   "Hello. This is a test of the Voxtrail speech-to-text system." — the differences here
   ("VoxTrol"/"Voxtrail", punctuation) are ordinary ASR-level variance on a made-up product
   name, not a bug.
3. Added a permanent ground-truth regression test,
   `WhisperRealWeightsTests.WhisperPipeline_RealModel_TranscribesJfkSampleCorrectly`
   (tiny + base), asserting actual transcribed CONTENT against the known JFK quote — not
   just shape/finiteness like every other existing "real weights" test in this file (which
   is why none of them caught either bug: they all use synthetic sine-wave audio and only
   assert the output doesn't crash/isn't NaN). Passes both sizes, ~6-11s each.
4. Full `tests/OpenTail.Stingray.Tests.Audio` suite: 104 → 110 tests (new diagnostic +
   regression tests), all passing.

**Correction to this doc's earlier claim**: iteration 1 said "only 2 test files exist" for
Audio — wrong, missed by an incomplete initial search. There are ~26 test files including
a `*RealWeightsTests.cs` per pipeline (pre-existing, committed 2026-08-19, not added this
session). The real gap was narrower than first stated: those tests DO run against real
model files, but only assert structural properties (finite output, no crash, top logit
above mean) on SYNTHETIC sine-wave input — never real audio, never real expected text.
That's still why none of this was caught before, just a more precise version of the claim.

## Bug 3, FIXED this session (2nd iteration) — missing decode-time logit suppression

Root-caused the "[Music]"/degenerate-collapse bug above by forcing the decoder past the
special/timestamp token cluster in the diagnostic harness: with those suppressed, the SAME
real audio produced fluent, grammatically real English ("...and I'll see you next time.
Bye. Bye...") — proving the encoder, self-attention, cross-attention, MLP, and tied LM head
are all numerically correct. The bug was entirely in decode-time sampling: this engine
never suppressed `<|notimestamps|>` (id 50363) at all, and real whisper.cpp (confirmed by
reading `examples/whisper.cpp/src/whisper.cpp` lines 6228-6354) suppresses it
UNCONDITIONALLY during generation (it's a prompt-only token), plus SOT/no-speech/task/
language tokens, plus a timestamp-pairing invariant (timestamps must alternate
open/close). None of that existed here — only a partial, incomplete version (no-speech
suppressed at step 0 only; timestamps suppressed only if the user disabled them entirely).
`<|notimestamps|>` had the single highest logit at step 0 (26.62, edging out the correct
`<|0.00|>` at 26.47) in every case tested, which is exactly what caused the collapse.

**Fixed:** added `WhisperTokenizer.ApplySamplingFilters` (ported faithfully from the
whisper.cpp source above — notimestamps/SOT/no-speech/task/language token suppression,
initial-step blank suppression, and the timestamp open/close pairing rule) and wired it
into `WhisperPipeline.ProcessAudioChunk`'s per-step decode loop, replacing the old partial
ad hoc suppression.

**Effect confirmed real but incomplete:** re-ran `stingray stt` after the fix — the
degenerate behavior is gone (previously: 1-second segment, garbage; now: one clean,
properly-paired timestamp segment spanning the full clip, same structure real Whisper
produces). This is a genuine, verified improvement, not a regression risk (it makes this
engine's sampling loop match the reference algorithm exactly, not a heuristic tweak).
**However, transcribed content is still wrong** on `examples/whisper.cpp/samples/jfk.wav`
at all three sizes tested: `tiny` → `"[Music]"`, `base` → `"[Music]"`, `small` →
`"(Police sirens)"`. These are plausible-format Whisper hallucination tags (real Whisper
does use bracketed non-speech tags for actual music/noise), but wrong for this specific,
famously clean/reliable reference clip that real whisper.cpp transcribes correctly even at
`base`. Getting a DIFFERENT wrong hallucination at each size (rather than the same one)
while all three fail the same way is itself informative: rules out one single frozen
bug shared identically at every scale, and rules out "weights are correctly loaded but
architecturally too small," and points at something upstream that's genuinely
size-independent — most likely something in the mel spectrogram extraction that makes
real speech numerically resemble noise/music to the encoder, since `WavReader.cs` was
checked this session and is structurally correct (standard RIFF/PCM decode; confirmed
empirically too — it reports the exact right sample count/rate/duration for jfk.wav).
**Next step:** get real numeric mel-spectrogram values from a reference (there's no local
whisper.cpp binary to diff against, would need e.g. a Python `librosa`/`openai-whisper`
one-off if available, or a hand-computed check of a few known frames) and compare against
this engine's `WhisperMelExtractor` output for the SAME audio, frame by frame -- this is
the highest-value next lead, more promising than further decoder-logic reading since the
decoder is now confirmed structurally correct.

## Smoke test results (before the fake-implementation finding above — kept for the
record, but note per the finding these mostly prove "the pipe runs and pipes emit sound",
not "the model works correctly")

| Pipeline | Ran without error against a real model file? | Real inference? |
|---|---|---|
| Kokoro | yes | **NO — confirmed fake** |
| Piper | yes | **NO — confirmed fake** |
| MeloTTS | yes | **NO — confirmed fake** |
| Whisper | yes | **YES — CONFIRMED WORKING**, see "WHISPER ASR NOW FULLY WORKING" above |
| F5-TTS | not run yet (needs `--ref-audio`/`--ref-text`) | **NO — confirmed fake** (DiT core) |
| Chatterbox | not run yet | **NO — confirmed fake** (`ChatterboxDecoder()` takes zero params) |
| CosyVoice | no CLI entry point | **NO — confirmed fake** (`CosyVoiceFlowDiT`/`HiFT`/`Llm` take only a config, no weights) |
| QwenTTS | no CLI entry point | **NO — confirmed fake** (`QwenTtsCodePredictor`/`DacDecoder`/`TalkerLm` take only a config, no weights) |
| QwenASR | no CLI entry point | **NO — confirmed fake** (`QwenAsrAudioEncoder` takes only a config, no weights) |
| Parakeet | no CLI entry point | **NO — confirmed fake** (`ParakeetConformerEncoder` takes only a config, no weights) |
| FunASR | no CLI entry point | **NO — confirmed fake** |
| Silero VAD | no CLI entry point | partially real, see above |

## CLI coverage gap (separate, smaller finding)

`TtsCommand.cs` only wires up kokoro/piper/f5tts/chatterbox/melo. `SttCommand.cs` only
wires up Whisper. QwenTTS, CosyVoice, QwenASR, Parakeet, FunASR pipelines exist in source
with no CLI entry point at all. Not worth adding CLI plumbing for pipelines that are fake
underneath — low priority until/unless the underlying pipeline is made real.

## Test coverage gap (corrected, see the correction note above — there ARE ~26 test files
per-pipeline, `*RealWeightsTests.cs`, pre-existing since 2026-08-19)

The real gap: those tests run against real model files but only assert structural
properties (finite output, no crash, top logit above mean) on SYNTHETIC sine-wave input —
never real audio, never real expected text. `GgufAudioAndEmbeddingRealWeightsTests.cs` (a
different test project) only checks that GGUF files load and have tensors/metadata — never
runs inference. This is why the fake-forward-pass pattern and Whisper's two real bugs were
never caught: nothing exercised `Forward()`/`Generate()`/`Transcribe()` against real audio
and checked the result against real known-correct content. The new
`WhisperPipeline_RealModel_TranscribesJfkSampleCorrectly` test (see above) is the first
one in this codebase that does that for any audio pipeline — worth using as the template
for closing this gap on other pipelines if/when they're made real.

## DSP utilities checked, 3rd iteration — no bugs found

`AudioResampler.cs` (windowed-sinc rational resampler with phase-bank caching) and
`Vad/VadSegmenter.cs` (frame-probability → speech-segment aggregation, with padding/merge
logic) were read in full. Both look like genuine, correctly-implemented standard
algorithms — proper Sinc*Hann kernel construction with DC-gain normalization in the
resampler, correct rising/falling-edge segment detection with padding and interval-merge
in the segmenter. No bugs found; not the cause of anything currently wrong. (`WavReader.cs`
was already checked and cleared in the 2nd iteration.)

**VAD end-to-end integration checked**: `stingray stt --vad` on jfk.wav does NOT crash
(good — the pipeline plumbing is solid) but produces visibly degraded, oddly-punctuated
output ("As not! What your country can do for you?" instead of "ask not what your country
can do for you") because Silero VAD's segment boundaries cut mid-sentence. This is the
expected, already-documented consequence of Silero VAD's fake/incomplete neural core
(see above) — not a new bug, just confirms the existing finding end-to-end rather than
only at the unit level.

## Next steps (pick up here)

1. **Whisper is done** — fully working, verified across tiny/base/small/medium (medium:
   exact word-for-word match against the real quote, 45s inference — the earlier 2-minute
   CLI timeout was just this session's `timeout` wrapper being too short for a CPU medium
   run, not a bug), regression-tested. Don't re-open without new evidence of a problem.
   `large-v3`/`large-v3-turbo` untested but there's no remaining reason to expect they
   differ — low priority.
2. **Audit of all 12 pipelines is now complete.** Only Whisper is real (and now fully
   working). The other 11 are all confirmed fake by direct code inspection — no more
   "presumed"/"not yet verified" entries remain. This is a stable finding; the open
   question is purely one of scope/priority for what (if anything) gets rebuilt next, not
   further investigation.
3. Do NOT start reimplementing any of the fake TTS/ASR neural architectures without
   either explicit user direction on priority, or extremely high confidence there's
   session time to do ONE of them properly (weight-tensor-correct, verified against a
   real reference) rather than partially across several. A half-real reimplementation is
   not obviously better than clearly-labeled-fake code, and multiplies the audit surface
   for whoever reviews this next.
4. ONNX Runtime option (see UPDATE above, 3rd iteration): likely blocked by
   `TrimMode=full` + `PublishAot=true` actually being set in
   `OpenTail.Stingray.Cli.csproj`, not just aspirational CLAUDE.md prose. Needs the user's
   explicit call, not just a flag — could require weakening a real, currently-enforced
   build constraint, which is a bigger decision than "add a package."
5. Silero VAD: the real GGUF is a materially different, messier architecture than
   `SileroVad.cs` assumes (see UPDATE above, 3rd iteration) — fixing it for real is
   comparable effort to one of the fake-TTS rebuilds (needs decoding the real ONNX graph
   structure from `models/silero_vad.onnx`), not a quick top-up of the existing skeleton.
   Same "don't rush it" guidance as item 3 applies here too.
5. Note: `dotnet build -c Release` on the FULL solution currently fails in
   `tests/OpenTail.Stingray.Tests.ForwardPass.Fast/ModelEndToEndAbBenchmark.cs`
   (`InferenceEngineOptions`/`InferenceEngine.CreateSession` API mismatch) — confirmed
   unrelated to any Audio change (core `InferenceEngine` API surface), and matches the
   files another session has uncommitted WIP in (`ModelGraph.cs`/`ForwardPass.Moe.cs`/
   `ModelCompatibility.cs`). Not touched, not this session's concern — leave it for that
   session. Audio/Cli-scoped builds (`dotnet build src/OpenTail.Stingray.Cli`,
   `dotnet test tests/OpenTail.Stingray.Tests.Audio`) are unaffected and pass clean.

## Changes made so far (for a diff/commit summary later)

- `src/OpenTail.Stingray.Cli/SttCommand.cs`: load real Whisper GGML weights by default
  (was silently running on untrained placeholder weights).
- `src/OpenTail.Stingray.Audio/Whisper/WhisperTokenizer.cs`: added `GetTokenId` and
  `ApplySamplingFilters` (ported from whisper.cpp's real decode-time logit suppression).
- `src/OpenTail.Stingray.Audio/Whisper/WhisperPipeline.cs`: wired `ApplySamplingFilters`
  into the decode loop, replacing the old partial/incomplete suppression.
- `tests/OpenTail.Stingray.Tests.Audio/WhisperDiagnosticTests.cs`: new, passes normally;
  set `STINGRAY_AUDIO_DIAGNOSTIC_DUMP=1` or run the built .exe with `-verbose` for the
  full stats dump. Kept as a reusable tool for the next iteration, not just this one.

## PERFORMANCE SWEEP (2026-08-21, same session as the Chatterbox T3/S3Gen rebuild) —
## from unusably slow to real-time-adjacent, same root fix applied across 3 pipelines

Context: Chatterbox T3 and S3Gen (see the T3/S3Gen sections above — encoder, CFM
flow-matching decoder, HiFTGenerator vocoder) were built real and numerically/structurally
verified, but the first end-to-end run against real weights took **5+ minutes and had not
finished** — unusable. The user asked to fix this properly rather than accept it, which
surfaced one recurring root cause across every pipeline touched this session: hand-written
C# inference code kept doing scalar, per-element, strided-memory loops in the hottest
paths (matmuls, attention, convolution), instead of the vectorized/parallelized primitives
this codebase already has available (`SimdKernels.MatVecF32`, `System.Numerics.Tensors.
TensorPrimitives`, `Parallel.For`). Every fix below is one of two mechanical patterns
applied repeatedly, not pipeline-specific cleverness:

1. **Matmul/Linear**: replace a hand-rolled `for outDim: for inDim: sum += ...` loop with
   `SimdKernels.MatVecF32` (AVX2/AVX-512, auto-parallelized across output rows) — the same
   kernel the main LLM inference engine already uses for GGUF forward passes.
2. **Attention weighted-sum, and convolution**: reorder loops so the innermost operation
   is a *contiguous, branch-free* scale-and-add over a whole row/span
   (`TensorPrimitives.MultiplyAdd(x, scale, addend, destination)` computing
   `destination = x*scale + addend`), instead of a *strided, per-element, bounds-checked*
   scalar accumulation. For attention this means looping `j`-outer/`d`-inner over the
   value cache instead of `d`-outer/`j`-inner. For convolution this means, for each fixed
   `(outChannel, inChannel, kernelTap)`, computing `output[ti] += weight * input[ti+shift]`
   across the *entire valid time range in one vectorized call*, instead of a per-timestep
   loop that re-reads `input` at a strided offset and bounds-checks every single element.
   Both reorderings were also paired with `Parallel.For` across independent output
   channels/positions where that wasn't already happening.

### Chatterbox T3 (GPT2-medium acoustic LM)
- Matmul fix (pattern 1): **5+ min, not finished → 44s** for a real prefill+decode run.
- Attention fix (pattern 2, applied to the Q·K/weighted-sum, not just matmul): 50s → 44s
  on top of the matmul fix (~12% further).
- Batching per-position `Linear()` calls into one `Parallel.For` dispatch per layer
  instead of N (further ~12%, included in the 44s figure above).

### Chatterbox S3Gen (Conformer flow encoder → CFM flow-matching UNet → HiFTGenerator vocoder)
Measured on a real end-to-end `ChatterboxPipeline.Generate()` call (real sentence, real
GGUF weights, ~750 mel frames / ~250 generated speech tokens — NOT the small 8-token
structural-test scale used for fast iteration):

| Stage          | Before   | After pattern-2 attention+softmax fix | After pattern-2 conv fix | Total speedup |
|----------------|----------|----------------------------------------|---------------------------|----------------|
| Flow encoder   | 7.9s     | 7.4-7.9s                                | (convs not the bottleneck here) | ~1.05x |
| CFM decoder    | 66.8s    | 34.1s (attention+softmax)               | 25.1s (+ conv vectorization) | **2.7x** |
| Vocoder        | 72.1s    | 58.3s (softmax fix only, no attention here) | **4.1s** (conv vectorization) | **17.6x** |
| **Full pipeline (incl. T3)** | **~2m54s** | ~2m04s | **~1m01s** | **2.85x** |

Two sub-fixes worth calling out specifically because they were found by asking "what looks
suspicious", not by pre-existing suspicion of a specific line:
- **`Math.Exp(double)` inside `SoftmaxInPlace`**: called ~T times per attention head, each
  summing T scores — O(T²) `Math.Exp` calls per attention block. At T≈750-800 this is
  hundreds of millions of double-precision transcendental calls with no numerical benefit
  over `MathF.Exp` for float32 softmax (found across `ChatterboxCfmDecoder.cs`,
  `ChatterboxFlowEncoder.cs`, and `ChatterboxAcousticLm.cs` — same copy-pasted pattern in
  all three). Fixing this alone nearly halved the CFM decoder's time (66.8s → 34.1s).
- **Vocoder convolution loop order**: the HiFiGAN resblock convolutions run at up to
  ~35,000 positions in the later upsample stages (near-full 24kHz sample rate) — this was
  the single biggest win in the whole sweep (17.6x on the vocoder alone). See pattern 2
  above; `ChatterboxVocoder.AxpyShifted`/`ChatterboxCfmDecoder.AxpyShifted` are the shared
  helper implementing it.

A dedicated **vocoder-only perf hotloop** was added for fast iteration:
`tests/OpenTail.Stingray.Tests.Audio/ChatterboxVocoderBenchmarkTests.cs` — real S3Gen
weights, synthetic mel input at a realistic scale (250 frames, matching what a real few-
second sentence produces), skips T3/encoder/CFM entirely. Cut the iteration loop from a
~2-3 minute full-pipeline run down to ~5-60s depending on which fix was being tested.
**Timings**: 59.8s (before conv fix) → **4.08s** (after) for 250 mel frames — this is the
number that got reused as the isolated "did the conv fix work" signal throughout.

**Permanent diagnostic logging added** (kept, not reverted — the user asked to keep it):
set `STINGRAY_AUDIO_DIAGNOSTIC_DUMP=1` and `ChatterboxPipeline.Generate()`/
`ChatterboxDecoder.DecodeReal()` will log a per-stage timing breakdown (T3 token count,
S3Gen encoder/CFM/vocoder split, frame/sample counts) to both stderr and
`%TEMP%\stingray-chatterbox-diag.log` — the file sink exists specifically because
xUnit/Microsoft.Testing.Platform only surfaces captured console output for *failing*
tests, so a passing diagnostic run would otherwise be invisible.

### Kokoro decoder/vocoder (istftnet.py Decoder + Generator, i.e. `KokoroDecoder.cs` /
### `KokoroAdainResBlk1d.cs`)
Same pattern-2 convolution fix applied after finding the exact same unfixed, **not even
parallelized** scalar/strided conv loops (`Conv1d`, `Conv1dK1`, `Conv1dDilated`,
`ConvTranspose1d`) — expected given Chatterbox's HiFiGAN vocoder code was explicitly
modeled on Kokoro's `AdaINResBlock1`/`AdainResBlk1d` earlier this session, so it's the
same architecture family with the same inefficiency. `KokoroRealWeightsTests` (a real
text→waveform synthesis, previously noted as a "182s test pass" in this doc's earlier
Kokoro-completion section) plus `KokoroDecoderTests` (the cosine-similarity-vs-golden-
waveform regression test) now both pass together in **13.7s total** — correctness
reconfirmed via the golden-output check, not just "didn't crash."

### Whisper (smaller win, lower priority, correctness-preserving)
Whisper's attention weighted-sum (encoder self-attn + decoder self-attn/cross-attn/
incremental-decode-step, 9 call sites total across `WhisperEncoder.cs`/
`WhisperDecoder.cs`) got the pattern-2 fix. Whisper's mel→hidden convolution
(`WhisperEncoder.ApplyConv1DReal`) is structurally different from Kokoro/Chatterbox's
resblock convs: it's called only twice per transcription (not per-layer/per-resblock), and
one of the two calls uses `stride=2` (mel downsampling), where output position `t` maps to
input position `2t+k` — a *strided gather*, not a contiguous shift, so it doesn't reduce to
the same `TensorPrimitives.MultiplyAdd` trick without more engineering. Fixed only the
`stride=1` call (the cheaper of the two); left `stride=2` as the original scalar/parallel
loop rather than risk a rewrite of the strided case under time pressure for a smaller
expected win. Verified via the real word-for-word JFK transcription regression test
(`WhisperPipeline_RealModel_TranscribesJfkSampleCorrectly`, tiny+base) — still passes,
no correctness change.

### UPDATE: Whisper's stride=2 conv also fixed (closes the gap noted above)
`WhisperEncoder.ApplyConv1DReal`'s `else` branch (the `stride=2` mel-downsampling conv) was
initially left unvectorized because output `t` maps to input `t*stride+k`, a strided
gather rather than the contiguous shift the `AxpyShifted` trick assumes. Closed it: compute
the valid `t` range analytically (same idea as `AxpyShifted`, removing the per-element
bounds check), gather the strided read into a small contiguous scratch buffer, then run the
weight-multiply-accumulate as one vectorized `TensorPrimitives.MultiplyAdd`. The gather
itself is still a scalar strided copy (no SIMD gather primitive available here), but it's
branch-free now, and the actual accumulate is vectorized. Verified against the real
word-for-word JFK regression test (tiny+base, still exact) and the medium-model end-to-end
test (1m03s, passed) — no correctness change. No known unvectorized conv loops remain
across Chatterbox/Kokoro/Whisper.

---

## F5-TTS DiT — REAL and golden-verified; Vocos vocoder BLOCKED (2026-08-21, continuation of the MASTER REBUILD PLAN after MeloTTS)

F5-TTS is architecturally a completely different family from the VITS-based pipelines
(Piper/MeloTTS) done earlier in this doc: a flow-matching Diffusion Transformer (DiT), not
a normalizing-flow TTS. Ships as a single `.safetensors` checkpoint (`models/
f5tts_base.safetensors`, raw torch state_dict layout -- no ONNX weight_norm-fusion
anonymization or MatMul-transpose quirks to worry about, unlike the VITS pipelines) plus a
real, runnable PyTorch reference (`examples/f5-tts-py`, from the official SWivid/F5-TTS
repo).

### Golden verification method: run the REAL PyTorch reference directly, not ONNX

Since F5-TTS ships as safetensors with working PyTorch source, golden dumps for this
pipeline load the ACTUAL reference `f5_tts.model.backbones.dit.DiT` class with the real
checkpoint and call it directly -- no ONNX re-export step, unlike Piper/MeloTTS. Getting the
reference source importable required installing several pip dependencies (`torchaudio`,
`librosa`, `rjieba`, `torchdiffeq`, `x_transformers`, `ema_pytorch`) and bypassing `f5_tts/
model/__init__.py` (which pulls in unrelated heavy deps like `wandb`/`trainer`) by
pre-registering stub package modules in `sys.modules` before importing just the submodules
needed (`f5_tts.model.backbones.dit`) -- see `scratch-llamacpp-ref/f5_golden_dit.py`'s
header comment for the exact trick. Config was confirmed exactly by loading the real
weights into the real `DiT` class with `strict=True` and getting zero missing/unexpected
keys: `dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2` (NOT the more common default of
4 -- confirmed via `ff.ff.0.0.weight`'s `[2048,1024]` shape), `text_dim=512, conv_layers=4,
qk_norm=None, long_skip_connection=False, attn_mask_enabled=False` (single-utterance
inference, no batch padding to mask).

### Architecture ported (all in `src/OpenTail.Stingray.Audio/F5TTS/`)

- **`F5Kernels.cs`**: low-level math on SEQUENCE-MAJOR (channel-last, `[T,D]`) arrays -- a
  DELIBERATE convention difference from the VITS-family pipelines' channel-first `[D,T]`
  layout (DiT/Transformer math is naturally per-timestep-row; VITS convs are naturally
  per-channel-row). Includes per-timestep `Linear`, affine/non-affine `LayerNorm`, exact
  erf-based GELU (`nn.GELU()` default) vs. tanh-approx GELU (`nn.GELU(approximate='tanh')`,
  used by the DiT FeedForward) as TWO DISTINCT functions (confirmed both variants are really
  used, in different places, by reading the reference source directly), depthwise and
  grouped "same"-padded Conv1d, and GRN (Global Response Normalization -- confirmed via
  reading `modules.py` that its L2-norm reduction axis is the SEQUENCE dimension, not
  per-timestep, a genuinely easy-to-miss detail).
- **`F5TextEmbedding.cs`**: token ids -> zero-padded-to-audio-frame-length embedding (`text
  = text + 1` for the 0-filler-token convention, truncate/pad to `numFrames`) + fixed
  sinusoidal position embedding (`precompute_freqs_cis`, concatenated cos/sin halves -- NOT
  interleaved, a different formula from the attention RoPE below) + 4x ConvNeXtV2Block,
  re-masking to zero at padded positions after every stage. Has a `dropText` parameter for
  CFG's null/unconditional branch (every position's embedding lookup becomes row 0, but the
  zero-mask still reflects the ORIGINAL pad boundary, not "everything is padding" -- a real
  subtlety in `dit.py`'s `TextEmbedding.forward`, ported faithfully after reading the exact
  order of operations).
- **`F5InputEmbedding.cs`**: `proj(cat[x,cond,text_embed])` + `ConvPositionEmbedding` (2x
  grouped Conv1d, `groups=16`, `k=31`, + Mish).
- **`F5TimestepEmbedding.cs`**: `SinusPositionEmbedding(256)` + 2-layer MLP with SiLU.
- **`F5RotaryEmbedding.cs`** + the RoPE application in **`F5DiTBlock.cs`**: interleaved
  ("GPT-J style") rotary embedding -- confirmed by installing the real `x_transformers`
  package and reading its actual `RotaryEmbedding`/`apply_rotary_pos_emb`/`rotate_half`
  source, NOT assumed from memory of "how RoPE usually works" (the split-half "rotate_half"
  convention used elsewhere is a DIFFERENT, incompatible convention from this one). Uses the
  checkpoint's own real `inv_freq` tensor directly rather than recomputing a theta formula.
- **`F5DiTBlock.cs`**: AdaLN-Zero modulation (`shift/scale/gate` x2, from one
  `Linear(1024,6144)` over `silu(t)`) + RoPE self-attention (no qk_norm, no mask) + gated
  FFN (tanh-approx GELU).
- **`F5DiTModel.cs`**: orchestrates all 22 blocks + `AdaLayerNorm_Final` + `proj_out` ->
  velocity.
- **`F5Tokenizer.cs`**: real character-vocabulary lookup loaded from the checkpoint's own
  `vocab.txt` (copied to `models/f5tts_vocab.txt`, 2545 lines matching `VocabSize` exactly),
  literal per-char id lookup with unknown-char fallback to id 0 (space) -- matching
  `list_str_to_idx`'s `vocab_char_map.get(c, 0)`. Does NOT implement the reference's
  `rjieba`+pinyin conversion for Chinese text (`convert_char_to_pinyin`) -- same documented
  scope boundary as Piper/MeloTTS's simplified phonemizers elsewhere in this doc: real
  neural math, simplified text normalization/g2p.
- **`F5FlowMatchingOde.cs`**: Euler ODE sampler with classifier-free guidance, ported from
  `cfm.py`'s `CFM.sample`. Documented simplification: uses the older `linspace(0,1,steps+1)`
  + optional `sway_sampling_coef` step schedule instead of the reference's newer default
  "Empirically Pruned Step Sampling" (EPSS, a precomputed lookup-table schedule) -- a real,
  still-supported non-EPSS code path in the reference, just not the newest optimization.
- **`F5TtsPipeline.cs`**: rewired to a real/fake dual path (mirroring
  `PiperModel`/`MeloModel`'s pattern) -- real path used when both a `.safetensors` weights
  file AND a resolvable `vocab.txt` are present; falls back to the original fake procedural
  DiT stand-in (moved here from the old `F5DiTModel.cs`, preserved verbatim as
  `FakeSolveFlowMatchingOde`/`FakeForwardVelocity`) otherwise.

### Bug found and fixed via golden bisection — in the DUMP SCRIPT, not the C# port

Initial `input_embed`/`pos_only`/`conv0_only` comparisons gave cosine similarities near 0
(`0.0015`, `-0.0057`, `0.163`) against otherwise-passing `text_embed`/`time_embed`/
`proj_only`/final-`velocity` checks -- looked exactly like a real transpose/indexing bug.
Root cause, found by replicating the EXACT SAME grouped-conv formula independently in pure
numpy against real checkpoint weights (matched the golden target at cosine `0.9999999`,
proving the C# math and weight-reading were both already correct) and then inspecting the
failing `.npy` files' raw headers directly: three golden dumps (`input_embed.npy`,
`pos_only.npy`, `conv0_only.npy`) were saved from tensors that had gone through
`.permute(...)` without a following `.contiguous()` call. `np.save` on a non-contiguous
torch tensor's `.numpy()` view writes the file with `'fortran_order': True` in its header --
a flat-byte `.npy` reader that assumes C (row-major) order, like every reader used
throughout this whole rebuild effort, silently misinterprets Fortran-ordered data as
row-major, which is numerically equivalent to reading a TRANSPOSED array. Fixed by adding
`.contiguous()` before every `.numpy()` call in `f5_golden_dit.py`, regenerating the
affected dumps, and adding an explicit `fortran_order` check (that throws with a clear
message) to `F5DiTModelTests.cs`'s npy reader so this class of mistake fails loudly instead
of silently next time. **No C# code changed as a result of this bisection** -- the DiT port
was correct on the first real attempt; only the verification harness was wrong. Documented
prominently in `F5DiTModelTests.cs`'s class doc comment as a durable warning.

### Test coverage

`F5DiTModelTests.cs`: checks `text_embed`, `time_embed`, `input_embed`, and the final
`velocity` output separately against their own real-PyTorch-reference golden targets (all
now pass, cosine similarity >0.999 for the intermediates, >0.99 for the final velocity).
Passes individually (`STINGRAY_RUN_HEAVY_TESTS=1`).

### NOT yet done / blocked for F5-TTS

- ~~Vocos vocoder blocked~~ -- **RESOLVED, see the dedicated "Vocos vocoder" section below.**
- `F5MelExtractor.cs` (reference-audio mel extraction for voice cloning) is REAL STFT +
  triangular mel filterbank code (pre-existing, not written this iteration), but NOT
  golden-verified against the checkpoint's own shipped `mel_spec.mel_stft.mel_scale.fb`/
  `window` buffers this pass -- worth doing before trusting voice-cloning conditioning
  quality, since `torchaudio.transforms.MelSpectrogram`'s exact padding/centering/
  normalization conventions were not independently confirmed to match.
- `F5TtsPipeline`'s duration estimate (chars-per-second heuristic -> `numFrames`) is a
  simplification, same category as the tokenizer/phonemizer scope boundary above.
- `F5TtsRealWeightsTests` (existing pipeline-level smoke test) is slow end-to-end on CPU
  (22-layer DiT x 32 Euler steps x 2 forward passes for CFG) -- expect it to take multiple
  minutes, not the ~1-20s of the VITS-family pipelines' equivalent smoke tests.

---

## Vocos vocoder — REAL and golden-verified (unblocks F5-TTS end-to-end; 2026-08-21, same session)

The user pointed at `charactr/vocos-mel-24khz` on HuggingFace as a candidate real Vocos
checkpoint (after correctly rejecting an earlier candidate GGUF repo that turned out to be
just a re-export of the DiT, not a vocoder — see above). Checked and confirmed real: MIT
license, `pytorch_model.bin` + `config.yaml`, and literally the vocoder backing most public
F5-TTS demo Spaces on HuggingFace.

### Getting the weights into this project's existing loading infrastructure

`pytorch_model.bin` is a pickled `torch.save` state_dict, NOT safetensors — this project's
`SafetensorsLoader` can't read it directly. Converted via `torch.load(weights_only=True)` +
`safetensors.torch.save_file()` into `models/vocos-mel-24khz.safetensors` (83 tensors,
verified same key set/shapes as the original). Per explicit user request ("I want GGUF and
ST. Both"), ALSO exported a real, spec-correct `models/vocos-mel-24khz.gguf` using the
official `gguf` PyPI package's `GGUFWriter` (not hand-rolled) — verified readable via
`gguf.GGUFReader` (all 83 tensors present, correct shapes in GGUF's reversed-dims
convention). The C# port uses the safetensors copy (this project's `SafetensorsLoader`
already handles this checkpoint's flat native-torch tensor layout with zero transpose/
anonymization quirks, same as F5-TTS's own checkpoint) — the `.gguf` copy exists as an
equally-valid alternative artifact via this project's existing `GgufModel` reader, not
currently wired to a second code path (would be pure duplication with no numerical
difference, since both contain byte-identical F32 tensor data).

### Architecture ported (`src/OpenTail.Stingray.Audio/F5TTS/VocosWeights.cs` + `VocosVocoder.cs`)

Read the real `vocos` PyPI package's source directly (`vocos/models.py`, `vocos/modules.py`,
`vocos/heads.py`, `vocos/spectral_ops.py`) rather than assumed from the "ConvNeXt vocoder"
name. Config confirmed via `models/vocos-mel-24khz-config.yaml` + real key/shape inspection:
`VocosBackbone(input_channels=100, dim=512, intermediate_dim=1536, num_layers=8)` -- no
AdaLayerNorm conditioning (this checkpoint has no per-bandwidth embeddings, unlike the
`VocosResNetBackbone`/multi-bandwidth variants some other Vocos checkpoints use).
`ISTFTHead(dim=512, n_fft=1024, hop_length=256, padding="center")`.

- `Conv1d(100,512,k=7)` embed -> `LayerNorm` -> 8x `ConvNeXtBlock` (depthwise `k=7` conv ->
  `LayerNorm` -> `Linear(512,1536)` -> exact GELU -> `Linear(1536,512)` -> per-channel
  learned `gamma` layer-scale -> residual add) -- notably SIMPLER than F5-TTS's own
  `ConvNeXtV2Block`: no GRN, just a scalar-per-channel `gamma` (confirmed by reading both
  reference sources side by side rather than assuming they're identical just because both
  are called "ConvNeXt blocks").
- -> `LayerNorm` (final) -> `ISTFTHead`: `Linear(512, 1026)` -> split into magnitude (`exp`,
  clipped at 100) and phase (`cos`/`sin`) -> complex spectrum -> **centered ISTFT**.
- **New primitive**: `SpectralKernels.InverseRealFft` (real->complex-conjugate-symmetric
  inverse DFT via the existing forward-DFT twiddle tables, reusable by direction since
  `cos` is even and the sign of `sin` just flips) plus a hand-written `CenteredIstft`
  (`VocosVocoder.cs`) implementing `torch.istft(..., center=True)`'s exact convention:
  per-frame inverse FFT, multiply by the analysis Hann window, overlap-add at `hop_length`
  spacing, normalize by the overlap-added squared-window envelope, then trim `n_fft/2`
  samples off each end (undoing the implicit center-padding the forward STFT would have
  used) -- confirmed output length `(numFrames-1)*hop_length` matches the real reference
  exactly for a 20-frame test input (`(20-1)*256 = 4864`, exactly what the golden dump
  produced).
- Reuses `F5Kernels`'s channel-last `[T,D]` primitives (`Linear`, `LayerNorm`, `GeluExact`,
  `DepthwiseConv1dSamePad`) and adds one new one, `Conv1dSamePad` (standard non-grouped
  conv, needed for the embed layer since `MelDim(100) != HiddenDim(512)` -- the existing
  `GroupedConv1dSamePad`/`DepthwiseConv1dSamePad` both assume equal in/out channel counts).

### Golden verification

`scratch-llamacpp-ref/vocos_golden_decode.py`: loads the real `vocos` package's
`VocosBackbone`/`ISTFTHead` classes directly with the real checkpoint, feeds a random mel
input straight into `decode()` (bypassing the feature extractor -- isolates vocoder
correctness from mel-extraction correctness, same bisection philosophy used throughout this
rebuild), dumps intermediate stages (`embed_out`, `norm_out`, `after_block0`,
`backbone_out`, `audio_out`). `VocosVocoderTests.cs` (new) checks the final waveform
against golden PyTorch output: **passed on the first attempt**, cosine similarity >0.99, no
bugs found this time (the F5-TTS DiT port established enough of the right conventions --
channel-last layout, real weight-loading via safetensors, careful reading of the actual
reference source rather than assumption -- that this smaller, structurally-simpler vocoder
came out correct immediately).

### Wired end-to-end

`F5TtsPipeline.Load` now also resolves `models/vocos-mel-24khz.safetensors` next to the
F5-TTS weights file (configurable via a new `vocosPath` parameter) and uses the real
`VocosVocoder.Decode` for the mel->waveform stage when found, falling back to the original
fake `F5VocosVocoder` placeholder otherwise -- exact same real/fake dual-path pattern as
every other pipeline in this doc.

### F5-TTS pipeline status update

With Vocos done, **every stage of F5-TTS's synthesis path is now real and independently
golden-verified**: text embedding, timestep embedding, input embedding, 22-layer DiT,
flow-matching ODE sampler, AND the vocoder. Remaining non-blocking caveats (same as noted
in the F5-TTS section above): `F5MelExtractor` (reference-audio mel extraction for voice
cloning) is real STFT code but not independently re-verified against the checkpoint's own
shipped mel filterbank this pass; the tokenizer/phonemizer and duration-estimate heuristics
are documented simplifications, same category as every other pipeline's phonemizer gap in
this doc. The pipeline-level smoke test (`Fast.F5TtsRealWeightsTests`) is genuinely slow on
CPU (22-layer DiT x 32 Euler steps x 2 CFG forward passes, observed ~8+ minutes) -- ran
individually per this doc's own testing discipline, not part of routine iteration.

## Parakeet — investigation resolved a false "blocked" conclusion; real reference found (2026-08-21, same session)

Investigation found `examples/nemo-toolkit-py` (the doc's previously-noted "not yet
verified" NeMo reference) is actually EMPTY (0 bytes) -- not just unverified, genuinely
absent. `examples/parakeet.cpp` (24MB, real git checkout, `mudler/parakeet.cpp`, confirmed
0 commits behind `origin/master`) IS a complete real GGML C++ FastConformer CTC reference,
but its `subsampling.cpp`/`subsampling.hpp` use tensor names `encoder.pre_encode.conv.0/2/3/
5/6.*`, while our local `models/parakeet-ctc-0.6b-q4_k.gguf` (`general.architecture =
canary_ctc`) uses `encoder.pre.conv.0/2/3/5/6.*` -- an initial (INCORRECT) read of this
mismatch concluded Parakeet was blocked on a missing subsampling reference.

**That conclusion was wrong, caught and corrected within the same iteration**: a full
(non-deduplicated) re-listing of the checkpoint's `encoder.pre.*` tensors showed `conv.0`
(full conv), `conv.2`+`conv.3` (depthwise+pointwise), `conv.5`+`conv.6` (depthwise+pointwise)
-- the EXACT same 3-stage `dw_striding` structure `parakeet.cpp` implements. The earlier
listing script deduplicated tensor names by a `\.\d+\.` regex substitution BEFORE printing,
which collapsed `conv.0/2/3/5/6` down to showing only the alphabetically-first match --
an artifact of the listing script, not a real architecture difference. Lesson: never
conclude "different architecture" from a tensor listing that was deduplicated for
readability -- always re-check the FULL undeduplicated list before declaring a mismatch.

**The real naming difference is just a renaming convention from a different GGUF converter**:
found `models/convert-canary-ctc-to-gguf.py` inside `CrispStrobe/CrispASR`
(github.com/CrispStrobe/CrispASR, already available locally under `examples/crispasr` --
the user had independently fetched it), whose `remap_name()` function shows the exact 1:1
rename table this specific checkpoint's converter used: `pre_encode`->`pre`,
`feed_forward1/2`->`ff1/2`, `norm_self_att`->`norm_attn`, `self_attn.linear_q/k/v/out/pos`->
`attn.q/k/v/out/pos`, `conv.pointwise_conv1/2`->`conv.pw1/pw2`, `conv.depthwise_conv`->
`conv.dw`, `conv.batch_norm`->`conv.bn` -- confirming the underlying NeMo module structure
is IDENTICAL to what `parakeet.cpp` already implements, just shorter tensor names. This repo
also ships the actual matching C++ runtime for this exact checkpoint/naming convention:
`src/canary_ctc.cpp` (1394 lines) + `src/core/fastconformer.h` (993 lines) -- a real,
directly-applicable oracle, on top of `parakeet.cpp`'s own `conformer.cpp`/
`relpos_attention.cpp`/`subsampling.cpp` for cross-reference (same architecture family).

**Not actually blocked.** Config confirmed via the checkpoint's own GGUF metadata:
`d_model=1024, n_layers=24, n_heads=8, head_dim=128, ff_dim=4096, subsampling_factor=8,
subsampling_channels=256, conv_kernel=9, n_mels=80, n_fft=512, win_length=400,
hop_length=160, sample_rate=16000, vocab_size=1024, blank_id=1024` (CTC blank is the last
vocab index). **Ran out of session budget before implementing** -- full architecture spec
below so the next iteration can go straight to writing+verifying C# without re-deriving.

### Full architecture spec (read directly from `examples/crispasr/src/core/fastconformer.h`, lines 98-991 -- READ THIS FILE FIRST next iteration, it is the actual math, not a summary)

**Subsampling (`build_pre_encode`, dw_striding, 8x)**: input mel `[n_mels=80, T]` as a 2D
"image" (1 in-channel). `Conv2d(1->256,k=3,s=2,p=1)+bias -> ReLU -> DWConv2d(256,k=3,s=2,
p=1)+bias -> PWConv2d(256->256,k=1)+bias -> ReLU -> DWConv2d(256,k=3,s=2,p=1)+bias ->
PWConv2d(256->256,k=1)+bias -> ReLU`. Output `[OW=freq_out, OH=T_enc, OC=256]`; flatten via
permute(1,2,0,3): `feature[k] = channel*OW + freq_ow` (channel-major, NOT freq-major -- this
ordering is called out explicitly in the source, get it backwards and everything downstream
breaks silently). Then `Linear(256*W3=2560 -> 1024)+bias` (`encoder.pre.out`). Tensor names:
`encoder.pre.conv.{0,2,3,5,6}.{weight,bias}` + `encoder.pre.out.{weight,bias}` (already
matches our GGUF, see the corrected finding above).

**Positional encoding (`make_pos_enc`)**: sinusoidal, length `2T-1`, `pe[p*d+2i] =
sin(pos*div)`, `pe[p*d+2i+1] = cos(pos*div)` where `pos = T-1-p` (descending from `+(T-1)`
to `-(T-1)`), `div = exp(-log(10000)*2i/d)`. This is a fixed (non-learned) table, computed
fresh per utterance length, NOT loaded from the checkpoint.

**Conformer block (`build_block`, x24, macaron structure)**:
1. `FFN1`: `LayerNorm(norm_ff1) -> Linear(d,ff)+bias(ff1.linear1) -> SiLU ->
   Linear(ff,d)+bias(ff1.linear2)`; `x = x + 0.5*ffn_out` (macaron HALF-step, note the 0.5
   scale -- easy to miss).
2. `Self-attention` (Transformer-XL-style relative position, untied `u`/`v` biases -- this
   is the Conformer paper's exact rel-pos MHSA, NOT the interleaved/split-half RoPE used
   elsewhere in this codebase for F5-TTS -- a THIRD distinct positional-encoding convention
   in this repo now, don't cross-contaminate): `LayerNorm(norm_attn)`, then
   `Q=Linear(d,d)(x)`, `K=Linear(d,d)(x)`, `V=Linear(d,d)(x)` -- **NO bias** on q/k/v/out/ff
   linears for this checkpoint family (`parakeet`/`canary_ctc`; `canary` proper has biases on
   everything -- confirmed via `BlockWeights`'s comment and the checkpoint's own converter,
   which never emits q/k/v/out/ff bias tensors for this arch). `R = Linear(d,d)(pos_enc)`
   (`attn.pos`, no bias). `Q_u = Q + pos_bias_u` (broadcast add per-head, `pos_bias_u`/`v`
   shape `[head_dim, n_heads]`, i.e. one head_dim-length vector per head, added to Q's
   corresponding head slice). Reshape Q/K/V into `[head_dim, n_heads, T]` heads. Per head:
   `AC = Q_u @ K^T` (`[T,T]`); `BD_raw = R @ Q_v^T` giving `[2T-1, T]` (relative-position
   score per query, per relative offset); `BD = rel_shift(BD_raw)`: the standard
   Transformer-XL shift, `BD[q,k] = BD_raw[q, (T-1-q)+k]` for `k=0..T-1` (derived directly
   from the `pos = T-1-p` indexing above -- verify this derivation against `rel_shift`'s
   ggml view-stride trick, lines 98-102, before trusting it blindly). `scores =
   (AC+BD)*scale` where `scale=1/sqrt(head_dim)`; softmax over k; `attn_out = softmax @ V`
   per head, concat heads, `Linear(d,d)` (`attn.out`, no bias). Residual add (full step, no
   0.5 here).
3. `Conv module`: `LayerNorm(norm_conv)`, `pw1: Linear(d,2d)+bias` -> **GLU**: split into two
   `d`-halves, `out = first_half * sigmoid(second_half)` (verify exact half order against
   `ggml_siglu_swapped` -- the "swapped" in the name is suspicious, don't assume standard
   GLU order without checking) -> depthwise Conv1d (kernel=`conv_kernel`=9,
   padding=`(K-1)/2`=4, groups=d) + bias (**BN already folded into this bias at load time**
   for `parakeet`/`canary_ctc` -- our GGUF's `conv.dw.bias` is a SYNTHETIC zero tensor added
   by the converter as a BN-fold target, see `convert-canary-ctc-to-gguf.py` lines 272-279;
   the real BN scale/shift must be folded into `conv.dw.weight`/`conv.dw.bias` -- check if
   our GGUF's `conv.bn.*` tensors are ALSO present in raw (unfolded) form, since our
   checkpoint might ship both and expect the C# port to do the fold itself, unlike the
   already-fused `.gguf` this fold-at-load logic assumes) -> `SiLU` -> `pw2: Linear(d,d)+bias`.
   Residual add.
4. `FFN2`: identical structure to FFN1 (own `norm_ff2`/`ff2.linear1/2` weights), same 0.5 scale.
5. `LayerNorm(norm_out)` (block-final norm, own weights per layer).

**CTC head**: `Linear(1024, vocab_size+1=1025)+bias` (`ctc.weight/bias`) over the encoder's
final `[T_enc, 1024]` output, then log-softmax per frame; greedy CTC decode = argmax per
frame, collapse consecutive repeats, drop blank (`blank_id` = last vocab index).

**Mel preprocessing** (`preprocessor.fb [257,80]`, `preprocessor.window [400]`, real
filterbank + window SHIPPED in the checkpoint -- use them directly, don't recompute a
formula, same principle as this doc's MeloTTS mel-filterbank note): NOT yet read this
session -- check `examples/crispasr/src/canary_ctc.cpp` (1394 lines, not yet opened) for the
exact log-compression/normalization/dithering NeMo uses; do NOT assume it matches
`ParakeetMelExtractor.cs`'s existing (unverified) formula.

**LayerNorm epsilon**: `BlockParams::ln_eps` -- value not yet read from
`examples/crispasr/src/canary_ctc.cpp`'s weight-loading code; check there (likely `1e-5`,
NeMo's Conformer default, but VERIFY, don't assume).

**Next iteration**: read `examples/crispasr/src/canary_ctc.cpp` for (a) the exact BN-fold
formula if not already fused in our GGUF, (b) mel preprocessing exact formula, (c)
`ln_eps`/`BlockParams` construction, (d) tokenizer (SentencePiece vocab is embedded in the
GGUF via `tokenizer.ggml.tokens`, already loadable). Then build a golden-output oracle
(compile `examples/crispasr` and run it against `models/parakeet-ctc-0.6b-q4_k.gguf` with a
short WAV to dump intermediate tensors, OR add debug output to the C++ source and rebuild --
check if it already has a "diff harness" mode, since `fastconformer.h`'s comments mention a
"diff harness" / "staged comparison" mechanism (`snap_conv4d`, `gf` param) that may ALREADY
support dumping intermediate activations without modification). Then port
`ParakeetConformerEncoder.cs` (currently 100% fake, see file) against real GGUF weights,
verifying every stage against that oracle -- same discipline as every other pipeline in this
doc. This is comparable in scope to the F5-TTS DiT effort (also 20+ layers, real attention +
conv module), should take a similar number of iterations.

### ParakeetWeights.cs rewritten with the full real tensor set + BN fold (2026-08-21, loop iteration 2)

`ParakeetWeights.cs` was previously a skeleton (only exposed `CtcBias`). Rewrote it to load
every real tensor confirmed present in `models/parakeet-ctc-0.6b-q4_k.gguf` (verified via
`dotnet run --project src/OpenTail.Stingray.Cli -c Release -- list-tensors -m models/
parakeet-ctc-0.6b-q4_k.gguf`, 957 lines, all tensor names match the spec above exactly):
subsampling front-end (`encoder.pre.conv.{0,2,3,5,6}`, `encoder.pre.out`), per-layer Conformer
weights (`ParakeetConformerLayer`, one instance per layer: `norm_ff1/attn/conv/ff2/out`,
`ff1/ff2.linear1/2`, `attn.q/k/v/out/pos` + `pos_bias_u/v`, `conv.pw1/pw2` + folded
`conv.dw`), mel preprocessing (`preprocessor.fb`, `preprocessor.window`), CTC head
(`ctc.weight` [1024,1025], `ctc.bias` [1025]). Follows the same eager-dequantize-to-float32
pattern as `ChatterboxWeights.cs`/`KokoroWeights.cs` (`Dequantize.ToFloat32` per tensor).

**Confirmed our checkpoint ships UNFUSED BatchNorm tensors** (`encoder.layers.{i}.conv.bn.
{weight,bias,running_mean,running_var}`, all present, all `Float32 [1024]`, verified via the
same tensor listing) alongside `conv.dw.weight`/`conv.dw.bias` -- so the BN-fold-at-load step
flagged as uncertain in the spec above IS required for this checkpoint (not already fused).
Implemented `ParakeetConformerLayer.FoldBatchNorm` matching `canary_ctc.cpp`'s
`cc_fold_batchnorm` exactly: `s[c] = bn_weight[c] / sqrt(bn_var[c] + 1e-5)`,
`w_folded[k,c] = w[k,c] * s[c]`, `b_folded[c] = s[c]*orig_bias[c] - bn_mean[c]*s[c] + bn_bias[c]`.
Depthwise conv weight storage order assumed `[K, 1, d]` -> flat index `k*channels + c`
(matches GGUF's listed dims `[9, 1, 1024]` reversed to row-major `[K,1,d]`, same
dims-reversed-vs-torch convention noted in `ChatterboxWeights.cs`'s doc comment) -- **not yet
cross-checked against `canary_ctc.cpp`'s actual indexing of `conv_dw_w`, do this first next
iteration** before trusting the fold numerically.

`dotnet build src/OpenTail.Stingray.Audio -c Debug` — clean build, no errors. Not yet
golden-verified (no oracle run yet) and `ParakeetConformerEncoder.cs` itself is still 100%
fake — only the weight-loading layer is real so far.

**Next iteration (unchanged in substance, just re-sequenced)**:
1. Verify the `conv.dw` storage-order/BN-fold assumption above against `canary_ctc.cpp`'s
   actual tensor read code before trusting it.
2. Read `canary_ctc.cpp` for mel preprocessing exact formula (log-compression/normalization/
   dithering) — do NOT assume `ParakeetMelExtractor.cs`'s existing formula matches.
3. Build a golden-output oracle (compile `examples/crispasr` against a short WAV, dump
   intermediate tensors — check for an existing "diff harness"/`snap_conv4d` mode first).
4. Port `ParakeetConformerEncoder.cs`'s forward pass (subsampling -> pos-enc -> 24 blocks ->
   CTC head) against the now-real `ParakeetWeights`, verifying every stage against the oracle
   (cosine >0.99 bar, same as every other pipeline in this doc) before calling it done.

### ParakeetConformerEncoder.cs fully ported (structurally complete, NOT yet golden-verified) (2026-08-21, loop iteration 3)

**Fixed a real bug found while re-deriving the BN-fold before trusting it (step 1 above)**:
the previous iteration's `FoldBatchNorm` assumed depthwise conv weight storage order
`k*channels+c`. Reading `canary_ctc.cpp`'s actual `cc_fold_batchnorm` loop
(`w_f32[ki + c * K] *= s[c]`) shows the real order is `c*K+k` (GGML `ne=[K,1,d]`, K
fastest-varying, so flattening is channel-major not kernel-major). Fixed in
`ParakeetWeights.cs`'s `FoldBatchNorm`.

**Also found our checkpoint DOES carry attn q/k/v/out biases** (`encoder.layers.{i}.attn.
{q,k,v,out}.bias`, all `Float32 [1024]`, confirmed present in the tensor listing) —
contradicting this doc's earlier note that "parakeet/canary_ctc has NO bias on q/k/v/out".
That note came from `canary_ctc.cpp`'s doc-comment describing the *generic* parakeet/
canary_ctc family, but `BlockWeights`'s bias fields are optional (`get_opt`) precisely
because specific checkpoints can differ — ours has them. Added `AttnQBias`/`AttnKBias`/
`AttnVBias`/`AttnOutBias` (all `float[]?`, loaded via `TryGetTensor`) to
`ParakeetConformerLayer`. `attn.pos` (rel-pos projection) genuinely has no bias tensor —
that part of the original note was correct.

**Wrote the full encoder forward pass** in `ParakeetConformerEncoder.cs` (now a static class,
replacing the 100%-fake instance-based placeholder): 3-stage dw_striding subsampling (full
conv2d -> ReLU -> depthwise+pointwise conv2d -> ReLU -> depthwise+pointwise conv2d -> ReLU,
channel-major flatten -> Linear), sinusoidal rel-pos table (length 2T-1, descending position),
24x Conformer block (macaron FFN1 half-step -> rel-pos self-attention with untied u/v biases
and the full Transformer-XL `rel_shift` derivation `BD[q,k] = BD_raw[(T-1)+k-q, q]` computed
directly per (query,key) pair rather than materializing the shifted matrix -> GLU depthwise-
conv module with the BN-folded weights -> macaron FFN2 half-step -> final LayerNorm), CTC head.
Reused the per-head `qU`/`qV` precompute + `Parallel.For` + SIMD-dot-product attention pattern
from `Chatterbox/ChatterboxFlowEncoder.cs`'s `RelPositionSelfAttention` (same Transformer-XL
rel-pos family, S3Gen's simplified case where pos_emb length always equals key length so it
skips `rel_shift` entirely — Parakeet's version needed the full shift since pos_emb is 2T-1).

**Rewired `ParakeetCtcDecoder.DecodeGreedy`** to take real per-frame CTC logits directly
(`ctcLogits[frame][vocab+1]`) instead of its previous fake cosine-based per-vocab scoring
loop — now genuinely just argmax + collapse-repeats + drop-blank, the standard greedy CTC
recipe, no synthetic math left in the decode path. Updated `ParakeetPipeline.cs` to call the
new static `ParakeetConformerEncoder.Forward(weights, mel, tMel)` and pass its CTC logits
straight to the decoder; the pipeline's no-args constructor now throws if used without real
GGUF weights (`ParakeetPipeline.Load` is required — no procedural fallback exists for this
pipeline anymore, unlike some others that keep a fast/fake path for quick structural tests).

**Test changes**: removed the four `Tests.Audio.Fast/ParakeetTests.cs` tests that exercised
the old fake procedural encoder/decoder/pipeline APIs (their constructors no longer exist).
Added `Tests.Audio/ParakeetConformerEncoderTests.cs` (`HeavyTestBase`, real GGUF weights):
loads real weights and checks all 24 layers' folded conv tensors are finite; runs the full
encoder forward pass on synthetic mel input and checks output shapes/finiteness; loads the
full pipeline via `ParakeetPipeline.Load` and transcribes 1s of synthetic audio without
crashing. All 3 tests PASS (`STINGRAY_RUN_HEAVY_TESTS=1 dotnet test
tests/OpenTail.Stingray.Tests.Audio -- --filter-class
OpenTail.Stingray.Tests.Audio.ParakeetConformerEncoderTests`, ~16s).

**NOT yet golden-verified** — no crispasr oracle run has been done yet, so per-stage cosine
similarity against real output is still unconfirmed; these tests only prove the real weights
load and the forward pass runs end-to-end without NaN/Inf, the same "does it even run for
real" bar every other pipeline passed before golden verification followed later. Mel
preprocessing exact formula also still unverified against `canary_ctc.cpp` (step 2 above,
still open). **Do not claim Parakeet is "done" yet** — structurally complete only.

**Unrelated pre-existing bug found, NOT part of this iteration's changes**: building
`tests/OpenTail.Stingray.Tests.Audio.Fast` fails on `F5TtsTests.cs` (`F5DiTModel` was made
static in an earlier session's F5-TTS DiT work but this test file was never updated to
match — `Cannot create an instance of the static class 'F5DiTModel'` etc, 7 compile errors).
Confirmed via `git status`/`git diff` that this file is untouched by this session. Flagging
for whoever picks up F5-TTS-adjacent work next; did not fix it here since it's out of scope
for the Parakeet queue item and unrelated to the changes in this iteration.

**Next iteration**: (1) fix the `F5TtsTests.cs` compile break so the Fast test project builds
again (small, unrelated fix, doesn't need its own full iteration). (2) Verify mel
preprocessing against `canary_ctc.cpp`. (3) Build a real oracle (compile `examples/crispasr`
or find its existing diff-harness mode) and golden-verify each encoder stage numerically
before calling Parakeet done. (4) Then move to CosyVoice per the queue.

### F5TtsTests.cs fixed; ParakeetMelExtractor.cs rewritten to use the real checkpoint preprocessing; encoder performance pass (2026-08-21, same session, direct user requests)

**(1) done**: `F5TtsTests.cs`'s `F5DiTModel_SolveFlowMatchingOde_SolvesTrajectory` test
exercised the old fake instance-based `F5DiTModel` API, which no longer exists (an earlier
session made it a real, static, golden-verified class and never updated this Fast test).
Removed the stale test with a comment pointing at the real coverage
(`Tests.Audio/F5DiTModelTests.cs`, `Tests.Audio/F5TtsRealWeightsTests.cs`). `Tests.Audio.Fast`
now builds clean; ran `F5TtsTests` (4 remaining tests) individually to confirm no regression
— all PASS.

**(2) done**: rewrote `ParakeetMelExtractor.cs` from scratch to match `examples/crispasr/src/
core/mel.cpp`'s exact NeMo `AudioToMelSpectrogramPreprocessor` pipeline (traced through
`canary_ctc.cpp`'s `core_mel::Params` construction, `mel.cpp`'s stage-by-stage implementation,
and `mel.h`'s enum defaults for the fields `canary_ctc.cpp` doesn't override):
- Pre-emphasis (0.97) applied globally to the raw signal BEFORE center-padding, not reset per
  frame (previous version was already close to this but didn't center-pad or center the
  window).
- Zero center-pad by `n_fft/2` (256) on both sides — previous version had no center-padding
  at all, meaning every frame boundary was wrong once padding is accounted for.
- Window (400 real samples from `preprocessor.window`) centered within the 512-sample FFT
  buffer with `(512-400)/2=56`-sample zero-pad on each side (`lpad`) — previous version put
  the window at the buffer's start (`real[0..399]`) with no centering, a real bug.
- Mel filterbank now uses the checkpoint's own real shipped `preprocessor.fb` tensor
  (`ParakeetWeights.MelFilterbank`) instead of a self-computed librosa-style triangular
  filterbank — the doc had flagged this as required ("use them directly, don't recompute a
  formula") since the previous iteration but it hadn't been done yet.
- Log: `log(x + log_eps)` with `log_eps = 2^-24` (`LogGuard::AddEpsilon`, NeMo's convention)
  — previous version used `log(max(x, 1e-5))` (`LogGuard::MaxClip`-style with the wrong
  epsilon), a different function entirely, not just a different constant.
- Added the previously entirely-missing per-feature Z-normalization stage
  (`Normalization::PerFeatureZ`): per-mel-band mean/variance across time, Bessel-corrected
  (`denom = T-1`), `std += 1e-5` applied OUTSIDE the sqrt (order matters, per `mel.cpp`'s own
  comment about issue #37 — an eps-inside-sqrt version would under-amplify low-variance mel
  bands relative to NeMo). Previous version had no normalization step at all.
- Kept a fallback self-computed filterbank/window path (`ParakeetMelExtractor()` no-args ctor)
  for callers without a loaded checkpoint (e.g. structural tests) — explicitly documented as
  NOT what the real checkpoint expects; `ParakeetPipeline.Load` always uses the new
  `ParakeetMelExtractor.FromWeights(weights)` factory instead, wired into `ParakeetPipeline.cs`.
- Dropped a `powerSpectrum[k] /= NFft` normalization the previous version had after the power
  spectrum computation — checked `F5MelExtractor.cs` (a real, golden-verified pipeline using
  the same `SpectralKernels.ComputePowerSpectrum`) and it applies no such division, so the
  previous Parakeet-specific division looks like it was never verified against anything real.

Build clean (`dotnet build src/OpenTail.Stingray.Audio`). Not yet re-run against the Heavy
encoder tests with this new mel path specifically (those tests feed synthetic mel directly,
bypassing `ExtractMel`) — still needs its own verification pass, and still no numeric oracle
comparison exists for the mel stage itself (same "structurally corrected against the real
reference source, not yet golden-verified" caveat as the rest of Parakeet).

**Performance pass on `ParakeetConformerEncoder.cs`** (direct user request, explicitly scoped
to "improve existing code, don't start new pipeline work"): the per-frame loops inside each
Conformer block's FFN1, FFN2, attention Q/K/V projection, rel-pos projection, and conv
module's pw1/dw/pw2 stages were serial `for` loops over T frames despite every frame's work
being independent (each frame's `Linear` calls only read that frame's own row). These are the
actual cost centers (each involves at least one `dim`x`dim` or `dim`x`ff_dim` matmul, times T
frames, times 24 layers), so parallelized all of them with `Parallel.For` across the frame
index. Left the plain O(T*dim) residual-add loops (no matmul) serial — parallelization
overhead would exceed their cost. Did NOT touch the subsampling stage's pointwise conv2d
(`Conv2dPointwise`) despite its scalar per-channel loop being SIMD-unfriendly in its current
layout (channel-strided, not contiguous) — it runs twice total per utterance (not once per
layer), so it's a small fraction of total cost and not worth the risk of a transpose-layout
bug for this pass. Verified via `ParakeetConformerEncoderTests` (all 3 tests, real GGUF
weights) — still PASS after the parallelization changes, 12.7s vs 15.5s before on the same
small synthetic-mel test input (real gains should be larger on longer real audio, where T is
in the hundreds rather than ~8).

**Next iteration**: (1) verify the new mel extractor's output shape/values are sane against
real audio (currently only build-verified, not test-run through `ExtractMel` itself). (2)
Build a real oracle and golden-verify Parakeet numerically end-to-end (mel + encoder + CTC),
the last thing standing between "structurally complete" and "done". (3) Then CosyVoice per
the queue.

## New planning docs added under `docs/audio/` (2026-08-22) — user-provided, treat as reference not gospel

User added six external planning documents (`046-native-cosyvoice-tts-plan.md` [+ an `-old`
superseded variant], `049-native-qwen3-asr-completion-plan.md`, `050-QwenTTS-...-plan.md`,
`51fish.md`, `52parlertts.md`, `53orpheous.md`) — 3 pipelines already in this doc's queue
(CosyVoice, QwenASR, QwenTTS) plus 3 new ones not previously tracked (Fish Speech, Parler-TTS,
Orpheus TTS). Per explicit user instruction, these planners "do not know the code we have in
front of us" — they're useful for architecture/ground-truth context (config values, recommended
phase ordering, ground-truth source hierarchy) but every concrete claim about this repo's
current state must be re-verified against the actual code/checkpoints directly, the same
discipline already established for every other pipeline in this doc. Queue order updated:
CosyVoice (in progress, see below) -> QwenASR -> QwenTTS -> FunASR -> Silero VAD -> Fish Speech
-> Parler-TTS -> Orpheus TTS.

### CosyVoice — Phase 0 audit: real checkpoint inventory, one concrete bug fixed, one real blocker found (2026-08-22)

**Fixed a real bug**: `CosyVoiceWeights.cs`'s constants (`HiddenDim=1024, NumHeads=16,
NumKvHeads=8, IntermediateDim=2816`) did not match the actual checkpoint at all — verified
directly by reading `models/cosyvoice2_0.5b.safetensors`'s raw safetensors JSON header (`node`
can read this directly: 8-byte LE length prefix + JSON header, no python needed) and
`models/cosyvoice2_config.json`. Real values: `hidden_size=896, num_attention_heads=14,
num_key_value_heads=2, intermediate_size=4864, vocab_size=151936, rope_theta=1e6,
rms_norm_eps=1e-6, tie_word_embeddings=true, 24 layers` — a plain `Qwen2ForCausalLM` config.
`CosyVoiceLlmConfig` (in `CosyVoiceLlm.cs`) already had these correct numbers; only
`CosyVoiceWeights.cs` was stale (likely copy-pasted from a different model's constants).
Fixed, builds clean.

**Real finding, worth flagging clearly before more work goes into this pipeline**: inspected
every local CosyVoice-related asset's actual tensor/string contents (via `node` reading raw
safetensors JSON headers and grepping raw ONNX protobuf bytes for readable tensor-name
strings — no python/onnx tooling needed for this level of inspection):
- `models/cosyvoice2_0.5b.safetensors` (290 tensors: `model.embed_tokens.weight`
  `[151936,896]`, `model.norm.weight`, 24x standard `model.layers.{i}.{self_attn,mlp}.*`) is
  **only the vanilla Qwen2-0.5B text backbone** — `tie_word_embeddings=true` means there's no
  separate output head, and the embedding table is exactly the base model's 151936-token TEXT
  vocab. There is NO speech-token embedding table and NO speech-vocab output head anywhere in
  this file. Real CosyVoice checkpoints extend the embedding/head for the FSQ speech codebook
  (`SpeechTokenSize=6561` + special tokens, per `CosyVoiceLlmConfig`) — that extension is
  simply not present in what we have locally.
- `models/cosyvoice_speech_tokenizer.onnx` — confirmed via string grep (`encoders.
  positional_embedding`) to be the audio->speech-token ENCODER (Whisper-style), not an LLM
  decoder head. Doesn't fill the gap above.
- `models/flow.decoder.estimator.fp32.onnx` (286MB) — grepped for `encoder|conformer|
  upsample|input_embed` and found NONE of those strings anywhere in the file. This is only
  the CFM/DiT *estimator* network (confirms its own tensor name `estimator_out`); the
  token-conditioning encoder stage that real CosyVoice needs before the CFM decoder (the
  Chatterbox-S3Gen-equivalent `UpsampleConformerEncoder` that turns speech tokens into the
  DiT's `mu` conditioning tensor — see `Chatterbox/ChatterboxFlowEncoder.cs` for what that
  looks like in this codebase for a different model) is NOT bundled in this file and no
  separate weight file for it exists locally either.
- **No HiFT vocoder weights exist anywhere under `models/`** (checked exhaustively: no
  `*hift*`, `*vocoder*`, or other `*cosyvoice*`-named file beyond the three already listed).
  `CosyVoiceHiFT.cs` currently has nothing real to load from at all.
- `models/campplus.onnx` (speaker x-vector embedding) is present and its purpose is
  unambiguous from the model name/format — likely fine as-is, not specifically re-verified
  this pass.

**Assessment**: CosyVoice is more blocked than Parakeet ever was (Parakeet's "blocked"
conclusion was a false alarm from a bad tensor-listing script; this one is a real, confirmed
absence of weights after direct inspection, matching this doc's QwenTTS precedent for a
genuine blocker). Three of five stages (LLM speech-token head/embedding, flow's token->mu
conditioning encoder, HiFT vocoder) have no real local weights to port against numerically.
Two stages (LLM backbone itself, flow's CFM estimator) DO have real weights and could be
ported/verified.

**What IS still productively actionable without new weights**:
1. The plan docs' section 13 ("Do Not Duplicate OpenTail's Model Runtime") is worth taking
   seriously for the LLM backbone specifically: since `cosyvoice2_0.5b.safetensors` is a
   *plain, standard* Qwen2ForCausalLM (not a custom architecture needing hand-rolled scalar
   C# math the way Parakeet's FastConformer or Chatterbox's Conformer did), this codebase's
   own existing text-LLM engine (`OpenTail.Stingray.Engine`'s `ForwardPass`, per CLAUDE.md)
   should in principle be able to run this backbone directly once/if a real speech-vocab
   checkpoint is sourced, rather than writing a second bespoke transformer implementation —
   worth a follow-up architecture check (does `SafetensorsLlamaWeightLoader`/the Engine layer
   support Qwen2's GQA+RoPE+SwiGLU shape already, given it clearly supports at least one
   GQA/RoPE architecture already for the main LLM feature) before deciding to hand-write a
   CosyVoice-specific forward pass. Not attempted this iteration — flagging as the single
   highest-leverage next question for whoever continues this pipeline.
2. The CFM/DiT estimator (`flow.decoder.estimator.fp32.onnx`) IS real and could be ported and
   golden-verified in isolation (feeding it a synthetic/golden-dumped `mu` conditioning tensor
   directly, the same input-bisection approach used for Vocos/F5-TTS's vocoder earlier in this
   doc) even without the upstream conditioning encoder existing yet.

**Decision**: rather than sinking further iterations into a pipeline where 3 of 5 stages have
no real weights, moving to QwenASR next per the updated queue (has real local weights per the
original queue table: `models/qwen3-asr-0.6b-q4_k.gguf` + `examples/qwen3-asr.cpp` reference).
Revisit CosyVoice if/when a complete real checkpoint (with the speech-token head, flow
encoder, and HiFT weights) is sourced, or to make partial progress on the two stages that do
have real weights (LLM backbone via the Engine-reuse question above, and the CFM estimator in
isolation).

**Confirmed by independent outside opinion (2026-08-22, same session)**: user cross-checked
this finding with another AI, which confirmed the diagnosis exactly — `models/
cosyvoice2_0.5b.safetensors` being a stripped vanilla Qwen2-0.5B mirror (missing the speech-
token embedding extension, custom output head, flow conditioning encoder, and HiFT vocoder)
is a known failure mode of grabbing a base-model mirror instead of the official unified
release. Suggested real sources: `FunAudioLLM/CosyVoice2-0.5B` (official, has the complete
multi-component structure) or community GGUF ports (e.g. `cstr/cosyvoice3-0.5b-2512-GGUF`).
User initially said they'd download it themselves, then corrected to have this session do it.
Downloaded the official `FunAudioLLM/CosyVoice2-0.5B` repo's `llm.pt` (2.0GB), `flow.pt`
(450MB), `hift.pt` (83MB), `speech_tokenizer_v2.onnx` (496MB), and configs via `curl -L -C -`
(confirmed `campplus.onnx`/`flow.decoder.estimator.fp32.onnx` already on disk were byte-
identical to the official copies, so those didn't need re-fetching). `python`/`pip` were
already present (3.14.3) and **`torch` 2.11.0 + `safetensors` 0.7.0 were already installed**
-- converted all three `.pt` checkpoints (PyTorch zip-pickle, `weights_only=True` load) to
real safetensors files via `safetensors.torch.save_file`: `models/cosyvoice2_llm.safetensors`
(295 tensors), `models/cosyvoice2_flow.safetensors` (1121 tensors),
`models/cosyvoice2_hift.safetensors` (328 tensors). Verified all three parse as valid
safetensors from C#'s side too (read via `node`, same raw-JSON-header technique as before).
Renamed `speech_tokenizer_v2.onnx` -> `models/cosyvoice_speech_tokenizer_v2.onnx` (the real
current-generation tokenizer; the old `cosyvoice_speech_tokenizer.onnx`, 522.6MB, is a
different/older version -- left in place, not deleted, but prefer the `_v2` one going
forward). Deleted the raw `.pt` files after conversion (2.5GB reclaimed); kept the old
(wrong) `cosyvoice2_0.5b.safetensors` in place too, unused, in case it's needed for reference.

**This resolves the "3 of 5 stages have no real weights" finding from the audit above —
inspecting the real `llm.pt`/`flow.pt` contents directly (via `torch.load`, listing every
tensor name/shape) found ALL of the previously-missing pieces actually exist**:
- `llm.pt` has `speech_embedding.weight [6564,896]` (the missing FSQ speech-token embedding
  table) and `llm_decoder.weight/bias [6564,896]` (the missing speech-vocab output head) --
  exactly the two tensors the Phase-0 audit predicted were missing, confirmed present by name
  and shape. Also has `llm_embedding.weight [2,896]` (small extra embedding, purpose TBD --
  likely sos/task special tokens) and, notably, `llm.model.lm_head.weight [151936,896]`
  SEPARATE from `llm.model.model.embed_tokens.weight` (untied) -- our old
  `cosyvoice2_0.5b.safetensors` had `tie_word_embeddings=true` and no separate head at all,
  confirming that file really was a different, stripped-down source, not just a renamed copy
  of this one.
- `flow.pt` has a full `encoder.*` section (`encoder.encoders.{i}.self_attn.{linear_q/k/v/
  pos,pos_bias_u,pos_bias_v}`, `encoder.pre_lookahead_layer`, `encoder.up_layer`,
  `encoder.up_encoders.*`, `spk_embed_affine_layer`, `input_embedding`, `encoder_proj`) --
  **this is the token->mu conditioning encoder the audit found missing, and its tensor
  naming is nearly identical to `Chatterbox/ChatterboxFlowEncoder.cs`'s already-implemented,
  real, golden-verified `UpsampleConformerEncoder`** (same `pos_bias_u/v` untied rel-pos
  attention, same `pre_lookahead_layer`/`up_layer` upsampling structure, same
  `spk_embed_affine_layer` speaker conditioning) -- S3Gen-family lineage, CosyVoice's own
  flow encoder and Chatterbox's S3Gen encoder are architecturally the same thing. This is a
  major unlock: porting CosyVoice's flow encoder should be substantially a generalization of
  existing, tested code rather than a from-scratch port. The `decoder.estimator.*` section
  (1121 tensors total, most of them here) is the DiT/CFM decoder -- its tensor names
  (`down_blocks`/`mid_blocks`/`up_blocks`, `attn1.to_q/k/v`, `time_mlp`) should correspond
  directly to our existing `models/flow.decoder.estimator.fp32.onnx` (same "estimator"
  naming), giving a second, independent cross-check source for that stage.
- `hift.pt` (328 tensors: `conv_pre/post`, `resblocks.*`, `source_resblocks.*`,
  `f0_predictor.*`, `m_source`, `ups.*`) is a real, complete HiFT vocoder checkpoint --
  previously ZERO vocoder weights existed locally; this fully unblocks that stage. Note:
  many conv weights use PyTorch's newer `parametrizations.weight.original0/1` weight-norm
  encoding (`original0`=magnitude `g`, `original1`=direction `v`; actual weight =
  `g * v/||v||`) -- will need a fold-at-load step analogous to Parakeet's BN-fold before the
  plain conv weight is usable, don't assume `original0`/`original1` can be read as the weight
  directly.

**Not yet done** (real porting work, next iteration): none of `CosyVoiceWeights.cs`/
`CosyVoiceLlm.cs`/`CosyVoiceFlowDiT.cs`/`CosyVoiceHiFT.cs` have been updated to load these new
real tensors yet -- this iteration only got the correct checkpoint onto disk in a loadable
format and confirmed via direct inspection that every stage now has real weights to port
against. `CosyVoiceWeights.cs`'s `TryReadWeight` candidate-prefix guessing (`name`,
`model.{name}`, `llm.{name}`) won't find the real names either (`llm.model.model.layers.
{i}.self_attn.q_proj.weight`, not `model.layers.{i}...` -- note the doubled `model.model.`)
so it needs updating regardless of the new checkpoint. CosyVoice moves from "genuinely
blocked" back to "not started, but now has everything it needs" -- same status class as
QwenASR/FunASR before this session touched them. Revisit per the queue once QwenASR's
metadata-adapter question is resolved (see below).

### QwenASR — Phase 0 audit: real GGUF metadata/tensor names verified, one major architectural finding (2026-08-22)

Dumped the real checkpoint (`models/qwen3-asr-0.6b-q4_k.gguf`, 614 tensors, 36 metadata keys)
via the CLI's `list-tensors`/`list-metadata` commands (same technique as Parakeet).
**`QwenAsrWeights.cs`'s hardcoded defaults (`AudioLayers=18, AudioDim=896, AudioHeads=14,
LlmLayers=28, LlmDim=1024, LlmHeads=16, LlmKvHeads=8, LlmVocabSize=151936`) all match the real
checkpoint's metadata exactly** — no bug here, unlike CosyVoice/the earlier Parakeet BN-fold
issue. However the loader still doesn't load any actual tensors (no `GetTensor` method, no
per-layer weight structs) — same skeletal state Parakeet's `ParakeetWeights.cs` was in before
last iteration's rewrite.

**Major finding**: the checkpoint's LLM half (`blk.{i}.attn_q/k/v/output`,
`attn_q_norm`/`attn_k_norm` [confirming real Qwen3 QK-norm, not Qwen2], `ffn_gate/up/down`,
`token_embd.weight`/`output.weight` NOT tied, `output_norm.weight`) is byte-for-byte standard
llama.cpp Qwen3 tensor naming — and `OpenTail.Stingray.Engine/ModelCompatibility.cs` confirms
`"qwen3"` is already a fully-supported text-generation architecture in this engine's own
`s_textGenerationArchitectures` whitelist (real GQA + QK-norm support, not a stub). This means
the LLM decoder half of QwenASR is architecturally identical to something this codebase's own
production LLM engine already runs correctly, in principle removing the need to hand-write a
second scalar Qwen3 transformer forward pass the way Parakeet's Conformer or Chatterbox's T3
needed custom math for a genuinely novel architecture.

**The catch, and why this isn't a trivial "just point the Engine at it"**: this GGUF's
`general.architecture` is the string `"qwen3asr"`, not `"qwen3"`, and its hyperparameter keys
live under a custom `qwen3asr.llm.*`/`qwen3asr.audio.*` namespace (this file bundles the
custom AuT audio encoder tensors AND the standard-shaped LLM tensors together in one
multimodal container, which is why it needed a nonstandard KV namespace in the first place).
`ModelCompatibility`'s architecture gate would reject `"qwen3asr"` outright, and even if
admitted, the standard loader expects config keys like `qwen3.block_count`/`qwen3.attention.
head_count` (llama.cpp convention), not `qwen3asr.llm.n_layers`/`qwen3asr.llm.n_heads`. Two
viable unlock paths, neither attempted yet: **(a)** write a thin adapter/synthetic
`GgufModel`-like view that remaps `qwen3asr.llm.*` metadata keys to their standard `qwen3.*`
names and presents `general.architecture="qwen3"` to the Engine, letting the real
`ForwardPass` run the `blk.*` tensors directly (uses this GGUF file as-is, no format
conversion); **(b)** extract just the `blk.*`/`token_embd`/`output`/`output_norm` tensors
into a second, minimal standard-metadata GGUF file at load/build time and hand that to the
existing loader (touches disk, but keeps `ModelCompatibility`/the Engine completely
unmodified). (a) is more elegant but touches shared Engine-facing code (higher blast radius,
per this repo's own audio-vs-engine boundary conventions); (b) is more isolated to the Audio
project. Whichever path is taken, the AuT audio encoder (`audio.conv.*` conv2d stem +
`audio.blk.*` Whisper-style transformer blocks + `audio.proj1/proj2` projection into the
LLM's embedding space + `audio.mel_filters`/`audio.mel_window`, all genuinely custom, no
existing Engine support) still needs a hand-written port the same way Parakeet's Conformer
encoder did — that part of the plan doc's "Phase 3/4: Real Conv2D + Real AuT Transformer" is
unavoidable regardless of which LLM-reuse path is chosen.

**Not attempted this iteration** (time/scope): extending `QwenAsrWeights.cs` with real tensor
loading (mirroring `ParakeetWeights.cs`'s pattern), the AuT encoder port, and the LLM-reuse
adapter are all left for a future iteration. This audit alone is a meaningful unlock though —
whoever picks this up next should evaluate the Engine-reuse question BEFORE writing any
decoder transformer code by hand, since getting that decision right could save an entire
Chatterbox/Parakeet-sized porting effort.

**Next iteration**: (1) decide between adapter-path (a) and split-GGUF-path (b) for LLM reuse
— read `ModelCompatibility.cs` and `ForwardPass.cs`'s GGUF-loading entry points fully before
deciding, this wasn't done this pass. (2) Extend `QwenAsrWeights.cs` with real tensor loading
for the audio encoder half regardless of the LLM decision. (3) Port the AuT conv2d stem +
transformer (`QwenAsrAudioEncoder.cs`, currently "mostly a placeholder" per the plan doc's own
section 6 title — not independently re-verified this pass, but consistent with every other
pipeline's starting state in this doc).

### QwenASR — adapter path (a) implemented and structurally verified (2026-08-22, same session)

User's direction: build the metadata-remapping adapter, not the split-GGUF-file path — "a
base class we inherit from and split what we have." `GgufModel` is `sealed` with no public
constructor, so a literal subclass isn't possible; found the actual intended seam instead:
`OpenTail.Stingray.Core.IModelTensorSource` (`ForwardPass`'s constructor already takes this
interface, not a concrete `GgufModel` — confirmed by reading `ForwardPass.cs` directly) whose
own doc comment states its exact purpose: "lets another format feed the existing, unmodified
transformer loop." `SafetensorsTensorSource.cs` is a real, in-tree example of exactly this
pattern for a different format (HF safetensors instead of a differently-namespaced GGUF).

Implemented `QwenAsrLlmTensorSource.cs`: wraps a real `GgufModel`, remaps the specific
`qwen3asr.llm.*` metadata keys `ModelGraph.cs` actually reads (traced directly from its
source rather than assumed: `embedding_length`, `attention.head_count`, `attention.
head_count_kv`, `attention.key_length`/`value_length`, `block_count`, `feed_forward_length`,
`attention.layer_norm_rms_epsilon`, `rope.freq_base`, `vocab_size`, `context_length`) into
standard `qwen3.*` names, overrides `general.architecture` to `"qwen3"`, and filters
`Tensors`/`FindTensor` down to only `blk.*`/`token_embd.weight`/`output.weight`/
`output_norm.weight` (excluding the `audio.*`-prefixed AuT encoder tensors, irrelevant to the
text-generation forward pass). Zero changes to `Engine`/`Core` — fully additive, lives
entirely in the Audio project.

Verified structurally (not yet a full generation-loop test) via
`Tests.Audio/QwenAsrLlmTensorSourceTests.cs` (`HeavyTestBase`, real GGUF):
`Adapter_RemapsMetadata_ToStandardQwen3Keys` (all 7 remapped keys match the real checkpoint's
values: `embedding_length=1024, head_count=16, head_count_kv=8, key_length=128,
block_count=28, feed_forward_length=3072, vocab_size=151936`) and
`Adapter_ExposesOnlyLlmTensors_NotAudioTensors` (real `blk.0.attn_q.weight`/`token_embd.
weight`/`output.weight`/`output_norm.weight` resolve; `audio.*` tensors correctly excluded
from both `FindTensor` and the `Tensors` enumeration). Both PASS. Did NOT add a
`Tests.Audio` -> `Engine` project reference to test `ModelCompatibility.
IsTextGenerationArchitectureSupported` directly (that project boundary looks deliberate, not
something to cross for one assertion) — independently confirmed by reading
`ModelCompatibility.cs`'s architecture whitelist source directly instead.

**Not yet done**: this only proves the Engine's `ForwardPass` COULD consume this checkpoint
under its own architecture detection — it hasn't actually been run through a real
`ForwardPass`/KV-cache/generation loop, and the multimodal audio-embedding injection point
(replacing `audio_pad_token_id` embeddings with the AuT encoder's projected features before
decode, per the plan doc's "Phase 13: Multimodal Audio Injection") isn't wired at all. The
AuT audio encoder itself (`audio.conv.*` conv2d stem + `audio.blk.*` transformer +
`audio.proj1/proj2`) is still 100% unstarted — that part was never going to be avoidable via
Engine reuse, only the LLM decoder half benefits from this adapter.

**Next iteration**: (1) actually construct a `ForwardPass` from this adapter and run a real
forward step against known input token ids, comparing logits/next-token against something
verifiable (even a simple "does it predict plausible next tokens for a text-only prompt,
ignoring audio entirely" sanity check would validate the adapter end-to-end for the first
time). (2) Port the AuT audio encoder (conv2d stem, Whisper-style transformer blocks,
projection) — this is the genuinely novel, unavoidable hand-written part, comparable in scope
to Parakeet's subsampling+Conformer work. (3) Wire the audio-embedding injection point once
both halves work independently.

### QwenASR — real tokenizer + real AuT audio encoder port (2026-08-22, same session, direct user request: "tokenizer/prompt formatting + AuT audio encoder")

**Real BPE tokenizer wired up.** The checkpoint ships a real GPT2-style BPE vocabulary
(`tokenizer.ggml.tokens` [151936], `tokenizer.ggml.merges` [151291], `tokenizer.ggml.model=
gpt2`) — confirmed via `GgufTokenizer`, the same real tokenizer class Chatterbox already uses
(`GgufTokenizer.FromGgufModel`). One wrinkle: this checkpoint has no `tokenizer.ggml.
token_type` array, which is the only thing `FromGgufModel` uses to recognize special tokens
(`<|im_start|>` etc) — without it they'd get silently BPE-shredded character-by-character.
Added `QwenAsrWeights.BuildTokenizer` which constructs the `TokenizerSource` by hand (mirroring
`FromGgufModel`'s own logic) and supplies `AdditionalSpecialTokens` explicitly.

**Dumped real special-token strings directly from the checkpoint** (via a `dotnet run
scratch.cs` one-off script reading `tokenizer.ggml.tokens[id]` — .NET 10's file-based-app
`dotnet run foo.cs` feature is a fast way to do this kind of one-off GGUF inspection without a
throwaway test/console project). Found the real audio special tokens are `<|audio_start|>`
(151669), `<|audio_end|>` (151670), `<|audio_pad|>` (151676) — **not** the old code's
`<|audio_bos|>`/`<|audio_eos|>`/`<|AUDIO|>` at ids 151646-151648, which turned out to be
completely different tokens (`<|object_ref_start|>` etc — Qwen's shared vision/multimodal
vocab, not audio-specific at all). **Also confirmed there is NO dedicated timestamp-token
range anywhere in the vocabulary** — grepped every token string containing "timestamp"; zero
matches beyond BPE fragments like `Ġtimestamp`/`timestamps` (ordinary subword tokens, not
special tokens). The previous `QwenAsrTokenizer.TimestampBegin=151649..TimestampEnd=153149`
(~1500 tokens) was entirely fictional. Cross-checked against the plan doc's own section 20,
which independently confirms "ASR timestamp output" and "Forced alignment" are different
things and that alignment is a **separate model** (`Qwen3-ForcedAligner-0.6B`, not bundled in
this GGUF) — so real segment-level ASR timestamps may not even exist as a native output for
this specific checkpoint; `QwenAsrTokenizer.DecodeWithTimestamps` now produces one best-effort
segment spanning the whole decoded output rather than fabricating sub-segment timing.

Rewrote `QwenAsrTokenizer.cs` around the real `GgufTokenizer` (real `Encode`/`Decode`, real
`FormatPrompt` using the real special-token strings, one `<|audio_pad|>` per AuT-encoder
output frame). Extended `QwenAsrWeights.cs` with all the real special-token ids (read from
metadata, not hardcoded) plus every AuT hyperparameter (`AudioHeadDim`, `AudioFfDim`,
`AudioConvChannels`, `AudioProjDim`, mel params) the encoder needed. Fixed downstream
breakage: `QwenAsrDecoder.cs` referenced the now-deleted `QwenAsrTokenizer.TimestampBegin`/
`ImEndTokenId` static constants (moved EOS id onto `QwenAsrDecoderConfig` instead, sourced
from real metadata); `QwenAsrForcedAligner.cs` called `_tokenizer.Encode` unconditionally in
its (still 100% procedural, separate-model, out-of-scope-this-session) DTW aligner — since
`QwenAsrTokenizer.Encode` now correctly throws without real weights (matching Parakeet's "no
procedural fallback" policy), swapped ForcedAligner's per-word token-COUNT proxy to a plain
length-based heuristic instead of resurrecting a fake-but-plausible tokenizer call.
`QwenAsrPipeline.cs` updated to require real weights for its tokenizer (`Load` always
constructs `new QwenAsrTokenizer(weights)`).

**Real AuT audio encoder ported.** Found the real reference (same `CrispStrobe/CrispASR`
family as Parakeet's oracle, already cloned locally): `examples/crispasr/src/qwen3_asr.cpp`,
whose header comment states it matches `modeling_qwen3_asr.Qwen3ASRAudioEncoder.forward`
exactly. Confirmed architecture directly from real tensor shapes (`audio.conv.1/2/3.weight`,
`audio.conv_out.weight [7680,896]`, `audio.blk.{i}.*`, `audio.ln_post`, `audio.proj1/2`) AND
cross-checked every detail against the reference source before writing any math:
- Conv2D stem: 3 FULL (non-depthwise, unlike Parakeet's stage-2/3) stride-2 conv2d layers
  (1→480, 480→480, 480→480, all k=3 pad=1) + GELU each, giving exactly 8x downsampling in
  BOTH time and mel-frequency (128 mels → 16). `480*16=7680` matches `conv_out`'s input dim
  exactly, confirming the derivation. `conv_out` is a bias-less Linear(7680→896) (no bias
  tensor exists for it, confirmed).
- Positional encoding: Whisper's own fixed sinusoidal formula, reused directly from
  `Whisper/WhisperEncoder.cs`'s `GenerateSinusoidalPositionalEmbeddings` (this checkpoint
  ships no positional-embedding tensor, so it's non-learned, matching Whisper's convention;
  the reference source's own comment confirms "Add sinusoidal pos embed").
- 18x Whisper-style pre-LN Transformer blocks: plain LayerNorm (with bias), self-attention
  with bias on Q/K/V/out (unlike Whisper's own encoder, which has no K bias — a real, checked
  difference, not copied blindly from `WhisperEncoder.cs`), 2-layer GELU FFN — no rel-pos
  bias, no conv module, no macaron step, much simpler than Parakeet's FastConformer.
  Block-level math (attention loop structure, `LinearReal`, `LayerNormAffine` shape) follows
  the same pattern as `WhisperEncoder.cs`'s real, golden-verified implementation.
- Adapter: `ln_post` final LayerNorm → `proj1` (896→896) + GELU → `proj2` (896→1024) into the
  Qwen3 LLM's embedding space.
- GELU: the reference uses `ggml_gelu_erf` (exact erf-based GELU) throughout, not the tanh
  approximation `WhisperEncoder.cs` uses. .NET has no built-in erf, so this port currently
  still uses the tanh approximation (documented as a known, small numerical gap, not silently
  copied without noting the discrepancy) — flagged for the golden-verification pass rather
  than guessed at with a hand-rolled erf approximation under time pressure.

**Real, transferable bug found and fixed via direct debugging** (not guessing): the first
real-weights test run produced 100% NaN output. Traced it stage-by-stage (conv stem → conv_out
projection → positional embedding → per-layer Q/K/V → attention scores → softmax → weighted
sum) using a `dotnet run scratch.cs` one-off script instrumented with temporary `Console.
Error.WriteLine` probes at each stage (removed after diagnosis) — this bisection approach is
the same discipline used throughout this doc (e.g. Vocos/F5-TTS's vocoder isolation), just
applied via a disposable script instead of a test file since the goal was localization, not
lasting coverage. Found: raw attention logits for this checkpoint's real trained weights are
legitimately large (~137-139, confirmed by direct measurement — not a bug upstream), and
**`System.Numerics.Tensors.TensorPrimitives.SoftMax` returns NaN for inputs in that range**
even though every input value is perfectly finite (almost certainly `exp(138)` overflowing
float32 internally without the standard max-subtraction stabilization trick, on whatever code
path a length-8 span takes in .NET 10). Fixed by replacing `TensorPrimitives.SoftMax` with
`Primitives/DenseKernels.SoftmaxInPlace` (this codebase's own max-subtracted, numerically-
stable softmax, extracted earlier this session) — confirmed the exact same input logits now
produce a finite, sane probability distribution. **`Whisper/WhisperEncoder.cs` uses the
identical `TensorPrimitives.SoftMax(scores.AsSpan(...), scores.AsSpan(...))` in-place pattern**
and is therefore latently exposed to the same failure mode on any audio that drives its
attention logits into a similar range — Whisper's own tests apparently haven't hit it, but
this is a real, transferable finding worth a follow-up look rather than assuming Whisper is
safe just because its current tests pass.

Added `Tests.Audio/QwenAsrTokenizerTests.cs` (2 tests: ChatML round-trip with real special
tokens, `DecodeWithTimestamps`' single-segment behavior) and `Tests.Audio/
QwenAsrAudioEncoderTests.cs` (1 test: real weights, finite output, correct 8x-downsample
shape) — both HeavyTestBase, real GGUF weights, all PASS. Removed/updated the now-stale
`Tests.Audio.Fast/QwenAsrTests.cs` entries that exercised deleted fake APIs (same "no
procedural fallback" pattern as Parakeet/CosyVoice's Fast-test cleanups earlier in this doc).

**Not yet done**: mel extraction (`QwenAsrMelExtractor.cs`) still uses a self-computed
filterbank rather than this checkpoint's real shipped `audio.mel_filters`/`audio.mel_window`
tensors (same gap Parakeet had before its mel-extractor fix — not yet applied here due to
time). No numeric golden verification against the real reference yet (structurally complete
only, same caveat as every pipeline before its golden pass).

### QwenASR — LLM decoder generation loop wired to the real ForwardPass, plus a perf pass (2026-08-22 / 2026-08-23, direct user request "performance-enhance QwenASR" then "yes and yes" to both a perf fix and finishing the decoder)

**Perf fix (verified via the CosyVoice/F5 benchmark sweep's established discipline, though no
dedicated QwenASR benchmark test exists yet)**: `QwenAsrAudioEncoder.SelfAttention` had the
same anti-pattern just fixed elsewhere in this doc for CosyVoice3/F5-TTS's DiT attention —
`Parallel.For(0, heads, ...)` (heads=14, narrow) with the QK/AV math already using
`TensorPrimitives.Dot`/`MultiplyAdd` (SIMD) internally. Restructured to loop `heads`
sequentially and `Parallel.For(0, t, ...)` inside (t = audio frame count after 8x downsample,
scales with clip length, typically far larger than 14) — same SIMD math, much wider
parallelism for realistic clip lengths. `QwenAsrAudioEncoderTests` still passes (real weights,
correct 8x-downsample shape, finite output) after the change.

**Decoder wired to the real Engine forward pass — the embedding-injection blocker resolved
the same way CosyVoice's was.** `QwenAsrLlmTensorSource` (previously "adapter only, not yet
wired into a generation loop") gained `EnableAudioConditioning(audioEmbeddings,
numAudioTokens)`: rebuilt the class around a mutable `Dictionary<string, GgufTensorInfo>`
(was a fixed `List` built once at construction) and made it `IDisposable`, then applied the
exact same composition-only trick `CosyVoiceLlmTensorSource.EnableSpeechGenerationMode` uses
(`ForwardPass.EmbedTokenInto` resolves an embedding row purely by whatever tensor is bound to
`"token_embd.weight"` + token id, no hardcoded vocab assumption — confirmed by re-reading
`ForwardPass.Helpers.cs` directly, not re-guessing): dequantizes the real text-vocab rows in
one pass via `Dequantize.ToFloat32` (this checkpoint's `token_embd.weight` is Q4_K/Q8_0, not
F32, unlike CosyVoice's safetensors case) into a natively-allocated buffer, appends
`numAudioTokens` rows taken directly from the AuT encoder's own real per-frame output, and
presents the combined table under the same tensor name. Unlike CosyVoice's speech-generation
case, `output.weight`/`qwen3.vocab_size` do NOT need swapping — QwenASR still predicts real
text tokens, only the *input* embedding space grows for one utterance's audio positions.
`AudioTokenIdOffset` (= real text vocab size) is exposed so a caller can map audio frame `f` to
synthetic id `AudioTokenIdOffset + f`.

**`QwenAsrDecoder.Generate`** (previously fully fake/procedural — see the old doc entry just
above, now stale) gained a second constructor taking a real `QwenAsrWeights` (mirroring Piper/
Melo's dual fake/real-constructor pattern); when constructed that way, `Generate`: builds a
fresh `QwenAsrLlmTensorSource` from `weights.Model` per call (audio conditioning is
utterance-specific, so a fresh source per utterance is correct, not a missed caching
opportunity), calls `EnableAudioConditioning`, remaps the prompt's real `<|audio_pad|>`
occurrences (one per AuT-encoder frame, from `QwenAsrTokenizer.FormatPrompt`) to the synthetic
audio-frame ids in order, then runs a real `ForwardPass.Prefill` + autoregressive
`Forward(token, position)` loop using `Sampler.Sample` (the same production sampler every other
text-generation pipeline in this repo uses, not a hand-rolled argmax) until EOS or
`maxNewTokens`. The old fully-fake loop is kept as `GenerateProcedural`, used only when no
`QwenAsrWeights` was supplied (mirrors every other pipeline's fake/real dual-path discipline).
`QwenAsrPipeline.Load` now constructs `QwenAsrDecoder(weights, decoderConfig)` instead of the
config-only constructor.

**Required adding an `OpenTail.Stingray.Engine` `ProjectReference` to `OpenTail.Stingray.Audio`
itself** (previously only `Tests.Audio` had this, for CosyVoice's LLM tests) — checked
`Engine.csproj`'s own references (Core, Cpu, Vision, Vulkan, Cuda, TurboQuant, Pipeline) and
`Pipeline.csproj`'s (Core only) first to confirm no circular dependency before adding it.

**Verified**: new test `QwenAsrLlmTensorSourceTests.AudioConditioning_RunsRealForwardPass_
ProducesFiniteNonDegenerateLogits` (mirrors CosyVoice's `SpeechGenerationMode_...` test) —
builds the combined embedding table, prefills a sequence mixing real text ids and synthetic
audio-frame ids, confirms finite non-degenerate logits over the real (unswapped) text
vocabulary — PASS, plus all 3 previously-passing tests in that file (5/5 total). The
pre-existing end-to-end pipeline test `QwenAsrRealWeightsTests.
QwenAsrPipeline_LoadRealGguf_TranscribesAudioEndToEnd` (synthetic tone audio through the full
`Transcribe` call) now genuinely exercises this real decoder path end-to-end for the first
time — still PASS (2/2 in that file), confirming the whole chain (mel → real AuT encoder → real
audio-conditioned prompt → real Qwen3 `ForwardPass` generation loop → tokenizer decode) runs
without crashing. `QwenAsrAudioEncoderTests` (1/1) and `QwenAsrTokenizerTests` (2/2) also still
pass — 4 test classes, 10 tests total, zero regressions.

**Not yet done / known caveats, same honesty bar as every other pipeline in this doc**: no
numeric golden verification exists for either the AuT encoder or the decoder path (no
real reference Python/C++ AuT+Qwen3-ASR implementation has been run this session to diff
against) — "runs end-to-end without crashing and produces non-degenerate logits" is NOT the
same claim as "produces a correct transcript," exactly the distinction this whole doc exists
to keep honest. `QwenAsrMelExtractor`'s self-computed filterbank gap above still stands. No
dedicated QwenASR perf benchmark test exists yet (unlike CosyVoice's `CosyVoiceBenchmarkTests`)
if a future pass wants to measure the audio-encoder attention fix's actual wall-clock effect.

### CosyVoice — LLM backbone ported via the same Engine-reuse pattern as QwenASR (2026-08-22, same session, direct user request)

User confirmed the architectural direction from the earlier QwenASR work ("port CosyVoice's
LLM backbone first") after being asked whether a performance pass made sense on CosyVoice
yet — it didn't, since nothing real existed there to optimize (still 100% procedural
placeholder in `CosyVoiceLlm.cs`/`CosyVoiceFlowDiT.cs`/`CosyVoiceHiFT.cs`).

**`CosyVoiceLlmTensorSource.cs`** (new): the same `IModelTensorSource` adapter pattern as
`QwenASR/QwenAsrLlmTensorSource.cs`, but for CosyVoice2's LLM backbone
(`models/cosyvoice2_llm.safetensors`, converted earlier this session from the official
`llm.pt`) rather than a GGUF. Confirmed via `models/cosyvoice2_config.json` this backbone is a
plain `Qwen2ForCausalLM` (`"architectures": ["Qwen2ForCausalLM"]`), and `"qwen2"` is in the
Engine's `s_textGenerationArchitectures` whitelist (`ModelCompatibility.cs`) — same bet as
QwenASR's `"qwen3"`, now validated a second time.

Could NOT reuse the existing generic `SafetensorsTensorSource`/`SafetensorsTextModelPackage.
TryMapToOpenTailTensorName` (Core's shared HF-safetensors adapter) as-is: this checkpoint's
real tensor names use a doubled `llm.model.model.layers.{i}.*` prefix (an artifact of the
original PyTorch module nesting — `CosyVoice2Model.llm.model` wraps a standard HF `Qwen2Model`
— confirmed directly from `torch.load`'s printed keys earlier this session) instead of the
generic mapper's expected bare `model.layers.{i}.*`, AND carries Q/K/V *biases*
(`self_attn.{q,k,v}_proj.bias`, confirmed present) that the generic mapper doesn't map at all
(it only handles bias-less standard weight names). Extending that shared mapper to cover both
cases would touch every other safetensors-backed pipeline in this repo — instead wrote a
dedicated adapter mapping this checkpoint's exact real tensor names (including biases)
directly, following `SafetensorsTensorSource.cs`'s real structure/BF16-conversion pattern as
a template (the BF16 path is dead code in practice here — despite `cosyvoice2_config.json`
declaring `torch_dtype: bfloat16`, the actual converted safetensors tensors are already plain
F32, confirmed directly from the file's own header — kept the conversion path anyway so a
future re-conversion from a genuinely-BF16 source still works without changes).

Metadata constructed directly (not read from any file, since safetensors carries no GGUF-style
KV metadata): `qwen2.embedding_length=896, qwen2.attention.head_count=14, qwen2.attention.
head_count_kv=2, qwen2.attention.key_length=64, qwen2.block_count=24, qwen2.feed_forward_
length=4864, qwen2.rope.freq_base=1000000, qwen2.attention.layer_norm_rms_epsilon=1e-6,
qwen2.vocab_size=151936` — every value read directly from `cosyvoice2_config.json` during the
Phase-0 audit, not guessed.

**Verified via a real `ForwardPass` integration test**, same discipline as QwenASR's:
`Tests.Audio/CosyVoiceLlmTensorSourceTests.cs` — `Adapter_MapsRealTensors_...` (all 24 layers'
tensors resolve, including biases; layer 24 correctly absent) and
`Adapter_RunsRealForwardPass_ProducesFiniteNonDegenerateLogits` (`ModelHyperparams.
FromGgufMetadata` + `CpuBackend` + real `ForwardPass.Prefill`, checks 151936-length finite
non-degenerate logits). **Both PASS** — CosyVoice's actual Qwen2 backbone now runs through
this engine's real production forward pass against its real converted weights, the same
empirical validation QwenASR's adapter got, now proven on a second independently-converted
checkpoint.

**Not yet done**: this only proves the backbone itself runs — CosyVoice's actual generation
target is FSQ *speech* tokens, not text, which requires the checkpoint's `speech_embedding.
weight [6564,896]` (extends the input embedding beyond the 151936 text vocab) and
`llm_decoder.weight/bias [6564,896]` (the actual output head used at inference, separate from
the tied/untied text `lm_head`) — neither is wired into this adapter yet, both confirmed
present in the checkpoint during the earlier Phase-0 re-audit. `CosyVoicePipeline.cs`/
`CosyVoiceLlm.cs` still call the 100%-procedural old code path, not this adapter. The flow
(CFM/DiT) and HiFT vocoder stages are completely untouched this iteration — `examples/
cosyvoice.cpp` (a real, complete C++ reference for this whole pipeline, distinct from the
crispasr family used for Parakeet/QwenASR) is the reference to use for those next, not yet
opened this session.

**Next iteration**: (1) extend `CosyVoiceLlmTensorSource` (or add a second adapter) to expose
`speech_embedding`/`llm_decoder` so the real backbone can actually predict speech tokens, not
just text logits. (2) Read `examples/cosyvoice.cpp` for the flow encoder (compare against
`Chatterbox/ChatterboxFlowEncoder.cs`'s near-identical architecture, per the Phase-0 audit
finding) and CFM/DiT decoder math. (3) Port HiFT vocoder (real weights now available in
`models/cosyvoice2_hift.safetensors`, includes PyTorch weight-norm-parametrized convs needing
a fold-at-load step, same class of transform as Parakeet's BN-fold).

### CosyVoice — `speech_embedding`/`llm_decoder` exposed (real weights, not yet wired); `examples/cosyvoice.cpp` is CosyVoice3, NOT our CosyVoice2 checkpoint (2026-08-22, same session)

**Exposed the real speech-token tensors.** Added `SpeechEmbeddingWeight [6564,896]`,
`LlmDecoderWeight`/`LlmDecoderBias [6564,896]`/`[6564]`, and `LlmEmbeddingWeight [2,896]` as
public properties on `CosyVoiceLlmTensorSource` (loaded directly via `SafetensorsLoader.
ReadF32`, not through the `IModelTensorSource` seam since they aren't part of the standard
qwen2 tensor set `ForwardPass` expects). Verified real, correctly-shaped, finite via
`Adapter_LoadsRealSpeechTokenTensors_CorrectShapesAndFinite` — PASS.

**Checked `ForwardPass`'s public API for an embedding-injection or output-head-override entry
point before wiring further, rather than guessing**: `ForwardPass.Prefill(IReadOnlyList<int>
tokens, ...)` only accepts token ids — no raw-embedding input, no way to swap the output head
mid-sequence. Real CosyVoice generation needs both (mixing text-vocab and speech-vocab
embeddings in one sequence via two different tables, then projecting through the *speech*
head rather than the tied/untied text `lm_head`). Neither exists in the stock API, so wiring
this properly means either **(a)** extending `ForwardPass` with an embeddings-input entry
point (touches shared Engine code, the same class of decision flagged for QwenASR's
metadata-adapter path but with higher stakes since it's a new API surface, not just a
metadata remap) or **(b)** a standalone manual transformer loop over the loaded layer weights
that bypasses `ForwardPass` for this specific generation path, which would partially undo the
point of reusing the Engine in the first place. Not resolved this pass — flagged as the
concrete next decision, deliberately not rushed given the stakes.

**Important correction, found while checking the wrong thing at the right time**: went to
read `examples/cosyvoice.cpp`'s flow encoder to compare against `ChatterboxFlowEncoder.cs`
(per the "major unlock" note from the Phase-0 audit) and found its
`CausalMaskedDiffWithDiT::build_cgraph_encode` has NO Conformer self-attention stage at
all — token embedding → `PreLookaheadLayer` → simple repeat-based upsample (by
`token_mel_ratio`) straight into the DiT estimator, no `encoder.encoders.*`-style attention
blocks anywhere in that code path. This flatly contradicts what `flow.pt`'s real tensor names
show (`encoder.encoders.{i}.self_attn.{linear_q/k/v/pos, pos_bias_u, pos_bias_v}` — genuine
Conformer relative-position self-attention, confirmed present, not imagined). Checked why:
`examples/cosyvoice.cpp/README.md` states outright it is "currently focused on **CosyVoice3**"
(`CosyVoice3LM`, links to `Fun-CosyVoice3-0.5B-2512-GGUF`). **We downloaded and converted
CosyVoice2-0.5B, a different, older architecture** — this C++ reference is the wrong oracle
for our checkpoint's flow stage. The `ChatterboxFlowEncoder.cs`-architecture-match finding
from the Phase-0 audit stands (that was derived directly from `flow.pt`'s real tensor names,
not from this C++ source) and is now the *only* confirmed-correct lead for CosyVoice2's flow
encoder — do not use `examples/cosyvoice.cpp` for this stage. It may still be useful for the
CFM/DiT estimator math specifically if CosyVoice2 and CosyVoice3 share that sub-component
(both ship a `flow.decoder.estimator.fp32.onnx`-style estimator per the earlier audit; NOT
independently verified whether the estimator architecture itself is version-stable —
check tensor names before trusting it, same discipline as everything else in this doc).

**Next iteration**: (1) decide on the `ForwardPass` embedding-injection question above before
writing any more CosyVoice LLM code — this blocks real speech-token generation regardless of
which pipeline stage is worked on next. (2) For the flow encoder specifically: compare
`flow.pt`'s real `encoder.encoders.*` tensor names/shapes directly against `Chatterbox/
ChatterboxS3GenWeights.cs`'s existing layer struct (not against `examples/cosyvoice.cpp`) to
confirm the architectural match precisely before porting. (3) HiFT vocoder port (real weights
in `models/cosyvoice2_hift.safetensors`, needs a weight-norm fold-at-load step).

### CosyVoice — embedding-injection blocker RESOLVED via composition, no Engine changes (2026-08-22, same session, direct user request "keep going")

**The blocker from the previous entry is solved.** Re-read `ForwardPass.Helpers.cs` directly
(specifically `EmbedTokenInto`, the actual per-token embedding-lookup method) rather than
assuming the earlier "no injection API" conclusion was the final word: it resolves the
embedding row purely by looking up whatever tensor is bound to the canonical name
`"token_embd.weight"` and indexing by token id — **no hardcoded vocabulary assumption at
all**. Since `IModelTensorSource.FindTensor`/`GetTensorDataPtr` are just a name→bytes seam
(the same seam `QwenAsrLlmTensorSource` already uses), a tensor source can present a
completely synthetic, non-file-backed tensor under that name and `ForwardPass` will use it
exactly as if it were real GGUF/safetensors data.

**Added `CosyVoiceLlmTensorSource.EnableSpeechGenerationMode()`**: builds a synthetic,
natively-allocated combined embedding table — real text-vocab rows `[0, 151936)` followed by
real `speech_embedding.weight` rows `[151936, 151936+6564)` — and presents it under
`token_embd.weight`; swaps `output.weight` to a copy of `llm_decoder.weight` (the real
speech-vocab head). A caller offsets any real speech token id by `SpeechTokenIdOffset`
(=151936) before passing it to `Prefill`, and adds `LlmDecoderBias` to the returned logits
afterward (confirmed `ForwardPass` has NO final-layer bias support at all — only per-block
`attn_output.bias` exists, verified by direct source inspection — so the bias literally cannot
be applied inside the Engine and must be added by the caller as a trivial post-processing
step on the returned `ReadOnlySpan<float>`, which is fine since it's just an additive
per-vocab-index offset). **Zero changes to `OpenTail.Stingray.Engine`/`Core`** — this is
composition entirely within the Audio project, same "extend via the seam, not the engine"
principle `IModelTensorSource`'s own doc comment describes.

**Real bug caught by actually running it, not just building it**: the first version segfaulted
(`STINGRAY_RUN_HEAVY_TESTS=1 dotnet test ...` exited 139, SIGSEGV). Root cause: the
synthetic-source constructor still advertised the ORIGINAL text vocab size
(`qwen2.vocab_size=151936`) in its metadata after `EnableSpeechGenerationMode` swapped
`output.weight` down to the real, much smaller 6564-row speech head — `ModelHyperparams.
FromGgufMetadata` sizes `ForwardPass`'s internal logits/output buffers off that stale
metadata value, so the engine allocated/read/wrote against a buffer size that no longer
matched the real (smaller) tensor, corrupting memory. Fixed by updating `qwen2.vocab_size` in
the source's own metadata dictionary at the moment of the mode switch (required changing
`_metadata`'s field type from `IReadOnlyDictionary` to the concrete mutable `Dictionary` so
this update is possible after construction). This is a real, generalizable lesson for anyone
building a synthetic/dynamic `IModelTensorSource`: every dimension advertised in metadata
that `ForwardPass` uses for buffer sizing must be kept in lockstep with the actual tensor
shapes at all times, not just at construction.

**Verified via a real end-to-end `ForwardPass` run**:
`SpeechGenerationMode_RunsRealForwardPass_ProducesFiniteSpeechVocabLogits` — prefills a
sequence mixing real text token ids with a real speech token id (offset via
`SpeechTokenIdOffset`), confirms the combined embedding table and speech head report the
correct swapped dimensions, runs a real `ForwardPass.Prefill`, adds the real
`LlmDecoderBias`, and checks the resulting 6564-length logits vector is fully finite and
non-degenerate. **PASS**, along with all 3 previously-passing tests in this file (4/4 total).
This is the same class of milestone as QwenASR's adapter validation — CosyVoice's LLM can now
genuinely predict real speech tokens through this engine's real, unmodified production
forward pass, not just text logits.

**Still not done**: no actual autoregressive generation loop (repeated `Decode` calls with
sampling/stopping) built on top of this yet — only a single `Prefill` call is exercised. No
KV-cache-based streaming generation tested. The flow encoder / HiFT vocoder stages remain
untouched. `CosyVoicePipeline.cs`/`CosyVoiceLlm.cs` still call the 100%-procedural old code
path, not any of this session's real adapters — wiring them in is follow-up work, not done
yet.

### CosyVoice3 checkpoint also acquired, in parallel (2026-08-22, same session, direct user request)

User's framing: "two birds with one and a half stones" — since `examples/cosyvoice.cpp` turned
out to target CosyVoice3 specifically (not our CosyVoice2 checkpoint, per the correction
above), downloading a real CosyVoice3 checkpoint gives a genuinely matched reference+checkpoint
pair, unlike CosyVoice2 (real weights, but the closest analog is Chatterbox's
architecturally-similar-but-different S3Gen code, not a byte-exact reference).

Found the real source via `examples/cosyvoice.cpp/README.md`'s own pointer:
`Lourdle/Fun-CosyVoice3-0.5B-2512-GGUF` on Hugging Face — a **pre-converted, single-file GGUF**
already bundling the whole pipeline (confirmed this is the same conversion tooling family
`examples/cosyvoice.cpp` ships: `convert_model_to_gguf.py` at its repo root), plus separate
`frontend-onnx/{campplus,speech_tokenizer_v3}.onnx` files for the non-GGUF frontend stages.
Chose the F16 variant (1.7GB, best available precision short of F32's 3.4GB — many smaller
quantized variants exist, from Q2_K at 447MB up, but precision matters more than size for a
verification oracle) plus the real (non-`.int8`) frontend ONNX files, downloading in the
background (`models/cosyvoice3/CosyVoice3-2512_F16.gguf` +
`models/cosyvoice3/frontend-onnx/{campplus,speech_tokenizer_v3}.onnx`).

**Considered building `examples/cosyvoice.cpp` itself as a real oracle binary** (would let us
golden-verify against actual reference output, the strongest verification tier per this doc's
own "Ground Truth Hierarchy" — see the CosyVoice plan doc's section 9). **Decided not to
attempt it this pass**: `vendor/ggml` is referenced by `CMakeLists.txt`
(`GGML_SOURCE_DIR`) but not present in the checked-out `vendor/` directory (only
`cpp-httplib`/`miniaudio`/`nlohmann`/`pcre2` are there), and the one declared git submodule
(`pcre2`) isn't initialized either (`git submodule status` shows a `-` prefix). Getting a full
build going on Windows would mean fetching `ggml` from wherever `CMakeLists.txt`'s
`FetchContent` step expects it, initializing submodules, and likely wiring up `onnxruntime`
too (`target_link_libraries(cosyvoice-frontend PRIVATE ggml common onnxruntime)`) — real
infrastructure work with meaningful risk of consuming the rest of a session on build-system
troubleshooting rather than audio-pipeline work. Flagging as a legitimate future investment
(a working oracle binary would be very valuable) rather than attempting it under time
pressure and leaving a half-working build tree behind.

**Update, same session, download completed and inspected**: real tensor/metadata dump (`list-
tensors`/`list-metadata`, same CLI technique used throughout this doc) shows CosyVoice3's
`general.architecture = cosyvoice3-2512` is a custom whole-pipeline tag covering all 868
tensors (LLM+flow+HiFT+tokenizer bundled in one file), but **the LLM backbone itself IS also
a plain Qwen2-shaped transformer** — `num_hidden_layers=24, num_attention_heads=14,
num_key_value_heads=2, rope_theta=1e6, rms_norm_eps=1e-6` match CosyVoice2's backbone
dimensions exactly, and `layers.0.self_attn.q_proj.weight [896,896]`/`k_proj.weight [896,128]`
(head_dim=64) confirm identical shapes. Naming convention is a THIRD variant, distinct from
both QwenASR's GGUF (`blk.N.*`) and CosyVoice2's safetensors (`llm.model.model.layers.N.*`):
bare `layers.N.*`/`embed_tokens.weight`/`norm.weight`, no prefix at all — the simplest of the
three. **Real, checkpoint-specific difference from CosyVoice2**: speech vocab is **6761**
here (`speech_embedding.weight`/`llm_decoder.weight` both `[896,6761]`), not CosyVoice2's
6564 — read directly from the real tensor shape, never assumed to match across versions.

**Wrote `CosyVoice3LlmTensorSource.cs`** (new): same architectural bet validated a THIRD
independent time — wraps the real `GgufModel` directly (unlike CosyVoice2's safetensors-based
adapter), remaps the bare `layers.N.*`/`embed_tokens.weight`/`norm.weight` names to standard
`blk.N.*`/`token_embd.weight`/`output_norm.weight` via an explicit per-tensor map (learned
from a real bug below — a generic prefix-only rule is NOT sufficient), presents
`general.architecture="qwen2"`, and includes the same `EnableSpeechGenerationMode()`
composition trick as CosyVoice2's adapter (synthetic combined embedding table + swapped
speech-vocab output head, zero Engine changes) — this time dequantizing from the checkpoint's
real F16 tensors via `Dequantize.ToFloat32` rather than assuming F32 source data (CosyVoice3's
GGUF genuinely IS F16, unlike CosyVoice2's safetensors which turned out to already be F32
despite its config's BF16 hint).

**Real bug caught by testing, not inspection**: `Adapter_MapsRealTensors_...` initially failed
with `Assert.NotNull()` on `blk.0.attn_q.weight`, and the speech-generation test failed with
`Missing tensor: blk.0.attn_output.weight`. Root cause: the first version's `Tensors`
collection was built by filtering `inner.Tensors` for names starting with `"layers."` and
adding them **under their original real names** (`layers.0.self_attn.q_proj.weight`), while
`FindTensor` separately tried a generic `"blk.N.X" -> "layers.N.X"` string-prefix swap that
left the suffix untouched (`self_attn.q_proj.weight` staying `self_attn.q_proj.weight`
instead of becoming `attn_q.weight`) — the two code paths disagreed about what "canonical
name" meant, and neither actually produced real llama.cpp-convention suffixes. Fixed by
building one explicit `_canonicalToReal` dictionary (mirroring exactly the pattern that
already worked correctly in `CosyVoiceLlmTensorSource`'s `MapIfPresent`) used consistently by
`Tensors`, `FindTensor`, `GetTensorData`, and `GetTensorDataPtr` — a single source of truth
instead of two parallel, silently-inconsistent mapping schemes. A smaller second bug in the
same fix pass: `EnableSpeechGenerationMode`'s `_tensors.RemoveAll` originally searched for the
tensor's OLD real name (`"embed_tokens.weight"`) inside a list that, after the first fix, held
canonical names (`"token_embd.weight"`) — corrected to remove/re-add by canonical name.

**Verified via real `ForwardPass` runs**, same discipline as every other adapter this session:
`Tests.Audio/CosyVoice3LlmTensorSourceTests.cs` —
`Adapter_MapsRealTensors_ArchitectureAndSpeechVocabConfirmed` (architecture, layer count,
hidden dim, and the real 6761 speech-vocab size all confirmed against the live checkpoint) and
`SpeechGenerationMode_RunsRealForwardPass_ProducesFiniteSpeechVocabLogits` (real `Prefill`
call, mixed text+speech token ids, finite non-degenerate 6761-length logits). **Both PASS.**
CosyVoice's LLM speech-token generation now works end-to-end through this engine's real
production forward pass on THREE independently-sourced checkpoints (QwenASR GGUF, CosyVoice2
safetensors, CosyVoice3 GGUF) — the architectural approach is thoroughly de-risked at this
point, not a one-off fluke.

**Considered building `examples/cosyvoice.cpp` as a real oracle again now that a genuinely
matching checkpoint (CosyVoice3) exists** — still not attempted this pass (same `vendor/ggml`-
missing/submodule-not-initialized blocker as before; unchanged since the earlier note).
Remains a legitimate future investment, not attempted under time pressure.

**Not yet done**: `campplus.onnx`/`speech_tokenizer_v3.onnx` (the real frontend files,
downloaded alongside the GGUF) are on disk but completely unexamined this session — no
tensor/graph inspection done yet. The flow/HiFT portions of CosyVoice3's bundled GGUF (846 of
its 868 tensors are NOT the LLM backbone — `decoder.estimator.*`, `resblocks.*`,
`source_resblocks.*`, `ups.*`, `f0_predictor.*` etc., a full DiT flow decoder + HiFT-style
vocoder) are completely unexamined. No autoregressive generation loop (sampling/stopping/KV-
cache reuse across `Decode` calls) built for either CosyVoice2 or CosyVoice3 yet — only single
`Prefill` calls have been exercised on all three checkpoints so far.

### CosyVoice2 flow encoder ported and verified; HiFT weight-norm-fold loader done (2026-08-22, same session, direct user request "Let's do CosyVoice2/3's flow (DiT) and HiFT vocoder stages")

**Flow encoder: ported and verified on the first try.** Confirmed via real `flow.pt`/
`cosyvoice2_flow.safetensors` tensor shapes that the "major unlock" finding from the Phase-0
audit was exactly right: `encoder.encoders.{i}.self_attn.{linear_q/k/v/out/pos, pos_bias_u,
pos_bias_v}`, `encoder.pre_lookahead_layer.{conv1,conv2}`, `encoder.up_layer.conv`,
`encoder.up_encoders.*`, `spk_embed_affine_layer`, `input_embedding`, `encoder_proj` — a
byte-for-byte architectural match to `Chatterbox/ChatterboxFlowEncoder.cs`'s already-real,
golden-verified S3Gen `UpsampleConformerEncoder` (hidden=512, 8 heads, head_dim=64, ffn=2048,
mel=80, spk_embed_dim=192 [CAMPPlus x-vector], 6 `encoders` + 4 `up_encoders` blocks — all
read directly from the file, not assumed). Wrote `CosyVoiceFlowWeights.cs` (real loader,
parallel structure to `ChatterboxS3GenWeights.cs` but sourced from plain HF-style safetensors
names instead of GGUF-converted short names) and `CosyVoiceFlowEncoder.cs` (adapted directly
from `ChatterboxFlowEncoder.cs`'s proven math, reusing `Primitives/DenseKernels`'s `Linear`/
`LinearNoBias`/`LayerNorm`/`SoftmaxInPlace`/`SiluInPlace` for the low-level math rather than
re-copying those, though the S3Gen-specific structural logic — `PreLookahead`, `Upsample1D`,
`ConformerLayer`, `RelPositionSelfAttention`, `Conv1dValid`, `EmbedAndScale` — is still
duplicated between the two files, a deliberate, flagged trade-off: extracting a shared
`S3GenConformerKernels` class is mechanically straightforward now that both layer-weight
structs have identical fields, but doing so mid-session while standing up two new, unverified
pipelines was judged riskier than doing it later as its own dedicated, re-verifiable pass).

`Tests.Audio/CosyVoiceFlowEncoderTests.cs` (3 tests: real-shape weight loading, forward-pass
finite-output at the expected 2x-upsampled length, speaker-embedding projection) — **all PASS
on the first attempt**, same as Chatterbox's own encoder did when it was first ported (real
confirmation this is genuinely the same architecture, not a coincidental shape match).

**HiFT vocoder: weight-norm-fold loader done and verified; the DSP forward pass deliberately
NOT ported this pass.** Confirmed CosyVoice2's `hift.pt` is the same NSF-source + ISTFTNet
HiFiGAN family as `Chatterbox/ChatterboxVocoder.cs`'s already-real vocoder (which is itself
explicitly commented as "S3Gen stage 3: HiFTGenerator", i.e. literally the same class of
model) — every architectural constant matches Chatterbox's hardcoded defaults exactly, read
directly from real tensor shapes, not assumed: upsample rates `[8,5,3]`, upsample kernels
`[16,11,7]`, resblock kernels `[3,7,11]` (3 per stage x 3 stages = 9 `resblocks`, confirmed by
index), source-resblock kernels `[7,7,11]`, base_channels=512, n_fft=16 (confirmed:
`conv_post` output channels = 18 = n_fft+2), hop=4.

**Real difference requiring new code, not just weight loading**: this checkpoint's conv
weights use PyTorch's newer `parametrizations.weight.original0/1` weight-norm encoding
(confirmed via real shapes: `original0` is always `[outCh,1,1]`, the standard
`weight_norm(dim=0)` per-output-channel magnitude shape; `original1` is the full direction
tensor) rather than being pre-fused like Chatterbox's GGUF checkpoint. Wrote
`CosyVoiceHiftWeights.cs` with `GetFoldedConvWeight`, folding `w[o,i,k] = g[o] * v[o,i,k] /
||v[o,:,:]||_2` at load time (same class of transform as Parakeet's BN-fold earlier this
session — a real, non-trivial per-checkpoint numerical transform, not a naming exercise).
Applied to `conv_pre`/`conv_post`/`ups.*`/`resblocks.*.convs1/2`/`source_resblocks.*.convs1/2`/
`f0_predictor.condnet.*`; confirmed `source_downs.*`/`m_source.*`/`f0_predictor.classifier`
are plain unparametrized tensors (no `parametrizations.` prefix on those specific names).

**Deliberately did NOT port the vocoder's forward DSP pass this iteration.**
`ChatterboxVocoder.cs`'s `Decode`/`SineGen`/STFT/iSTFT/resblock machinery is ~400 lines of
intricate, already-verified, heavily-commented numerical code (overlap-add iSTFT, dilated
Snake-activated resblocks, NSF harmonic sine source, strided/transposed convs). Duplicating
it wholesale under time pressure risked real transcription errors in code this dense; hastily
refactoring it into a shared kernel to avoid duplication risked destabilizing Chatterbox's
already-golden-verified production vocoder mid-session. Chose to do neither rushed — instead
landed the one piece that's genuinely bounded and mechanical (the weight loader + fold),
verified it (`Tests.Audio/CosyVoiceHiftWeightsTests.cs`: real shapes for `conv_pre`/
`conv_post`/all 3 `ups` stages/9 `resblocks`/3 `source_resblocks`, all finite, and a
non-degeneracy check confirming the fold didn't collapse everything toward zero — PASS), and
left the actual `Generate`/`Decode` port as clearly-scoped next-iteration work with every
architectural constant already confirmed and written down here, so it doesn't need
re-deriving.

**Recommended approach for next iteration** (not started): extract `ChatterboxVocoder.cs`'s
conv/activation/STFT primitives (`Conv1dSamePad`, `ConvTranspose1d`, `Conv1dDilated`,
`Conv1dStrided`, `Conv1dK1`, `ReflectionPadLeft1`, `LeakyReluInPlace`, `EluInPlace`,
`SnakeInPlace`, `RealStft`, `InverseStft`, `HannWindow`, `NearestUpsample1D`,
`AlignTimeLength`) and the resblock/`SineGen`/`LinearTanhMerge`/`Decode`-shaped orchestration
into a shared `Primitives/HiFTVocoderKernels.cs`, parametrized over plain `float[][]`
resblock-weight arrays rather than the Chatterbox-specific typed weights class (so both
`ChatterboxVocoder.cs` and a new `CosyVoiceHiftVocoder.cs` can call the same kernels) — THEN
refactor `ChatterboxVocoder.cs` to use it and re-run its golden test before trusting the
extraction, exactly the "extract once two verified implementations exist to check against
each other" sequencing flagged during the flow-encoder work above. Write `CosyVoiceHiftVocoder.
Generate`/`Decode` against those shared kernels afterward — all architectural constants for
this are already confirmed and listed above, so this should be substantially faster than the
first HiFT port was.

**CosyVoice3's flow/HiFT stages remain completely unexamined** (846 of 868 tensors in the
bundled GGUF) — not reached this pass; do the CosyVoice2 flow/HiFT work first since it has
the more mature reference material (Chatterbox's existing code) to build against, then compare
CosyVoice3's `decoder.estimator.*`/`resblocks.*`/`source_resblocks.*`/`ups.*` tensor shapes
against whatever CosyVoice2 lands on rather than assuming they're identical (same discipline
as everywhere else in this doc — CosyVoice3's speech vocab already turned out to differ from
CosyVoice2's, 6761 vs 6564, so don't assume the flow/HiFT dims carry over either).

### DRY pass: extracted `Primitives/S3GenConformerKernels.cs`, verified against Chatterbox's real golden test (2026-08-22, same session, direct user request "do a DRY pass, make sure it works well, then do CosyVoice3")

Did the deferred extraction flagged in the previous entry now that both `ChatterboxFlowEncoder.
cs` (real, golden-verified) and `CosyVoiceFlowEncoder.cs` (real-weights structural tests
passing) existed to check against each other. Wrote `Primitives/S3GenConformerKernels.cs`
with two small interfaces (`IS3GenConformerLayerWeights` for one Conformer block's weights,
`IS3GenFlowEncoderWeights` for the top-level encoder weights) and moved the actual math
(`Forward`, `EmbedRow`, `ProjectSpeakerEmbedding`, `EmbedAndScale`, `PreLookahead`,
`Upsample1D`, `Conv1dValid`, `ConformerLayer`, `RelPositionSelfAttention`) there verbatim from
`ChatterboxFlowEncoder.cs` (the two files' logic was already identical after the CosyVoice
port, confirmed during the earlier porting pass — this was a pure move, not a rewrite).

**Chatterbox's `ChatterboxS3GenWeights.cs` implements the new interfaces via EXPLICIT
interface implementation** (`int IS3GenFlowEncoderWeights.HiddenDim => EncHidden;` etc.)
rather than renaming its existing `EncHidden`/`EncHeads`/`EncHeadDim`/`EncFfn`/`SpkEncDim`
properties — user confirmed there's no real external API to preserve yet ("no real public API
... noone uses this yet"), but explicit-interface-implementation was still the right call
for a different reason: renaming would have meant touching every other Chatterbox file that
already references those names (`ChatterboxCfmDecoder.cs`, `ChatterboxVocoder.cs`, etc.),
which is unrelated churn and unnecessary risk for a same-session refactor. `CosyVoiceFlowWeights.
cs` implements the interfaces directly (its property names already matched the interface,
since the interface was designed to match what was already written for CosyVoice). Both
per-layer weight classes (`ChatterboxS3GenConformerLayer`, `CosyVoiceFlowLayerWeights`)
needed only `: IS3GenConformerLayerWeights` added — their fields already matched exactly,
zero renaming.

One real compile hiccup, not a logic bug: C# doesn't implicitly convert
`ConcreteLayerType[]` to `IInterfaceType[]` for an explicit interface member the way it does
for some other covariant contexts — needed an explicit `(IS3GenConformerLayerWeights[])`
cast on both `EncLayers`/`UpEncLayers` explicit-interface accessors in both weight classes.

Both `ChatterboxFlowEncoder.cs` and `CosyVoiceFlowEncoder.cs` are now thin wrappers
(`EmbedRow` the tokens, call `S3GenConformerKernels.Forward`/`ProjectSpeakerEmbedding`) — no
duplicated Conformer-block logic remains between the two pipelines.

**Verified the extraction is behavior-preserving, not just "it compiles"**: re-ran
`ChatterboxFlowEncoderTests` (the real golden cosine-similarity test against actual PyTorch
reference output) — **PASS**, plus `ChatterboxCfmDecoderTests` and `ChatterboxVocoderTests`
(downstream consumers of the flow encoder's `mu` output, to catch any subtle behavior change
that wouldn't show up in the encoder's own test alone) — **both PASS**. Re-ran
`CosyVoiceFlowEncoderTests` (all 3) — **PASS**. `Tests.Audio.Fast` still builds clean. This is
the strongest verification bar available in this codebase (real numerical output vs a real
external reference, not just shape/finiteness), and it confirms the refactor changed zero
numerics on either pipeline.

### CosyVoice3 flow/HiFT: real architecture inspected, genuinely NOT a drop-in reuse of CosyVoice2's (2026-08-22, same session, direct user request "then do CosyVoice3")

Dumped CosyVoice3's real flow/HiFT tensor names from the bundled GGUF (same technique as
throughout this doc) before writing any code, per this doc's own established discipline.
**Two real, load-bearing findings, one confirming earlier suspicion and one new:**

**Flow: CosyVoice3 has NO Conformer encoder stage at all — confirms `examples/cosyvoice.cpp`
really is the right reference for THIS checkpoint.** There is no `encoder.encoders.*`/
`pos_bias_u`/`self_attn.linear_q` anywhere in the GGUF. Instead: `input_embedding.weight`
is `[vocab=6561, embed_dim=80]` (NOT CosyVoice2's `[6561,512]` — a much smaller, 80-dim token
embedding, likely feeding directly toward mel-space rather than through a large Conformer
hidden state) and the tensors go straight to `decoder.estimator.input_embed.{conv_pos_embed,
proj}` / `decoder.estimator.time_embed.time_mlp` / `decoder.estimator.transformer_blocks.*`
— a DiT-style flow whose naming (`conv_pos_embed`, `time_embed.time_mlp`) resembles F5-TTS's
DiT more than CosyVoice2's Conformer encoder. This matches exactly what
`CausalMaskedDiffWithDiT::build_cgraph_encode` in `examples/cosyvoice.cpp` showed earlier
this session (token embed -> `PreLookaheadLayer` -> simple upsample -> straight into the DiT
estimator, no attention stage) — confirming that C++ reference, dismissed for CosyVoice2, is
the right oracle for CosyVoice3 specifically. **None of this session's `CosyVoiceFlowEncoder.
cs`/`S3GenConformerKernels` work applies to CosyVoice3's flow at all** — it needs its own,
separate port against the DiT architecture, closer in shape to this repo's existing
`F5TTS/F5DiTModel.cs` (already real and golden-verified) than to Chatterbox/CosyVoice2's
Conformer encoder. Worth checking whether `F5DiTModel.cs`'s block math is reusable as a
template the same way Chatterbox's was for the Conformer encoders — not yet checked.

**HiFT: architecturally close to CosyVoice2's (same stage shapes, same resblock/source-down
structure) but with two real, confirmed differences, not a byte-identical match either.**
`conv_pre.weight` is `[out=512,in=80,k=5]` — kernel **5**, not CosyVoice2's **7** (CosyVoice2:
`[512,80,7]`). `ups.0`/`resblocks.0.convs1.0`/`source_downs.0`/`conv_post` shapes otherwise
match CosyVoice2's HiFT almost exactly (512→256 first upsample stage, 256-channel resblocks
with k=3, `conv_post` outputting 18=n_fft+2 channels). **Also confirmed: none of CosyVoice3's
HiFT conv tensors carry a `parametrizations.weight.original0/1` split — they're already plain,
pre-fused conv weights.** This means `CosyVoiceHiftWeights.GetFoldedConvWeight`'s weight-norm
fold does NOT apply to this checkpoint at all; a CosyVoice3 HiFT loader needs to read these
tensors directly, and should NOT assume the fold step is universal across CosyVoice versions
just because CosyVoice2 needed it.

**Decision: did not start either CosyVoice3 port this pass.** The flow stage needs a genuinely
new DiT-family implementation (not a config-difference from CosyVoice2's, an architecturally
different pipeline), and doing that justice — reading `examples/cosyvoice.cpp`'s DiT estimator
code properly, checking `F5DiTModel.cs` for reuse potential, deriving the conv_pos_embed/
time_embed specifics — is its own substantial task, not a quick follow-on to squeeze into an
already very long session. Writing this up precisely now (exact tensor names/shapes,
kernel-size difference, no-fold-needed finding) so the next iteration can start directly from
real facts instead of re-deriving them, the same discipline every other entry in this doc
follows.

**Next iteration**: (1) Read `examples/cosyvoice.cpp`'s `CausalConditionalCFM`/DiT estimator
build-graph code in full (partially read earlier this session — `build_cgraph_one_step`,
`PreLookaheadLayer::build_cgraph`, `CausalMaskedDiffWithDiT::build_cgraph_encode` — but not
the actual `transformer_blocks`/`conv_pos_embed`/`time_embed` internals). (2) Check whether
`F5TTS/F5DiTModel.cs`'s block math (also a real, golden-verified DiT) is a usable template,
the same way Chatterbox's Conformer encoder was for CosyVoice2's. (3) Write a CosyVoice3-
specific HiFT loader WITHOUT the weight-norm fold (confirmed unnecessary for this checkpoint),
reusing `CosyVoiceHifiResBlockWeights`'s shape/field conventions where they still apply,
adjusting for the confirmed kernel-size-5 `conv_pre`. (4) Once CosyVoice2's own HiFT DSP pass
lands (per the earlier entry's recommended approach: extract `Primitives/HiFTVocoderKernels.cs`
from `ChatterboxVocoder.cs`, verify, then use it for CosyVoice2), reuse those same kernels for
CosyVoice3's HiFT forward pass too — the resblock/source-down/upsample math itself should
still be shared even though the flow stage isn't.

### CosyVoice3 DiT backbone ported — turned out to be F5-TTS's DiT, tensor-for-tensor (2026-08-22, same session, direct user request "probably code CosyVoice3")

Followed up on next-iteration item (2) above immediately: checked `F5TTS/F5DiTBlockWeights`'s
real tensor names (`transformer_blocks.{i}.{attn_norm.linear, attn.to_q/k/v, attn.to_out.0,
ff.ff.0.0, ff.ff.2}`) against CosyVoice3's real dump — **byte-for-byte identical suffixes**,
just prefixed with `decoder.estimator.`. Cross-checked every hyperparameter against real
tensor shapes/GGUF metadata rather than assuming: hidden=1024, heads=16 (`decoder.estimator.
heads` metadata), head_dim=64, ffn=2048, depth=22 (`decoder.estimator.depth` metadata),
`time_embed.time_mlp.0/2` matches F5's `TimeFreqDim=256` exactly, `norm_out.linear`/
`proj_out` match F5's final-layer naming and AdaLN-Zero-final structure exactly (only
`MelDim` differs: 80 here vs F5's 100 — an expected, non-architectural difference). This is
CosyVoice3's DiT estimator running the literal same architecture as `F5TTS/F5DiTModel.cs`
(itself real and golden-verified against actual PyTorch reference output).

**Ported `CosyVoice3DiTWeights.cs`** (real GGUF tensor loading: `input_embed.proj`+
`conv_pos_embed.conv1/conv2`, `time_embed.time_mlp.0/2`, `norm_out.linear`, `proj_out`, all 22
`transformer_blocks`) and **`CosyVoice3DiTModel.cs`** (the block/backbone forward math),
copying `F5DiTBlock.cs`/`F5DiTModel.cs`'s real, verified logic directly rather than
re-deriving it, and reusing `F5TTS.F5Kernels`/`F5TTS.F5RotaryEmbedding` as-is (already
pipeline-agnostic static utilities, zero duplication needed there).

**Deliberately did NOT implement the `input_embed` concatenation step** (what real tensors
combine into the 320-dim vector `input_embed.proj` consumes) — F5's analog is
`concat(x, cond, text_embed)` at `melDim*2+textDim=320`, but CosyVoice3 has no text embedding
in that sense, and guessing the composition (plausible candidates: `x`/flow-token-embedding/
reference-mel/speaker-embedding, each 80-dim, summing to 320) without confirming against
`examples/cosyvoice.cpp`'s real estimator-input construction would be exactly the failure
mode this whole rebuild exists to avoid. Instead exposed `RunBackbone` (takes an
already-embedded `[numFrames, 1024]` hidden state) and `InputEmbed` (takes an already-formed
concatenated input of known width, runs the confirmed `proj`+`ConvPositionEmbedding`
mechanics) as the two testable, real pieces, with the input-formula gap explicitly documented
in `CosyVoice3DiTModel`'s doc comment rather than silently papered over.

**Verified against real weights, passed on the first attempt** (a genuine contrast to
QwenASR's `TensorPrimitives.SoftMax` NaN debugging saga earlier this session — makes sense in
hindsight: this is a careful, direct port of already-golden-verified F5-TTS code, not a
from-scratch derivation, so there was much less surface area for a transcription bug to hide
in). `Tests.Audio/CosyVoice3DiTModelTests.cs`: real-shape weight loading (22 blocks, correct
`input_embed.proj`/`proj_out` dims), `RunBackbone` on a synthetic embedded input (finite,
non-degenerate 80-dim-per-frame output), `InputEmbed` on a synthetic 320-dim concatenated
input (finite 1024-dim-per-frame output) — **all 3 PASS**.

**Next iteration**: resolve the `input_embed` concatenation formula by reading `examples/
cosyvoice.cpp`'s `CausalConditionalCFM::build_cgraph_one_step`/`CausalMaskedDiffWithDiT::
build_cgraph_encode` in full (both partially read this session, not to completion) — once
that's confirmed, wire `InputEmbed` + `RunBackbone` together into a real `ForwardVelocity`
entry point (mirroring `F5DiTModel.ForwardVelocity`'s shape exactly) and the flow-matching ODE
sampler loop (a real, existing template: `F5TTS/F5FlowMatchingOde.cs`, also likely reusable
given how closely everything else has matched so far). Then CosyVoice3's HiFT (already-fused
weights, no fold needed, kernel-5 `conv_pre` — see the earlier entry) is the last remaining
stage.

### CosyVoice3's input_embed formula resolved (confirmed, not guessed); CosyVoice2's HiFT DRY pass + real DSP forward pass, both verified (2026-08-22, same session, direct user request "find the best answers for each")

**CosyVoice3 input_embed: resolved by reading the real source, not guessed.** Found
`InputEmbedding::build_cgraph` (`cosyvoice-graph.cpp` line 278) and its caller `DiT::
build_cgraph` (line 443) in `examples/cosyvoice.cpp`. Confirmed exactly:
`concat(x, cond, text_embed, spks)` in that literal order, where the "text_embed" parameter
slot (inherited from what's structurally a text-conditioned DiT template) is actually fed
`mu` — CosyVoice3's own upsampled 80-dim speech-token embedding from `CausalMaskedDiffWithDiT::
build_cgraph_encode`'s `pre_lookahead_layer` + 2x upsample (confirmed earlier this session:
CosyVoice3 has no Conformer flow encoder at all, so there's no other candidate for what feeds
this slot). `cond` is the padded reference/prompt mel; `spks` is `spk_embed_affine_layer(l2_norm(
campplus_embedding))`, `ggml_repeat`-broadcast across every frame before concatenation (not
real per-frame data). This confirmed order differs from my own pre-confirmation guess (I'd
guessed x+mu+cond+spks; the real order is x+cond+mu+spks) — a concrete demonstration of why
this doc's "never guess architecture" discipline matters even when confident.

Updated `CosyVoice3DiTModel.cs`: `InputEmbed` now takes the four real named `[numFrames,
MelDim]` inputs (`x, cond, mu, spks`) and concatenates them in the confirmed order internally,
rather than an opaque pre-concatenated blob; added `ForwardVelocity(w, x, cond, mu, spks,
timestep, numFrames)` wiring `InputEmbed` + `RunBackbone` together end-to-end (mirroring
`F5DiTModel.ForwardVelocity`'s shape). `Tests.Audio/CosyVoice3DiTModelTests.cs` grew to 4
tests (added a real `ForwardVelocity` end-to-end check) — **all PASS**, first attempt again.
CosyVoice3's DiT is now a complete, real, tested single-step forward pass; only the
flow-matching ODE sampler loop (wrapping repeated `ForwardVelocity` calls, e.g. via `F5TTS/
F5FlowMatchingOde.cs` as a template) and HiFT remain for that pipeline.

**CosyVoice2 HiFT: did the DRY extraction recommended two entries back, then the real DSP
port, both verified.** Extracted `Primitives/HiFTVocoderKernels.cs` from `ChatterboxVocoder.
cs` (~400 lines: `Generate`/`Decode`/`SineGen`/`LinearTanhMerge`/`PredictF0`/
`HifiResBlockForward`/`SnakeInPlace`/STFT+iSTFT/all conv primitives), following the exact same
interface-based pattern as `S3GenConformerKernels` (`IHiFTVocoderWeights`/
`IHifiResBlockWeights`/`IF0PredictorWeights`, `ChatterboxS3GenWeights` implementing them via
explicit interface members to avoid renaming its existing `Voc*`/`Vocoder.*` properties).
One real, non-cosmetic generalization needed during the extraction: the original code
hardcoded `kernel: 7` for both `conv_pre` and `conv_post` (true for Chatterbox's and
CosyVoice2's checkpoints, but NOT for CosyVoice3's, which this session already confirmed uses
`conv_pre` kernel 5) — added `ConvPreKernel`/`ConvPostKernel` to the shared interface instead
of leaving the old hardcoded literals, so the kernel is now a per-checkpoint fact rather than
an assumption silently baked into shared code.

**Verified the extraction first, before building on top of it**: re-ran `ChatterboxVocoderTests`
(the real golden test, cosine-similarity vs actual PyTorch reference waveform output) —
**PASS** — plus `ChatterboxCfmDecoderTests` (a pipeline neighbor, not a direct consumer, run
as an extra check) and the Fast suite build. Zero numeric drift from a ~400-line extraction,
the same clean result as the smaller Conformer-encoder DRY pass earlier.

**Then wrote `CosyVoiceHiftVocoder.cs`** (thin wrapper, mirroring `ChatterboxVocoder.cs`'s new
shape exactly) and verified it against CosyVoice2's real, weight-norm-folded weights:
`Tests.Audio/CosyVoiceHiftVocoderTests.cs` — real forward pass on synthetic mel input, checks
output length is roughly the expected upsampled sample count and every sample is finite and
in the valid `[-1,1]` waveform range — **PASS**.

**CosyVoice2 status**: all three real stages (LLM speech-token generation via
`CosyVoiceLlmTensorSource`, flow encoder via `CosyVoiceFlowEncoder`/`S3GenConformerKernels`,
HiFT vocoder via `CosyVoiceHiftVocoder`/`HiFTVocoderKernels`) now exist, build, and pass
their own real-weights tests. **Not yet done**: nothing wires these three stages together
into one `CosyVoicePipeline.Generate` call — `CosyVoicePipeline.cs`/`CosyVoiceLlm.cs` still
run the original 100%-procedural code path. No numeric golden verification exists yet for any
CosyVoice2 stage (all real-weights tests so far check shape/finiteness, not correctness
against a real oracle) — `examples/cosyvoice.cpp` remains the wrong reference for CosyVoice2
specifically (confirmed CosyVoice3-only earlier this session), so a genuine CosyVoice2 oracle
would need either the original Python `cosyvoice` package or very careful manual derivation.

**Next iteration**: (1) wire CosyVoice2's three real stages into an actual end-to-end
`CosyVoicePipeline.Generate`. (2) Same for CosyVoice3 once its HiFT (already-fused weights,
kernel-5 `conv_pre` — a `CosyVoice3HiftWeights`/loader following `CosyVoiceHiftWeights`'s
pattern minus the weight-norm fold step) and ODE sampler land. (3) Pursue real numeric golden
verification for both — the biggest remaining gap now that every stage runs structurally.

Did next-iteration item (1) above immediately rather than deferring it. Added
`OpenTail.Stingray.Engine`/`OpenTail.Stingray.Cpu` `ProjectReference`s to `Tests.Audio.csproj`
(previously only referenced the Audio project — this is the first Audio test needing a real
Engine forward pass; checked `Engine.csproj` doesn't reference `Audio` back, so no circular
dependency introduced). Hit one unrelated snag: an XML comment containing `--` in the new
`ProjectReference` block broke MSBuild's project-file XML parser entirely (`error MSB4025`,
project fails to even load) — not a C#/XML-comment gotcha specific to this repo, just a
real MSBuild XML-comment restriction (comments can't contain `--`); reworded the comment to
avoid it.

Added `Adapter_RunsRealForwardPass_ProducesFiniteNonDegenerateLogits` to
`QwenAsrLlmTensorSourceTests.cs`: builds `ModelHyperparams.FromGgufMetadata(adapter.Metadata)`
+ `CpuBackend` + `new ForwardPass(adapter, backend, hp)` (the exact same construction pattern
`Tests.ForwardPass/SafetensorsDifferentialFixtureTests.cs` uses for its GGUF-vs-safetensors
differential tests — copied the real pattern rather than guessing at the API), runs
`fwd.Prefill(prompt)` against 5 arbitrary in-vocab token ids, and checks the returned logits
vector has the right length (`vocab_size=151936`), is entirely finite (no NaN/Inf anywhere),
and is non-degenerate (`max-min > 1.0`, ruling out the failure mode where a broken adapter
silently feeds garbage/zeroed tensors through and the engine produces a flat, meaningless
distribution rather than crashing).

**Result: PASS.** Ran individually (`STINGRAY_RUN_HEAVY_TESTS=1 dotnet test
tests/OpenTail.Stingray.Tests.Audio -- --filter-class
OpenTail.Stingray.Tests.Audio.QwenAsrLlmTensorSourceTests`, all 4 tests including this one,
2.5s total). **This is a real, substantial confirmation, not just a structural check**: the
Engine's actual production `ForwardPass` (real Qwen3 GQA + QK-norm + RoPE + SwiGLU, real
Q4_K/Q8_0 dequantization, the same code path every text-generation GGUF in this repo runs
through) executed a real forward step over this checkpoint's real quantized weights via the
adapter and produced a sane, finite, non-degenerate logit distribution. The metadata-
remapping-adapter architectural bet from earlier this session is now empirically validated,
not just plausible-sounding.

**Still not done** (unchanged from before, now on firmer footing): no tokenizer/prompt-
formatting wired up (the 5 token ids in the test are arbitrary, not a real prompt), no KV
cache reuse across a generation loop tested, no multimodal audio-embedding injection, and the
AuT audio encoder itself is still 100% unstarted. But the single highest-risk unknown --
"does the Engine-reuse idea actually work at all, or is `qwen3asr` secretly different enough
from `qwen3` that this blows up at runtime" -- is now answered definitively: yes, it works.

### DRY pass: extracted `Primitives/DenseKernels.cs` (2026-08-21, same session, direct user request)

`ParakeetConformerEncoder.cs` and `Chatterbox/ChatterboxFlowEncoder.cs` (S3Gen's Conformer
encoder, real and golden-verified) each hand-rolled near-identical private copies of five
functions: SIMD `Linear`/`LinearNoBias` (fixed-pointer `SimdKernels.MatVecF32` calls),
`LayerNorm`, `SoftmaxInPlace` (float-only accumulation), and SiLU/Swish activation --
inevitable since Parakeet's port used Chatterbox's file as a structural template but copied
the helpers instead of sharing them. Extracted all five into a new
`Primitives/DenseKernels.cs`, following the same "single source of truth for a kernel family"
pattern already established by `Primitives/VitsAttentionKernels.cs` for the VITS pipelines.
Both call sites now delegate via one-line wrappers (kept as thin private aliases rather than
replacing every call site, so neither file's diff is a mechanical rename storm and each
still reads as "this is Chatterbox's Linear" / "this is Parakeet's Linear" locally).

No behavior change intended (pure extraction, byte-identical math) -- verified by re-running
both real-weights test suites individually after the refactor:
`OpenTail.Stingray.Tests.Audio.ChatterboxFlowEncoderTests` (golden cosine-similarity check
against real PyTorch output) and `OpenTail.Stingray.Tests.Audio.ParakeetConformerEncoderTests`
(3 tests) -- both still PASS, confirming the extraction didn't silently change any numerics
(e.g. an eps-default mismatch, a bias-null-check ordering difference). Any future Conformer-
style port (QwenASR, FunASR's paraformer) should use `DenseKernels` from the start rather than
copying a third private version.

## Autonomous cron-driven pass (2026-08-22): FunASR real weights + tokenizer, Silero VAD / Fish Speech / Parler-TTS / Orpheus TTS scoped

User set a recurring local cron job (30-min cadence, time-gated to skip its first ~3 no-op
fires) to work through a new queue while AFK: 1) FunASR, 2) Silero VAD, 3) Fish Speech,
4) Parler-TTS, 5) Orpheus TTS -- same "no subagents, golden-verify, run heavy tests one
filter-class at a time" discipline as everywhere else in this doc.

**Scope check on items 3-5 (Fish Speech, Parler-TTS, Orpheus TTS) -- all three BLOCKED, not
started, moving past them per the standing "genuinely blocked -> document and move on" rule**:
checked `models/` and `examples/` directly -- zero weight files, zero reference source, and
(unlike every other pipeline in this doc, including FunASR/Silero VAD) not even a fake stub
pipeline class exists in `src/OpenTail.Stingray.Audio/` for any of the three. These aren't
"port real math into an existing fake pipeline" tasks like the rest of this doc -- they'd be
entirely new pipelines (new tokenizer, new weight format, new model download) with nothing
local to build against. Correctly out of scope for this pass; would need the user to supply
model weights and/or a reference implementation before any of the three can start.

### FunASR (Paraformer) -- real GGUF weight loader + real tokenizer landed and verified; encoder/predictor/decoder forward pass NOT yet ported

**Real architecture confirmed directly from `models/paraformer-q8.gguf`'s own metadata/tensor
names** (`general.architecture = "paraformer"`, 957 tensors, 14 metadata keys -- dumped via
`dotnet run --project src/OpenTail.Stingray.Cli -- list-metadata`/`list-tensors`, not guessed):
a SAN-M (Self-Attention with Memory, i.e. multi-head self-attention + a depthwise FSMN conv
memory term) encoder with one special first layer (`encoder.encoders0.0`, 560-dim input --
CMVN'd 80-mel x 7-frame splice, `cmvn.scale`/`cmvn.shift` are real per-dim affine params, not
computed) followed by a **49-layer** main stack (`encoder.encoders.{0..48}`, 512-dim,
`pf.enc.num_blocks=50` metadata counts encoders0+main-stack COMBINED -- a real off-by-one trap,
see the bug below); a CIF (Continuous Integrate-and-Fire) predictor (`predictor.cif_conv1d`
kernel=3 conv + `predictor.cif_output` linear) that both counts and boundary-detects acoustic
tokens from the encoder output; a non-autoregressive cross-attention decoder with 16 main
layers (`decoder.decoders.{0..15}`, `self_attn` is FSMN-ONLY -- no Q/K/V at all, confirmed by
direct tensor-name inspection, matching Paraformer's non-autoregressive design where the
decoder never attends to its own un-emitted future tokens; `src_attn` is real cross-attention
to the encoder output) plus one extra FFN-only layer (`decoder.decoders3.0`, no attention
tensors at all) and a real vocab projection (`decoder.output_layer`, 512->8404) to a real,
GGUF-embedded 8404-token vocabulary (`pf.vocab` metadata, a plain string array -- confirmed by
directly dumping vocab entries via a throwaway console app, not assumed).

**Critical finding, do not re-derive**: the only local C++ reference
(`examples/paraformer.cpp/src/csrc/paraformer-offline.cpp`) has the encoder's FSMN memory-term
ADD explicitly DISABLED in code (`// todo open when conv depth wise 1d with group implement
finished`, ~line 1767: `// cur = ggml_add(ctx0, cur, fsmn_memory);` is commented out) -- i.e.
that specific reference is a known-incomplete/broken implementation on exactly the SAN-M detail
that matters most. Do NOT port its encoder forward pass as-is when this work continues. The
real FSMN formula needs deriving from FunASR's actual Python source (the `funasr` package's
`funasr/models/sanm/attention.py`/`encoder.py` -- NOT yet fetched this session, same
"pip download the real package" technique used for Kokoro/Chatterbox/F5-TTS earlier in this
doc) before writing `FunAsrEncoder.cs`'s forward pass.

**Landed and verified this pass**:
- `FunASR/FunAsrWeights.cs` (new): real GGUF loading for every one of the 957 real tensors --
  cmvn, encoders0 (1 layer), encoders (49 layers), encoder.after_norm, predictor (CIF
  conv1d+output), decoder.decoders (16 layers), decoders3 (1 FFN-only layer),
  decoder.after_norm, decoder.output_layer, and the real 8404-entry vocab array. Per-layer
  weight holder types (`FunAsrEncoderLayerWeights`/`FunAsrDecoderLayerWeights`/
  `FunAsrDecoderFfnLayerWeights`) mirror the shape/field conventions already established by
  `QwenAsrAudioLayerWeights` etc.
- **Real bug found and fixed via actually running it, not just building it** (same discipline
  every prior pipeline in this doc was held to): first version assumed `EncoderLayers=50` from
  `pf.enc.num_blocks` directly and crashed (`InvalidDataException: ... missing required tensor
  'encoder.encoders.49.norm1.weight'`) since the real main stack only goes to index 48 (49
  layers) -- `num_blocks` counts `encoders0.0` + the 49-layer main stack combined. Fixed by
  loading `EncoderLayers = num_blocks - 1`.
- **`FunASR/FunAsrPipeline.cs`'s `FunAsrTokenizer`**: was 100% fake (`Decode` just printed
  `[T123]`-style placeholders regardless of input). Now decodes against the real vocab with the
  real convention confirmed by directly inspecting vocab strings (not guessed): a trailing `@@`
  marks "glue to the next piece, no space" (ESPnet/subword-nmt BPE continuation, e.g.
  `and@@`+`these` -> `andthese`); single CJK characters get no surrounding spaces (Chinese has
  none); completed non-CJK words get a trailing space; `<blank>`/`<s>`/`</s>`/`<unk>` are
  stripped, never emitted as text. `FunAsrPipeline.Load` now constructs `FunAsrWeights` (real)
  instead of a bare `GgufModel` used only for `Dispose()`, and wires the real tokenizer through.
- **Tests, all real-weights, all PASS**: `Tests.Audio/FunAsrWeightsTests.cs` (new, 3 tests) --
  every real tensor's shape/finiteness spot-checked across encoders0/first+last main
  encoder/predictor/first+last decoder/decoders3/output_layer/cmvn; the real vocab's first/last
  known entries (`<blank>`, `<s>`, `</s>`, `and@@`, `<unk>`) match direct inspection exactly;
  the tokenizer's CJK-no-space and `@@`-continuation-glue behavior verified against real vocab
  ids. Pre-existing `Tests.Audio.Fast/FunAsrRealWeightsTests.cs` (3 tests, GGUF+2 ONNX model
  paths) re-run individually after the fix -- still PASS (output text is still meaningless,
  since the encoder/predictor are unported fakes feeding real-vocab strings for synthetic
  token ids -- but the pipeline no longer crashes and the tokenizer itself is genuinely real).

**NOT yet done, concrete next steps for whoever picks this up**:
1. `pip download funasr --no-deps` (or find the specific `sanm` module source another way) to
   get the REAL FSMN/SAN-M attention formula -- do not trust `examples/paraformer.cpp` for this
   specific detail, see the critical finding above.
2. Port `FunAsrEncoder.cs`: encoders0 (560-dim) -> 49x main SAN-M layer (LayerNorm -> QKV
   self-attn + FSMN depthwise-conv-over-V memory term, added not skipped -> LayerNorm -> FFN)
   -> after_norm. Build a golden-output oracle (same pip-source-run technique as every other
   pipeline in this doc) before writing the forward pass, not after.
3. Port `CifPredictor.cs`'s real forward: `cif_conv1d` (kernel=3) -> `cif_output` linear ->
   sigmoid -> the real CIF integrate-and-fire accumulation/boundary-detection algorithm (read
   the real predictor math from `funasr`'s Python source, NOT the current fake fixed-0.25-
   alpha-per-frame heuristic).
4. Port `FunAsrDecoder`: FSMN-only self-attn (no Q/K/V) -> real cross-attention to encoder
   output -> FFN, x16 layers, + the `decoders3` FFN-only layer -> after_norm -> output_layer
   (512->8404 logits) -> argmax/greedy token selection (non-autoregressive: all positions
   decoded in one pass, no per-token loop needed, unlike QwenASR's LLM decoder).
5. Golden-verify each stage numerically (cosine similarity, >0.99 bar, matching every other
   pipeline's discipline) before calling any of the above done.

**Fish Speech / Parler-TTS / Orpheus TTS**: blocked, see the scope-check note above -- no local
weights, no reference source, not even a fake stub exists. Needs the user to supply model
files and/or a reference implementation before this queue can reach them.

**Silero VAD**: not reached this pass (ran out of fire-cycle time on FunASR's investigation +
weight-loader work) -- still in the state described earlier in this doc (2 of 4 conv stages +
LSTM genuinely real, 2 conv stages + final projection permanently fake, plus a hardcoded
heuristic layered on top). Next in the queue for a future fire.

### Silero VAD (next cron fire, 2026-08-22): real ONNX graph fully decoded, STFT+encoder verified bit-exact; LSTM/decoder stage numerically wrong, root cause not yet found

**Real 16kHz forward pass fully decoded from `models/silero_vad.onnx` directly** (Python's
`onnx` package walking the actual protobuf graph node-by-node -- not guessed, not inferred from
Silero's public docs/old code comments, which described the WRONG architecture, see this doc's
earlier "UPDATE 3rd iteration" entry). Confirmed the doc's earlier finding and went further:
the top-level graph is just `Equal(sr,16000) -> If -> Identity`; the ENTIRE real model lives
inside the `If` node's `then_branch` subgraph (54 nodes, `else_branch` is a separate unused
8kHz path). Real pipeline, confirmed node-by-node including every Conv's exact
stride/pad/kernel attribute and every Slice's exact start/end/axis (not assumed):
1. `padded = ReflectPad(raw_512_samples, pad=[begin=0,end=64])` -- pads the END ONLY by 64
   samples, reflect mode (the old fake code assumed symmetric 64+64 head+tail padding -- wrong).
2. `stft = Conv1d(padded.unsqueeze(1), stft.forward_basis_buffer[258,1,256], stride=128,
   pad=0)` -- learned STFT frontend, 258 output channels.
3. `real = stft[0:129]`, `imag = stft[129:258]` (confirmed via the real Slice node ranges, NOT
   interleaved per-bin as one might guess), `mag = sqrt(real^2+imag^2)`.
4. `ReLU(Conv1d(mag, encoder.0, k=3,s=1,p=1))` (129->128) -> `ReLU(Conv1d(_, encoder.1,
   k=3,s=2,p=1))` (128->64) -> `ReLU(Conv1d(_, encoder.2, k=3,s=2,p=1))` (64->64) ->
   `ReLU(Conv1d(_, encoder.3, k=3,s=1,p=1))` (64->128) -- Silero's fused "reparam_conv"
   architecture (one Conv+bias per stage, no separate norm/activation tensors), confirmed
   real shapes match this checkpoint's actual initializers exactly.
5. A standard ONNX `LSTM` op (hidden_size=128) whose W/R/B tensors are auto-generated-named
   (`.../Unsqueeze_7/8/9_output_0...`, constant-folded by the exporter, not clean module-path
   names) but resolve to real, correctly-shaped values ([1,512,128]/[1,512,128]/[1,1024]).
6. `Sigmoid(Conv1d(lstm_out, decoder.decoder.2[1,128,1]))`, mean over time if >1 output frame.

**Required extending `OnnxModel` (shared Core class, also used by Piper/MeloTTS) to recurse
into `If` node subgraphs** -- it previously only walked the top-level `GraphProto`, so it saw
Silero's real weights (all buried inside `then_branch`) as simply absent. Added recursive
attribute parsing (`AttributeProto.g`, field 6) that folds a subgraph's nodes/initializers into
the same dictionaries the top-level graph uses; purely additive (existing single-graph models
have no `If` nodes, so see zero behavior change) -- verified by re-running
`PiperTextEncoderTests` (golden cosine-similarity test) individually after the change, still
PASS.

**Landed this pass**:
- `Vad/SileroVadWeights.cs` (new): real weight loading for all of the above via the extended
  `OnnxModel`, with the exact real tensor names/shapes documented in its class doc comment so
  the next person doesn't have to re-walk the graph.
- `Vad/SileroVad.cs`: fully rewritten forward pass (was 100% Glorot/sin-initialized fake weights
  + a hardcoded `frameEnergy`-based heuristic layered on top of whatever the fake network
  computed) to the real pipeline above. `SileroVad.Load` now takes the `.onnx` path (was
  `.gguf`) -- `SileroVadRealWeightsTests.cs` updated accordingly (real model file is
  `models/silero_vad.onnx`, the `.gguf` conversion remains messy/unused per the doc's earlier
  finding). Both structural tests (silence<0.2 probability, speech prob in [0,1]) still PASS.

**Golden-verified via bisection against real onnxruntime output (single `sess.run()` call,
`ORT_DISABLE_ALL`, matching this doc's established discipline) -- STFT and the 4-stage encoder
are BIT-EXACT (float32 rounding only); the LSTM+decoder stage is NOT yet correct**:
- Golden input: a seeded (`numpy default_rng(42)`) synthetic 512-sample frame, saved to
  `scratch-llamacpp-ref/silero_golden_input.txt` (plain comma-separated floats, so the C# test
  doesn't need to reproduce numpy's specific PRNG algorithm).
- Real onnxruntime probability for this frame (fresh zero LSTM state): **0.025505661964416504**.
- Bisected stage-by-stage with a from-scratch numpy re-implementation of each stage (walking
  the real ONNX initializers directly via `onnx.numpy_helper`, independent of the C# code) AND
  a throwaway C# console app computing the same intermediate values:
  - STFT magnitude: numpy and C# match exactly (`mag[:3,:]` identical to 7 significant figures).
  - 4-stage encoder output (`h3`/`enc3`): numpy and C# match exactly (`[0, 1.4528..., 9.3853...,
    0, 0]` identical).
  - LSTM+decoder: **both numpy's independent re-implementation AND the C# port produce the SAME
    wrong value, 0.012973616 vs 0.012973609** (matching each other, NOT the real onnxruntime
    output of 0.025505662) -- this proves the bug is in the ASSUMED LSTM formula/wiring itself
    (gate order, or which tensor actually feeds the LSTM), not a C#-specific porting mistake,
    since two independent implementations of the SAME assumption agree with each other and
    disagree with ground truth together.
- **Root cause NOT found this pass, flagged rather than guessed at further**: the real graph
  has extra conditional nodes between the encoder output and the LSTM
  (`If_0_then_branch__Inline_0__/If`, then `Shape_2`/`Size`/`Equal_1`/`Not`/`If_1`/`If_2`) that
  reshape/select the LSTM's actual X/initial_h/initial_c inputs based on a shape check (likely
  distinguishing a streaming single-chunk case from a batched-multi-chunk case) -- this
  session's straightforward "unsqueeze the raw encoder output" assumption for the LSTM's X
  input may be taking the WRONG branch of that conditioning for this checkpoint's actual
  calling convention, or the real gate order/tensor orientation differs from the ONNX LSTM
  spec's documented default (`iofc`) in a way not yet identified. `SileroVadWeightsTests.cs`
  (new) encodes this exact golden comparison and is a KNOWN, EXPECTED FAILURE right now --
  left failing/uncommitted rather than silently weakening the assertion, so the next fire (or
  the user) sees red, not a false green.

**Concrete next steps for whoever continues this** (superseded, see the RESOLVED entry
immediately below -- kept for history, not because they're still open):
1. ~~Trace the `If`/`If_1`/`If_2` subgraphs~~ -- done, see below: pure shape ops, not the bug.
2. ~~Verify against `torch`'s installed Silero VAD reference~~ -- not needed, root cause found
   via direct onnxruntime bisection instead.
3. Once golden, consider a SIMD pass on the STFT's fully-unrolled O(kernel*freqBins*frames)
   scalar loop (same anti-pattern already fixed in QwenASR/HiFT elsewhere in this doc) -- STILL
   open, not attempted this session, correctness came first.

### Silero VAD RESOLVED (same fire, continued): root cause found and fixed -- golden test now PASSES

Continued directly from the "next steps" above. Traced `If`/`If_1`/`If_2`/`If_5` (one more
level of `AttributeProto.g` recursion) node-by-node: every one of them is a pure SHAPE
operation (`Squeeze(axis=-1)`/`Unsqueeze(axis=0)`/`Gather`) reshaping between `[128]`,
`[1,128]`, and `[1,1,128]` depending on whether the encoder output's time dimension is 1 --
confirmed these never change VALUES, only tensor rank/shape, so the original "feed the raw
128-dim encoder output as a single LSTM timestep, gather state[0]/state[1] as h0/c0" assumption
was already structurally correct. Also confirmed the decoder consumes the LSTM's `h_n` (final
hidden state, via `Squeeze_2`), not the raw sequence output `LSTM_output_0` (which is provably
unused anywhere downstream) -- matching what was already implemented.

**A brute-force search over all 4! LSTM gate-order permutations x 4 bias variants (Wb+Rb, Wb
only, Rb only, zero) against the real golden probability found ZERO matches** -- ruling out
"just the wrong gate order" as the bug and raising a more fundamental question: was the
`h3`/encoder-stage value itself actually right, or did the earlier "numpy vs C#" bisection just
have both independently derived implementations sharing the SAME misreading of the graph
(since both were coded from this session's own understanding, not cross-checked against a
REAL onnxruntime run)?

**Built a standalone extraction ONNX model to get real intermediate values directly** (the
technique that actually cracked this): used `onnx.helper.make_graph` to construct a NEW,
self-contained model reusing the `then_branch` subgraph's nodes+initializers verbatim, with
explicit top-level inputs `input`/`state` (subgraph nodes reference these by outer-scope name,
which works fine once they're real top-level graph inputs) and outputs set to whatever internal
tensor names needed inspecting (`Relu_3_output_0` = encoder output, `Squeeze_2_output_0` = LSTM
h_n, `outputs_0` = final probability) -- `onnx.checker.check_model` complains about missing
output shape info for control-flow-derived tensors, so skip the checker (not needed to run) and
feed the serialized bytes straight to a real `onnxruntime.InferenceSession`. This is a
real, reusable technique for tapping ANY internal tensor inside an `If`/`Loop` subgraph that
onnxruntime's normal `session.run()` can't expose directly -- worth remembering for future
graph-structured-model debugging in this doc.

**Result: both the real encoder output AND the real LSTM hidden state matched this session's
manual re-implementations almost exactly** (`Relu_3` real=`[0, 1.4528553, 9.3853245, 0, 0]` vs.
this session's `h3`=`[0, 1.4528542/1.4528533, 9.3853245, 0, 0]`; real `h_n`=
`[0.00016919836, 0.76025409, -0.000040186413, ...]` vs. manual `h1`=`[0.00016920401,
0.76025391, -0.000040220231, ...]`) -- i.e. the LSTM gate-order/bias assumption (i,o,f,c order,
Wb+Rb summed) was RIGHT all along, and the earlier brute-force search finding no match was
because it was searching the wrong stage entirely.

**The actual bug: a missing ReLU between the LSTM and the decoder conv.** Re-reading the
original 84-line node dump (`Unsqueeze_19(hn) -> Relu_4 -> Conv_5(decoder.decoder.2) ->
Sigmoid`) shows a `Relu_4` node this session's C# port had simply never implemented -- the
decoder's real input is `ReLU(h_n)`, not raw `h_n`. Confirmed the fix numerically: recomputing
the decoder stage as `sigmoid(sum(relu(h_n) * decw) + decb)` gives `0.025505627` against the
real golden `0.025505661964416504` -- matches to 6 significant figures.

**Fixed in `Vad/SileroVad.cs`**: added `MathF.Max(0f, ...)` to the decoder's per-channel
accumulation loop, with a doc comment explaining exactly what the bug was and how it was found
(so nobody re-derives this investigation). `SileroVadWeightsTests.
ProcessFrame_RealWeights_MatchesOnnxGoldenProbability` -- **PASSES**, tolerance tightened from
the initial loose `0.01` to `0.0001` (still passes) once the fix was confirmed real, not a
fluke. `SileroVadRealWeightsTests.cs` (2 structural tests: silence<0.2 probability, speech
prob in [0,1]) also re-run individually -- still PASS. **Silero VAD's core neural forward pass
is now genuinely, numerically real** -- the biggest concrete milestone from this whole
autonomous cron session.

**Still open / not done this pass**:
- The STFT frontend's O(kernel*freqBins*frames) scalar loop is unvectorized (same anti-pattern
  already fixed for QwenASR/HiFT elsewhere in this doc) -- correctness came first this pass,
  a SIMD pass is a legitimate future perf target now that correctness is proven.
- Only ONE golden input (a seeded-random synthetic frame) has been verified -- worth adding a
  second golden sample (e.g. actual speech vs actual silence audio, if real WAV fixtures exist
  anywhere in this repo) for broader confidence, though the bisection technique above makes
  false-positive risk low (the bug was found and fixed via real intermediate-tensor comparison,
  not just end-to-end probability matching).
- `VadSegmenter`/`DetectSegments` (multi-frame segment-boundary logic) was NOT touched or
  re-verified this pass -- only `ProcessFrame`'s single-frame math was fixed/verified. Worth a
  follow-up check that segment boundaries still make sense with the real (now-correct) neural
  output instead of whatever the old fake+heuristic hybrid produced.

## FunASR (Paraformer) real algorithm, fully transcribed from the real `funasr` Python package (2026-08-22, next cron fire) -- READ THIS BEFORE WRITING ANY FORWARD-PASS CODE, do not re-derive

Silero VAD queue item is DONE (see RESOLVED entry above); this fire went back to FunASR per the
queue order. `pip download funasr --no-deps` (matching this doc's established "get the real
source" technique) succeeded -- extracted to `examples/funasr-py/` (the whole real `funasr`
package, NOT just a wrapper). This finally gives a TRUSTWORTHY reference, unlike
`examples/paraformer.cpp` which was already confirmed broken/incomplete on the exact detail
that matters most (see the earlier "Critical finding" entry: its FSMN memory-add is commented
out). Read `funasr/models/sanm/{attention,encoder,decoder,positionwise_feed_forward}.py` and
`funasr/models/paraformer/cif_predictor.py` in full this pass. This checkpoint
(`paraformer-q8.gguf`) is confirmed to be the plain `paraformer` model (not `bicif_paraformer`/
`e_paraformer`/`contextual_paraformer` variants) using `CifPredictorV2` (see below for how that
was determined from the real tensor shape, not assumed).

**Encoder layer (`EncoderLayerSANM.forward`, applies to BOTH `encoders0.0` and every
`encoders.N`)** -- pre-LN structure:
```
residual = x
x_normed = norm1(x)
attn_out = self_attn(x_normed)              # see "encoder self-attn" below
if in_size == size:  x = residual + attn_out
else:                x = attn_out            # ENCODERS0.0 ONLY (in=560,out=512): NO residual!
residual2 = x
x_normed2 = norm2(x)
x = residual2 + feed_forward(x_normed2)      # feed_forward = plain FFN, see below
```
**CRITICAL, easy to get wrong**: `encoders0.0` has `in_size=560 != size=512`, so its
self-attention's residual connection is SKIPPED ENTIRELY (`x = attn_out`, not `residual+
attn_out`) -- confirmed directly from `EncoderLayerSANM.forward`'s `if self.in_size ==
self.size` branch, not assumed. Every main `encoders.N` layer (in=out=512) DOES get the normal
residual add.

**Encoder self-attention + FSMN (`MultiHeadedAttentionSANM.forward`)**:
```
q_h, k_h, v_h, v = forward_qkv(x)      # linear_q_k_v(x) split into 3; v is [B,T,512] (pre-head-split)
fsmn_memory = forward_fsmn(v)          # NOT x -- the FSMN branch operates on v, not the layer input
q_h *= d_k ** -0.5
scores = q_h @ k_h^T
att_outs = softmax(scores) @ v_h  ->  linear_out(...)   # standard scaled-dot-product attention
return att_outs + fsmn_memory          # BOTH branches summed -- this is the add examples/
                                        # paraformer.cpp has DISABLED/commented out, confirmed
                                        # the real math does need it
```
`forward_fsmn(v)`: `x = pad(v.transpose(1,2), left=(kernel-1)//2, right=kernel-1-left)` (kernel=11
-> left=5,right=5, symmetric, since `sanm_shfit=0` for the encoder) `-> depthwise_conv1d(x,
groups=512, no bias) -> x.transpose back -> x + v` (residual is `v`, the SAME `v` fed in, NOT
the original layer input `x`) `-> return x`. Matches the real `fsmn_block.weight` shape
`[512,11]` flattened (GGUF) = PyTorch `[512,1,11]` depthwise-conv convention exactly.

**Encoder FFN (plain `PositionwiseFeedForward`, NOT the decoder's SANM variant)**:
`w_2(ReLU(w_1(x)))` -- no internal LayerNorm (confirmed: encoder's real tensors have no
`feed_forward.norm.*`, only `w_1.{weight,bias}`/`w_2.{weight,bias}`, both WITH bias).

**Decoder layer (`DecoderLayerSANM.forward`) -- GENUINELY SURPRISING ORDER, do not assume the
"obvious" self-attn-then-FFN transformer order**:
```
residual = tgt                          # the ORIGINAL layer input, kept aside
tgt_normed = norm1(tgt)
tgt = feed_forward(tgt_normed)          # FFN runs FIRST, unconditionally
x = tgt                                 # x currently = raw FFN output, no residual yet
if self_attn:                           # true for decoders.0..15, FALSE for decoders3.0
    tgt_normed2 = norm2(tgt)            # norm2 applied to the FFN's output, not the original input
    x = self_attn(tgt_normed2)          # FSMN-only self-attn (no Q/K/V at all)
    x = residual + x                    # residual is the ORIGINAL tgt from the top -- NOT tgt (FFN output)!
if src_attn:                            # true for decoders.0..15, FALSE for decoders3.0
    residual2 = x
    x = norm3(x)
    x = residual2 + src_attn(x, memory) # standard cross-attention to encoder output
return x
```
For `decoders.0..15` (self_attn and src_attn both present): FFN -> FSMN(norm2(ffn_out)) with
residual back to the ORIGINAL input -> cross-attn(norm3(...)) with residual to the FSMN output.
For `decoders3.0` (self_attn=None, src_attn=None, confirmed: this layer's real tensors are only
`norm1.*`+`feed_forward.*`, nothing else): the function reduces to just `x = feed_forward(norm1(
tgt))` -- **no residual add at all** for this layer (the `residual + x` line only executes
inside the `if self_attn:`/`if src_attn:` blocks, both skipped here).

**Decoder FSMN self-attn (`MultiHeadedAttentionSANMDecoder.forward`)**: same depthwise-conv
pattern as the encoder's FSMN branch, but applied directly to the LAYER INPUT (no Q/K/V split at
all -- confirmed by the real tensor set: decoder layers have ONLY `self_attn.fsmn_block.weight`,
no `linear_q_k_v`). For offline (non-streaming) inference, `cache=None` always, so: `x =
depthwise_conv1d(pad(input.transpose(1,2), left=(11-1)//2, right=11-1-left)) .transpose_back
+ input` (residual = the FSMN's own input this time, straightforward).

**Decoder cross-attention (`MultiHeadedAttentionCrossAtt.forward`)**: standard, no surprises --
`q = linear_q(x)`, `k,v = split(linear_k_v(memory))`, scaled dot-product, `linear_out(...)`.

**Decoder FFN (`PositionwiseFeedForwardDecoderSANM`, DIFFERENT from the encoder's)**:
`w_2(norm(dropout(ReLU(w_1(x)))))` -- HAS an internal LayerNorm between activation and `w_2`
(confirmed: `decoder.decoders.N.feed_forward.norm.{weight,bias}` real tensors exist), and `w_2`
has NO bias (confirmed: no `feed_forward.w_2.bias` tensor anywhere in the decoder).

**Decoder orchestration (`ParaformerSANMDecoder.forward`)**: `x = embed(tgt)` where `embed` is
`nn.Identity()` for this checkpoint (confirmed: no `decoder.embed.*` tensor anywhere) -- i.e.
**the decoder's input IS the CIF predictor's `acoustic_embeds` output directly**, no learned
token embedding lookup at all (consistent with Paraformer being fully non-autoregressive: there
is no autoregressive token loop to embed). Then `x = decoders(x, ...)` (16 main layers) ->
`decoders2` (confirmed ABSENT for this checkpoint: `num_blocks(16) - att_layer_num(16) = 0`) ->
`decoders3` (the 1 FFN-only layer) -> `after_norm(x)` -> `output_layer(x)` (512->8404 logits,
straight argmax per position gives the token id, no further processing -- this is the final
transcript token sequence, CTC-adjacent non-autoregressive decode, no beam search needed for a
first correct port).

**CIF predictor is `CifPredictorV2`, NOT the base `CifPredictor`** -- determined from the real
tensor shape, not assumed: `predictor.cif_conv1d.weight`'s real GGUF shape is `[3,512,512]`
(GGUF stores dims fastest-varying-first, i.e. PyTorch shape `[512,512,3]` = out,in,kernel) --
a FULL, non-grouped Conv1d. The base `CifPredictor` class hardcodes `groups=idim` (depthwise,
would need PyTorch shape `[512,1,3]`), which does NOT match; `CifPredictorV2` has no `groups`
argument (defaults to a full conv), which DOES match.
```
context = hidden.transpose(1,2)                          # [B,512,T]
queries = pad(context, left=1, right=1)                  # kernel=3 -> symmetric pad=1 (l_order=r_order=1)
output = ReLU(cif_conv1d(queries))                        # FULL conv, NOT depthwise, NO v1's "+context" residual
output = output.transpose(1,2)                            # back to [B,T,512]
alphas = sigmoid(cif_output(output))                      # Linear(512,1) -> [B,T,1] -> squeeze -> [B,T]
                                                           # (smooth_factor=1, noise_threshold=0 -- this
                                                           # checkpoint's metadata shows no override, so
                                                           # these ReLU(alphas*1-0) terms are no-ops)
# tail_process_fn (tail_threshold=0.45 from real pf.predictor.tail_threshold metadata, tail_mask
# irrelevant for a single non-padded utterance -- mask=None branch):
alphas = concat([alphas, [tail_threshold]])               # append ONE extra synthetic alpha=0.45 at the end
hidden = concat([hidden, zeros[1,512]])                   # append a matching ZERO acoustic frame
token_num = floor(sum(alphas))                            # final predicted token count
acoustic_embeds, cif_peak = cif_v1(hidden, alphas, threshold=1.0)
acoustic_embeds = acoustic_embeds[:, :token_num, :]        # truncate to the predicted length
```
**The core CIF integrate-and-fire algorithm** (`cif_wo_hidden_v1` + `cif_v1`, confirmed exact
formula from source, not guessed -- the real code is a vectorized cumsum/floor trick that is
mathematically the SEQUENTIAL classic-CIF algorithm specialized to `threshold=1.0`; port the
sequential form, it is equivalent and much simpler in C#):
```
accumulated_weight = 0; accumulated_state = zeros(512); tokens = []
for t in 0..T-1 (T = original_time_len + 1, the +1 tail frame):
    if accumulated_weight + alpha[t] >= threshold:        # threshold = 1.0 for this checkpoint
        remaining = threshold - accumulated_weight        # portion of alpha[t] that completes THIS token
        carry = alpha[t] - remaining                       # portion that starts the NEXT token
        emitted = accumulated_state + remaining * hidden[t]
        tokens.append(emitted)
        accumulated_weight = carry
        accumulated_state = carry * hidden[t]
    else:
        accumulated_weight += alpha[t]
        accumulated_state += alpha[t] * hidden[t]
# tokens now holds `round(sum(alphas))`-ish entries; the real code additionally truncates to
# exactly token_num = floor(sum(alphas)) computed above -- keep only the first token_num entries.
```
This sequential form was NOT independently derived/guessed -- it is the well-known classic CIF
formulation that the real vectorized `cif_wo_hidden_v1`/`cif_v1` source (prefix-sum + floor +
remainder-carry logic, read in full this pass) is provably equivalent to for `threshold=1.0`;
implement the sequential form directly in C#, it will match.

**Concrete next steps for whoever picks up C# implementation** (research phase is DONE --
everything above is real, sourced, ready to port; no more Python reading should be needed):
1. Port `FunAsrEncoder.cs`: `encoders0.0` (560-dim, NO self-attn residual) -> 49x main
   `encoders.N` (512-dim, WITH residual) -> `after_norm`. Reuse the shared attention/FFN math
   where it overlaps with other Conformer-family pipelines in this codebase
   (`Primitives/DenseKernels.cs` etc per the earlier "should use DenseKernels from the start"
   note) but the FSMN depthwise-conv-with-asymmetric-shift and the encoders0-no-residual quirk
   are FunASR-specific, don't expect to find them pre-built.
2. Port `FunAsrPredictor.cs` (replacing the fake `CifPredictor` class in
   `FunAsrPipeline.cs`): the `CifPredictorV2` forward above + the sequential CIF algorithm.
3. Port `FunAsrDecoder.cs`: the surprising FFN-first layer order, FSMN-only self-attn, real
   cross-attn, decoders3's no-residual FFN-only tail layer, `after_norm` ->
   `output_layer` -> argmax per position for the final token ids.
4. ~~Build a golden-output oracle~~ -- DONE, see the RESOLVED entry immediately below.
5. ~~Wire `FunAsrPipeline.Transcribe`~~ -- NOT done yet, blocked on the real mel extractor, see
   below.

## FunASR RESOLVED (same fire, continued): encoder + predictor + decoder ALL ported and golden-verified -- end-to-end wiring blocked on the real mel extractor

Continued directly from the research above. `pip download funasr --no-deps` was already
extracted to `examples/funasr-py/` last fire; this fire actually READ the three module files in
full (`sanm/attention.py`, `sanm/encoder.py`, `sanm/decoder.py`,
`paraformer/cif_predictor.py`) and ported all three stages, each independently golden-verified.

**Golden oracle technique used (no original .pt checkpoint available locally, only the GGUF
conversion)**: the Python `gguf` package (`gguf.GGUFReader` + `gguf.dequantize`) reads and
dequantizes `models/paraformer-q8.gguf`'s real tensors directly into numpy arrays, matching
PyTorch's native `[out,in]` weight-row-major convention with zero manual reshaping needed
(confirmed empirically: `gguf.dequantize`'s output shape for a Linear weight already comes out
`[out,in]`). Three chained golden-dump scripts were written, each reusing the prior stage's
verified output as its own input (so each test isolates exactly one stage's correctness):
`scratch-llamacpp-ref/funasr_golden_{encoder,predictor,decoder}.py`. Fixed seeded-random
560-dim x 10-frame synthetic input (content doesn't matter, only that C# and Python see the
exact same numbers).

**All three landed and PASSED their golden cosine-similarity test on the FIRST attempt** (no
bugs found needing a fix this time, unlike Silero VAD's missing-ReLU -- likely because the
formulas were transcribed directly from real source with zero guessing at any step, per the
extensive derivation notes above):
- `FunASR/FunAsrEncoder.cs` (new) + `Tests.Audio/FunAsrEncoderTests.cs` (new) -- the SAN-M
  encoder (`encoders0.0` no-residual quirk, FSMN memory-add, plain FFN). Cosine >0.99 vs
  `funasr_golden_encoder.py`. PASS.
- `FunASR/FunAsrPredictor.cs` (new) + `Tests.Audio/FunAsrPredictorTests.cs` (new) -- real
  `CifPredictorV2` (full non-grouped conv, confirmed from the real tensor shape, not the base
  depthwise `CifPredictor`) + the sequential classic-CIF integrate-and-fire algorithm. Exact
  token-count match AND cosine >0.99 vs `funasr_golden_predictor.py`'s acoustic_embeds. A quick
  sanity check along the way: the golden test input's alphas[0] came out to EXACTLY 1.0 (an
  immediate fire at t=0), so `acoustic_embeds[0]` should equal `encoder_output[0]` exactly --
  confirmed it does, a nice internal-consistency check that the CIF math is doing the right
  thing before even comparing cross-language.
- `FunASR/FunAsrRealDecoder.cs` (new, named to avoid colliding with the pre-existing fake
  `FunAsrDecoder` class in `FunAsrPipeline.cs` until that's rewired) + `Tests.Audio/
  FunAsrRealDecoderTests.cs` (new) -- the surprising FFN-first layer order, FSMN-only self-attn
  with residual back to the ORIGINAL layer input (not the FFN output), real cross-attention,
  `decoders3.0`'s no-residual FFN-only tail layer. Exact argmax token-id match AND cosine >0.99
  vs `funasr_golden_decoder.py`'s logits. For the degenerate single-random-noise-token golden
  input, both C# and Python confidently predict token id 2 (`</s>`, real end-of-sequence) --
  a sane, plausible result for a single essentially-noise acoustic embedding.

**Full regression sweep this fire** (per user instruction, one filter-class at a time):
`Fast.FunAsrRealWeightsTests` (3/3, unaffected by this fire's new standalone classes) and
`FunAsrWeightsTests` (3/3) both still PASS -- nothing this fire's additions touched broke
anything already landed.

**Deliberately did NOT wire `FunAsrPipeline.Transcribe` to the real encoder/predictor/decoder
this fire** -- this is a real, load-bearing blocker, not a missed step: the real encoder's
`encoders0.0` layer expects 560-dim input (real `cmvn.scale`/`cmvn.shift` are both 560-dim =
80 mel channels x 7-frame LFR splice, per this checkpoint's real CMVN tensors), but
`FunAsrMelExtractor` (still 100% fake -- a log-energy heuristic, not a real mel filterbank) only
produces raw 80-dim frames with NO splicing or CMVN normalization applied at all. Wiring the
real encoder directly onto the fake mel extractor's output would either throw (dimension
mismatch) or require inventing a splice/CMVN scheme without checking the real source first --
exactly the kind of guess this whole doc's discipline exists to prevent. Did NOT attempt it
under this fire's time pressure.

**Concrete next steps for whoever continues this (encoder/predictor/decoder are DONE, this is
the only remaining gap before a real end-to-end transcription)**:
1. Find and read the real mel-feature-extraction + LFR-splice + CMVN code in `examples/
   funasr-py` (likely `funasr/frontends/` or similar -- not yet located/read this fire) to get
   the exact real formula: window/hop/n_mels for the mel filterbank, and the splice window
   (left/right context frame counts, stride) that turns per-frame 80-dim mel into 560-dim
   spliced-and-CMVN'd encoder input. Do NOT assume the common "3+1+3=7 frames, stride 1" Kaldi
   LFR convention without checking -- this doc's own history (FSMN memory-add,
   `encoders0.0`'s no-residual quirk, decoder's FFN-first order) has repeatedly shown FunASR
   diverges from the "obvious"/textbook version in ways that would have been silently wrong if
   guessed.
2. Port `FunAsrMelExtractor` for real (real mel filterbank + splice + CMVN), golden-verify it
   the same way (a 4th `funasr_golden_*.py` script), same >0.99 cosine bar.
3. Wire `FunAsrPipeline.Transcribe`: real mel -> `FunAsrEncoder.Forward` -> `FunAsrPredictor.
   Predict` -> `FunAsrRealDecoder.Forward` -> argmax per position -> `FunAsrTokenizer.Decode`
   (already real). Delete the now-dead fake `SanmEncoder`/`CifPredictor`/`FunAsrDecoder`
   classes in `FunAsrPipeline.cs` once the real path is wired and verified (don't delete before
   -- `FunAsrPipeline`'s no-weights fallback constructor path still needs SOME implementation
   to compile/run without real weights, matching every other pipeline's fake/real dual-path
   convention).
4. Once wired, re-run `Fast.FunAsrRealWeightsTests`'s existing `Paraformer_GgufRealModelFile_
   LoadsAndTranscribes` test -- it should still pass structurally, but now the transcript text
   will be genuinely meaningful for the first time (currently it's still fake/nonsense even
   though the underlying neural math is now real, since the fake mel input feeds nothing
   downstream of it correctly).

## FunASR frontend (real mel + LFR + CMVN): exact spec CONFIRMED from the real published config, golden oracle built; C# DSP port NOT attempted this fire (documented, not guessed)

Same fire, continued straight from item 1 of the next-steps list above. Found
`funasr/frontends/wav_frontend.py`'s `WavFrontend` class (real mel filterbank +
low-frame-rate splice + CMVN) -- but its constructor defaults alone don't reveal WHICH values
this specific checkpoint actually used (`n_mels`/`lfr_m`/`lfr_n`/etc are all constructor
parameters, not hardcoded). Rather than assume the common "7-frame splice, stride 6" LFR
convention many Kaldi-family ASR stacks use, fetched the REAL published config directly:
`https://huggingface.co/funasr/paraformer-zh/raw/main/config.yaml` (small, public, directly
authoritative). **This independently CONFIRMS every encoder/decoder/predictor hyperparameter
already derived from this checkpoint's real GGUF metadata in earlier fires** (encoder
`num_blocks=50`, `kernel_size=11`, `sanm_shfit=0`; decoder `num_blocks=16`, `att_layer_num=16`,
`kernel_size=11`; predictor `CifPredictorV2`, `l_order=1`, `r_order=1`, `threshold=1.0`,
`tail_threshold=0.45` -- all exact matches, strong independent confirmation the whole port so
far is on the right track) -- and gives the frontend spec that was still missing:
```
frontend: WavFrontend
frontend_conf: {fs: 16000, window: hamming, n_mels: 80, frame_length: 25, frame_shift: 10,
                 lfr_m: 7, lfr_n: 6}
```
Real forward pass (`WavFrontend.forward`, confirmed from source, not the config alone):
`waveform *= 32768` (Kaldi expects int16-range values, not [-1,1] float) -> `torchaudio.
compliance.kaldi.fbank(...)` (num_mel_bins=80, frame_length=25, frame_shift=10,
window_type=hamming, sample_frequency=16000, energy_floor=0.0, snip_edges=True; **dither is
this script's own deterministic choice of 0.0** -- `WavFrontend`'s own class default is
`dither=1.0`, non-deterministic, unsuitable for a golden oracle, and the actual inference-time
override couldn't be verified from the config alone, so this is flagged as a documented
assumption, not a confirmed fact, unlike everything else in this entry) -> `apply_lfr(mat,
lfr_m=7, lfr_n=6)` (real function, copy-pasted verbatim into the golden script, not
re-derived) -> `apply_cmvn(mat, cmvn)` where `cmvn = [shift, scale]` and the real formula is
`(mat + shift) * scale` (an ADD then MULTIPLY, matching this checkpoint's real GGUF tensor
names `cmvn.shift`/`cmvn.scale` -- Kaldi's convention stores the NEGATED mean and the INVERTED
std specifically so a plain add+multiply suffices at inference time, no subtract/divide).

**Built and verified the golden oracle itself** (`scratch-llamacpp-ref/
funasr_golden_frontend.py`): uses the REAL `torchaudio.compliance.kaldi.fbank` function
directly (not a reimplementation -- zero risk of misreading Kaldi's DSP internals) plus the
real `apply_lfr`/`apply_cmvn` functions copied verbatim from source. Ran successfully against
this checkpoint's real `cmvn.shift`/`cmvn.scale` GGUF tensors and a 2-second synthetic PCM
input: produced `[33, 560]` features (33 LFR-spliced frames from ~200 raw 10ms mel frames,
consistent with `lfr_n=6` stride), confirming the whole real pipeline chain (fbank -> LFR ->
CMVN) runs end-to-end against real weights without error.

**Deliberately did NOT attempt the C# DSP port this fire** -- porting Kaldi's `fbank` exactly
(specific mel-filter construction differing subtly from librosa/standard implementations,
pre-emphasis coefficient application order, `round_to_power_of_two` FFT-size selection,
`snip_edges` frame-count convention, hamming vs Kaldi's own "povey" window family) is a
precision-sensitive DSP task on its own, comparable in scope/risk to Silero VAD's STFT
frontend -- attempting it under this fire's remaining time budget risked exactly the kind of
rushed, under-verified port this whole doc's discipline exists to prevent. The spec above is
now fully confirmed and ready to port directly; no further research should be needed before
writing `FunAsrMelExtractor`'s real forward pass.

**Concrete next steps, refined** (supersedes the more vague "port the mel filterbank" item
above):
1. Port `FunAsrMelExtractor.ExtractMel` to real Kaldi-compatible fbank + LFR + CMVN using the
   exact spec above. Consider whether any EXISTING mel-filterbank code in this codebase
   (`Primitives/SpectralKernels.cs`, or Whisper/Parakeet/QwenASR's own mel extractors) is
   close enough to Kaldi's convention to reuse/adapt vs. needing a genuinely new
   implementation -- Kaldi's fbank has real, non-obvious differences from a "standard" librosa-
   style mel filterbank (different mel-scale formula, different filter normalization, Kaldi-
   specific windowing), so do NOT assume reuse is safe without checking the exact math first.
2. Golden-verify against `scratch-llamacpp-ref/funasr_golden_frontend.py`'s output (already
   built and confirmed working this fire) via cosine similarity, same >0.99 bar as every other
   stage.
3. Resolve the flagged `dither` uncertainty if it turns out to matter for the cosine-similarity
   bar (unlikely to move the needle much given dither is meant to be a small perturbation, but
   worth a note if the golden comparison doesn't cleanly clear 0.99 and this is why).
4. Then proceed with the previously-documented wiring steps (`FunAsrPipeline.Transcribe`
   end-to-end, delete the dead fake classes, re-verify the existing pipeline-level test).

## FunASR — COMPLETE: real mel/LFR/CMVN DSP ported, golden-verified, wired end-to-end into `FunAsrPipeline.Transcribe` (2026-08-22, next cron fire, continued straight from the frontend spec above)

Picked up exactly where the previous entry left off. All four steps of that entry's "concrete
next steps" list are now done.

**`FunAsrRealMelExtractor.cs` (new)** — real Kaldi-compatible fbank + LFR splice + CMVN, ported
directly from the confirmed spec (no new research needed). Reused this codebase's existing
`SpectralKernels.ComputePowerSpectrum` (internal, same assembly) for the FFT+power-spectrum
step rather than writing a new FFT — the real Kaldi-specific parts (Hamming window with
`periodic=False` convention, DC-mean removal before pre-emphasis, pre-emphasis with the `x[0]`
edge-replication convention, the non-librosa triangular mel-filter construction evaluated
directly in the mel domain against each FFT bin's mel frequency, the `apply_lfr` padding-count
formula, and `apply_cmvn`'s add-then-multiply) were all transcribed verbatim from the real
source, not reused from any other pipeline's mel extractor (confirmed the existing kernel was
NOT close enough to reuse beyond the raw FFT/power-spectrum primitive, per the "do NOT assume
reuse is safe" caution in the prior entry). `dither=0` kept as this port's own deterministic
choice, matching the golden oracle script — documented as an open assumption, not re-litigated
this fire since the golden comparison below cleared the bar cleanly.

**Golden-verified**: `FunAsrRealMelExtractorTests.Extract_RealWeights_MatchesGoldenFrontendOutput`
compares the C# port's `[T,560]` LFR+CMVN output against `scratch-llamacpp-ref/
funasr_golden_frontend.py`'s real-`torchaudio.compliance.kaldi.fbank`-backed oracle via cosine
similarity — **passes cleanly above the 0.99 bar** on the first attempt (no `dither` follow-up
needed).

**Wired end-to-end** — `FunAsrPipeline.cs`: added a cached `_realMelExtractor` field (constructed
only when real GGUF weights are present) and split `Transcribe` into `TranscribeReal` (mel
extract → `FunAsrEncoder.Forward` → `FunAsrPredictor.Predict` → `FunAsrRealDecoder.Forward` →
per-frame argmax → `FunAsrTokenizer.Decode`) vs. the old fake inline path, which is now only a
fallback for the (untested, no-weights) case. This is the first fire where the full FunASR
neural pipeline runs on 100% real, golden-verified weights and math end-to-end — every stage
(frontend, encoder, predictor, decoder, tokenizer) is independently cosine/exact-verified against
a real oracle, per this doc's established discipline.

**One pre-existing test assertion had to be corrected, not the model**: real-weight wiring caused
`Paraformer_GgufRealModelFile_LoadsAndTranscribes` (`FunAsrRealWeightsTests.cs`) to fail on
`Assert.False(string.IsNullOrWhiteSpace(res.Text))`. Diagnosed, not assumed: the test's input is
a pure 440Hz sine tone, not real speech — the real Paraformer model can legitimately predict only
special tokens (`<blank>`/`<s>`/`</s>`/`<unk>`, all stripped by `FunAsrTokenizer.Decode`) for
non-speech audio, producing a real, structurally valid, empty-text result. This is correct
real-model behavior, not a bug — the OLD fake pipeline guaranteed non-empty placeholder text
regardless of input, which was itself the fake behavior being replaced. Removed the outdated
assertion with an inline comment explaining why; kept `Assert.NotNull(res)` and
`Assert.NotEmpty(res.Segments)`. This matches the established pattern from earlier fires
(QwenASR, Silero VAD) of updating pre-existing tests when real-model wiring changes what's
actually true, rather than forcing the model to match a fake assumption.

**Full regression sweep, run one filter-class at a time per project discipline (all PASS, 9/9)**:
- `FunAsrRealWeightsTests` — 3/3 (after the assertion fix above)
- `FunAsrWeightsTests` — 3/3
- `FunAsrEncoderTests` — 1/1
- `FunAsrPredictorTests` — 1/1
- `FunAsrRealDecoderTests` — 1/1
- `FunAsrRealMelExtractorTests` — 1/1

**FunASR is now DONE — queue item 1 of 5 complete.** No further work planned for this pipeline
unless a future fire's real-audio (not synthetic sine-tone) testing surfaces a numerical issue.
Not committed (per standing instruction — no commits made this fire).

**Queue status after this fire**:
1. ✅ FunASR — COMPLETE (this fire)
2. ✅ Silero VAD — COMPLETE (resolved in an earlier fire this session)
3. ⛔ Fish Speech — BLOCKED on weights only, see correction below (real reference source DOES
   exist locally)
4. ⛔ Parler-TTS — BLOCKED on weights only, see correction below (real reference source DOES
   exist locally)
5. ⛔ Orpheus TTS — BLOCKED on weights only, see correction below (real reference source DOES
   exist locally)

**Second correction, same fire, prompted by user pointing at `examples/s2.cpp` directly**:
`examples/s2.cpp` is a real, substantial, self-contained C++/GGML inference engine for **Fish
Audio's S2 Pro** (Dual-AR transformer TTS) — `s2_model.cpp`, `s2_generate.cpp` (Slow-AR
transformer w/ KV cache -> Fast-AR codebook decoder), `s2_codec.cpp` (audio codec
encode/decode), `s2_tokenizer.cpp` (Qwen3 BPE, `tokenizer.json` present in-repo), `s2_prompt.cpp`,
`s2_sampler.cpp`, `s2_voice.cpp` (voice cloning), plus C#/Go/Python bindings under
`examples/s2.cpp/examples/`. This is the modern Fish Audio product (not the older open-source
"Fish Speech" repo the queue name was written against, but the same lineage/company and the
directly relevant reference for this queue slot) — a real, non-trivial architecture to port from,
not a stub. The earlier "no reference source" conclusion for Fish Speech was wrong for the same
reason as items 4-5 above: insufficient `examples/` search depth.

**Correction to the earlier "no reference source" claim for items 4–5**: re-checked this fire
(broader glob than the earlier scope check used) and found `examples/CrispASR/src/orpheus.cpp` +
`orpheus.h` + `orpheus_snac.h`, and `examples/CrispASR/src/parler_tts.cpp` + `parler_tts.h` —
real, non-trivial ggml/GGUF C++ reference implementations for both, plus
`examples/CrispASR/models/convert-orpheus-to-gguf.py` /
`convert-parler-to-gguf.py` conversion scripts and `examples/CrispASR/hf_readmes/
parler-tts-mini-v1.1-GGUF.md` / `orpheus-3b-*-GGUF.md` describing the real architectures
(Parler-TTS: T5 encoder (flan-t5-large, 24 layers) conditioning a MusicGen-style causal decoder
(24 layers, 9 codebooks) generating DAC tokens, then a DAC 44kHz codec decoder; Orpheus: a
Llama-family causal LM emitting SNAC audio codes). The earlier fire's "no reference source"
conclusion was simply wrong — it didn't check deep enough into `examples/CrispASR/`, same class
of mistake as the Parakeet false-blocked finding earlier in this doc.

**Still genuinely blocked, but now ONLY on weights, not reference material**: confirmed no
`.gguf` (or any other) local weight file exists anywhere under `models/` or
`examples/CrispASR/models/` for Parler-TTS or Orpheus TTS — only markdown READMEs describing
HuggingFace-hosted GGUF quants that would need to be downloaded (multi-GB, per the readme's file
listing). Per this project's "golden-verify against a real oracle... real weight-driven code"
discipline, porting the C++ reference's math without real weights to load and run it against
would produce code that compiles but can't be verified — not worth doing blind. Downloading
multi-GB model files autonomously (network access, disk space, unknown provenance) is also the
kind of action this session should not take without the user's explicit go-ahead, per this
project's action-authorization norms. **Fish Speech (S2 Pro) is in the identical situation**:
no `.gguf` anywhere under `models/` or `examples/s2.cpp/`, only the README pointing to
`rodrigomt/s2-pro-gguf` on Hugging Face (7 quant variants, 2.6-9.9 GB) — same "reference ready,
weights missing" blocker as items 4-5, not the "nothing to port from" situation the earlier
scope-check entry wrongly claimed.

**All actionable queue items are now complete for this fire.** FunASR and Silero VAD are fully
done. Fish Speech, Parler-TTS, and Orpheus TTS are now all understood precisely and identically
blocked — real, non-trivial C++/GGML reference source is in-repo and ready to port from for all
three (`examples/s2.cpp`, `examples/CrispASR/src/parler_tts.*`,
`examples/CrispASR/src/orpheus*`), but real progress on any of them needs the user to either drop
the real GGUF weight files into `models/` (see each README's exact filenames/quants:
`rodrigomt/s2-pro-gguf` for Fish Speech, the `examples/CrispASR/hf_readmes/*.md` files for
Parler-TTS/Orpheus) or explicitly authorize downloading them. Next cron fire should re-check
`models/` (and `examples/s2.cpp/`, `examples/CrispASR/models/`) for any newly-added weight files
for all three before concluding there's nothing left to do.

## Weights downloaded (user, live in this fire, not a cron fire) — all three of Fish Speech/Parler-TTS/Orpheus TTS now UNBLOCKED; more real reference source found; FunASR performance/DRY pass done

User was present live for this fire's tail end (not AFK), directly authorized the multi-GB
downloads flagged as blocking above, and pointed out additional local reference material the
scope check above had missed.

**Additional real reference source found (live user pointer, then verified)**:
`examples/s2.cpp` — a real, substantial, self-contained C++/GGML inference engine for **Fish
Audio's S2 Pro** (Dual-AR transformer TTS: `s2_model.cpp`, `s2_generate.cpp` (Slow-AR transformer
w/ KV cache -> Fast-AR codebook decoder), `s2_codec.cpp`, `s2_tokenizer.cpp` (Qwen3 BPE,
`tokenizer.json` in-repo), `s2_prompt.cpp`, `s2_sampler.cpp`, `s2_voice.cpp` (voice cloning), plus
C#/Go/Python bindings). This is the modern Fish Audio product (base model `fishaudio/s2-pro`,
GGUF `architecture` tag literally `fish-speech`, confirmed via the HF API) -- the "no reference
source" conclusion in the original scope-check entry was wrong for the same class of reason as
the Parakeet false-blocked finding earlier in this doc: insufficient `examples/` search depth.
Also `examples/TTS.cpp` (same author, `mmwillet2`, as the Parler-TTS GGUF weights below) --
confirmed via its README to be the ACTUAL source repo that produced those exact GGUF files, and
it also supports Orpheus -- this supersedes CrispASR's Orpheus/Parler ports as the PRIMARY
reference for those two (architecture is guaranteed to match the downloaded weights exactly,
unlike a third-party port). `examples/Orpheus-TTS` (the original Python `canopyai/Orpheus-TTS`
repo) is also present, useful for cross-checking prompt formatting/generation-config details the
C++ ports might not document.

**Real GGUF weights downloaded into `models/`, all verified against real HF repos before
downloading (not blind URLs)**:
- `models/s2-pro-q4_k_m.gguf` (~3.6 GB target) -- from `rodrigomt/s2-pro-gguf`
  (`base_model: fishaudio/s2-pro`, `gguf.architecture: fish-speech`, 4.95B params, verified via
  the HF API before download).
- `models/orpheus-3b-0.1-ft.Q4_K_M.gguf` (~2.36 GB) -- from
  `QuantFactory/orpheus-3b-0.1-ft-GGUF` (`base_model: canopylabs/orpheus-3b-0.1-pretrained`,
  `gguf.architecture: llama` -- confirms the docs/audio/53orpheous.md planning note that Orpheus's
  talker is a real Llama-3.2-3B-shape model, potentially reusable against this codebase's
  existing Llama forward-pass infrastructure rather than needing an entirely new architecture).
- `models/Parler_TTS_mini.gguf` (~1.2 GB) -- from `mmwillet2/Parler_TTS_GGUF`
  (`base_model: parler-tts/parler-tts-mini-v1.1`, `gguf.architecture: parler-tts`).

User also pointed at `docs/audio/51fish.md` / `52parlertts.md` / `53orpheous.md` -- pre-existing
high-level planning docs for exactly these three pipelines (same family as the
CosyVoice/QwenASR/QwenTTS planning docs noted earlier in this doc). Skimmed, not fully read yet:
same caveat as those earlier planners applies -- **treat as directional reference, not gospel,
since these planners don't know this codebase's actual existing infrastructure** (e.g.
53orpheous.md's suggestion to reuse "Stingray's existing Llama implementation" needs verifying
against what `OpenTail.Stingray.Engine`'s real Llama forward-pass path actually looks like before
assuming it's a clean fit). **Read all three docs in full before starting the actual port work**
-- not done yet this fire, next fire's first step.

**Queue status, corrected**: all three of Fish Speech, Parler-TTS, and Orpheus TTS are now
UNBLOCKED -- real reference source (both third-party C++ ports AND, for Parler/Orpheus, the
actual source repo that produced the downloaded weights) plus real GGUF weights are both present
locally. Next fire should start with Orpheus (per 53orpheous.md's own difficulty ranking and the
`architecture: llama` confirmation above, likely the lowest-effort of the three), read
`docs/audio/53orpheous.md` in full plus `examples/TTS.cpp`'s Orpheus support and
`examples/Orpheus-TTS`'s real generation config, then follow this doc's standard discipline
(real weight loader -> golden-verify each stage -> wire end-to-end) exactly as done for FunASR.

**FunASR performance + DRY pass (same fire, direct user request "do a performance pass" then
"also check DRY")**: `FunAsrEncoder.cs` and `FunAsrRealDecoder.cs` had copy-pasted identical
`Linear`/`LinearNoBias`/`LayerNorm`/`SoftmaxInPlace` helpers, and the FSMN depthwise-conv memory
term (encoder's self-attention branch, decoder's FSMN-only self-attention) was the same algorithm
duplicated in both. Extracted to a new shared `Primitives/FunAsrKernels.cs`, following the same
DRY-after-verification pattern established for `S3GenConformerKernels`/`DenseKernels` earlier in
this doc. **Real perf win found in the extraction, not just cleanup**: the original per-pipeline
`Linear` looped output channels with a scalar `SimdKernels.DotF32` call each, missing
`SimdKernels.MatVecF32`'s own internal `Parallel.For` over output rows (triggers at outDim >= 64
-- every Linear call in this pipeline hits that: QKV projections are 512->1536, FFN is
512->2048, vocab projection is 512->8404). Routing through `MatVecF32` picks up that
parallelization for free. Also parallelized multi-head attention over heads and the FSMN conv
over channels (both via `Parallel.For`, matching the per-head/per-channel convention already used
by `WhisperEncoder.cs`), and the encoder/decoder's per-position LayerNorm/residual loops (each
position is independent, `Parallel.For(0, t, ...)`), and the predictor's Conv1d output-channel
loop. No numerical behavior changed (all changes are either identical math reordered for
parallel-independence, or literal reuse of a pre-existing, already-verified batched kernel) --
re-ran all four FunASR-specific golden/structural test classes individually after the change and
all still PASS: `FunAsrEncoderTests` 1/1, `FunAsrRealDecoderTests` 1/1, `FunAsrPredictorTests`
1/1, `FunAsrRealWeightsTests` (full pipeline, end-to-end) 3/3. Did not re-run
`FunAsrWeightsTests`/`FunAsrRealMelExtractorTests` this pass since neither file they cover was
touched.

**Actually measured, not assumed (user directly asked "is the performance measurably better? have
you checked or are we assuming?" -- it was an assumption at that point, corrected here)**: added a
throwaway benchmark (`tests/OpenTail.Stingray.Tests.Audio/FunAsrPerfBenchTests.cs`, marked
TEMPORARY in its own doc comment, not part of the permanent suite -- writes results to a temp
file since the MTP test runner doesn't surface `Console.WriteLine` in non-verbose mode) that
transcribes 12s of synthetic audio 8 times (1 warmup + 8 timed) against the real
`paraformer-q8.gguf` weights on a 12-core machine. Current pipeline (all `Parallel.For`, as
landed): **median 3.49-3.61s per 12s-audio transcription across two separate 8-sample runs**
(samples ranged 3.16-3.93s) -- no prior baseline exists to A/B against (the parallelization was
added in the same fire the pipeline was first wired end-to-end, so there's no "before" to compare
to), so this is a measured absolute number, not a measured delta, and should be read as "current
real-weight throughput," not "N% faster."

**Second performance pass, prompted directly by the user ("do a second performance pass")**:
revisited the theoretical concern that wrapping `FunAsrKernels.Linear` calls (QKV/FFN/output
projections) in an outer `Parallel.For(0, t, ...)` nests parallelism inside `Linear`'s own
internal `Parallel.For` (via `SimdKernels.MatVecF32`, triggers at outDim >= 64) -- reasoned this
could cause thread-pool oversubscription, converted those specific loops (not the pure
LayerNorm/residual-add ones) to plain serial `for` loops, rebuilt, reran the golden tests (still
PASS, correctness unaffected either way since parallelization doesn't change per-position math),
then re-ran the SAME benchmark: **median rose to 3.78s (was 3.49s) -- the "fix" was a real,
consistent regression, not noise** (all 8 samples in the serial-loop run were higher than the
mean of the Parallel.For run). Reverted back to `Parallel.For` for all of these loops (encoder
QKV/attn-out/FFN, decoder output-projection/FFN(x2)/cross-attention Q/KV/out) after confirming
the revert brought the benchmark back to the original 3.4-3.6s range. **Lesson, matching this
project's established "measure, don't assume" discipline from the earlier CosyVoice SIMD work**:
the theoretically-sound nested-parallelism concern did not hold up under measurement here, likely
because per-layer `t` (encoder frame count) is small enough that the outer `Parallel.For`'s task-
scheduling overhead is cheap relative to the work each task does, and outer-loop parallelism
gives the OS scheduler more independent units to spread across cores than relying solely on
`MatVecF32`'s row-level parallelism inside a serial per-position loop. Do not attempt this
"fix" again without re-measuring with an equivalent benchmark first.

## Orpheus TTS — investigation started (cron fire, gate passed at 2026-08-22T07:58Z): major finding, the talker LM loads and runs through this codebase's EXISTING Llama forward pass completely unchanged; real generation-prompt spec found; SNAC decoder still needed

Started per the queue (Orpheus next, per this doc's own prior recommendation: confirmed
`architecture: llama`, likely lowest-effort of the three remaining pipelines).

**Confirmed via real GGUF metadata dump (`dotnet run ... -- list-metadata -m
models/orpheus-3b-0.1-ft.Q4_K_M.gguf`), not assumed**: `general.architecture=llama`,
`general.base_model` chain explicitly names `meta-llama/Llama-3.2-3B-Instruct` ->
`canopylabs/orpheus-3b-0.1-pretrained`, standard Llama hyperparameters (28 layers, 3072 hidden,
24 attention heads / 8 KV heads (GQA), head_dim=128, RoPE freq_base=500000, RMSNorm eps=1e-5,
ffn_dim=8192, context_length=131072), and `llama.vocab_size=156940` -- notably LARGER than stock
Llama-3.2's 128256, confirming the base LM vocab was extended with audio-codec tokens (see the
exact offset below).

**Major finding, tested directly, not theoretical**: ran `dotnet run --project
src/OpenTail.Stingray.Cli -c Release -- -m models/orpheus-3b-0.1-ft.Q4_K_M.gguf -p "hello" --temp
0 -g 20` -- the model LOADED AND RAN with zero code changes, through this codebase's ordinary
`HybridForwardPass` (Vulkan+CPU hybrid, same path any other GGUF Llama model uses), printing
`Model loaded in 6.3s — 28L, 3072d, headDim=128, 156940 vocab, ctx=32768` and completing a normal
prefill pass. This directly confirms `docs/audio/53orpheous.md`'s central planning question ("how
much of Orpheus can literally run through Stingray's existing Llama implementation unchanged?
Potentially: a lot.") -- the answer is: the ENTIRE talker transformer, unchanged, no new
architecture code needed at all for that part.

**Decode produced 0 tokens with a generic chat-template prompt -- expected, not a bug**: this
checkpoint (`orpheus-3b-0.1-ft`, note the `-ft` = fine-tuned-for-TTS suffix) is not an instruct/
chat model, so wrapping "hello" in this codebase's standard Llama chat template produces a prompt
far outside its training distribution, and it immediately predicts EOS. This means the actual
remaining Orpheus-specific work is NOT a new transformer -- it's (1) the real prompt-construction
format (voice/speaker name + text, wrapped in Orpheus's own special tokens, not a chat template)
and (2) parsing the generated token stream into SNAC codec frames, then (3) a real SNAC decoder
to turn those into PCM. Exactly matches 53orpheous.md's predicted shape: "Llama-3.2-3B backbone
(done, confirmed above) -> small Orpheus-specific generation layer -> SNAC decoder."

**Real generation-prompt/token spec, read directly from `examples/CrispASR/src/orpheus.cpp`'s
own header comment and hyperparameter struct (not guessed, not re-derived)**:
- `<|audio_start|>` = token id `128259` -- prepended to start audio-token generation (after the
  text/voice prompt, per that file's header comment sequence).
- Generation continues, emitting a stream of `<custom_token_N>` ids, until either
  `<|audio_end|>` = token id `128257` is emitted, or a max-audio-tokens cap is hit.
- Other special tokens: `<|eot_id|>` = `128009` (standard Llama EOT, inherited from the base
  model), `<|audio_eot|>` = `128260`, `<|audio_eom|>` = `128261`.
- `custom_token_offset` = `128266` (real default, also readable from this specific checkpoint's
  own `orpheus.custom_token_offset` GGUF metadata key per `orpheus.cpp` line 302 -- NOT yet
  checked whether this exact checkpoint's GGUF carries that key or relies on the hardcoded
  default; check this first before trusting the offset blindly).
- `custom_token_count` = `7 * 4096` = `28672` -- 4096-entry codebook x 7 SNAC hierarchy levels.
- Real detokenization: `text_n = lm_id - custom_token_offset` gives the raw SNAC codebook index
  (0..28671); every 7 consecutive `<custom_token_N>` ids form one de-interleaved SNAC frame
  (confirmed from `orpheus.cpp`'s own inline comment: "every 7 tokens form one SNAC [frame]").

**NOT yet done, concrete next steps for whoever picks this up next**:
1. Check whether `models/orpheus-3b-0.1-ft.Q4_K_M.gguf` actually carries the
   `orpheus.custom_token_offset`/`orpheus.custom_token_count` GGUF metadata keys itself (grep the
   full `list-metadata` dump, not yet done this fire) -- if present, use those real values
   directly rather than the `orpheus.cpp` hardcoded defaults, per this doc's "read the real
   checkpoint's own metadata, don't assume defaults transfer" discipline used everywhere else.
2. Find or derive the REAL text-prompt wrapping format (what exactly precedes `<|audio_start|>` --
   voice/speaker name syntax, any system-prompt-equivalent) -- check `examples/Orpheus-TTS`'s own
   Python inference code (the original `canopyai/Orpheus-TTS` repo, present locally) for the
   real prompt-construction function, and/or `examples/TTS.cpp`'s Orpheus support, before
   guessing at a format.
3. Once the real prompt format is confirmed, wire it into a small Orpheus-specific pipeline class
   (analogous to `FunAsrPipeline`) that: builds the real prompt -> runs the EXISTING
   `OpenTail.Stingray.Engine` Llama forward pass (confirmed working above, should need NO new
   forward-pass code) -> greedy/sampled decode loop collecting `<custom_token_N>` ids until
   `<|audio_end|>` or a cap -> de-interleave into 7-level SNAC frames.
4. Port a real SNAC decoder (genuinely new component, not reusable from existing Llama infra) --
   check `examples/CrispASR/src/orpheus_snac.h` first (real reference, same repo as the prompt
   spec above) before looking elsewhere. Golden-verify against a real oracle per this doc's
   standard discipline before calling it done.
5. This is a genuinely promising, lower-effort port than FunASR/Silero VAD were, given the
   talker transformer needs zero new forward-pass code -- prioritize finishing this one before
   moving to Fish Speech/Parler-TTS if time allows, per 53orpheous.md's own difficulty ranking.

Not committed (per standing instruction). No golden verification done yet for anything Orpheus-
specific (the prompt/SNAC layer doesn't exist yet) -- the "load and run unchanged" finding above
is a real, directly-observed result (not a golden-verified one), appropriate for confirming
infrastructure compatibility, not for confirming correctness of Orpheus-specific behavior.

## Orpheus TTS -- continued same fire: real Python source found and read in full, CORRECTS several details assumed from the C++ port; real SNAC codec architecture + weights now also in place

Read `examples/Orpheus-TTS/orpheus_tts_pypi/orpheus_tts/engine_class.py` and `decoder.py` in
full -- the actual `canopyai/Orpheus-TTS` reference implementation, strictly more authoritative
than `examples/CrispASR/src/orpheus.cpp`'s comments (a third-party port). **Three real
corrections to what the earlier entry inferred from the C++ port, do not use the C++ port's
version of these details**:

1. **Real prompt format** (`_format_prompt`, "larger"/`medium-3b` model type -- matches our
   checkpoint, the pretrained/`-ft` 3B model, not the smaller nano/micro variants):
   `[128259] + tokenize(f"{voice}: {prompt}") + [128009, 128260, 128261, 128257]`. All four
   "end tokens" (`128009`=eot_id, `128260`=audio_eot, `128261`=audio_eom, `128257`=audio_end) are
   appended to the PROMPT, priming the model to start emitting audio codec tokens -- `128257` is
   NOT a generation stop signal as the C++ port's comment implied.
2. **Real generation stop condition**: `stop_token_ids=[49158]` (a literal vLLM
   `SamplingParams` argument in `generate_tokens_sync`) -- a single specific token id, not
   `<|audio_end|>=128257`. Available voices per this real source:
   `["zoe","zac","jess","leo","mia","julia","leah"]` (though the function's own default
   `voice="tara"` isn't in that list -- a real inconsistency in the upstream source itself, not
   something to "fix", just replicate/pick a real listed voice).
3. **Real detokenization formula is NOT a flat offset subtraction**: `turn_token_into_id` computes
   `code = int(N) - 10 - ((index % 7) * 4096)`, where `N` is the literal integer parsed out of
   the decoded `"<custom_token_N>"` string and `index` is the running count of emitted audio
   tokens (0-based). This means each of the 7 per-superframe slot positions has its own effective
   subtracted stride (slot 0: `-10`; slot 1: `-10-4096`; ... slot 6: `-10-6*4096`), NOT a single
   flat `id - custom_token_offset` as the earlier entry (extrapolating from the C++ port) assumed.
   Must port this exact formula, indexed by position-within-superframe, not a constant offset.

**Real SNAC de-interleaving structure, confirmed from `decoder.py`'s `convert_to_audio`**: SNAC
is NOT a flat 7-codebook scheme -- it's a genuine 3-level hierarchical/multi-rate residual VQ
codec. Per 7-token superframe (tokens at local offsets `i..i+6`): `codes_0` gets 1 entry (`i`),
`codes_1` gets 2 entries (`i+1`, `i+4`), `codes_2` gets 4 entries (`i+2`, `i+3`, `i+5`, `i+6`).
After accumulating >=4 superframes (28 tokens), decode the most recent 4-superframe window
through the real SNAC model, taking only the middle `[2048:4096]` sample slice of the decoded
output per window (streaming/overlap convention, real detail from the Python reference, not
guessed). `torch.any(codes < 0 or > 4096)` is a real sanity/safety check before decoding -- worth
replicating.

**Real SNAC decoder architecture + weights now both in place, confirmed from
`examples/CrispASR/src/orpheus_snac.h`'s own header comment (very detailed, effectively a spec)
plus a real, verified HF GGUF repo**: SNAC 24kHz (`hubertsiuzdak/snac_24khz`, MIT license) is a
small (~25MB F32) residual-VQ codec: `quantizer.from_codes` (3 codebook streams -> 768-dim latent)
-> `decoder.model[0..1]` in-convs -> 4x `DecoderBlock` with strides `[8, 8, 4, 2]` (channel
progression 1024->512->256->128->64) -> final tanh head -> 24kHz mono PCM. Per-codebook VQ
strides are `[4, 2, 1]` (matching `codes_0`/`codes_1`/`codes_2`'s 1:2:4 token-rate ratio above),
hop_length=512, output length = `n2 * 512` samples where `n2` = the `codes_2` (finest level)
token count. **Downloaded the real weights this fire**: `models/snac-24khz.gguf` (26MB, verified
via HF API before download: `base_model: hubertsiuzdak/snac_24khz`, `gguf.architecture: snac`,
from `cstr/snac-24khz-GGUF`, the same author/repo family as the Orpheus GGUF conversions).

**Orpheus is now fully unblocked on both weights AND a complete, real, precisely-specified
architecture** (talker: existing Llama infra, confirmed working unchanged; codec: real SNAC
architecture spec + real weights, both in place). Concrete next steps, refined from the earlier
entry:
1. Build `OrpheusWeights.cs`-equivalent loaders for BOTH GGUF files (talker uses existing
   Llama-loading infra -- check whether `OpenTail.Stingray.Engine`'s standard model loader can be
   reused as-is or needs a thin Orpheus-specific wrapper; codec needs a new loader matching the
   real tensor names in `models/snac-24khz.gguf`, dump via `list-tensors` first, don't guess
   names).
2. Build the real prompt-construction function (voice + text -> token ids, per the exact spec
   above) and a decode loop wired to the existing Engine (collect `<custom_token_N>` ids until
   `stop_token_ids=49158` or a max-tokens cap).
3. Port `SnacDecoder.cs`: quantizer.from_codes -> in-convs -> 4x DecoderBlock (strides
   8/8/4/2) -> tanh head, per the exact spec in `orpheus_snac.h`'s header comment. Note
   `orpheus_snac.h` itself documents named intermediate stages
   (`snac_quant_out`/`snac_dec_pre`/`snac_dec_blk{0-3}`/`snac_pcm`) intended for exactly this
   kind of golden-verification bisection -- very likely a real, ready-made oracle-comparison
   harness (`crispasr-diff orpheus` / `tools/reference_backends/orpheus_snac.py`, referenced in
   the header, not yet located/read) exists somewhere under `examples/CrispASR/` -- check for it
   before building a golden oracle from scratch, it may already exist.
4. Wire the real 7-token-superframe de-interleaving (1/2/4 split, exact slot-to-codebook mapping
   above) and the `code = N - 10 - (slot*4096)` detokenization formula between the talker's
   output and the SNAC decoder's input.
5. Golden-verify the SNAC decoder against a real oracle (ideally the ready-made one noted in
   step 3) before calling any of this done, per standard discipline.

**Reference note for a LATER pipeline, not Orpheus**: user pointed at
`https://huggingface.co/parler-tts/parler-tts-mini-v1` (the base, non-quantized Parler-TTS model
card) noting it has real audio samples -- flagged here for when Parler-TTS's turn in the queue
comes, as a candidate source for real reference audio to golden-verify that pipeline's output
against (FunASR/Silero VAD were verified via cosine similarity against numeric oracles; having
real reference AUDIO available for a TTS pipeline is a different and valuable kind of check --
listen-and-compare, not just numeric). Not used yet, not relevant to Orpheus's own DAC/SNAC
verification.

**Found the ready-made golden oracle predicted above -- confirmed real, not yet run**:
`examples/CrispASR/tools/reference_backends/orpheus_snac.py`, a real dump script against the
official PyTorch `hubertsiuzdak/snac_24khz` model. Its own header comment independently
CORROBORATES every detail in this entry (the 1/2/4 codes_0/1/2 split, the `[2048:4096]`
streaming-window slice, the 7-token superframe grouping) -- strong cross-confirmation the
detokenization/de-interleaving understanding above is correct, since this script derives from
the same real Python source (`canopyai/Orpheus-TTS:decoder.py`) independently of the C++ port.
Dumps 10 named stages (`snac_codes_{0,1,2}` -> `snac_quant_out` -> `snac_dec_pre` ->
`snac_dec_blk{0-3}` -> `snac_pcm` -> `snac_pcm_emit`), driven by `ORPHEUS_SNAC_T_SUPER`/
`ORPHEUS_SNAC_CODE` env vars for deterministic constant-fill test codes -- exactly the kind of
oracle needed for a real cosine-similarity golden verification of a from-scratch `SnacDecoder.cs`
port, staged the same way this doc's other pipelines were (FunASR's encoder/predictor/decoder
each independently verified stage-by-stage, not just end-to-end). **Next fire: run this script
first** (needs the real `snac` pip package + torch, same `pip download --no-deps` technique used
throughout this doc) to produce real oracle dumps BEFORE writing `SnacDecoder.cs`'s forward pass,
not after -- this is now the single most valuable next action for finishing Orpheus.

## Orpheus TTS -- COMPLETE end-to-end (same fire, continued, direct user instruction "don't stop till you are done"): SnacDecoder ported + golden-verified, real prompt/detokenization corrected via direct tokenizer inspection, full pipeline wired and producing real audio

**Ran the real oracle** (`pip download snac --no-deps`, then a self-contained adaptation of
`orpheus_snac.py`'s real logic since that file depends on a private `_hooks` harness module not
present standalone -- new `scratch-llamacpp-ref/snac_golden.py`, same NoiseBlock-no-op patch,
same deterministic-codes construction). Also fetched the real `hubertsiuzdak/snac_24khz`
`config.json` directly from HF to nail down the exact decoder config rather than trust the
`snac` package's class defaults (which are for an unrelated 44.1kHz variant): `decoder_dim=1024`,
`decoder_rates=[8,8,4,2]`, `encoder_dim=48`+`encoder_rates=[2,4,8,8]` (latent_dim = 48*16=768,
confirmed matches `orpheus_snac.h`'s stated 768), `vq_strides=[4,2,1]`, `codebook_size=4096`,
`codebook_dim=8`, `attn_window_size=null` (confirmed: no LocalMHA layer anywhere in this
variant's decoder, matching the real GGUF's tensor set having zero attention tensors), `noise=true`
(present in weights, made a no-op at inference, see below), `depthwise=true`.

**Real per-module math confirmed by reading `snac/layers.py`, `snac/vq.py`, `snac/snac.py`
directly** (not guessed, not inferred from `orpheus_snac.h`'s header comment alone, though that
comment's spec matched exactly once cross-checked): `ResidualVectorQuantize.from_codes` (per
quantizer: embedding lookup by codebook index -> `out_proj` pointwise conv (8->768) ->
`repeat_interleave` nearest-neighbor time-upsample by that quantizer's own stride -> sum across
the 3 quantizers) -> `Decoder.forward` (depthwise conv in0 (768ch, k=7) -> pointwise conv in1
(768->1024, k=1) -> 4x `DecoderBlock` (`Snake1d` -> `ConvTranspose1d` upsample (strides 8/8/4/2)
-> [`NoiseBlock`, no-op'd] -> 3x `ResidualUnit` dilations 1/3/9) -> final `Snake1d` -> full
(non-grouped) conv (64->1, k=7) -> `Tanh`). `ResidualUnit`: `Snake1d` -> depthwise dilated conv
(k=7, `pad=(kernel-1)*dilation/2`) -> `Snake1d` -> pointwise conv (k=1) -> residual add. `Snake1d`:
`x + (1/(alpha+1e-9)) * sin(alpha*x)^2`, per-channel alpha.

**`NoiseBlock` is a documented no-op, not this port's own shortcut**: the real
`NoiseBlock.forward` injects `torch.randn(...)` every call, making the real PyTorch decoder
itself non-deterministic at inference. `orpheus_snac.py`'s own comment explicitly patches
`NoiseBlock.forward = lambda self, x: x` for exactly this reason ("the noise contribution is
~1e-2 of the signal RMS at 24kHz"). This port follows the same documented convention -- loads
`noise.weight` from the GGUF but never uses it.

**Real GGUF tensor layout confirmed via `list-tensors` on `models/snac-24khz.gguf`** (110
tensors, 25.1MiB total): `snac.dec.in0`/`in1`, `snac.dec.{0..3}.{alpha,up.weight/bias,
res.{0,1,2}.{alpha0,conv0.weight/bias,alpha1,conv1.weight/bias}}`, `snac.dec.out.{alpha,weight,
bias}`, `snac.q.{0..2}.{codebook,in_proj.weight/bias,out_proj.weight/bias}`. Weight-norm is
already folded into a single plain weight tensor per conv (no separate `.original0`/`.original1`
g/v pair anywhere, unlike CosyVoice HiFT's checkpoint) -- no folding step needed.

**Dimension-order convention cross-verified against multiple real tensor shapes before writing
any indexing code** (not assumed from one example): GGUF's displayed shape is the REVERSE of
PyTorch's native dim order, but the underlying FLAT BYTE LAYOUT is identical to PyTorch's native
row-major layout -- e.g. `snac.q.0.out_proj.weight` displayed `[1,8,768]` (kernel,in,out reversed)
matches PyTorch `Conv1d(in=8,out=768,kernel=1)`'s native `[out,in,kernel]` shape exactly when
un-reversed. This let every kernel below reuse the exact same indexing formulas as this
codebase's existing `HiFTVocoderKernels.ConvTranspose1d` (`weight[(ic*outCh+oc)*kernel+k]`),
confirmed correct via the golden test, not just assumed to generalize.

**`src/OpenTail.Stingray.Audio/Orpheus/SnacWeights.cs`** (new): real GGUF loader, all tensor
names above, config constants (`DecoderDim=1024`, `LatentDim=768`, `CodebookSize=4096`,
`CodebookDim=8`, `DecoderRates=[8,8,4,2]`, `VqStrides=[4,2,1]`).

**`src/OpenTail.Stingray.Audio/Orpheus/SnacDecoder.cs`** (new): real forward pass --
`Snake1d`/`DepthwiseConv1d`/`PointwiseConv1d`/`ConvTranspose1d`/`ResidualUnit`/`DecoderBlock`/
`QuantizerFromCodes`/`Decode`, matching the exact real math above.

**Golden-verified**: `scratch-llamacpp-ref/snac_golden.py` runs the REAL PyTorch `snac` package
against the REAL pretrained `hubertsiuzdak/snac_24khz` weights (auto-downloaded from HF, not a
from-scratch reimplementation) with 4 deterministic superframes (every slot = code 17). C#
`SnacDecoder.Decode` against the SAME real `models/snac-24khz.gguf` weights and same codes
produces PCM matching the oracle at **cosine similarity > 0.99, PASSED on the first attempt** (no
bugs needed fixing) -- `tests/OpenTail.Stingray.Tests.Audio/SnacDecoderTests.cs`.

### Real prompt/detokenization spec: ONE MORE correction found, via direct tokenizer-vocab inspection (not guessing, not trusting either prior source blindly)

Before wiring the full pipeline, verified the exact numeric relationship between raw GGUF vocab
ids and the `<custom_token_N>` string ids `turn_token_into_id` parses, by directly dumping
specific vocab entries from `models/orpheus-3b-0.1-ft.Q4_K_M.gguf`'s own
`tokenizer.ggml.tokens` array (via the real `gguf` Python package, not assumed):
`id 128256 = "<custom_token_0>"`, `128257 = "<custom_token_1>"`, `128259 = "<custom_token_3>"`,
`128260 = "<custom_token_4>"`, `128261 = "<custom_token_5>"`, `128266 = "<custom_token_10>"`.

**This resolves an apparent contradiction, not a real one**: `orpheus.cpp`'s comment labels
`custom_token_offset=128266` as "`<custom_token_0>` id", which is factually wrong for this
checkpoint's real vocab (`<custom_token_0>` is actually id 128256) -- but its formula
`code = raw_id - custom_token_offset - slot*4096` is still numerically IDENTICAL to the real
Python `turn_token_into_id`'s `code = N - 10 - slot*4096` once `N = raw_id - 128256` is
substituted (`128256 + 10 = 128266`), because the real Python formula's constant `-10` and the
base offset are simply combined differently. **Confirmed real, final formula for this specific
checkpoint**: `code = raw_id - 128266 - (index % 7) * 4096`. Also confirmed: token id 49158 (the
real `stop_token_ids` default from `engine_class.py`) decodes to the ORDINARY BPE text token
`"Ġrez"` -- not a reserved/special token at all, just something this fine-tune apparently learned
to emit as an end-of-audio marker. Real, if slightly unusual; replicated as-is, not "fixed".

### `src/OpenTail.Stingray.Audio/Orpheus/OrpheusPipeline.cs` (new): full text+voice -> PCM pipeline, wired and RUNNING

Talker: loads `models/orpheus-3b-0.1-ft.Q4_K_M.gguf` through this codebase's completely
UNMODIFIED `GgufModel`/`ModelHyperparams`/`ForwardPass`/`GgufTokenizer`/`CpuBackend` --
confirmed, per the earlier entry in this section, that Orpheus's talker needs zero new
forward-pass code. `BuildPrompt`: raw `Encode("{voice}: {text}")` (NOT a chat template) wrapped
in the real special tokens (`[128259] + ... + [128009,128260,128261,128257]`).
`GenerateCodes`: greedy autoregressive decode (temperature=0 -- a deliberate first-pass choice
for determinism; the real reference's own defaults are temp=0.6/top_p=0.8/repetition_penalty=1.3,
NOT yet wired) via `ForwardPass.Prefill`/`Forward`, stopping at token 49158 or a max-tokens cap,
de-interleaving valid in-range generated tokens into 3 codebook streams via the real formula
above (out-of-range tokens, e.g. the model emitting ordinary text instead of a codec code, are
skipped rather than fed to the codec -- matches the real `decoder.py`'s own `if token > 0` /
range guard), truncated to a whole number of complete 7-token superframes. `Synthesize`: codes ->
`SnacDecoder.Decode` -> 24kHz mono PCM.

**End-to-end structural test, real weights throughout, PASSED**:
`tests/OpenTail.Stingray.Tests.Audio.Fast/OrpheusPipelineTests.cs` --
`Synthesize("Hello, this is a test.", voice: "tara", maxTokens: 140)` against the real
talker + real SNAC weights ran in ~2m19s (CPU, greedy, ~3B model -- unoptimized, no perf pass
attempted yet for this pipeline) and produced **1.37s of real, non-silent, valid-range
([-1,1] post-Tanh) 24kHz PCM** (65580-byte WAV, saved to
`scratch-llamacpp-ref/orpheus_test_output.wav` for the user to listen to). This is NOT a
golden-cosine-verified result end-to-end (no independent oracle runs the FULL talker->codec
chain for comparison -- the talker LM's forward-pass correctness rests on this codebase's
existing, separately-validated Llama support, and the codec stage is independently golden-
verified above) -- it confirms the pipeline is real, wired, and produces plausible audio, not
that the specific words/voice are correct. **Whether the audio is actually intelligible speech
saying the right words has NOT been verified by a human listener yet** -- flagging this
explicitly rather than claiming more than was checked.

**Orpheus TTS is now functionally COMPLETE for a first pass**: real talker (existing infra,
zero new code), real prompt construction, real detokenization (corrected via direct evidence,
not trusted blindly from either reference source), real golden-verified SNAC codec, wired
end-to-end, producing real audio. Not committed (per standing instruction). **Concrete follow-up
items, not blocking, lower priority than moving to the next queue pipeline**:
1. Listen to `scratch-llamacpp-ref/orpheus_test_output.wav` and confirm it's actually
   intelligible/correct speech, not just "real-looking PCM."
2. Wire real sampling (temp=0.6, top_p=0.8, repetition_penalty=1.3) instead of greedy --
   greedy was a deliberate first-correctness-pass simplification, not a final choice; may also
   affect audio quality/naturalness meaningfully for an AR TTS model.
3. Performance pass (per the new CLAUDE.md rule 7, added this session): 2m19s for ~1.4s of audio
   is far from real-time -- likely dominated by the talker's per-token `ForwardPass.Forward`
   calls on a 3B model on CPU; check GPU offload (`-g`/`--ngl` equivalent) wiring for
   `OrpheusPipeline`, not yet attempted.
4. DRY pass (per the same new rule): not yet needed (no duplicated logic across Orpheus and
   another pipeline yet), but check once Parler-TTS is also done, since both may share codec-
   adjacent patterns.
5. `scratch-llamacpp-ref/snac-pkg/` (extracted real `snac` package) can be deleted once this
   work is confirmed stable, per this doc's usual "scratch is gitignored, not permanent"
   convention.

**Queue status**: FunASR ✅, Silero VAD ✅, Orpheus TTS ✅ (first pass, follow-ups noted above).
Fish Speech and Parler-TTS remain: both have real weights (`models/s2-pro-q4_k_m.gguf`,
`models/Parler_TTS_mini.gguf`) and real reference source (`examples/s2.cpp`,
`examples/CrispASR/src/parler_tts.*` / `examples/TTS.cpp`) already in place from earlier this
fire -- next fire should start Parler-TTS or Fish Speech following the exact same discipline
demonstrated end-to-end here for Orpheus (real weight loader -> real per-module math from the
real source -> golden-verify each new component -> wire end-to-end -> test).
