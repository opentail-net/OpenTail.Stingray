# Audio subsystem review — progress log

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

### Remaining known opportunity (not pursued, flagged for whoever picks this up next)
Whisper's `stride=2` conv (`WhisperEncoder.ApplyConv1DReal`, the `else` branch) is the one
conv loop across all four pipelines that was deliberately left unvectorized — it's the more
expensive of Whisper's two conv calls at large model sizes (dModel² channel scaling vs.
conv1's dModel×numMels), but the strided-gather access pattern needs a different technique
(e.g. pre-extracting even/odd strided views into contiguous temp buffers before the
vectorized MAC, or accepting the smaller win of vectorizing only within each stride class)
than the simple contiguous-shift trick used everywhere else. Not attempted this session;
Whisper's overall performance was already reasonable before this sweep (unlike Chatterbox/
Kokoro, which were the actual motivating problem), so this is optional polish, not a
known-broken gap.
