# CosyVoice3 LLM forward-pass bug — request for help finding the root cause

## Context

I'm porting CosyVoice3's TTS pipeline from a real C++ reference implementation
(`examples/cosyvoice.cpp`, a llama.cpp-style GGUF/GGML port) to a native C#
engine (`OpenTail.Stingray`, a from-scratch LLM/diffusion/audio inference
engine in C#/.NET). The full pipeline is: text → LLM generates speech tokens →
flow encoder → DiT (conditional flow matching) → HiFT vocoder → waveform.

This session already found and fixed two real bugs (a HiFT vocoder padding bug,
and a missing LLM zero-shot conditioning gap), each verified by comparing
against the real C++ reference numerically before trusting the fix. Both are
committed. **Every stage of the pipeline except one is now proven correct**
against the reference, given identical inputs:

- Frontend extraction (speaker x-vector, reference-audio speech tokenizer,
  mel extractor): **cosine similarity 1.000000**, and the speech tokenizer
  produces an **exact 65/65 token match** against the reference.
- Flow encoder (`mu`, `spks`, `conds` tensors): **cosine similarity 1.000000**
  on all three, given identical token/embedding inputs.
- DiT (conditional flow matching ODE solve) + HiFT vocoder, run end-to-end:
  when fed the reference's own real tokens/embedding/prompt-mel, produces
  **correctly-transcribed, clean speech** (verified via Whisper ASR: "This is
  a test of voice coding[cloning]. This is a test of voice synthesis." for
  the reference's own real 160 generated tokens).

**The one broken piece: the LLM's own forward pass** (a Qwen2-architecture
backbone, `CosyVoice3Llm.GenerateSpeechTokens` / `CosyVoice3LlmTensorSource` in
the C# code, `cosyvoice_model_3::llm_prefill`/`llm_decode` in the C++
reference — `src/cosyvoice-llm.cpp`). Given the IDENTICAL real composed input
token sequence (sos + instruction-prefix + endofprompt + reference transcript
+ new synthesis text + task-token + reference's own real prompt speech
tokens — all real values, dumped from an actual reference CLI run), our C#
`ForwardPass.Prefill`'s first-step logits do NOT match the reference's own
dumped logits for the same sequence:

- **Cosine similarity: 0.30** (should be ~1.0)
- **Argmax completely different** (our top pick: token 2387; reference's:
  token 4011 — not a near-miss)
- Our top logit is ~4x larger in magnitude than the reference's (17.9 vs 4.3)
  — but since the argmax itself differs, this is not just a scale/temperature
  issue; the underlying computation diverges in kind, not just magnitude.

## The audible symptom (from real, listened-to output)

Before these fixes, output was a high-pitched "drone"/"drill" noise with no
words. After the HiFT + LLM-conditioning fixes above, **words are now audible**
— real, if heavily distorted, progress. The remaining distortion is described
by the person listening as: **"like lowest-quality MP3 noise sprinkled around
the clip, like a stream of digits"** — i.e. real speech content is there, but
interspersed/corrupted by short bursts of harsh, bitcrushed-sounding noise,
rather than a uniform tone or total gibberish. This is consistent with the
LLM occasionally emitting a badly-wrong speech token (from the diverging
logits above) that decodes, through an otherwise-correct flow/DiT/HiFT chain,
into a short burst of harsh/wrong-sounding audio, while nearby correctly (or
near-correctly) chosen tokens still produce recognizable speech.

## What's already been ruled out for this specific bug (checked directly, not assumed)

- **CORRECTION, found after this doc was first written**: this doc originally
  claimed "CosyVoice2's own LLM was already independently verified working,
  so a shared-engine bug is unlikely." That claim was WRONG and has been
  retracted. The existing `CosyVoiceLlmTensorSourceTests.cs` only checks that
  logits are finite and non-degenerate (`max-min > 1.0`) — it never compared
  against any real oracle. When actually asked to generate real audio via a
  new `CosyVoice2GenerateWavDebugTest.cs` and checked with Whisper ASR,
  CosyVoice2's real end-to-end output transcribes as **`[Music]`** — no
  words at all, i.e. worse/more degenerate than CosyVoice3's output (which
  has audible, if distorted, real words). So CosyVoice2's LLM is ALSO
  broken — but given the qualitatively different severity (no words at all
  vs. words present with noise bursts), **do not assume this is the same
  bug as CosyVoice3's** — it could easily be a second, independent bug
  (e.g. in CosyVoice2's own tensor adapter/prompt composition, which is
  different code from CosyVoice3's) rather than a shared root cause. Treat
  "is this the same bug in both, or two separate ones" as an open question
  to investigate, not a conclusion — the shared engine is now merely back
  on the table as ONE possibility among several, not confirmed.
- **Not a metadata/hyperparameter mapping bug**: `CosyVoice3LlmTensorSource`'s
  constructor (`src/OpenTail.Stingray.Audio/CosyVoice/CosyVoice3LlmTensorSource.cs`)
  explicitly translates this checkpoint's own bare metadata keys
  (`num_hidden_layers`, `rms_norm_eps`, `rope_theta`, etc. — NOT the standard
  GGUF `{arch}.*` naming convention) into the standard `qwen2.*`-prefixed keys
  the generic hyperparameter parser expects. Checked this translation
  carefully; it looks correct (head counts, RoPE theta, RMS norm eps, and a
  vocab-size override for the combined text+speech embedding table all map
  correctly).
- **Not the token-composition sequence itself**: the exact sequence of ids
  fed to the LLM (sos, instruction-prefix tokens, endofprompt token, real
  reference transcript tokens, new text tokens, task-token, real reference
  prompt speech tokens) was transcribed directly from the reference's own
  `cosyvoice-llm-job.cpp`'s `llm_job_ext`, not guessed, and this session
  separately confirmed (via `list-metadata`) that `sos_token_id`/
  `task_token_id`/`stop_token_ids` are read correctly from the real GGUF
  metadata.
- **Not the embedding-table composition trick**: `CosyVoice3LlmTensorSource.
  EnableSpeechGenerationMode()` builds one combined [text-vocab rows ;
  speech-vocab rows] embedding table so integer token ids (offset for speech
  ids) can be used directly with the ordinary `ForwardPass` API instead of
  needing raw-embedding injection (unlike the C++ reference, whose two
  embedding tables are genuinely separate weight tensors). This same pattern
  is already used successfully by `CosyVoiceLlmGeneration.cs` (CosyVoice2) and
  `QwenAsrLlmTensorSource.cs`.

## What's NOT yet been tried (the natural next steps, in priority order)

This project's own methodology (documented in `docs/qwentts-cosyvoice3-handoff.md`,
proven effective for a structurally similar bug in a sibling pipeline,
QwenTTS's Talker LM) is:

1. **Narrow by sequence length first.** The comparison so far used the REAL,
   long (~140+ token) prefill. QwenTTS's actual bug was only visible with 2+
   cached positions — a single-position test passed with cosine 0.9999. Try
   forcing a much SHORTER real sequence (e.g. no prompt speech tokens, no
   reference transcript — just `sos + instruction-prefix + endofprompt +
   short text + task-token`, maybe 15-20 positions) through both the
   reference CLI (may need a `--mode cross-lingual` invocation instead of
   zero-shot, or a synthetic minimal test harness) and our C# port, dumping
   logits from both. If a short sequence matches (cosine ~1.0) but the long
   one doesn't, the bug is specifically triggered by something that scales
   with sequence length or KV-cache size (attention masking, RoPE position
   ids beyond some threshold, KV-cache read/write consistency for many
   positions). If even the short sequence diverges, the bug is more
   fundamental and should be even easier to find.
2. **Dump post-RoPE Q/K vectors directly**, not just final logits, to narrow
   "somewhere in the transformer" down to a specific op: Q after RoPE? K
   after RoPE? attention scores? weighted V-sum? output projection? MLP?
   Final norm? This needs new env-var-gated dump hooks added to
   `examples/cosyvoice.cpp/src/cosyvoice-llm.cpp`'s `build_qwen2_decoder_layer`
   (mirrors the same env-var-gated dump pattern already used elsewhere in
   this reference for the HiFT vocoder — see `g_dump_*` globals in
   `cosyvoice-graph.cpp`/`cosyvoice-token2wav.cpp` for the pattern to copy).
3. **KV-cache write/read consistency.** Verify the cached K/V for an early
   position (written during the multi-token prefill) is byte-identical to
   what a fresh, uncached computation of that same position would produce.
   This is exactly the kind of bug that's invisible in a position-0-only test
   but shows up once real multi-position caching is exercised.
4. **Check the causal attention mask construction specifically for this long
   sequence.** The reference's `llm_prefill` builds a custom causal mask
   (`build_causal_mask` in `cosyvoice-llm.cpp`) with its own indexing scheme
   (`visible_prefix_end = seq_len - n_batch`). Compare this against however
   the C# `ForwardPass.Prefill` constructs its own causal mask for a
   multi-token prefill with no prior KV cache — a subtle indexing mismatch
   here (e.g. off-by-one in how many prior positions are visible) would
   produce exactly this symptom: correct-ish for short/simple cases,
   increasingly wrong as the real sequence gets longer and more
   context-dependent.

## Where to find everything (all real, already built and working)

- Reference C++ CLI: `examples/cosyvoice.cpp/build/bin/Release/cosyvoice-cli.exe`
  (already built; rebuild via `cmake --build build --config Release --parallel 8
  --target cosyvoice-cli` from an MSVC dev environment — plain `cl.exe` from a
  bare shell fails on a missing `<cstdint>`; use `vcvars64.bat` first).
- Real prompt audio already extracted and ready to reuse:
  `examples/cosyvoice.cpp/prompt16k.pcm`/`prompt24k.pcm` (raw 32-bit FLOAT
  PCM — NOT 16-bit int, a mistake made once already this session, see the
  CLI's own `--help` text: "Reference audio in 16kHz PCM float format").
  Real prompt text used with these: `"this is a test of voice cloning"`.
- Existing env-var-gated dump hooks in the reference (already added this
  session, still in the tree — these are NOT committed since `examples/` is
  gitignored, but they're sitting there uncommitted and working):
  `COSY_DUMP_NEWTOKENS_PATH`, `COSY_DUMP_PROMPTTOKENS_PATH`,
  `COSY_DUMP_PROMPTFEAT_PATH`, `COSY_DUMP_EMBEDDING_PATH`, `COSY_DUMP_MU_PATH`,
  `COSY_DUMP_SPKS_PATH`, `COSY_DUMP_CONDS_PATH` (all in
  `cosyvoice-token2wav.cpp`), and `COSY_DUMP_LLM_LOGITS_PATH` (in
  `cosyvoice-llm.cpp`, dumps the raw pre-softmax logits from the FIRST
  `llm_decode` call after a prefill). Example real invocation used to produce
  the comparison data already gathered:
  ```
  cd examples/cosyvoice.cpp
  $env:COSY_DUMP_LLM_LOGITS_PATH="$PWD\llm_logits.bin"
  .\build\bin\Release\cosyvoice-cli.exe --model ..\..\models\cosyvoice3\CosyVoice3-2512_F16.gguf `
    --speech-tokenizer ..\..\models\cosyvoice_speech_tokenizer_v2.onnx --campplus ..\..\models\campplus.onnx `
    --prompt-audio-16k prompt16k.pcm --prompt-audio-24k prompt24k.pcm `
    --prompt-text "this is a test of voice cloning" --text "This is a test of voice synthesis." `
    --output ref_llmlogits_check.wav --seed 42
  ```
- C# comparison test already built and passing (numerically proves the
  divergence): `tests/OpenTail.Stingray.Tests.Audio/CosyVoice3LlmLogitsCompareDebugTest.cs`
  — run via (note: never pass `--nologo`, and heavy/real-weight tests need
  `STINGRAY_RUN_HEAVY_TESTS=1`):
  ```
  dotnet test tests/OpenTail.Stingray.Tests.Audio -c Release -- --filter-class "OpenTail.Stingray.Tests.Audio.CosyVoice3LlmLogitsCompareDebugTest"
  ```
- The debug-only method this test uses to get raw first-step logits:
  `CosyVoice3Llm.GetFirstStepLogitsForTest` (internal, in
  `src/OpenTail.Stingray.Audio/CosyVoice/CosyVoice3Llm.cs`) — reuse or extend
  this for any narrower/shorter-sequence test.
- Other real comparison tests from this session, useful as templates for the
  same dump-then-compare pattern: `CosyVoice3FlowEncoderCompareDebugTest.cs`,
  `CosyVoice3FrontendExtractionCompareDebugTest.cs`,
  `CosyVoice3FullChainFromRefInputsDebugTest.cs`,
  `CosyVoiceHiftStage0CompareDebugTest.cs`.
- ASR-based listening loop (much faster than asking a human to listen every
  time — use Whisper to sanity-check any new candidate fix before asking for
  a human listen): `dotnet run --project src/OpenTail.Stingray.Cli -c Release
  -- stt -i <wav> --model-file models/ggml-medium.bin`.

## What I'd like help with

Given the numeric evidence above (every other stage proven correct, the LLM's
own forward pass diverges specifically for a long, multi-position, real
sequence), which of the four "not yet tried" steps above would you pursue
first, and why? Is there a known class of bug in causal-mask construction or
KV-cache indexing for multi-token prefills (as opposed to single-token decode
steps) that commonly produces exactly this "fine at short range, degrades
with more context" symptom? I have full read/write access to both the real
C++ reference source and the C# port, and can add more env-var-gated dumps to
either side to test whatever hypothesis is suggested.
