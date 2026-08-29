# CosyVoice3 HiFT vocoder "mosquito"/buzz bug — request for advice

## Context

I'm porting CosyVoice3's TTS pipeline from a real C++ reference implementation
(`cosyvoice.cpp`, a llama.cpp-style GGUF/GGML port) to a native C# engine
(OpenTail.Stingray). The full pipeline (LLM speech-token generation → DiT/CFM
flow-matching mel decoder → HiFT vocoder) is architecturally complete, and I
just fixed a real structural gap in the zero-shot conditioning (reference-audio
speech-token concatenation before flow encoding). That fix works correctly.

But independent of that: **the final audio output has always had a
"mosquito"/high-pitched buzzing-whine quality**, both with and without
reference-audio conditioning. The words are sometimes intelligible under the
buzz, but it's clearly wrong.

## What I've already ruled out (numerically verified against the real C++ reference)

I built the real C++ reference CLI (`cosyvoice.cpp`'s `cosyvoice-cli`) against
the *exact same GGUF model weights and ONNX frontend files* my C# port uses,
and it produces clean, correct speech ("this is a test of voice cloning",
clearly spoken) — so the model files themselves are fine, and the bug is
entirely in my C# vocoder port.

I then instrumented the reference C++ code with env-var-gated tensor dumps at
several checkpoints, and cross-checked each one numerically (cosine
similarity) against the equivalent point in my C# `HiFTVocoderKernels.cs`,
feeding the **exact same real inputs** through both sides:

1. **F0 (pitch) prediction**: reference vs. mine — **numerically identical**
   (min/max/mean match to 2 decimal places, e.g. mean=122.71Hz vs 122.72Hz).
2. **Mel spectrogram input to the vocoder**: fed the reference's own real
   ground-truth mel (dumped from its DiT output) directly into my C# HiFT —
   **still buzzy**. So the DiT/mel stage isn't the cause either.
3. **Harmonic source excitation** (the NSF sine-generator's output, before the
   final conv/ISTFT decode stage): fed the reference's own dumped excitation
   signal directly into my C# `Decode()` (completely bypassing my own
   `SineGen`) — **still buzzy**. So my `SineGen`/source-generation isn't the
   cause either.
4. **`conv_pre` output** (first conv layer of the decode stage, applied to the
   real mel): reference vs. mine — **cosine similarity 0.9999999848** (i.e.
   numerically identical). So `conv_pre`'s weights and causal-conv logic are
   correct.
5. **End of upsample stage 0** (after `ups[0]` ConvTranspose1d, source
   injection via `source_downs[0]`+`source_resblocks[0]`, and the
   `resblocks[0*num_kernels + j]` HiFiGAN residual blocks, for
   `j=0,1,2` covering kernel sizes `[3,7,11]`, averaged): reference vs.
   mine — **cosine similarity only 0.568**, and the value ranges are
   qualitatively different (reference: small alternating-sign values roughly
   in `[-8, 8]`; mine: larger, mostly-positive values roughly `[0, 20]`).

**So the bug is confirmed to be introduced somewhere inside upsample stage 0**
— i.e. in one (or more) of:
- `ups[0]`: a causal `ConvTranspose1d` (kernel 16, stride 8, `BaseChannels=512`
  → `256` channels)
- `source_downs[0]` + `source_resblocks[0]`: strided injection of the
  excitation's own STFT (`s_stft`) into the upsampled path
- `resblocks[0*3 + j]` for `j=0..2` (kernel sizes 3, 7, 11): standard HiFiGAN
  `ResBlock`s with Snake activations and dilated causal convs (dilations
  1,3,5), residual-summed and averaged by `1/3`

I was in the middle of adding one more, finer-grained dump (right after
`ups[0].build_cgraph` alone, before source injection/resblocks) to narrow it
down to exactly one of those three sub-components when I paused to write this.

## Specific things I suspect but haven't confirmed

1. **`ConvTranspose1d` weight layout/orientation.** PyTorch's native
   `nn.ConvTranspose1d` weight shape is `[in_channels, out_channels, kernel]`
   — note this is **reversed** from regular `Conv1d`'s
   `[out_channels, in_channels, kernel]`. My C# `ConvTranspose1d` kernel
   assumes the GGUF-stored flat buffer is row-major `[inCh, outCh, kernel]`
   (i.e. untransposed from PyTorch's native transposed-conv layout). The GGUF
   exporter (`convert_model_to_gguf.py`) does a generic `add_tensor(name, value)`
   dump with no per-layer-type transpose, and I *believe* GGUF preserves the
   original PyTorch row-major byte order regardless of the `ne[]` display
   order — but this is exactly the kind of orientation assumption that's easy
   to get backwards, and would produce structurally wrong output while still
   "looking like a signal" (which matches what I'm seeing — not silence, not a
   crash, just wrong/buzzy).

2. **`source_downs[0]`'s strided conv reading the excitation's own STFT
   (`s_stft`)** — the layout convention of the complex real/imaginary STFT
   representation. My C# `RealStft` returns shape `[2*specBins, frames]`
   (real for channels `0..specBins-1`, imag for `specBins..2*specBins-1`,
   channel-outer). The reference computes `s_stft` via `ggml_stft(...)`, then
   `ggml_reshape_2d(s_stft, s_stft->ne[0], s_stft->ne[1] * 2)` — i.e. it
   reshapes by **merging the frame dimension and the complex-part dimension**
   together (`ne[1] * 2`), which only makes sense if the tensor's *pre-reshape*
   axis order was `[specBins, frames, 2]` (freq fastest-varying... or
   whichever GGML's `ne[0]` convention makes fastest). I was **not able to
   fully verify** this reshape's resulting real/imag memory interleaving
   against my own `[2*specBins, frames]` channel-doubled convention before
   pausing — this could easily be a real mismatch (e.g. real/imag
   interleaved per-frame vs. blocked by channel), and it feeds every single
   upsample stage's source injection, matching the "introduced within stage
   0, but not before" symptom.

3. Something in the three `ResBlock`s (Snake activation formula, dilated
   causal-conv padding amount, or residual-summation order) — though I have
   independently verified my Snake activation formula against another,
   unrelated model's codec in this same codebase already (Fish Speech), so I
   consider this less likely but haven't ruled it out here specifically.

## What I'd like advice on

- Given the numeric evidence above (matches through `conv_pre`, diverges by
  end of upsample stage 0), which of the three sub-components would you
  investigate first, and why?
- Is there a known GGUF/GGML convention (or PyTorch/GGML interop gotcha) for
  how `ConvTranspose1d` weights get laid out that I should check literally,
  rather than reasoning about it abstractly?
- Is there a standard/known layout convention for `ggml_stft`'s output shape
  and its real/imaginary interleaving after a `reshape_2d(ne0, ne1*2)` that
  would tell me definitively whether my `[2*specBins, frames]`
  channel-doubled convention is equivalent or different?
- Any other classic HiFiGAN/ISTFTNet-style vocoder porting bugs that produce
  specifically a "buzzing"/"mosquito whine" artifact (as opposed to silence,
  clipping, or garbled-but-recognizable speech) that I should consider?

I have full read access to both the real C++ reference source
(`examples/cosyvoice.cpp/src/cosyvoice-graph.cpp`,
`cosyvoice-loader.cpp`, `cosyvoice-token2wav.cpp`) and my own C# port
(`src/OpenTail.Stingray.Audio/Primitives/HiFTVocoderKernels.cs`,
`src/OpenTail.Stingray.Audio/CosyVoice/CosyVoice3HiftWeights.cs`), and I'm
already set up to add more env-var-gated numeric dumps on both sides to test
whatever hypothesis you suggest.
