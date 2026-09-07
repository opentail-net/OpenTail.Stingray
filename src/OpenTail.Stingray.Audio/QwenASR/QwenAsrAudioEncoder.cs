
namespace OpenTail.Stingray.Audio.QwenASR;

/// <summary>
/// Configuration for the Qwen3-ASR Audio Transformer (AuT) Encoder.
/// </summary>
public sealed record QwenAsrEncoderConfig
{
    public int InMelChannels { get; init; } = 128;
    public int EncoderDim { get; init; } = 896;     // 896 for 0.6B, 1024 for 1.7B
    public int NumLayers { get; init; } = 18;       // 18 for 0.6B, 24 for 1.7B
    public int NumHeads { get; init; } = 14;        // 14 for 0.6B, 16 for 1.7B
    public int QwenHiddenDim { get; init; } = 1024; // 1024 for 0.6B, 2048 for 1.7B
    public int WindowSizeInfer { get; init; } = 800; // 8 seconds attention window
    /// <summary>Real chunk-size divisor (`audio_config.n_window`, confirmed via
    /// examples/audio.cpp's real qwen3_asr_audio_encoder_token_count/audio_encoder.cpp:
    /// `chunk_frame_limit = n_window * 2`): the conv stem does NOT run over the whole mel
    /// sequence at once -- it splits input frames into independent chunks of at most
    /// `NWindow*2` frames each, runs the SAME 3-stage conv stem on every chunk separately (its
    /// own edge padding per chunk, not just at the sequence ends), and restarts the sinusoidal
    /// positional embedding at position 0 for every chunk, before concatenating all chunks'
    /// tokens. See <see cref="Forward"/>'s doc comment for why this matters (a single
    /// continuous-sequence conv gives the WRONG token count and content for any audio longer
    /// than one chunk).</summary>
    public int NWindow { get; init; } = 50;
}

/// <summary>
/// Audio Transformer (AuT) Encoder for Qwen3-ASR: 3-stage full Conv2D stem (each stride-2,
/// giving 8x downsampling in BOTH time and mel-frequency: 128 mel bins -> 16, confirmed
/// against the real checkpoint's tensor shapes -- `audio.conv_out.weight` is `[7680,896]` and
/// `480*16=7680` exactly, see docs/audio-review-progress.md's QwenASR section) -> Linear
/// projection into encoder dim -> sinusoidal absolute positional embedding (Whisper's own
/// convention, reused since this checkpoint ships no positional-embedding tensor, i.e. it's
/// non-learned) -> 18x pre-LN Whisper-style Transformer blocks (plain LayerNorm with bias,
/// GELU FFN, no rel-pos bias, no conv module, no macaron step -- simpler than Parakeet's
/// FastConformer) -> final LayerNorm -> 2-layer GELU adapter MLP into the Qwen3 LLM's
/// embedding space. Block-level math (attention/FFN/LayerNorm/GELU) follows the same real,
/// golden-verified pattern as `Whisper/WhisperEncoder.cs` (this checkpoint's AuT is
/// architecturally a Whisper-style encoder, just with a 2D rather than 1D conv stem, and
/// WITH a key bias unlike Whisper's encoder which omits it).
///
/// NOT YET golden-verified against a real oracle (no reference Python/C++ AuT implementation
/// has been run this session) -- structurally complete only, same caveat as every other
/// pipeline in this doc before its golden-verification pass.
/// </summary>
public sealed unsafe class QwenAsrAudioEncoder : IDisposable
{
    public QwenAsrEncoderConfig Config { get; }
    private readonly QwenAsrWeights? _weights;

    public QwenAsrAudioEncoder(QwenAsrEncoderConfig? config = null, QwenAsrWeights? weights = null)
    {
        Config = config ?? new QwenAsrEncoderConfig();
        _weights = weights;
    }

    /// <summary>
    /// Processes mel frames (frame-major [numMelFrames, InMelChannels], as produced by
    /// <see cref="QwenAsrMelExtractor"/>) through the real conv stem + AuT transformer +
    /// adapter projection into Qwen3 LLM token embeddings.
    /// Output: [numAudioTokens, QwenHiddenDim].
    ///
    /// <para><b>Real chunking, confirmed against `examples/audio.cpp`'s
    /// `qwen3_asr_audio_encoder_token_count`/`audio_encoder.cpp` (root-caused 2026-09-06 chasing
    /// a Qwen3-ForcedAligner off-by-2-audio-tokens bug that turned out to be this, affecting the
    /// base ASR pipeline too for any audio longer than one chunk)</b>: the conv stem does NOT
    /// run over the whole mel sequence as one continuous conv -- input frames are split into
    /// independent chunks of at most <see cref="QwenAsrEncoderConfig.NWindow"/>*2 frames each,
    /// the SAME 3-stage conv stem runs on every chunk SEPARATELY (its own edge padding at every
    /// chunk boundary, not just the sequence's own start/end), and the sinusoidal positional
    /// embedding restarts at position 0 for every chunk, before all chunks' tokens are
    /// concatenated. A single continuous-sequence conv (this method's previous behavior)
    /// produces a WRONG token count and content for any audio longer than one chunk (confirmed:
    /// 75 tokens here vs the real reference's 77, for a 593-mel-frame/~5.9s clip) -- the extra
    /// per-chunk edge padding accumulates real extra tokens a single continuous conv can't
    /// produce.</para>
    ///
    /// <para>NOT yet implemented: the real windowed (not full) self-attention mask the reference
    /// applies across chunks when the total token count exceeds one attention window
    /// (`max_chunk_tokens * (WindowSizeInfer/NWindow/2)` &#8776; 104 tokens for the default
    /// config) -- full attention across the whole concatenated sequence (this method's existing
    /// behavior) is only correct up to that many tokens (roughly a 13-second clip). Longer audio
    /// needs the real windowing restriction ported too; flagged rather than guessed.</para>
    /// </summary>
    public (float[] ProjectedTokens, int NumTokens) Forward(ReadOnlySpan<float> mel, int numMelFrames)
    {
        if (_weights == null)
            throw new InvalidOperationException("QwenAsrAudioEncoder requires real QwenAsrWeights -- no procedural fallback exists for the AuT encoder.");
        if (numMelFrames <= 0 || mel.IsEmpty)
            return ([], 0);

        var w = _weights;
        int nMels = Config.InMelChannels;
        int c = 480; // audio.conv_channels
        int encDim = Config.EncoderDim;
        int chunkFrameLimit = Config.NWindow * 2;

        // Real chunk-padding fix, 2026-09-07 (see docs/audio-review-progress.md's audio-encoder
        // bisection entries): the reference pads EVERY chunk's mel data to a UNIFORM
        // `chunk_frames_ = max(chunk_lengths_)` width (zero-padded on the right, real samples
        // first) before convolving -- this port previously convolved each chunk at its own
        // actual (unpadded) length, which only differs from the reference for a non-uniform
        // FINAL chunk (interior chunks are already exactly `chunkFrameLimit` long). A small,
        // edge-only effect (only the last chunk's own boundary tokens), confirmed via this
        // session's real per-frame dump comparison to be a minor contributor next to the mel-
        // extraction/conv-orientation bugs already fixed, but real and worth closing.
        int numChunks = (numMelFrames + chunkFrameLimit - 1) / chunkFrameLimit;
        int paddedChunkFrames = chunkFrameLimit;
        if (numChunks == 1) paddedChunkFrames = Math.Min(chunkFrameLimit, numMelFrames);

        var swEnc = System.Diagnostics.Stopwatch.StartNew();
        var chunkTokens = new List<float[]>();
        for (int chunkStart = 0; chunkStart < numMelFrames; chunkStart += chunkFrameLimit)
        {
            int chunkLen = Math.Min(chunkFrameLimit, numMelFrames - chunkStart);

            var chunkMel = new float[nMels * paddedChunkFrames];
            for (int m = 0; m < nMels; m++)
                mel.Slice(m * numMelFrames + chunkStart, chunkLen).CopyTo(chunkMel.AsSpan(m * paddedChunkFrames, chunkLen));

            var stage1 = Conv2dFull(chunkMel, cin: 1, hin: nMels, win: paddedChunkFrames, w.Conv1WeightCL, w.Conv1Bias, c, k: 3, stride: 2, pad: 1, out int h1, out int w1);
            GeluInPlace(stage1);

            var stage2 = Conv2dFull(stage1, cin: c, hin: h1, win: w1, w.Conv2WeightCL, w.Conv2Bias, c, k: 3, stride: 2, pad: 1, out int h2, out int w2);
            GeluInPlace(stage2);

            var stage3 = Conv2dFull(stage2, cin: c, hin: h2, win: w2, w.Conv3WeightCL, w.Conv3Bias, c, k: 3, stride: 2, pad: 1, out int h3, out int w3);
            GeluInPlace(stage3);

            // h3 = downsampled mel-freq count (128/8=16), w3 = downsampled time-token count --
            // the TIME axis (w3) is now the real per-token index, matching the reference. Real
            // per-chunk valid-token trim (`qwen3_asr_audio_encoder_token_count`, ported exactly):
            // padding a short last chunk up to `paddedChunkFrames` makes the raw conv output
            // wider than this chunk's real valid token count -- trim to match.
            int chunkT = Math.Min(w3, EncoderTokenCount(chunkLen));
            int flatDim = c * h3; // 480 * 16 = 7680, matches audio.conv_out.weight's input dim

            var chunkPosEmb = SinusoidalPositionalEmbeddings(chunkT, encDim);
            var allRows = new float[chunkT * flatDim];
            for (int ti = 0; ti < chunkT; ti++)
            {
                int rowOff = ti * flatDim;
                for (int ch = 0; ch < c; ch++)
                {
                    int stageChOff = ch * h3 * w3;
                    int chFlatOff = ch * h3;
                    for (int fh = 0; fh < h3; fh++)
                        allRows[rowOff + chFlatOff + fh] = stage3[stageChOff + fh * w3 + ti];
                }
            }
            var allTokens = new float[chunkT * encDim];
            fixed (float* pAllRows = allRows, pAllTokens = allTokens)
                w.ConvOutWeight.MatMul(pAllRows, chunkT, pAllTokens);
            for (int ti = 0; ti < chunkT; ti++)
            {
                var tokenVec = new float[encDim];
                for (int d = 0; d < encDim; d++)
                    tokenVec[d] = allTokens[ti * encDim + d] + chunkPosEmb[ti * encDim + d];
                chunkTokens.Add(tokenVec);
            }
        }
        long tConvMs = swEnc.ElapsedMilliseconds;

        int t = chunkTokens.Count;
        var x = new float[t * encDim];
        for (int ti = 0; ti < t; ti++)
            chunkTokens[ti].CopyTo(x.AsSpan(ti * encDim, encDim));

        if (Environment.GetEnvironmentVariable("STINGRAY_FA_DUMP_DIR") is { } dumpDirConvPos)
        {
            var bytes = new byte[x.Length * 4];
            Buffer.BlockCopy(x, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dumpDirConvPos, "our_enc_after_conv_pos.bin"), bytes);
        }

        swEnc.Restart();
        int ffnDim = w.AudioLayerWeights[0].FfnUpBias.Length;
        var normed = new float[t * encDim];
        var q = new float[t * encDim];
        var k = new float[t * encDim];
        var v = new float[t * encDim];
        var attnRaw = new float[t * encDim];
        var attnOut = new float[t * encDim];
        var ffnUp = new float[t * ffnDim];

        foreach (var layer in w.AudioLayerWeights)
            EncoderBlock(x, layer, t, encDim, Config.NumHeads, normed, q, k, v, attnRaw, attnOut, ffnUp);
        long tLayersMs = swEnc.ElapsedMilliseconds;

        if (Environment.GetEnvironmentVariable("STINGRAY_FA_DUMP_DIR") is { } dumpDirTransformer)
        {
            var bytes = new byte[x.Length * 4];
            Buffer.BlockCopy(x, 0, bytes, 0, bytes.Length);
            File.WriteAllBytes(Path.Combine(dumpDirTransformer, "our_enc_after_transformer.bin"), bytes);
        }

        swEnc.Restart();
        fixed (float* px = x, pLnW = w.LnPostWeight, pLnB = w.LnPostBias)
        {
            nint xAddr = (nint)px, lnWAddr = (nint)pLnW, lnBAddr = (nint)pLnB;
            System.Threading.Tasks.Parallel.For(0, t, ti =>
            {
                OpenTail.Stingray.Cpu.SimdKernels.LayerNorm((float*)xAddr + (long)ti * encDim, (float*)xAddr + (long)ti * encDim, (float*)lnWAddr, (float*)lnBAddr, encDim, 1e-5f);
            });
        }

        // Adapter: proj1 (encDim -> encDim) + GELU -> proj2 (encDim -> QwenHiddenDim).
        int qwenDim = Config.QwenHiddenDim;
        var projected = new float[t * qwenDim];
        fixed (float* px = x, pProj1 = normed, pProj = projected, pP1B = w.Proj1Bias, pP2B = w.Proj2Bias)
        {
            w.Proj1Weight.MatMul(px, t, pProj1, pP1B);
            GeluInPlace(normed);
            w.Proj2Weight.MatMul(pProj1, t, pProj, pP2B);
        }
        long tAdapterMs = swEnc.ElapsedMilliseconds;
        if (Environment.GetEnvironmentVariable("STINGRAY_FORCEDALIGNER_PERF") == "1" || Environment.GetEnvironmentVariable("STINGRAY_FORCEDALIGNER_TRACE") == "1")
            Console.Error.WriteLine($"[Enc-Perf] Conv={tConvMs}ms Layers={tLayersMs}ms Adapter={tAdapterMs}ms (t={t})");

        return (projected, t);
    }

    private static void EncoderBlock(
        float[] x,
        QwenAsrAudioLayerWeights l,
        int t,
        int dim,
        int heads,
        float[] normed,
        float[] q,
        float[] k,
        float[] v,
        float[] attnRaw,
        float[] attnOut,
        float[] ffnUp)
    {
        fixed (float* px = x, pNormed = normed, pQ = q, pK = k, pV = v,
                      pAttnRaw = attnRaw, pAttnOut = attnOut, pFfnUp = ffnUp)
        fixed (float* pAttnNormW = l.AttnNormWeight, pAttnNormB = l.AttnNormBias,
                      pQBias = l.AttnQBias, pKBias = l.AttnKBias, pVBias = l.AttnVBias,
                      pAttnOutBias = l.AttnOutBias,
                      pFfnNormW = l.FfnNormWeight, pFfnNormB = l.FfnNormBias,
                      pFfnUpBias = l.FfnUpBias, pFfnDownBias = l.FfnDownBias)
        {
            nint xAddr = (nint)px, normedAddr = (nint)pNormed, qAddr = (nint)pQ, kAddr = (nint)pK, vAddr = (nint)pV;
            nint attnRawAddr = (nint)pAttnRaw, attnOutAddr = (nint)pAttnOut;
            nint anWAddr = (nint)pAttnNormW, anBAddr = (nint)pAttnNormB;
            nint fnWAddr = (nint)pFfnNormW, fnBAddr = (nint)pFfnNormB;

            // 1. Pre-attention LayerNorm
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                OpenTail.Stingray.Cpu.SimdKernels.LayerNorm((float*)normedAddr + (long)i * dim, (float*)xAddr + (long)i * dim, (float*)anWAddr, (float*)anBAddr, dim, 1e-5f);
            });

            // 2. Q, K, V Projections (batched, parallel, hardware F16C with bias)
            l.AttnQWeight.MatMul(pNormed, t, pQ, pQBias);
            l.AttnKWeight.MatMul(pNormed, t, pK, pKBias);
            l.AttnVWeight.MatMul(pNormed, t, pV, pVBias);

            // 3. Multi-head self-attention
            int headDim = dim / heads;
            float scale = 1f / MathF.Sqrt(headDim);
            Array.Clear(attnRaw, 0, t * dim);

            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                Span<float> scores = stackalloc float[t];
                Span<float> probs = stackalloc float[t];
                for (int h = 0; h < heads; h++)
                {
                    int off = h * headDim;
                    float* qi = (float*)qAddr + (long)i * dim + off;
                    for (int j = 0; j < t; j++)
                    {
                        float* kj = (float*)kAddr + (long)j * dim + off;
                        scores[j] = OpenTail.Stingray.Cpu.SimdKernels.DotF32(qi, kj, headDim) * scale;
                    }
                    scores.CopyTo(probs);
                    OpenTail.Stingray.Audio.Primitives.DenseKernels.SoftmaxInPlace(probs);

                    float* weighted = (float*)attnRawAddr + (long)i * dim + off;
                    for (int j = 0; j < t; j++)
                    {
                        float* vj = (float*)vAddr + (long)j * dim + off;
                        float pj = probs[j];
                        System.Numerics.Tensors.TensorPrimitives.MultiplyAdd(
                            new ReadOnlySpan<float>(vj, headDim),
                            pj,
                            new ReadOnlySpan<float>(weighted, headDim),
                            new Span<float>(weighted, headDim));
                    }
                }
            });

            // 4. Output projection
            l.AttnOutWeight.MatMul(pAttnRaw, t, pAttnOut, pAttnOutBias);

            // 5. Residual 1: x += attnOut
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                OpenTail.Stingray.Cpu.SimdKernels.AddInPlace((float*)xAddr + (long)i * dim, (float*)attnOutAddr + (long)i * dim, dim);
            });

            // 6. Post-attention LayerNorm (before FFN)
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                OpenTail.Stingray.Cpu.SimdKernels.LayerNorm((float*)normedAddr + (long)i * dim, (float*)xAddr + (long)i * dim, (float*)fnWAddr, (float*)fnBAddr, dim, 1e-5f);
            });

            // 7. FFN Up projection
            l.FfnUpWeight.MatMul(pNormed, t, pFfnUp, pFfnUpBias);

            // 8. FFN GELU
            GeluInPlace(ffnUp);

            // 9. FFN Down projection (reuse pAttnOut buffer for down output)
            l.FfnDownWeight.MatMul(pFfnUp, t, pAttnOut, pFfnDownBias);

            // 10. Residual 2: x += ffnDown
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                OpenTail.Stingray.Cpu.SimdKernels.AddInPlace((float*)xAddr + (long)i * dim, (float*)attnOutAddr + (long)i * dim, dim);
            });
        }
    }

    // -----------------------------------------------------------------
    // Conv2d stem, same convention as ParakeetConformerEncoder's (IDX(c,h,w) = c*H*W+h*W+w,
    // weight order idx = kw+kh*K+cin*K*K+cout*K*K*Cin). All three AuT conv layers are FULL
    // (dense) convs, not depthwise, unlike Parakeet's stage-2/3.
    // -----------------------------------------------------------------
    /// <summary>
    /// Channel-last (im2col-lite) formulation: cin (up to 480 for this checkpoint's stages 2/3)
    /// was previously a scalar, non-contiguous innermost-but-one loop axis in both the weight
    /// and channel-first input layouts -- effectively unvectorized despite being the dominant
    /// cost dimension (measured ~70% of total AuT-encoder wall time on stage 2 alone before this
    /// change). Repacks input to [hin, win, cin] and weight to [cout, kh, kw, cin] once per call
    /// (O(cin*hin*win)/O(cout*cin*k*k), trivial next to the O(cout*hout*wout*cin*k*k) conv
    /// itself) so cin becomes contiguous on both operands, then uses SimdKernels.DotF32 (AVX2/
    /// FMA) for the per-(kh,kw) accumulation instead of a scalar inner loop.
    /// </summary>
    private static unsafe float[] Conv2dFull(float[] input, int cin, int hin, int win, float[] weightCL, float[] bias, int cout, int k, int stride, int pad, out int hout, out int wout)
    {
        int houtLocal = (hin + 2 * pad - k) / stride + 1;
        int woutLocal = (win + 2 * pad - k) / stride + 1;
        hout = houtLocal;
        wout = woutLocal;

        var inputCL = new float[hin * win * cin];
        if (cin == 1)
        {
            Array.Copy(input, inputCL, input.Length);
        }
        else
        {
            int hwStride = hin * win;
            System.Threading.Tasks.Parallel.For(0, hin, hi =>
            {
                for (int wi = 0; wi < win; wi++)
                {
                    int hwOffset = (hi * win + wi) * cin;
                    int inHw = hi * win + wi;
                    for (int ci = 0; ci < cin; ci++)
                    {
                        inputCL[hwOffset + ci] = input[ci * hwStride + inHw];
                    }
                }
            });
        }

        var output = new float[cout * houtLocal * woutLocal];
        fixed (float* inCLp = inputCL, wCLp = weightCL, outp = output)
        {
            float* inCL = inCLp; float* wCL = wCLp; float* outPtr = outp;
            System.Threading.Tasks.Parallel.For(0, cout, co =>
            {
                float* wBase = wCL + (long)co * k * k * cin;
                for (int ho = 0; ho < houtLocal; ho++)
                {
                    for (int wo = 0; wo < woutLocal; wo++)
                    {
                        float sum = bias[co];
                        for (int kh = 0; kh < k; kh++)
                        {
                            int hi = ho * stride - pad + kh;
                            if (hi < 0 || hi >= hin) continue;
                            for (int kw = 0; kw < k; kw++)
                            {
                                int wi = wo * stride - pad + kw;
                                if (wi < 0 || wi >= win) continue;
                                float* wSlice = wBase + (long)(kh * k + kw) * cin;
                                float* inSlice = inCL + (long)(hi * win + wi) * cin;
                                sum += OpenTail.Stingray.Cpu.SimdKernels.DotF32(wSlice, inSlice, cin);
                            }
                        }
                        outPtr[(long)co * houtLocal * woutLocal + ho * woutLocal + wo] = sum;
                    }
                }
            });
        }
        return output;
    }

    /// <summary>
    /// Real exact-erf GELU, corrected 2026-09-07 (see docs/audio-review-progress.md's audio-
    /// encoder bisection entries): the reference's `GeluModule` default is
    /// `GeluApproximation::ExactErf` (`0.5*x*(1+erf(x/sqrt(2)))`), used for every GELU in this
    /// encoder (conv stem, FFN, projector) -- NOT `OpenTail.Stingray.Cpu.SimdKernels.
    /// GeluInPlace`'s tanh approximation this class previously used, which "diverges by a
    /// measurable margin" from exact-erf per that kernel's own doc comment. Applied 5 times per
    /// forward pass (3 conv stages + FFN + projector), so even a small per-call difference
    /// compounds into a real, measurable end-to-end error.
    /// </summary>
    private static void GeluInPlace(float[] x)
    {
        const float invSqrt2 = 0.70710678118654752f;
        if (x.Length >= 4096)
        {
            int chunkSize = 2048;
            int numChunks = (x.Length + chunkSize - 1) / chunkSize;
            System.Threading.Tasks.Parallel.For(0, numChunks, chunkIdx =>
            {
                int start = chunkIdx * chunkSize;
                int end = Math.Min(x.Length, start + chunkSize);
                for (int i = start; i < end; i++)
                {
                    float v = x[i];
                    x[i] = 0.5f * v * (1f + Erf(v * invSqrt2));
                }
            });
            return;
        }
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + Erf(v * invSqrt2));
        }
    }

    /// <summary>Abramowitz &amp; Stegun 7.1.26 erf approximation (max absolute error ~1.5e-7,
    /// well within float32 precision) -- .NET's Math/MathF has no built-in erf.</summary>
    private static float Erf(float x)
    {
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float sign = x < 0f ? -1f : 1f;
        float ax = MathF.Abs(x);
        float t = 1f / (1f + p * ax);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-ax * ax);
        return sign * y;
    }

    /// <summary>
    /// Real `sinusoidal_positions` layout, corrected 2026-09-07 (see
    /// docs/audio-review-progress.md's "ROOT CAUSE FOUND, PRECISELY" entry): the reference
    /// (`audio_encoder.cpp`'s `sinusoidal_positions`) writes SPLIT-HALF sin/cos -- all `sin`
    /// values in the first half of the channel dimension `[0, channels/2)`, all `cos` values in
    /// the second half `[channels/2, channels)` -- NOT interleaved `(sin,cos,sin,cos,...)` pairs.
    /// This port previously used the interleaved layout, which silently scrambled every
    /// dimension of the positional contribution added to every single encoder token -- found via
    /// a real frame-by-frame comparison against the reference's dumped audio-encoder output that
    /// showed every one of 77 frames badly diverging despite matching aggregate statistics.
    /// </summary>
    /// <summary>Real per-chunk valid-token count, ported exactly from
    /// `qwen3_asr_audio_encoder_token_count` (`types.h`, not guessed): each full 100-frame chunk
    /// always yields exactly 13 tokens; the remainder (`frames % 100`) goes through the same
    /// 3x floor-div-by-2 downsample formula as the conv stem's real stride-2 halving.</summary>
    private static int EncoderTokenCount(int inputFrames)
    {
        int leave = inputFrames % 100;
        int featLengths = FloorDiv(leave - 1, 2) + 1;
        return FloorDiv(FloorDiv(featLengths - 1, 2) + 1 - 1, 2) + 1 + (inputFrames / 100) * 13;
    }

    private static int FloorDiv(int numerator, int denominator)
    {
        int quotient = numerator / denominator;
        int remainder = numerator % denominator;
        if (remainder != 0 && (remainder < 0) != (denominator < 0)) quotient--;
        return quotient;
    }

    private static float[] SinusoidalPositionalEmbeddings(int length, int channels)
    {
        var pe = new float[length * channels];
        int half = channels / 2;
        float logTimescale = MathF.Log(10000.0f) / (half - 1);
        for (int p = 0; p < length; p++)
        {
            int off = p * channels;
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Exp(-i * logTimescale);
                float angle = p * invFreq;
                pe[off + i] = MathF.Sin(angle);
                pe[off + half + i] = MathF.Cos(angle);
            }
        }
        return pe;
    }

    /// <summary>Linear layer via a <see cref="CfmLinearWeight"/> (real hardware F16C when available, F32 fallback otherwise) plus bias.</summary>
    private static float[] LinearBias(float[] input, CfmLinearWeight weight, float[] bias)
    {
        var output = weight.MatVec(input);
        for (int o = 0; o < output.Length; o++) output[o] += bias[o];
        return output;
    }

    private static unsafe float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f)
    {
        var output = new float[x.Length];
        fixed (float* op = output, xp = x, wp = weight, bp = bias)
        {
            OpenTail.Stingray.Cpu.SimdKernels.LayerNorm(op, xp, wp, bp, x.Length, eps);
        }
        return output;
    }

    public void Dispose()
    {
    }
}
