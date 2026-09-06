
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
public sealed class QwenAsrAudioEncoder : IDisposable
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

        var chunkTokens = new List<float[]>();
        for (int chunkStart = 0; chunkStart < numMelFrames; chunkStart += chunkFrameLimit)
        {
            int chunkLen = Math.Min(chunkFrameLimit, numMelFrames - chunkStart);
            var chunkMel = mel.Slice(chunkStart * nMels, chunkLen * nMels).ToArray();

            // mel input as a 1-channel image, IDX(0,h=t,w=f) = t*nMels + f (frame-major, matches
            // QwenAsrMelExtractor's layout directly, same convention as ParakeetConformerEncoder).
            var stage1 = Conv2dFull(chunkMel, cin: 1, hin: chunkLen, win: nMels, w.Conv1WeightCL, w.Conv1Bias, c, k: 3, stride: 2, pad: 1, out int h1, out int w1);
            GeluInPlace(stage1);

            var stage2 = Conv2dFull(stage1, cin: c, hin: h1, win: w1, w.Conv2WeightCL, w.Conv2Bias, c, k: 3, stride: 2, pad: 1, out int h2, out int w2);
            GeluInPlace(stage2);

            var stage3 = Conv2dFull(stage2, cin: c, hin: h2, win: w2, w.Conv3WeightCL, w.Conv3Bias, c, k: 3, stride: 2, pad: 1, out int h3, out int w3);
            GeluInPlace(stage3);

            int chunkT = h3;
            int flatDim = c * w3; // 480 * 16 = 7680, matches audio.conv_out.weight's input dim

            // Flatten channel-major (matches Parakeet's identical permute+flatten convention):
            // feature[k] = channel*w3 + freq_w.
            var chunkPosEmb = SinusoidalPositionalEmbeddings(chunkT, encDim);
            for (int ti = 0; ti < chunkT; ti++)
            {
                var row = new float[flatDim];
                for (int ch = 0; ch < c; ch++)
                    for (int fw = 0; fw < w3; fw++)
                        row[ch * w3 + fw] = stage3[ch * h3 * w3 + ti * w3 + fw];

                var tokenVec = w.ConvOutWeight.MatVec(row);
                for (int d = 0; d < encDim; d++)
                    tokenVec[d] += chunkPosEmb[ti * encDim + d];
                chunkTokens.Add(tokenVec);
            }
        }

        int t = chunkTokens.Count;
        var x = chunkTokens.ToArray();

        foreach (var layer in w.AudioLayerWeights)
            x = EncoderBlock(x, layer, t, encDim, Config.NumHeads);

        for (int ti = 0; ti < t; ti++)
            x[ti] = LayerNorm(x[ti], w.LnPostWeight, w.LnPostBias);

        // Adapter: proj1 (encDim -> encDim) + GELU -> proj2 (encDim -> QwenHiddenDim).
        int qwenDim = Config.QwenHiddenDim;
        var projected = new float[t * qwenDim];
        for (int ti = 0; ti < t; ti++)
        {
            var p1 = LinearBias(x[ti], w.Proj1Weight, w.Proj1Bias);
            GeluInPlace(p1);
            var p2 = LinearBias(p1, w.Proj2Weight, w.Proj2Bias);
            p2.CopyTo(projected.AsSpan(ti * qwenDim, qwenDim));
        }

        return (projected, t);
    }

    private static float[][] EncoderBlock(float[][] x, QwenAsrAudioLayerWeights l, int t, int dim, int heads)
    {
        var afterAttn = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            var normed = LayerNorm(x[i], l.AttnNormWeight, l.AttnNormBias);
            afterAttn[i] = normed;
        });

        var attnOut = SelfAttention(afterAttn, l, t, dim, heads);
        var afterResidual1 = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d];
            afterResidual1[i] = row;
        }

        var output = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            var normed = LayerNorm(afterResidual1[i], l.FfnNormWeight, l.FfnNormBias);
            var h = LinearBias(normed, l.FfnUpWeight, l.FfnUpBias);
            GeluInPlace(h);
            var down = LinearBias(h, l.FfnDownWeight, l.FfnDownBias);
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterResidual1[i][d] + down[d];
            output[i] = row;
        });

        return output;
    }

    private static float[][] SelfAttention(float[][] normed, QwenAsrAudioLayerWeights l, int t, int dim, int heads)
    {
        int headDim = dim / heads;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = new float[t][];
        var k = new float[t][];
        var v = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            q[i] = LinearBias(normed[i], l.AttnQWeight, l.AttnQBias);
            k[i] = LinearBias(normed[i], l.AttnKWeight, l.AttnKBias);
            v[i] = LinearBias(normed[i], l.AttnVWeight, l.AttnVBias);
        });

        var attnRaw = new float[t][];
        for (int i = 0; i < t; i++) attnRaw[i] = new float[dim];

        // Parallelize over t (audio frame count -- scales with clip length, typically far larger
        // than heads=14) rather than heads, matching the pattern in F5Kernels.
        // MultiHeadSelfAttention: much more parallel width for longer clips, same SIMD dot/AV
        // math either way (TensorPrimitives.Dot/MultiplyAdd), just a different loop nesting.
        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                var scores = new float[t];
                var probs = new float[t];
                var qi = q[i].AsSpan(off, headDim);
                for (int j = 0; j < t; j++)
                    scores[j] = TensorPrimitives.Dot(qi, k[j].AsSpan(off, headDim)) * scale;
                // NOT TensorPrimitives.SoftMax: this checkpoint's real Q/K weights produce raw
                // dot-product logits well into the 130-140 range (confirmed by direct
                // measurement, not a bug in Q/K -- large logits are a legitimate output of a
                // real trained attention head), and .NET 10's TensorPrimitives.SoftMax returns
                // NaN for inputs in that range (no max-subtraction stabilization on whatever
                // code path a length-8 span takes) even though the inputs themselves are
                // perfectly finite. DenseKernels.SoftmaxInPlace subtracts the max before
                // exponentiating (standard numerically-stable softmax), which the same
                // TensorPrimitives call apparently skips -- use that instead.
                scores.AsSpan(0, t).CopyTo(probs);
                OpenTail.Stingray.Audio.Primitives.DenseKernels.SoftmaxInPlace(probs);

                var weighted = attnRaw[i].AsSpan(off, headDim);
                for (int j = 0; j < t; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, headDim), probs[j], weighted, weighted);
            });
        }

        var output = new float[t][];
        for (int i = 0; i < t; i++)
            output[i] = LinearBias(attnRaw[i], l.AttnOutWeight, l.AttnOutBias);
        return output;
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
            for (int ci = 0; ci < cin; ci++)
                for (int hi = 0; hi < hin; hi++)
                    for (int wi = 0; wi < win; wi++)
                        inputCL[(hi * win + wi) * cin + ci] = input[ci * hin * win + hi * win + wi];
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

    private static unsafe void GeluInPlace(float[] x)
    {
        fixed (float* xp = x)
        {
            OpenTail.Stingray.Cpu.SimdKernels.GeluInPlace(xp, x.Length);
        }
    }

    private static float[] SinusoidalPositionalEmbeddings(int length, int channels)
    {
        var pe = new float[length * channels];
        float logTimescale = MathF.Log(10000.0f) / (channels / 2 - 1);
        for (int p = 0; p < length; p++)
        {
            int off = p * channels;
            for (int i = 0; i < channels / 2; i++)
            {
                float invFreq = MathF.Exp(-i * logTimescale);
                float angle = p * invFreq;
                pe[off + 2 * i] = MathF.Sin(angle);
                pe[off + 2 * i + 1] = MathF.Cos(angle);
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
