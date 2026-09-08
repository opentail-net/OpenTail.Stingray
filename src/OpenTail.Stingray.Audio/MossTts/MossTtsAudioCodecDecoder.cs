namespace OpenTail.Stingray.Audio.MossTts;

/// <summary>Decoded stereo waveform (interleaved L/R samples, real 48kHz).</summary>
public readonly struct MossTtsAudioWaveform(float[] left, float[] right)
{
    public float[] Left { get; } = left;
    public float[] Right { get; } = right;
}

/// <summary>
/// Native C# port of MOSS-Audio-Tokenizer-Nano's decoder forward pass, from
/// `examples/audio.cpp/src/framework/codecs/moss_audio_tokenizer_codec_runtime.cpp`'s
/// `MossAudioTokenizerDecoder::decode`/`Impl::prepare_graph` (not guessed). Real pipeline,
/// genuinely "CNN-free" (the reference's own description): dequantize RVQ codes to a
/// `[frames, CodeDim]` latent (see `MossTtsAudioCodecQuantizerWeights.DecodeFrame`) -&gt;
/// `PatchUpsample` by 4 (`decoder_initial_patch`) -&gt; four stages of [Linear input_proj -&gt; N
/// causal-RoPE-attention/LayerScale/exact-erf-GELU-MLP transformer layers -&gt; optional Linear
/// output_proj] each followed by its own `PatchUpsample` (by 2, 2, 2, 240 for the real nano
/// config) -&gt; a final `[interleavedFrames, 1]` stream that is really `[frames, 2]`
/// left/right-interleaved 48kHz stereo (`Channels=2`, matching the reference's "channel 0 = even
/// samples, channel 1 = odd samples" de-interleave).
///
/// <para><b>Not yet implemented</b>: the reference's chunked/windowed attention path for
/// sequences longer than its `kAttentionQueryChunk` (used only when generation runs long enough
/// that a stage's local step count exceeds that chunk size -- a real, separate optimization for
/// long-form generation, not needed for short/moderate outputs and precisely flagged here rather
/// than silently approximated). This port always builds one full `steps x steps` causal mask,
/// which is mathematically identical to the reference's windowed path for any sequence short
/// enough to stay under the chunk threshold.</para>
/// </summary>
public static class MossTtsAudioCodecDecoder
{
    /// <summary>
    /// Decodes `[NumQuantizers][frames]` RVQ codes (row per quantizer, matching
    /// <see cref="MossTtsAudioCodes"/>'s flattened `[frame * Codebooks + codebook]` layout after
    /// transposition -- see the overload below for that convenience) into a real stereo waveform.
    /// </summary>
    public static MossTtsAudioWaveform Decode(
        MossTtsAudioCodecQuantizerWeights quantizer,
        MossTtsAudioCodecDecoderWeights decoderWeights,
        int[][] codesPerQuantizer)
    {
        int numQuantizers = codesPerQuantizer.Length;
        if (numQuantizers != MossTtsAudioCodecQuantizerWeights.NumQuantizers)
            throw new ArgumentException($"Expected {MossTtsAudioCodecQuantizerWeights.NumQuantizers} codebooks, got {numQuantizers}.");
        int frames = codesPerQuantizer[0].Length;
        if (frames <= 0) throw new ArgumentException("MOSS codec decoder requires a non-empty code sequence.");

        // Dequantize: [frames][CodeDim].
        var latent = new float[frames][];
        var perFrameCodes = new int[numQuantizers];
        for (int f = 0; f < frames; f++)
        {
            for (int q = 0; q < numQuantizers; q++) perFrameCodes[q] = codesPerQuantizer[q][f];
            latent[f] = quantizer.DecodeFrame(perFrameCodes);
        }

        var hidden = PatchUpsample(latent, MossTtsAudioCodecDecoderWeights.DecoderInitialPatch);

        foreach (var stage in decoderWeights.Stages)
        {
            var afterStage = RunTransformerStage(hidden, stage);
            hidden = PatchUpsample(afterStage, stage.Patch);
        }

        // hidden is now [interleavedSamples][1] (packed=1 after the final patch=240 upsample) --
        // flatten to the interleaved stereo sample stream and de-interleave.
        int interleaved = hidden.Length;
        int perChannel = interleaved / MossTtsAudioCodecDecoderWeights.Channels;
        var left = new float[perChannel];
        var right = new float[perChannel];
        for (int i = 0; i < perChannel; i++)
        {
            left[i] = hidden[MossTtsAudioCodecDecoderWeights.Channels * i][0];
            right[i] = hidden[MossTtsAudioCodecDecoderWeights.Channels * i + 1][0];
        }
        return new MossTtsAudioWaveform(left, right);
    }

    /// <summary>
    /// Real `PatchedPretransform` reshape-upsample: `[frames][packed]` -&gt; `[frames*patch][packed/patch]`,
    /// where `output[frame*patch + p][c] = input[frame][c*patch + p]` (derived directly from the
    /// reference's `reshape(len,channels,patch) -&gt; transpose(2,3) -&gt; reshape(len*patch,channels)`,
    /// not guessed).
    /// </summary>
    public static float[][] PatchUpsample(float[][] input, int patch)
    {
        int frames = input.Length;
        int packed = input[0].Length;
        int channels = packed / patch;
        var output = new float[frames * patch][];
        for (int f = 0; f < frames; f++)
        {
            for (int p = 0; p < patch; p++)
            {
                var row = new float[channels];
                for (int c = 0; c < channels; c++) row[c] = input[f][c * patch + p];
                output[f * patch + p] = row;
            }
        }
        return output;
    }

    /// <summary>Real `PatchedPretransform` reshape-downsample (encoder direction, the exact inverse
    /// of <see cref="PatchUpsample"/>): `[l][d]` -&gt; `[l/patch][d*patch]`, where
    /// `output[lt][d_idx*patch+h_idx] = input[lt*patch+h_idx][d_idx]` (derived directly from the
    /// reference's `patch_downsample`, not guessed).</summary>
    public static float[][] PatchDownsample(float[][] input, int patch)
    {
        int totalLength = input.Length;
        int channels = input[0].Length;
        int length = totalLength / patch;
        var output = new float[length][];
        for (int lt = 0; lt < length; lt++)
        {
            var row = new float[channels * patch];
            for (int h = 0; h < patch; h++)
            {
                var src = input[lt * patch + h];
                for (int d = 0; d < channels; d++) row[d * patch + h] = src[d];
            }
            output[lt] = row;
        }
        return output;
    }

    internal static float[][] RunTransformerStage(float[][] input, MossTtsAudioCodecTransformerStageWeights stage)
    {
        int n = input.Length;
        int dModel = stage.DModel;
        int headDim = dModel / stage.NumHeads;
        float scale = 1f / MathF.Sqrt(headDim);

        var hidden = MossTtsGlobalTransformer.LinearBatched(input, stage.InputProjWeight, bias: new float[dModel], stage.InputDim, dModel);

        foreach (var layer in stage.Layers)
        {
            var normed = new float[n][];
            for (int i = 0; i < n; i++) normed[i] = MossTtsGlobalTransformer.LayerNorm(hidden[i], layer.Norm1Weight, layer.Norm1Bias);
            var qkvAll = MossTtsGlobalTransformer.LinearBatched(normed, layer.InProjWeight, bias: new float[dModel * 3], dModel, dModel * 3);

            var q = new float[n][];
            var k = new float[n][];
            var v = new float[n][];
            for (int i = 0; i < n; i++)
            {
                q[i] = new float[dModel];
                k[i] = new float[dModel];
                v[i] = new float[dModel];
                Array.Copy(qkvAll[i], 0, q[i], 0, dModel);
                Array.Copy(qkvAll[i], dModel, k[i], 0, dModel);
                Array.Copy(qkvAll[i], dModel * 2, v[i], 0, dModel);
                MossTtsGlobalTransformer.ApplyRopeAdjacentPairs(q[i], stage.NumHeads, headDim, position: i);
                MossTtsGlobalTransformer.ApplyRopeAdjacentPairs(k[i], stage.NumHeads, headDim, position: i);
            }

            int context = stage.Context;
            var contexts = new float[n][];
            Parallel.For(0, n, i =>
            {
                var ctxOut = new float[dModel];
                // Real causal_context_mask: key valid iff key<=query && query-key<context.
                int keyStart = Math.Max(0, i - context + 1);
                int availableKeys = i - keyStart + 1;
                var scores = new float[availableKeys];
                unsafe
                {
                    fixed (float* qp = q[i])
                    {
                        for (int hOffIdx = 0; hOffIdx < stage.NumHeads; hOffIdx++)
                        {
                            int hOff = hOffIdx * headDim;
                            for (int t = 0; t < availableKeys; t++)
                            {
                                fixed (float* kp = k[keyStart + t])
                                    scores[t] = SimdKernels.DotF32(qp + hOff, kp + hOff, headDim) * scale;
                            }
                            MossTtsGlobalTransformer.SoftmaxInPlace(scores);
                            for (int t = 0; t < availableKeys; t++)
                            {
                                var vt = v[keyStart + t];
                                float p = scores[t];
                                for (int d = 0; d < headDim; d++) ctxOut[hOff + d] += p * vt[hOff + d];
                            }
                        }
                    }
                }
                contexts[i] = ctxOut;
            });

            var attnOut = MossTtsGlobalTransformer.LinearBatched(contexts, layer.OutProjWeight, bias: new float[dModel], dModel, dModel);
            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < dModel; d++) attnOut[i][d] *= layer.LayerScale1[d];
                TensorPrimitives.Add(hidden[i], attnOut[i], hidden[i]);
            }

            var ffnNormed = new float[n][];
            for (int i = 0; i < n; i++) ffnNormed[i] = MossTtsGlobalTransformer.LayerNorm(hidden[i], layer.Norm2Weight, layer.Norm2Bias);
            var fcAll = MossTtsGlobalTransformer.LinearBatched(ffnNormed, layer.Fc1Weight, bias: new float[stage.IntermediateDim], dModel, stage.IntermediateDim);
            for (int i = 0; i < n; i++) GeluExactErfInPlace(fcAll[i]);
            var projAll = MossTtsGlobalTransformer.LinearBatched(fcAll, layer.Fc2Weight, bias: new float[dModel], stage.IntermediateDim, dModel);
            for (int i = 0; i < n; i++)
            {
                for (int d = 0; d < dModel; d++) projAll[i][d] *= layer.LayerScale2[d];
                TensorPrimitives.Add(hidden[i], projAll[i], hidden[i]);
            }
        }

        if (stage.OutputProjWeight is { } outputProjWeight)
            hidden = MossTtsGlobalTransformer.LinearBatched(hidden, outputProjWeight, bias: new float[stage.OutputDim], dModel, stage.OutputDim);

        return hidden;
    }

    /// <summary>Exact-erf GELU: 0.5*x*(1+erf(x/sqrt(2))) -- NOT the tanh approximation used by the
    /// global/local transformers' `gelu_new` (a real, deliberate difference between the two model
    /// families ported this session, confirmed by the reference's own `GeluApproximation::ExactErf`
    /// vs `::Tanh`).</summary>
    private static void GeluExactErfInPlace(float[] x)
    {
        const float invSqrt2 = 0.70710678118654752f;
        for (int i = 0; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = 0.5f * v * (1f + Erf(v * invSqrt2));
        }
    }

    /// <summary>Abramowitz-Stegun 7.1.26 rational erf approximation (max error ~1.5e-7).</summary>
    private static float Erf(float x)
    {
        const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f, a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);
        float t = 1f / (1f + p * x);
        float y = 1f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);
        return sign * y;
    }
}
