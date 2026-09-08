using OpenTail.Stingray.Audio.F5TTS;

namespace OpenTail.Stingray.Audio.NeuTts;

/// <summary>
/// Real NeuTTS FSQ acoustic-decoder codec forward pass, ported from the shared
/// `fsq_audio_codec_runtime.cpp`'s `decode_fsq_audio_codec_levels`/`resnet_block`/
/// `transformer_layer`/`CodecHeadGraph` (not guessed). Real pipeline: FSQ dequant (deterministic
/// mixed-radix code -&gt; per-level value, no learned table) -&gt; `quantizer.project_out` Linear -&gt;
/// `acoustic_decoder.fc` Linear -&gt; `embed` Conv1d(k=7) -&gt; `prior_net` (2x ResNet: GroupNorm-&gt;
/// SiLU-&gt;Conv1d(k=3)-&gt;GroupNorm-&gt;SiLU-&gt;Conv1d(k=3)-&gt;residual) -&gt; 12x Transformer layer
/// (RMSNorm-&gt;non-causal-MHA with NEOX RoPE-&gt;residual-&gt;RMSNorm-&gt;plain SiLU-MLP (fc1-&gt;SiLU-&gt;fc2,
/// NOT gated SwiGLU)-&gt;residual) -&gt; `post_net` (2x ResNet) -&gt; final LayerNorm -&gt; `istft_head`
/// Linear(hidden-&gt;hop*4+2) -&gt; Vocos-style log-magnitude/phase ISTFT (reuses
/// `VocosVocoder`'s real formula, same technique, different real n_fft/hop for this checkpoint).
/// Channel-last `[T,D]` throughout -- unlike the reference's ggml graph, no transpose choreography
/// is needed since every kernel here (`F5Kernels.Conv1dSamePad` included) already works in that
/// layout directly.
/// </summary>
public static class NeuTtsAudioDecoder
{
    public static float[] Decode(NeuTtsAudioDecoderWeights w, int[] codes)
    {
        int frames = codes.Length;
        int numLevels = NeuTtsAudioDecoderWeights.QuantizationLevels.Length;

        // Real FSQ dequant: mixed-radix decomposition of `code` into per-dimension level indices,
        // mapped to [-1, 1] via `levelIndex * (2/(level-1)) - 1`.
        var levels = new float[frames * numLevels];
        long codebookSize = 1;
        foreach (var lv in NeuTtsAudioDecoderWeights.QuantizationLevels) codebookSize *= lv;
        for (int f = 0; f < frames; f++)
        {
            long code = codes[f];
            if (code < 0 || code >= codebookSize) throw new ArgumentOutOfRangeException(nameof(codes), $"code {code} out of range for codebook size {codebookSize}.");
            long basis = 1;
            for (int d = 0; d < numLevels; d++)
            {
                int level = NeuTtsAudioDecoderWeights.QuantizationLevels[d];
                long levelIndex = (code / basis) % level;
                levels[f * numLevels + d] = levelIndex * (2f / (level - 1)) - 1f;
                basis *= level;
            }
        }

        var x = F5Kernels.Linear(levels, frames, numLevels, w.QuantizerProjectOutWeight, w.QuantizerProjectOutBias, NeuTtsAudioDecoderWeights.QuantizationDim);
        x = F5Kernels.Linear(x, frames, NeuTtsAudioDecoderWeights.QuantizationDim, w.AcousticFcWeight, w.AcousticFcBias, NeuTtsAudioDecoderWeights.HiddenSize);
        x = F5Kernels.Conv1dSamePad(x, frames, NeuTtsAudioDecoderWeights.HiddenSize, w.EmbedConvWeight, w.EmbedConvBias, NeuTtsAudioDecoderWeights.HiddenSize, kernel: 7);

        foreach (var block in w.PriorNet) x = ResnetBlock(x, frames, block);

        var (cos, sin) = BuildNeoxRope(frames);
        foreach (var layer in w.Layers) x = TransformerLayer(x, frames, layer, cos, sin);

        foreach (var block in w.PostNet) x = ResnetBlock(x, frames, block);

        x = F5Kernels.LayerNorm(x, frames, NeuTtsAudioDecoderWeights.HiddenSize, w.FinalNormWeight, w.FinalNormBias, eps: 1e-6f);

        int outDim = NeuTtsAudioDecoderWeights.HopLength * 4 + 2;
        var head = F5Kernels.Linear(x, frames, NeuTtsAudioDecoderWeights.HiddenSize, w.IstftHeadWeight, w.IstftHeadBias, outDim);

        return IstftHead(head, frames, outDim);
    }

    private static float[] ResnetBlock(float[] x, int t, NeuTtsResnetBlockWeights b)
    {
        int c = NeuTtsAudioDecoderWeights.HiddenSize;
        var h = GroupNorm(x, t, c, b.Norm1Weight, b.Norm1Bias);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.SiLU(h[i]);
        h = F5Kernels.Conv1dSamePad(h, t, c, b.Conv1Weight, b.Conv1Bias, c, kernel: 3);
        h = GroupNorm(h, t, c, b.Norm2Weight, b.Norm2Bias);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.SiLU(h[i]);
        h = F5Kernels.Conv1dSamePad(h, t, c, b.Conv2Weight, b.Conv2Bias, c, kernel: 3);
        var output = new float[x.Length];
        for (int i = 0; i < output.Length; i++) output[i] = x[i] + h[i];
        return output;
    }

    /// <summary>Real GroupNorm (32 groups, eps=1e-6, affine), channel-last [T,C]: normalizes
    /// each group of `C/32` channels JOINTLY across the time axis (matches PyTorch's
    /// `nn.GroupNorm` -- statistics are per (batch, group), pooled over both spatial/time AND the
    /// group's channels, not per-channel like LayerNorm).</summary>
    private static float[] GroupNorm(float[] x, int t, int channels, float[] weight, float[] bias, int numGroups = 32, float eps = 1e-6f)
    {
        int groupSize = channels / numGroups;
        var output = new float[x.Length];
        for (int g = 0; g < numGroups; g++)
        {
            double sum = 0, sumSq = 0;
            long count = (long)t * groupSize;
            for (int ti = 0; ti < t; ti++)
            {
                int baseIdx = ti * channels + g * groupSize;
                for (int c = 0; c < groupSize; c++)
                {
                    float v = x[baseIdx + c];
                    sum += v;
                    sumSq += (double)v * v;
                }
            }
            double mean = sum / count;
            double variance = sumSq / count - mean * mean;
            float invStd = (float)(1.0 / Math.Sqrt(variance + eps));
            for (int ti = 0; ti < t; ti++)
            {
                int baseIdx = ti * channels + g * groupSize;
                for (int c = 0; c < groupSize; c++)
                {
                    int ch = g * groupSize + c;
                    output[baseIdx + c] = (float)((x[baseIdx + c] - mean) * invStd) * weight[ch] + bias[ch];
                }
            }
        }
        return output;
    }

    private static float[] RmsNorm(float[] x, int t, int dim, float[] weight, float eps = NeuTtsAudioDecoderWeights.RmsNormEps)
    {
        var output = new float[x.Length];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            double sumSq = 0;
            for (int d = 0; d < dim; d++) sumSq += (double)x[off + d] * x[off + d];
            float invRms = (float)(1.0 / Math.Sqrt(sumSq / dim + eps));
            for (int d = 0; d < dim; d++) output[off + d] = x[off + d] * invRms * weight[d];
        }
        return output;
    }

    private static (float[] Cos, float[] Sin) BuildNeoxRope(int t)
    {
        int half = NeuTtsAudioDecoderWeights.HeadDim / 2;
        var cos = new float[t * half];
        var sin = new float[t * half];
        for (int pos = 0; pos < t; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float freq = MathF.Pow(NeuTtsAudioDecoderWeights.RopeTheta, -2f * i / NeuTtsAudioDecoderWeights.HeadDim);
                float angle = pos * freq;
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    /// <summary>Real NEOX RoPE: pairs `x[i]` with `x[i+halfHead]` (split-half), NOT the interleaved
    /// `x[2i]`/`x[2i+1]` convention `F5Kernels.ApplyRotary` implements for F5-TTS's DiT --
    /// deliberately NOT reused here (this session's VoxCPM2 bisection found this exact
    /// interleaved-vs-NEOX mismatch to be a real, previously-shipped bug elsewhere).</summary>
    private static void ApplyNeoxRope(float[] x, int t, int heads, int headDim, float[] cos, float[] sin)
    {
        int dim = heads * headDim;
        int half = headDim / 2;
        for (int ti = 0; ti < t; ti++)
        {
            int angleBase = ti * half;
            for (int h = 0; h < heads; h++)
            {
                int hOff = ti * dim + h * headDim;
                for (int i = 0; i < half; i++)
                {
                    float c = cos[angleBase + i], s = sin[angleBase + i];
                    float x0 = x[hOff + i], x1 = x[hOff + half + i];
                    x[hOff + i] = x0 * c - x1 * s;
                    x[hOff + half + i] = x1 * c + x0 * s;
                }
            }
        }
    }

    private static float[] TransformerLayer(float[] x, int t, NeuTtsCodecTransformerLayerWeights layer, float[] cos, float[] sin)
    {
        int dim = NeuTtsAudioDecoderWeights.HiddenSize, heads = NeuTtsAudioDecoderWeights.NumHeads, headDim = NeuTtsAudioDecoderWeights.HeadDim;
        var normed = RmsNorm(x, t, dim, layer.AttnNormWeight);
        var q = F5Kernels.Linear(normed, t, dim, layer.QWeight, null, dim);
        var k = F5Kernels.Linear(normed, t, dim, layer.KWeight, null, dim);
        var v = F5Kernels.Linear(normed, t, dim, layer.VWeight, null, dim);
        ApplyNeoxRope(q, t, heads, headDim, cos, sin);
        ApplyNeoxRope(k, t, heads, headDim, cos, sin);
        var context = F5Kernels.MultiHeadSelfAttention(q, k, v, t, heads, headDim);
        var attnOut = F5Kernels.Linear(context, t, dim, layer.OutWeight, null, dim);
        var hidden = new float[x.Length];
        for (int i = 0; i < hidden.Length; i++) hidden[i] = x[i] + attnOut[i];

        var ffnNormed = RmsNorm(hidden, t, dim, layer.FfnNormWeight);
        var ff = F5Kernels.Linear(ffnNormed, t, dim, layer.Fc1Weight, null, NeuTtsAudioDecoderWeights.IntermediateSize);
        for (int i = 0; i < ff.Length; i++) ff[i] = F5Kernels.SiLU(ff[i]);
        ff = F5Kernels.Linear(ff, t, NeuTtsAudioDecoderWeights.IntermediateSize, layer.Fc2Weight, null, dim);
        var output = new float[hidden.Length];
        for (int i = 0; i < output.Length; i++) output[i] = hidden[i] + ff[i];
        return output;
    }

    /// <summary>Real Vocos-style ISTFT head: `head` is channel-last [T, hop*4+2] (first half
    /// log-magnitude, second half phase). Same real formula as `VocosVocoder.IstftHead`/
    /// `CenteredIstft` (`mag=clip(exp(x),100)`, `S=mag*(cos+isin)`, centered overlap-add ISTFT),
    /// reimplemented here with this checkpoint's own real n_fft/hop rather than reusing
    /// `VocosVocoder` directly (different, unrelated model/config -- same technique, not shared
    /// code, to avoid coupling two independent model ports).</summary>
    private static float[] IstftHead(float[] head, int t, int outDim)
    {
        int nFft = NeuTtsAudioDecoderWeights.NFft, hop = NeuTtsAudioDecoderWeights.HopLength;
        int numBins = nFft / 2 + 1;

        var specReal = new float[t * numBins];
        var specImag = new float[t * numBins];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * outDim;
            for (int kBin = 0; kBin < numBins; kBin++)
            {
                float mag = MathF.Exp(head[off + kBin]);
                if (mag > 100f) mag = 100f;
                float phase = head[off + numBins + kBin];
                specReal[ti * numBins + kBin] = mag * MathF.Cos(phase);
                specImag[ti * numBins + kBin] = mag * MathF.Sin(phase);
            }
        }

        // Periodic Hann window (Kokoro STFT family): denom = nFft
        var window = new float[nFft];
        for (int i = 0; i < nFft; i++)
            window[i] = 0.5f - 0.5f * MathF.Cos(2f * MathF.PI * i / nFft);

        int pad = (nFft - hop) / 2;
        int rawLen = (t - 1) * hop + nFft;
        var ola = new float[rawLen];
        var envelope = new float[rawLen];
        var frameReal = new float[numBins];
        var frameImag = new float[numBins];
        var frameTime = new float[nFft];
        for (int ti = 0; ti < t; ti++)
        {
            Array.Copy(specReal, ti * numBins, frameReal, 0, numBins);
            Array.Copy(specImag, ti * numBins, frameImag, 0, numBins);
            SpectralKernels.InverseRealFft(frameReal, frameImag, frameTime);

            int start = ti * hop;
            for (int n = 0; n < nFft; n++)
            {
                ola[start + n] += frameTime[n] * window[n];
                envelope[start + n] += window[n] * window[n];
            }
        }

        int outLen = rawLen - 2 * pad;
        var output = new float[outLen];
        for (int i = 0; i < outLen; i++)
        {
            float env = envelope[pad + i];
            output[i] = env > 1e-11f ? ola[pad + i] / env : 0f;
        }
        return output;
    }
}
