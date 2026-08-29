
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// Real Vocos vocoder (charactr/vocos-mel-24khz) forward pass: VocosBackbone (Conv1d embed +
/// LayerNorm + 8x ConvNeXtBlock + final LayerNorm) + ISTFTHead (Linear -> mag/phase -> centered
/// ISTFT). Ported from the real `vocos` PyPI package's `vocos/models.py` and `vocos/heads.py`,
/// verified against the real PyTorch reference (see scratch-llamacpp-ref/vocos_golden_decode.py).
///
/// Reuses <see cref="F5Kernels"/> (channel-last `[T,D]` primitives, shared with the F5-TTS DiT
/// port -- both operate on the same sequence-major layout).
/// </summary>
public static class VocosVocoder
{
    /// <summary>mel is channel-last [numFrames, MelDim] (100 channels). Returns mono waveform samples, length (numFrames-1)*HopLength.</summary>
    public static float[] Decode(VocosWeights w, float[] mel, int numFrames)
    {
        int dim = VocosWeights.HiddenDim;

        var x = F5Kernels.Conv1dSamePad(mel, numFrames, VocosWeights.MelDim, w.EmbedWeight, w.EmbedBias, dim, kernel: 7);
        x = F5Kernels.LayerNorm(x, numFrames, dim, w.NormWeight, w.NormBias);

        for (int i = 0; i < w.Blocks.Length; i++)
            x = ConvNeXtBlock(x, numFrames, dim, w.Blocks[i]);

        x = F5Kernels.LayerNorm(x, numFrames, dim, w.FinalNormWeight, w.FinalNormBias);

        return IstftHead(w, x, numFrames);
    }

    /// <summary>ConvNeXtBlock.forward: residual + gamma * pwconv2(gelu(pwconv1(layernorm(dwconv(x))))).</summary>
    private static float[] ConvNeXtBlock(float[] x, int t, int dim, VocosConvNeXtBlockWeights bw)
    {
        var h = F5Kernels.DepthwiseConv1dSamePad(x, t, dim, bw.DwConvWeight, bw.DwConvBias, kernel: 7);
        h = F5Kernels.LayerNorm(h, t, dim, bw.NormWeight, bw.NormBias);

        int inter = VocosWeights.IntermediateDim;
        h = F5Kernels.Linear(h, t, dim, bw.PwConv1Weight, bw.PwConv1Bias, inter);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.GeluExact(h[i]);
        h = F5Kernels.Linear(h, t, inter, bw.PwConv2Weight, bw.PwConv2Bias, dim);

        var output = new float[x.Length];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * dim;
            for (int d = 0; d < dim; d++) output[off + d] = x[off + d] + bw.Gamma[d] * h[off + d];
        }
        return output;
    }

    /// <summary>
    /// ISTFTHead.forward: out = Linear(dim, n_fft+2); split into mag/phase (n_fft/2+1 each);
    /// mag = clip(exp(mag), max=100); S = mag*(cos(phase) + i*sin(phase)); audio = centered ISTFT(S).
    /// </summary>
    private static float[] IstftHead(VocosWeights w, float[] x, int t)
    {
        int nFft = VocosWeights.NFft;
        int numBins = VocosWeights.NumBins;
        int outDim = nFft + 2;

        var proj = F5Kernels.Linear(x, t, VocosWeights.HiddenDim, w.HeadOutWeight, w.HeadOutBias, outDim);

        // proj is channel-last [T, n_fft+2]; first half (numBins) = magnitude logits, second half = phase.
        var specReal = new float[t * numBins];
        var specImag = new float[t * numBins];
        for (int ti = 0; ti < t; ti++)
        {
            int off = ti * outDim;
            for (int k = 0; k < numBins; k++)
            {
                float mag = MathF.Exp(proj[off + k]);
                if (mag > 100f) mag = 100f;
                float phase = proj[off + numBins + k];
                specReal[ti * numBins + k] = mag * MathF.Cos(phase);
                specImag[ti * numBins + k] = mag * MathF.Sin(phase);
            }
        }

        return CenteredIstft(specReal, specImag, t, nFft, VocosWeights.HopLength);
    }

    /// <summary>
    /// torch.istft(..., center=True) semantics: per-frame irfft, multiply by the analysis
    /// window, overlap-add at hop_length spacing into a (t-1)*hop+n_fft buffer, normalize by the
    /// overlap-added squared-window envelope, then trim n_fft/2 samples off each end (undoing the
    /// implicit center-padding the forward STFT would have used) -- final length (t-1)*hop.
    /// </summary>
    private static float[] CenteredIstft(float[] specReal, float[] specImag, int t, int nFft, int hop)
    {
        int numBins = nFft / 2 + 1;
        // torch.hann_window(n) default is periodic=True: 0.5*(1-cos(2*pi*i/n)).
        var window = SpectralKernels.CreateHannWindow(nFft);

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

        int pad = nFft / 2;
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
