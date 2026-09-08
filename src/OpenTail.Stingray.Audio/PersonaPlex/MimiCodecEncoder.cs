namespace OpenTail.Stingray.Audio.PersonaPlex;

/// <summary>
/// Real forward pass for the Mimi neural codec's ENCODE path (waveform -&gt; RVQ codes), ported from
/// `mimi_codec_runtime.cpp`'s `MimiEncoderPreTransformerGraph`/`MimiEncoderPostTransformerGraph`/
/// `quantize_projected` (not guessed) -- the real mirror direction of the already-ported
/// <see cref="MimiCodecDecoder"/>, needed for PersonaPlex's live-duplex user-audio conditioning. A
/// ONE-SHOT (non-streaming) port, matching <see cref="MimiCodecDecoder"/>'s own established
/// simplification: every real "stateful" conv degenerates to a plain causal (left-zero-pad) Conv1d
/// for a single full-sequence call with no prior state (see
/// <see cref="MimiCodecDecoder.CausalConv1dStrided"/>'s doc comment for the exact real formula).
/// </summary>
public static class MimiCodecEncoder
{
    /// <summary>Encodes a mono waveform (real 24kHz input, `[samples]`) into `[frames][8]` RVQ
    /// codes (codebook 0 = semantic, 1-7 = acoustic, matching <see cref="MimiCodecDecoder.Decode"/>'s
    /// own expected input layout).</summary>
    public static int[][] Encode(MimiCodecEncoderWeights w, float[] monoWaveform)
    {
        var hidden = new float[MimiCodecEncoderWeights.InputChannels][] { monoWaveform };
        hidden = MimiCodecDecoder.CausalConv1d(hidden, MimiCodecEncoderWeights.InputChannels, 64, w.InputProjWeight, w.InputProjBias, kernel: 7, dilation: 1);

        for (int stage = 0; stage < 4; stage++)
        {
            int inCh = MimiCodecEncoderWeights.StageResidualChannels[stage];
            int hiddenCh = MimiCodecEncoderWeights.StageHiddenChannels[stage];
            int outCh = stage < 3 ? MimiCodecEncoderWeights.StageResidualChannels[stage + 1] : 1024;
            int kernel = MimiCodecEncoderWeights.StageDownsampleKernels[stage];
            int stride = MimiCodecEncoderWeights.StageDownsampleStrides[stage];
            var sw = w.Stages[stage];

            hidden = MimiCodecDecoder.SeanetResidualUnit(hidden, inCh, hiddenCh, sw.ResidualBlock);
            MimiCodecDecoder.Elu(hidden);
            hidden = MimiCodecDecoder.CausalConv1dStrided(hidden, inCh, outCh, sw.DownsampleWeight, sw.DownsampleBias, kernel, stride);
        }

        MimiCodecDecoder.Elu(hidden);
        hidden = MimiCodecDecoder.CausalConv1d(hidden, 1024, MimiCodecDecoderWeights.HiddenDim, w.OutputProjWeight, w.OutputProjBias, kernel: 3, dilation: 1);

        var transformed = MimiCodecDecoder.RunTransformer(w.TransformerLayers, hidden);

        var downsampled = MimiCodecDecoder.CausalConv1dStrided(
            transformed, MimiCodecDecoderWeights.HiddenDim, MimiCodecDecoderWeights.HiddenDim,
            w.DownsampleConvWeight, w.DownsampleConvBias, kernel: 4, stride: 2);

        int frames = downsampled[0].Length;
        int hiddenDim = MimiCodecDecoderWeights.HiddenDim;
        int latentSize = MimiCodecDecoderWeights.LatentSize;

        var codes = new int[frames][];
        for (int t = 0; t < frames; t++)
        {
            var latent = new float[hiddenDim];
            for (int c = 0; c < hiddenDim; c++) latent[c] = downsampled[c][t];

            var semanticLatent = Linear(latent, w.SemanticInputProjWeight, w.SemanticInputProjBias, hiddenDim, latentSize);
            var acousticLatent = Linear(latent, w.AcousticInputProjWeight, w.AcousticInputProjBias, hiddenDim, latentSize);

            var frameCodes = new int[MimiCodecDecoderWeights.ActiveCodebooks];
            frameCodes[0] = NearestCode(semanticLatent, w.SemanticCodebook.Embedding, latentSize);
            var residual = acousticLatent;
            for (int q = 0; q < w.AcousticCodebooks.Length; q++)
            {
                var table = w.AcousticCodebooks[q].Embedding;
                int best = NearestCode(residual, table, latentSize);
                frameCodes[1 + q] = best;
                int baseIdx = best * latentSize;
                for (int d = 0; d < latentSize; d++) residual[d] -= table[baseIdx + d];
            }
            codes[t] = frameCodes;
        }
        return codes;
    }

    /// <summary>Real plain-Euclidean nearest-code search (`quantize_projected`'s real formula --
    /// minimize squared distance, NOT cosine similarity/L2-normalized like MOSS-TTS-Nano's RVQ).</summary>
    private static int NearestCode(float[] latent, float[] table, int latentSize)
    {
        int best = 0;
        float bestDist = float.PositiveInfinity;
        int codebookSize = table.Length / latentSize;
        for (int code = 0; code < codebookSize; code++)
        {
            int baseIdx = code * latentSize;
            float dist = 0f;
            for (int d = 0; d < latentSize; d++)
            {
                float delta = latent[d] - table[baseIdx + d];
                dist += delta * delta;
            }
            if (dist < bestDist) { bestDist = dist; best = code; }
        }
        return best;
    }

    private static float[] Linear(float[] input, float[] weight, float[]? bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias?[o] ?? 0f;
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }
}
