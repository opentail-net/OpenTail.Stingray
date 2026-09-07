using OpenTail.Stingray.Audio.OmniVoice;

namespace OpenTail.Stingray.Audio.HiggsAudio;

/// <summary>
/// Real encode-direction primitives for Higgs Audio TTS's acoustic codec, ported from
/// `codec.cpp`'s `codec_encode`/`quantizer_encode` (not guessed). This class covers the
/// `codec_project` combination and the real RVQ `quantizer_encode` loop -- the two encoders that
/// FEED it (real hidden-state-averaged HuBERT semantic forward pass, and the separate real
/// `semantic_encoder` residual-conv post-network) are NOT yet ported (see
/// docs/audio-review-progress.md's "real scope correction" entry, 2026-09-07) -- so this is not
/// yet a full voice-cloning `Encode(waveform) -> codes` entry point, only the self-contained
/// project+quantize tail that CONSUMES those encoders' real `[832]`-wide concatenated output.
/// </summary>
public static class HiggsCodecEncoder
{
    /// <summary>Real `codec_project`: combines one frame's concatenated
    /// `[acousticHidden(256), semanticHidden(768)]` vector into the codec's real
    /// `[1024]`-wide hidden space.</summary>
    public static float[] Project(HiggsCodecDecoderWeights w, float[] concatenatedFrame)
    {
        if (concatenatedFrame.Length != HiggsCodecDecoderWeights.CodecProjectInputSize)
            throw new ArgumentException($"Expected length {HiggsCodecDecoderWeights.CodecProjectInputSize}.", nameof(concatenatedFrame));
        return LinearBias(concatenatedFrame, w.CodecProjectWeight, w.CodecProjectBias,
            HiggsCodecDecoderWeights.CodecProjectInputSize, HiggsCodecDecoderWeights.CodecHiddenSize);
    }

    /// <summary>
    /// Real `quantizer_encode`: a standard residual vector quantization loop across
    /// `NumCodebooks=8` real levels. Each step projects the current residual to the codebook's
    /// `[64]`-dim space (`project_in`), finds the nearest codebook entry via the real expanded-
    /// squared-distance argmax (`2*dot - ||x||^2 - ||e||^2`, mathematically `argmin ||x-e||^2`,
    /// matching the reference's real ggml formula exactly rather than a naive O(size*dim) direct
    /// distance loop -- same asymptotic cost, same result), projects the quantized entry back to
    /// `[1024]` (`project_out`), and subtracts it from the residual before the next codebook.
    /// Real per-frame call (`hiddenFrame` is one post-`Project` `[1024]` vector); the caller loops
    /// over frames.
    /// </summary>
    public static int[] QuantizeFrame(HiggsCodecDecoderWeights w, float[] hiddenFrame)
    {
        if (hiddenFrame.Length != HiggsCodecDecoderWeights.CodecHiddenSize)
            throw new ArgumentException($"Expected length {HiggsCodecDecoderWeights.CodecHiddenSize}.", nameof(hiddenFrame));

        var codes = new int[HiggsCodecDecoderWeights.NumCodebooks];
        var residual = (float[])hiddenFrame.Clone();

        for (int cb = 0; cb < HiggsCodecDecoderWeights.NumCodebooks; cb++)
        {
            var q = w.Quantizers[cb];
            var projected = LinearBias(residual, q.ProjectInWeight, q.ProjectInBias,
                HiggsCodecDecoderWeights.CodecHiddenSize, HiggsCodecDecoderWeights.CodebookDim);

            double xNormSq = 0;
            for (int d = 0; d < projected.Length; d++) xNormSq += (double)projected[d] * projected[d];

            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int e = 0; e < HiggsCodecDecoderWeights.CodebookSize; e++)
            {
                int eBase = e * HiggsCodecDecoderWeights.CodebookDim;
                double dot = 0, eNormSq = 0;
                for (int d = 0; d < HiggsCodecDecoderWeights.CodebookDim; d++)
                {
                    float ev = q.CodebookEmbed[eBase + d];
                    dot += (double)projected[d] * ev;
                    eNormSq += (double)ev * ev;
                }
                double score = 2.0 * dot - xNormSq - eNormSq;
                if (score > bestScore) { bestScore = score; best = e; }
            }
            codes[cb] = best;

            var quantizedLatent = new float[HiggsCodecDecoderWeights.CodebookDim];
            Array.Copy(q.CodebookEmbed, best * HiggsCodecDecoderWeights.CodebookDim, quantizedLatent, 0, HiggsCodecDecoderWeights.CodebookDim);
            var quantized = LinearBias(quantizedLatent, q.ProjectOutWeight, q.ProjectOutBias,
                HiggsCodecDecoderWeights.CodebookDim, HiggsCodecDecoderWeights.CodecHiddenSize);

            for (int d = 0; d < residual.Length; d++) residual[d] -= quantized[d];
        }

        return codes;
    }

    /// <summary>
    /// Real, FULL `codec_encode` chain, added 2026-09-07 once every real sub-piece existed (real
    /// acoustic encoder, real hidden-state-averaged semantic encoder, real semantic post-network,
    /// real `codec_project`+RVQ quantize) -- ported from `codec.cpp`'s `codec_encode` exactly:
    /// acoustic-encode the real 24kHz waveform (its own real frame count becomes `targetFrames`),
    /// hidden-state-mean-encode the real 16kHz waveform then run it through the semantic post-
    /// network, concatenate `[acoustic(256), semantic(768)]` per frame, `Project` to `[1024]`,
    /// `QuantizeFrame` to real 8-codebook codes. Callers are responsible for real resampling
    /// to 24kHz/16kHz and the real `kSemanticPadSamples=160` zero-padding on the 16kHz branch
    /// (matching `prepare_semantic_audio_16k`/`prepare_codec_audio_24k`, not done here).
    /// </summary>
    public static int[][] Encode(
        OmniVoiceAcousticEncoderWeights acousticWeights,
        OmniVoiceSemanticWeights semanticWeights,
        HiggsSemanticPostEncoder.Weights semanticPostWeights,
        HiggsCodecDecoderWeights codecWeights,
        ReadOnlySpan<float> waveform24k,
        ReadOnlySpan<float> waveform16k)
    {
        var (acousticLatent, targetFrames) = OmniVoiceAcousticEncoder.Encode(acousticWeights, waveform24k);

        var semanticMean = OmniVoiceSemanticEncoder.ForwardHiddenStateMean(semanticWeights, waveform16k, targetFrames);
        var semanticPost = HiggsSemanticPostEncoder.Forward(semanticPostWeights, semanticMean);

        var codes = new int[HiggsCodecDecoderWeights.NumCodebooks][];
        for (int cb = 0; cb < HiggsCodecDecoderWeights.NumCodebooks; cb++) codes[cb] = new int[targetFrames];

        const int acousticDim = 256;
        for (int f = 0; f < targetFrames; f++)
        {
            var concat = new float[HiggsCodecDecoderWeights.CodecProjectInputSize];
            for (int c = 0; c < acousticDim; c++) concat[c] = acousticLatent[c * targetFrames + f];
            for (int c = 0; c < HiggsSemanticPostEncoder.HiddenDim; c++) concat[acousticDim + c] = semanticPost[f][c];

            var hidden = Project(codecWeights, concat);
            var frameCodes = QuantizeFrame(codecWeights, hidden);
            for (int cb = 0; cb < HiggsCodecDecoderWeights.NumCodebooks; cb++) codes[cb][f] = frameCodes[cb];
        }
        return codes;
    }

    private static float[] LinearBias(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += weight[wBase + i] * input[i];
            output[o] = sum;
        }
        return output;
    }
}
