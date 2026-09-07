using OpenTail.Stingray.Audio.HiggsAudio;

namespace OpenTail.Stingray.Audio.OmniVoice;

/// <summary>
/// Real, FULL `codec_encode` chain for OmniVoice's OWN reference-audio encode path, ported from
/// `audio_tokenizer.cpp` (not guessed), added 2026-09-07 once every real sub-piece was confirmed.
/// This mirrors <see cref="HiggsAudio.HiggsCodecEncoder"/> structurally (both models share the
/// same real HuBERT/DAC lineage -- see docs/audio-review-progress.md's scope-correction entry),
/// but is NOT a blind copy: two real differences were checked against OmniVoice's own real
/// checkpoint (`models/_models/omnivoice/audio_tokenizer/config.json`) before reuse:
/// <list type="bullet">
/// <item>Semantic downsample stride: OmniVoice's real formula is `semantic_downsample_factor() =
/// hop_length / (sample_rate/semantic_sample_rate) / downsample_factor` = `960 / (24000/16000) /
/// 320` = `2` for THIS checkpoint's real config values (hop_length=960, sample_rate=24000,
/// semantic_sample_rate=16000, downsample_factor=320) -- confirmed to equal Higgs's hardcoded 2,
/// not assumed.</item>
/// <item>Semantic post-encoder (`encoder_semantic.*`) block shape: OmniVoice's real config
/// (`kernel_size=3`, `strides=[1,1]`, `block_dilations=[1,1]`, `unit_kernel_size=3`) makes its
/// real `build_semantic_encoder` degenerate EXACTLY to Higgs's fixed 2-block/2-residual-unit/
/// kernel=3/stride=1/dilation=1 structure (real tensor names also match:
/// `encoder_semantic.conv`/`encoder_semantic.conv_blocks.N.res_units.M.conv1|conv2`/
/// `encoder_semantic.conv_blocks.N.conv`) -- so <see cref="HiggsSemanticPostEncoder"/> is reused
/// directly against OmniVoice's own (bare, unprefixed) tensor names rather than duplicated. This
/// was checked, not assumed -- a different real checkpoint with non-1 `strides` would need its
/// own post-encoder class.</item>
/// </list>
/// The quantizer `score` nearest-neighbor layer is likewise confirmed mathematically identical to
/// Higgs's direct expanded-squared-distance argmax (see `OmniVoiceQuantizerWeights.ProjectInWeight`'s
/// doc comment), so <see cref="HiggsCodecEncoder"/>'s per-frame quantize math is reused verbatim
/// with OmniVoice's own weights rather than reimplemented.
/// </summary>
public static class OmniVoiceCodecEncoder
{
    public static int[][] Encode(
        OmniVoiceAcousticEncoderWeights acousticWeights,
        OmniVoiceSemanticWeights semanticWeights,
        HiggsSemanticPostEncoder.Weights semanticPostWeights,
        OmniVoiceAcousticDecoderWeights codecWeights,
        ReadOnlySpan<float> waveform24k,
        ReadOnlySpan<float> waveform16k)
    {
        var (acousticLatent, targetFrames) = OmniVoiceAcousticEncoder.Encode(acousticWeights, waveform24k);

        var semanticMean = OmniVoiceSemanticEncoder.ForwardHiddenStateMean(semanticWeights, waveform16k, targetFrames, downsampleStride: 2);
        var semanticPost = HiggsSemanticPostEncoder.Forward(semanticPostWeights, semanticMean);

        var codes = new int[OmniVoiceAcousticDecoderWeights.NumCodebooks][];
        for (int cb = 0; cb < OmniVoiceAcousticDecoderWeights.NumCodebooks; cb++) codes[cb] = new int[targetFrames];

        const int acousticDim = OmniVoiceAcousticDecoderWeights.AcousticLatentDim; // 256
        int concatWidth = acousticDim + HiggsSemanticPostEncoder.HiddenDim; // 256+768=1024, matches Fc's real input width
        for (int f = 0; f < targetFrames; f++)
        {
            var concat = new float[concatWidth];
            for (int c = 0; c < acousticDim; c++) concat[c] = acousticLatent[c * targetFrames + f];
            for (int c = 0; c < HiggsSemanticPostEncoder.HiddenDim; c++) concat[acousticDim + c] = semanticPost[f][c];

            var hidden = LinearBias(concat, codecWeights.FcWeight, codecWeights.FcBias, concatWidth, OmniVoiceAcousticDecoderWeights.QuantizerLatentDim);
            var frameCodes = QuantizeFrame(codecWeights, hidden);
            for (int cb = 0; cb < OmniVoiceAcousticDecoderWeights.NumCodebooks; cb++) codes[cb][f] = frameCodes[cb];
        }
        return codes;
    }

    /// <summary>Real RVQ encode loop, identical math to <see cref="HiggsCodecEncoder.QuantizeFrame"/>
    /// (see that method's doc comment for the derivation), against OmniVoice's own weights.</summary>
    private static int[] QuantizeFrame(OmniVoiceAcousticDecoderWeights w, float[] hiddenFrame)
    {
        var codes = new int[OmniVoiceAcousticDecoderWeights.NumCodebooks];
        var residual = (float[])hiddenFrame.Clone();

        for (int cb = 0; cb < OmniVoiceAcousticDecoderWeights.NumCodebooks; cb++)
        {
            var q = w.Quantizers[cb];
            var projected = LinearBias(residual, q.ProjectInWeight, q.ProjectInBias,
                OmniVoiceAcousticDecoderWeights.QuantizerLatentDim, OmniVoiceAcousticDecoderWeights.CodebookDim);

            double xNormSq = 0;
            for (int d = 0; d < projected.Length; d++) xNormSq += (double)projected[d] * projected[d];

            int best = 0;
            double bestScore = double.NegativeInfinity;
            for (int e = 0; e < OmniVoiceAcousticDecoderWeights.CodebookSize; e++)
            {
                int eBase = e * OmniVoiceAcousticDecoderWeights.CodebookDim;
                double dot = 0, eNormSq = 0;
                for (int d = 0; d < OmniVoiceAcousticDecoderWeights.CodebookDim; d++)
                {
                    float ev = q.CodebookEmbed[eBase + d];
                    dot += (double)projected[d] * ev;
                    eNormSq += (double)ev * ev;
                }
                double score = 2.0 * dot - xNormSq - eNormSq;
                if (score > bestScore) { bestScore = score; best = e; }
            }
            codes[cb] = best;

            var quantizedLatent = new float[OmniVoiceAcousticDecoderWeights.CodebookDim];
            Array.Copy(q.CodebookEmbed, best * OmniVoiceAcousticDecoderWeights.CodebookDim, quantizedLatent, 0, OmniVoiceAcousticDecoderWeights.CodebookDim);
            var quantized = LinearBias(quantizedLatent, q.ProjectOutWeight, q.ProjectOutBias,
                OmniVoiceAcousticDecoderWeights.CodebookDim, OmniVoiceAcousticDecoderWeights.QuantizerLatentDim);

            for (int d = 0; d < residual.Length; d++) residual[d] -= quantized[d];
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
