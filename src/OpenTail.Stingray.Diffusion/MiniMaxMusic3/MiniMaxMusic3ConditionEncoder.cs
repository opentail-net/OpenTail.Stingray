namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Real MiniMax Music 3 condition encoder (`MiniMaxMusic3ConditionEncoder`), transcribed directly
/// from the real, already-installed `diffusers==0.40.0` source
/// (`diffusers/models/condition_embedders/condition_embedder_minimax_music3.py`) -- see
/// docs/066-minimax-music3-future-plan.md.
///
/// <para>Tiny, real formula: each generated frame carries `num_condition_layers`(8) hidden states
/// of size `condition_hidden_dim`(4096) -- one from the global language model plus one per residual
/// codebook step (real, `real per-frame concatenation` this project's earlier "hidden-state fusion"
/// finding already resolved, see `MiniMaxMusic3RvqDepthDecoder`'s doc comment). This class mixes
/// those 8 layers with LEARNED softmax weights (`layer_weight_logits`, ELMo-style), scales
/// (`layer_scale`), projects `condition_hidden_dim -&gt; out_dim` via a real `Conv1d(k=3,pad=1)`,
/// then nearest-neighbor-resamples from the language model's real frame rate (25Hz, `24000/960`)
/// to the real Flow-VAE latent frame rate (`44100/512 ≈ 86.13Hz`) -- real ratio
/// `≈3.4453125`.</para>
/// </summary>
public sealed class MiniMaxMusic3ConditionEncoderWeights
{
    public required float[] LayerWeightLogits { get; init; } // [numConditionLayers(8)]
    public required float LayerScale { get; init; }
    public required float[] ProjWeight { get; init; } // [outDim(2048), condHiddenDim(4096), kernel(3)]
    public required float[] ProjBias { get; init; }

    public static MiniMaxMusic3ConditionEncoderWeights Load(SafetensorsLoader loader) => new()
    {
        LayerWeightLogits = loader.ReadF32("layer_weight_logits"),
        LayerScale = loader.ReadF32("layer_scale")[0],
        ProjWeight = loader.ReadF32("proj.weight"),
        ProjBias = loader.ReadF32("proj.bias"),
    };
}

public static class MiniMaxMusic3ConditionEncoder
{
    /// <summary>Real forward: `hiddenStates[frame][layer][condHiddenDim]` (already split per real
    /// condition layer, matching the real reference's own `reshape` of the concatenated
    /// `[frames, numLayers*condHiddenDim]` input) -&gt; softmax-weighted layer mix -&gt; scale -&gt;
    /// `Conv1d(k=3,pad=1)` projection -&gt; nearest-neighbor resample to the real Flow-VAE latent
    /// frame count. Returns `[latentLength][outDim]`.</summary>
    public static float[][] Forward(MiniMaxMusic3ConditionEncoderWeights w, float[][][] hiddenStates)
    {
        int numFrames = hiddenStates.Length;
        int numLayers = MiniMaxMusic3Config.ConditionEncoderNumLayers;
        int condHiddenDim = MiniMaxMusic3Config.ConditionEncoderHiddenDim;
        int outDim = MiniMaxMusic3Config.ConditionEncoderOutDim;

        var layerWeights = Softmax(w.LayerWeightLogits);

        // Real: mix the 8 condition layers per frame with the learned softmax weights, then scale.
        var mixed = new float[numFrames][];
        for (int f = 0; f < numFrames; f++)
        {
            var row = new float[condHiddenDim];
            for (int l = 0; l < numLayers; l++)
            {
                float lw = layerWeights[l];
                var src = hiddenStates[f][l];
                for (int c = 0; c < condHiddenDim; c++) row[c] += lw * src[c];
            }
            for (int c = 0; c < condHiddenDim; c++) row[c] *= w.LayerScale;
            mixed[f] = row;
        }

        // Real Conv1d(k=3, pad=1) projection, condHiddenDim -> outDim, applied along the frame axis.
        var projected = new float[numFrames][];
        for (int f = 0; f < numFrames; f++)
        {
            var row = new float[outDim];
            for (int oc = 0; oc < outDim; oc++)
            {
                float acc = w.ProjBias[oc];
                for (int k = 0; k < 3; k++)
                {
                    int srcF = f + k - 1; // pad=1
                    if (srcF < 0 || srcF >= numFrames) continue;
                    int wBase = oc * condHiddenDim * 3;
                    var srcRow = mixed[srcF];
                    for (int ic = 0; ic < condHiddenDim; ic++)
                        acc += w.ProjWeight[wBase + ic * 3 + k] * srcRow[ic];
                }
                row[oc] = acc;
            }
            projected[f] = row;
        }

        // Real nearest-neighbor resample: language-model frame rate -> Flow-VAE latent frame rate.
        int latentLength = Math.Max(1, (int)(
            numFrames
            * (double)MiniMaxMusic3Config.ConditionEncoderOutputSampleRate / MiniMaxMusic3Config.ConditionEncoderInputSampleRate
            * MiniMaxMusic3Config.ConditionEncoderInputHopLength / MiniMaxMusic3Config.ConditionEncoderOutputHopLength));

        var output = new float[latentLength][];
        for (int t = 0; t < latentLength; t++)
        {
            // Real `F.interpolate(..., mode="nearest")`: source index = floor(t * numFrames / latentLength).
            int srcF = Math.Min(numFrames - 1, (int)((long)t * numFrames / latentLength));
            output[t] = projected[srcF];
        }
        return output;
    }

    private static float[] Softmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        foreach (var v in logits) if (v > max) max = v;
        var exp = new float[logits.Length];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++) { exp[i] = MathF.Exp(logits[i] - max); sum += exp[i]; }
        for (int i = 0; i < exp.Length; i++) exp[i] /= sum;
        return exp;
    }
}
