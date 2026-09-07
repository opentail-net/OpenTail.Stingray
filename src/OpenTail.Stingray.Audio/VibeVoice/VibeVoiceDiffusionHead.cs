using OpenTail.Stingray.Audio.Primitives;

namespace OpenTail.Stingray.Audio.VibeVoice;

/// <summary>Real per-layer weights for VibeVoice TTS's diffusion prediction head, ported from
/// `diffusion_head.cpp`'s `load_layer_weights` (not guessed): a SwiGLU FFN block with AdaLN-Zero
/// modulation (`adaLN_modulation.1.weight`, no bias -- real reference passes `SiLU(condition)`
/// through a single Linear producing `[shift,scale,gate]` concatenated on the last axis, each
/// `hiddenSize` wide).</summary>
public sealed class VibeVoiceDiffusionHeadLayerWeights
{
    public required float[] NormWeight { get; init; }
    public required float[] AdaLnWeight { get; init; } // [3*hidden, hidden], no bias
    public required float[] GateProjWeight { get; init; } // [ffn, hidden]
    public required float[] UpProjWeight { get; init; } // [ffn, hidden]
    public required float[] DownProjWeight { get; init; } // [hidden, ffn]
}

/// <summary>Real weights for the whole diffusion head, ported from `diffusion_head.cpp`'s
/// `load_vibevoice_diffusion_head_weights` (not guessed). No biases anywhere in this module --
/// every Linear is bias-free, matching the real reference's `linear_config(..., false)` calls
/// throughout.</summary>
public sealed class VibeVoiceDiffusionHeadWeights
{
    public required int HiddenSize { get; init; }
    public required int LatentSize { get; init; }
    public required int FfnDim { get; init; }
    public required float RmsNormEps { get; init; }

    public required float[] NoisyImagesProjWeight { get; init; } // [hidden, latent]
    public required float[] CondProjWeight { get; init; } // [hidden, hidden]
    public required float[] TimestepFc1Weight { get; init; } // [hidden, 256]
    public required float[] TimestepFc2Weight { get; init; } // [hidden, hidden]
    public required VibeVoiceDiffusionHeadLayerWeights[] Layers { get; init; }
    public required float[] FinalAdaLnWeight { get; init; } // [2*hidden, hidden], no bias
    public required float[] FinalLinearWeight { get; init; } // [latent, hidden]

    public static VibeVoiceDiffusionHeadWeights Load(int hiddenSize, int latentSize, int headLayers,
        float headFfnRatio, float rmsNormEps, Func<string, float[]> get)
    {
        int ffnDim = (int)(hiddenSize * (double)headFfnRatio);
        var layers = new VibeVoiceDiffusionHeadLayerWeights[headLayers];
        for (int i = 0; i < headLayers; i++)
        {
            string p = $"model.prediction_head.layers.{i}.";
            layers[i] = new VibeVoiceDiffusionHeadLayerWeights
            {
                NormWeight = get(p + "norm.weight"),
                AdaLnWeight = get(p + "adaLN_modulation.1.weight"),
                GateProjWeight = get(p + "ffn.gate_proj.weight"),
                UpProjWeight = get(p + "ffn.up_proj.weight"),
                DownProjWeight = get(p + "ffn.down_proj.weight"),
            };
        }

        return new VibeVoiceDiffusionHeadWeights
        {
            HiddenSize = hiddenSize,
            LatentSize = latentSize,
            FfnDim = ffnDim,
            RmsNormEps = rmsNormEps,
            NoisyImagesProjWeight = get("model.prediction_head.noisy_images_proj.weight"),
            CondProjWeight = get("model.prediction_head.cond_proj.weight"),
            TimestepFc1Weight = get("model.prediction_head.t_embedder.mlp.0.weight"),
            TimestepFc2Weight = get("model.prediction_head.t_embedder.mlp.2.weight"),
            Layers = layers,
            FinalAdaLnWeight = get("model.prediction_head.final_layer.adaLN_modulation.1.weight"),
            FinalLinearWeight = get("model.prediction_head.final_layer.linear.weight"),
        };
    }
}

/// <summary>
/// Real forward pass for VibeVoice TTS's diffusion prediction head, ported from
/// `diffusion_head.cpp`'s `build_vibevoice_diffusion_head` (not guessed). A small, attention-
/// free DiT: per-frame-independent AdaLN-Zero-modulated SwiGLU blocks conditioned on
/// `cond_proj(condition) + timestep_embedding(t)` (real sinusoidal timestep embedding, 256-wide,
/// `exp(-log(10000)*i/128)` frequencies, through a 2-layer SiLU MLP). Each block: `RMSNorm(no
/// affine) -> modulate(shift,scale) -> SwiGLU -> x + gate*out`. Final layer: `RMSNorm(no affine)
/// -> modulate(shift,scale from a 2-way [not 3-way] AdaLN) -> Linear(hidden->latent)`.
/// </summary>
public static class VibeVoiceDiffusionHead
{
    private const int TimestepFreqEmbedSize = 256;
    private const float TimestepMaxPeriod = 10000.0f;

    /// <summary>Runs the head for `frames` independent rows: `noisy[frames][latent]`,
    /// `condition[frames][hidden]`, single shared scalar `timestep`. Returns `[frames][latent]`
    /// predicted v (v-prediction target, per the scheduler's `prediction_type=v_prediction`).</summary>
    public static float[][] Predict(VibeVoiceDiffusionHeadWeights w, float[][] noisy, float[][] condition, float timestep)
    {
        int frames = noisy.Length;
        var timestepEmbedding = TimestepEmbedding(w, timestep);

        var output = new float[frames][];
        for (int t = 0; t < frames; t++)
        {
            var x = DenseKernels.LinearNoBias(noisy[t], w.NoisyImagesProjWeight, w.LatentSize, w.HiddenSize);
            var projectedCondition = DenseKernels.LinearNoBias(condition[t], w.CondProjWeight, w.HiddenSize, w.HiddenSize);
            var c = Add(projectedCondition, timestepEmbedding);

            foreach (var layer in w.Layers) x = HeadLayer(w, layer, x, c);

            output[t] = FinalLayer(w, x, c);
        }
        return output;
    }

    private static float[] TimestepEmbedding(VibeVoiceDiffusionHeadWeights w, float timestep)
    {
        int half = TimestepFreqEmbedSize / 2;
        var embedding = new float[TimestepFreqEmbedSize];
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(TimestepMaxPeriod) * i / half);
            float arg = timestep * freq;
            embedding[i] = MathF.Cos(arg);
            embedding[half + i] = MathF.Sin(arg);
        }
        var hidden = DenseKernels.LinearNoBias(embedding, w.TimestepFc1Weight, TimestepFreqEmbedSize, w.HiddenSize);
        DenseKernels.SiluInPlace(hidden);
        return DenseKernels.LinearNoBias(hidden, w.TimestepFc2Weight, w.HiddenSize, w.HiddenSize);
    }

    private static float[] HeadLayer(VibeVoiceDiffusionHeadWeights w, VibeVoiceDiffusionHeadLayerWeights layer, float[] x, float[] c)
    {
        int hidden = w.HiddenSize;
        var siluC = (float[])c.Clone();
        DenseKernels.SiluInPlace(siluC);
        var modulation = DenseKernels.LinearNoBias(siluC, layer.AdaLnWeight, hidden, 3 * hidden);
        var shift = modulation.AsSpan(0, hidden).ToArray();
        var scale = modulation.AsSpan(hidden, hidden).ToArray();
        var gate = modulation.AsSpan(2 * hidden, hidden).ToArray();

        var normed = RmsNormNoAffine(x, w.RmsNormEps, layer.NormWeight);
        var modulated = Modulate(normed, shift, scale);
        var ffnOut = SwiGlu(layer, modulated, w.FfnDim, hidden);

        var result = new float[hidden];
        for (int i = 0; i < hidden; i++) result[i] = x[i] + gate[i] * ffnOut[i];
        return result;
    }

    private static float[] FinalLayer(VibeVoiceDiffusionHeadWeights w, float[] x, float[] c)
    {
        int hidden = w.HiddenSize;
        var siluC = (float[])c.Clone();
        DenseKernels.SiluInPlace(siluC);
        var modulation = DenseKernels.LinearNoBias(siluC, w.FinalAdaLnWeight, hidden, 2 * hidden);
        var shift = modulation.AsSpan(0, hidden).ToArray();
        var scale = modulation.AsSpan(hidden, hidden).ToArray();

        // Final RMSNorm has NO learnable scale (real reference: norm_data(nullopt,nullopt)).
        var normed = RmsNormUnweighted(x, w.RmsNormEps);
        var modulated = Modulate(normed, shift, scale);
        return DenseKernels.LinearNoBias(modulated, w.FinalLinearWeight, hidden, w.LatentSize);
    }

    /// <summary>Real `modulate(x, shift, scale) = x * (1 + scale) + shift`.</summary>
    private static float[] Modulate(float[] x, float[] shift, float[] scale)
    {
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * (1f + scale[i]) + shift[i];
        return output;
    }

    private static float[] SwiGlu(VibeVoiceDiffusionHeadLayerWeights layer, float[] input, int ffnDim, int hidden)
    {
        var gate = DenseKernels.LinearNoBias(input, layer.GateProjWeight, hidden, ffnDim);
        DenseKernels.SiluInPlace(gate);
        var up = DenseKernels.LinearNoBias(input, layer.UpProjWeight, hidden, ffnDim);
        for (int i = 0; i < gate.Length; i++) gate[i] *= up[i];
        return DenseKernels.LinearNoBias(gate, layer.DownProjWeight, ffnDim, hidden);
    }

    private static float[] RmsNormNoAffine(float[] x, float eps, float[] weight)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + eps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms * weight[i];
        return output;
    }

    private static float[] RmsNormUnweighted(float[] x, float eps)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + eps));
        var output = new float[x.Length];
        for (int i = 0; i < x.Length; i++) output[i] = x[i] * invRms;
        return output;
    }

    private static float[] Add(float[] a, float[] b)
    {
        var output = new float[a.Length];
        for (int i = 0; i < a.Length; i++) output[i] = a[i] + b[i];
        return output;
    }
}
