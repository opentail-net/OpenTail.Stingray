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
        => PredictProjected(w, noisy, ProjectCondition(w, condition), timestep);

    /// <summary>`cond_proj(condition)` for all rows, flat `[rows, hidden]`. It doesn't depend on the
    /// timestep, so a sampler computes it once per frame, not once per solver step.</summary>
    public static float[] ProjectCondition(VibeVoiceDiffusionHeadWeights w, float[][] condition)
    {
        int m = condition.Length, h = w.HiddenSize;
        var projected = new float[m * h];
        DenseKernels.LinearBatchedNoBias(Flatten(condition, h), w.CondProjWeight, projected, m, h, h);
        return projected;
    }

    /// <summary>
    /// Same math as the per-row version, with all rows (the CFG cond/uncond pair) batched through every
    /// Linear so each weight matrix is read once per solver step instead of once per row
    /// (<see cref="DenseKernels.LinearBatchedNoBias"/>; 2026-09-25 perf pass).
    /// </summary>
    public static float[][] PredictProjected(VibeVoiceDiffusionHeadWeights w, float[][] noisy, float[] projectedCondition, float timestep)
    {
        int m = noisy.Length, h = w.HiddenSize, f = w.FfnDim, lat = w.LatentSize;
        var timestepEmbedding = TimestepEmbedding(w, timestep);

        // c = cond_proj(condition) + t_emb; every AdaLN uses silu(c), which is the same for all layers.
        var siluC = new float[m * h];
        for (int r = 0; r < m; r++)
        {
            var row = siluC.AsSpan(r * h, h);
            System.Numerics.Tensors.TensorPrimitives.Add(projectedCondition.AsSpan(r * h, h), timestepEmbedding, row);
        }
        DenseKernels.SiluInPlace(siluC);

        var x = new float[m * h];
        DenseKernels.LinearBatchedNoBias(Flatten(noisy, lat), w.NoisyImagesProjWeight, x, m, lat, h);

        var modulation = new float[m * 3 * h];
        var modulated = new float[m * h];
        var gate = new float[m * f];
        var up = new float[m * f];
        var ffnOut = new float[m * h];
        foreach (var layer in w.Layers)
        {
            DenseKernels.LinearBatchedNoBias(siluC, layer.AdaLnWeight, modulation, m, h, 3 * h);
            for (int r = 0; r < m; r++)
            {
                var mod = modulation.AsSpan(r * 3 * h, 3 * h);
                RmsNormModulateRow(x.AsSpan(r * h, h), layer.NormWeight, mod[..h], mod.Slice(h, h), w.RmsNormEps, modulated.AsSpan(r * h, h));
            }
            DenseKernels.LinearBatchedNoBias(modulated, layer.GateProjWeight, gate, m, h, f);
            DenseKernels.SiluInPlace(gate);
            DenseKernels.LinearBatchedNoBias(modulated, layer.UpProjWeight, up, m, h, f);
            System.Numerics.Tensors.TensorPrimitives.Multiply(gate, up, gate);
            DenseKernels.LinearBatchedNoBias(gate, layer.DownProjWeight, ffnOut, m, f, h);
            for (int r = 0; r < m; r++)
            {
                var xr = x.AsSpan(r * h, h);
                var g = modulation.AsSpan(r * 3 * h + 2 * h, h);
                for (int i = 0; i < h; i++) xr[i] += g[i] * ffnOut[r * h + i];
            }
        }

        // Final layer: 2-way AdaLN, RMSNorm with NO learnable scale, Linear(hidden -> latent).
        var finalMod = new float[m * 2 * h];
        DenseKernels.LinearBatchedNoBias(siluC, w.FinalAdaLnWeight, finalMod, m, h, 2 * h);
        for (int r = 0; r < m; r++)
        {
            var mod = finalMod.AsSpan(r * 2 * h, 2 * h);
            RmsNormModulateRow(x.AsSpan(r * h, h), null, mod[..h], mod.Slice(h, h), w.RmsNormEps, modulated.AsSpan(r * h, h));
        }
        var outFlat = new float[m * lat];
        DenseKernels.LinearBatchedNoBias(modulated, w.FinalLinearWeight, outFlat, m, h, lat);

        var output = new float[m][];
        for (int r = 0; r < m; r++) output[r] = outFlat.AsSpan(r * lat, lat).ToArray();
        return output;
    }

    /// <summary>`modulate(RMSNorm(x) [* weight], shift, scale) = n * (1 + scale) + shift`, with the
    /// RMS statistic accumulated in double exactly as the per-row helpers do.</summary>
    private static void RmsNormModulateRow(ReadOnlySpan<float> x, float[]? weight, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale, float eps, Span<float> output)
    {
        double sumSq = 0;
        for (int i = 0; i < x.Length; i++) sumSq += (double)x[i] * x[i];
        float invRms = (float)(1.0 / Math.Sqrt(sumSq / x.Length + eps));
        for (int i = 0; i < x.Length; i++)
        {
            float n = weight is null ? x[i] * invRms : x[i] * invRms * weight[i];
            output[i] = n * (1f + scale[i]) + shift[i];
        }
    }

    private static float[] Flatten(float[][] rows, int dim)
    {
        var flat = new float[rows.Length * dim];
        for (int r = 0; r < rows.Length; r++) rows[r].AsSpan(0, dim).CopyTo(flat.AsSpan(r * dim, dim));
        return flat;
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

}
