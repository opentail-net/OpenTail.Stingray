using System.Numerics.Tensors;
using System.Text.Json;
using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Engine.Encoders;

/// <summary>
/// Chronos-Bolt time-series forecaster (amazon/chronos-bolt-*), ported from the vendored reference
/// <c>examples/chronos-forecasting/src/chronos/chronos_bolt.py</c>: instance-normalize the context (nan-aware mean/std,
/// zero scale → 1e-5), left-pad with NaN to whole patches, patch (16 values + their observed mask → 32 features),
/// <c>input_patch_embedding</c> (ResidualBlock), append the [REG] embedding, run the <see cref="T5Model"/> encoder with the
/// patch attention mask, decode ONE step from <c>decoder_start_token_id</c>, map the decoder output through
/// <c>output_patch_embedding</c> to [quantiles × prediction_length] and un-normalize. Horizons beyond the model's
/// prediction length use the reference's heuristic (every quantile path fed back, empirical quantiles of the Q² paths).
/// </summary>
public sealed class ChronosBoltModel : IDisposable
{
    private readonly T5Model _t5;
    private readonly ChronosResidualBlock _inPatch, _outPatch;
    private readonly float[] _regEmbedding; // shared[reg_token_id], or empty when use_reg_token is false

    public int ContextLength { get; }
    public int PredictionLength { get; }
    public int PatchSize { get; }
    public float[] Quantiles { get; }

    private ChronosBoltModel(T5Model t5, ChronosResidualBlock inPatch, ChronosResidualBlock outPatch, float[] reg, int ctx, int pred, int patch, float[] q)
    {
        (_t5, _inPatch, _outPatch, _regEmbedding) = (t5, inPatch, outPatch, reg);
        (ContextLength, PredictionLength, PatchSize, Quantiles) = (ctx, pred, patch, q);
    }

    public static ChronosBoltModel Load(string dir)
    {
        string configPath = Path.Combine(dir, "config.json");
        using var doc = JsonDocument.Parse(File.ReadAllBytes(configPath));
        var root = doc.RootElement;
        if (!root.TryGetProperty("chronos_config", out var cc) || !root.TryGetProperty("architectures", out var arch)
            || arch[0].GetString() != "ChronosBoltModelForForecasting")
            throw new NotSupportedException($"{configPath}: not a ChronosBoltModelForForecasting checkpoint.");
        string act = root.TryGetProperty("dense_act_fn", out var a) ? a.GetString() ?? "relu" : "relu";
        if (act != "relu") throw new NotSupportedException($"Chronos-Bolt dense_act_fn '{act}' not supported (relu).");
        int ctx = cc.GetProperty("context_length").GetInt32(), pred = cc.GetProperty("prediction_length").GetInt32();
        int patch = cc.GetProperty("input_patch_size").GetInt32(), stride = cc.GetProperty("input_patch_stride").GetInt32();
        if (stride != patch) throw new NotSupportedException("Chronos-Bolt: only input_patch_stride == input_patch_size is supported.");
        bool reg = cc.TryGetProperty("use_reg_token", out var r) && r.ValueKind == JsonValueKind.True;
        var quantiles = cc.GetProperty("quantiles").EnumerateArray().Select(x => x.GetSingle()).ToArray();

        var c = T5Config.FromJson(configPath);
        using var st = SafetensorsLoader.OpenDirectory(dir);
        var t5 = T5Model.Load(c, st, loadLmHead: false);
        float[] shared = st.ReadF32("shared.weight");
        float[] regEmb = reg ? shared.AsSpan(1 * c.DModel, c.DModel).ToArray() : [];
        return new ChronosBoltModel(t5,
            ChronosResidualBlock.Load(st, "input_patch_embedding", 2 * patch, c.DFf, c.DModel),
            ChronosResidualBlock.Load(st, "output_patch_embedding", c.DModel, c.DFf, quantiles.Length * pred),
            regEmb, ctx, pred, patch, quantiles);
    }

    /// <summary>One model call: quantile forecasts [Quantiles.Length, PredictionLength] (row-major, quantile-major) for
    /// <paramref name="context"/> (NaN = missing; only the last <see cref="ContextLength"/> values are used).</summary>
    public float[] Forward(ReadOnlySpan<float> context)
    {
        if (context.Length > ContextLength) context = context[^ContextLength..];
        int d = _t5.Config.DModel, n = context.Length;

        var norm = ChronosInstanceNorm.Fit(context, arcsinh: false);

        // Patch: left-pad with NaN (context) / NaN→0 (mask) to a multiple of PatchSize.
        int pad = n % PatchSize == 0 ? 0 : PatchSize - n % PatchSize;
        int patches = (n + pad) / PatchSize, p = PatchSize;
        var features = new float[patches * 2 * p];
        var keyMask = new bool[patches + (_regEmbedding.Length > 0 ? 1 : 0)];
        for (int i = 0; i < patches; i++)
        {
            float observed = 0;
            for (int j = 0; j < p; j++)
            {
                int src = i * p + j - pad;
                float v = src < 0 ? float.NaN : context[src];
                bool ok = !float.IsNaN(v);
                features[i * 2 * p + j] = ok ? norm.Apply(v) : 0f;
                features[i * 2 * p + p + j] = ok ? 1f : 0f;
                observed += ok ? 1f : 0f;
            }
            keyMask[i] = observed > 0;
        }

        var embeds = _inPatch.Forward(features, patches);
        int len = patches;
        if (_regEmbedding.Length > 0)
        {
            embeds = [.. embeds, .. _regEmbedding];
            keyMask[len++] = true;
        }

        var hidden = _t5.EncodeEmbeddings(embeds, len, keyMask);
        var dec = _t5.StartDecoder(hidden, len, 1, keyMask).Step([_t5.Config.DecoderStartTokenId]);
        var q = _outPatch.Forward(dec, 1);
        norm.InverseInPlace(q);
        return q;
    }

    /// <summary>Quantile forecasts [Quantiles.Length, <paramref name="predictionLength"/>] (as <c>ChronosBoltPipeline.predict</c>).</summary>
    public float[] Predict(float[] context, int predictionLength)
    {
        int nq = Quantiles.Length, P = PredictionLength;
        var first = Forward(context);
        var result = new float[nq * predictionLength];
        int filled = Math.Min(P, predictionLength);
        for (int k = 0; k < nq; k++) Array.Copy(first, k * P, result, k * predictionLength, filled);

        // Long horizon: extend every quantile path with its own prediction, forecast again, take the empirical
        // quantiles (torch.quantile, linear interpolation) of the nq·nq resulting paths.
        var paths = Enumerable.Range(0, nq).Select(k => (float[])[.. context, .. first.AsSpan(k * P, P)]).ToArray();
        while (filled < predictionLength)
        {
            var samples = new float[nq * nq][];
            Parallel.For(0, nq, k =>
            {
                var f = Forward(paths[k]);
                for (int j = 0; j < nq; j++) samples[k * nq + j] = f.AsSpan(j * P, P).ToArray();
            });
            var next = new float[nq * P];
            var column = new float[nq * nq];
            for (int t = 0; t < P; t++)
            {
                for (int s = 0; s < column.Length; s++) column[s] = samples[s][t];
                Array.Sort(column);
                for (int k = 0; k < nq; k++)
                {
                    float pos = Quantiles[k] * (column.Length - 1);
                    int lo = (int)MathF.Floor(pos), hi = Math.Min(lo + 1, column.Length - 1);
                    next[k * P + t] = column[lo] + (pos - lo) * (column[hi] - column[lo]);
                }
            }
            int take = Math.Min(P, predictionLength - filled);
            for (int k = 0; k < nq; k++)
            {
                Array.Copy(next, k * P, result, k * predictionLength + filled, take);
                paths[k] = [.. paths[k], .. next.AsSpan(k * P, P)];
            }
            filled += take;
        }
        return result;
    }

    public void Dispose()
    {
        _t5.Dispose();
        _inPatch.Dispose();
        _outPatch.Dispose();
    }
}
