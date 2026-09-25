
namespace OpenTail.Stingray.Audio.Rvc;

/// <summary>
/// Real HuBERT-base (Wav2Vec2-family) content encoder forward pass for RVC, transcribed directly
/// from `examples/audio.cpp/src/framework/modules/speech_encoders/hubert_encoder.cpp`'s real
/// `build_hubert_graph` (not guessed): 7-stage valid (no-padding) conv1d feature extractor with
/// GroupNorm only after layer 0 (`FirstLayerGroupNorm`) and exact-erf GELU after every layer ->
/// LayerNorm + Linear feature projection (512-&gt;768) -> real weight-normalized grouped
/// positional conv (16 groups, kernel 128, added as a residual, NOT gated) -> encoder input
/// LayerNorm -> 12 standard post-LayerNorm (`attn -&gt; +residual -&gt; self_attn_layer_norm -&gt;
/// ffn -&gt; +residual -&gt; final_layer_norm`) transformer blocks, non-causal (bidirectional)
/// self-attention.
/// </summary>
public static class RvcHubertEncoder
{
    /// <summary>
    /// Runs the encoder on a 16kHz mono waveform. Returns frame-major [numFrames, 768] hidden
    /// states after all 12 layers (the real reference's `output_hidden_layer = 12` case, i.e.
    /// `v1_features=false` -- RVC v2's content features).
    /// </summary>
    public static float[][] Forward(RvcHubertWeights w, ReadOnlySpan<float> waveform16k)
    {
        // 1. Feature extractor: 7 valid (no-padding) conv1d layers, GroupNorm only after layer 0,
        // exact-erf GELU after every layer.
        //    Convs run as im2col + packed GEMM (shared Wav2Vec2FrontendKernels, 2026-09-25; they were scalar loops).
        float[] flat = waveform16k.ToArray();
        int frames = flat.Length, channels = 1;
        for (int layerIdx = 0; layerIdx < 7; layerIdx++)
        {
            flat = Primitives.Wav2Vec2FrontendKernels.Conv1dValid(flat, frames, channels, RvcHubertWeights.ConvKernel[layerIdx],
                RvcHubertWeights.ConvStride[layerIdx], w.PackedConv[layerIdx], out frames);
            channels = RvcHubertWeights.ConvDim[layerIdx];
            if (layerIdx == 0)
                Primitives.Wav2Vec2FrontendKernels.GroupNormPerChannel(flat, frames, channels, w.ConvLayer0GroupNormWeight, w.ConvLayer0GroupNormBias);
            Parallel.For(0, frames, i => { var row = flat.AsSpan(i * channels, channels); for (int c = 0; c < row.Length; c++) row[c] = GeluErf(row[c]); });
        }
        float[][] x = new float[frames][];
        for (int i = 0; i < frames; i++) x[i] = flat.AsSpan(i * channels, channels).ToArray();

        // 2. Feature projection: LayerNorm(512) + Linear(512->768).
        int t = x.Length;
        for (int i = 0; i < t; i++)
            x[i] = LayerNorm(x[i], w.LayerNormWeight, w.LayerNormBias);
        var hidden = new float[t][];
        for (int i = 0; i < t; i++)
            hidden[i] = LinearBias(x[i], w.PostExtractProjWeight, w.PostExtractProjBias, inDim: 512, outDim: RvcHubertWeights.HiddenDim);

        // 3. Positional conv: grouped conv1d (16 groups, kernel 128, pad=64 each side, output
        // trimmed by 1 trailing frame since kernel is even), exact-erf GELU, added as a residual.
        int hd = RvcHubertWeights.HiddenDim;
        var hiddenFlat = new float[t * hd];
        for (int i = 0; i < t; i++) hidden[i].CopyTo(hiddenFlat, i * hd);
        var posFlat = Primitives.Wav2Vec2FrontendKernels.GroupedConvSamePad(hiddenFlat, t, hd, w.PackedPosConv, w.PosConvBias, RvcHubertWeights.ConvPosKernel);
        var pos = new float[t][];
        for (int i = 0; i < t; i++) pos[i] = posFlat.AsSpan(i * hd, hd).ToArray();
        GeluErfInPlaceRows(pos);
        for (int i = 0; i < t; i++)
            for (int d = 0; d < RvcHubertWeights.HiddenDim; d++)
                hidden[i][d] += pos[i][d];

        // 4. Encoder input LayerNorm.
        for (int i = 0; i < t; i++)
            hidden[i] = LayerNorm(hidden[i], w.EncoderLayerNormWeight, w.EncoderLayerNormBias);

        // Real reference (`pad_odd_tokens_with_attention_mask`) pads an odd token count by one
        // zero frame, masked out of every position's attention (score -1e4 added for the padded
        // key) so it can't influence any real position's output, then trims it back off before
        // returning.
        int rawTokens = t;
        bool padded = t % 2 != 0;
        if (padded)
        {
            var withPad = new float[t + 1][];
            Array.Copy(hidden, withPad, t);
            withPad[t] = new float[RvcHubertWeights.HiddenDim];
            hidden = withPad;
            t++;
        }

        // 5. 12 standard post-LayerNorm transformer blocks, non-causal self-attention.
        foreach (var layer in w.Layers)
            hidden = EncoderBlock(hidden, layer, t, RvcHubertWeights.HiddenDim, RvcHubertWeights.NumHeads, maskLastKey: padded);

        if (padded)
        {
            var trimmed = new float[rawTokens][];
            Array.Copy(hidden, trimmed, rawTokens);
            hidden = trimmed;
        }

        return hidden;
    }




    private static void GeluErfInPlaceRows(float[][] x)
    {
        foreach (var row in x)
            for (int i = 0; i < row.Length; i++)
                row[i] = GeluErf(row[i]);
    }

    private static float GeluErf(float x) => (float)(0.5 * x * (1.0 + Erf(x / Math.Sqrt(2.0))));

    // Abramowitz-Stegun erf approximation (matches project convention where a real math-library
    // erf isn't already available -- max error ~1.5e-7, ample for this use).
    private static double Erf(double x)
    {
        double sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        const double a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
        return sign * y;
    }


    private static float[][] EncoderBlock(float[][] x, RvcHubertLayerWeights l, int t, int dim, int heads, bool maskLastKey)
    {
        var attnOut = SelfAttention(x, l, t, dim, heads, maskLastKey);
        var afterAttn = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = x[i][d] + attnOut[i][d];
            afterAttn[i] = LayerNorm(row, l.SelfAttnLayerNormWeight, l.SelfAttnLayerNormBias);
        }

        // FFN as two batched GEMMs over all frames (was a mat-vec per frame; 2026-09-25 perf pass).
        int ffnDim = l.Fc1Bias.Length;
        var hidden = OpenTail.Stingray.Audio.Primitives.DenseKernels.LinearBatchedRows(afterAttn, l.Fc1Weight, l.Fc1Bias, dim, ffnDim);
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            var h = hidden[i];
            for (int d = 0; d < h.Length; d++) h[d] = GeluErf(h[d]);
        });
        var down = OpenTail.Stingray.Audio.Primitives.DenseKernels.LinearBatchedRows(hidden, l.Fc2Weight, l.Fc2Bias, ffnDim, dim);
        var output = new float[t][];
        System.Threading.Tasks.Parallel.For(0, t, i =>
        {
            var row = new float[dim];
            for (int d = 0; d < dim; d++) row[d] = afterAttn[i][d] + down[i][d];
            output[i] = LayerNorm(row, l.FinalLayerNormWeight, l.FinalLayerNormBias);
        });
        return output;
    }

    private static float[][] SelfAttention(float[][] x, RvcHubertLayerWeights l, int t, int dim, int heads, bool maskLastKey)
    {
        int headDim = dim / heads;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = OpenTail.Stingray.Audio.Primitives.DenseKernels.LinearBatchedRows(x, l.AttnQWeight, l.AttnQBias, dim, dim);
        var k = OpenTail.Stingray.Audio.Primitives.DenseKernels.LinearBatchedRows(x, l.AttnKWeight, l.AttnKBias, dim, dim);
        var v = OpenTail.Stingray.Audio.Primitives.DenseKernels.LinearBatchedRows(x, l.AttnVWeight, l.AttnVBias, dim, dim);

        var context = new float[t][];
        for (int i = 0; i < t; i++) context[i] = new float[dim];

        for (int h = 0; h < heads; h++)
        {
            int off = h * headDim;
            System.Threading.Tasks.Parallel.For(0, t, i =>
            {
                var scores = new float[t];
                var qi = q[i].AsSpan(off, headDim);
                for (int j = 0; j < t; j++)
                    scores[j] = TensorPrimitives.Dot(qi, k[j].AsSpan(off, headDim)) * scale;
                if (maskLastKey) scores[t - 1] += -1.0e4f;
                OpenTail.Stingray.Audio.Primitives.DenseKernels.SoftmaxInPlace(scores);

                var weighted = context[i].AsSpan(off, headDim);
                for (int j = 0; j < t; j++)
                    TensorPrimitives.MultiplyAdd(v[j].AsSpan(off, headDim), scores[j], weighted, weighted);
            });
        }

        return OpenTail.Stingray.Audio.Primitives.DenseKernels.LinearBatchedRows(context, l.AttnOutWeight, l.AttnOutBias, dim, dim);
    }

    // Perf-sweep horizontal pass (docs/perf-sweep-plan.md): was a naive O(outDim*inDim) scalar
    // loop, same anti-pattern found and fixed in Voxtral -- delegates to the shared SIMD/parallel
    // helper (OpenTail.Stingray.Audio.Primitives.DenseKernels).
    private static float[] LinearBias(float[] input, float[] weight, float[] bias, int inDim, int outDim) =>
        OpenTail.Stingray.Audio.Primitives.DenseKernels.Linear(input, weight, bias, inDim, outDim);

    private static float[] LayerNorm(float[] x, float[] weight, float[] bias, float eps = 1e-5f)
    {
        int n = x.Length;
        double sum = 0;
        for (int i = 0; i < n; i++) sum += x[i];
        double mean = sum / n;
        double sumSq = 0;
        for (int i = 0; i < n; i++) { double d = x[i] - mean; sumSq += d * d; }
        double invStd = 1.0 / Math.Sqrt(sumSq / n + eps);
        var output = new float[n];
        for (int i = 0; i < n; i++)
            output[i] = (float)(((x[i] - mean) * invStd) * weight[i] + bias[i]);
        return output;
    }
}
