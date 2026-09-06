
namespace OpenTail.Stingray.Audio.FunASR;

/// <summary>
/// Real SAN-M (Self-Attention + FSMN-augmented) block for the CURRENT Fun-ASR-Nano-2512
/// architecture's audio tower, transcribed directly from
/// `examples/audio.cpp/src/framework/modules/speech_encoders/sanm.cpp`'s
/// `build_block`/`build_attention_branch`/`build_ffn_residual` (not guessed) -- NOT the same
/// architecture as the old CIF-Paraformer files in this same folder (`FunAsrEncoder.cs` et al),
/// which target a different, no-longer-published checkpoint format; see
/// `docs/audio-review-progress.md`'s "FunASR-nano" section for the full correction. Real per-block
/// math: `LayerNorm -> {Q,K,V=Linear(bias)} -> standard non-causal scaled-dot-product attention ->
/// Linear(bias) out-projection` ADDED to `FSMN depthwise-conv1d(V, kernel, causal-symmetric pad,
/// NO bias) + V` (the FSMN "memory" branch operates on V, not the attention output), then
/// (for residual blocks only) added to the block's own input, then a real post-LN FFN
/// (`LayerNorm -> Linear(bias) -> ReLU -> Linear(bias)`, added to its own input).
/// </summary>
public static class FunAsrNanoSanmBlock
{
    /// <summary>Stem/projection block: input_size may differ from model_size, so there is NO
    /// input-to-output residual add around the attention branch (only the FFN has its own
    /// internal residual, added to the attention branch's own output).</summary>
    public static float[][] ProjectionBlock(float[][] input, FunAsrNanoSanmWeights w, int inputSize, int modelSize, int numHeads, int ffnSize, int kernelSize, float eps)
        => Build(input, w, inputSize, modelSize, numHeads, ffnSize, kernelSize, eps, addInputResidual: false);

    /// <summary>Main/timestamp layer block: input_size == model_size, with the attention branch's
    /// output added to the block's own input before the FFN.</summary>
    public static float[][] ResidualBlock(float[][] input, FunAsrNanoSanmWeights w, int modelSize, int numHeads, int ffnSize, int kernelSize, float eps)
        => Build(input, w, modelSize, modelSize, numHeads, ffnSize, kernelSize, eps, addInputResidual: true);

    private static float[][] Build(float[][] input, FunAsrNanoSanmWeights w, int inputSize, int modelSize, int numHeads, int ffnSize, int kernelSize, float eps, bool addInputResidual)
    {
        int t = input.Length;
        var normed = LayerNormRows(input, inputSize, w.SelfAttnLayerNormWeight, w.SelfAttnLayerNormBias, eps);

        var q = LinearRows(normed, w.QWeight, w.QBias, inputSize, modelSize);
        var k = LinearRows(normed, w.KWeight, w.KBias, inputSize, modelSize);
        var v = LinearRows(normed, w.VWeight, w.VBias, inputSize, modelSize);

        var attn = SelfAttention(q, k, v, numHeads, modelSize);
        var attnOut = LinearRows(attn, w.OutWeight, w.OutBias, modelSize, modelSize);

        var fsmn = FsmnDepthwiseConv(v, w.FsmnConvWeight, modelSize, kernelSize);
        for (int i = 0; i < t; i++)
            for (int c = 0; c < modelSize; c++)
                fsmn[i][c] += v[i][c];

        var branchOutput = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[modelSize];
            for (int c = 0; c < modelSize; c++) row[c] = attnOut[i][c] + fsmn[i][c];
            branchOutput[i] = row;
        }

        var residual = branchOutput;
        if (addInputResidual)
        {
            residual = new float[t][];
            for (int i = 0; i < t; i++)
            {
                var row = new float[modelSize];
                for (int c = 0; c < modelSize; c++) row[c] = input[i][c] + branchOutput[i][c];
                residual[i] = row;
            }
        }

        var ffnNorm = LayerNormRows(residual, modelSize, w.FinalLayerNormWeight, w.FinalLayerNormBias, eps);
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var h = LinearRow(ffnNorm[i], w.Fc1Weight, w.Fc1Bias, modelSize, ffnSize);
            for (int c = 0; c < ffnSize; c++) if (h[c] < 0f) h[c] = 0f;
            var ffnOut = LinearRow(h, w.Fc2Weight, w.Fc2Bias, ffnSize, modelSize);
            var row = new float[modelSize];
            for (int c = 0; c < modelSize; c++) row[c] = residual[i][c] + ffnOut[c];
            output[i] = row;
        }
        return output;
    }

    private static float[][] SelfAttention(float[][] q, float[][] k, float[][] v, int numHeads, int modelSize)
    {
        int t = q.Length;
        int headDim = modelSize / numHeads;
        float scale = 1f / MathF.Sqrt(headDim);
        var output = new float[t][];
        for (int i = 0; i < t; i++) output[i] = new float[modelSize];

        var scores = new float[t];
        for (int h = 0; h < numHeads; h++)
        {
            int hBase = h * headDim;
            for (int i = 0; i < t; i++)
            {
                float maxScore = float.NegativeInfinity;
                for (int j = 0; j < t; j++)
                {
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++) dot += q[i][hBase + d] * k[j][hBase + d];
                    dot *= scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }
                float sum = 0f;
                for (int j = 0; j < t; j++) { scores[j] = MathF.Exp(scores[j] - maxScore); sum += scores[j]; }
                float invSum = 1f / sum;
                for (int d = 0; d < headDim; d++)
                {
                    float acc = 0f;
                    for (int j = 0; j < t; j++) acc += scores[j] * invSum * v[j][hBase + d];
                    output[i][hBase + d] = acc;
                }
            }
        }
        return output;
    }

    /// <summary>Real FSMN depthwise conv1d: kernel `kernelSize` (odd), symmetric pad
    /// `(kernelSize-1)/2`, NO bias, weight layout `[channels, 1, kernelSize]` (PyTorch depthwise
    /// Conv1d convention, flatten idx = `c*kernelSize+k`).</summary>
    private static float[][] FsmnDepthwiseConv(float[][] v, float[] weight, int channels, int kernelSize)
    {
        int t = v.Length;
        int pad = (kernelSize - 1) / 2;
        var output = new float[t][];
        for (int i = 0; i < t; i++)
        {
            var row = new float[channels];
            for (int c = 0; c < channels; c++)
            {
                float sum = 0f;
                int wBase = c * kernelSize;
                for (int kk = 0; kk < kernelSize; kk++)
                {
                    int srcT = i - pad + kk;
                    if (srcT < 0 || srcT >= t) continue;
                    sum += weight[wBase + kk] * v[srcT][c];
                }
                row[c] = sum;
            }
            output[i] = row;
        }
        return output;
    }

    private static float[][] LayerNormRows(float[][] xRows, int dim, float[] weight, float[] bias, float eps)
    {
        var output = new float[xRows.Length][];
        for (int i = 0; i < xRows.Length; i++)
        {
            var row = xRows[i];
            double mean = 0;
            for (int c = 0; c < dim; c++) mean += row[c];
            mean /= dim;
            double variance = 0;
            for (int c = 0; c < dim; c++) { double d = row[c] - mean; variance += d * d; }
            variance /= dim;
            float invStd = (float)(1.0 / Math.Sqrt(variance + eps));
            var outRow = new float[dim];
            for (int c = 0; c < dim; c++) outRow[c] = (float)((row[c] - mean) * invStd) * weight[c] + bias[c];
            output[i] = outRow;
        }
        return output;
    }

    private static float[][] LinearRows(float[][] xRows, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[xRows.Length][];
        for (int i = 0; i < xRows.Length; i++) output[i] = LinearRow(xRows[i], weight, bias, inDim, outDim);
        return output;
    }

    private static float[] LinearRow(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            float sum = bias[o];
            int wBase = o * inDim;
            for (int i = 0; i < inDim; i++) sum += input[i] * weight[wBase + i];
            output[o] = sum;
        }
        return output;
    }
}

public sealed class FunAsrNanoSanmWeights
{
    public float[] SelfAttnLayerNormWeight { get; set; } = [];
    public float[] SelfAttnLayerNormBias { get; set; } = [];
    public float[] QWeight { get; set; } = [];
    public float[] QBias { get; set; } = [];
    public float[] KWeight { get; set; } = [];
    public float[] KBias { get; set; } = [];
    public float[] VWeight { get; set; } = [];
    public float[] VBias { get; set; } = [];
    public float[] OutWeight { get; set; } = [];
    public float[] OutBias { get; set; } = [];
    public float[] FsmnConvWeight { get; set; } = []; // [modelSize, 1, kernelSize], no bias
    public float[] FinalLayerNormWeight { get; set; } = [];
    public float[] FinalLayerNormBias { get; set; } = [];
    public float[] Fc1Weight { get; set; } = [];
    public float[] Fc1Bias { get; set; } = [];
    public float[] Fc2Weight { get; set; } = [];
    public float[] Fc2Bias { get; set; } = [];
}
