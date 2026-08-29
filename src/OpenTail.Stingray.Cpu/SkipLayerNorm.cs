
namespace OpenTail.Stingray.Cpu;

/// <summary>
/// Fused Skip-Layer Normalization and Skip-RMSNorm operator kernels.
/// Reference: ONNX Runtime contrib_ops/cpu/skip_layer_norm.cc &amp; skip_layer_norm_helper.h
/// 
/// Why it's useful:
/// In transformer decoder layers, every attention and MLP block computes:
///     x = LayerNorm(x + skip + bias)   or   x = RmsNorm(x + skip)
/// Fusing the residual vector addition, mean/variance reduction, and normalization into a single memory pass
/// eliminates intermediate tensor allocation and halves memory bandwidth consumption across all decoder layers.
/// </summary>
public static unsafe class SkipLayerNorm
{
    /// <summary>
    /// Computes fused Skip-Layer Normalization: y = LayerNorm(input + skip + bias) * gamma + beta.
    /// Optionally preserves the intermediate residual addition: (input + skip + bias) into skipPlusInputOutput.
    /// Source: onnxruntime::contrib::SkipLayerNorm::ComputeJob (skip_layer_norm.cc).
    /// </summary>
    public static void Compute(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> skip,
        ReadOnlySpan<float> gamma,
        ReadOnlySpan<float> beta,
        ReadOnlySpan<float> bias,
        Span<float> output,
        Span<float> skipPlusInputOutput,
        int rows,
        int hiddenSize,
        float epsilon = 1e-5f)
    {
        bool hasBeta = !beta.IsEmpty;
        bool hasBias = !bias.IsEmpty;
        bool saveResidual = !skipPlusInputOutput.IsEmpty;
        int skipSize = skip.Length;

        fixed (float* pIn = input)
        fixed (float* pSkip = skip)
        fixed (float* pGamma = gamma)
        fixed (float* pBeta = beta)
        fixed (float* pBias = bias)
        fixed (float* pOut = output)
        fixed (float* pResOut = skipPlusInputOutput)
        {
            var inPtr = pIn;
            var skipPtr = pSkip;
            var gammaPtr = pGamma;
            var betaPtr = pBeta;
            var biasPtr = pBias;
            var outPtr = pOut;
            var resOutPtr = pResOut;

            Parallel.For(0, rows, r =>
            {
                long offset = (long)r * hiddenSize;
                long skipOffset = (offset % skipSize);

                float* rowIn = inPtr + offset;
                float* rowSkip = skipPtr + skipOffset;
                float* rowOut = outPtr + offset;
                float* rowRes = saveResidual ? (resOutPtr + offset) : null;

                float sum = 0.0f;
                float sumSq = 0.0f;

                // Pass 1: Element-wise addition + Accumulate mean & variance
                for (int h = 0; h < hiddenSize; h++)
                {
                    float val = rowIn[h] + rowSkip[h];
                    if (hasBias) val += biasPtr[h];

                    if (rowRes != null) rowRes[h] = val;
                    rowOut[h] = val;

                    sum += val;
                    sumSq += val * val;
                }

                float mean = sum / hiddenSize;
                float variance = MathF.Max(0.0f, (sumSq / hiddenSize) - (mean * mean));
                float invStd = 1.0f / MathF.Sqrt(variance + epsilon);

                // Pass 2: Normalize and scale
                for (int h = 0; h < hiddenSize; h++)
                {
                    float normVal = (rowOut[h] - mean) * invStd * gammaPtr[h];
                    if (hasBeta) normVal += betaPtr[h];
                    rowOut[h] = normVal;
                }
            });
        }
    }

    /// <summary>
    /// Computes fused Skip-RMSNorm (Simplified Layer Normalization): y = RMSNorm(input + skip) * gamma.
    /// Source: onnxruntime::contrib::SkipSimplifiedLayerNormalization (skip_layer_norm.cc).
    /// </summary>
    public static void ComputeRmsNorm(
        ReadOnlySpan<float> input,
        ReadOnlySpan<float> skip,
        ReadOnlySpan<float> gamma,
        Span<float> output,
        Span<float> skipPlusInputOutput,
        int rows,
        int hiddenSize,
        float epsilon = 1e-5f)
    {
        bool saveResidual = !skipPlusInputOutput.IsEmpty;
        int skipSize = skip.Length;

        fixed (float* pIn = input)
        fixed (float* pSkip = skip)
        fixed (float* pGamma = gamma)
        fixed (float* pOut = output)
        fixed (float* pResOut = skipPlusInputOutput)
        {
            var inPtr = pIn;
            var skipPtr = pSkip;
            var gammaPtr = pGamma;
            var outPtr = pOut;
            var resOutPtr = pResOut;

            Parallel.For(0, rows, r =>
            {
                long offset = (long)r * hiddenSize;
                long skipOffset = (offset % skipSize);

                float* rowIn = inPtr + offset;
                float* rowSkip = skipPtr + skipOffset;
                float* rowOut = outPtr + offset;
                float* rowRes = saveResidual ? (resOutPtr + offset) : null;

                float sumSq = 0.0f;

                // Pass 1: Element-wise addition + Accumulate sum of squares
                for (int h = 0; h < hiddenSize; h++)
                {
                    float val = rowIn[h] + rowSkip[h];
                    if (rowRes != null) rowRes[h] = val;
                    rowOut[h] = val;
                    sumSq += val * val;
                }

                float invRms = 1.0f / MathF.Sqrt((sumSq / hiddenSize) + epsilon);

                // Pass 2: Normalize and scale
                for (int h = 0; h < hiddenSize; h++)
                {
                    rowOut[h] = rowOut[h] * invRms * gammaPtr[h];
                }
            });
        }
    }
}
