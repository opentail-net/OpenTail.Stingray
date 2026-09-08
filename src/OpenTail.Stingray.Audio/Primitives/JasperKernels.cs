namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// Common base weight descriptor for 1D convolutions across NVIDIA NeMo's "Jasper" family
/// (MarbleNet VAD, Citrinet ASR, QuartzNet, Jasper), where weights can have BatchNorm folded
/// in at load time and can be depthwise, pointwise 1x1, or general dilated/strided 1D convs.
/// </summary>
public class JasperConv1dWeights
{
    public required int InChannels { get; init; }
    public required int OutChannels { get; init; }
    public required int Kernel { get; init; }
    public required int Stride { get; init; }
    public required int Dilation { get; init; }
    public required int Padding { get; init; }
    public required bool Depthwise { get; init; }
    /// <summary>Depthwise: [channels][kernel]. Regular (incl. pointwise 1x1): [out][in][kernel] flattened.</summary>
    public required float[] Weight { get; init; }
    public required float[] Bias { get; init; }
}

/// <summary>
/// Shared high-performance primitives for NeMo "Jasper"-family 1D convolutional networks
/// (MarbleNet VAD, Citrinet ASR). Unifies Conv1d, ReLU, in-place residual addition, and
/// Linear projection decoders across models, with SIMD-vectorized pointwise 1x1 conv paths
/// and multi-threaded channel-parallel evaluation.
/// </summary>
public static class JasperKernels
{
    /// <summary>
    /// Generic and optimized 1D conv (channel-major [channels][frames] input and output).
    /// Dispatches to a fast SIMD MultiplyAdd path for pointwise 1x1 convs, and parallelizes
    /// across output channels for depthwise and general convolutions.
    /// </summary>
    public static float[][] Conv1d(float[][] input, JasperConv1dWeights conv)
    {
        int inFrames = input[0].Length;
        int outFrames = (inFrames + 2 * conv.Padding - conv.Dilation * (conv.Kernel - 1) - 1) / conv.Stride + 1;
        var output = new float[conv.OutChannels][];
        for (int c = 0; c < conv.OutChannels; c++)
            output[c] = new float[outFrames];

        if (conv.Depthwise)
        {
            Parallel.For(0, conv.OutChannels, c =>
            {
                var inRow = input[c];
                var outRow = output[c];
                int wBase = c * conv.Kernel;
                float bias = conv.Bias[c];
                for (int t = 0; t < outFrames; t++)
                {
                    float sum = bias;
                    int inStart = t * conv.Stride - conv.Padding;
                    for (int k = 0; k < conv.Kernel; k++)
                    {
                        int inIdx = inStart + k * conv.Dilation;
                        if ((uint)inIdx < (uint)inFrames)
                            sum += conv.Weight[wBase + k] * inRow[inIdx];
                    }
                    outRow[t] = sum;
                }
            });
        }
        else if (conv.Kernel == 1 && conv.Stride == 1 && conv.Padding == 0 && conv.Dilation == 1)
        {
            // Pointwise 1x1 fast path: vectorizable across frames with contiguous sequential memory
            Parallel.For(0, conv.OutChannels, oc =>
            {
                var outRow = output[oc];
                float bias = conv.Bias[oc];
                outRow.AsSpan().Fill(bias);
                int wBase = oc * conv.InChannels;
                for (int ic = 0; ic < conv.InChannels; ic++)
                {
                    float w = conv.Weight[wBase + ic];
                    if (w == 0f) continue;
                    TensorPrimitives.MultiplyAdd((ReadOnlySpan<float>)input[ic], w, outRow, outRow);
                }
            });
        }
        else
        {
            // General regular 1D conv with kernel > 1, stride, or dilation
            Parallel.For(0, conv.OutChannels, oc =>
            {
                var outRow = output[oc];
                float bias = conv.Bias[oc];
                for (int t = 0; t < outFrames; t++)
                {
                    float sum = bias;
                    int inStart = t * conv.Stride - conv.Padding;
                    for (int ic = 0; ic < conv.InChannels; ic++)
                    {
                        var inRow = input[ic];
                        int wBase = (oc * conv.InChannels + ic) * conv.Kernel;
                        for (int k = 0; k < conv.Kernel; k++)
                        {
                            int inIdx = inStart + k * conv.Dilation;
                            if ((uint)inIdx < (uint)inFrames)
                                sum += conv.Weight[wBase + k] * inRow[inIdx];
                        }
                    }
                    outRow[t] = sum;
                }
            });
        }

        return output;
    }

    /// <summary>In-place ReLU activation across [channels][frames].</summary>
    public static void Relu(float[][] x)
    {
        Parallel.For(0, x.Length, c =>
        {
            var row = x[c];
            for (int t = 0; t < row.Length; t++)
            {
                if (row[t] < 0f) row[t] = 0f;
            }
        });
    }

    /// <summary>In-place residual addition: x[c][t] += residual[c][t] using SIMD.</summary>
    public static void AddResidualInPlace(float[][] x, float[][] residual)
    {
        Parallel.For(0, x.Length, c =>
        {
            TensorPrimitives.Add((ReadOnlySpan<float>)x[c], residual[c], x[c]);
        });
    }

    /// <summary>
    /// Projects [channels][frames] hidden representation to [frames][numClasses] logits.
    /// Uses SIMD MultiplyAdd across frames per output class.
    /// </summary>
    public static float[][] LinearDecoder(float[][] x, float[] weight, float[] bias, int numClasses)
    {
        int frames = x[0].Length, inChannels = x.Length;
        var logits = new float[frames][];
        for (int t = 0; t < frames; t++)
            logits[t] = new float[numClasses];

        Parallel.For(0, numClasses, c =>
        {
            float b = bias[c];
            int wBase = c * inChannels;
            var classLogits = new float[frames];
            classLogits.AsSpan().Fill(b);
            for (int ic = 0; ic < inChannels; ic++)
            {
                float w = weight[wBase + ic];
                if (w == 0f) continue;
                TensorPrimitives.MultiplyAdd((ReadOnlySpan<float>)x[ic], w, classLogits, classLogits);
            }
            for (int t = 0; t < frames; t++)
            {
                logits[t][c] = classLogits[t];
            }
        });

        return logits;
    }
}
