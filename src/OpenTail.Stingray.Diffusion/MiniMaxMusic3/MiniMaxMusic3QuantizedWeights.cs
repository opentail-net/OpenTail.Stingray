using System.Buffers;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Diffusion.MiniMaxMusic3;

/// <summary>
/// Holds Q8_0 (8-bit blockwise symmetric) quantized weights for a single transformer block.
/// Weight row format matches <see cref="SimdKernels.DotQ8_0_Q8_0"/> (34 bytes per 32-element block).
/// </summary>
public sealed class MiniMaxMusic3QuantizedBlockWeights
{
    public required byte[] QWeight { get; init; }
    public required byte[] KWeight { get; init; }
    public required byte[] VWeight { get; init; }
    public required byte[] OWeight { get; init; }
    public required byte[] FfInWeight { get; init; }
    public required byte[] FfOutWeight { get; init; }

    // Small enough (a few KB/block) that keeping these as plain FP32 is not worth quantizing --
    // kept directly here (not via a reference to the parent MiniMaxMusic3TransformerWeights) so
    // QuantizeFrom's caller can drop its FP32 weights object and actually free the big Q/K/V/O/
    // FfIn/FfOut arrays. See MiniMaxMusic3QuantizedTransformerWeights's own doc comment.
    public required float[] Norm1Weight { get; init; }
    public required float[] Norm1Bias { get; init; }
    public required float[] Norm2Weight { get; init; }
    public required float[] Norm2Bias { get; init; }
    public required float[] FfInBias { get; init; }
    public required float[] FfOutBias { get; init; }
}

/// <summary>
/// Quantized container for <see cref="MiniMaxMusic3Transformer"/> weights in Q8_0 format.
/// Linear weights take 34 bytes per 32 elements (~1.0625 bytes/weight vs 4.0 in FP32),
/// yielding 3.76x reduction in the SIZE of those weights specifically.
///
/// <para><b>Real memory-saving contract</b>: this type deliberately does NOT hold a reference to
/// the source <see cref="MiniMaxMusic3TransformerWeights"/> (an earlier version kept the whole
/// FP32 object alive via an `OriginalFp32` field so the small per-block norms/biases and the
/// timestep-embedding weights had somewhere to come from -- which meant the ~6GB of FP32 Q/K/V/O/
/// FfIn/FfOut/proj/conv arrays never actually became eligible for GC, so quantizing only ever
/// ADDED ~1.6GB on top of the FP32 baseline instead of replacing it). The small FP32 tensors this
/// forward pass genuinely still needs (LayerNorm weights/biases, FF biases, the timestep Fourier/
/// MLP weights) are copied out by reference into this type and <see cref="MiniMaxMusic3QuantizedBlockWeights"/>
/// directly, so once the CALLER drops its own reference to the source
/// <see cref="MiniMaxMusic3TransformerWeights"/>, the big FP32 matrices are free to be collected --
/// this type never re-references them.</para>
/// </summary>
public sealed class MiniMaxMusic3QuantizedTransformerWeights
{
    public required float[] TimeProjWeight { get; init; }
    public required TimestepEmbeddingWeights TimeEmbed { get; init; }
    public required byte[] PreprocessConvWeight { get; init; }
    public required byte[] ProjInWeight { get; init; }
    public required MiniMaxMusic3QuantizedBlockWeights[] Blocks { get; init; }
    public required byte[] ProjOutWeight { get; init; }
    public required byte[] PostprocessConvWeight { get; init; }

    /// <summary>
    /// Quantizes full-precision FP32 weights into Q8_0 in parallel. Does not retain <paramref name="w"/>
    /// (or any of its big matrices) -- once the caller drops its own reference to <paramref name="w"/>,
    /// those FP32 arrays are free to be collected. Real memory saving requires the caller to actually
    /// do that (e.g. don't keep both the FP32 and quantized weights resident at once unless you need
    /// both paths available simultaneously).
    /// </summary>
    public static unsafe MiniMaxMusic3QuantizedTransformerWeights QuantizeFrom(MiniMaxMusic3TransformerWeights w)
    {
        const int inChannels = MiniMaxMusic3Config.TransformerInChannels; // 128
        const int condDim = MiniMaxMusic3Config.TransformerConditionDim; // 2048
        const int concatChannels = 2 * inChannels + condDim; // 2304
        const int innerDim = MiniMaxMusic3Config.TransformerNumAttentionHeads * MiniMaxMusic3Config.TransformerAttentionHeadDim; // 2048
        const int ffn = MiniMaxMusic3Config.TransformerFfInnerDim; // 8192

        var prepConv = QuantizeMatrixQ8(w.PreprocessConvWeight, concatChannels, concatChannels);
        var projIn = QuantizeMatrixQ8(w.ProjInWeight, innerDim, concatChannels);

        var blocks = new MiniMaxMusic3QuantizedBlockWeights[w.Blocks.Length];
        System.Threading.Tasks.Parallel.For(0, w.Blocks.Length, i =>
        {
            var b = w.Blocks[i];
            blocks[i] = new MiniMaxMusic3QuantizedBlockWeights
            {
                QWeight = QuantizeMatrixQ8(b.QWeight, innerDim, innerDim),
                KWeight = QuantizeMatrixQ8(b.KWeight, innerDim, innerDim),
                VWeight = QuantizeMatrixQ8(b.VWeight, innerDim, innerDim),
                OWeight = QuantizeMatrixQ8(b.OWeight, innerDim, innerDim),
                FfInWeight = QuantizeMatrixQ8(b.FfInWeight, 2 * ffn, innerDim),
                FfOutWeight = QuantizeMatrixQ8(b.FfOutWeight, innerDim, ffn),
                Norm1Weight = b.Norm1Weight,
                Norm1Bias = b.Norm1Bias,
                Norm2Weight = b.Norm2Weight,
                Norm2Bias = b.Norm2Bias,
                FfInBias = b.FfInBias,
                FfOutBias = b.FfOutBias
            };
        });

        var projOut = QuantizeMatrixQ8(w.ProjOutWeight, inChannels, innerDim);
        var postConv = QuantizeMatrixQ8(w.PostprocessConvWeight, inChannels, inChannels);

        return new MiniMaxMusic3QuantizedTransformerWeights
        {
            TimeProjWeight = w.TimeProjWeight,
            TimeEmbed = w.TimeEmbed,
            PreprocessConvWeight = prepConv,
            ProjInWeight = projIn,
            Blocks = blocks,
            ProjOutWeight = projOut,
            PostprocessConvWeight = postConv
        };
    }

    /// <summary>
    /// Quantizes an FP32 matrix of [rows, cols] to Q8_0 format (34 bytes per 32-element block).
    /// </summary>
    public static unsafe byte[] QuantizeMatrixQ8(float[] matrix, int rows, int cols)
    {
        if (cols % 32 != 0)
            throw new ArgumentException($"Columns ({cols}) must be a multiple of 32 for Q8_0 quantization.");

        int numBlocksPerRow = cols / 32;
        int bytesPerRow = numBlocksPerRow * 34;
        byte[] result = GC.AllocateUninitializedArray<byte>(rows * bytesPerRow);

        fixed (float* pSrc = matrix)
        fixed (byte* pDst = result)
        {
            nint srcAddr = (nint)pSrc;
            nint dstAddr = (nint)pDst;

            System.Threading.Tasks.Parallel.For(0, rows, r =>
            {
                float* rowSrc = (float*)srcAddr + (long)r * cols;
                byte* rowDst = (byte*)dstAddr + (long)r * bytesPerRow;

                for (int b = 0; b < numBlocksPerRow; b++)
                {
                    float* x = rowSrc + b * 32;
                    byte* wb = rowDst + b * 34;

                    float amax = 0f;
                    for (int i = 0; i < 32; i++)
                    {
                        float a = MathF.Abs(x[i]);
                        if (a > amax) amax = a;
                    }

                    float d = amax / 127f;
                    Half dHalf = (Half)d;
                    float id = d != 0f ? 1f / (float)dHalf : 0f;

                    *(ushort*)wb = BitConverter.HalfToUInt16Bits(dHalf);
                    sbyte* qs = (sbyte*)(wb + 2);
                    for (int i = 0; i < 32; i++)
                    {
                        qs[i] = (sbyte)MathF.Round(x[i] * id);
                    }
                }
            });
        }

        return result;
    }
}
