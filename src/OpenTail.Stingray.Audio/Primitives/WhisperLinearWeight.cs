using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// A Whisper linear-layer weight matrix that dispatches to the real hardware F16C kernel
/// (<see cref="F16CNative"/>) when the on-disk tensor was genuinely F16 and the native shim loaded
/// successfully, falling back to the existing parallel-per-row F32 <see cref="SimdKernels.MatVecF32"/>
/// path otherwise (Safetensors-sourced weights, non-Windows/non-x64, or missing/failed native
/// shim). See docs/audio-review-progress.md's ggml/F16C investigation entry: this is the one real
/// win out of this session's several F16/quantization attempts, measured 4.5-4.7x faster on
/// Whisper's actual encoder projection shapes and 2x faster on the decoder's tied LM head.
/// </summary>
public sealed class WhisperLinearWeight
{
    private readonly short[]? _f16Bits;
    private readonly float[]? _f32;
    private readonly int _outDim;
    private readonly int _inDim;

    public int OutDim => _outDim;
    public int InDim => _inDim;

    private WhisperLinearWeight(short[]? f16Bits, float[]? f32, int outDim, int inDim)
    {
        _f16Bits = f16Bits;
        _f32 = f32;
        _outDim = outDim;
        _inDim = inDim;
    }

    /// <summary>
    /// Builds from a Whisper ggml/GGUF model tensor: uses the real F16C path when the tensor's
    /// on-disk dtype was F16 and the native shim is available, otherwise falls back to the
    /// already-dequantized F32 copy (Safetensors, or F16C unavailable on this machine).
    /// </summary>
    public static WhisperLinearWeight FromTensor(Whisper.WhisperGgmlModel model, string name, int outDim, int inDim, float[] f32Fallback)
    {
        if (F16CNative.IsAvailable && model.TryGetTensorRawF16(name, out var f16Bits))
            return new WhisperLinearWeight(f16Bits, null, outDim, inDim);
        return new WhisperLinearWeight(null, f32Fallback, outDim, inDim);
    }

    /// <summary>
    /// Batched linear layer: output[seqLen, outDim] = input[seqLen, inDim] @ weight[outDim, inDim]^T.
    /// Same parallel-per-row dispatch shape as the pre-existing F32 path (parallelize over seqLen
    /// when seqLen &gt;= 8; parallelize over output rows instead when seqLen == 1, the decode-step
    /// case, since there's nothing else to parallelize over).
    /// </summary>
    public unsafe void MatVecBatched(ReadOnlySpan<float> input, int seqLen, Span<float> output)
    {
        if (_f16Bits is { } f16)
        {
            fixed (float* pIn = input, pOut = output)
            fixed (short* pW = f16)
            {
                ushort* wBase = (ushort*)pW;
                if (seqLen == 1)
                {
                    float* rowIn = pIn;
                    nint inAddr = (nint)rowIn, wAddr = (nint)wBase, outAddr = (nint)pOut;
                    int inDim = _inDim;
                    System.Threading.Tasks.Parallel.For(0, _outDim, o =>
                    {
                        ushort* wRow = (ushort*)wAddr + (nuint)o * (nuint)inDim;
                        ((float*)outAddr)[o] = F16CNative.Dot((float*)inAddr, wRow, inDim);
                    });
                }
                else
                {
                    nint inAddr = (nint)pIn, wAddr = (nint)wBase, outAddr = (nint)pOut;
                    int inDim = _inDim, outDim = _outDim;
                    System.Threading.Tasks.Parallel.For(0, seqLen, t =>
                    {
                        float* rowIn = (float*)inAddr + (nuint)t * (nuint)inDim;
                        float* rowOut = (float*)outAddr + (nuint)t * (nuint)outDim;
                        for (int o = 0; o < outDim; o++)
                        {
                            ushort* wRow = (ushort*)wAddr + (nuint)o * (nuint)inDim;
                            rowOut[o] = F16CNative.Dot(rowIn, wRow, inDim);
                        }
                    });
                }
            }
            return;
        }

        var f32 = _f32!;
        fixed (float* pIn = input, pW = f32, pOut = output)
        {
            if (seqLen >= 8)
            {
                nint inAddr = (nint)pIn, wAddr = (nint)pW, outAddr = (nint)pOut;
                int inDim = _inDim, outDim = _outDim;
                System.Threading.Tasks.Parallel.For(0, seqLen, t =>
                {
                    float* rowIn = (float*)inAddr + (nuint)t * (nuint)inDim;
                    float* rowOut = (float*)outAddr + (nuint)t * (nuint)outDim;
                    SimdKernels.MatVecF32(rowOut, (float*)wAddr, rowIn, outDim, inDim);
                });
            }
            else
            {
                SimdKernels.MatMulBatchedF32(pOut, pW, pIn, seqLen, _outDim, _inDim);
            }
        }
    }
}
