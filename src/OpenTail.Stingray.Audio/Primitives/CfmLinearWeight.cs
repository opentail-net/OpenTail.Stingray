using CoreTensor = OpenTail.Stingray.Core.Tensor;
using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// A CFM UNet linear-layer weight (see <see cref="CfmUNetKernels"/>, shared by CosyVoice2's and
/// Chatterbox's CFM decoders) that dispatches to the real hardware F16C kernel
/// (<see cref="F16CNative"/>) when available, falling back to the existing F32
/// <see cref="SimdKernels.MatVecF32"/> path otherwise.
///
/// <para>Unlike <see cref="WhisperLinearWeight"/> (which reuses raw F16 bits already present in
/// Whisper's ggml/GGUF files at zero conversion cost), CosyVoice/Chatterbox's weights are
/// Safetensors-sourced -- real F32 on disk. This class instead converts F32 -&gt; F16 ONCE at
/// construction time (a small one-time cost, amortized over the full lifetime of the loaded
/// pipeline) and dispatches every subsequent call through the same proven native kernel. Measured
/// (docs/audio-review-progress.md's ggml/F16C investigation entry) 2.6-3.1x faster than plain F32
/// at the CFM decoder's actual shapes (t=128 timesteps, dim 256-1024) -- a smaller multiplier than
/// Whisper's 4.5-4.7x (the win scales with row count, and CFM's per-call row count is much smaller
/// than Whisper's), but real.</para>
///
/// <para>Only used for the CFM UNet's linear projections (self-attention Q/K/V/Out, feed-forward
/// up/down, and the resnet block's time-embedding MLP) -- NOT the causal convolutions
/// (<c>CausalConv1d</c>/<c>Conv1dK1</c>), which are a structurally different (channel-parallel,
/// kernel-window) operation this class doesn't address.</para>
/// </summary>
public sealed class CfmLinearWeight
{
    private readonly short[]? _f16Bits;
    private readonly float[]? _f32;
    private readonly int _outDim;
    private readonly int _inDim;

    public int OutDim => _outDim;
    public int InDim => _inDim;

    // Lazily-uploaded, persistent GPU copy of this weight (--backend vulkan path). Uploaded once
    // on first GPU use and reused for every subsequent call (weights are static across the whole
    // Euler ODE solve), matching ZImageDiT's existing weight-caching convention.
    private CoreTensor? _gpuWeight;
    private IComputeBackend? _gpuBackend;

    private CfmLinearWeight(short[]? f16Bits, float[]? f32, int outDim, int inDim)
    {
        _f16Bits = f16Bits;
        _f32 = f32;
        _outDim = outDim;
        _inDim = inDim;
    }

    /// <summary>Batch linear layer via a GPU backend: outputMatrix[T, outDim] = inputMatrix[T, inDim] * weight^T + bias.
    /// Weight is uploaded once and cached on this instance; bias is added on the CPU after download
    /// (bias-add is negligible next to the matmul and IComputeBackend has no fused bias-add).</summary>
    public unsafe void GpuMatMul(IComputeBackend backend, float* inputMatrix, int t, float* outputMatrix, float* bias = null)
    {
        if (_f32 is not { } w) throw new InvalidOperationException("GpuMatMul requires an F32-backed CfmLinearWeight.");

        if (_gpuWeight is null || !ReferenceEquals(_gpuBackend, backend))
        {
            _gpuWeight = backend.Upload(w, TensorShape.D1(w.Length), exact: true);
            _gpuBackend = backend;
        }

        var xGpu = backend.Upload(new ReadOnlySpan<float>(inputMatrix, t * _inDim), TensorShape.D1(t * _inDim));
        var cGpu = backend.Allocate(TensorShape.D1(t * _outDim));
        try
        {
            backend.Sgemm(cGpu, xGpu, _gpuWeight, t, _inDim, _outDim);
            backend.Synchronize();
            backend.Download(cGpu, new Span<float>(outputMatrix, t * _outDim));
        }
        finally
        {
            backend.Free(xGpu);
            backend.Free(cGpu);
        }

        if (bias is not null)
        {
            for (int row = 0; row < t; row++)
            {
                float* outRow = outputMatrix + (nint)row * _outDim;
                for (int o = 0; o < _outDim; o++) outRow[o] += bias[o];
            }
        }
    }

    /// <summary>Creates a CFM UNet linear-layer weight preserving full Float32 precision.</summary>
    public static CfmLinearWeight FromF32(float[] weightF32, int outDim, int inDim)
    {
        return new CfmLinearWeight(null, weightF32, outDim, inDim);
    }

    /// <summary>Batch-of-2 row-major matmul: streams each weight row ONCE from RAM and applies it to
    /// BOTH input vectors (e.g. a CFG conditional/unconditional pair) via a single `Parallel.For`
    /// dispatch over output rows, instead of two separate <see cref="MatMul"/> calls (each of which
    /// independently re-streams the whole weight matrix and dispatches its own `Parallel.For`).
    ///
    /// <para>Added for single-token (`t=1`) incremental decode call sites where a weight matrix is
    /// too large to stay resident in cache across sequential calls (e.g. MiniMax-Music3's RVQ depth
    /// decoder: a 4096x4096 F32 weight is 64MB, far larger than this machine's L3 cache, so two
    /// sequential single-token <see cref="MatMul"/> calls each pay the full RAM-bandwidth cost of
    /// streaming it -- exactly the class of fix already landed for
    /// <c>MiniMaxMusic3Transformer.MatMulRowMajor</c>, applied here to the shared kernel instead of
    /// being duplicated inline). Additive: does not change <see cref="MatMul"/> or any other existing
    /// call site's behavior.</para>
    ///
    /// <para>F32 weights only (this project's real MiniMax-Music3 depth-decoder weights are F32, not
    /// F16) -- throws if this instance was constructed from F16 bits.</para>
    /// </summary>
    public unsafe void MatMulPairRowMajor(float* input0, float* input1, float* output0, float* output1, float* bias = null)
    {
        if (_f32 is not { } w) throw new NotSupportedException("MatMulPairRowMajor requires an F32-backed CfmLinearWeight.");
        int inDim = _inDim, outDim = _outDim;

        fixed (float* wp = w)
        {
            int numThreads = Math.Min(Environment.ProcessorCount, (outDim + 63) / 64);
            if (numThreads <= 1)
            {
                RunRows(wp, 0, outDim);
                return;
            }

            int chunkSize = (outDim + numThreads - 1) / numThreads;
            nint wAddr = (nint)wp, o0Addr = (nint)output0, o1Addr = (nint)output1,
                i0Addr = (nint)input0, i1Addr = (nint)input1, bAddr = (nint)bias;

            System.Threading.Tasks.Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(outDim, start + chunkSize);
                RunRowsThreadLocal((float*)wAddr, (float*)i0Addr, (float*)i1Addr, (float*)o0Addr, (float*)o1Addr, (float*)bAddr, inDim, start, end);
            });
        }

        void RunRows(float* weights, int start, int end) =>
            RunRowsThreadLocal(weights, input0, input1, output0, output1, bias, inDim, start, end);

        static void RunRowsThreadLocal(float* weights, float* in0, float* in1, float* out0, float* out1, float* b, int inDim, int start, int end)
        {
            for (int r = start; r < end; r++)
            {
                float* wRow = weights + (long)r * inDim;
                float v0 = SimdKernels.DotF32(wRow, in0, inDim);
                float v1 = SimdKernels.DotF32(wRow, in1, inDim);
                if (b != null) { v0 += b[r]; v1 += b[r]; }
                out0[r] = v0;
                out1[r] = v1;
            }
        }
    }

    /// <summary>Batch linear layer across T rows: outputMatrix[T, outDim] = inputMatrix[T, inDim] * weight^T + bias.</summary>
    public unsafe void MatMul(float* inputMatrix, int t, float* outputMatrix, float* bias = null)
    {
        int inDim = _inDim, outDim = _outDim;
        if (_f16Bits is { } f16)
        {
            fixed (short* pW = f16)
            {
                nint inAddr = (nint)inputMatrix;
                nint outAddr = (nint)outputMatrix;
                nint wAddr = (nint)pW;
                nint bAddr = (nint)bias;

                System.Threading.Tasks.Parallel.For(0, t, ti =>
                {
                    float* inRow = (float*)inAddr + (nuint)ti * (nuint)inDim;
                    float* outRow = (float*)outAddr + (nuint)ti * (nuint)outDim;
                    ushort* wBase = (ushort*)wAddr;
                    float* b = (float*)bAddr;

                    for (int o = 0; o < outDim; o++)
                    {
                        float val = F16CNative.Dot(inRow, wBase + (nuint)o * (nuint)inDim, inDim);
                        if (b != null) val += b[o];
                        outRow[o] = val;
                    }
                });
            }
            return;
        }

        fixed (float* w = _f32)
        {
            for (int ti = 0; ti < t; ti++)
            {
                float* inRow = inputMatrix + (nuint)ti * (nuint)inDim;
                float* outRow = outputMatrix + (nuint)ti * (nuint)outDim;
                SimdKernels.MatVecF32(outRow, w, inRow, outDim, inDim);
                if (bias != null)
                {
                    for (int o = 0; o < outDim; o++) outRow[o] += bias[o];
                }
            }
        }
    }

    /// <summary>Single-row linear layer: output[outDim] = weight[outDim, inDim] . input[inDim] (no bias -- callers add bias separately, matching CfmUNetKernels' existing convention).</summary>
    public unsafe float[] MatVec(float[] input)
    {
        var output = new float[_outDim];
        int inDim = _inDim, outDim = _outDim;

        if (_f16Bits is { } f16)
        {
            fixed (float* pIn = input)
            fixed (short* pW = f16)
            {
                ushort* wBase = (ushort*)pW;
                if (outDim >= 64)
                {
                    nint inAddr = (nint)pIn, wAddr = (nint)wBase;
                    System.Threading.Tasks.Parallel.For(0, outDim, o =>
                    {
                        ushort* wRow = (ushort*)wAddr + (nuint)o * (nuint)inDim;
                        output[o] = F16CNative.Dot((float*)inAddr, wRow, inDim);
                    });
                }
                else
                {
                    for (int o = 0; o < outDim; o++)
                        output[o] = F16CNative.Dot(pIn, wBase + (nuint)o * (nuint)inDim, inDim);
                }
            }
            return output;
        }

        fixed (float* w = _f32, x = input, y = output)
            SimdKernels.MatVecF32(y, w, x, outDim, inDim);
        return output;
    }
}
