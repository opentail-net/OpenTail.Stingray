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

    /// <summary>Converts a plain F32 weight matrix to F16 ONCE at construction time and dispatches
    /// every subsequent call through the native F16C kernel (falling back to F32 automatically when
    /// <see cref="F16CNative.IsAvailable"/> is false). Deliberately a SEPARATE, explicitly-opted-into
    /// factory from <see cref="FromF32"/> rather than a change to that method's default behavior:
    /// <see cref="FromF32"/> itself briefly did this automatically (`ad98570`), then was reverted
    /// back to plain F32 (`f54e907`, "chatterbox speed improvement") for every caller of this shared
    /// class at once, with no written rationale beyond a same-commit change that started running a
    /// CFM decoder's CFG-conditional/unconditional branches in parallel via `Parallel.Invoke` -- the
    /// three `.wav` files committed alongside that revert suggest a real audible A/B listening
    /// comparison drove it, plausibly specific to CFM's iterative multi-step flow-matching solve
    /// (where small per-step precision loss can compound across steps) rather than to F16C itself.
    /// Given that ambiguity, this method exists so a genuinely single-pass encoder architecture
    /// (already proven safe for Whisper's/QwenASR's/CosyVoice2's encoders before the revert) can opt
    /// back into the technique explicitly, without silently re-enabling it for CFM/flow-matching
    /// callers that may have been reverted for a real, undocumented quality reason. Verify real
    /// correctness AND performance before keeping any use of this at a new call site -- do not
    /// assume it is safe just because it worked elsewhere.</summary>
    public static CfmLinearWeight FromF32WithF16Conversion(float[] weightF32, int outDim, int inDim)
    {
        if (!F16CNative.IsAvailable)
            return new CfmLinearWeight(null, weightF32, outDim, inDim);

        var f16Bits = new short[weightF32.Length];
        for (int i = 0; i < weightF32.Length; i++)
            f16Bits[i] = unchecked((short)BitConverter.HalfToUInt16Bits((Half)weightF32[i]));
        return new CfmLinearWeight(f16Bits, null, outDim, inDim);
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
                Cpu.SimdKernels.DotF32_2In(in0, in1, wRow, inDim, out float v0, out float v1);
                if (b != null) { v0 += b[r]; v1 += b[r]; }
                out0[r] = v0;
                out1[r] = v1;
            }
        }
    }

    /// <summary>Batch-of-2 fused QKV projection: projects input0 and input1 through Q, K, and V weight matrices simultaneously.
    /// Reduces thread pool dispatches from 3 to 1 and keeps the input vectors hot in cache across all 3 projections.</summary>
    public static unsafe void MatMulQkvPairRowMajor(
        CfmLinearWeight qWeight, CfmLinearWeight kWeight, CfmLinearWeight vWeight,
        float* input0, float* input1,
        float* qOut0, float* qOut1,
        float* kOut0, float* kOut1,
        float* vOut0, float* vOut1)
    {
        if (qWeight._f32 is not { } qw || kWeight._f32 is not { } kw || vWeight._f32 is not { } vw)
            throw new NotSupportedException("MatMulQkvPairRowMajor requires F32-backed weights.");

        int inDim = qWeight._inDim;
        int outDim = qWeight._outDim;
        if (kWeight._inDim != inDim || kWeight._outDim != outDim || vWeight._inDim != inDim || vWeight._outDim != outDim)
            throw new ArgumentException("Q, K, V dimensions must match.");

        fixed (float* qwp = qw, kwp = kw, vwp = vw)
        {
            int numThreads = Math.Min(Environment.ProcessorCount, (outDim + 63) / 64);
            if (numThreads <= 1)
            {
                RunRowsThreadLocal(qwp, kwp, vwp, input0, input1, qOut0, qOut1, kOut0, kOut1, vOut0, vOut1, inDim, 0, outDim);
                return;
            }

            int chunkSize = (outDim + numThreads - 1) / numThreads;
            nint qwAddr = (nint)qwp, kwAddr = (nint)kwp, vwAddr = (nint)vwp;
            nint i0Addr = (nint)input0, i1Addr = (nint)input1;
            nint qo0Addr = (nint)qOut0, qo1Addr = (nint)qOut1;
            nint ko0Addr = (nint)kOut0, ko1Addr = (nint)kOut1;
            nint vo0Addr = (nint)vOut0, vo1Addr = (nint)vOut1;

            System.Threading.Tasks.Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(outDim, start + chunkSize);
                RunRowsThreadLocal((float*)qwAddr, (float*)kwAddr, (float*)vwAddr,
                                   (float*)i0Addr, (float*)i1Addr,
                                   (float*)qo0Addr, (float*)qo1Addr,
                                   (float*)ko0Addr, (float*)ko1Addr,
                                   (float*)vo0Addr, (float*)vo1Addr,
                                   inDim, start, end);
            });

            static void RunRowsThreadLocal(
                float* qWeights, float* kWeights, float* vWeights,
                float* in0, float* in1,
                float* qo0, float* qo1,
                float* ko0, float* ko1,
                float* vo0, float* vo1,
                int inDim, int start, int end)
            {
                for (int r = start; r < end; r++)
                {
                    float* qRow = qWeights + (long)r * inDim;
                    Cpu.SimdKernels.DotF32_2In(in0, in1, qRow, inDim, out float qv0, out float qv1);
                    qo0[r] = qv0;
                    qo1[r] = qv1;

                    float* kRow = kWeights + (long)r * inDim;
                    Cpu.SimdKernels.DotF32_2In(in0, in1, kRow, inDim, out float kv0, out float kv1);
                    ko0[r] = kv0;
                    ko1[r] = kv1;

                    float* vRow = vWeights + (long)r * inDim;
                    Cpu.SimdKernels.DotF32_2In(in0, in1, vRow, inDim, out float vv0, out float vv1);
                    vo0[r] = vv0;
                    vo1[r] = vv1;
                }
            }
        }
    }

    /// <summary>Batch-of-2 fused Gate + Up + SiLU projection for SwiGLU MLP: computes gate and up projections
    /// simultaneously and writes `Silu(gate) * up` directly to output0 and output1 in a single pass.</summary>
    public static unsafe void MatMulGateUpSiluPairRowMajor(
        CfmLinearWeight gateWeight, CfmLinearWeight upWeight,
        float* input0, float* input1,
        float* output0, float* output1)
    {
        if (gateWeight._f32 is not { } gw || upWeight._f32 is not { } uw)
            throw new NotSupportedException("MatMulGateUpSiluPairRowMajor requires F32-backed weights.");

        int inDim = gateWeight._inDim;
        int outDim = gateWeight._outDim;
        if (upWeight._inDim != inDim || upWeight._outDim != outDim)
            throw new ArgumentException("Gate and Up dimensions must match.");

        fixed (float* gwp = gw, uwp = uw)
        {
            int numThreads = Math.Min(Environment.ProcessorCount, (outDim + 63) / 64);
            if (numThreads <= 1)
            {
                RunRowsThreadLocal(gwp, uwp, input0, input1, output0, output1, inDim, 0, outDim);
                return;
            }

            int chunkSize = (outDim + numThreads - 1) / numThreads;
            nint gwAddr = (nint)gwp, uwAddr = (nint)uwp;
            nint i0Addr = (nint)input0, i1Addr = (nint)input1;
            nint o0Addr = (nint)output0, o1Addr = (nint)output1;

            System.Threading.Tasks.Parallel.For(0, numThreads, t =>
            {
                int start = t * chunkSize;
                int end = Math.Min(outDim, start + chunkSize);
                RunRowsThreadLocal((float*)gwAddr, (float*)uwAddr,
                                   (float*)i0Addr, (float*)i1Addr,
                                   (float*)o0Addr, (float*)o1Addr,
                                   inDim, start, end);
            });

            static void RunRowsThreadLocal(
                float* gWeights, float* uWeights,
                float* in0, float* in1,
                float* out0, float* out1,
                int inDim, int start, int end)
            {
                for (int r = start; r < end; r++)
                {
                    float* gRow = gWeights + (long)r * inDim;
                    Cpu.SimdKernels.DotF32_2In(in0, in1, gRow, inDim, out float g0, out float g1);

                    float* uRow = uWeights + (long)r * inDim;
                    Cpu.SimdKernels.DotF32_2In(in0, in1, uRow, inDim, out float u0, out float u1);

                    out0[r] = (g0 / (1f + MathF.Exp(-g0))) * u0;
                    out1[r] = (g1 / (1f + MathF.Exp(-g1))) * u1;
                }
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

    /// <summary>Returns this weight's data as a plain float32 array, reconstructing from F16 bits
    /// if that's how it was stored (read-only, additive accessor -- does not change this
    /// instance's own CPU dispatch path or precision, only exposes the value for a caller that
    /// needs raw float32, e.g. uploading to a GPU-resident tensor via <c>backend.Upload</c>).</summary>
    public float[] ToF32Array()
    {
        if (_f32 is { } f32) return f32;
        var f16 = _f16Bits!;
        var result = new float[f16.Length];
        for (int i = 0; i < f16.Length; i++)
            result[i] = (float)BitConverter.UInt16BitsToHalf(unchecked((ushort)f16[i]));
        return result;
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
