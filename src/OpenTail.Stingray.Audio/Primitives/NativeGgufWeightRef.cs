using OpenTail.Stingray.Core;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// A weight matrix read directly off a memory-mapped GGUF file's raw bytes -- no dequantization,
/// no copy, no re-quantization -- paired with <see cref="SimdKernels.MatVec"/>'s existing
/// per-dtype dispatch (F32/F16/BF16/Q8_0/Q4_K/Q6_K/Q5_K/Q3_K/Q2_K/Q4_0/IQ4_NL/...). Mirrors
/// `OpenTail.Stingray.Vision`'s `VisionTensorRef`/`VisionOps.GetTensor`/`MatVecAny` pattern,
/// applied to Audio.
///
/// <para><b>Where this genuinely wins (unlike the earlier, reverted Fish Speech attempt)</b>: the
/// real community GGUF conversion of `parler-tts/parler-tts-mini-v1`
/// (`ecyht2/parler-tts-mini-v1-GGUF`) stores its decoder's big matrices as REAL on-disk Q8_0 --
/// not a K-quant like Fish Speech's default checkpoint, which measured slower to decode per-
/// element despite being smaller (see docs/audio-review-progress.md). Reading Q8_0 natively here
/// has no compute-cost tradeoff to weigh against the bandwidth savings: it's the exact same
/// `DotQ8_0` kernel this codebase already uses for its OWN re-quantized Q8_0 weights, just without
/// the float32 round-trip Safetensors-sourced weights require at load time.</para>
/// </summary>
public readonly unsafe struct NativeGgufWeightRef : IQuantWeightRef
{
    public readonly byte* Data;
    public readonly DType Dtype;

    private NativeGgufWeightRef(byte* data, DType dtype)
    {
        Data = data;
        Dtype = dtype;
    }

    /// <summary>Resolves a tensor's raw data pointer and real on-disk dtype directly from the GGUF's memory mapping -- no copy.</summary>
    public static NativeGgufWeightRef Get(GgufModel model, string name)
    {
        var info = model.FindTensor(name) ?? throw new System.IO.InvalidDataException($"GGUF missing required tensor '{name}'.");
        return new NativeGgufWeightRef(model.GetTensorDataPtr(info), info.DType);
    }

    public float[] MatVec(float[] input, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* xp = input, op = output)
        {
            SimdKernels.MatVec(op, Data, xp, outDim, inDim, Dtype);
        }
        return output;
    }
}
