namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>
/// A weight matrix consumable by a dtype-specific fused mat-vec kernel, abstracting over WHERE the
/// bytes live -- a Q8_0-quantized managed copy (see <see cref="Q8_0WeightQuantizer"/>, used when
/// the source file is plain float32, e.g. Safetensors) or a pointer straight into a memory-mapped
/// GGUF file already carrying its own real on-disk dtype (see <see cref="NativeGgufWeightRef"/>).
/// Letting a decoder's forward-pass code depend on this interface instead of either concrete type
/// means the SAME forward-pass implementation (KV cache, attention, FFN, etc.) works unchanged
/// whichever loader produced the weights -- see <c>ParlerDecoderWeights</c>'s two constructors.
/// </summary>
public interface IQuantWeightRef
{
    /// <summary>Real dtype-dispatching mat-vec: <c>output[outDim] = weight[outDim, inDim] . input[inDim]</c>, no bias.</summary>
    float[] MatVec(float[] input, int inDim, int outDim);
}
