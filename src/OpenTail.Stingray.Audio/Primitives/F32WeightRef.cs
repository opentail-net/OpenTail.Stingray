namespace OpenTail.Stingray.Audio.Primitives;

/// <summary>Managed F32 matrix wrapper for callers that need an unquantized precision control.</summary>
public sealed class F32WeightRef : IQuantWeightRef
{
    private readonly float[] _data;

    public F32WeightRef(float[] data) => _data = data;

    public unsafe float[] MatVec(float[] input, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* wp = _data, xp = input, op = output)
            OpenTail.Stingray.Cpu.SimdKernels.MatVecF32(op, wp, xp, outDim, inDim);
        return output;
    }
}
