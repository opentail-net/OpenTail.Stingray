using OpenTail.Stingray.Audio.Primitives;
using OpenTail.Stingray.Cpu;

namespace OpenTail.Stingray.Audio.VoxCpm2;

public readonly struct VoxCpm2StepProjectionOutput(
    float[] fsqHidden, float[] currentResidualInput, float[] residualInput,
    float[] currentLmDitHidden, float[] fsqLmDitHidden, float[] residualDitHidden,
    float[] currentStopLogits, float[] fsqStopLogits)
{
    public float[] FsqHidden { get; } = fsqHidden;
    public float[] CurrentResidualInput { get; } = currentResidualInput;
    public float[] ResidualInput { get; } = residualInput;
    public float[] CurrentLmDitHidden { get; } = currentLmDitHidden;
    public float[] FsqLmDitHidden { get; } = fsqLmDitHidden;
    public float[] ResidualDitHidden { get; } = residualDitHidden;
    public float[] CurrentStopLogits { get; } = currentStopLogits;
    public float[] FsqStopLogits { get; } = fsqStopLogits;
}

/// <summary>
/// Real forward pass for VoxCPM2's per-step fusion/projection layers, ported from
/// `generator.cpp`'s `VoxCPM2StepProjectionRuntime::Impl::build` (not guessed). Real FSQ (finite
/// scalar quantization) bottleneck: `Linear(hidden-&gt;latentDim) -&gt; tanh -&gt; *scale -&gt; round
/// -&gt; /scale -&gt; Linear(latentDim-&gt;hidden)` -- a real straight-through quantization of the LM
/// hidden state into `scalarQuantizationScale`-many discrete levels per latent dimension, used by
/// the "residual LM" path (this session has not yet ported the residual LM itself, only this
/// shared projection math). Every other output is a plain `Linear` (with real bias), several
/// REUSING the same learned weight for both the raw LM hidden state and its FSQ-quantized
/// counterpart (not two separate learned projections) -- confirmed by the reference passing the
/// same `proj.fusion_concat_proj`/`proj.lm_to_dit_proj`/`proj.stop_proj`/`proj.stop_head` weight
/// objects to two different call sites.
/// </summary>
public static class VoxCpm2StepProjection
{
    public static VoxCpm2StepProjectionOutput Run(
        VoxCpm2StepProjectionWeights w,
        float[] lmHidden, float[] residualHidden, float[] currentEmbed,
        int hiddenDim, int ditHiddenDim, int latentDim, int scalarQuantizationScale)
    {
        var fsq = Linear(lmHidden, w.FsqInProjWeight, w.FsqInProjBias, hiddenDim, latentDim);
        for (int i = 0; i < fsq.Length; i++)
        {
            float v = MathF.Tanh(fsq[i]) * scalarQuantizationScale;
            fsq[i] = MathF.Round(v) / scalarQuantizationScale;
        }
        fsq = Linear(fsq, w.FsqOutProjWeight, w.FsqOutProjBias, latentDim, hiddenDim);

        var currentConcat = Concat(lmHidden, currentEmbed);
        var currentResidualInput = Linear(currentConcat, w.FusionConcatProjWeight, w.FusionConcatProjBias, hiddenDim * 2, hiddenDim);

        var residualConcat = Concat(fsq, currentEmbed);
        var residualInput = Linear(residualConcat, w.FusionConcatProjWeight, w.FusionConcatProjBias, hiddenDim * 2, hiddenDim);

        var currentLmDit = Linear(lmHidden, w.LmToDitProjWeight, w.LmToDitProjBias, hiddenDim, ditHiddenDim);
        var fsqLmDit = Linear(fsq, w.LmToDitProjWeight, w.LmToDitProjBias, hiddenDim, ditHiddenDim);
        var residualDit = Linear(residualHidden, w.ResToDitProjWeight, w.ResToDitProjBias, hiddenDim, ditHiddenDim);

        var currentStop = Linear(lmHidden, w.StopProjWeight, w.StopProjBias, hiddenDim, hiddenDim);
        SiluInPlace(currentStop);
        var currentStopLogits = Linear(currentStop, w.StopHeadWeight, bias: [], hiddenDim, 2);

        var fsqStop = Linear(fsq, w.StopProjWeight, w.StopProjBias, hiddenDim, hiddenDim);
        SiluInPlace(fsqStop);
        var fsqStopLogits = Linear(fsqStop, w.StopHeadWeight, bias: [], hiddenDim, 2);

        return new VoxCpm2StepProjectionOutput(
            fsq, currentResidualInput, residualInput,
            currentLmDit, fsqLmDit, residualDit,
            currentStopLogits, fsqStopLogits);
    }

    private static float[] Concat(float[] a, float[] b)
    {
        var output = new float[a.Length + b.Length];
        Array.Copy(a, output, a.Length);
        Array.Copy(b, 0, output, a.Length, b.Length);
        return output;
    }

    private static unsafe float[] Linear(float[] input, float[] weight, float[] bias, int inDim, int outDim)
    {
        var output = new float[outDim];
        fixed (float* outPtr = output, wPtr = weight, inPtr = input)
        {
            if (bias.Length > 0)
            {
                fixed (float* bPtr = bias)
                {
                    SimdKernels.MatVecF32(outPtr, wPtr, bPtr, inPtr, outDim, inDim);
                }
            }
            else
            {
                SimdKernels.MatVecF32(outPtr, wPtr, null, inPtr, outDim, inDim);
            }
        }
        return output;
    }

    private static void SiluInPlace(float[] x) => DenseKernels.SiluInPlace(x);
}
