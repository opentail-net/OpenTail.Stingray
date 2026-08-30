
namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// MMS-TTS's StochasticDurationPredictor (`duration_predictor`), inference/reverse path only.
/// Same math as <see cref="OpenTail.Stingray.Audio.Piper.PiperDurationPredictor"/> (shared
/// <see cref="VitsDurationFlowKernels"/>), adapted for this checkpoint's own flow-list length.
///
/// <para><b>Flow list structure (derived from the real reference's construction + real reverse-
/// mode pruning logic, cross-checked against Piper's own ONNX-graph-confirmed real execution
/// order for the same reference lineage -- NOT yet independently golden-verified end-to-end for
/// MMS specifically, only the text-encoder stage has been; verify this stage's own numeric output
/// against `scratch-llamacpp-ref/mms_tts_golden.py`'s dumped `logw`/durations before trusting it
/// blindly)</b>: HuggingFace stores 5 `duration_predictor.flows.N` entries (indices 0..4) --
/// unlike Piper's ONNX export, HF does not number the parameter-free Flip modules into the flow
/// list, so index 0 = ElementwiseAffine, indices 1..4 = the four ConvFlow blocks (construction
/// order). The real VITS reference builds `self.flows = [ElementwiseAffine()] + interleave(
/// ConvFlow, Flip) * 4`, then at inference reverses the list and drops the FIRST ConvFlow
/// (`flows[:-2] + [flows[-1]]` on the reversed list -- "remove a useless vflow") -- so
/// `duration_predictor.flows.1` (the first-constructed ConvFlow) is the one dropped, matching
/// Piper's own confirmed-via-ONNX-node-inspection pruning of its equivalent first ConvFlow.
/// Real execution order: Flip -&gt; ConvFlow(flows.4) -&gt; Flip -&gt; ConvFlow(flows.3) -&gt; Flip -&gt;
/// ConvFlow(flows.2) -&gt; Flip -&gt; ElementwiseAffine(flows.0).</para>
/// </summary>
public static class MmsTtsDurationPredictor
{
    /// <summary>
    /// encoderHidden is the TextEncoder's pre-proj output, channel-first [hidden, T]. Returns logw
    /// (pre-exp predicted log-duration), length T. noise must be a [2*T] array of raw N(0,1) draws
    /// (channel-first [2,T]).
    /// </summary>
    public static float[] Predict(MmsTtsWeights w, float[] encoderHidden, int t, float[] noise, float noiseScaleW)
    {
        int dim = w.HiddenDim;

        var x = VitsAttentionKernels.Conv1x1(encoderHidden, dim, t, w.DpConvPreWeight, w.DpConvPreBias, dim);
        x = VitsDurationFlowKernels.DDSConv(x, dim, t, w.DpConvDds);
        x = VitsAttentionKernels.Conv1x1(x, dim, t, w.DpConvProjWeight, w.DpConvProjBias, dim);

        var z = new float[2 * t];
        System.Numerics.Tensors.TensorPrimitives.Multiply(noise.AsSpan(0, 2 * t), noiseScaleW, z);

        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.DpFlow4);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.DpFlow3);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.DpFlow2);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ElementwiseAffineReverse(z, t, w.DpFlow0Translate, w.DpFlow0ExpNegLogScale);

        var logw = new float[t];
        Array.Copy(z, 0, logw, 0, t);
        return logw;
    }
}
