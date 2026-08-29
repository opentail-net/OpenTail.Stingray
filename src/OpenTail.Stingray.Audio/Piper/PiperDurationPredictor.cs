
namespace OpenTail.Stingray.Audio.Piper;

/// <summary>
/// Piper's VITS StochasticDurationPredictor (dp), inference/reverse path only. Ported from the
/// reference VITS `models.py StochasticDurationPredictor.forward(reverse=True)`: a normalizing
/// flow over a 2-channel noise sample, conditioned on the TextEncoder's pre-proj hidden state.
/// Math (DDSConv/ConvFlow/spline/Flip/ElementwiseAffine) lives in
/// <see cref="VitsDurationFlowKernels"/>, shared with MeloTTS's `sdp`.
///
/// Confirmed against the real ONNX graph (not just recollection of the reference): the exported
/// inference graph only exercises 8 of the 9 flow-list entries -- ConvFlow at list-index 1 is
/// fully pruned (no nodes/initializers reference `dp.flows.1.*`), matching the reference's
/// `flows = list(reversed(self.flows)); flows = flows[:-2] + [flows[-1]]` "remove a useless
/// vflow" trick. The real execution order, confirmed via each stage's Split node's input chain,
/// is: Flip(idx8) -&gt; ConvFlow(idx7) -&gt; Flip(idx6) -&gt; ConvFlow(idx5) -&gt; Flip(idx4) -&gt;
/// ConvFlow(idx3) -&gt; Flip(idx2) -&gt; ElementwiseAffine(idx0).
/// </summary>
public static class PiperDurationPredictor
{
    /// <summary>
    /// encoderHidden is the TextEncoder's pre-proj output, channel-first [hidden, T]. Returns
    /// logw (pre-exp predicted log-duration), length T. noise must be a [2*T] array of raw
    /// N(0,1) draws (channel-first [2,T]) -- caller supplies these (real RNG in production,
    /// golden-captured values for verification).
    /// </summary>
    public static float[] Predict(PiperOnnxWeights w, float[] encoderHidden, int t, float[] noise, float noiseScaleW)
    {
        int dim = w.HiddenDim;

        // x = proj(convs(pre(encoderHidden))) -- the shared conditioning context `g` for every flow.
        var x = VitsAttentionKernels.Conv1x1(encoderHidden, dim, t, w.DpPreWeight, w.DpPreBias, dim);
        x = VitsDurationFlowKernels.DDSConv(x, dim, t, w.DpConvs);
        x = VitsAttentionKernels.Conv1x1(x, dim, t, w.DpProjWeight, w.DpProjBias, dim);

        // z = noise * noise_w, channel-first [2, T].
        var z = new float[2 * t];
        for (int i = 0; i < z.Length; i++) z[i] = noise[i] * noiseScaleW;

        // Flip(8) -> ConvFlow(7) -> Flip(6) -> ConvFlow(5) -> Flip(4) -> ConvFlow(3) -> Flip(2) -> ElementwiseAffine(0)
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.DpFlow7);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.DpFlow5);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ConvFlowReverse(z, t, x, dim, w.DpFlow3);
        z = VitsDurationFlowKernels.Flip(z, t);
        z = VitsDurationFlowKernels.ElementwiseAffineReverse(z, t, w.DpFlow0M, w.DpFlow0ExpNegLogs);

        // logw = z[0, :] (z0); z[1, :] is discarded (matches torch.split(z, [1,1], 1)).
        var logw = new float[t];
        Array.Copy(z, 0, logw, 0, t);
        return logw;
    }
}
