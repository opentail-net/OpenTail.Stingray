
namespace OpenTail.Stingray.Audio.MeloTTS;

/// <summary>
/// MeloTTS's VITS2 normalizing flow (`flow`): `TransformerCouplingBlock`, NOT Piper's plain
/// `ResidualCouplingBlock` -- 4 `TransformerCouplingLayer`s interleaved with parameter-free
/// `Flip`s (8 sub-flows total: TCL0,Flip,TCL1,Flip,TCL2,Flip,TCL3,Flip), each coupling layer
/// carrying its own internal 3-layer relative-attention encoder (`enc`, reusing
/// <see cref="MeloRelativeEncoder"/>) and its own speaker-conditioning `spk_emb_linear`. Ported
/// from `examples/melotts-py/models.py`'s `TransformerCouplingBlock.forward(reverse=True)` and
/// `modules.py`'s `TransformerCouplingLayer.forward(reverse=True)` / `Flip.forward`.
///
/// Inference only ever runs the REVERSE direction (`SynthesizerTrn.infer`: `z = self.flow(z_p,
/// y_mask, g=g, reverse=True)`), so only that direction is implemented here.
///
/// `mean_only=True` for every coupling layer in this checkpoint (models.py line 139), so `logs`
/// is implicitly all-zero and the reverse update simplifies from `(x1-m)*exp(-logs)` to plain
/// `x1-m` -- confirmed against `TransformerCouplingLayer.__init__`'s `mean_only=True` call site
/// and `post`'s output channel count (half_channels, not half_channels*2).
/// </summary>
public static class MeloFlow
{
    /// <summary>
    /// zP is the frame-rate prior-latent sample (channel-first [HiddenDim, T], i.e. `m_p +
    /// exp(logs_p) * noise * noise_scale` after length-regulation -- NOT yet produced by this
    /// codebase, see docs/audio-review-progress.md; the accompanying test feeds a golden-dumped
    /// zP directly to isolate flow correctness from length-regulator correctness). g is the
    /// speaker embedding [GinChannels] (`emb_g[speakerId]`). Returns z, same shape as zP.
    /// </summary>
    public static float[] Reverse(MeloOnnxWeights w, float[] zP, int t, float[] g)
    {
        int channels = w.HiddenDim;
        int half = channels / 2;

        var z = (float[])zP.Clone();

        // TransformerCouplingBlock.forward(reverse=True): for flow in reversed(self.flows).
        // self.flows = [TCL0,Flip,TCL1,Flip,TCL2,Flip,TCL3,Flip] -> reversed:
        // Flip,TCL3,Flip,TCL2,Flip,TCL1,Flip,TCL0 -- 4 Flips and 4 TCLs alternating, ending on
        // TCL0 with NO trailing Flip (confirmed via golden bisection: an earlier version had a
        // spurious 5th Flip after this loop, which passed every per-TCL golden checkpoint but
        // then flipped the final returned tensor's channel order -- caught by comparing the
        // actual `Reverse()` return value against golden, not just the loop's intermediates).
        for (int idx = w.FlowLayers.Length - 1; idx >= 0; idx--)
        {
            z = Flip(z, channels, t);
            z = CouplingLayerReverse(w, w.FlowLayers[idx], z, t, half, channels, g);
        }
        return z;
    }

    private static float[] CouplingLayerReverse(
        MeloOnnxWeights w, MeloFlowLayerWeights fw, float[] x, int t, int half, int channels, float[] g)
    {
        var x0 = new float[half * t];
        Array.Copy(x, 0, x0, 0, half * t);

        var h = VitsAttentionKernels.Conv1x1(x0, half, t, fw.PreWeight, fw.PreBias, channels);

        var spkProjected = MeloRelativeEncoder.LinearVec(g, fw.SpkEmbLinearWeight, fw.SpkEmbLinearBias, channels, w.GinChannels);
        h = MeloRelativeEncoder.Forward(h, t, channels, w.NumHeads, w.WindowSize, MeloOnnxWeights.FlowFfnKernel,
            fw.Layers, w.SpkEmbCondLayerIdx, spkProjected);

        // post: hidden -> half_channels (mean_only=True, so this IS m; logs is implicitly zero).
        var m = VitsAttentionKernels.Conv1x1(h, channels, t, fw.PostWeight, fw.PostBias, half);

        var output = new float[channels * t];
        Array.Copy(x0, 0, output, 0, half * t);
        System.Numerics.Tensors.TensorPrimitives.Subtract(x.AsSpan(half * t, half * t), m, output.AsSpan(half * t, half * t));
        return output;
    }

    /// <summary>torch.flip(x, [1]): full channel-order reversal (c -> channels-1-c), NOT the
    /// 2-channel swap in <see cref="VitsDurationFlowKernels.Flip"/> (that one is specific to the
    /// SDP's 2-channel flow; this flow has `channels` (192) channels).</summary>
    private static float[] Flip(float[] x, int channels, int t)
    {
        var output = new float[channels * t];
        for (int c = 0; c < channels; c++)
            Array.Copy(x, (channels - 1 - c) * t, output, c * t, t);
        return output;
    }
}
