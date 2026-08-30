
namespace OpenTail.Stingray.Audio.MmsTts;

/// <summary>
/// MMS-TTS's ResidualCouplingBlock (`flow`), inference/reverse path only. Same math as
/// <see cref="OpenTail.Stingray.Audio.Piper.PiperFlow"/> (both real VITS `ResidualCouplingBlock`
/// implementations, gin_channels=0 for this single-speaker checkpoint -- confirmed via
/// `speaker_embedding_size: 0`/`num_speakers: 1` in the real config.json), adapted for this
/// checkpoint's own weight field names (<see cref="MmsTtsWeights"/>).
///
/// Real reference (`VitsResidualCouplingBlock.forward(reverse=True)`,
/// `transformers/models/vits/modeling_vits.py`): `for flow in reversed(self.flows): hidden_states,
/// _ = flow(hidden_states, ..., reverse=True)` where `self.flows` interleaves
/// `VitsResidualCouplingLayer` and `VitsFlip` (4 of each, `prior_encoder_num_flows=4` in
/// config.json). HuggingFace does not store the parameter-free Flip modules in the safetensors
/// checkpoint (only `flow.flows.0..3` = the 4 real coupling layers), so the real reversed
/// execution order is: Flip -&gt; Layer(3) -&gt; Flip -&gt; Layer(2) -&gt; Flip -&gt; Layer(1) -&gt; Flip -&gt;
/// Layer(0) -- same pattern as Piper's own confirmed order, just indices 0..3 instead of
/// Piper's ONNX-numbered 0,2,4,6.
/// </summary>
public static class MmsTtsFlow
{
    /// <summary>zp is the length-regulator-expanded latent, channel-first [hiddenDim, T]. Returns z, channel-first [hiddenDim, T].</summary>
    public static float[] Reverse(MmsTtsWeights w, float[] zp, int t)
    {
        int dim = w.HiddenDim;
        var z = zp;

        for (int i = MmsTtsWeights.FlowWnLayers - 1; i >= 0; i--)
        {
            z = Flip(z, dim, t);
            z = CouplingLayerReverse(z, t, w.FlowLayers[i], w.FlowHalfChannels);
        }

        return z;
    }

    private static float[] Flip(float[] z, int dim, int t)
    {
        var output = new float[dim * t];
        for (int c = 0; c < dim; c++)
            Array.Copy(z, (dim - 1 - c) * t, output, c * t, t);
        return output;
    }

    private static float[] CouplingLayerReverse(float[] z, int t, MmsCouplingLayerWeights lw, int half)
    {
        var x0 = new float[half * t];
        var x1 = new float[half * t];
        Array.Copy(z, 0, x0, 0, half * t);
        Array.Copy(z, half * t, x1, 0, half * t);

        var h = VitsAttentionKernels.Conv1x1(x0, half, t, lw.PreWeight, lw.PreBias, half * 2);
        h = WnForward(h, t, lw.Wavenet, half * 2);
        var m = VitsAttentionKernels.Conv1x1(h, half * 2, t, lw.PostWeight, lw.PostBias, half);

        var x1Out = new float[half * t];
        for (int i = 0; i < x1Out.Length; i++) x1Out[i] = x1[i] - m[i];

        var output = new float[2 * half * t];
        Array.Copy(x0, 0, output, 0, half * t);
        Array.Copy(x1Out, 0, output, half * t, half * t);
        return output;
    }

    /// <summary>WN (WaveNet-style dilated conv stack), gin_channels=0. 4 layers, kernel=5, dilation=1 constant (real `wavenet_kernel_size=5`/`wavenet_dilation_rate=1` from config.json).</summary>
    private static float[] WnForward(float[] x, int t, MmsWnWeights wn, int hidden)
    {
        const int kernel = MmsTtsWeights.FlowWnKernel;
        const int dilation = MmsTtsWeights.FlowWnDilation;

        var output = new float[hidden * t];
        var cur = x;

        for (int layer = 0; layer < MmsTtsWeights.FlowWnLayers; layer++)
        {
            var xIn = HifiGanKernels.Conv1dDilated(cur, hidden, t, wn.InWeight[layer], wn.InBias[layer], 2 * hidden, kernel, dilation);

            var acts = new float[hidden * t];
            var gateActs = new float[hidden * t];
            System.Numerics.Tensors.TensorPrimitives.Tanh(xIn.AsSpan(0, hidden * t), acts);
            System.Numerics.Tensors.TensorPrimitives.Sigmoid(xIn.AsSpan(hidden * t, hidden * t), gateActs);
            System.Numerics.Tensors.TensorPrimitives.Multiply(acts, gateActs, acts);

            bool isLast = layer == MmsTtsWeights.FlowWnLayers - 1;
            int resSkipOutCh = isLast ? hidden : 2 * hidden;
            var resSkip = VitsAttentionKernels.Conv1x1(acts, hidden, t, wn.ResSkipWeight[layer], wn.ResSkipBias[layer], resSkipOutCh);

            if (!isLast)
            {
                var next = new float[hidden * t];
                for (int i = 0; i < hidden * t; i++) next[i] = cur[i] + resSkip[i];
                for (int i = 0; i < hidden * t; i++) output[i] += resSkip[hidden * t + i];
                cur = next;
            }
            else
            {
                for (int i = 0; i < hidden * t; i++) output[i] += resSkip[i];
            }
        }

        return output;
    }
}
