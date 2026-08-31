using OpenTail.Stingray.Audio.F5TTS;

namespace OpenTail.Stingray.Audio.CosyVoice;

/// <summary>
/// CosyVoice3's flow-matching DiT backbone. Ports F5-TTS's `DiTBlock`/`DiT.forward` math
/// directly (`F5TTS/F5DiTBlock.cs`, `F5TTS/F5DiTModel.cs` -- real, golden-verified against the
/// actual PyTorch reference) since the two are tensor-for-tensor architecturally identical
/// (confirmed this session, see `CosyVoice3DiTWeights.cs`'s doc comment) -- reuses
/// `F5TTS.F5Kernels`/`F5TTS.F5RotaryEmbedding` directly (already pipeline-agnostic static
/// utilities) rather than re-deriving the same Linear/SiLU/LayerNorm/RoPE math a third time.
///
/// <para><b>Input composition confirmed directly from `examples/cosyvoice.cpp`</b>
/// (`InputEmbedding::build_cgraph`, `cosyvoice-graph.cpp` line ~278, and its caller
/// `DiT::build_cgraph` line ~446): `x = concat(x, cond, text_embed, spks)` in exactly that
/// order, where `text_embed` is CosyVoice's own repurposing of a parameter slot inherited
/// from a text-conditioned DiT template -- it actually receives `mu` (the flow's own
/// upsampled 80-dim speech-token embedding from `CausalMaskedDiffWithDiT::
/// build_cgraph_encode`'s `pre_lookahead_layer` + 2x upsample, NOT a text embedding at all;
/// CosyVoice3 has no Conformer flow encoder, see this class's earlier doc history in
/// docs/audio-review-progress.md). `cond` is the padded reference/prompt mel (`conds` in that
/// same function). `spks` is `spk_embed_affine_layer(l2_norm(campplus_embedding))`,
/// broadcast (`ggml_repeat`) across every frame before concatenation, not per-frame data.
/// All four are `MelDim` (80) wide, concatenating to the confirmed 320-wide `input_embed.
/// proj` input.</para>
/// </summary>
public static class CosyVoice3DiTModel
{
    /// <summary>
    /// Full DiT forward pass. x/cond/mu/spks are channel-last [numFrames, MelDim] (spks is
    /// logically per-utterance but passed pre-broadcast to every frame here, matching
    /// `ggml_repeat`'s effect in the real graph -- callers should broadcast a single [MelDim]
    /// speaker vector across all frames before calling). Returns predicted velocity,
    /// [numFrames, MelDim].
    /// </summary>
    public static float[] ForwardVelocity(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, float timestep, int numFrames, float[] rotaryCos, float[] rotarySin)
    {
        var h = InputEmbed(w, x, cond, mu, spks, numFrames);
        return RunBackbone(w, h, timestep, numFrames, rotaryCos, rotarySin);
    }

    public static float[] ForwardVelocity(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, float timestep, int numFrames)
    {
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(RotaryInvFreq(), numFrames);
        return ForwardVelocity(w, x, cond, mu, spks, timestep, numFrames, rotaryCos, rotarySin);
    }

    /// <summary>
    /// Real Euler-integrated CFM ODE solve (<c>CausalConditionalCFM::build_cgraph_one_step</c>,
    /// <c>get_t_and_dt</c> in <c>cosyvoice-graph.cpp</c>/<c>cosyvoice-loader.cpp</c>): starts from Gaussian
    /// noise and integrates the DiT's predicted velocity field over the real 10-step cosine
    /// schedule <c>t_span[i] = 1 - cos(0.05*pi*i)</c> for <c>i=0..10</c> (11 points, 10 real Euler steps).
    ///
    /// <para>Applies Classifier-Free Guidance (CFG) matching <c>cosyvoice-graph.cpp:529-533</c>:
    /// evaluates both conditional velocity <c>dphiDt</c> (with <paramref name="cond"/>, <paramref name="mu"/>, <paramref name="spks"/>)
    /// and unconditional velocity <c>cfgDphiDt</c> (with zero conditioning), combined via
    /// <c>dphi_dt = (1 + cfg_rate) * dphi_dt - cfg_rate * cfg_dphi_dt</c> with <paramref name="cfgRate"/> (default 0.7).</para>
    /// </summary>
    public static unsafe float[] SolveFlowMatchingOde(CosyVoice3DiTWeights w, float[] cond, float[] mu, float[] spks, int numFrames, int odeSteps, Random rng, float cfgRate = 0.7f)
    {
        int melLen = numFrames * CosyVoice3DiTWeights.MelDim;
        var tSpan = new float[odeSteps + 1];
        for (int i = 0; i <= odeSteps; i++)
            tSpan[i] = 1f - MathF.Cos(0.05f * MathF.PI * i * (10f / odeSteps));

        var x = new float[melLen];
        for (int i = 0; i < melLen; i++)
            x[i] = (float)(NextGaussian(rng));

        float[]? zeroCond = null;
        if (cfgRate > 0f)
        {
            zeroCond = new float[melLen]; // Shared all-zero buffer for unconditional cond, mu, spks
        }

        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(RotaryInvFreq(), numFrames);

        for (int step = 1; step <= odeSteps; step++)
        {
            float t = tSpan[step - 1];
            float dt = tSpan[step] - tSpan[step - 1];

            if (cfgRate > 0f && zeroCond != null)
            {
                float[] dphiDt = null!;
                float[] cfgDphiDt = null!;
                Parallel.Invoke(
                    () => dphiDt = ForwardVelocity(w, x, cond, mu, spks, t, numFrames, rotaryCos, rotarySin),
                    () => cfgDphiDt = ForwardVelocity(w, x, zeroCond, zeroCond, zeroCond, t, numFrames, rotaryCos, rotarySin)
                );

                float condScale = 1f + cfgRate;
                fixed (float* xp = x, dp = dphiDt, cp = cfgDphiDt)
                {
                    int i = 0;
                    int vecSize = System.Numerics.Vector<float>.Count;
                    var vDt = new System.Numerics.Vector<float>(dt);
                    var vCondScale = new System.Numerics.Vector<float>(condScale);
                    var vCfgRate = new System.Numerics.Vector<float>(cfgRate);
                    for (; i <= melLen - vecSize; i += vecSize)
                    {
                        var vx = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(xp + i, vecSize));
                        var vd = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(dp + i, vecSize));
                        var vc = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(cp + i, vecSize));
                        var vv = vCondScale * vd - vCfgRate * vc;
                        var vRes = vx + vDt * vv;
                        vRes.CopyTo(new Span<float>(xp + i, vecSize));
                    }
                    for (; i < melLen; i++)
                    {
                        float v = condScale * dp[i] - cfgRate * cp[i];
                        xp[i] += dt * v;
                    }
                }
            }
            else
            {
                var dphiDt = ForwardVelocity(w, x, cond, mu, spks, t, numFrames, rotaryCos, rotarySin);
                fixed (float* xp = x, dp = dphiDt)
                {
                    int i = 0;
                    int vecSize = System.Numerics.Vector<float>.Count;
                    var vDt = new System.Numerics.Vector<float>(dt);
                    for (; i <= melLen - vecSize; i += vecSize)
                    {
                        var vx = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(xp + i, vecSize));
                        var vd = new System.Numerics.Vector<float>(new ReadOnlySpan<float>(dp + i, vecSize));
                        var vRes = vx + vDt * vd;
                        vRes.CopyTo(new Span<float>(xp + i, vecSize));
                    }
                    for (; i < melLen; i++)
                        xp[i] += dt * dp[i];
                }
            }
        }

        return x;
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>h is the already-embedded hidden state [numFrames, HiddenDim] (post input_embed -- see this class's doc comment for what's NOT yet implemented upstream of this call). Returns the predicted velocity in mel-space [numFrames, MelDim].</summary>
    public static float[] RunBackbone(CosyVoice3DiTWeights w, float[] h, float timestep, int numFrames, float[] rotaryCos, float[] rotarySin)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var tEmb = TimestepEmbedding(w, timestep);

        for (int layer = 0; layer < w.NumLayers; layer++)
            h = DiTBlock(w.Blocks[layer], h, tEmb, numFrames, rotaryCos, rotarySin);

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = F5Kernels.Linear(siluT, 1, dim, w.NormOutLinearWeight, w.NormOutLinearBias, dim * 2);

        var normOut = F5Kernels.LayerNormNoAffine(h, numFrames, dim);
        F5Kernels.ApplyAffineModulationSlice(normOut, normOut, modulation, scaleOffset: 0, shiftOffset: dim, numFrames, dim);

        return F5Kernels.Linear(normOut, numFrames, dim, w.ProjOutWeight, w.ProjOutBias, CosyVoice3DiTWeights.MelDim);
    }

    public static float[] RunBackbone(CosyVoice3DiTWeights w, float[] h, float timestep, int numFrames)
    {
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(RotaryInvFreq(), numFrames);
        return RunBackbone(w, h, timestep, numFrames, rotaryCos, rotarySin);
    }

    /// <summary>
    /// concat(x, cond, mu, spks) [confirmed order/composition, see this class's doc comment]
    /// -> proj -> + ConvPositionEmbedding(kernel=31, groups=16). x/cond/mu/spks are each
    /// channel-last [numFrames, MelDim]; spks is expected already broadcast to every frame.
    /// </summary>
    public static float[] InputEmbed(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, int numFrames)
    {
        int melDim = CosyVoice3DiTWeights.MelDim;
        int concatDim = melDim * 4;
        var concatInput = new float[numFrames * concatDim];
        for (int ti = 0; ti < numFrames; ti++)
        {
            int outOff = ti * concatDim;
            int melOff = ti * melDim;
            Array.Copy(x, melOff, concatInput, outOff, melDim);
            Array.Copy(cond, melOff, concatInput, outOff + melDim, melDim);
            Array.Copy(mu, melOff, concatInput, outOff + 2 * melDim, melDim);
            Array.Copy(spks, melOff, concatInput, outOff + 3 * melDim, melDim);
        }

        int hidden = CosyVoice3DiTWeights.HiddenDim;
        var h = F5Kernels.Linear(concatInput, numFrames, concatDim, w.InputProjWeight, w.InputProjBias, hidden);

        var pos = CausalGroupedConv1d(h, numFrames, hidden, w.ConvPos1Weight, w.ConvPos1Bias, CosyVoice3DiTWeights.ConvPosKernel, CosyVoice3DiTWeights.ConvPosGroups);
        for (int i = 0; i < pos.Length; i++) pos[i] = F5Kernels.Mish(pos[i]);
        pos = CausalGroupedConv1d(pos, numFrames, hidden, w.ConvPos2Weight, w.ConvPos2Bias, CosyVoice3DiTWeights.ConvPosKernel, CosyVoice3DiTWeights.ConvPosGroups);
        for (int i = 0; i < pos.Length; i++) pos[i] = F5Kernels.Mish(pos[i]);

        var output = new float[h.Length];
        for (int i = 0; i < output.Length; i++) output[i] = h[i] + pos[i];
        return output;
    }

    /// <summary>
    /// Causal grouped 1D convolution matching <c>cosyvoice-graph.cpp:269</c>: left-pads by <c>kernel - 1</c> (30 frames),
    /// so the convolution is strictly causal and only consumes current and past frames (no future lookahead).
    /// </summary>
    public static float[] CausalGroupedConv1d(float[] x, int t, int dim, float[] weight, float[] bias, int kernel, int groups)
    {
        int pad = kernel - 1;
        int inPerGroup = dim / groups;
        int outPerGroup = dim / groups;
        var y = new float[t * dim];
        Parallel.For(0, dim, oc =>
        {
            int group = oc / outPerGroup;
            int inBase = group * inPerGroup;
            int wBase = oc * inPerGroup * kernel;
            for (int ti = 0; ti < t; ti++)
            {
                float sum = bias[oc];
                for (int ic = 0; ic < inPerGroup; ic++)
                {
                    int wcBase = wBase + ic * kernel;
                    int srcCh = inBase + ic;
                    for (int k = 0; k < kernel; k++)
                    {
                        int src = ti - pad + k;
                        if ((uint)src < (uint)t) sum += weight[wcBase + k] * x[src * dim + srcCh];
                    }
                }
                y[ti * dim + oc] = sum;
            }
        });
        return y;
    }

    private static float[] TimestepEmbedding(CosyVoice3DiTWeights w, float timestep)
    {
        int freqDim = CosyVoice3DiTWeights.TimeFreqDim;
        int halfDim = freqDim / 2;

        var sinusEmbed = new float[freqDim];
        float embConst = MathF.Log(10000f) / (halfDim - 1);
        for (int k = 0; k < halfDim; k++)
        {
            float freq = MathF.Exp(-k * embConst);
            float angle = 1000f * timestep * freq;
            sinusEmbed[k] = MathF.Sin(angle);
            sinusEmbed[halfDim + k] = MathF.Cos(angle);
        }

        var h = F5Kernels.Linear(sinusEmbed, 1, freqDim, w.TimeMlp0Weight, w.TimeMlp0Bias, CosyVoice3DiTWeights.HiddenDim);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.SiLU(h[i]);
        return F5Kernels.Linear(h, 1, CosyVoice3DiTWeights.HiddenDim, w.TimeMlp2Weight, w.TimeMlp2Bias, CosyVoice3DiTWeights.HiddenDim);
    }

    /// <summary>Same RoPE base/formula F5-TTS's `F5RotaryEmbedding` uses (theta=10000, standard `x_transformers` convention) -- confirmed applicable since head_dim (64) matches exactly; not yet cross-checked against `examples/cosyvoice.cpp`'s own RoPE construction, flagged alongside the input_embed gap above.</summary>
    private static float[] RotaryInvFreq()
    {
        int halfHead = CosyVoice3DiTWeights.HeadDim / 2;
        var invFreq = new float[halfHead];
        for (int k = 0; k < halfHead; k++)
            invFreq[k] = 1f / MathF.Pow(10000f, (float)(2 * k) / CosyVoice3DiTWeights.HeadDim);
        return invFreq;
    }

    private static float[] DiTBlock(CosyVoice3DiTBlockWeights bw, float[] x, float[] tEmb, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = F5Kernels.Linear(siluT, 1, dim, bw.AttnNormLinearWeight, bw.AttnNormLinearBias, dim * 6);

        var norm = F5Kernels.LayerNormNoAffine(x, t, dim);
        F5Kernels.ApplyAffineModulationSlice(norm, norm, modulation, 1 * dim, 0 * dim, t, dim);

        var attnOut = Attention(bw, norm, t, rotaryCos, rotarySin);

        var xAfterAttn = new float[x.Length];
        F5Kernels.ApplyGatedResidualSlice(xAfterAttn, x, modulation, 2 * dim, attnOut, t, dim);

        var ffNorm = F5Kernels.LayerNormNoAffine(xAfterAttn, t, dim);
        F5Kernels.ApplyAffineModulationSlice(ffNorm, ffNorm, modulation, 4 * dim, 3 * dim, t, dim);

        var ffOut = FeedForward(bw, ffNorm, t);

        var output = new float[x.Length];
        F5Kernels.ApplyGatedResidualSlice(output, xAfterAttn, modulation, 5 * dim, ffOut, t, dim);
        return output;
    }

    private static float[] FeedForward(CosyVoice3DiTBlockWeights bw, float[] x, int t)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int ffn = CosyVoice3DiTWeights.FfnDim;
        var h = F5Kernels.Linear(x, t, dim, bw.FfInWeight, bw.FfInBias, ffn);
        Parallel.For(0, t, ti =>
        {
            int off = ti * ffn;
            for (int d = 0; d < ffn; d++)
                h[off + d] = GeluErf(h[off + d]);
        });
        return F5Kernels.Linear(h, t, ffn, bw.FfOutWeight, bw.FfOutBias, dim);
    }

    /// <summary>
    /// Exact erf-based GELU matching <c>cosyvoice-graph.cpp:403</c> (<c>ggml_gelu_erf</c>).
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float GeluErf(float x)
    {
        return 0.5f * x * (1.0f + Erff(x * 0.7071067811865475f));
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float Erff(float x)
    {
        float a1 = 0.254829592f;
        float a2 = -0.284496736f;
        float a3 = 1.421413741f;
        float a4 = -1.453152027f;
        float a5 = 1.061405429f;
        float p = 0.3275911f;

        float sign = x < 0 ? -1f : 1f;
        x = MathF.Abs(x);

        float t = 1.0f / (1.0f + p * x);
        float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * MathF.Exp(-x * x);

        return sign * y;
    }

    /// <summary>See <see cref="F5Kernels.ApplyRotary"/>/<see cref="F5Kernels.MultiHeadSelfAttention"/> (shared with F5-TTS's tensor-for-tensor-identical DiT).</summary>
    private static float[] Attention(CosyVoice3DiTBlockWeights bw, float[] norm, int t, float[] rotaryCos, float[] rotarySin)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int heads = CosyVoice3DiTWeights.NumHeads;
        int headDim = CosyVoice3DiTWeights.HeadDim;

        var q = F5Kernels.Linear(norm, t, dim, bw.ToQWeight, bw.ToQBias, dim);
        var k = F5Kernels.Linear(norm, t, dim, bw.ToKWeight, bw.ToKBias, dim);
        var v = F5Kernels.Linear(norm, t, dim, bw.ToVWeight, bw.ToVBias, dim);

        F5Kernels.ApplyRotary(q, t, heads, headDim, rotaryCos, rotarySin);
        F5Kernels.ApplyRotary(k, t, heads, headDim, rotaryCos, rotarySin);

        var context = F5Kernels.MultiHeadSelfAttention(q, k, v, t, heads, headDim);
        return F5Kernels.Linear(context, t, dim, bw.ToOutWeight, bw.ToOutBias, dim);
    }
}
