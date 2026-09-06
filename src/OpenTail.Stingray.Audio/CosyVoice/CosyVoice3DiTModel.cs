using OpenTail.Stingray.Audio.F5TTS;
using OpenTail.Stingray.Core;

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
    /// <summary>Dispatches to <see cref="F5Kernels.LinearGpu"/> when <paramref name="backend"/> is
    /// supplied (--backend vulkan), else the existing CPU <see cref="F5Kernels.Linear"/> path.
    /// See docs/052-vulkan-backend-for-tts-engines-plan.md.</summary>
    private static float[] Lin(IComputeBackend? backend, float[] x, int t, int inDim, float[] weight, float[]? bias, int outDim) =>
        backend is not null
            ? F5Kernels.LinearGpu(backend, x, t, inDim, weight, bias, outDim)
            : F5Kernels.Linear(x, t, inDim, weight, bias, outDim);

    /// <summary>
    /// Full DiT forward pass. x/cond/mu/spks are channel-last [numFrames, MelDim] (spks is
    /// logically per-utterance but passed pre-broadcast to every frame here, matching
    /// `ggml_repeat`'s effect in the real graph -- callers should broadcast a single [MelDim]
    /// speaker vector across all frames before calling). Returns predicted velocity,
    /// [numFrames, MelDim].
    /// </summary>
    public static float[] ForwardVelocity(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, float timestep, int numFrames, float[] rotaryCos, float[] rotarySin, IComputeBackend? backend = null)
    {
        var h = InputEmbed(w, x, cond, mu, spks, numFrames, backend);
        return RunBackbone(w, h, timestep, numFrames, rotaryCos, rotarySin, backend);
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
    public static unsafe float[] SolveFlowMatchingOde(CosyVoice3DiTWeights w, float[] cond, float[] mu, float[] spks, int numFrames, int odeSteps, Random rng, float cfgRate = 0.7f, IComputeBackend? backend = null)
    {
        int melLen = numFrames * CosyVoice3DiTWeights.MelDim;
        var tSpan = new float[odeSteps + 1];
        for (int i = 0; i <= odeSteps; i++)
            tSpan[i] = 1f - MathF.Cos(0.05f * MathF.PI * i * (10f / odeSteps));

        // The flow-matching ODE's starting state is NOT fresh per-call randomness. The real
        // reference (`read_noise_prefix` in examples/audio.cpp's flow.cpp) reads a fixed,
        // pre-baked noise tensor (`decoder.rand_noise`, [15000, 80], CHANNEL-MAJOR -- 80
        // contiguous runs of up to 15000 frame values each) baked into the checkpoint and slices
        // a frame-count-sized prefix off each channel's run every call -- it never samples fresh
        // Gaussian noise at inference time. This matters because two independent PRNG
        // implementations (.NET's `Random` vs PyTorch's generator) can NEVER reproduce the same
        // "random" sequence even given a matching seed; a shared, frozen buffer baked into the
        // checkpoint is the only way two independent implementations can start the ODE from
        // literally the same state. Root-caused this session: fresh-per-call Gaussian noise here
        // produced a uniform ~1.3-3x excess mel variance across all 80 channels vs the real
        // reference, which is what the long-standing "wobbling/metallic" CosyVoice3 quality
        // issue traced back to.
        //
        // Use the checkpoint's own tensor when it carries one (spliced in via
        // CosyVoice3SpliceRandNoiseTool, or present natively as in the upstream Q8_0 checkpoint);
        // otherwise fall back to FallbackRandNoise, a buffer built ONCE with a fixed seed (NOT
        // matching the real reference's actual values, since that's impossible across languages,
        // but at least internally reproducible run-to-run for OUR OWN pipeline). Never regenerate
        // fresh randomness per call in either branch -- that reintroduces the exact bug this
        // replaces.
        float[] noiseSource = w.RandNoise ?? FallbackRandNoise;
        if (numFrames > NoiseCapacityFrames)
            throw new InvalidOperationException(
                $"CosyVoice3 requested {numFrames} mel frames, which exceeds the fixed flow noise buffer's capacity of {NoiseCapacityFrames} frames.");
        var x = new float[melLen];
        for (int f = 0; f < numFrames; f++)
            for (int c = 0; c < CosyVoice3DiTWeights.MelDim; c++)
                x[f * CosyVoice3DiTWeights.MelDim + c] = noiseSource[c * NoiseCapacityFrames + f];

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
                float[] dphiDt;
                float[] cfgDphiDt;
                // Same reasoning as ChatterboxCfmDecoder: IComputeBackend isn't verified
                // thread-safe for concurrent dispatch from two threads onto one instance, so
                // serialize the CFG cond/uncond branches when GPU-backed instead of Parallel.Invoke.
                if (backend is not null)
                {
                    dphiDt = ForwardVelocity(w, x, cond, mu, spks, t, numFrames, rotaryCos, rotarySin, backend);
                    cfgDphiDt = ForwardVelocity(w, x, zeroCond, zeroCond, zeroCond, t, numFrames, rotaryCos, rotarySin, backend);
                }
                else
                {
                    float[] dCond = null!, dUncond = null!;
                    Parallel.Invoke(
                        () => dCond = ForwardVelocity(w, x, cond, mu, spks, t, numFrames, rotaryCos, rotarySin),
                        () => dUncond = ForwardVelocity(w, x, zeroCond, zeroCond, zeroCond, t, numFrames, rotaryCos, rotarySin)
                    );
                    dphiDt = dCond;
                    cfgDphiDt = dUncond;
                }

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

                if (Environment.GetEnvironmentVariable("STINGRAY_CFG_TRACE") == "1")
                {
                    double sxsq = 0, sdsq = 0, scsq = 0;
                    for (int j = 0; j < melLen; j++) { sxsq += (double)x[j] * x[j]; sdsq += (double)dphiDt[j] * dphiDt[j]; scsq += (double)cfgDphiDt[j] * cfgDphiDt[j]; }
                    Console.Error.WriteLine($"[CfgTrace] step={step} t={t:F3} dt={dt:F3} x_rms={Math.Sqrt(sxsq / melLen):F4} condV_rms={Math.Sqrt(sdsq / melLen):F4} uncondV_rms={Math.Sqrt(scsq / melLen):F4}");

                    // Track a few representative channels' std growth across ALL ODE steps (not
                    // just the final one) to see WHERE the low-mel-frequency-channel excess
                    // variance first opens up relative to the reference, rather than only its
                    // end state.
                    {
                        int mel = CosyVoice3DiTWeights.MelDim;
                        int[] trackedCh = [0, 1, 2, 4, 21, 40, 60];
                        var stepSb = new System.Text.StringBuilder($"[XPerChStep] step={step} std[");
                        foreach (var c in trackedCh)
                        {
                            double sum = 0, sumsq = 0;
                            for (int f = 0; f < numFrames; f++) { double v = x[f * mel + c]; sum += v; sumsq += v * v; }
                            double mean = sum / numFrames;
                            double variance = Math.Max(0, sumsq / numFrames - mean * mean);
                            stepSb.Append($"ch{c}={Math.Sqrt(variance):F3} ");
                        }
                        stepSb.Append(']');
                        Console.Error.WriteLine(stepSb.ToString());
                    }
                    if (step == odeSteps)
                    {
                        int mel = CosyVoice3DiTWeights.MelDim;
                        int worstFrame = -1; float worstAbs = 0f;
                        for (int f = 0; f < numFrames; f++)
                        {
                            float m = 0f;
                            for (int c = 0; c < mel; c++) m = Math.Max(m, Math.Abs(x[f * mel + c]));
                            if (m > worstAbs) { worstAbs = m; worstFrame = f; }
                        }
                        Console.Error.WriteLine($"[CfgTrace] FINAL worstFrame={worstFrame}/{numFrames} worstAbs={worstAbs:F4} (whole seq, prompt+target)");

                        // Per-channel breakdown of the cond vs uncond velocity RMS on the final
                        // step, to test whether CFG amplification ((1+cfgRate)*cond - cfgRate*uncond)
                        // is what's inflating specific mel channels: if cond/uncond diverge sharply
                        // on the low-frequency channels, CFG's subtraction would amplify exactly
                        // those channels' variance in x.
                        var condSb = new System.Text.StringBuilder("[CfgTrace] condV_perCh_rms=");
                        var uncondSb = new System.Text.StringBuilder("[CfgTrace] uncondV_perCh_rms=");
                        for (int c = 0; c < mel; c++)
                        {
                            double sd = 0, su = 0;
                            for (int f = 0; f < numFrames; f++)
                            {
                                double dv = dphiDt[f * mel + c];
                                double uv = cfgDphiDt[f * mel + c];
                                sd += dv * dv;
                                su += uv * uv;
                            }
                            condSb.Append(Math.Sqrt(sd / numFrames).ToString("F3")).Append(',');
                            uncondSb.Append(Math.Sqrt(su / numFrames).ToString("F3")).Append(',');
                        }
                        Console.Error.WriteLine(condSb.ToString());
                        Console.Error.WriteLine(uncondSb.ToString());
                    }
                }
            }
            else
            {
                var dphiDt = ForwardVelocity(w, x, cond, mu, spks, t, numFrames, rotaryCos, rotarySin, backend);
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

    // Matches the real checkpoint's fixed noise tensor capacity (`50 * 300` frames in the real
    // reference's `read_noise_prefix`), so a checkpoint-provided and fallback buffer can be
    // sliced identically regardless of which one is in use.
    private const int NoiseCapacityFrames = 50 * 300;

    /// <summary>Frozen fallback noise buffer for checkpoints that don't carry the real
    /// `decoder.rand_noise` tensor -- built ONCE, with a fixed seed, the first time it's needed,
    /// and reused (sliced, never regenerated) on every subsequent call. See the doc comment at
    /// its call site in <see cref="SolveFlowMatchingOde"/> for why this must never be freshly
    /// randomized per call.</summary>
    private static readonly Lazy<float[]> FallbackRandNoiseLazy = new(() =>
    {
        var buf = new float[NoiseCapacityFrames * CosyVoice3DiTWeights.MelDim];
        var rng = new Random(0);
        for (int i = 0; i < buf.Length; i++)
            buf[i] = (float)NextGaussian(rng);
        return buf;
    });
    private static float[] FallbackRandNoise => FallbackRandNoiseLazy.Value;

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>h is the already-embedded hidden state [numFrames, HiddenDim] (post input_embed -- see this class's doc comment for what's NOT yet implemented upstream of this call). Returns the predicted velocity in mel-space [numFrames, MelDim].</summary>
    public static float[] RunBackbone(CosyVoice3DiTWeights w, float[] h, float timestep, int numFrames, float[] rotaryCos, float[] rotarySin, IComputeBackend? backend = null)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var tEmb = TimestepEmbedding(w, timestep, backend);

        for (int layer = 0; layer < w.NumLayers; layer++)
            h = DiTBlock(w.Blocks[layer], h, tEmb, numFrames, rotaryCos, rotarySin, backend);

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = Lin(backend, siluT, 1, dim, w.NormOutLinearWeight, w.NormOutLinearBias, dim * 2);

        var normOut = F5Kernels.LayerNormNoAffine(h, numFrames, dim);
        F5Kernels.ApplyAffineModulationSlice(normOut, normOut, modulation, scaleOffset: 0, shiftOffset: dim, numFrames, dim);

        return Lin(backend, normOut, numFrames, dim, w.ProjOutWeight, w.ProjOutBias, CosyVoice3DiTWeights.MelDim);
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
    public static float[] InputEmbed(CosyVoice3DiTWeights w, float[] x, float[] cond, float[] mu, float[] spks, int numFrames, IComputeBackend? backend = null)
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
        var h = Lin(backend, concatInput, numFrames, concatDim, w.InputProjWeight, w.InputProjBias, hidden);

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

    private static float[] TimestepEmbedding(CosyVoice3DiTWeights w, float timestep, IComputeBackend? backend = null)
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

        var h = Lin(backend, sinusEmbed, 1, freqDim, w.TimeMlp0Weight, w.TimeMlp0Bias, CosyVoice3DiTWeights.HiddenDim);
        for (int i = 0; i < h.Length; i++) h[i] = F5Kernels.SiLU(h[i]);
        return Lin(backend, h, 1, CosyVoice3DiTWeights.HiddenDim, w.TimeMlp2Weight, w.TimeMlp2Bias, CosyVoice3DiTWeights.HiddenDim);
    }

    /// <summary>Same RoPE base/formula F5-TTS's `F5RotaryEmbedding` uses (theta=10000, standard `x_transformers` convention). Cross-checked against the real reference's own RoPE construction
    /// (`examples/audio.cpp/src/models/cosyvoice3/flow.cpp:322-326`'s `attention_config()`:
    /// `rope_theta=10000.0F`, `rope_freq_scale` defaults to `1.0F`, `rope_type=GGML_ROPE_TYPE_NORMAL`
    /// -- confirmed via `ggml.h`'s own doc comment that NORMAL mode is interleaved-pairs rotation
    /// `[cscs...]` for the first `n_dims` elements, matching this file's/F5Kernels.ApplyRotary's
    /// convention exactly) -- all match. See Attention's numRopeHeads:1 call site for the one real
    /// discrepancy this cross-check found (RoPE applied only to head 0, not all 16 heads).</summary>
    private static float[] RotaryInvFreq()
    {
        int halfHead = CosyVoice3DiTWeights.HeadDim / 2;
        var invFreq = new float[halfHead];
        for (int k = 0; k < halfHead; k++)
            invFreq[k] = 1f / MathF.Pow(10000f, (float)(2 * k) / CosyVoice3DiTWeights.HeadDim);
        return invFreq;
    }

    private static float[] DiTBlock(CosyVoice3DiTBlockWeights bw, float[] x, float[] tEmb, int t, float[] rotaryCos, float[] rotarySin, IComputeBackend? backend = null)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;

        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = Lin(backend, siluT, 1, dim, bw.AttnNormLinearWeight, bw.AttnNormLinearBias, dim * 6);

        var norm = F5Kernels.LayerNormNoAffine(x, t, dim);
        F5Kernels.ApplyAffineModulationSlice(norm, norm, modulation, 1 * dim, 0 * dim, t, dim);

        var attnOut = Attention(bw, norm, t, rotaryCos, rotarySin, backend);

        var xAfterAttn = new float[x.Length];
        F5Kernels.ApplyGatedResidualSlice(xAfterAttn, x, modulation, 2 * dim, attnOut, t, dim);

        var ffNorm = F5Kernels.LayerNormNoAffine(xAfterAttn, t, dim);
        F5Kernels.ApplyAffineModulationSlice(ffNorm, ffNorm, modulation, 4 * dim, 3 * dim, t, dim);

        var ffOut = FeedForward(bw, ffNorm, t, backend);

        var output = new float[x.Length];
        F5Kernels.ApplyGatedResidualSlice(output, xAfterAttn, modulation, 5 * dim, ffOut, t, dim);
        return output;
    }

    private static float[] FeedForward(CosyVoice3DiTBlockWeights bw, float[] x, int t, IComputeBackend? backend = null)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int ffn = CosyVoice3DiTWeights.FfnDim;
        var h = Lin(backend, x, t, dim, bw.FfInWeight, bw.FfInBias, ffn);
        // EXPERIMENT (2026-09-05): examples/audio.cpp/src/models/cosyvoice3/flow.cpp explicitly
        // configures GeluApproximation::Tanh for this exact FeedForward, disagreeing with the
        // older examples/cosyvoice.cpp reference this file's GeluErf was verified against.
        // Gated behind an env var for A/B listening, not switched by default -- two independent
        // references disagreeing needs a real listen, not a coin flip.
        bool useTanhGelu = Environment.GetEnvironmentVariable("STINGRAY_COSYVOICE3_TANH_GELU") == "1";
        Parallel.For(0, t, ti =>
        {
            int off = ti * ffn;
            for (int d = 0; d < ffn; d++)
                h[off + d] = useTanhGelu ? F5Kernels.GeluTanh(h[off + d]) : GeluErf(h[off + d]);
        });
        return Lin(backend, h, t, ffn, bw.FfOutWeight, bw.FfOutBias, dim);
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
    private static float[] Attention(CosyVoice3DiTBlockWeights bw, float[] norm, int t, float[] rotaryCos, float[] rotarySin, IComputeBackend? backend = null)
    {
        int dim = CosyVoice3DiTWeights.HiddenDim;
        int heads = CosyVoice3DiTWeights.NumHeads;
        int headDim = CosyVoice3DiTWeights.HeadDim;

        var q = Lin(backend, norm, t, dim, bw.ToQWeight, bw.ToQBias, dim);
        var k = Lin(backend, norm, t, dim, bw.ToKWeight, bw.ToKBias, dim);
        var v = Lin(backend, norm, t, dim, bw.ToVWeight, bw.ToVBias, dim);

        // Real reference (examples/audio.cpp's projected_grouped_self_attention.cpp:97-111,
        // cosyvoice3/flow.cpp:322-326) applies RoPE to the UNSPLIT [t, 1024] q/k projection
        // (before reshape_projected_heads splits it into [t, 16, 64]) via
        // ggml_rope_ext(..., n_dims=head_dim=64, ...) -- confirmed via positional_modules.cpp:76-91
        // this maps straight to ggml_rope_ext with n_dims < the tensor's last dimension (1024).
        // Real GGML rope semantics for n_dims < ne[0] rotate ONLY the first n_dims elements and
        // pass the rest through unrotated (the same partial-rotary mechanism used by e.g.
        // GPT-NeoX-style models) -- so only elements [0,64) (head 0) are ever rotated; heads
        // 1-15 (elements [64,1024)) pass through completely unrotated. This earlier had NO
        // numRopeHeads argument (defaulting to all 16 heads), silently distorting 15 of 16
        // heads' attention on every one of the 22 layers -- an earlier comment on
        // F5Kernels.ApplyRotary claiming "CosyVoice3's own real config" rotates all heads was
        // wrong (never actually traced through to ggml_rope_ext's real n_dims-vs-ne[0] behavior).
        F5Kernels.ApplyRotary(q, t, heads, headDim, rotaryCos, rotarySin, numRopeHeads: 1);
        F5Kernels.ApplyRotary(k, t, heads, headDim, rotaryCos, rotarySin, numRopeHeads: 1);

        var context = F5Kernels.MultiHeadSelfAttention(q, k, v, t, heads, headDim);
        return Lin(backend, context, t, dim, bw.ToOutWeight, bw.ToOutBias, dim);
    }
}
