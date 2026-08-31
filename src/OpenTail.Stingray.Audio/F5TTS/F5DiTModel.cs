
namespace OpenTail.Stingray.Audio.F5TTS;

/// <summary>
/// F5-TTS's 22-layer Flow-Matching Diffusion Transformer (DiT), real weight-driven port of
/// `examples/f5-tts-py/f5_tts/model/backbones/dit.py`'s `DiT.forward`, verified against the real
/// PyTorch reference loaded with the real checkpoint (see scratch-llamacpp-ref/f5_golden_dit.py --
/// unlike the ONNX-exported VITS-family pipelines this session, F5-TTS ships as a safetensors
/// checkpoint with the original PyTorch source available, so golden verification runs the ACTUAL
/// reference model directly instead of an ONNX re-export).
///
/// x/cond are the noisy/conditioning mel spectrograms, channel-last [numFrames, MelDim]. text is
/// raw (unshifted) character ids. Returns the predicted velocity, same shape as x.
/// </summary>
public static class F5DiTModel
{
    public static float[] ForwardVelocity(F5TtsWeights w, float[] x, float[] cond, ReadOnlySpan<int> text, float timestep, int numFrames) =>
        ForwardVelocity(w, x, cond, text, timestep, numFrames, dropText: false);

    /// <summary>dropText selects CFG's null/unconditional text branch (see <see cref="F5TextEmbedding"/>'s dropText overload doc comment). Callers doing CFG should also pass an all-zero `cond` for the uncond branch (drop_audio_cond).</summary>
    public static float[] ForwardVelocity(F5TtsWeights w, float[] x, float[] cond, ReadOnlySpan<int> text, float timestep, int numFrames, bool dropText)
    {
        var textEmbed = F5TextEmbedding.Forward(w, text, numFrames, dropText);
        var (rotaryCos, rotarySin) = F5RotaryEmbedding.Precompute(w.RotaryInvFreq, numFrames);
        return ForwardVelocity(w, x, cond, textEmbed, timestep, numFrames, rotaryCos, rotarySin);
    }

    /// <summary>Optimized forward pass accepting precomputed/cached textEmbed, rotaryCos, and rotarySin to avoid redundant recomputations across ODE steps.</summary>
    public static float[] ForwardVelocity(
        F5TtsWeights w,
        float[] x,
        float[] cond,
        float[] textEmbed,
        float timestep,
        int numFrames,
        float[] rotaryCos,
        float[] rotarySin)
    {
        var h = F5InputEmbedding.Forward(w, x, cond, textEmbed, numFrames);
        var tEmb = F5TimestepEmbedding.Forward(w, timestep);

        for (int layer = 0; layer < F5TtsWeights.NumLayers; layer++)
            h = F5DiTBlock.Forward(w, w.Blocks[layer], h, tEmb, numFrames, rotaryCos, rotarySin);

        // norm_out: AdaLayerNorm_Final -- emb = linear(silu(t)) -> chunk2(scale,shift); x = LN_noaffine(x)*(1+scale)+shift.
        int dim = F5TtsWeights.HiddenDim;
        var siluT = new float[dim];
        for (int d = 0; d < dim; d++) siluT[d] = F5Kernels.SiLU(tEmb[d]);
        var modulation = F5Kernels.LinearQ8_0(siluT, 1, dim, w.NormOutLinearQ8, w.NormOutLinearBias, dim * 2);

        var normOut = F5Kernels.LayerNormNoAffine(h, numFrames, dim);
        F5Kernels.ApplyAffineModulationSlice(normOut, normOut, modulation, scaleOffset: 0, shiftOffset: dim, numFrames, dim);

        return F5Kernels.LinearQ8_0(normOut, numFrames, dim, w.ProjOutQ8, w.ProjOutBias, F5TtsWeights.MelDim);
    }
}
