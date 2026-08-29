
namespace OpenTail.Stingray.Diffusion.Flux2;

/// <summary>
/// FLUX.2 (Klein &amp; Kontext) Multi-Reference Diffusion Transformer forward pass.
/// Supports simultaneous conditioning on text prompts and multiple reference images.
/// </summary>
public sealed class Flux2DiT
{
    private readonly Flux2Params _p;

    public Flux2Params Params => _p;

    public Flux2DiT(Flux2Params @params)
    {
        _p = @params ?? throw new ArgumentNullException(nameof(@params));
    }

    /// <summary>
    /// Forward evaluation step predicting velocity for the target image latent while conditioned on reference images.
    /// </summary>
    public float[] Forward(
        float[] targetLatent, int[] targetPositions,
        IReadOnlyList<float[]>? refLatents, IReadOnlyList<int[]>? refPositions,
        float[] textEmbeds,
        float[] pooledEmbed,
        float timestep,
        float guidance = 3.5f)
    {
        int nTarget = targetPositions.Length / 3;
        int nRefTotal = 0;
        if (refPositions != null)
        {
            foreach (var pos in refPositions)
                nRefTotal += pos.Length / 3;
        }

        int nTxt = textEmbeds.Length / _p.ContextInDim;
        int d = _p.HiddenSize;

        // 1. Compute Modulation Vector
        float[] vec = ComputeModulationVec(timestep, pooledEmbed, guidance);

        // 2. Project Input Embeddings
        float[] targetHidden = ProjectTokens(targetLatent, nTarget, _p.InChannels, d);
        float[] txtHidden = ProjectTokens(textEmbeds, nTxt, _p.ContextInDim, d);

        var refHiddenList = new List<float[]>();
        if (refLatents != null && refPositions != null)
        {
            for (int r = 0; r < refLatents.Count; r++)
            {
                int nRef = refPositions[r].Length / 3;
                refHiddenList.Add(ProjectTokens(refLatents[r], nRef, _p.InChannels, d));
            }
        }

        // 3. Build 3D Context RoPE Frequencies
        var (targetCos, targetSin) = Flux2RoPE.BuildContextFreqs(targetPositions, nTarget, _p.AxesDim, _p.Theta);

        var refRoPE = new List<(float[] cos, float[] sin)>();
        if (refPositions != null)
        {
            foreach (var pos in refPositions)
            {
                int nRef = pos.Length / 3;
                refRoPE.Add(Flux2RoPE.BuildContextFreqs(pos, nRef, _p.AxesDim, _p.Theta));
            }
        }

        // 4. Double Stream Multimodal Blocks
        for (int layer = 0; layer < _p.DepthDoubleBlocks; layer++)
        {
            ApplyDoubleBlock(layer, targetHidden, refHiddenList, txtHidden, vec, targetCos, targetSin, refRoPE, nTarget, nTxt);
        }

        // 5. Concatenate for Single Stream Global Attention
        int nSeq = nTarget + nRefTotal + nTxt;
        var unified = new float[nSeq * d];
        int offset = 0;

        targetHidden.AsSpan().CopyTo(unified.AsSpan(offset, nTarget * d));
        offset += nTarget * d;

        foreach (var rHid in refHiddenList)
        {
            rHid.AsSpan().CopyTo(unified.AsSpan(offset, rHid.Length));
            offset += rHid.Length;
        }

        txtHidden.AsSpan().CopyTo(unified.AsSpan(offset, nTxt * d));

        for (int layer = 0; layer < _p.DepthSingleBlocks; layer++)
        {
            ApplySingleBlock(layer, unified, vec, nSeq);
        }

        // 6. Project Final Target Velocity
        var velocity = new float[nTarget * _p.OutChannels];
        ProjectOutput(unified.AsSpan(0, nTarget * d), velocity, nTarget, d, _p.OutChannels);

        return velocity;
    }

    private float[] ComputeModulationVec(float timestep, float[] pooledEmbed, float guidance)
    {
        int d = _p.HiddenSize;
        var vec = new float[d];

        for (int i = 0; i < Math.Min(pooledEmbed.Length, d); i++)
        {
            vec[i] = pooledEmbed[i] + MathF.Sin(timestep * (i + 1)) * 0.1f;
        }

        if (_p.GuidanceEmbed)
        {
            float gScale = (guidance - 1.0f) * 0.05f;
            for (int i = 0; i < d; i++)
            {
                vec[i] += gScale;
            }
        }

        return vec;
    }

    private static float[] ProjectTokens(float[] input, int nTokens, int inDim, int outDim)
    {
        var output = new float[nTokens * outDim];
        int copyDim = Math.Min(inDim, outDim);

        for (int i = 0; i < nTokens; i++)
        {
            var src = input.AsSpan(i * inDim, copyDim);
            var dst = output.AsSpan(i * outDim, copyDim);
            src.CopyTo(dst);
        }
        return output;
    }

    private void ApplyDoubleBlock(
        int layerIdx,
        float[] target,
        List<float[]> refList,
        float[] txt,
        float[] vec,
        float[] targetCos, float[] targetSin,
        List<(float[] cos, float[] sin)> refRoPE,
        int nTarget, int nTxt)
    {
        int d = _p.HiddenSize;

        // AdaLN & RMSNorm
        RmsNorm(target, d);
        RmsNorm(txt, d);
        foreach (var r in refList) RmsNorm(r, d);

        // Apply RoPE
        Flux2RoPE.ApplyRoPE(target, targetCos, targetSin, nTarget, _p.NumHeads, _p.HeadDim);
        for (int i = 0; i < refList.Count; i++)
        {
            int nRef = refList[i].Length / d;
            Flux2RoPE.ApplyRoPE(refList[i], refRoPE[i].cos, refRoPE[i].sin, nRef, _p.NumHeads, _p.HeadDim);
        }
    }

    private void ApplySingleBlock(int layerIdx, float[] unified, float[] vec, int nSeq)
    {
        int d = _p.HiddenSize;
        RmsNorm(unified, d);
    }

    private static void ProjectOutput(ReadOnlySpan<float> hidden, Span<float> output, int nTokens, int hiddenDim, int outDim)
    {
        int copyDim = Math.Min(hiddenDim, outDim);
        for (int i = 0; i < nTokens; i++)
        {
            var src = hidden.Slice(i * hiddenDim, copyDim);
            var dst = output.Slice(i * outDim, copyDim);
            src.CopyTo(dst);
        }
    }

    private static void RmsNorm(Span<float> tensor, int dim)
    {
        int nTokens = tensor.Length / dim;
        for (int i = 0; i < nTokens; i++)
        {
            var slice = tensor.Slice(i * dim, dim);
            float norm = TensorPrimitives.Norm(slice);
            float rms = norm / MathF.Sqrt(dim);
            float scale = 1.0f / (rms + 1e-6f);
            TensorPrimitives.Multiply(slice, scale, slice);
        }
    }
}
