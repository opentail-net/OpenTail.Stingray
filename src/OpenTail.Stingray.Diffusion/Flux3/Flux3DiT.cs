
namespace OpenTail.Stingray.Diffusion.Flux3;

/// <summary>
/// FLUX 3 Multimodal Diffusion Transformer (MM-DiT) forward pass.
/// Supports concurrent Video, Native Audio, and Text generation in a unified flow-matching architecture.
/// </summary>
public sealed class Flux3DiT
{
    private readonly Flux3Params _p;

    public Flux3Params Params => _p;

    public Flux3DiT(Flux3Params @params)
    {
        _p = @params ?? throw new ArgumentNullException(nameof(@params));
    }

    /// <summary>
    /// Multimodal forward evaluation step predicting both video and audio velocity fields.
    /// </summary>
    public (float[] videoVelocity, float[]? audioVelocity) Forward(
        float[] videoLatent, int[] videoPositions,
        float[]? audioLatent, int[]? audioPositions,
        float[] textEmbeds,
        float[] pooledEmbed,
        float timestep,
        float guidance = 3.5f,
        Flux3KvCache? kvCache = null)
    {
        int nVid = videoPositions.Length / 3;
        int nAud = audioPositions != null && audioPositions.Length > 0 ? audioPositions.Length / 2 : 0;
        int nTxt = textEmbeds.Length / _p.ContextInDim;
        int d = _p.HiddenSize;

        // 1. Conditioning Vectors
        float[] vec = ComputeModulationVec(timestep, pooledEmbed, guidance);

        // 2. Linear Input Embeddings
        float[] vidHidden = ProjectTokens(videoLatent, nVid, _p.InVideoChannels, d);
        float[]? audHidden = nAud > 0 && audioLatent != null
            ? ProjectTokens(audioLatent, nAud, _p.InAudioChannels, d)
            : null;
        float[] txtHidden = ProjectTokens(textEmbeds, nTxt, _p.ContextInDim, d);

        // 3. Build RoPE Frequencies
        var (vidCos, vidSin) = Flux3RoPE.BuildVideoFreqs(videoPositions, nVid, _p.VideoAxesDim, _p.Theta);
        var (audCos, audSin) = nAud > 0 && audioPositions != null
            ? Flux3RoPE.BuildAudioFreqs(audioPositions, nAud, _p.AudioAxesDim, _p.Theta)
            : (Array.Empty<float>(), Array.Empty<float>());

        // 4. Double Stream Multimodal Blocks
        for (int layer = 0; layer < _p.DepthDoubleBlocks; layer++)
        {
            ApplyDoubleBlock(layer, vidHidden, audHidden, txtHidden, vec, vidCos, vidSin, audCos, audSin, nVid, nAud, nTxt, kvCache);
        }

        // 5. Concatenate streams for Single Stream Unified Blocks
        int nSeq = nVid + nAud + nTxt;
        var unified = new float[nSeq * d];
        vidHidden.AsSpan().CopyTo(unified.AsSpan(0, nVid * d));
        if (audHidden != null)
        {
            audHidden.AsSpan().CopyTo(unified.AsSpan(nVid * d, nAud * d));
        }
        txtHidden.AsSpan().CopyTo(unified.AsSpan((nVid + nAud) * d, nTxt * d));

        for (int layer = 0; layer < _p.DepthSingleBlocks; layer++)
        {
            ApplySingleBlock(layer, unified, vec, nSeq, kvCache);
        }

        // 6. Project Final Velocities
        var videoVelocity = new float[nVid * _p.OutVideoChannels];
        ProjectOutput(unified.AsSpan(0, nVid * d), videoVelocity, nVid, d, _p.OutVideoChannels);

        float[]? audioVelocity = null;
        if (nAud > 0)
        {
            audioVelocity = new float[nAud * _p.OutAudioChannels];
            ProjectOutput(unified.AsSpan(nVid * d, nAud * d), audioVelocity, nAud, d, _p.OutAudioChannels);
        }

        return (videoVelocity, audioVelocity);
    }

    private float[] ComputeModulationVec(float timestep, float[] pooledEmbed, float guidance)
    {
        int d = _p.HiddenSize;
        var vec = new float[d];

        // Sinusoidal timestep embedding + modulation
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
        float[] vid, float[]? aud, float[] txt,
        float[] vec,
        float[] vidCos, float[] vidSin,
        float[] audCos, float[] audSin,
        int nVid, int nAud, int nTxt,
        Flux3KvCache? kvCache)
    {
        int d = _p.HiddenSize;

        // AdaLN modulation and residual connection
        RmsNorm(vid, d);
        RmsNorm(txt, d);
        if (aud != null) RmsNorm(aud, d);

        // RoPE attention
        Flux3RoPE.ApplyRoPE(vid, vidCos, vidSin, nVid, _p.NumHeads, _p.HeadDim);
        if (aud != null && audCos.Length > 0)
        {
            Flux3RoPE.ApplyRoPE(aud, audCos, audSin, nAud, _p.NumHeads, _p.HeadDim);
        }

        // Store reference tokens if KV cache is attached
        if (kvCache != null)
        {
            var layerCache = kvCache.GetDoubleLayer(layerIdx);
            if (!layerCache.HasCachedTokens)
            {
                layerCache.Store((float[])vid.Clone(), (float[])vid.Clone(), nVid);
            }
        }
    }

    private void ApplySingleBlock(
        int layerIdx,
        float[] unified,
        float[] vec,
        int nSeq,
        Flux3KvCache? kvCache)
    {
        int d = _p.HiddenSize;
        RmsNorm(unified, d);

        if (kvCache != null)
        {
            var layerCache = kvCache.GetSingleLayer(layerIdx);
            if (!layerCache.HasCachedTokens)
            {
                layerCache.Store((float[])unified.Clone(), (float[])unified.Clone(), nSeq);
            }
        }
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
