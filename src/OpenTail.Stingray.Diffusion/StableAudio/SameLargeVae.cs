namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Real Stable Audio 3 Medium SAME-L autoencoder (`TransformerResamplingBlock` in the real
/// `stable_audio_tools` reference, real class confirmed on GitHub `main`, NOT the older, similarly-
/// shaped `TAAEBlock` still present in the same file -- see docs/057-stable-audio-3-implementation-
/// plan.md's "CONFIDENCE RESTORED" section for the full derivation and the real trap of matching
/// the wrong class).
///
/// <para>Same real differential-attention/DynamicTanh machinery as the already golden-verified
/// Small `AcousticVae.cs` (real `differential=true` default applies to Small's VAE too, confirmed
/// this session), duplicated here rather than shared (CLAUDE.md rule 7 -- unify once this class is
/// itself verified). Structurally wider (`EmbedDim` 1536 vs 768, `NumHeads` 24 vs 12,
/// `TransformerDepth` 12 vs 6) AND uses a genuinely DIFFERENT windowing mechanism: Small's real
/// config has no `sliding_window` key (falls back to `None`), so its `TransformerResamplingBlock`
/// runs the real dual-pass ALIGNED-then-SHIFTED chunked local attention (`AcousticVae`'s own
/// `RunResamplingBlock`); Medium's real config sets `sliding_window: [1, 1]`, which the real
/// `_get_sliding_window_size` scales to `[(1*(stride+1)), (1*(stride+1))] = [17, 17]` -- when
/// `sliding_window is not None`, the real reference takes an entirely different branch: NO outer
/// chunking/splitting at all, a single pass through every transformer layer over the WHOLE folded
/// sequence, with each `Attention` call internally masked to a real banded (`±17`) local window
/// (confirmed by reading the real `TransformerResamplingBlock.forward`'s `if sliding_window is
/// None: ... else: ...` branch directly). This class implements that second, simpler (for a
/// from-scratch CPU port) mechanism -- a plain banded-softmax self-attention, the same shape as
/// this project's own `AceStepDiT` sliding-window self-attention.</para>
///
/// <para><b>Real, tensor-confirmed correction vs. assuming Small's shape</b>: Medium's DECODER
/// final mapping conv uses kernel=1 (confirmed via the real `mapping.weight_v` tensor's shape
/// `[512,1536,1]`), NOT kernel=3 like Small's -- `mapping_style="none"`/`conv_mapping=False` in the
/// real per-checkpoint config, real source `WNConv1d(..., 3 if conv_mapping else 1, ...)`. Checked
/// directly rather than copying Small's kernel assumption.</para>
///
/// <para><b>Known, deliberately deferred real gap</b>: `sinusoidal_blocks: [8]` on the real decoder
/// config (absent on the encoder) selects a real per-layer FeedForward variant
/// (`ff_kwargs={"sinusoidal": sinusoidal}`) for the LAST several decoder layers -- not yet ported
/// (this class's `FeedForward` always uses the plain SwiGLU path). Flagged, not implemented, same
/// as this project's standing practice for scoped-out real features (matches how ACE-Step's V1
/// scope explicitly deferred timbre conditioning before it was later added).</para>
/// </summary>
public sealed class SameLargeVae : IDisposable
{
    private const int PatchSize = 256;
    private const int AudioChannels = 2;
    private const int PatchedChannels = AudioChannels * PatchSize; // 512
    private const int EmbedDim = 1536; // channels(256) * c_mults[0](6)
    private const int LatentDim = 256;
    private const int Stride = 16;
    private const int SubChunkSize = Stride + 1; // 17
    private const int TransformerDepth = 12;
    private const int NumHeads = 24;
    private const int HeadDim = 64;
    private const int RopeRotDim = 32;
    private const int FfInner = 4608; // mult=3 * 1536
    private const float RopeTheta = 10000f;

    // Real `_get_sliding_window_size([1,1], stride=16) = [(1*(16+1)), (1*(16+1))] = [17,17]`.
    private const int SlidingWindowEachSide = SubChunkSize;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;

    public SameLargeVae(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
    }

    private SameLargeVae(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
    }

    public static SameLargeVae FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>Decodes real latent frames [latentSeqLen, 256] (token-major) into interleaved stereo PCM in [-1, 1].</summary>
    public float[] Decode(float[] latents, int latentSeqLen)
    {
        var x = BottleneckDecode(latents, latentSeqLen);

        var w = _st.ReadF32("pretransform.model.decoder.layers.1.weight");
        var b = _st.ReadF32("pretransform.model.decoder.layers.1.bias");
        var y = DiffusionOps.Linear(x, w, b, latentSeqLen, LatentDim, EmbedDim);

        // Real decoder padding when sliding_window is set: pad_modulo = input_seg_size = 1 (trivial, no-op).
        int n = latentSeqLen;

        var upsampled = RunResamplingBlockWindowed(y, n, "pretransform.model.decoder.layers.3", isEncoder: false);

        // Real, tensor-confirmed: Medium's decoder final mapping conv is kernel=1 (not Small's kernel=3).
        var mapped = MappingConv(
            upsampled, n * Stride, EmbedDim, PatchedChannels,
            "pretransform.model.decoder.layers.3.mapping.weight_g",
            "pretransform.model.decoder.layers.3.mapping.weight_v",
            "pretransform.model.decoder.layers.3.mapping.bias",
            kernel: 1);

        var pcm = Unpatchify(mapped, n * Stride);
        for (int i = 0; i < pcm.Length; i++) pcm[i] = Math.Clamp(pcm[i], -1f, 1f);
        return pcm;
    }

    /// <summary>Encodes interleaved stereo PCM into real latent frames [seqLen, 256] (token-major).</summary>
    public float[] Encode(ReadOnlySpan<float> pcmInterleaved, int numSamplesPerChannel)
    {
        var patched = Patchify(pcmInterleaved, numSamplesPerChannel, out int patchedLen);

        // Real encoder padding when sliding_window is set: pad_modulo = input_seg_size = Stride.
        int paddedLen = PadToMultiple(patchedLen, Stride);
        var patchedPadded = new float[paddedLen * PatchedChannels];
        patched.AsSpan().CopyTo(patchedPadded);

        var mapped = MappingConv(
            patchedPadded, paddedLen, PatchedChannels, EmbedDim,
            "pretransform.model.encoder.layers.0.mapping.weight_g",
            "pretransform.model.encoder.layers.0.mapping.weight_v",
            "pretransform.model.encoder.layers.0.mapping.bias",
            kernel: 1);

        int n = paddedLen / Stride;
        var downsampled = RunResamplingBlockWindowed(mapped, n, "pretransform.model.encoder.layers.0", isEncoder: true);

        var w = _st.ReadF32("pretransform.model.encoder.layers.2.weight");
        var b = _st.ReadF32("pretransform.model.encoder.layers.2.bias");
        var latents = DiffusionOps.Linear(downsampled, w, b, n, EmbedDim, LatentDim);

        return BottleneckEncode(latents, n);
    }

    private static float[] Patchify(ReadOnlySpan<float> pcmInterleaved, int numSamples, out int patchedLen)
    {
        patchedLen = PadToMultiple(numSamples, PatchSize) / PatchSize;
        var outp = new float[patchedLen * PatchedChannels];
        for (int l = 0; l < patchedLen; l++)
        {
            for (int c = 0; c < AudioChannels; c++)
            {
                for (int h = 0; h < PatchSize; h++)
                {
                    int t = l * PatchSize + h;
                    float v = t < numSamples ? pcmInterleaved[t * AudioChannels + c] : 0f;
                    outp[l * PatchedChannels + c * PatchSize + h] = v;
                }
            }
        }
        return outp;
    }

    private static float[] Unpatchify(float[] x, int patchedLen)
    {
        var outp = new float[patchedLen * PatchSize * AudioChannels];
        for (int l = 0; l < patchedLen; l++)
        {
            for (int c = 0; c < AudioChannels; c++)
            {
                for (int h = 0; h < PatchSize; h++)
                {
                    int t = l * PatchSize + h;
                    outp[t * AudioChannels + c] = x[l * PatchedChannels + c * PatchSize + h];
                }
            }
        }
        return outp;
    }

    private float[] BottleneckDecode(float[] latents, int n)
    {
        var runningStd = _st.ReadF32("pretransform.model.bottleneck.running_std")[0];
        var outp = new float[n * LatentDim];
        for (int t = 0; t < n; t++)
            for (int c = 0; c < LatentDim; c++)
                outp[t * LatentDim + c] = latents[t * LatentDim + c] * runningStd;
        return outp;
    }

    private float[] BottleneckEncode(float[] x, int n)
    {
        var scale = _st.ReadF32("pretransform.model.bottleneck.scaling_factor");
        var bias = _st.ReadF32("pretransform.model.bottleneck.bias");
        var runningStd = _st.ReadF32("pretransform.model.bottleneck.running_std")[0];
        var outp = new float[n * LatentDim];
        for (int t = 0; t < n; t++)
            for (int c = 0; c < LatentDim; c++)
                outp[t * LatentDim + c] = (x[t * LatentDim + c] * scale[c] + bias[c]) / runningStd;
        return outp;
    }

    private float[] MappingConv(float[] x, int seqLen, int inC, int outC, string wgKey, string wvKey, string biasKey, int kernel)
    {
        var wg = _st.ReadF32(wgKey);
        var wv = _st.ReadF32(wvKey);
        var bias = _st.ReadF32(biasKey);

        var weight = new float[outC * inC * kernel];
        for (int oc = 0; oc < outC; oc++)
        {
            double normSq = 0;
            int rowOff = oc * inC * kernel;
            for (int i = 0; i < inC * kernel; i++) normSq += (double)wv[rowOff + i] * wv[rowOff + i];
            float norm = (float)Math.Sqrt(normSq);
            float g = wg[oc];
            for (int i = 0; i < inC * kernel; i++) weight[rowOff + i] = g * wv[rowOff + i] / norm;
        }

        var outp = new float[seqLen * outC];
        int pad = kernel / 2;
        for (int t = 0; t < seqLen; t++)
        {
            for (int oc = 0; oc < outC; oc++)
            {
                float acc = bias[oc];
                for (int k = 0; k < kernel; k++)
                {
                    int srcT = t + k - pad;
                    if (srcT < 0 || srcT >= seqLen) continue;
                    int xOff = srcT * inC;
                    for (int ic = 0; ic < inC; ic++)
                        acc += weight[oc * inC * kernel + ic * kernel + k] * x[xOff + ic];
                }
                outp[t * outC + oc] = acc;
            }
        }
        return outp;
    }

    /// <summary>Real `TransformerResamplingBlock.forward`'s `sliding_window is not None` branch: fold
    /// into `(n*SubChunkSize)`-long sequence (same real construction as `AcousticVae`'s dual-pass
    /// path), then run ALL `TransformerDepth` layers in a SINGLE pass over the whole folded
    /// sequence, each with real banded (`±17`) self-attention -- no outer chunk/shift split at all
    /// (confirmed real branch difference from Small's `sliding_window is None` path).</summary>
    private float[] RunResamplingBlockWindowed(float[] input, int n, string prefix, bool isEncoder)
    {
        var newToken = _st.ReadF32($"{prefix}.new_tokens");

        var folded = new float[n * SubChunkSize * EmbedDim];
        for (int g = 0; g < n; g++)
        {
            int dst = g * SubChunkSize * EmbedDim;
            if (isEncoder)
            {
                input.AsSpan(g * Stride * EmbedDim, Stride * EmbedDim).CopyTo(folded.AsSpan(dst, Stride * EmbedDim));
                newToken.AsSpan().CopyTo(folded.AsSpan(dst + Stride * EmbedDim, EmbedDim));
            }
            else
            {
                input.AsSpan(g * EmbedDim, EmbedDim).CopyTo(folded.AsSpan(dst, EmbedDim));
                for (int i = 0; i < Stride; i++)
                    newToken.AsSpan().CopyTo(folded.AsSpan(dst + (1 + i) * EmbedDim, EmbedDim));
            }
        }

        int totalLen = n * SubChunkSize;
        var (cos, sin) = BuildPartialRope(totalLen);

        for (int li = 0; li < TransformerDepth; li++)
        {
            string layerPrefix = $"{prefix}.transformers.{li}";
            TransformerBlockForward(folded, totalLen, layerPrefix, cos, sin);
        }

        int outputSegSize = isEncoder ? 1 : Stride;
        var outp = new float[n * outputSegSize * EmbedDim];
        for (int g = 0; g < n; g++)
        {
            int srcOff = g * SubChunkSize * EmbedDim + (SubChunkSize - outputSegSize) * EmbedDim;
            int dstOff = g * outputSegSize * EmbedDim;
            folded.AsSpan(srcOff, outputSegSize * EmbedDim).CopyTo(outp.AsSpan(dstOff, outputSegSize * EmbedDim));
        }
        return outp;
    }

    private void TransformerBlockForward(float[] x, int seq, string p, float[] cos, float[] sin)
    {
        var preAlpha = _st.ReadF32($"{p}.pre_norm.alpha")[0];
        var preGamma = _st.ReadF32($"{p}.pre_norm.gamma");
        var preBeta = _st.ReadF32($"{p}.pre_norm.beta");

        var normed = x.ToArray();
        DynamicTanh(normed, preAlpha, preGamma, preBeta, EmbedDim);

        var attn = SelfAttentionDifferentialWindowed(normed, seq, p, cos, sin);
        for (int i = 0; i < x.Length; i++) x[i] += attn[i];

        var ffAlpha = _st.ReadF32($"{p}.ff_norm.alpha")[0];
        var ffGamma = _st.ReadF32($"{p}.ff_norm.gamma");
        var ffBeta = _st.ReadF32($"{p}.ff_norm.beta");

        var ffNormed = x.ToArray();
        DynamicTanh(ffNormed, ffAlpha, ffGamma, ffBeta, EmbedDim);

        var ff = FeedForward(ffNormed, seq, p);
        for (int i = 0; i < x.Length; i++) x[i] += ff[i];
    }

    private static void DynamicTanh(Span<float> x, float alpha, ReadOnlySpan<float> gamma, ReadOnlySpan<float> beta, int dim)
    {
        int n = x.Length / dim;
        for (int t = 0; t < n; t++)
        {
            var row = x.Slice(t * dim, dim);
            for (int i = 0; i < dim; i++) row[i] = gamma[i] * MathF.Tanh(alpha * row[i]) + beta[i];
        }
    }

    /// <summary>Real differential attention, same formula as `AcousticVae`'s (see its own doc
    /// comment), but with a real banded `±SlidingWindowEachSide` mask instead of Small's full
    /// (unmasked, within a fixed chunk) attention.</summary>
    private float[] SelfAttentionDifferentialWindowed(float[] x, int seq, string p, float[] cos, float[] sin)
    {
        var qkvW = _st.ReadF32($"{p}.self_attn.to_qkv.weight");
        var qNormAlpha = _st.ReadF32($"{p}.self_attn.q_norm.alpha")[0];
        var qNormGamma = _st.ReadF32($"{p}.self_attn.q_norm.gamma");
        var qNormBeta = _st.ReadF32($"{p}.self_attn.q_norm.beta");
        var kNormAlpha = _st.ReadF32($"{p}.self_attn.k_norm.alpha")[0];
        var kNormGamma = _st.ReadF32($"{p}.self_attn.k_norm.gamma");
        var kNormBeta = _st.ReadF32($"{p}.self_attn.k_norm.beta");
        var outW = _st.ReadF32($"{p}.self_attn.to_out.weight");

        var qkv = DiffusionOps.Linear(x, qkvW, null, seq, EmbedDim, 5 * EmbedDim);
        var q = new float[seq * EmbedDim];
        var k = new float[seq * EmbedDim];
        var v = new float[seq * EmbedDim];
        var qDiff = new float[seq * EmbedDim];
        var kDiff = new float[seq * EmbedDim];
        for (int t = 0; t < seq; t++)
        {
            int off = t * 5 * EmbedDim;
            qkv.AsSpan(off, EmbedDim).CopyTo(q.AsSpan(t * EmbedDim, EmbedDim));
            qkv.AsSpan(off + EmbedDim, EmbedDim).CopyTo(k.AsSpan(t * EmbedDim, EmbedDim));
            qkv.AsSpan(off + 2 * EmbedDim, EmbedDim).CopyTo(v.AsSpan(t * EmbedDim, EmbedDim));
            qkv.AsSpan(off + 3 * EmbedDim, EmbedDim).CopyTo(qDiff.AsSpan(t * EmbedDim, EmbedDim));
            qkv.AsSpan(off + 4 * EmbedDim, EmbedDim).CopyTo(kDiff.AsSpan(t * EmbedDim, EmbedDim));
        }

        PerHeadDynamicTanh(q, seq, qNormAlpha, qNormGamma, qNormBeta);
        PerHeadDynamicTanh(qDiff, seq, qNormAlpha, qNormGamma, qNormBeta);
        PerHeadDynamicTanh(k, seq, kNormAlpha, kNormGamma, kNormBeta);
        PerHeadDynamicTanh(kDiff, seq, kNormAlpha, kNormGamma, kNormBeta);

        ApplyPartialRope(q, seq, cos, sin);
        ApplyPartialRope(qDiff, seq, cos, sin);
        ApplyPartialRope(k, seq, cos, sin);
        ApplyPartialRope(kDiff, seq, cos, sin);

        var main = BandedDotProductAttention(q, k, v, seq);
        var diff = BandedDotProductAttention(qDiff, kDiff, v, seq);
        var combined = new float[main.Length];
        for (int i = 0; i < combined.Length; i++) combined[i] = main[i] - diff[i];

        return DiffusionOps.Linear(combined, outW, null, seq, EmbedDim, EmbedDim);
    }

    private static void PerHeadDynamicTanh(float[] qk, int seq, float alpha, float[] gamma, float[] beta)
    {
        for (int t = 0; t < seq; t++)
            for (int h = 0; h < NumHeads; h++)
                DynamicTanh(qk.AsSpan(t * EmbedDim + h * HeadDim, HeadDim), alpha, gamma, beta, HeadDim);
    }

    /// <summary>Banded self-attention: query position `i` only attends to keys in
    /// `[i-SlidingWindowEachSide, i+SlidingWindowEachSide]` (real flash-attn `window_size` symmetric
    /// convention) -- same shape as this project's `AceStepDiT` sliding-window attention.</summary>
    private static float[] BandedDotProductAttention(float[] q, float[] k, float[] v, int seq)
    {
        float scale = 1f / MathF.Sqrt(HeadDim);
        var outp = new float[seq * EmbedDim];
        int window = SlidingWindowEachSide;

        for (int h = 0; h < NumHeads; h++)
        {
            for (int i = 0; i < seq; i++)
            {
                int jStart = Math.Max(0, i - window);
                int jEnd = Math.Min(seq, i + window + 1);
                var scores = new float[jEnd - jStart];
                int qOff = i * EmbedDim + h * HeadDim;
                for (int j = jStart; j < jEnd; j++)
                {
                    int kOff = j * EmbedDim + h * HeadDim;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[j - jStart] = dot * scale;
                }
                DiffusionOps.Softmax(scores, scores.Length);

                int outOff = i * EmbedDim + h * HeadDim;
                for (int j = jStart; j < jEnd; j++)
                {
                    float w = scores[j - jStart];
                    if (w == 0f) continue;
                    int vOff = j * EmbedDim + h * HeadDim;
                    for (int d = 0; d < HeadDim; d++) outp[outOff + d] += w * v[vOff + d];
                }
            }
        }
        return outp;
    }

    private float[] FeedForward(float[] x, int seq, string p)
    {
        var w0 = _st.ReadF32($"{p}.ff.ff.0.proj.weight");
        var b0 = _st.ReadF32($"{p}.ff.ff.0.proj.bias");
        var w2 = _st.ReadF32($"{p}.ff.ff.2.weight");
        var b2 = _st.ReadF32($"{p}.ff.ff.2.bias");

        var proj = DiffusionOps.Linear(x, w0, b0, seq, EmbedDim, 2 * FfInner);
        var h = new float[seq * FfInner];
        for (int t = 0; t < seq; t++)
        {
            var val = proj.AsSpan(t * 2 * FfInner, FfInner);
            var gate = proj.AsSpan(t * 2 * FfInner + FfInner, FfInner);
            var dst = h.AsSpan(t * FfInner, FfInner);
            for (int i = 0; i < FfInner; i++) dst[i] = val[i] * DiffusionOps.Silu(gate[i]);
        }

        return DiffusionOps.Linear(h, w2, b2, seq, FfInner, EmbedDim);
    }

    /// <summary>Real `rope.forward_from_seq_len(seq_len)` -- unlike Small's fixed chunk-local table
    /// (no outer chunking here), positions are GLOBAL over the whole folded sequence, computed fresh
    /// per call.</summary>
    private static (float[] cos, float[] sin) BuildPartialRope(int seq)
    {
        int half = RopeRotDim / 2;
        var cos = new float[seq * half];
        var sin = new float[seq * half];
        for (int s = 0; s < seq; s++)
        {
            for (int i = 0; i < half; i++)
            {
                float invFreq = MathF.Pow(RopeTheta, -2.0f * i / RopeRotDim);
                float angle = s * invFreq;
                cos[s * half + i] = MathF.Cos(angle);
                sin[s * half + i] = MathF.Sin(angle);
            }
        }
        return (cos, sin);
    }

    private static void ApplyPartialRope(float[] qk, int seq, float[] cos, float[] sin)
    {
        int half = RopeRotDim / 2;
        for (int s = 0; s < seq; s++)
        {
            for (int h = 0; h < NumHeads; h++)
            {
                int headOff = s * EmbedDim + h * HeadDim;
                for (int i = 0; i < half; i++)
                {
                    float c = cos[s * half + i];
                    float sn = sin[s * half + i];
                    float x1 = qk[headOff + i];
                    float x2 = qk[headOff + half + i];
                    qk[headOff + i] = x1 * c - x2 * sn;
                    qk[headOff + half + i] = x1 * sn + x2 * c;
                }
            }
        }
    }

    private static int PadToMultiple(int value, int modulo) => ((value + modulo - 1) / modulo) * modulo;

    public void Dispose()
    {
        if (_ownsLoader) _st.Dispose();
    }
}
