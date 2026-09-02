namespace OpenTail.Stingray.Diffusion.StableAudio;

/// <summary>
/// Real Stable Audio 3 SAME autoencoder (`SAMEEncoder`/`SAMEDecoder`/`TransformerResamplingBlock`
/// in the real reference), transcribed directly from `stable_audio_3/models/autoencoders.py`,
/// `pretransforms.py`, and `bottleneck.py` against the real bundled checkpoint's resolved config
/// (see docs/057-stable-audio-3-implementation-plan.md's VAE section for the full derivation).
///
/// Hardcoded to this checkpoint's exact real shape (single `TransformerResamplingBlock` per side,
/// `chunk_size=32`, `stride=16`, `transformer_depth=6`, `dim_heads=64`, `differential=true`,
/// `dyt` norm, `variable_stride=true`, `chunk_midpoint_shift=true`, `sinusoidal_blocks=0`,
/// `mask_noise`/bottleneck noise-regularization deliberately NOT applied -- both are tiny-magnitude
/// (`1e-3`/`1e-2` std) real eval-time noise sources in the reference; omitting them keeps this
/// port and its golden test deterministic, matching how the reference itself is monkey-patched to
/// zero for golden-fixture generation), same shape/scope choice as other single-checkpoint ports
/// in this project (LTX/FLUX did not implement every optional branch on their first real pass).
///
/// The encoder and decoder resampling passes share the exact same dual-window local-attention
/// mechanism (`RunResamplingBlock`), only differing in how each per-group micro-chunk is built
/// (real vs. learned tokens, and their order) and where the input has real vs. inserted content.
/// </summary>
public sealed class AcousticVae : IDisposable
{
    private const int PatchSize = 256;
    private const int AudioChannels = 2;
    private const int PatchedChannels = AudioChannels * PatchSize; // 512
    private const int EmbedDim = 768; // resampling-block channel dim, both encoder and decoder
    private const int LatentDim = 256;
    private const int ChunkSize = 32;
    private const int Stride = 16;
    private const int SubChunkSize = Stride + 1; // 17
    private const int EffectiveChunkSize = 2 * SubChunkSize; // 34
    private const int Shift = EffectiveChunkSize / 2; // 17
    private const int TransformerDepth = 6;
    private const int SplitAt = TransformerDepth / 2; // 3
    private const int NumHeads = 12;
    private const int HeadDim = 64;
    private const int RopeRotDim = 32;
    private const int FfInner = 2304; // mult=3 * 768
    private const float RopeTheta = 10000f;

    private readonly IWeightLoader _st;
    private readonly bool _ownsLoader;
    private readonly (float[] cos, float[] sin) _windowRope;

    public AcousticVae(string path)
    {
        _st = SafetensorsLoader.Open(path);
        _ownsLoader = true;
        _windowRope = BuildPartialRope(EffectiveChunkSize);
    }

    private AcousticVae(IWeightLoader loader, bool ownsLoader)
    {
        _st = loader;
        _ownsLoader = ownsLoader;
        _windowRope = BuildPartialRope(EffectiveChunkSize);
    }

    public static AcousticVae FromLoader(IWeightLoader loader) => new(loader, ownsLoader: false);

    /// <summary>Decodes real latent frames [latentSeqLen, 256] (token-major, real DiT/rectified-flow
    /// output convention) into interleaved stereo PCM in [-1, 1].</summary>
    public float[] Decode(float[] latents, int latentSeqLen)
    {
        var x = BottleneckDecode(latents, latentSeqLen);

        var w = _st.ReadF32("pretransform.model.decoder.layers.1.weight");
        var b = _st.ReadF32("pretransform.model.decoder.layers.1.bias");
        var y = DiffusionOps.Linear(x, w, b, latentSeqLen, LatentDim, EmbedDim);

        int n = PadToMultiple(latentSeqLen, ChunkSize / Stride);
        var yPadded = new float[n * EmbedDim];
        y.AsSpan().CopyTo(yPadded);

        var upsampled = RunResamplingBlock(yPadded, n, "pretransform.model.decoder.layers.3", isEncoder: false);

        var mapped = MappingConv(
            upsampled, n * Stride, EmbedDim, PatchedChannels,
            "pretransform.model.decoder.layers.3.mapping.weight_g",
            "pretransform.model.decoder.layers.3.mapping.weight_v",
            "pretransform.model.decoder.layers.3.mapping.bias",
            kernel: 3);

        var pcm = Unpatchify(mapped, n * Stride);
        for (int i = 0; i < pcm.Length; i++) pcm[i] = Math.Clamp(pcm[i], -1f, 1f);
        return pcm;
    }

    /// <summary>Encodes interleaved stereo PCM into real latent frames [seqLen, 256] (token-major).</summary>
    public float[] Encode(ReadOnlySpan<float> pcmInterleaved, int numSamplesPerChannel)
    {
        var patched = Patchify(pcmInterleaved, numSamplesPerChannel, out int patchedLen);

        int paddedLen = PadToMultiple(patchedLen, ChunkSize);
        var patchedPadded = new float[paddedLen * PatchedChannels];
        patched.AsSpan().CopyTo(patchedPadded);

        var mapped = MappingConv(
            patchedPadded, paddedLen, PatchedChannels, EmbedDim,
            "pretransform.model.encoder.layers.0.mapping.weight_g",
            "pretransform.model.encoder.layers.0.mapping.weight_v",
            "pretransform.model.encoder.layers.0.mapping.bias",
            kernel: 1);

        int n = paddedLen / Stride;
        var downsampled = RunResamplingBlock(mapped, n, "pretransform.model.encoder.layers.0", isEncoder: true);

        var w = _st.ReadF32("pretransform.model.encoder.layers.2.weight");
        var b = _st.ReadF32("pretransform.model.encoder.layers.2.bias");
        var latents = DiffusionOps.Linear(downsampled, w, b, n, EmbedDim, LatentDim);

        return BottleneckEncode(latents, n);
    }

    // ── Patchify / unpatchify: real `PatchedPretransform` (no oversampling, no postfilter here) ──

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

    // ── Bottleneck: real `SoftNormBottleneck` (noise-regularization deliberately omitted, see class doc) ──

    private float[] BottleneckDecode(float[] latents, int n)
    {
        var runningStd = _st.ReadF32("pretransform.model.bottleneck.running_std")[0];
        var outp = new float[n * LatentDim];
        for (int t = 0; t < n; t++)
        {
            for (int c = 0; c < LatentDim; c++)
            {
                outp[t * LatentDim + c] = latents[t * LatentDim + c] * runningStd;
            }
        }
        return outp;
    }

    private float[] BottleneckEncode(float[] x, int n)
    {
        var scale = _st.ReadF32("pretransform.model.bottleneck.scaling_factor"); // [1,256,1] -> 256
        var bias = _st.ReadF32("pretransform.model.bottleneck.bias"); // [1,256,1] -> 256
        var runningStd = _st.ReadF32("pretransform.model.bottleneck.running_std")[0];
        var outp = new float[n * LatentDim];
        for (int t = 0; t < n; t++)
        {
            for (int c = 0; c < LatentDim; c++)
            {
                outp[t * LatentDim + c] = (x[t * LatentDim + c] * scale[c] + bias[c]) / runningStd;
            }
        }
        return outp;
    }

    // ── Real `WNConv1d` (PyTorch weight_norm, dim=0: one g-scalar and one v-vector per out channel) ──

    private float[] MappingConv(float[] x, int seqLen, int inC, int outC, string wgKey, string wvKey, string biasKey, int kernel)
    {
        var wg = _st.ReadF32(wgKey); // [outC,1,1] -> outC
        var wv = _st.ReadF32(wvKey); // [outC,inC,kernel]
        var bias = _st.ReadF32(biasKey); // [outC]

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
        int pad = kernel / 2; // 'same' padding
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
                    {
                        acc += weight[oc * inC * kernel + ic * kernel + k] * x[xOff + ic];
                    }
                }
                outp[t * outC + oc] = acc;
            }
        }
        return outp;
    }

    // ── Real `TransformerResamplingBlock` (encoder & decoder share this exact windowed dual-pass
    // local-attention mechanism -- only micro-chunk assembly/extraction differs) ──

    private float[] RunResamplingBlock(float[] input, int n, string prefix, bool isEncoder)
    {
        // n: number of stride-wide micro-groups (real `n` in the reference) -- for the encoder
        // direction this is `paddedPatchedLen / Stride`; for the decoder it is simply the (padded)
        // latent token count, since decoder micro-groups are single input tokens.
        var newToken = _st.ReadF32($"{prefix}.new_tokens"); // [1,1,768] -- broadcast to however many are needed

        // Build the folded (n * SubChunkSize)-long sequence: encoder = [16 real, 1 new] per group;
        // decoder = [1 real, 16 new] per group (real cat() order, see class doc derivation).
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
                {
                    newToken.AsSpan().CopyTo(folded.AsSpan(dst + (1 + i) * EmbedDim, EmbedDim));
                }
            }
        }

        int totalLen = n * SubChunkSize;

        // First pass: layers[0..SplitAt), aligned 34-wide windows.
        RunChunkedPass(folded, totalLen, prefix, layerStart: 0, layerCount: SplitAt);

        // Second pass: layers[SplitAt..depth), windows shifted by `Shift`, via edge-repeat padding.
        var shifted = new float[(totalLen + 2 * Shift) * EmbedDim];
        folded.AsSpan(0, Shift * EmbedDim).CopyTo(shifted.AsSpan(0, Shift * EmbedDim));
        folded.AsSpan().CopyTo(shifted.AsSpan(Shift * EmbedDim, totalLen * EmbedDim));
        folded.AsSpan((totalLen - Shift) * EmbedDim, Shift * EmbedDim).CopyTo(shifted.AsSpan((Shift + totalLen) * EmbedDim, Shift * EmbedDim));

        RunChunkedPass(shifted, totalLen + 2 * Shift, prefix, layerStart: SplitAt, layerCount: TransformerDepth - SplitAt);

        shifted.AsSpan(Shift * EmbedDim, totalLen * EmbedDim).CopyTo(folded);

        // Extract per-group output: encoder keeps only the last (new-token) position; decoder keeps
        // the last `Stride` positions (the new-tokens' outputs, skipping the real input token).
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

    private void RunChunkedPass(float[] x, int totalLen, string prefix, int layerStart, int layerCount)
    {
        int nc = totalLen / EffectiveChunkSize;
        for (int li = 0; li < layerCount; li++)
        {
            string layerPrefix = $"{prefix}.transformers.{layerStart + li}";
            for (int c = 0; c < nc; c++)
            {
                TransformerBlockForward(x.AsSpan(c * EffectiveChunkSize * EmbedDim, EffectiveChunkSize * EmbedDim), layerPrefix);
            }
        }
    }

    /// <summary>Real `TransformerBlock.forward`, no-adaLN / no-cross-attn / no-conformer branch:
    /// `x = x + self_attn(pre_norm(x)); x = x + ff(ff_norm(x))`, both norms `DynamicTanh`.</summary>
    private void TransformerBlockForward(Span<float> chunk, string p)
    {
        var preAlpha = _st.ReadF32($"{p}.pre_norm.alpha")[0];
        var preGamma = _st.ReadF32($"{p}.pre_norm.gamma");
        var preBeta = _st.ReadF32($"{p}.pre_norm.beta");

        var normed = chunk.ToArray();
        DynamicTanh(normed, preAlpha, preGamma, preBeta, EmbedDim);

        var attn = SelfAttentionDifferential(normed, EffectiveChunkSize, p);
        for (int i = 0; i < chunk.Length; i++) chunk[i] += attn[i];

        var ffAlpha = _st.ReadF32($"{p}.ff_norm.alpha")[0];
        var ffGamma = _st.ReadF32($"{p}.ff_norm.gamma");
        var ffBeta = _st.ReadF32($"{p}.ff_norm.beta");

        var ffNormed = chunk.ToArray();
        DynamicTanh(ffNormed, ffAlpha, ffGamma, ffBeta, EmbedDim);

        var ff = FeedForward(ffNormed, EffectiveChunkSize, p);
        for (int i = 0; i < chunk.Length; i++) chunk[i] += ff[i];
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

    /// <summary>Real differential attention (`Attention.forward`, `differential=true`): fused
    /// `to_qkv` splits into q,k,v,q_diff,k_diff (in that order); the same `q_norm`/`k_norm`
    /// (`DynamicTanh`) and the same RoPE apply identically to both the primary and "diff" pathways;
    /// `out = attn(q,k,v) - attn(q_diff,k_diff,v)` (same V both times, plain subtraction, no
    /// learned lambda).</summary>
    private float[] SelfAttentionDifferential(float[] x, int seq, string p)
    {
        var qkvW = _st.ReadF32($"{p}.self_attn.to_qkv.weight"); // [3840,768] = 5*768
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

        ApplyPartialRope(q, seq, _windowRope.cos, _windowRope.sin);
        ApplyPartialRope(qDiff, seq, _windowRope.cos, _windowRope.sin);
        ApplyPartialRope(k, seq, _windowRope.cos, _windowRope.sin);
        ApplyPartialRope(kDiff, seq, _windowRope.cos, _windowRope.sin);

        var main = DotProductAttention(q, k, v, seq);
        var diff = DotProductAttention(qDiff, kDiff, v, seq);
        var combined = new float[main.Length];
        for (int i = 0; i < combined.Length; i++) combined[i] = main[i] - diff[i];

        return DiffusionOps.Linear(combined, outW, null, seq, EmbedDim, EmbedDim);
    }

    private static void PerHeadDynamicTanh(float[] qk, int seq, float alpha, float[] gamma, float[] beta)
    {
        for (int t = 0; t < seq; t++)
        {
            for (int h = 0; h < NumHeads; h++)
            {
                DynamicTanh(qk.AsSpan(t * EmbedDim + h * HeadDim, HeadDim), alpha, gamma, beta, HeadDim);
            }
        }
    }

    private static float[] DotProductAttention(float[] q, float[] k, float[] v, int seq)
    {
        float scale = 1f / MathF.Sqrt(HeadDim);
        var outp = new float[seq * EmbedDim];
        for (int h = 0; h < NumHeads; h++)
        {
            var scores = new float[seq * seq];
            for (int i = 0; i < seq; i++)
            {
                int qOff = i * EmbedDim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    int kOff = j * EmbedDim + h * HeadDim;
                    float dot = 0f;
                    for (int d = 0; d < HeadDim; d++) dot += q[qOff + d] * k[kOff + d];
                    scores[i * seq + j] = dot * scale;
                }
            }
            DiffusionOps.Softmax(scores, seq);

            for (int i = 0; i < seq; i++)
            {
                int outOff = i * EmbedDim + h * HeadDim;
                for (int j = 0; j < seq; j++)
                {
                    float w = scores[i * seq + j];
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

    /// <summary>Same partial GPT-J-style rotary scheme as <c>StableAudioDiT</c> (only the first
    /// <see cref="RopeRotDim"/> of each 64-wide head vector rotate) -- but positions here are LOCAL
    /// to each windowed chunk (real per-`TransformerBlock` `self.rope.forward_from_seq_len` call,
    /// not a global sequence position), so a single cos/sin table sized to
    /// <see cref="EffectiveChunkSize"/> is reused for every chunk instance.</summary>
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
