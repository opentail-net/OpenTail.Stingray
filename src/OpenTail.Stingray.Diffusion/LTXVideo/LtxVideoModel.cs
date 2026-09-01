using OpenTail.Stingray.Core;

namespace OpenTail.Stingray.Diffusion.LTXVideo;

/// <summary>
/// Native LTX-Video (Lightricks) Transformer Backbone -- real checkpoint-driven port.
/// Reference: stable-diffusion.cpp:src/model/diffusion/ltxv.hpp (LTXAVConfig::detect_from_weights,
/// BasicTransformerBlock, CrossAttention, AdaLayerNormSingle, PixArtAlphaTextProjection).
///
/// Tensor shapes below verified directly against `ltx-video-2b-v0.9.1.safetensors`'s own
/// safetensors JSON header (see docs/055-ltx-video-implementation-plan.md) -- config is read FROM
/// the checkpoint (<see cref="DetectConfig"/>), not hardcoded, matching this project's own
/// established convention (<c>WanModel.DetectConfig</c>).
///
/// Scope of this pass: the DiT transformer core only (steps 0-4 of the implementation plan's
/// build order) -- patchify/caption projections, AdaLN-single timestep branch, continuous 3D RoPE,
/// and the 28-block transformer + final projection. NOT yet wired to a real T5-v1.1-XXL encoder
/// (google/t5-v1_1-xxl is not downloaded locally) or the VAE decoder (timestep-conditioned, its
/// own separate research pass per the plan doc) -- both remain deferred, per the plan.
/// </summary>
public sealed class LtxVideoModel : IDisposable
{
    private readonly IWeightLoader _weights;
    private readonly string _prefix;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private bool _disposed;

    public int InChannels { get; }
    public int OutChannels { get; }
    public int HiddenSize { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int NumLayers { get; }
    public int CrossAttentionDim { get; }
    public int CaptionChannels { get; }
    public bool CrossAttentionAdaln { get; }
    public bool SelfAttentionGated { get; }
    public bool CrossAttentionGated { get; }
    public float NormEps { get; } = 1e-6f;
    public float QkNormEps { get; } = 1e-5f;
    public float RopeTheta { get; } = 10000.0f;
    public float TimestepScale { get; } = 1000.0f;

    // Intermediate-tensor capture for golden/numeric-parity tests (LtxVideoGoldenParityTests) --
    // NOT used by the real inference path, populated unconditionally on each Forward() call since
    // the cost is negligible next to the transformer itself.
    internal float[]? LastProjInOut { get; private set; }
    internal float[]? LastCaptionProjOut { get; private set; }
    internal float[]? LastEmbeddedTimestep { get; private set; }
    internal float[]? LastTimestepProj { get; private set; }
    internal float[]? LastRopeCos { get; private set; }
    internal float[]? LastRopeSin { get; private set; }
    internal float[]? LastBlock0Out { get; private set; }

    public LtxVideoModel(IWeightLoader weights, string prefix = "model.diffusion_model")
    {
        _weights = weights;
        _prefix = prefix.Length > 0 && !prefix.EndsWith('.') ? prefix + "." : prefix;

        (InChannels, HiddenSize, NumHeads, HeadDim, OutChannels, CrossAttentionDim, CaptionChannels,
            NumLayers, CrossAttentionAdaln, SelfAttentionGated, CrossAttentionGated) = DetectConfig(_weights, _prefix);
    }

    /// <summary>Mirrors `LTXAVConfig::detect_from_weights` -- every dimension read from the real
    /// checkpoint's tensor shapes, not assumed. `patchify_proj.weight` is `[hiddenSize, inChannels]`
    /// row-major (GGUF/safetensors `Linear.weight` convention: `ne[1]=inFeatures` is NOT what's
    /// used here -- this project's `ReadF32` + row-major `Linear` helper below reads it as
    /// `[outDim, inDim]`, i.e. `weight.Length / outDim == inDim`).</summary>
    private static (int inCh, int hidden, int heads, int headDim, int outCh, int crossDim, int captionCh,
        int layers, bool crossAdaln, bool selfGated, bool crossGated) DetectConfig(IWeightLoader w, string prefix)
    {
        int inCh = 128, hidden = 2048, heads = 32, headDim = 64, outCh = 128, crossDim = 2048, captionCh = 4096;

        string patchKey = prefix + "patchify_proj.weight";
        if (w.Contains(patchKey))
        {
            var pw = w.ReadF32(patchKey);
            // Real tensor shape [hiddenSize, inChannels]. `IWeightLoader.ReadF32` returns only a
            // flat buffer (no shape metadata), so -- same approach as `WanModel.DetectConfig`
            // dividing by its known-fixed `InChannels` constant -- treat in_channels as the fixed
            // 128 the real config always uses for both video and audio variants, and derive
            // hidden_size from the flat length.
            inCh = 128;
            hidden = pw.Length / inCh;
        }

        int gateHeads = TryInferGateHeads(w, prefix + "transformer_blocks.0.attn1.to_gate_logits.bias", heads);
        (heads, headDim) = InferAttentionLayout(hidden, gateHeads);

        string projOutKey = prefix + "proj_out.weight";
        if (w.Contains(projOutKey))
        {
            var pw = w.ReadF32(projOutKey);
            outCh = pw.Length / hidden;
        }

        string attn2KKey = prefix + "transformer_blocks.0.attn2.to_k.weight";
        if (w.Contains(attn2KKey))
        {
            var kw = w.ReadF32(attn2KKey);
            crossDim = kw.Length / hidden;
        }

        string capProjKey = prefix + "caption_projection.linear_1.weight";
        if (w.Contains(capProjKey))
        {
            var cw = w.ReadF32(capProjKey);
            captionCh = cw.Length / hidden;
        }

        bool crossAdaln = w.Contains(prefix + "transformer_blocks.0.prompt_scale_shift_table");
        bool selfGated = w.Contains(prefix + "transformer_blocks.0.attn1.to_gate_logits.weight");
        bool crossGated = w.Contains(prefix + "transformer_blocks.0.attn2.to_gate_logits.weight");

        int layers = 0;
        for (int i = 0; i < 128; i++)
        {
            if (!w.Contains($"{prefix}transformer_blocks.{i}.attn1.to_q.weight")) break;
            layers = i + 1;
        }
        if (layers == 0) layers = 28;

        return (inCh, hidden, heads, headDim, outCh, crossDim, captionCh, layers, crossAdaln, selfGated, crossGated);
    }

    private static int TryInferGateHeads(IWeightLoader w, string biasName, int fallback)
    {
        if (!w.Contains(biasName)) return fallback;
        var b = w.ReadF32(biasName);
        return b.Length;
    }

    /// <summary>Real `LTXAVConfig::infer_attention_layout` -- picks the largest "nice" head_dim
    /// candidate that divides hiddenSize evenly with a head count in [8,64].</summary>
    private static (int heads, int headDim) InferAttentionLayout(int hiddenSize, int preferredHeads)
    {
        if (preferredHeads > 0 && hiddenSize % preferredHeads == 0)
            return (preferredHeads, hiddenSize / preferredHeads);
        foreach (int headDim in new[] { 128, 96, 80, 64, 48, 40, 32 })
        {
            if (hiddenSize % headDim == 0)
            {
                int heads = hiddenSize / headDim;
                if (heads is >= 8 and <= 64) return (heads, headDim);
            }
        }
        return (32, hiddenSize / 32);
    }

    private string Resolve(string name)
    {
        string direct = _prefix + name;
        if (_weights.Contains(direct)) return direct;
        return direct;
    }

    private float[] GetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        var data = _weights.ReadF32(fullName);
        _weightCache[fullName] = data;
        return data;
    }

    private float[]? TryGetWeight(string name)
    {
        string fullName = Resolve(name);
        if (_weightCache.TryGetValue(fullName, out var cached)) return cached;
        if (_weights.Contains(fullName))
        {
            var data = _weights.ReadF32(fullName);
            _weightCache[fullName] = data;
            return data;
        }
        return null;
    }

    /// <summary>
    /// Executes one forward denoising-velocity step of the LTX-Video DiT.
    /// </summary>
    /// <param name="latents">Input video latent tokens, already flattened to
    /// [numFrames*patchH*patchW, InChannels] (patch_size=1/patch_size_t=1 means this is a plain
    /// reshape of the VAE-encoded latent -- no spatial/temporal patch merge at this boundary).</param>
    /// <param name="timestep">Diffusion timestep t, expected pre-scaled by <see cref="TimestepScale"/>
    /// the same way the reference's own `timestep_scale_multiplier` config entry applies it.</param>
    /// <param name="captionEmbeds">Text conditioning tokens [textSeqLen, CaptionChannels] -- real
    /// T5-v1.1-XXL output (4096-dim); projected down to HiddenSize internally via
    /// `caption_projection`, per the real checkpoint (NOT pre-projected by the caller).</param>
    public float[] Forward(
        ReadOnlySpan<float> latents,
        float timestep,
        ReadOnlySpan<float> captionEmbeds,
        int numFrames,
        int patchH,
        int patchW)
    {
        int numTokens = numFrames * patchH * patchW;
        int d = HiddenSize;
        int numTxt = CaptionChannels > 0 ? captionEmbeds.Length / CaptionChannels : 0;

        // 1. patchify_proj: inChannels -> hiddenSize (real Linear, has bias)
        var x = Linear("patchify_proj", latents, InChannels, d);

        // 2. caption_projection: PixArtAlphaTextProjection(4096 -> 2048 -> 2048, GELU between)
        float[] captionProj = Array.Empty<float>();
        if (numTxt > 0)
        {
            var c1 = Linear("caption_projection.linear_1", captionEmbeds, CaptionChannels, d);
            DiffusionOps.GeluInPlace(c1);
            captionProj = Linear("caption_projection.linear_2", c1, d, d);
        }

        // 3. AdaLayerNormSingle: sinusoidal timestep embed -> Linear -> SiLU -> Linear(dim -> 6*dim)
        // Real: `embedded_timestep = TimestepEmbedder(t)`; `hidden = silu(embedded_timestep)`;
        // `out = linear(hidden)` -- the shared per-block modulation input added to each block's OWN
        // `scale_shift_table` constant (NOT itself the per-block modulation).
        var embeddedTimestep = TimestepEmbedder(timestep);
        var siluEmb = (float[])embeddedTimestep.Clone();
        DiffusionOps.SiluInPlace(siluEmb);
        var timestepProj = Linear("adaln_single.linear", siluEmb, d, d * 6);

        // 4. Continuous 3D RoPE (self-attention only; cross-attention gets none). Real:
        // `LTXVideoRotaryPosEmbed(dim=inner_dim, ...)` -- the rotation is computed over the FULL
        // hidden dim (2048), applied to q/k BEFORE the conceptual head split (`apply_rotary_emb`
        // runs on the un-split `[B,S,inner_dim]` tensor) -- NOT a per-head-width table repeated
        // identically across heads the way Wan's RoPE works. Confirmed directly against
        // `LTXVideoRotaryPosEmbed.__init__`'s `dim` argument in diffusers' real transformer_ltx.py
        // (found via golden-tensor mismatch: a per-head-width table gave near-zero cosine
        // similarity against the real reference's actual rope_cos/rope_sin dump).
        var (ropeCos, ropeSin) = LtxVideoRoPE.ComputeContinuous3DRoPE(numFrames, patchH, patchW, d, RopeTheta);

        LastProjInOut = (float[])x.Clone();
        LastCaptionProjOut = captionProj;
        LastEmbeddedTimestep = embeddedTimestep;
        LastTimestepProj = timestepProj;
        LastRopeCos = ropeCos;
        LastRopeSin = ropeSin;

        // 5. Transformer blocks
        for (int layer = 0; layer < NumLayers; layer++)
        {
            x = TransformerBlock($"transformer_blocks.{layer}", x, timestepProj, ropeCos, ropeSin,
                captionProj, numTokens, numTxt);
            if (layer == 0) LastBlock0Out = (float[])x.Clone();
        }

        // 6. Final layer: real `nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)`
        // (mean+variance normalized -- NOT RMSNorm, unlike every other norm in this model; confirmed
        // directly against `LTXVideoTransformer3DModel.norm_out` in diffusers' real
        // transformer_ltx.py) -> AdaLN-modulate with TOP-LEVEL scale_shift_table [2,hidden]
        // additively combined with the raw embedded_timestep -> proj_out (Linear dim->out)
        var topTable = GetWeight("scale_shift_table"); // [2, d]: shift, scale
        var finalShift = new float[d];
        var finalScale = new float[d];
        for (int i = 0; i < d; i++)
        {
            finalShift[i] = topTable[i] + embeddedTimestep[i];
            finalScale[i] = topTable[d + i] + embeddedTimestep[i];
        }

        var normed = (float[])x.Clone();
        LayerNormNoAffine(normed, d, NormEps);
        var modulated = Modulate(normed, numTokens, d, finalShift, finalScale);

        return Linear("proj_out", modulated, d, OutChannels);
    }

    private float[] TransformerBlock(
        string prefix,
        float[] x,
        float[] timestepProj,
        float[] ropeCos,
        float[] ropeSin,
        float[] context,
        int numTokens,
        int numTxt)
    {
        int d = HiddenSize;
        // Real: `mods = scale_shift_table[dim,6] + timestepProj[dim,6]` (broadcast add per-token),
        // chunked into shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp.
        var table = GetWeight($"{prefix}.scale_shift_table"); // [6, d] (or [9,d] if cross_attention_adaln)

        // 1. Self-attention: RMSNorm (non-affine) -> AdaLN modulate -> attn1 (full-width QK-norm,
        // continuous 3D RoPE) -> gated residual.
        var norm1 = (float[])x.Clone();
        RmsNormNoAffine(norm1, d, NormEps);
        var (shiftMsa, scaleMsa, gateMsa) = GetModTriple(table, timestepProj, numTokens, d, 0);
        var normed1 = Modulate(norm1, numTokens, d, shiftMsa, scaleMsa);
        var selfAttn = Attention($"{prefix}.attn1", normed1, normed1, ropeCos, ropeSin, numTokens, numTokens, applyRope: true);
        ApplyGatedResidual(x, selfAttn, numTokens, d, gateMsa);

        // 2. Cross-attention with the caption sequence: real reference runs attn2 on RAW x (no
        // pre-norm), K/V from the SAME projected caption_projection output at every block, no RoPE,
        // plain (ungated) residual add -- `cross_attention_adaln=false` for the real checkpoint
        // (no `prompt_scale_shift_table` tensor found), so the simpler branch applies.
        if (numTxt > 0)
        {
            var crossAttn = Attention($"{prefix}.attn2", x, context, null, null, numTokens, numTxt, applyRope: false);
            for (int i = 0; i < x.Length; i++) x[i] += crossAttn[i];
        }

        // 3. FFN: RMSNorm (non-affine) -> AdaLN modulate -> ordinary 2-layer GELU MLP (2048->8192->2048,
        // NOT gated SiLU) -> gated residual.
        var norm2 = (float[])x.Clone();
        RmsNormNoAffine(norm2, d, NormEps);
        var (shiftMlp, scaleMlp, gateMlp) = GetModTriple(table, timestepProj, numTokens, d, 3);
        var normed2 = Modulate(norm2, numTokens, d, shiftMlp, scaleMlp);
        var ffn = FeedForward($"{prefix}.ff", normed2);
        ApplyGatedResidual(x, ffn, numTokens, d, gateMlp);

        return x;
    }

    /// <summary>Computes one (shift, scale, gate) triple of the 6-way `scale_shift_table + timestepProj`
    /// modulation, per-token (both operands are additively broadcast per real reference semantics --
    /// `table` is a constant [6,d] shared across tokens, `timestepProj` varies with the (currently
    /// single, batch=1) timestep but is likewise shared across all tokens of this forward call).</summary>
    private static (float[] shift, float[] scale, float[] gate) GetModTriple(
        float[] table, float[] timestepProj, int numTokens, int d, int startChunk)
    {
        var shift = new float[d];
        var scale = new float[d];
        var gate = new float[d];
        int shiftOff = startChunk * d;
        int scaleOff = (startChunk + 1) * d;
        int gateOff = (startChunk + 2) * d;
        for (int i = 0; i < d; i++)
        {
            shift[i] = table[shiftOff + i] + timestepProj[shiftOff + i];
            scale[i] = table[scaleOff + i] + timestepProj[scaleOff + i];
            gate[i] = table[gateOff + i] + timestepProj[gateOff + i];
        }
        return (shift, scale, gate);
    }

    private float[] Attention(string prefix, float[] xQuery, float[] xContext,
        float[]? ropeCos, float[]? ropeSin, int qSeq, int kvSeq, bool applyRope)
    {
        int d = HiddenSize;
        var q = Linear($"{prefix}.to_q", xQuery, d, d);
        var k = Linear($"{prefix}.to_k", xContext, d, d);
        var v = Linear($"{prefix}.to_v", xContext, d, d);

        // Real: RMSNorm(inner_dim=hidden, eps=1e-5), FULL WIDTH (all heads concatenated), applied
        // BEFORE the conceptual head split -- same convention this project already established for
        // Wan (`WanModel.RmsNormHeads`'s doc comment), reused here rather than a per-head version.
        var qNorm = GetWeight($"{prefix}.q_norm.weight");
        var kNorm = GetWeight($"{prefix}.k_norm.weight");
        DiffusionOps.RmsNorm(q, qNorm, d, QkNormEps);
        DiffusionOps.RmsNorm(k, kNorm, d, QkNormEps);

        // Real: `apply_rotary_emb` runs on q/k BEFORE the head split (`x.unflatten(2,(heads,-1))`
        // happens only afterward, in the attention processor) -- the rotation table spans the FULL
        // hidden dim, so apply it as ONE "head" of width `d`, not per-head with a repeated table.
        if (applyRope && ropeCos is not null && ropeSin is not null)
        {
            LtxVideoRoPE.ApplyRoPE(q, ropeCos, ropeSin, qSeq, numHeads: 1, headDim: d);
            LtxVideoRoPE.ApplyRoPE(k, ropeCos, ropeSin, kvSeq, numHeads: 1, headDim: d);
        }

        var attnOut = MultiHeadAttention(q, k, v, qSeq, kvSeq, NumHeads, HeadDim);
        return Linear($"{prefix}.to_out.0", attnOut, d, d);
    }

    private static float[] MultiHeadAttention(float[] q, float[] k, float[] v, int qSeq, int kvSeq, int numHeads, int headDim)
    {
        float scale = 1.0f / MathF.Sqrt(headDim);
        var output = new float[qSeq * numHeads * headDim];

        Parallel.For(0, numHeads, h =>
        {
            var scores = new float[kvSeq];
            for (int i = 0; i < qSeq; i++)
            {
                int qRow = (i * numHeads + h) * headDim;
                var qSpan = q.AsSpan(qRow, headDim);
                float maxScore = float.NegativeInfinity;

                for (int j = 0; j < kvSeq; j++)
                {
                    int kRow = (j * numHeads + h) * headDim;
                    float dot = TensorPrimitives.Dot(qSpan, k.AsSpan(kRow, headDim)) * scale;
                    scores[j] = dot;
                    if (dot > maxScore) maxScore = dot;
                }

                float sumExp = 0f;
                for (int j = 0; j < kvSeq; j++)
                {
                    scores[j] = MathF.Exp(scores[j] - maxScore);
                    sumExp += scores[j];
                }
                float invSum = 1f / sumExp;

                int outRow = (i * numHeads + h) * headDim;
                for (int dIdx = 0; dIdx < headDim; dIdx++)
                {
                    float sum = 0f;
                    for (int j = 0; j < kvSeq; j++)
                        sum += scores[j] * v[(j * numHeads + h) * headDim + dIdx];
                    output[outRow + dIdx] = sum * invSum;
                }
            }
        });

        return output;
    }

    /// <summary>Ordinary 2-layer GELU MLP, 2048->8192->2048 -- real `ff.net.0.proj` / `ff.net.2`
    /// tensor names (`FeedForward(dim, dim, mult=4, Activation::GELU)`), NOT a gated SiLU FFN like
    /// Wan's (`w1`/`w2`/`w3`) -- confirmed by the checkpoint's own tensor inventory (no `w3`).</summary>
    private float[] FeedForward(string prefix, float[] x)
    {
        int d = HiddenSize;
        int ffnDim = d * 4;
        var h1 = Linear($"{prefix}.net.0.proj", x, d, ffnDim);
        DiffusionOps.GeluInPlace(h1);
        return Linear($"{prefix}.net.2", h1, ffnDim, d);
    }

    /// <summary>Real `TimestepEmbedder`: sinusoidal(256) -> Linear(256,hidden) -> SiLU -> Linear(hidden,hidden).</summary>
    private float[] TimestepEmbedder(float timestep)
    {
        const int freqEmbedSize = 256;
        var emb = new float[freqEmbedSize];
        int half = freqEmbedSize / 2;
        float scaledT = timestep;
        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-MathF.Log(10000.0f) * i / half);
            float angle = scaledT * freq;
            emb[i] = MathF.Cos(angle);
            emb[half + i] = MathF.Sin(angle);
        }

        var h1 = Linear("adaln_single.emb.timestep_embedder.linear_1", emb, freqEmbedSize, HiddenSize);
        DiffusionOps.SiluInPlace(h1);
        return Linear("adaln_single.emb.timestep_embedder.linear_2", h1, HiddenSize, HiddenSize);
    }

    /// <summary>Block-level `RMSNorm(dim, eps, elementwise_affine=False)` (no learned weight) --
    /// used for `norm1`/`norm2` inside each transformer block. Distinct from the FULL-WIDTH,
    /// weighted q_norm/k_norm RMSNorm used inside attention, and from the model's own final
    /// `norm_out` (a real mean-centered LayerNorm, not RMSNorm -- see <see cref="LayerNormNoAffine"/>).</summary>
    private static void RmsNormNoAffine(float[] x, int dim, float eps)
    {
        int n = x.Length / dim;
        Parallel.For(0, n, row =>
        {
            var rowSpan = x.AsSpan(row * dim, dim);
            float sumSq = TensorPrimitives.SumOfSquares(rowSpan);
            float invRms = 1f / MathF.Sqrt(sumSq / dim + eps);
            TensorPrimitives.Multiply(rowSpan, invRms, rowSpan);
        });
    }

    /// <summary>Real `nn.LayerNorm(dim, eps, elementwise_affine=False)` -- mean-centered AND
    /// variance-normalized (unlike every other norm in this model, which is RMS-only). Used ONLY
    /// for the model's final `norm_out`, confirmed directly against
    /// `LTXVideoTransformer3DModel.norm_out` in diffusers' real `transformer_ltx.py` -- a
    /// plausible-looking RMSNorm substitution here would silently corrupt the final projection's
    /// input distribution.</summary>
    private static void LayerNormNoAffine(float[] x, int dim, float eps)
    {
        int n = x.Length / dim;
        Parallel.For(0, n, row =>
        {
            var rowSpan = x.AsSpan(row * dim, dim);
            float mean = TensorPrimitives.Sum(rowSpan) / dim;
            TensorPrimitives.Subtract(rowSpan, mean, rowSpan);
            float sumSq = TensorPrimitives.SumOfSquares(rowSpan);
            float scale = 1f / MathF.Sqrt(sumSq / dim + eps);
            TensorPrimitives.Multiply(rowSpan, scale, rowSpan);
        });
    }

    private static float[] Modulate(float[] x, int numTokens, int dim, ReadOnlySpan<float> shift, ReadOnlySpan<float> scale)
    {
        var outF = new float[x.Length];
        for (int t = 0; t < numTokens; t++)
        {
            int off = t * dim;
            for (int i = 0; i < dim; i++)
                outF[off + i] = x[off + i] * (1.0f + scale[i]) + shift[i];
        }
        return outF;
    }

    private static void ApplyGatedResidual(float[] x, float[] branch, int numTokens, int dim, ReadOnlySpan<float> gate)
    {
        for (int t = 0; t < numTokens; t++)
        {
            int off = t * dim;
            for (int i = 0; i < dim; i++)
                x[off + i] += branch[off + i] * gate[i];
        }
    }

    private float[] Linear(string name, ReadOnlySpan<float> x, int inDim, int outDim)
    {
        var w = GetWeight($"{name}.weight");
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;
        var outF = new float[rows * outDim];
        var xCopy = x.ToArray(); // captured for the parallel closure below

        Parallel.For(0, outDim, o =>
        {
            float bVal = b is not null ? b[o] : 0f;
            var wRow = w.AsSpan(o * inDim, inDim);
            for (int r = 0; r < rows; r++)
            {
                var xRow = xCopy.AsSpan(r * inDim, inDim);
                outF[r * outDim + o] = bVal + TensorPrimitives.Dot(xRow, wRow);
            }
        });
        return outF;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weights.Dispose();
        }
    }
}
