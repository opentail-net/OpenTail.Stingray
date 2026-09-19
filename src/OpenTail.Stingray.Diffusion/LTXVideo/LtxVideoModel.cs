using OpenTail.Stingray.Core;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

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
    private readonly IComputeBackend? _backend;
    private readonly QuantizedWeightCache _quantizedCache;
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);
    private LtxVideoGpuWeights? _gpuWeights;
    private LtxVideoGpuWorkspace? _gpuWorkspace;
    private readonly object _gpuLock = new();
    private CoreTensor? _cachedRopeCosGpu;
    private CoreTensor? _cachedRopeSinGpu;
    private (int numFrames, int patchH, int patchW) _cachedRopeKey;
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
    // they cost negligible extra RAM for the test harness and save needing separate test-only accessors.
    internal float[]? LastProjInOut { get; private set; }
    internal float[]? LastCaptionProjOut { get; private set; }
    internal float[]? LastEmbeddedTimestep { get; private set; }
    internal float[]? LastTimestepProj { get; private set; }
    internal float[]? LastRopeCos { get; private set; }
    internal float[]? LastRopeSin { get; private set; }
    internal float[]? LastBlock0Out { get; private set; }

    public LtxVideoModel(IWeightLoader weights, string prefix = "model.diffusion_model", IComputeBackend? backend = null)
    {
        _weights = weights;
        _prefix = prefix.Length > 0 && !prefix.EndsWith('.') ? prefix + "." : prefix;
        _backend = backend;
        _quantizedCache = new QuantizedWeightCache(weights, _prefix);

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
        if (_backend is not null && _backend is IVisionOpsBackend visionOps && _backend is IImageOpsBackend imageOps)
        {
            try
            {
                lock (_gpuLock)
                {
                    return ForwardGpu(visionOps, imageOps, latents, timestep, captionEmbeds, numFrames, patchH, patchW);
                }
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                Console.WriteLine($"[LtxVideo GPU Exception] {ex}");
            }
        }

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
        DiffusionOps.LayerNormNoAffine(normed, d, NormEps);
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
        DiffusionOps.RmsNormNoAffine(norm1, d, NormEps);
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
        DiffusionOps.RmsNormNoAffine(norm2, d, NormEps);
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

        var attnOut = DiffusionOps.MultiHeadAttention(q, k, v, qSeq, kvSeq, NumHeads, HeadDim);
        return Linear($"{prefix}.to_out.0", attnOut, d, d);
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

    /// <summary>Real `TimestepEmbedder`: sinusoidal(256) -> Linear(256,hidden) -> SiLU -> Linear(hidden,hidden).
    /// Sinusoidal embedding uses `flip_sin_to_cos=true` (2026-09-14 fix, found via a golden-parity
    /// diagnostic: the shared `timestep_embedding` helper this model's real C++ reference calls
    /// (`examples/stable-diffusion.cpp/src/core/ggml_extend.hpp`) defaults `flip_sin_to_cos=true`
    /// -- [cos,sin] order, not [sin,cos] -- the same real convention already confirmed and fixed for
    /// Wan's own timestep embedding this session; this call site had been left at this project's
    /// own default (false/[sin,cos]), which measured cosine-sim 0.79 against the real dumped
    /// `embedded_timestep` golden fixture before this fix (see docs/077's 2026-09-14 update) and is
    /// very likely the root cause of the block0/full-output golden-parity failure, since AdaLN
    /// modulation derives from this value in every block.</summary>
    private float[] TimestepEmbedder(float timestep)
    {
        const int freqEmbedSize = 256;
        var emb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, freqEmbedSize, flipSinToCos: true);
        var h1 = Linear("adaln_single.emb.timestep_embedder.linear_1", emb, freqEmbedSize, HiddenSize);
        DiffusionOps.SiluInPlace(h1);
        return Linear("adaln_single.emb.timestep_embedder.linear_2", h1, HiddenSize, HiddenSize);
    }

    private static float[] Modulate(float[] x, int numTokens, int dim, float[] shift, float[] scale)
    {
        var outF = new float[x.Length];
        var scalePlusOne = new float[dim];
        for (int i = 0; i < dim; i++) scalePlusOne[i] = 1.0f + scale[i];

        Parallel.For(0, numTokens, t =>
        {
            int off = t * dim;
            var xSpan = x.AsSpan(off, dim);
            var outSpan = outF.AsSpan(off, dim);
            TensorPrimitives.Multiply(xSpan, scalePlusOne, outSpan);
            TensorPrimitives.Add(outSpan, shift, outSpan);
        });
        return outF;
    }

    private static void ApplyGatedResidual(float[] x, float[] branch, int numTokens, int dim, float[] gate)
    {
        Parallel.For(0, numTokens, t =>
        {
            int off = t * dim;
            var xSpan = x.AsSpan(off, dim);
            var bSpan = branch.AsSpan(off, dim);
            TensorPrimitives.MultiplyAdd(bSpan, gate, xSpan, xSpan);
        });
    }

    private float[] Linear(string name, float[] x, int inDim, int outDim)
    {
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;
        var result = new float[rows * outDim];
        _quantizedCache.Linear($"{name}.weight", x.AsSpan(0, rows * inDim), b ?? ReadOnlySpan<float>.Empty, result.AsSpan(), rows, inDim, outDim);
        return result;
    }

    private float[] Linear(string name, ReadOnlySpan<float> x, int inDim, int outDim)
    {
        var b = TryGetWeight($"{name}.bias");
        int rows = x.Length / inDim;
        var result = new float[rows * outDim];
        _quantizedCache.Linear($"{name}.weight", x.Slice(0, rows * inDim), b ?? ReadOnlySpan<float>.Empty, result.AsSpan(), rows, inDim, outDim);
        return result;
    }

    private void EnsureGpuResident(IVisionOpsBackend visionOps, IImageOpsBackend imageOps, int numTokens, int numTxt)
    {
        if (_gpuWeights is null && _backend is not null)
        {
            _gpuWeights = new LtxVideoGpuWeights(_backend, GetWeight, TryGetWeight, InChannels, HiddenSize, CaptionChannels, NumLayers);
        }
        if (_gpuWorkspace is null || _gpuWorkspace.NumTokens != numTokens || _gpuWorkspace.NumTxt != numTxt)
        {
            _gpuWorkspace?.Dispose();
            _gpuWorkspace = new LtxVideoGpuWorkspace(_backend!, numTokens, numTxt, HiddenSize, InChannels);
        }
    }

    private float[] ForwardGpu(
        IVisionOpsBackend visionOps,
        IImageOpsBackend imageOps,
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

        EnsureGpuResident(visionOps, imageOps, numTokens, numTxt);
        var gw = _gpuWeights!;
        var ws = _gpuWorkspace!;

        // 1. patchify_proj: latents [numTokens, InChannels] -> x [numTokens, d]
        var latentsGpu = imageOps.Upload(latents.ToArray(), TensorShape.D2(numTokens, InChannels), exact: true);
        imageOps.Sgemm(ws.X, latentsGpu, gw.PatchifyProjWeight, numTokens, InChannels, d);
        if (gw.PatchifyProjBias is not null)
            imageOps.AddRowBroadcastInPlace(ws.X, gw.PatchifyProjBias, numTokens, d);
        _backend!.Free(latentsGpu);

        // 2. caption_projection: PixArtAlphaTextProjection(4096 -> 2048 -> 2048, GELU between)
        if (numTxt > 0)
        {
            var capGpu = imageOps.Upload(captionEmbeds.ToArray(), TensorShape.D2(numTxt, CaptionChannels), exact: true);
            imageOps.Sgemm(ws.CapIntermediate, capGpu, gw.CapProj1Weight, numTxt, CaptionChannels, d);
            if (gw.CapProj1Bias is not null)
                imageOps.AddRowBroadcastInPlace(ws.CapIntermediate, gw.CapProj1Bias, numTxt, d);
            visionOps.VisionGeluInPlace(ws.CapIntermediate);
            imageOps.Sgemm(ws.CaptionProj, ws.CapIntermediate, gw.CapProj2Weight, numTxt, d, d);
            if (gw.CapProj2Bias is not null)
                imageOps.AddRowBroadcastInPlace(ws.CaptionProj, gw.CapProj2Bias, numTxt, d);
            _backend.Free(capGpu);
        }

        // 3. AdaLayerNormSingle
        const int freqEmbedSize = 256;
        var emb = DiffusionOps.SinusoidalTimestepEmbedding(timestep, freqEmbedSize, flipSinToCos: true);
        var embGpu = imageOps.Upload(emb, TensorShape.D1(freqEmbedSize), exact: true);
        imageOps.Sgemm(ws.EmbeddedTimestep, embGpu, gw.TimeEmb1Weight, 1, freqEmbedSize, d);
        if (gw.TimeEmb1Bias is not null)
            imageOps.AddRowBroadcastInPlace(ws.EmbeddedTimestep, gw.TimeEmb1Bias, 1, d);
        imageOps.SiLU(ws.EmbeddedTimestep);

        imageOps.Sgemm(ws.Normed, ws.EmbeddedTimestep, gw.TimeEmb2Weight, 1, d, d);
        if (gw.TimeEmb2Bias is not null)
            imageOps.AddRowBroadcastInPlace(ws.Normed, gw.TimeEmb2Bias, 1, d);
        imageOps.ScaleInPlace(ws.EmbeddedTimestep, 0f);
        imageOps.AddInPlace(ws.EmbeddedTimestep, ws.Normed); // ws.EmbeddedTimestep is now hidden

        // silu(embedded_timestep) -> adaln_single.linear
        imageOps.ScaleInPlace(ws.Normed, 0f);
        imageOps.AddInPlace(ws.Normed, ws.EmbeddedTimestep);
        imageOps.SiLU(ws.Normed);
        imageOps.Sgemm(ws.TimestepProj, ws.Normed, gw.TimeProjWeight, 1, d, 6 * d);
        if (gw.TimeProjBias is not null)
            imageOps.AddRowBroadcastInPlace(ws.TimestepProj, gw.TimeProjBias, 1, 6 * d);
        _backend.Free(embGpu);

        // 4. RoPE (cached across denoising steps)
        if (_cachedRopeCosGpu is null || _cachedRopeSinGpu is null || _cachedRopeKey != (numFrames, patchH, patchW))
        {
            if (_cachedRopeCosGpu is not null) _backend.Free(_cachedRopeCosGpu);
            if (_cachedRopeSinGpu is not null) _backend.Free(_cachedRopeSinGpu);
            var (compactCos, compactSin) = LtxVideoRoPE.ComputeContinuous3DRoPECompact(numFrames, patchH, patchW, d, RopeTheta);
            _cachedRopeCosGpu = imageOps.Upload(compactCos, TensorShape.D2(numTokens, d / 2), exact: true);
            _cachedRopeSinGpu = imageOps.Upload(compactSin, TensorShape.D2(numTokens, d / 2), exact: true);
            _cachedRopeKey = (numFrames, patchH, patchW);
        }

        // 5. 28 Transformer Blocks (100% GPU Resident, Batched Command Buffer)
        imageOps.BeginBatch();
        bool batchSuccess = false;
        try
        {
            for (int l = 0; l < NumLayers; l++)
            {
                var bw = gw.Blocks[l];

                // BlockMod = ScaleShiftTable + TimestepProj (both [1, 6*d])
                imageOps.ScaleInPlace(ws.BlockMod, 0f);
                imageOps.AddInPlace(ws.BlockMod, bw.ScaleShiftTable);
                imageOps.AddInPlace(ws.BlockMod, ws.TimestepProj);

                // 5.1 Self-attention: RMSNorm -> AdaLN modulate -> Q/K/V -> QK-norm -> RoPE -> Attn -> OutProj -> GatedResidual
                visionOps.AdaLNModulate(ws.Normed, ws.X, ws.BlockMod, numTokens, d, shiftOffset: 0, scaleOffset: d, isRmsNorm: true, eps: NormEps);

                imageOps.Sgemm(ws.Q, ws.Normed, bw.Attn1QWeight, numTokens, d, d);
                if (bw.Attn1QBias is not null) imageOps.AddRowBroadcastInPlace(ws.Q, bw.Attn1QBias, numTokens, d);

                imageOps.Sgemm(ws.K, ws.Normed, bw.Attn1KWeight, numTokens, d, d);
                if (bw.Attn1KBias is not null) imageOps.AddRowBroadcastInPlace(ws.K, bw.Attn1KBias, numTokens, d);

                imageOps.Sgemm(ws.V, ws.Normed, bw.Attn1VWeight, numTokens, d, d);
                if (bw.Attn1VBias is not null) imageOps.AddRowBroadcastInPlace(ws.V, bw.Attn1VBias, numTokens, d);

                visionOps.RmsNormBatched(ws.Q, ws.Q, bw.Attn1QNorm, d, numTokens, QkNormEps);
                visionOps.RmsNormBatched(ws.K, ws.K, bw.Attn1KNorm, d, numTokens, QkNormEps);

                visionOps.Flux2DRoPE(ws.Q, ws.K, _cachedRopeCosGpu, _cachedRopeSinGpu, 0, numTokens, 1, d);

                imageOps.MultiHeadAttentionTiled(ws.AttnOut, ws.Q, ws.K, ws.V, numTokens, numTokens, NumHeads, HeadDim);

                imageOps.Sgemm(ws.Normed, ws.AttnOut, bw.Attn1OutWeight, numTokens, d, d);
                if (bw.Attn1OutBias is not null) imageOps.AddRowBroadcastInPlace(ws.Normed, bw.Attn1OutBias, numTokens, d);

                visionOps.ScaleGateAdd(ws.X, ws.Normed, ws.BlockMod, numTokens, d, gateOffset: 2 * d);

                // 5.2 Cross-attention
                if (numTxt > 0)
                {
                    imageOps.Sgemm(ws.CrossQ, ws.X, bw.Attn2QWeight, numTokens, d, d);
                    if (bw.Attn2QBias is not null) imageOps.AddRowBroadcastInPlace(ws.CrossQ, bw.Attn2QBias, numTokens, d);

                    imageOps.Sgemm(ws.CrossK, ws.CaptionProj, bw.Attn2KWeight, numTxt, d, d);
                    if (bw.Attn2KBias is not null) imageOps.AddRowBroadcastInPlace(ws.CrossK, bw.Attn2KBias, numTxt, d);

                    imageOps.Sgemm(ws.CrossV, ws.CaptionProj, bw.Attn2VWeight, numTxt, d, d);
                    if (bw.Attn2VBias is not null) imageOps.AddRowBroadcastInPlace(ws.CrossV, bw.Attn2VBias, numTxt, d);

                    visionOps.RmsNormBatched(ws.CrossQ, ws.CrossQ, bw.Attn2QNorm, d, numTokens, QkNormEps);
                    visionOps.RmsNormBatched(ws.CrossK, ws.CrossK, bw.Attn2KNorm, d, numTxt, QkNormEps);

                    imageOps.MultiHeadAttentionTiled(ws.CrossAttnOut, ws.CrossQ, ws.CrossK, ws.CrossV, numTokens, numTxt, NumHeads, HeadDim);

                    imageOps.Sgemm(ws.Normed, ws.CrossAttnOut, bw.Attn2OutWeight, numTokens, d, d);
                    if (bw.Attn2OutBias is not null) imageOps.AddRowBroadcastInPlace(ws.Normed, bw.Attn2OutBias, numTokens, d);

                    imageOps.AddInPlace(ws.X, ws.Normed);
                }

                // 5.3 FFN
                visionOps.AdaLNModulate(ws.Normed, ws.X, ws.BlockMod, numTokens, d, shiftOffset: 3 * d, scaleOffset: 4 * d, isRmsNorm: true, eps: NormEps);

                imageOps.Sgemm(ws.Ffn1, ws.Normed, bw.Ffn0Weight, numTokens, d, d * 4);
                if (bw.Ffn0Bias is not null) imageOps.AddRowBroadcastInPlace(ws.Ffn1, bw.Ffn0Bias, numTokens, d * 4);

                visionOps.VisionGeluInPlace(ws.Ffn1);

                imageOps.Sgemm(ws.FfnOut, ws.Ffn1, bw.Ffn2Weight, numTokens, d * 4, d);
                if (bw.Ffn2Bias is not null) imageOps.AddRowBroadcastInPlace(ws.FfnOut, bw.Ffn2Bias, numTokens, d);

                visionOps.ScaleGateAdd(ws.X, ws.FfnOut, ws.BlockMod, numTokens, d, gateOffset: 5 * d);
            }

            // 6. Final Layer (100% GPU resident inside the single-submission batch):
            // FinalMod = TopScaleShiftTable + EmbeddedTimestep (broadcast over the 2 rows [shift, scale])
            imageOps.ScaleInPlace(ws.FinalMod, 0f);
            imageOps.AddInPlace(ws.FinalMod, gw.TopScaleShiftTable);
            imageOps.AddRowBroadcastInPlace(ws.FinalMod, ws.EmbeddedTimestep, 2, d);

            visionOps.AdaLNModulate(ws.Normed, ws.X, ws.FinalMod, numTokens, d, shiftOffset: 0, scaleOffset: d, isRmsNorm: false, eps: NormEps);

            imageOps.Sgemm(ws.ProjOut, ws.Normed, gw.ProjOutWeight, numTokens, d, OutChannels);
            if (gw.ProjOutBias is not null)
                imageOps.AddRowBroadcastInPlace(ws.ProjOut, gw.ProjOutBias, numTokens, OutChannels);

            imageOps.EndBatch();
            batchSuccess = true;
        }
        finally
        {
            if (!batchSuccess)
            {
                try { imageOps.EndBatch(); } catch { }
            }
        }

        var result = new float[numTokens * OutChannels];
        imageOps.Download(ws.ProjOut, result);
        return result;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            if (_cachedRopeCosGpu is not null) { _backend?.Free(_cachedRopeCosGpu); _cachedRopeCosGpu = null; }
            if (_cachedRopeSinGpu is not null) { _backend?.Free(_cachedRopeSinGpu); _cachedRopeSinGpu = null; }
            _gpuWorkspace?.Dispose();
            _gpuWeights?.Dispose();
            _quantizedCache.Dispose();
            _weights.Dispose();
        }
    }
}

