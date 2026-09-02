using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion;

/// <summary>
/// FLUX Multi-Modal Diffusion Transformer (MM-DiT) forward pass.
///
/// Architecture (FLUX.1-schnell / FLUX.1-dev):
///   img_in:     Linear(64, 3072)           — project image patches
///   txt_in:     Linear(4096, 3072)          — project T5 text embeddings
///   time_in:    Timestep MLP → [3072]
///   vector_in:  Pooled CLIP MLP → [3072]
///   double_blocks × 19: img/txt separate streams with cross-attention
///   single_blocks × 38: concatenated img+txt stream
///   final_layer: AdaLN + Linear(3072, 64)
///
/// Weights are loaded from a GGUF file using the existing GgufModel infrastructure.
/// All matrix multiplications go through <see cref="IComputeBackend"/> (CpuBackend or VulkanBackend).
/// </summary>
public sealed class FluxDiT : IDisposable
{
    private readonly GgufModel _model;
    private readonly FluxParams _p;
    private readonly IComputeBackend _backend;
    private bool _disposed;

    // Cached tensor lookups (lazy on first use)
    private readonly Dictionary<string, float[]> _weightCache = new(StringComparer.Ordinal);

    /// <summary>Minimum token batch size to route a MatQ call through the GPU backend.</summary>
    private const int MinGpuBatch = 16;
    /// <summary>bf16 weights cached on GPU — uploaded once on first denoising step, reused every step.</summary>
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsBf16;
    /// <summary>fp8 E4M3 weights cached on GPU — uploaded once on first step (sm_89+, 2× smaller than bf16).</summary>
    private readonly Dictionary<string, CoreTensor>? _gpuWeightsFp8;

    public FluxParams Params => _p;

    public FluxDiT(GgufModel model, FluxParams p, IComputeBackend backend)
    {
        _model   = model;
        _p       = p;
        _backend = backend;
        if (backend?.BestSgemmPrecision == SgemmPrecision.Bf16)
            _gpuWeightsBf16 = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
        if (backend?.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
            _gpuWeightsFp8 = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    // ── Entry point ───────────────────────────────────────────────────────

    /// <summary>
    /// Single denoising forward pass.
    /// Returns predicted velocity field (same shape as <paramref name="imgLatent"/>).
    /// </summary>
    /// <param name="imgLatent">Packed image patches [nImg, inChannels] (= [H/2·W/2, 64]).</param>
    /// <param name="imgIds">Patch (row,col) position ids [nImg, 2].</param>
    /// <param name="txtEmbeds">T5 text embeddings [nTxt, 4096].</param>
    /// <param name="txtIds">Text token ids (all zero for FLUX) [nTxt, 2].</param>
    /// <param name="pooledEmbed">CLIP pooled embed [768].</param>
    /// <param name="timestep">Scalar timestep ∈ [0, 1].</param>
    /// <param name="guidance">Guidance scale (ignored for schnell, used by dev).</param>
    public float[] Forward(
        float[] imgLatent, int[] imgIds,
        float[] txtEmbeds, int[] txtIds,
        float[] pooledEmbed,
        float timestep, float guidance = 3.5f)
    {
        int nImg = imgIds.Length / 2;
        int nTxt = txtIds.Length / 2;
        int d    = _p.HiddenSize;   // 3072
        int nSeq = nTxt + nImg;

        using var ws = new Workspace(nSeq, nImg, nTxt, d);

        // ── Encode conditioning ───────────────────────────────────────────
        float[] vec = ComputeVec(timestep, pooledEmbed, guidance);     // [d]
        float[] txtHidden = ProjectTxt(txtEmbeds, nTxt);               // [nTxt, d]
        float[] imgHidden = ProjectImg(imgLatent, nImg);               // [nImg, d]

        // ── Build RoPE freqs ──────────────────────────────────────────────
        // Combine txt and img ids for single-stream RoPE: [txtIds (zeros), imgIds (spatial)]
        var allIds = new int[nSeq * 2];
        // text positions [0, nTxt*2) are all zeros (already zero from array init)
        imgIds.CopyTo(allIds, nTxt * 2);
        var (ropeC, ropeS) = Flux2DRoPE.BuildFreqs(allIds, nSeq, _p.HeadDim);

        // Also build per-image RoPE for double stream blocks
        var (imgRopeC, imgRopeS) = Flux2DRoPE.BuildFreqs(imgIds, nImg, _p.HeadDim);

        // ── Double stream blocks ──────────────────────────────────────────
        for (int i = 0; i < _p.DoubleBlocks; i++)
            DoubleBlock(i, imgHidden, txtHidden, vec, imgRopeC, imgRopeS, nImg, nTxt, ws);

        // ── Single stream blocks ──────────────────────────────────────────
        // Concatenate txt + img → [nSeq, d] (matches BFL FLUX convention: txt first, img second)
        var x = new float[nSeq * d];
        txtHidden.AsSpan().CopyTo(x.AsSpan(0, nTxt * d));
        imgHidden.AsSpan().CopyTo(x.AsSpan(nTxt * d, nImg * d));

        for (int i = 0; i < _p.SingleBlocks; i++)
            SingleBlock(i, x, vec, ropeC, ropeS, nSeq, nTxt, nImg, ws);

        // ── Final layer ───────────────────────────────────────────────────
        // Extract image portion (tokens after nTxt)
        imgHidden = x.AsSpan(nTxt * d, nImg * d).ToArray();
        return FinalLayer(imgHidden, vec, nImg, ws);
    }

    // ── Workspace for zero-copy memory reuse ───────────────────────────────

    private sealed class Workspace : IDisposable
    {
        public readonly float[] Q;
        public readonly float[] K;
        public readonly float[] V;
        public readonly float[] AttnOut;
        public readonly float[] Lin1;
        public readonly float[] Combined;
        public readonly float[] NormedImg;
        public readonly float[] NormedTxt;
        public readonly float[] QkvTxt;
        public readonly float[] QkvImg;
        public readonly float[] OutTxt;
        public readonly float[] OutImg;
        public readonly float[] MlpBuf;

        public Workspace(int nSeq, int nImg, int nTxt, int d)
        {
            Q = ArrayPool<float>.Shared.Rent(nSeq * d);
            K = ArrayPool<float>.Shared.Rent(nSeq * d);
            V = ArrayPool<float>.Shared.Rent(nSeq * d);
            AttnOut = ArrayPool<float>.Shared.Rent(nSeq * d);
            Lin1 = ArrayPool<float>.Shared.Rent(nSeq * 7 * d);
            Combined = ArrayPool<float>.Shared.Rent(nSeq * 5 * d);
            NormedImg = ArrayPool<float>.Shared.Rent(nImg * d);
            NormedTxt = ArrayPool<float>.Shared.Rent(nTxt * d);
            QkvTxt = ArrayPool<float>.Shared.Rent(nTxt * 3 * d);
            QkvImg = ArrayPool<float>.Shared.Rent(nImg * 3 * d);
            OutTxt = ArrayPool<float>.Shared.Rent(nTxt * d);
            OutImg = ArrayPool<float>.Shared.Rent(nImg * d);
            int maxMlp = Math.Max(nSeq, Math.Max(nImg, nTxt)) * 4 * d;
            MlpBuf = ArrayPool<float>.Shared.Rent(maxMlp);
        }

        public void Dispose()
        {
            ArrayPool<float>.Shared.Return(Q);
            ArrayPool<float>.Shared.Return(K);
            ArrayPool<float>.Shared.Return(V);
            ArrayPool<float>.Shared.Return(AttnOut);
            ArrayPool<float>.Shared.Return(Lin1);
            ArrayPool<float>.Shared.Return(Combined);
            ArrayPool<float>.Shared.Return(NormedImg);
            ArrayPool<float>.Shared.Return(NormedTxt);
            ArrayPool<float>.Shared.Return(QkvTxt);
            ArrayPool<float>.Shared.Return(QkvImg);
            ArrayPool<float>.Shared.Return(OutTxt);
            ArrayPool<float>.Shared.Return(OutImg);
            ArrayPool<float>.Shared.Return(MlpBuf);
        }
    }

    // ── Conditioning embedding ────────────────────────────────────────────

    private float[] ComputeVec(float timestep, float[] pooled, float guidance)
    {
        int d = _p.HiddenSize;

        // Timestep sinusoidal embedding → MLP → [d]
        float[] tEmb  = TimestepEmbedding(timestep, 256);
        float[] tProj = MlpProj("model.diffusion_model.time_in", tEmb, 256, d);

        // CLIP pooled embedding → MLP → [d]
        float[] vProj = MlpProj("model.diffusion_model.vector_in", pooled, _p.VecDim, d);

        var vec = new float[d];
        for (int i = 0; i < d; i++) vec[i] = tProj[i] + vProj[i];

        if (_p.HasGuidanceIn)
        {
            float[] gEmb  = TimestepEmbedding(guidance, 256);
            float[] gProj = MlpProj("model.diffusion_model.guidance_in", gEmb, 256, d);
            for (int i = 0; i < d; i++) vec[i] += gProj[i];
        }
        return vec;
    }

    private float[] ProjectImg(float[] imgLatent, int nImg) =>
        MatQ(imgLatent, nImg, _p.InChannels, "model.diffusion_model.img_in.weight", _p.HiddenSize,
             W("model.diffusion_model.img_in.bias"));

    private float[] ProjectTxt(float[] txtEmb, int nTxt) =>
        MatQ(txtEmb, nTxt, _p.ContextDim, "model.diffusion_model.txt_in.weight", _p.HiddenSize,
             W("model.diffusion_model.txt_in.bias"));

    // ── Double stream block ───────────────────────────────────────────────

    private void DoubleBlock(int idx,
        float[] img, float[] txt, float[] vec,
        float[] imgRopeC, float[] imgRopeS,
        int nImg, int nTxt, Workspace ws)
    {
        int d  = _p.HiddenSize;
        int nh = _p.NumHeads;
        int hd = _p.HeadDim;
        int nSeq = nTxt + nImg;
        string pi = $"model.diffusion_model.double_blocks.{idx}";

        // adaLN modulation: Linear(d, 6d) × silu for each stream
        float[] imgMod = AdaLNMod($"{pi}.img_mod.lin", vec, d, 6);
        float[] txtMod = AdaLNMod($"{pi}.txt_mod.lin", vec, d, 6);

        // Normalize txt & img directly into ws buffers
        DiffusionOps.AdaLNModulate(ws.NormedTxt.AsSpan(0, nTxt * d), txt, txtMod.AsSpan(0, d), txtMod.AsSpan(d, d), nTxt, d, isRmsNorm: true, eps: 1e-6f);
        DiffusionOps.AdaLNModulate(ws.NormedImg.AsSpan(0, nImg * d), img, imgMod.AsSpan(0, d), imgMod.AsSpan(d, d), nImg, d, isRmsNorm: true, eps: 1e-6f);

        // Txt QKV: compute fused [nTxt, 3d] and unpack directly into rows [0..nTxt) of Q, K, V
        MatQ(ws.NormedTxt.AsSpan(0, nTxt * d), nTxt, d, $"{pi}.txt_attn.qkv.weight", d * 3, null, ws.QkvTxt.AsSpan(0, nTxt * d * 3));
        UnpackQkv(ws.QkvTxt, ws.Q, ws.K, ws.V, 0, nTxt, d);
        QKNorm($"{pi}.txt_attn.norm", ws.Q, ws.K, 0, nTxt, nh, hd);

        // Img QKV: compute fused [nImg, 3d] and unpack directly into rows [nTxt..nSeq) of Q, K, V
        MatQ(ws.NormedImg.AsSpan(0, nImg * d), nImg, d, $"{pi}.img_attn.qkv.weight", d * 3, null, ws.QkvImg.AsSpan(0, nImg * d * 3));
        UnpackQkv(ws.QkvImg, ws.Q, ws.K, ws.V, nTxt, nImg, d);
        QKNorm($"{pi}.img_attn.norm", ws.Q, ws.K, nTxt, nImg, nh, hd);

        // Apply 2D RoPE to img Q,K only (located at [nTxt, nTxt+nImg))
        Flux2DRoPE.ApplyInPlace(ws.Q, imgRopeC, imgRopeS, nSeq, nh, hd, startToken: nTxt, tokenCount: nImg);
        Flux2DRoPE.ApplyInPlace(ws.K, imgRopeC, imgRopeS, nSeq, nh, hd, startToken: nTxt, tokenCount: nImg);

        // Joint attention directly into ws.AttnOut [nSeq, d]
        DiffusionOps.MultiHeadAttention(ws.Q, ws.K, ws.V, ws.AttnOut.AsSpan(0, nSeq * d), nSeq, nSeq, nh, hd);

        // Project + residual for txt (tokens 0..nTxt)
        LinearBias($"{pi}.txt_attn.proj", ws.AttnOut.AsSpan(0, nTxt * d), nTxt, d, d, ws.OutTxt.AsSpan(0, nTxt * d));
        ScaleGateAdd(txt, ws.OutTxt, txtMod, nTxt, d, gateIdx: 2);

        // txt MLP
        DiffusionOps.AdaLNModulate(ws.NormedTxt.AsSpan(0, nTxt * d), txt, txtMod.AsSpan(3 * d, d), txtMod.AsSpan(4 * d, d), nTxt, d, isRmsNorm: true, eps: 1e-6f);
        GeluMlpDirect($"{pi}.txt_mlp", ws.NormedTxt.AsSpan(0, nTxt * d), nTxt, d, ws.MlpBuf, ws.OutTxt.AsSpan(0, nTxt * d));
        ScaleGateAdd(txt, ws.OutTxt, txtMod, nTxt, d, gateIdx: 5);

        // Project + residual for img (tokens nTxt..nSeq)
        LinearBias($"{pi}.img_attn.proj", ws.AttnOut.AsSpan(nTxt * d, nImg * d), nImg, d, d, ws.OutImg.AsSpan(0, nImg * d));
        ScaleGateAdd(img, ws.OutImg, imgMod, nImg, d, gateIdx: 2);

        // img MLP
        DiffusionOps.AdaLNModulate(ws.NormedImg.AsSpan(0, nImg * d), img, imgMod.AsSpan(3 * d, d), imgMod.AsSpan(4 * d, d), nImg, d, isRmsNorm: true, eps: 1e-6f);
        GeluMlpDirect($"{pi}.img_mlp", ws.NormedImg.AsSpan(0, nImg * d), nImg, d, ws.MlpBuf, ws.OutImg.AsSpan(0, nImg * d));
        ScaleGateAdd(img, ws.OutImg, imgMod, nImg, d, gateIdx: 5);
    }

    // ── Single stream block ───────────────────────────────────────────────

    private void SingleBlock(int idx, float[] x, float[] vec,
                              float[] ropeC, float[] ropeS, int nSeq, int nTxt, int nImg,
                              Workspace ws)
    {
        int d  = _p.HiddenSize;
        int nh = _p.NumHeads;
        int hd = _p.HeadDim;
        string p = $"model.diffusion_model.single_blocks.{idx}";

        // adaLN modulation: Linear(d, 3d)
        float[] mod = AdaLNMod($"{p}.modulation.lin", vec, d, 3);

        // Modulate directly into ws.AttnOut
        DiffusionOps.AdaLNModulate(ws.AttnOut.AsSpan(0, nSeq * d), x, mod.AsSpan(0, d), mod.AsSpan(d, d), nSeq, d, isRmsNorm: true, eps: 1e-6f);

        // Fused linear1 into ws.Lin1 [nSeq, 7d]
        MatQ(ws.AttnOut.AsSpan(0, nSeq * d), nSeq, d, $"{p}.linear1.weight", d * 7, null, ws.Lin1.AsSpan(0, nSeq * d * 7));

        // Unpack lin1 into Q, K, V and the MLP portion of Combined
        UnpackSingleLin1(ws.Lin1, ws.Q, ws.K, ws.V, ws.Combined, nSeq, d);

        // QK norm (per-head)
        QKNorm($"{p}.norm", ws.Q, ws.K, 0, nSeq, nh, hd);

        // 2D RoPE on full sequence [txt (identity), img (spatial)]
        Flux2DRoPE.ApplyInPlace(ws.Q, ropeC, ropeS, nSeq, nh, hd);
        Flux2DRoPE.ApplyInPlace(ws.K, ropeC, ropeS, nSeq, nh, hd);

        // Self-attention directly into the first part of ws.Combined [0..nSeq*d]
        DiffusionOps.MultiHeadAttention(ws.Q, ws.K, ws.V, ws.Combined.AsSpan(0, nSeq * d), nSeq, nSeq, nh, hd);

        // GELU on the MLP portion of ws.Combined [nSeq*d .. nSeq*5d]
        DiffusionOps.GeluInPlace(ws.Combined.AsSpan(nSeq * d, nSeq * d * 4));

        // linear2 directly from ws.Combined [nSeq, 5d] into ws.AttnOut [nSeq, d]
        MatQ(ws.Combined.AsSpan(0, nSeq * d * 5), nSeq, d * 5, $"{p}.linear2.weight", d, null, ws.AttnOut.AsSpan(0, nSeq * d));

        // Gate and residual
        ScaleGateAdd(x, ws.AttnOut, mod, nSeq, d, gateIdx: 2);
    }

    // ── Final layer ───────────────────────────────────────────────────────

    private float[] FinalLayer(float[] img, float[] vec, int nImg, Workspace ws)
    {
        int d = _p.HiddenSize;
        string p = "model.diffusion_model.final_layer";

        // adaLN modulation: shift + scale
        var mod = AdaLNMod($"{p}.adaLN_modulation.1", vec, d, 2);
        DiffusionOps.AdaLNModulate(ws.NormedImg.AsSpan(0, nImg * d), img, mod.AsSpan(0, d), mod.AsSpan(d, d), nImg, d, isRmsNorm: true, eps: 1e-6f);

        // Linear(d, outChannels)
        return MatQ(ws.NormedImg, nImg, d, $"{p}.linear.weight", _p.OutChannels, W($"{p}.linear.bias"));
    }

    // ── Attention helpers ─────────────────────────────────────────────────

    private void QKNorm(string normPath, float[] q, float[] k, int tokenStart, int tokenCount, int nh, int hd)
    {
        var qScale = W($"{normPath}.query_norm.scale");
        var kScale = W($"{normPath}.key_norm.scale");
        // Norm each head's q and k independently in parallel
        Parallel.For(0, tokenCount * nh, idx =>
        {
            int t = idx / nh;
            int h = idx % nh;
            int globalToken = tokenStart + t;
            int off = (globalToken * nh + h) * hd;
            DiffusionOps.RmsNorm(q.AsSpan(off, hd), qScale, hd, _p.QkNormEps);
            DiffusionOps.RmsNorm(k.AsSpan(off, hd), kScale, hd, _p.QkNormEps);
        });
    }

    private static void UnpackQkv(float[] qkv, float[] q, float[] k, float[] v, int dstTokenStart, int tokenCount, int d)
    {
        Parallel.For(0, tokenCount, i =>
        {
            int srcOff = i * d * 3;
            int dstOff = (dstTokenStart + i) * d;
            qkv.AsSpan(srcOff, d).CopyTo(q.AsSpan(dstOff, d));
            qkv.AsSpan(srcOff + d, d).CopyTo(k.AsSpan(dstOff, d));
            qkv.AsSpan(srcOff + d * 2, d).CopyTo(v.AsSpan(dstOff, d));
        });
    }

    private static void UnpackSingleLin1(float[] lin1, float[] q, float[] k, float[] v, float[] combined, int nSeq, int d)
    {
        Parallel.For(0, nSeq, i =>
        {
            int srcOff = i * d * 7;
            int dstDOff = i * d;
            int dstMlpOff = nSeq * d + i * d * 4;
            lin1.AsSpan(srcOff, d).CopyTo(q.AsSpan(dstDOff, d));
            lin1.AsSpan(srcOff + d, d).CopyTo(k.AsSpan(dstDOff, d));
            lin1.AsSpan(srcOff + d * 2, d).CopyTo(v.AsSpan(dstDOff, d));
            lin1.AsSpan(srcOff + d * 3, d * 4).CopyTo(combined.AsSpan(dstMlpOff, d * 4));
        });
    }

    // ── MLP helpers ───────────────────────────────────────────────────────

    private void GeluMlpDirect(string prefix, ReadOnlySpan<float> x, int n, int d, float[] mlpBuf, Span<float> dst)
    {
        int hidden = d * 4;
        MatQ(x, n, d, $"{prefix}.0.weight", hidden, W($"{prefix}.0.bias"), mlpBuf.AsSpan(0, n * hidden));
        DiffusionOps.GeluInPlace(mlpBuf.AsSpan(0, n * hidden));
        MatQ(mlpBuf.AsSpan(0, n * hidden), n, hidden, $"{prefix}.2.weight", d, W($"{prefix}.2.bias"), dst);
    }

    private float[] MlpProj(string prefix, float[] x, int inDim, int outDim)
    {
        var h = DiffusionOps.Linear(x, W($"{prefix}.in_layer.weight"), W($"{prefix}.in_layer.bias"), 1, inDim, outDim);
        DiffusionOps.SiluInPlace(h);
        return DiffusionOps.Linear(h, W($"{prefix}.out_layer.weight"), W($"{prefix}.out_layer.bias"), 1, outDim, outDim);
    }

    // ── adaLN helpers ─────────────────────────────────────────────────────

    private float[] AdaLNMod(string linPath, float[] vec, int d, int nOut)
    {
        // Real FLUX (AdaLayerNormZero/AdaLayerNormZeroSingle.forward,
        // diffusers/models/normalization.py): `emb = self.linear(self.silu(emb))` -- SiLU gates
        // the shared conditioning vector BEFORE the modulation linear projects it into the
        // shift/scale/gate chunks. This previously applied SiLU AFTER the linear instead, directly
        // squashing the projected shift/scale/gate values through a nonlinearity they were never
        // meant to pass through -- wrong on every one of the ~4 modulation calls per block, across
        // all 19 double + 38 single blocks. `vec` is reused across multiple AdaLNMod calls (e.g.
        // DoubleBlock's img_mod and txt_mod both read the same vec), so SiLU must run on a COPY,
        // not in place.
        var gated = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(gated);
        return DiffusionOps.Linear(gated, W($"{linPath}.weight"), W($"{linPath}.bias"), 1, d, nOut * d);
    }

    private static unsafe void ScaleGateAdd(float[] x, float[] update, float[] mod,
                                             int n, int d, int gateIdx)
    {
        int gateOff = gateIdx * d;
        fixed (float* pX = x, pU = update, pMod = mod)
        {
            float* pXLocal = pX;
            float* pULocal = pU;
            float* pGate = pMod + gateOff;
            Parallel.For(0, n, i =>
            {
                int off = i * d;
                var xSpan = new Span<float>(pXLocal + off, d);
                var uSpan = new ReadOnlySpan<float>(pULocal + off, d);
                var gSpan = new ReadOnlySpan<float>(pGate, d);
                TensorPrimitives.MultiplyAdd(uSpan, gSpan, xSpan, xSpan);
            });
        }
    }

    private void LinearNoBias(string path, ReadOnlySpan<float> x, int n, int inDim, int outDim, Span<float> dst)
        => MatQ(x, n, inDim, $"{path}.weight", outDim, null, dst);

    private void LinearBias(string path, ReadOnlySpan<float> x, int n, int inDim, int outDim, Span<float> dst)
        => MatQ(x, n, inDim, $"{path}.weight", outDim, OptW($"{path}.bias"), dst);

    private float[] LinearBias(string path, ReadOnlySpan<float> x, int n, int inDim, int outDim)
    {
        var result = new float[n * outDim];
        LinearBias(path, x, n, inDim, outDim, result.AsSpan());
        return result;
    }

    // ── Timestep sinusoidal embedding ─────────────────────────────────────

    private static float[] TimestepEmbedding(float t, int dim)
    {
        // Standard sinusoidal embedding at fractional timestep t ∈ [0,1]
        var emb = new float[dim];
        int halfDim = dim / 2;
        float logMax = MathF.Log(10000f);
        for (int i = 0; i < halfDim; i++)
        {
            float freq = MathF.Exp(-logMax * i / (halfDim - 1));
            float v = t * 1000f * freq;   // scale t to [0, 1000]
            emb[i]           = MathF.Cos(v);
            emb[i + halfDim] = MathF.Sin(v);
        }
        return emb;
    }

    // ── Weight access ─────────────────────────────────────────────────────

    /// <summary>
    /// Some GGUF exports of FLUX (e.g. city96's ComfyUI-GGUF converter) strip the redundant
    /// "model.diffusion_model." prefix since the file only ever contains DiT tensors -- while
    /// this class's own naming (and real diffusers-format safetensors checkpoints) keep it. Try
    /// the requested name as-is first, then with that prefix stripped, so either convention loads.
    /// </summary>
    private const string DiffusionModelPrefix = "model.diffusion_model.";

    private GgufTensorInfo? FindTensor(string name)
    {
        var info = _model.FindTensor(name);
        if (info is not null) return info;
        return name.StartsWith(DiffusionModelPrefix, StringComparison.Ordinal)
            ? _model.FindTensor(name[DiffusionModelPrefix.Length..])
            : null;
    }

    private float[] W(string name)
    {
        if (_weightCache.TryGetValue(name, out var cached)) return cached;
        var info = FindTensor(name) ?? throw new KeyNotFoundException($"DiT weight not found: {name}");
        var data = DequantGguf(info);
        _weightCache[name] = data;
        return data;
    }

    private float[]? OptW(string name)
    {
        if (_weightCache.TryGetValue(name, out var cached)) return cached;
        var info = FindTensor(name);
        if (info is null) return null;
        var data = DequantGguf(info.Value);
        _weightCache[name] = data;
        return data;
    }

    private float[] DequantGguf(GgufTensorInfo info)
    {
        var raw = _model.GetTensorData(info);
        var dst = new float[info.ElementCount];
        Dequantize.ToFloat32(raw, dst, info.DType, info.ElementCount);
        return dst;
    }

    // ── GPU-accelerated matmul with weight caching ────────────────────────

    /// <summary>
    /// Multiply <paramref name="x"/> [n × inDim] by weight tensor <paramref name="wName"/>
    /// [outDim × inDim], optionally adding bias, and return the result [n × outDim].
    ///
    /// On a CUDA backend: dequantizes the weight (fp8 / bf16 / fp16 / fp32 depending on
    /// device capability), uploads it once and caches on GPU, then dispatches cuBLAS SGEMM.
    /// Falls back to CPU (SimdKernels.MatMulBatched) when no GPU backend or n &lt; MinGpuBatch.
    /// </summary>
    private unsafe void MatQ(ReadOnlySpan<float> x, int n, int inDim, string wName, int outDim,
                             float[]? bias, Span<float> result)
    {
        var info = FindTensor(wName);
        if (info.HasValue)
        {
            var ti       = info.Value;
            int rows     = (int)ti.Dimensions[1];  // outDim — output features (ne1)
            int cols     = (int)ti.Dimensions[0];  // inDim  — input features  (ne0)
            var rawBytes = _model.GetTensorData(ti);

            if (_backend is not CpuBackend && n >= MinGpuBatch)
            {
                if (_backend.BestSgemmPrecision == SgemmPrecision.Fp8E4M3)
                {
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    byte[]   xFp8     = ArrayPool<byte>.Shared.Rent(xCount);
                    ushort[] cBf16Buf = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        for (int i = 0; i < xCount; i++)
                            xFp8[i] = Fp8Converter.FloatToFp8E4M3(x[i]);

                        CoreTensor wGpu;
                        bool ownW;
                        if (_gpuWeightsFp8 != null && _gpuWeightsFp8.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW; ownW = false;
                        }
                        else
                        {
                            float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            byte[]  wFp8   = ArrayPool<byte>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(rawBytes, wBuf32.AsSpan(0, wCount), ti.DType, wCount);
                                for (int i = 0; i < wCount; i++)
                                    wFp8[i] = Fp8Converter.FloatToFp8E4M3(wBuf32[i]);
                                wGpu = _backend.UploadFp8(wFp8.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<byte>.Shared.Return(wFp8);
                            }
                            if (_gpuWeightsFp8 != null)
                                _gpuWeightsFp8[wName] = wGpu;
                            ownW = _gpuWeightsFp8 == null;
                        }

                        var xGpu = _backend.UploadFp8(xFp8.AsSpan(0, xCount), TensorShape.D1(xCount));
                        // fp8 GEMM output must be bf16 (cuBLAS restriction); convert to fp32 on download
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.BFloat16);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.DownloadBf16(cGpu, cBf16Buf.AsSpan(0, n * rows));
                            int cCount = n * rows;
                            for (int i = 0; i < cCount; i++)
                            {
                                uint bits = (uint)cBf16Buf[i] << 16;
                                result[i] = BitConverter.UInt32BitsToSingle(bits);
                            }
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            if (ownW) _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        ArrayPool<byte>.Shared.Return(xFp8);
                        ArrayPool<ushort>.Shared.Return(cBf16Buf);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Bf16)
                {
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    ushort[] xBf16 = ArrayPool<ushort>.Shared.Rent(xCount);
                    ushort[] cBf16 = ArrayPool<ushort>.Shared.Rent(n * rows);
                    try
                    {
                        for (int i = 0; i < xCount; i++)
                        {
                            uint bits = BitConverter.SingleToUInt32Bits(x[i]);
                            xBf16[i] = (ushort)(bits >> 16);
                        }

                        CoreTensor wGpu;
                        bool ownW;
                        if (_gpuWeightsBf16 != null && _gpuWeightsBf16.TryGetValue(wName, out var cachedW))
                        {
                            wGpu = cachedW; ownW = false;
                        }
                        else
                        {
                            float[]  wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                            ushort[] wBf16  = ArrayPool<ushort>.Shared.Rent(wCount);
                            try
                            {
                                Dequantize.ToFloat32(rawBytes, wBuf32.AsSpan(0, wCount), ti.DType, wCount);
                                for (int i = 0; i < wCount; i++)
                                {
                                    uint bits = BitConverter.SingleToUInt32Bits(wBuf32[i]);
                                    wBf16[i] = (ushort)(bits >> 16);
                                }
                                wGpu = _backend.UploadBf16(wBf16.AsSpan(0, wCount), TensorShape.D1(wCount));
                            }
                            finally
                            {
                                ArrayPool<float>.Shared.Return(wBuf32);
                                ArrayPool<ushort>.Shared.Return(wBf16);
                            }
                            if (_gpuWeightsBf16 != null)
                                _gpuWeightsBf16[wName] = wGpu;
                            ownW = _gpuWeightsBf16 == null;
                        }

                        var xGpu = _backend.UploadBf16(xBf16.AsSpan(0, xCount), TensorShape.D1(xCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.BFloat16);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.DownloadBf16(cGpu, cBf16.AsSpan(0, n * rows));
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            if (ownW) _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }

                        for (int i = 0, cnt = n * rows; i < cnt; i++)
                        {
                            uint bits = (uint)cBf16[i] << 16;
                            result[i] = BitConverter.UInt32BitsToSingle(bits);
                        }
                    }
                    finally
                    {
                        ArrayPool<ushort>.Shared.Return(xBf16);
                        ArrayPool<ushort>.Shared.Return(cBf16);
                    }
                }
                else if (_backend.BestSgemmPrecision == SgemmPrecision.Fp16)
                {
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    float[] wBuf32 = ArrayPool<float>.Shared.Rent(wCount);
                    Half[]  wHalf  = ArrayPool<Half>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(rawBytes, wBuf32.AsSpan(0, wCount), ti.DType, wCount);
                        for (int i = 0; i < wCount; i++) wHalf[i] = (Half)wBuf32[i];

                        var xGpu = _backend.Upload(x.Slice(0, xCount), TensorShape.D1(xCount));
                        var wGpu = _backend.UploadHalf(wHalf.AsSpan(0, wCount), TensorShape.D1(wCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows), DType.Float32);
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.Download(cGpu, result);
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(wBuf32);
                        ArrayPool<Half>.Shared.Return(wHalf);
                    }
                }
                else
                {
                    // fp32 GPU path
                    int wCount = rows * cols;
                    int xCount = n * cols;
                    float[] wBuf = ArrayPool<float>.Shared.Rent(wCount);
                    try
                    {
                        Dequantize.ToFloat32(rawBytes, wBuf.AsSpan(0, wCount), ti.DType, wCount);
                        var xGpu = _backend.Upload(x.Slice(0, xCount), TensorShape.D1(xCount));
                        var wGpu = _backend.Upload(wBuf.AsSpan(0, wCount), TensorShape.D1(wCount));
                        var cGpu = _backend.Allocate(TensorShape.D1(n * rows));
                        try
                        {
                            _backend.Sgemm(cGpu, xGpu, wGpu, n, cols, rows);
                            _backend.Synchronize();
                            _backend.Download(cGpu, result);
                        }
                        finally
                        {
                            _backend.Free(xGpu);
                            _backend.Free(wGpu);
                            _backend.Free(cGpu);
                        }
                    }
                    finally { ArrayPool<float>.Shared.Return(wBuf); }
                }
            }
            else
            {
                // CPU path: zero-copy via unsafe pointer into mmap'd buffer
                fixed (byte* rawPtr = rawBytes)
                fixed (float* xPtr = x, rPtr = result)
                    SimdKernels.MatMulBatched(rPtr, rawPtr, xPtr, n, rows, cols, ti.DType);
            }
        }
        else
        {
            // Fallback: dequantize + naive multiply (should not be reached in normal operation)
            var w = W(wName);
            var xArr = x.ToArray();
            DiffusionOps.Linear(xArr, w, null, n, inDim, outDim).AsSpan().CopyTo(result);
        }

        if (bias is not null)
        {
            for (int b = 0; b < n; b++)
                TensorPrimitives.Add(result.Slice(b * outDim, outDim),
                                     bias.AsSpan(), result.Slice(b * outDim, outDim));
        }
    }

    private float[] MatQ(float[] x, int n, int inDim, string wName, int outDim, float[]? bias)
    {
        var result = new float[n * outDim];
        MatQ(x.AsSpan(0, n * inDim), n, inDim, wName, outDim, bias, result.AsSpan());
        return result;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _disposed = true;
            _weightCache.Clear();
            if (_gpuWeightsBf16 != null)
            {
                foreach (var t in _gpuWeightsBf16.Values) _backend.Free(t);
                _gpuWeightsBf16.Clear();
            }
            if (_gpuWeightsFp8 != null)
            {
                foreach (var t in _gpuWeightsFp8.Values) _backend.Free(t);
                _gpuWeightsFp8.Clear();
            }
            _model.Dispose();
        }
    }
}
