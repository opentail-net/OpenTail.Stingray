using System.Buffers;
using System.Numerics.Tensors;
using CoreTensor = OpenTail.Stingray.Core.Tensor;

namespace OpenTail.Stingray.Diffusion.SD3;

/// <summary>
/// Multimodal Diffusion Transformer (MMDiT) for Stable Diffusion 3 / 3.5.
/// Supports dual-stream joint transformer blocks and single-stream self-attention blocks.
/// Supports CPU SIMD and Vulkan GPU SGEMM.
/// </summary>
public sealed class MMDiTModel : IDisposable
{
    private readonly CachedWeightReader _weightReader;
    private readonly IComputeBackend? _backend;
    private readonly Dictionary<string, CoreTensor>? _gpuWeights;

    public int HiddenSize { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int Depth { get; }
    public int InChannels { get; }
    public int OutChannels { get; }
    public int PatchSize { get; }
    public int ContextSize { get; }
    public int AdmInChannels { get; }

    public MMDiTModel(
        IWeightLoader weights,
        string prefix = "model.diffusion_model.",
        int hiddenSize = 1536,
        int numHeads = 24,
        int depth = 24,
        int inChannels = 16,
        int outChannels = 16,
        int patchSize = 2,
        int contextSize = 4096,
        int admInChannels = 2048,
        IComputeBackend? backend = null)
    {
        _weightReader = new CachedWeightReader(weights, prefix);
        HiddenSize = hiddenSize;
        NumHeads = numHeads;
        HeadDim = hiddenSize / numHeads;
        Depth = depth;
        InChannels = inChannels;
        OutChannels = outChannels;
        PatchSize = patchSize;
        ContextSize = contextSize;
        AdmInChannels = admInChannels;
        _backend = backend;
        if (_backend is not null)
            _gpuWeights = new Dictionary<string, CoreTensor>(StringComparer.Ordinal);
    }

    private float[] GetWeight(string name) => _weightReader.Get(name);

    private float[]? TryGetWeight(string name) => _weightReader.TryGet(name);

    private sealed class Workspace : IDisposable
    {
        public readonly float[] Q;
        public readonly float[] K;
        public readonly float[] V;
        public readonly float[] AttnOut;
        public readonly float[] NormedImg;
        public readonly float[] NormedTxt;
        public readonly float[] QkvImg;
        public readonly float[] QkvTxt;
        public readonly float[] OutImg;
        public readonly float[] OutTxt;
        public readonly float[] MlpBuf;
        public readonly float[] ImgMod;
        public readonly float[] TxtMod;
        public readonly float[] Scores;
        public readonly float[] TVecSilu;

        public Workspace(int totalTokens, int numImgTokens, int numTextTokens, int hiddenSize, int numHeads)
        {
            int maxTokens = Math.Max(totalTokens, Math.Max(numImgTokens, numTextTokens));
            int mlpHidden = hiddenSize * 4;
            Q = ArrayPool<float>.Shared.Rent(totalTokens * hiddenSize);
            K = ArrayPool<float>.Shared.Rent(totalTokens * hiddenSize);
            V = ArrayPool<float>.Shared.Rent(totalTokens * hiddenSize);
            AttnOut = ArrayPool<float>.Shared.Rent(totalTokens * hiddenSize);
            NormedImg = ArrayPool<float>.Shared.Rent(numImgTokens * hiddenSize);
            NormedTxt = ArrayPool<float>.Shared.Rent(numTextTokens * hiddenSize);
            QkvImg = ArrayPool<float>.Shared.Rent(numImgTokens * 3 * hiddenSize);
            QkvTxt = ArrayPool<float>.Shared.Rent(numTextTokens * 3 * hiddenSize);
            OutImg = ArrayPool<float>.Shared.Rent(numImgTokens * hiddenSize);
            OutTxt = ArrayPool<float>.Shared.Rent(numTextTokens * hiddenSize);
            MlpBuf = ArrayPool<float>.Shared.Rent(maxTokens * mlpHidden);
            ImgMod = ArrayPool<float>.Shared.Rent(9 * hiddenSize);
            TxtMod = ArrayPool<float>.Shared.Rent(6 * hiddenSize);
            Scores = ArrayPool<float>.Shared.Rent(numHeads * totalTokens);
            TVecSilu = ArrayPool<float>.Shared.Rent(hiddenSize);
        }

        public void Dispose()
        {
            ArrayPool<float>.Shared.Return(Q);
            ArrayPool<float>.Shared.Return(K);
            ArrayPool<float>.Shared.Return(V);
            ArrayPool<float>.Shared.Return(AttnOut);
            ArrayPool<float>.Shared.Return(NormedImg);
            ArrayPool<float>.Shared.Return(NormedTxt);
            ArrayPool<float>.Shared.Return(QkvImg);
            ArrayPool<float>.Shared.Return(QkvTxt);
            ArrayPool<float>.Shared.Return(OutImg);
            ArrayPool<float>.Shared.Return(OutTxt);
            ArrayPool<float>.Shared.Return(MlpBuf);
            ArrayPool<float>.Shared.Return(ImgMod);
            ArrayPool<float>.Shared.Return(TxtMod);
            ArrayPool<float>.Shared.Return(Scores);
            ArrayPool<float>.Shared.Return(TVecSilu);
        }
    }

    private static unsafe void UnpackQkv(ReadOnlySpan<float> qkv, Span<float> q, Span<float> k, Span<float> v, int dstTokenStart, int tokenCount, int dim)
    {
        fixed (float* pQkv = qkv, pQ = q, pK = k, pV = v)
        {
            float* pQkvLocal = pQkv;
            float* pQLocal = pQ;
            float* pKLocal = pK;
            float* pVLocal = pV;

            Parallel.For(0, tokenCount, i =>
            {
                int srcOff = i * 3 * dim;
                int dstOff = (dstTokenStart + i) * dim;
                new ReadOnlySpan<float>(pQkvLocal + srcOff, dim).CopyTo(new Span<float>(pQLocal + dstOff, dim));
                new ReadOnlySpan<float>(pQkvLocal + srcOff + dim, dim).CopyTo(new Span<float>(pKLocal + dstOff, dim));
                new ReadOnlySpan<float>(pQkvLocal + srcOff + 2 * dim, dim).CopyTo(new Span<float>(pVLocal + dstOff, dim));
            });
        }
    }

    /// <summary>Per-head RMSNorm (no bias) applied in place, using a real ln_q/ln_k.weight tensor
    /// if present; no-op if the checkpoint doesn't declare it for this block.</summary>
    private unsafe void ApplyHeadRmsNorm(Span<float> x, string weightName, int n, int numHeads, int headDim)
    {
        var w = TryGetWeight($"{weightName}.weight");
        if (w is null) return;

        fixed (float* pX = x, pW = w)
        {
            float* pXLocal = pX;
            float* pWLocal = pW;

            Parallel.For(0, n * numHeads, idx =>
            {
                int t = idx / numHeads;
                int h = idx % numHeads;
                int off = (t * numHeads + h) * headDim;
                var slice = new Span<float>(pXLocal + off, headDim);
                var weightSpan = new ReadOnlySpan<float>(pWLocal, headDim);
                float sumSq = TensorPrimitives.SumOfSquares(slice);
                float invStd = 1f / MathF.Sqrt(sumSq / headDim + 1e-6f);
                for (int d = 0; d < headDim; d++)
                    slice[d] = slice[d] * invStd * weightSpan[d];
            });
        }
    }

    private CoreTensor GetGpuWeight(string name, float[] cpuWeight)
    {
        string fullName = _weightReader.Prefix + name;
        lock (_gpuWeights!)
        {
            if (_gpuWeights.TryGetValue(fullName, out var wGpu)) return wGpu;
            wGpu = _backend!.Upload(cpuWeight.AsSpan(), TensorShape.D1(cpuWeight.Length));
            _gpuWeights[fullName] = wGpu;
            return wGpu;
        }
    }

    public unsafe void Lin(string name, ReadOnlySpan<float> x, Span<float> dst, int n, int inDim, int outDim)
    {
        var wF = GetWeight($"{name}.weight");
        var bF = TryGetWeight($"{name}.bias");

        // Real checkpoint has never been run against this port before this pass -- a wrong
        // n/inDim/outDim here previously corrupted memory silently (DiffusionOps.Linear indexes
        // raw pointers with no bounds check) instead of failing loudly. Catch the mismatch here
        // with the exact call site and buffer sizes instead of an opaque AccessViolationException
        // deep in a vectorized dot product.
        if (wF.Length != (long)outDim * inDim)
            throw new InvalidOperationException(
                $"MMDiTModel.Lin(\"{name}\"): weight buffer has {wF.Length} elements, expected " +
                $"outDim*inDim = {outDim}*{inDim} = {(long)outDim * inDim}. n={n}, x.Length={x.Length} " +
                $"(expected n*inDim={(long)n * inDim}).");
        if (x.Length < (long)n * inDim)
            throw new InvalidOperationException(
                $"MMDiTModel.Lin(\"{name}\"): input buffer has {x.Length} elements, expected at least " +
                $"n*inDim = {n}*{inDim} = {(long)n * inDim}.");
        if (dst.Length < (long)n * outDim)
            throw new InvalidOperationException(
                $"MMDiTModel.Lin(\"{name}\"): dest buffer has {dst.Length} elements, expected at least " +
                $"n*outDim = {n}*{outDim} = {(long)n * outDim}.");
        if (bF is not null && bF.Length != outDim)
            throw new InvalidOperationException(
                $"MMDiTModel.Lin(\"{name}\"): bias buffer has {bF.Length} elements, expected outDim={outDim}.");

        if (_backend is null)
        {
            DiffusionOps.Linear(x.Slice(0, n * inDim), wF.AsSpan(0, outDim * inDim), bF is not null ? bF.AsSpan(0, outDim) : ReadOnlySpan<float>.Empty, dst.Slice(0, n * outDim), n, inDim, outDim);
            return;
        }

        var wGpu = GetGpuWeight($"{name}.weight", wF);
        var xGpu = _backend.Upload(x.Slice(0, n * inDim), TensorShape.D1(n * inDim));
        var cGpu = _backend.Allocate(TensorShape.D1(n * outDim));

        try
        {
            _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
            _backend.Synchronize();
            _backend.Download(cGpu, dst.Slice(0, n * outDim));
        }
        finally
        {
            _backend.Free(xGpu);
            _backend.Free(cGpu);
        }

        if (bF is not null)
        {
            fixed (float* pDst = dst, pB = bF)
            {
                float* pDstLocal = pDst;
                float* pBLocal = pB;
                Parallel.For(0, n, i =>
                {
                    int off = i * outDim;
                    for (int o = 0; o < outDim; o++)
                        pDstLocal[off + o] += pBLocal[o];
                });
            }
        }
    }

    public float[] Lin(string name, float[] x, int n, int inDim, int outDim)
    {
        var res = new float[n * outDim];
        Lin(name, x.AsSpan(0, n * inDim), res.AsSpan(), n, inDim, outDim);
        return res;
    }

    public float[] ComputeTimeAndPooledEmbedding(float timestep, float[] pooledY)
    {
        // 1. Timestep Fourier embedding: [256] -> [hiddenSize]
        int dim = 256;
        var sinEmb = new float[dim];
        int half = dim / 2;
        float logMaxPeriod = MathF.Log(10000.0f);

        for (int i = 0; i < half; i++)
        {
            float freq = MathF.Exp(-logMaxPeriod * i / half);
            float arg = timestep * freq;
            sinEmb[i]        = MathF.Cos(arg);
            sinEmb[half + i] = MathF.Sin(arg);
        }

        var tEmb = Lin("t_embedder.mlp.0", sinEmb, 1, dim, HiddenSize);
        DiffusionOps.SiluInPlace(tEmb);
        tEmb = Lin("t_embedder.mlp.2", tEmb, 1, HiddenSize, HiddenSize);

        // 2. Pooled Y embedding: [2048] -> [hiddenSize]
        var yEmb = Lin("y_embedder.mlp.0", pooledY, 1, AdmInChannels, HiddenSize);
        DiffusionOps.SiluInPlace(yEmb);
        yEmb = Lin("y_embedder.mlp.2", yEmb, 1, HiddenSize, HiddenSize);

        for (int i = 0; i < HiddenSize; i++)
            tEmb[i] += yEmb[i];

        return tEmb;
    }

    public float[] Forward(float[] latents, float timestep, float[] textContext, float[] pooledY, int latH, int latW, int numTextTokens)
    {
        int p = PatchSize;
        int imgH = latH / p;
        int imgW = latW / p;
        int numImgTokens = imgH * imgW;
        int totalTokens = numImgTokens + numTextTokens;
        int inPatchDim = InChannels * p * p;

        using var ws = new Workspace(totalTokens, numImgTokens, numTextTokens, HiddenSize, NumHeads);

        // 1. Patchify latents: [1, 16, latH, latW] -> [numImgTokens, inPatchDim]
        var imgTokens = new float[numImgTokens * inPatchDim];
        for (int py = 0; py < imgH; py++)
        for (int px = 0; px < imgW; px++)
        {
            int tokenIdx = py * imgW + px;
            int dstBase = tokenIdx * inPatchDim;
            int idx = 0;

            for (int ic = 0; ic < InChannels; ic++)
            for (int dy = 0; dy < p; dy++)
            for (int dx = 0; dx < p; dx++)
            {
                int iy = py * p + dy;
                int ix = px * p + dx;
                imgTokens[dstBase + idx++] = latents[ic * latH * latW + iy * latW + ix];
            }
        }

        // Linear x_embedder projection: inPatchDim -> HiddenSize
        var x = Lin("x_embedder.proj", imgTokens, numImgTokens, inPatchDim, HiddenSize);

        // 2. Project text context: ContextSize (4096) -> HiddenSize
        var c = Lin("context_embedder", textContext, numTextTokens, ContextSize, HiddenSize);

        // 3. Time + Pooled Embedding
        var tVec = ComputeTimeAndPooledEmbedding(timestep, pooledY);
        tVec.AsSpan().CopyTo(ws.TVecSilu.AsSpan(0, HiddenSize));
        DiffusionOps.SiluInPlace(ws.TVecSilu.AsSpan(0, HiddenSize));

        // 4. Joint MMDiT Transformer Blocks
        for (int b = 0; b < Depth; b++)
        {
            string blk = $"joint_blocks.{b}";

            // SD3.5-medium's first 13 of 24 blocks are "dual-attention" (real diffusers:
            // JointTransformerBlock(use_dual_attention=True) / SD35AdaLayerNormZeroX) -- an EXTRA
            // image-only self-attention pass alongside the normal joint attention. Detected via
            // real tensor presence (this block's own attn2.qkv.weight), not a hardcoded layer-index
            // list, matching this project's established convention. When present, x_block's
            // modulation is 9*HiddenSize (shift/scale/gate for msa, mlp, AND msa2) instead of 6;
            // confirmed via the real crash this pass surfaced (x_block.adaLN_modulation.1's real
            // weight buffer was 9*HiddenSize*HiddenSize, not the assumed 6*HiddenSize*HiddenSize).
            bool dualAttn = TryGetWeight($"{blk}.x_block.attn2.qkv.weight") is not null;
            int imgModChunks = dualAttn ? 9 : 6;

            // Real: the LAST block has context_pre_only=True (confirmed by this pass's own crash:
            // the real last block's context_block.adaLN_modulation.1 weight buffer is 2*HiddenSize
            // wide, not 6x) -- its context_block uses a plain AdaLayerNormContinuous (shift+scale
            // only, no gate) instead of AdaLayerNormZero, joint attention still runs normally using
            // that normed context (image tokens still attend to text), but the text stream's
            // attention output is then DISCARDED entirely (real: `encoder_hidden_states = None`) --
            // no gate/residual, no MLP, since nothing reads the text stream's value after the last
            // block anyway.
            bool contextPreOnly = b == Depth - 1;
            int txtModChunks = contextPreOnly ? 2 : 6;

            // Modulations. Real AdaLayerNormZero/AdaLayerNormZeroX/AdaLayerNormContinuous:
            // `emb = self.linear(self.silu(emb))` -- SiLU gates tVec BEFORE the modulation linear,
            // matching the real ".1" tensor-name suffix (nn.Sequential(SiLU(), Linear()), index 0
            // = the parameter-free SiLU, index 1 = the real Linear whose weights are what's
            // actually in the checkpoint).
            Lin($"{blk}.x_block.adaLN_modulation.1", ws.TVecSilu.AsSpan(0, HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 1, HiddenSize, imgModChunks * HiddenSize);
            Lin($"{blk}.context_block.adaLN_modulation.1", ws.TVecSilu.AsSpan(0, HiddenSize), ws.TxtMod.AsSpan(0, txtModChunks * HiddenSize), 1, HiddenSize, txtModChunks * HiddenSize);

            // ── Self/Joint Attention ────────────────────────────────────────
            ModulateNorm(x.AsSpan(0, numImgTokens * HiddenSize), ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 0, numImgTokens, HiddenSize);
            ModulateNorm(c.AsSpan(0, numTextTokens * HiddenSize), ws.NormedTxt.AsSpan(0, numTextTokens * HiddenSize), ws.TxtMod.AsSpan(0, txtModChunks * HiddenSize), 0, numTextTokens, HiddenSize);

            // Real checkpoint (confirmed via list-tensors on both the safetensors and GGUF forms)
            // stores ONE fused qkv.weight [dim, 3*dim], not three separate qkv.0/1/2 matrices --
            // this had never been tested against a real checkpoint before (SD3 "has never actually
            // been run once", per docs/00-current-work.md), so this wrong assumption was never
            // caught. Real mmdit.py: `qkv = self.qkv(x)` then `.reshape(B,N,3,heads,head_dim)` --
            // a flat 3*dim output split into three contiguous dim-wide blocks [q|k|v], same
            // convention as FLUX's fused qkv.
            Lin($"{blk}.x_block.attn.qkv", ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.QkvImg.AsSpan(0, numImgTokens * 3 * HiddenSize), numImgTokens, HiddenSize, 3 * HiddenSize);
            Lin($"{blk}.context_block.attn.qkv", ws.NormedTxt.AsSpan(0, numTextTokens * HiddenSize), ws.QkvTxt.AsSpan(0, numTextTokens * 3 * HiddenSize), numTextTokens, HiddenSize, 3 * HiddenSize);

            UnpackQkv(ws.QkvImg.AsSpan(0, numImgTokens * 3 * HiddenSize), ws.Q.AsSpan(0, totalTokens * HiddenSize), ws.K.AsSpan(0, totalTokens * HiddenSize), ws.V.AsSpan(0, totalTokens * HiddenSize), 0, numImgTokens, HiddenSize);
            UnpackQkv(ws.QkvTxt.AsSpan(0, numTextTokens * 3 * HiddenSize), ws.Q.AsSpan(0, totalTokens * HiddenSize), ws.K.AsSpan(0, totalTokens * HiddenSize), ws.V.AsSpan(0, totalTokens * HiddenSize), numImgTokens, numTextTokens, HiddenSize);

            // QK-RMSNorm (per-head, no bias): real checkpoint tensors attn.ln_q.weight/
            // attn.ln_k.weight [headDim] exist (confirmed via list-tensors) but were never read at
            // all previously -- a second real gap alongside the fused-QKV one above.
            ApplyHeadRmsNorm(ws.Q.AsSpan(0, numImgTokens * HiddenSize), $"{blk}.x_block.attn.ln_q", numImgTokens, NumHeads, HeadDim);
            ApplyHeadRmsNorm(ws.K.AsSpan(0, numImgTokens * HiddenSize), $"{blk}.x_block.attn.ln_k", numImgTokens, NumHeads, HeadDim);
            ApplyHeadRmsNorm(ws.Q.AsSpan(numImgTokens * HiddenSize, numTextTokens * HiddenSize), $"{blk}.context_block.attn.ln_q", numTextTokens, NumHeads, HeadDim);
            ApplyHeadRmsNorm(ws.K.AsSpan(numImgTokens * HiddenSize, numTextTokens * HiddenSize), $"{blk}.context_block.attn.ln_k", numTextTokens, NumHeads, HeadDim);

            JointMultiHeadAttention(ws.Q.AsSpan(0, totalTokens * HiddenSize), ws.K.AsSpan(0, totalTokens * HiddenSize), ws.V.AsSpan(0, totalTokens * HiddenSize), ws.AttnOut.AsSpan(0, totalTokens * HiddenSize), ws.Scores, totalTokens, HiddenSize, NumHeads, HeadDim);

            var xAttn = ws.AttnOut.AsSpan(0, numImgTokens * HiddenSize);
            var cAttn = ws.AttnOut.AsSpan(numImgTokens * HiddenSize, numTextTokens * HiddenSize);

            Lin($"{blk}.x_block.attn.proj", xAttn, ws.OutImg.AsSpan(0, numImgTokens * HiddenSize), numImgTokens, HiddenSize, HiddenSize);
            ApplyGateAndResidual(x.AsSpan(0, numImgTokens * HiddenSize), ws.OutImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 2, numImgTokens, HiddenSize);

            // context_pre_only (last block): text stream's attention output is computed as part
            // of the joint attention (image tokens still attend to it) but then DISCARDED --
            // real: no gate/residual/proj is applied on the text side, no MLP, `encoder_hidden_
            // states = None` at the end of this block. txtMod only has 2 chunks (shift,scale) here
            // -- reading a gate chunk[2] would be out of bounds.
            if (!contextPreOnly)
            {
                Lin($"{blk}.context_block.attn.proj", cAttn, ws.OutTxt.AsSpan(0, numTextTokens * HiddenSize), numTextTokens, HiddenSize, HiddenSize);
                ApplyGateAndResidual(c.AsSpan(0, numTextTokens * HiddenSize), ws.OutTxt.AsSpan(0, numTextTokens * HiddenSize), ws.TxtMod.AsSpan(0, txtModChunks * HiddenSize), 2, numTextTokens, HiddenSize);
            }

            // ── Second, image-only self-attention (dual-attention blocks only) ──────────────
            // Real: `attn_output2 = self.attn2(hidden_states=norm_hidden_states2, ...)`, gated by
            // gate_msa2 and added as a SECOND residual on the image stream, AFTER the joint
            // attention's residual and BEFORE the MLP -- real attn2 has its own separate
            // qkv/proj/ln_q/ln_k weights and never sees the text tokens at all (image-only,
            // ordinary non-joint self-attention).
            if (dualAttn)
            {
                ModulateNorm(x.AsSpan(0, numImgTokens * HiddenSize), ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 6, numImgTokens, HiddenSize);
                Lin($"{blk}.x_block.attn2.qkv", ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.QkvImg.AsSpan(0, numImgTokens * 3 * HiddenSize), numImgTokens, HiddenSize, 3 * HiddenSize);
                UnpackQkv(ws.QkvImg.AsSpan(0, numImgTokens * 3 * HiddenSize), ws.Q.AsSpan(0, numImgTokens * HiddenSize), ws.K.AsSpan(0, numImgTokens * HiddenSize), ws.V.AsSpan(0, numImgTokens * HiddenSize), 0, numImgTokens, HiddenSize);
                ApplyHeadRmsNorm(ws.Q.AsSpan(0, numImgTokens * HiddenSize), $"{blk}.x_block.attn2.ln_q", numImgTokens, NumHeads, HeadDim);
                ApplyHeadRmsNorm(ws.K.AsSpan(0, numImgTokens * HiddenSize), $"{blk}.x_block.attn2.ln_k", numImgTokens, NumHeads, HeadDim);
                JointMultiHeadAttention(ws.Q.AsSpan(0, numImgTokens * HiddenSize), ws.K.AsSpan(0, numImgTokens * HiddenSize), ws.V.AsSpan(0, numImgTokens * HiddenSize), ws.AttnOut.AsSpan(0, numImgTokens * HiddenSize), ws.Scores, numImgTokens, HiddenSize, NumHeads, HeadDim);
                Lin($"{blk}.x_block.attn2.proj", ws.AttnOut.AsSpan(0, numImgTokens * HiddenSize), ws.OutImg.AsSpan(0, numImgTokens * HiddenSize), numImgTokens, HiddenSize, HiddenSize);
                ApplyGateAndResidual(x.AsSpan(0, numImgTokens * HiddenSize), ws.OutImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 8, numImgTokens, HiddenSize);
            }

            // ── FeedForward (MLP) ───────────────────────────────────────────
            ModulateNorm(x.AsSpan(0, numImgTokens * HiddenSize), ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 3, numImgTokens, HiddenSize);

            int mlpHidden = HiddenSize * 4;
            Lin($"{blk}.x_block.mlp.fc1", ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.MlpBuf.AsSpan(0, numImgTokens * mlpHidden), numImgTokens, HiddenSize, mlpHidden);
            DiffusionOps.GeluInPlace(ws.MlpBuf.AsSpan(0, numImgTokens * mlpHidden));
            Lin($"{blk}.x_block.mlp.fc2", ws.MlpBuf.AsSpan(0, numImgTokens * mlpHidden), ws.OutImg.AsSpan(0, numImgTokens * HiddenSize), numImgTokens, mlpHidden, HiddenSize);
            ApplyGateAndResidual(x.AsSpan(0, numImgTokens * HiddenSize), ws.OutImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, imgModChunks * HiddenSize), 5, numImgTokens, HiddenSize);

            // Real: context_pre_only blocks apply no MLP to the text stream at all (it's discarded
            // right after attention above); txtMod has no chunk 3/4/5 to read here either.
            if (!contextPreOnly)
            {
                ModulateNorm(c.AsSpan(0, numTextTokens * HiddenSize), ws.NormedTxt.AsSpan(0, numTextTokens * HiddenSize), ws.TxtMod.AsSpan(0, txtModChunks * HiddenSize), 3, numTextTokens, HiddenSize);
                Lin($"{blk}.context_block.mlp.fc1", ws.NormedTxt.AsSpan(0, numTextTokens * HiddenSize), ws.MlpBuf.AsSpan(0, numTextTokens * mlpHidden), numTextTokens, HiddenSize, mlpHidden);
                DiffusionOps.GeluInPlace(ws.MlpBuf.AsSpan(0, numTextTokens * mlpHidden));
                Lin($"{blk}.context_block.mlp.fc2", ws.MlpBuf.AsSpan(0, numTextTokens * mlpHidden), ws.OutTxt.AsSpan(0, numTextTokens * HiddenSize), numTextTokens, mlpHidden, HiddenSize);
                ApplyGateAndResidual(c.AsSpan(0, numTextTokens * HiddenSize), ws.OutTxt.AsSpan(0, numTextTokens * HiddenSize), ws.TxtMod.AsSpan(0, txtModChunks * HiddenSize), 5, numTextTokens, HiddenSize);
            }
        }

        // 5. Final Layer: modulation + linear projection back to patch channels
        Lin("final_layer.adaLN_modulation.1", ws.TVecSilu.AsSpan(0, HiddenSize), ws.ImgMod.AsSpan(0, 2 * HiddenSize), 1, HiddenSize, 2 * HiddenSize);
        ModulateNorm(x.AsSpan(0, numImgTokens * HiddenSize), ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), ws.ImgMod.AsSpan(0, 2 * HiddenSize), 0, numImgTokens, HiddenSize);

        int outPatchDim = OutChannels * p * p;
        var unpatchified = new float[numImgTokens * outPatchDim];
        Lin("final_layer.linear", ws.NormedImg.AsSpan(0, numImgTokens * HiddenSize), unpatchified.AsSpan(), numImgTokens, HiddenSize, outPatchDim);

        // 6. Unpatchify back to [1, 16, latH, latW]
        var outLatent = new float[OutChannels * latH * latW];
        for (int py = 0; py < imgH; py++)
        for (int px = 0; px < imgW; px++)
        {
            int tokenIdx = py * imgW + px;
            int srcBase = tokenIdx * outPatchDim;
            int idx = 0;

            for (int ch = 0; ch < OutChannels; ch++)
            for (int dy = 0; dy < p; dy++)
            for (int dx = 0; dx < p; dx++)
            {
                int y = py * p + dy;
                int xCoord = px * p + dx;
                outLatent[ch * latH * latW + y * latW + xCoord] = unpatchified[srcBase + idx++];
            }
        }

        return outLatent;
    }

    private static unsafe void ModulateNorm(ReadOnlySpan<float> tokens, Span<float> normed, ReadOnlySpan<float> modVec, int modIdxOffset, int numTokens, int dim)
    {
        // modVec contains shift [dim] and scale [dim] starting at modIdxOffset * dim
        int shiftOff = modIdxOffset * dim;
        int scaleOff = (modIdxOffset + 1) * dim;
        var shift = modVec.Slice(shiftOff, dim);
        var scale = modVec.Slice(scaleOff, dim);

        fixed (float* pTokens = tokens, pNormed = normed, pShift = shift, pScale = scale)
        {
            float* pTokLocal = pTokens;
            float* pNormLocal = pNormed;
            float* pShiftLocal = pShift;
            float* pScaleLocal = pScale;

            Parallel.For(0, numTokens, i =>
            {
                int off = i * dim;
                var inRow = new ReadOnlySpan<float>(pTokLocal + off, dim);
                var outRow = new Span<float>(pNormLocal + off, dim);

                float mean = TensorPrimitives.Sum(inRow) / dim;
                float sumSq = 0f;
                for (int d = 0; d < dim; d++)
                {
                    float diff = inRow[d] - mean;
                    sumSq += diff * diff;
                }
                float invStd = 1f / MathF.Sqrt(sumSq / dim + 1e-6f);

                for (int d = 0; d < dim; d++)
                {
                    float n = (inRow[d] - mean) * invStd;
                    outRow[d] = n * (1f + pScaleLocal[d]) + pShiftLocal[d];
                }
            });
        }
    }

    private static unsafe void ApplyGateAndResidual(Span<float> target, ReadOnlySpan<float> branch, ReadOnlySpan<float> modVec, int gateIdx, int numTokens, int dim)
    {
        int gateOff = gateIdx * dim;
        var gate = modVec.Slice(gateOff, dim);

        fixed (float* pT = target, pB = branch, pG = gate)
        {
            float* pTLocal = pT;
            float* pBLocal = pB;
            float* pGLocal = pG;

            Parallel.For(0, numTokens, i =>
            {
                int off = i * dim;
                var tSpan = new Span<float>(pTLocal + off, dim);
                var bSpan = new ReadOnlySpan<float>(pBLocal + off, dim);
                var gSpan = new ReadOnlySpan<float>(pGLocal, dim);
                TensorPrimitives.MultiplyAdd(bSpan, gSpan, tSpan, tSpan);
            });
        }
    }

    private static unsafe void JointMultiHeadAttention(
        ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v, Span<float> output,
        float[] threadScores, int totalTokens, int dim, int nHeads, int headDim)
    {
        float scale = 1f / MathF.Sqrt(headDim);

        fixed (float* pQ = q, pK = k, pV = v, pOut = output, pScores = threadScores)
        {
            float* pQLocal = pQ;
            float* pKLocal = pK;
            float* pVLocal = pV;
            float* pOutLocal = pOut;
            float* pScoresLocal = pScores;

            Parallel.For(0, nHeads, h =>
            {
                int headOffset = h * headDim;
                float* scores = pScoresLocal + h * totalTokens;

                for (int qi = 0; qi < totalTokens; qi++)
                {
                    int qBase = qi * dim + headOffset;
                    var qSpan = new ReadOnlySpan<float>(pQLocal + qBase, headDim);
                    float maxScore = float.NegativeInfinity;

                    for (int kj = 0; kj < totalTokens; kj++)
                    {
                        int kBase = kj * dim + headOffset;
                        var kSpan = new ReadOnlySpan<float>(pKLocal + kBase, headDim);
                        float dot = TensorPrimitives.Dot(qSpan, kSpan) * scale;
                        scores[kj] = dot;
                        if (dot > maxScore) maxScore = dot;
                    }

                    float sumExp = 0f;
                    for (int kj = 0; kj < totalTokens; kj++)
                    {
                        float exp = MathF.Exp(scores[kj] - maxScore);
                        scores[kj] = exp;
                        sumExp += exp;
                    }
                    float invSum = 1f / sumExp;

                    int outBase = qi * dim + headOffset;
                    var outSpan = new Span<float>(pOutLocal + outBase, headDim);
                    outSpan.Clear();

                    for (int kj = 0; kj < totalTokens; kj++)
                    {
                        float s = scores[kj] * invSum;
                        if (s == 0f) continue;
                        int vBase = kj * dim + headOffset;
                        var vSpan = new ReadOnlySpan<float>(pVLocal + vBase, headDim);
                        TensorPrimitives.MultiplyAdd(vSpan, s, outSpan, outSpan);
                    }
                }
            });
        }
    }

    public void Dispose()
    {
        if (_gpuWeights is not null)
        {
            lock (_gpuWeights)
            {
                foreach (var t in _gpuWeights.Values) _backend!.Free(t);
                _gpuWeights.Clear();
            }
        }
        _weightReader.Clear();
    }
}
