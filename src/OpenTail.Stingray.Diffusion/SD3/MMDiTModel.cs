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

    /// <summary>Returns a SiLU-gated COPY of a conditioning vector reused across multiple
    /// adaLN_modulation Linear calls (real: `self.linear(self.silu(emb))`).</summary>
    private static float[] SiluGate(float[] vec)
    {
        var gated = (float[])vec.Clone();
        DiffusionOps.SiluInPlace(gated);
        return gated;
    }

    /// <summary>Splits a fused [n, 3*dim] qkv projection into three contiguous [n, dim] blocks.</summary>
    private static (float[] q, float[] k, float[] v) SplitQkv(float[] qkv, int n, int dim)
    {
        var q = new float[n * dim];
        var k = new float[n * dim];
        var v = new float[n * dim];
        for (int t = 0; t < n; t++)
        {
            int src = t * 3 * dim;
            Array.Copy(qkv, src, q, t * dim, dim);
            Array.Copy(qkv, src + dim, k, t * dim, dim);
            Array.Copy(qkv, src + 2 * dim, v, t * dim, dim);
        }
        return (q, k, v);
    }

    /// <summary>Per-head RMSNorm (no bias) applied in place, using a real ln_q/ln_k.weight tensor
    /// if present; no-op if the checkpoint doesn't declare it for this block.</summary>
    private void ApplyHeadRmsNorm(float[] x, string weightName, int n, int numHeads, int headDim)
    {
        var w = TryGetWeight($"{weightName}.weight");
        if (w is null) return;
        for (int t = 0; t < n; t++)
        {
            for (int h = 0; h < numHeads; h++)
            {
                int off = t * numHeads * headDim + h * headDim;
                float sumSq = 0f;
                for (int d = 0; d < headDim; d++) sumSq += x[off + d] * x[off + d];
                float invStd = 1f / MathF.Sqrt(sumSq / headDim + 1e-6f);
                for (int d = 0; d < headDim; d++) x[off + d] = x[off + d] * invStd * w[d];
            }
        }
    }

    private CoreTensor GetGpuWeight(string name, float[] cpuWeight)
    {
        string fullName = _weightReader.Prefix + name;
        if (_gpuWeights!.TryGetValue(fullName, out var wGpu)) return wGpu;

        wGpu = _backend!.Upload(cpuWeight.AsSpan(), TensorShape.D1(cpuWeight.Length));
        _gpuWeights[fullName] = wGpu;
        return wGpu;
    }

    public float[] Lin(string name, float[] x, int n, int inDim, int outDim)
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
        if (bF is not null && bF.Length != outDim)
            throw new InvalidOperationException(
                $"MMDiTModel.Lin(\"{name}\"): bias buffer has {bF.Length} elements, expected outDim={outDim}.");

        if (_backend is null)
        {
            return DiffusionOps.Linear(x, wF, bF, n, inDim, outDim);
        }

        var wGpu = GetGpuWeight($"{name}.weight", wF);
        var xGpu = _backend.Upload(x.AsSpan(0, n * inDim), TensorShape.D1(n * inDim));
        var cGpu = _backend.Allocate(TensorShape.D1(n * outDim));
        var result = new float[n * outDim];

        try
        {
            _backend.Sgemm(cGpu, xGpu, wGpu, n, inDim, outDim);
            _backend.Synchronize();
            _backend.Download(cGpu, result);
        }
        finally
        {
            _backend.Free(xGpu);
            _backend.Free(cGpu);
        }

        if (bF is not null)
        {
            Parallel.For(0, n, i =>
            {
                int off = i * outDim;
                for (int o = 0; o < outDim; o++)
                    result[off + o] += bF[o];
            });
        }

        return result;
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
        int inPatchDim = InChannels * p * p;

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
            // actually in the checkpoint). This was previously MISSING entirely (not just
            // misordered, unlike FLUX's equivalent bug fixed earlier this session) at all three
            // call sites (img/txt per-block modulation + the final layer's). tVec is reused across
            // every block and the final layer, so SiLU must run on a copy each time.
            var imgMod = Lin($"{blk}.x_block.adaLN_modulation.1", SiluGate(tVec), 1, HiddenSize, imgModChunks * HiddenSize);
            var txtMod = Lin($"{blk}.context_block.adaLN_modulation.1", SiluGate(tVec), 1, HiddenSize, txtModChunks * HiddenSize);

            // ── Self/Joint Attention ────────────────────────────────────────
            var xNorm1 = ModulateNorm(x, imgMod, 0, numImgTokens, HiddenSize);
            var cNorm1 = ModulateNorm(c, txtMod, 0, numTextTokens, HiddenSize);

            // Real checkpoint (confirmed via list-tensors on both the safetensors and GGUF forms)
            // stores ONE fused qkv.weight [dim, 3*dim], not three separate qkv.0/1/2 matrices --
            // this had never been tested against a real checkpoint before (SD3 "has never actually
            // been run once", per docs/00-current-work.md), so this wrong assumption was never
            // caught. Real mmdit.py: `qkv = self.qkv(x)` then `.reshape(B,N,3,heads,head_dim)` --
            // a flat 3*dim output split into three contiguous dim-wide blocks [q|k|v], same
            // convention as FLUX's fused qkv.
            var (xQ, xK, xV) = SplitQkv(Lin($"{blk}.x_block.attn.qkv", xNorm1, numImgTokens, HiddenSize, 3 * HiddenSize), numImgTokens, HiddenSize);
            var (cQ, cK, cV) = SplitQkv(Lin($"{blk}.context_block.attn.qkv", cNorm1, numTextTokens, HiddenSize, 3 * HiddenSize), numTextTokens, HiddenSize);

            // QK-RMSNorm (per-head, no bias): real checkpoint tensors attn.ln_q.weight/
            // attn.ln_k.weight [headDim] exist (confirmed via list-tensors) but were never read at
            // all previously -- a second real gap alongside the fused-QKV one above.
            ApplyHeadRmsNorm(xQ, $"{blk}.x_block.attn.ln_q", numImgTokens, NumHeads, HeadDim);
            ApplyHeadRmsNorm(xK, $"{blk}.x_block.attn.ln_k", numImgTokens, NumHeads, HeadDim);
            ApplyHeadRmsNorm(cQ, $"{blk}.context_block.attn.ln_q", numTextTokens, NumHeads, HeadDim);
            ApplyHeadRmsNorm(cK, $"{blk}.context_block.attn.ln_k", numTextTokens, NumHeads, HeadDim);

            // Concatenate image + text tokens
            int totalTokens = numImgTokens + numTextTokens;
            var jointQ = ConcatSeq(xQ, cQ, numImgTokens, numTextTokens, HiddenSize);
            var jointK = ConcatSeq(xK, cK, numImgTokens, numTextTokens, HiddenSize);
            var jointV = ConcatSeq(xV, cV, numImgTokens, numTextTokens, HiddenSize);

            var jointAttn = JointMultiHeadAttention(jointQ, jointK, jointV, totalTokens, HiddenSize, NumHeads, HeadDim);

            var xAttn = jointAttn.AsSpan(0, numImgTokens * HiddenSize).ToArray();
            var cAttn = jointAttn.AsSpan(numImgTokens * HiddenSize, numTextTokens * HiddenSize).ToArray();

            var xProj = Lin($"{blk}.x_block.attn.proj", xAttn, numImgTokens, HiddenSize, HiddenSize);
            ApplyGateAndResidual(x, xProj, imgMod, 2, numImgTokens, HiddenSize);

            // context_pre_only (last block): text stream's attention output is computed as part
            // of the joint attention (image tokens still attend to it) but then DISCARDED --
            // real: no gate/residual/proj is applied on the text side, no MLP, `encoder_hidden_
            // states = None` at the end of this block. txtMod only has 2 chunks (shift,scale) here
            // -- reading a gate chunk[2] would be out of bounds.
            if (!contextPreOnly)
            {
                var cProj = Lin($"{blk}.context_block.attn.proj", cAttn, numTextTokens, HiddenSize, HiddenSize);
                ApplyGateAndResidual(c, cProj, txtMod, 2, numTextTokens, HiddenSize);
            }

            // ── Second, image-only self-attention (dual-attention blocks only) ──────────────
            // Real: `attn_output2 = self.attn2(hidden_states=norm_hidden_states2, ...)`, gated by
            // gate_msa2 and added as a SECOND residual on the image stream, AFTER the joint
            // attention's residual and BEFORE the MLP -- real attn2 has its own separate
            // qkv/proj/ln_q/ln_k weights and never sees the text tokens at all (image-only,
            // ordinary non-joint self-attention).
            if (dualAttn)
            {
                var xNorm1b = ModulateNorm(x, imgMod, 6, numImgTokens, HiddenSize);
                var (x2Q, x2K, x2V) = SplitQkv(Lin($"{blk}.x_block.attn2.qkv", xNorm1b, numImgTokens, HiddenSize, 3 * HiddenSize), numImgTokens, HiddenSize);
                ApplyHeadRmsNorm(x2Q, $"{blk}.x_block.attn2.ln_q", numImgTokens, NumHeads, HeadDim);
                ApplyHeadRmsNorm(x2K, $"{blk}.x_block.attn2.ln_k", numImgTokens, NumHeads, HeadDim);
                var x2Attn = JointMultiHeadAttention(x2Q, x2K, x2V, numImgTokens, HiddenSize, NumHeads, HeadDim);
                var x2Proj = Lin($"{blk}.x_block.attn2.proj", x2Attn, numImgTokens, HiddenSize, HiddenSize);
                ApplyGateAndResidual(x, x2Proj, imgMod, 8, numImgTokens, HiddenSize);
            }

            // ── FeedForward (MLP) ───────────────────────────────────────────
            var xNorm2 = ModulateNorm(x, imgMod, 3, numImgTokens, HiddenSize);

            int mlpHidden = HiddenSize * 4;
            var xMlp1 = Lin($"{blk}.x_block.mlp.fc1", xNorm2, numImgTokens, HiddenSize, mlpHidden);
            DiffusionOps.GeluInPlace(xMlp1);
            var xMlp2 = Lin($"{blk}.x_block.mlp.fc2", xMlp1, numImgTokens, mlpHidden, HiddenSize);
            ApplyGateAndResidual(x, xMlp2, imgMod, 5, numImgTokens, HiddenSize);

            // Real: context_pre_only blocks apply no MLP to the text stream at all (it's discarded
            // right after attention above); txtMod has no chunk 3/4/5 to read here either.
            if (!contextPreOnly)
            {
                var cNorm2 = ModulateNorm(c, txtMod, 3, numTextTokens, HiddenSize);
                var cMlp1 = Lin($"{blk}.context_block.mlp.fc1", cNorm2, numTextTokens, HiddenSize, mlpHidden);
                DiffusionOps.GeluInPlace(cMlp1);
                var cMlp2 = Lin($"{blk}.context_block.mlp.fc2", cMlp1, numTextTokens, mlpHidden, HiddenSize);
                ApplyGateAndResidual(c, cMlp2, txtMod, 5, numTextTokens, HiddenSize);
            }
        }

        // 5. Final Layer: modulation + linear projection back to patch channels
        var finalMod = Lin("final_layer.adaLN_modulation.1", SiluGate(tVec), 1, HiddenSize, 2 * HiddenSize);
        var finalNorm = ModulateNorm(x, finalMod, 0, numImgTokens, HiddenSize);

        int outPatchDim = OutChannels * p * p;
        var unpatchified = Lin("final_layer.linear", finalNorm, numImgTokens, HiddenSize, outPatchDim);

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

    private static float[] ModulateNorm(float[] tokens, float[] modVec, int modIdxOffset, int numTokens, int dim)
    {
        // modVec contains shift [dim] and scale [dim] starting at modIdxOffset * dim
        int shiftOff = modIdxOffset * dim;
        int scaleOff = (modIdxOffset + 1) * dim;

        var normed = (float[])tokens.Clone();
        for (int i = 0; i < numTokens; i++)
        {
            int off = i * dim;
            // LayerNorm over token
            float mean = 0f;
            for (int d = 0; d < dim; d++) mean += normed[off + d];
            mean /= dim;

            float var = 0f;
            for (int d = 0; d < dim; d++)
            {
                float diff = normed[off + d] - mean;
                var += diff * diff;
            }
            float invStd = 1f / MathF.Sqrt(var / dim + 1e-6f);

            for (int d = 0; d < dim; d++)
            {
                float n = (normed[off + d] - mean) * invStd;
                normed[off + d] = n * (1f + modVec[scaleOff + d]) + modVec[shiftOff + d];
            }
        }
        return normed;
    }

    private static void ApplyGateAndResidual(float[] target, float[] branch, float[] modVec, int gateIdx, int numTokens, int dim)
    {
        int gateOff = gateIdx * dim;
        Parallel.For(0, numTokens, i =>
        {
            int off = i * dim;
            for (int d = 0; d < dim; d++)
                target[off + d] += branch[off + d] * modVec[gateOff + d];
        });
    }

    private static float[] ConcatSeq(float[] seq1, float[] seq2, int n1, int n2, int dim)
    {
        var cat = new float[(n1 + n2) * dim];
        Array.Copy(seq1, 0, cat, 0, n1 * dim);
        Array.Copy(seq2, 0, cat, n1 * dim, n2 * dim);
        return cat;
    }

    private static float[] JointMultiHeadAttention(float[] q, float[] k, float[] v, int totalTokens, int dim, int nHeads, int headDim)
    {
        float scale = 1f / MathF.Sqrt(headDim);
        var output = new float[totalTokens * dim];

        Parallel.For(0, nHeads, h =>
        {
            int headOffset = h * headDim;
            var scores = new float[totalTokens];

            for (int qi = 0; qi < totalTokens; qi++)
            {
                int qBase = qi * dim + headOffset;

                for (int kj = 0; kj < totalTokens; kj++)
                {
                    int kBase = kj * dim + headOffset;
                    float dot = 0f;
                    for (int d = 0; d < headDim; d++)
                        dot += q[qBase + d] * k[kBase + d];
                    scores[kj] = dot * scale;
                }

                DiffusionOps.Softmax(scores, 0, totalTokens);

                int outBase = qi * dim + headOffset;
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0f;
                    for (int kj = 0; kj < totalTokens; kj++)
                        sum += scores[kj] * v[kj * dim + headOffset + d];
                    output[outBase + d] = sum;
                }
            }
        });

        return output;
    }

    public void Dispose()
    {
        if (_gpuWeights is not null)
        {
            foreach (var t in _gpuWeights.Values) _backend!.Free(t);
            _gpuWeights.Clear();
        }
        _weightReader.Clear();
    }
}

